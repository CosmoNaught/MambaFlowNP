"""MambaFlowNP networks, base distributions and checkpoint I/O.

The samples of ``MambaFlowNP`` are the joint values of one random function, so predictions are
consistent across query sets.
"""

import dataclasses
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def _tf32(fn):
    """Runs ``fn`` with TF32 matrix multiplication, as used to train the released models."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        with jax.default_matmul_precision("tensorfloat32"):
            return fn(*args, **kwargs)

    return wrapped


@dataclasses.dataclass(frozen=True)
class Config:
    """Model hyperparameters. ``max_points`` sets the Fourier feature bandwidth."""

    hidden_size: int
    d_state: int
    num_layers: int
    rff_freqs: int
    max_points: int
    num_scan_dirs: int = 4
    num_heads: int = 4
    chunk_size: int = 64
    c_scale: float = 1.0
    scan_offsets: int = 8
    scan_norm: bool = True

    @property
    def d_min(self) -> float:
        return max(0.5 / self.max_points ** 0.5, 1e-4)


def hilbert_index(x, y, order: int = 16):
    """Index of each point along a Hilbert curve on the unit square."""
    n = 2**order
    ix = jnp.clip((x * n).astype(jnp.int32), 0, n - 1)
    iy = jnp.clip((y * n).astype(jnp.int32), 0, n - 1)

    def index_one(xv, yv):
        def step(carry, _):
            d, xc, yc, scale = carry
            rx = jnp.where((xc & scale) > 0, jnp.int32(1), jnp.int32(0))
            ry = jnp.where((yc & scale) > 0, jnp.int32(1), jnp.int32(0))
            d = d + scale * scale * ((3 * rx) ^ ry)
            swap = ry == 0
            flip = swap & (rx == 1)
            xc = jnp.where(flip, scale - 1 - xc, xc)
            yc = jnp.where(flip, scale - 1 - yc, yc)
            xc, yc = jnp.where(swap, yc, xc), jnp.where(swap, xc, yc)
            return (d, xc, yc, scale // 2), None

        init = (jnp.int32(0), xv, yv, jnp.int32(n // 2))
        (d, _, _, _), _ = jax.lax.scan(step, init, None, length=order)
        return d

    return jax.vmap(index_one)(ix, iy)


def sort_indices(coords, num_scan_dirs: int):
    """Hilbert scan orders: forward, reverse, transposed, transposed reverse."""
    forward = jnp.argsort(hilbert_index(coords[:, 0], coords[:, 1])).astype(jnp.int32)
    if num_scan_dirs == 2:
        return (forward, forward[::-1])
    transposed = jnp.argsort(hilbert_index(coords[:, 1], coords[:, 0])).astype(jnp.int32)
    return (forward, forward[::-1], transposed, transposed[::-1])[:num_scan_dirs]


def segsum(x):
    cumulative = jnp.cumsum(x, axis=-1)
    pairwise = cumulative[..., :, None] - cumulative[..., None, :]
    length = x.shape[-1]
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))
    return jnp.where(causal, pairwise, -jnp.inf)


def _pad_seq(x, pad):
    if pad == 0:
        return x
    width = [(0, 0)] * x.ndim
    width[1] = (0, pad)
    return jnp.pad(x, width)


def ssd_scan(x, dt, A, B_mat, C_mat, D, dt_bias, chunk_size, mask=None):
    """Chunked Mamba-2 scan. x: (B, L, H, P); dt: (B, L, H); B_mat, C_mat: (B, L, H, N)."""
    _, seq_len, num_heads, _ = x.shape
    pad = (chunk_size - seq_len % chunk_size) % chunk_size

    dt = jnp.clip(jax.nn.softplus(dt + dt_bias), 1e-3, 10.0)
    x, dt = _pad_seq(x, pad), _pad_seq(dt, pad)
    B_mat, C_mat = _pad_seq(B_mat, pad), _pad_seq(C_mat, pad)

    if mask is not None:
        m = _pad_seq(mask[:, :, None].astype(x.dtype), pad)
        x = x * m[:, :, None, :]
        dt = dt * m

    skip = D.reshape(1, 1, num_heads, 1) * x
    x_disc = x * dt[..., None]
    A_disc = A.astype(x_disc.dtype) * dt

    def to_chunks(t):
        batch, length, *rest = t.shape
        return t.reshape(batch, length // chunk_size, chunk_size, *rest)

    xb, Bb, Cb = to_chunks(x_disc), to_chunks(B_mat), to_chunks(C_mat)
    Ab = jnp.transpose(to_chunks(A_disc), (0, 3, 1, 2))
    A_cum = jnp.cumsum(Ab, axis=-1)

    within = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cb, Bb, jnp.exp(segsum(Ab)), xb)
    decay = jnp.exp(A_cum[..., -1:] - A_cum)
    chunk_states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", Bb, decay, xb)
    chunk_states = jnp.transpose(chunk_states, (1, 0, 2, 3, 4))

    def advance(state, inputs):
        chunk_state, a_end = inputs
        state = state * jnp.exp(a_end)[..., None, None] + chunk_state
        return state, state

    init = jnp.zeros_like(chunk_states[0])
    chunk_decay = jnp.transpose(A_cum[..., -1], (2, 0, 1))
    _, outgoing = jax.lax.scan(advance, init, (chunk_states, chunk_decay))
    arriving = jnp.concatenate([init[None], outgoing[:-1]], axis=0)
    across = jnp.einsum(
        "bclhn,bchpn,bhcl->bclhp", Cb, jnp.transpose(arriving, (1, 0, 2, 3, 4)), jnp.exp(A_cum)
    )

    y = within + across
    batch, chunks, chunk_len, heads, head_dim = y.shape
    y = y.reshape(batch, chunks * chunk_len, heads, head_dim) + skip
    return y[:, :seq_len] if pad else y


class FrozenRFF(nnx.Module):
    """Random Fourier features with fixed frequencies."""

    def __init__(self, in_features: int, num_freqs: int, d_min: float, c_scale: float = 1.0,
                 seed: int = 42):
        sigma = max(0.0, c_scale / (d_min + 1e-6))
        self.B = jax.random.normal(jax.random.PRNGKey(seed), (num_freqs, in_features)) * sigma

    def __call__(self, coords):
        proj = 2 * jnp.pi * jnp.dot(coords, self.B.T)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


TIME_EMBED_DIM = 256


def timestep_embedding(t, dim: int = TIME_EMBED_DIM):
    half = dim // 2
    freqs = jnp.exp(jnp.arange(half) * -(jnp.log(10000) / (half - 1)))
    angles = (t * 1000.0)[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=1)


class ZeroLinear(nnx.Module):
    """Affine layer initialised to zero."""

    def __init__(self, in_features: int, out_features: int):
        self.kernel = nnx.Param(jnp.zeros((in_features, out_features)))
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x):
        return x @ self.kernel + self.bias


class DistanceAwareMambaBlock(nnx.Module):
    """Mamba-2 block with step size conditioned on inter-point distance. With ``scan_norm``, as in
    Mamba-2, the gated scan output is RMS-normalised before the output projection."""

    def __init__(self, d_model: int, d_state: int, num_heads: int, chunk_size: int,
                 scan_norm: bool = True, *, rngs: nnx.Rngs):
        self.d_model, self.d_state = d_model, d_state
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.chunk_size = chunk_size

        self.in_proj = nnx.Linear(d_model, d_model * 2, rngs=rngs)
        self.norm = nnx.RMSNorm(d_model, epsilon=1e-5, rngs=rngs) if scan_norm else None
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dt_proj = nnx.Linear(d_model + 1, num_heads, rngs=rngs)
        self.bc_proj = nnx.Linear(d_model, d_state * 2, rngs=rngs)

        a_init = jax.random.uniform(rngs.params(), (num_heads,), minval=1.0, maxval=16.0)
        self.A_log = nnx.Param(jnp.log(a_init))
        self.D = nnx.Param(jnp.ones(num_heads))
        self.dt_bias = nnx.Param(jnp.zeros(num_heads))

    def __call__(self, x, dists, mask=None, readers=None):
        """``mask`` marks the tokens that enter the scan state; ``readers`` marks further tokens
        that only read it, adding no input and no decay."""
        batch, length, _ = x.shape
        value, gate = jnp.split(self.in_proj(x), 2, axis=-1)
        value = jax.nn.silu(value)

        dt = self.dt_proj(jnp.concatenate([value, dists], axis=-1))
        B_mat, C_mat = jnp.split(self.bc_proj(value), 2, axis=-1)
        head_shape = (batch, length, self.num_heads, self.d_state)
        heads = value.reshape(batch, length, self.num_heads, self.head_dim)

        y = ssd_scan(
            x=heads,
            dt=dt,
            A=-jnp.exp(self.A_log[:].astype(jnp.float32)),
            B_mat=jnp.broadcast_to(B_mat[:, :, None, :], head_shape),
            C_mat=jnp.broadcast_to(C_mat[:, :, None, :], head_shape),
            D=self.D[:],
            dt_bias=self.dt_bias[:],
            chunk_size=self.chunk_size,
            mask=mask,
        )
        if readers is not None:
            y = y + readers[:, :, None, None] * self.D[:].reshape(1, 1, -1, 1) * heads
        y = y.reshape(batch, length, self.d_model) * jax.nn.silu(gate)
        return self.out_proj(y if self.norm is None else self.norm(y))


LATENT = 256


class MambaFlowNP(nnx.Module):
    """Velocity field whose samples are the joint values of one random function.

    Queries only read the scan state, and the gaps and density factor use the context only, so no
    query affects another. All randomness shared between queries is one latent c ~ N(0, I) per
    sample: the base noise is the random function f0(x) = phi(x)^T c, and c conditions every
    token through the adaptive normalisation. Samples for any set of queries are therefore the
    joint values of one random function of the context.
    """

    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        self.num_scan_dirs = cfg.num_scan_dirs
        self.scan_offsets = cfg.scan_offsets
        h = cfg.hidden_size

        self.rff = FrozenRFF(2, cfg.rff_freqs, cfg.d_min, cfg.c_scale)
        token_dim = 2 * cfg.rff_freqs + 1
        self.ctx_proj = nnx.Linear(token_dim, h, rngs=rngs)
        self.qry_proj = nnx.Linear(token_dim, h, rngs=rngs)
        self.time_mlp = nnx.Sequential(
            nnx.Linear(TIME_EMBED_DIM, h, rngs=rngs), nnx.silu, nnx.Linear(h, h, rngs=rngs)
        )

        self.mamba_layers = nnx.List(
            [
                DistanceAwareMambaBlock(h, cfg.d_state, cfg.num_heads, cfg.chunk_size,
                                        cfg.scan_norm, rngs=rngs)
                for _ in range(cfg.num_layers)
            ]
        )
        self.layer_norms = nnx.List(
            [nnx.LayerNorm(h, use_bias=False, use_scale=False, rngs=rngs)
             for _ in range(cfg.num_layers)]
        )
        self.adaln_projs = nnx.List([ZeroLinear(h, 3 * h) for _ in range(cfg.num_layers)])

        self.dir_merge = nnx.Linear(cfg.num_scan_dirs * h, h, rngs=rngs)
        self.final_norm = nnx.LayerNorm(h, use_bias=False, use_scale=False, rngs=rngs)
        self.final_adaln = ZeroLinear(h, 2 * h)
        self.flow_out = ZeroLinear(h, 1)
        self.latent_mlp = nnx.Sequential(
            nnx.Linear(LATENT, h, rngs=rngs), nnx.silu, nnx.Linear(h, h, rngs=rngs)
        )
        key = jax.random.PRNGKey(7)
        self.noise_freq = jax.random.normal(key, (LATENT, 2)) * np.sqrt(cfg.max_points)
        self.noise_phase = jax.random.uniform(jax.random.fold_in(key, 1), (LATENT,),
                                              maxval=2 * jnp.pi)

    def base_noise(self, c, coords):
        """f0(x) = phi(x)^T c, shape (batch, n, 1): a unit-variance Gaussian-process sample with
        lengthscale 1 / sqrt(max_points)."""
        phi = jnp.sqrt(2.0 / LATENT) * jnp.cos(coords @ self.noise_freq.T + self.noise_phase)
        return jnp.einsum("bnm,bm->bn", phi, c)[..., None]

    def precompute_routing(self, x_ctx, x_qry, mask=None, offset=None):
        """Scan orders over all points; each gap is to the previous context point in the order and
        the density factor counts the context points."""
        coords = jnp.concatenate([x_ctx, x_qry], axis=1)
        n_ctx, dirs = x_ctx.shape[1], self.num_scan_dirs
        if mask is None:
            mask = jnp.ones(coords.shape[:2], bool)
        is_ctx = mask & (jnp.arange(coords.shape[1]) < n_ctx)[None]
        num_ctx = jnp.sum(is_ctx, axis=1, keepdims=True).astype(jnp.float32)
        keyed = jnp.where(mask[:, :, None], coords, jnp.finfo(jnp.float32).max)
        grid = keyed if offset is None else 0.5 * keyed + offset[:, None, :]
        idx = jax.vmap(partial(sort_indices, num_scan_dirs=dirs))(grid)
        inv = tuple(jax.vmap(lambda i: jnp.argsort(i).astype(jnp.int32))(idx[d])
                    for d in range(dirs))

        def gaps(crd, order, ctx, n):
            ordered, flag = crd[order], ctx[order]
            last = jax.lax.cummax(jnp.where(flag, jnp.arange(ordered.shape[0]), -1))
            prev = jnp.concatenate([jnp.array([-1]), last[:-1]])
            step = ordered - ordered[jnp.maximum(prev, 0)]
            dist = jnp.sqrt(jnp.sum(step ** 2, axis=-1, keepdims=True) + 1e-8)
            dist = jnp.where((prev >= 0)[:, None], jnp.minimum(dist, 10.0), 0.0)
            return dist * jnp.sqrt(jnp.maximum(n, 1.0))

        dists = jnp.concatenate([jax.vmap(gaps)(coords, idx[d], is_ctx, num_ctx)
                                 for d in range(dirs)], axis=0)
        return idx, inv, dists

    @_tf32
    def __call__(self, x_ctx, y_ctx, x_qry, y_t, t, c, routing_cache=None, mask=None):
        """Returns the velocity at each query, shape (batch, n_qry, 1).

        Args:
          x_ctx: Context coordinates in [0, 1], shape (batch, n_ctx, 2).
          y_ctx: Context values, shape (batch, n_ctx, 1).
          x_qry: Query coordinates in [0, 1], shape (batch, n_qry, 2).
          y_t: Flow state at the queries, shape (batch, n_qry, 1).
          t: Flow time, shape (batch,).
          c: Latent of each sample, shape (batch, LATENT).
          routing_cache: Output of ``precompute_routing``. Computed if omitted.
          mask: Validity of each point, shape (batch, n_ctx + n_qry), context first.
        """
        dirs, n_ctx = self.num_scan_dirs, x_ctx.shape[1]
        if mask is None:
            mask = jnp.ones((x_ctx.shape[0], n_ctx + x_qry.shape[1]), bool)
        tokens = jnp.concatenate([
            self.ctx_proj(jnp.concatenate([self.rff(x_ctx), y_ctx], axis=-1)),
            self.qry_proj(jnp.concatenate([self.rff(x_qry), y_t], axis=-1))], axis=1)
        cond = self.time_mlp(timestep_embedding(t)) + self.latent_mlp(c)
        if routing_cache is None:
            routing_cache = self.precompute_routing(x_ctx, x_qry, mask=mask)
        idx, inv, dists = routing_cache
        order = jnp.concatenate([idx[d] for d in range(dirs)], axis=0)
        take = jax.vmap(lambda seq, o: seq[o])

        write = mask & (jnp.arange(mask.shape[1]) < n_ctx)[None]
        hidden = take(jnp.tile(tokens, (dirs, 1, 1)), order)
        write = take(jnp.tile(write, (dirs, 1)), order)
        valid = take(jnp.tile(mask, (dirs, 1)), order)
        cond_all = jnp.tile(cond, (dirs, 1))
        for norm, layer, adaln in zip(self.layer_norms, self.mamba_layers, self.adaln_projs):
            shift, scale, gate = jnp.split(adaln(jax.nn.silu(cond_all)), 3, axis=-1)
            modulated = norm(hidden) * (1 + scale[:, None, :]) + shift[:, None, :]
            hidden = hidden + gate[:, None, :] * layer(modulated, dists, mask=write,
                                                       readers=valid & ~write)

        per_dir = jnp.split(hidden, dirs, axis=0)
        fused = self.dir_merge(
            jnp.concatenate([take(per_dir[d], inv[d]) for d in range(dirs)], axis=-1)
        ) * mask[:, :, None]
        shift, scale = jnp.split(self.final_adaln(jax.nn.silu(cond)), 2, axis=-1)
        return self.flow_out(self.final_norm(fused[:, n_ctx:]) * (1 + scale[:, None, :])
                             + shift[:, None, :])


def save_checkpoint(model, path):
    """Saves model parameters to an ``.npz`` file."""
    flat = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]
    np.savez(path, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})


def load_checkpoint(model, path):
    """Loads model parameters from an ``.npz`` file."""
    flat, treedef = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    names = [jax.tree_util.keystr(k) for k, _ in flat]
    with np.load(path) as stored:
        missing = [n for n in names if n not in stored]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        leaves = [jnp.asarray(stored[n], dtype=v.dtype) for n, (_, v) in zip(names, flat)]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, leaves))
    return model
