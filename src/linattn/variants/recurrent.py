"""Fixed-size-state sequence mixers: Mamba-2/SSD, Gated DeltaNet and KDA.

This module implements the compressed-state branch of the taxonomy in one place,
because the three models are three settings of one recurrence:

.. math::
    S_t = \\mathrm{Diag}(\\alpha_t)\\,(I - \\beta_t k_t k_t^{\\top})\\,S_{t-1}
          + \\beta_t k_t v_t^{\\top}, \\qquad o_t = q_t^{\\top} S_t

* ``kind="ssd"``  -- Mamba-2 / semiseparable: drop the erase term
  (``beta`` also scales the write).  Dao & Gu, *Transformers are SSMs* (2024),
  https://arxiv.org/abs/2405.21060
* ``kind="gdn"``  -- Gated DeltaNet: scalar decay ``alpha`` per head plus the
  delta-rule erase.  Yang et al., *Gated Delta Networks* (ICLR 2025),
  https://arxiv.org/abs/2412.06464
* ``kind="kda"``  -- Kimi Delta Attention: channel-wise decay plus the erase.
  Kimi Team, *Kimi Linear* (2025), https://arxiv.org/abs/2510.26692
* ``kind="gdn2"`` -- Gated DeltaNet-2: channel-wise decay with *separate*
  channel-wise erase and write gates.  Generalises KDA (both gates equal) and GDN
  (gates collapse to a scalar).  Yang et al., *Gated DeltaNet-2: Decoupling Erase
  and Write in Linear Attention* (2026), https://arxiv.org/abs/2605.22791

Two forward paths are provided and the test-suite asserts they agree:

``forward_recurrent``
    the literal per-token recurrence -- the definition, and the numerical oracle.
``forward_chunked``
    a chunkwise algorithm that needs only matrix products and **one**
    ``C x C`` triangular solve per chunk.

The chunkwise derivation, which is what the kernel in
``kernels/triton/delta_chunk.py`` implements.  Note the order of the two
operations in the recurrence: the decay is applied to the state *first*, and the
delta-rule erase then acts on the decayed state,
``S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T``.
For a scalar decay the order is irrelevant (diagonal and rank-one terms commute),
which is why the same code covers Mamba-2 and Gated DeltaNet; for *channel-wise*
decay (KDA) it is not, and getting it backwards produces an implementation that is
exact for scalar decay and wrong for KDA -- a bug that is easy to miss.

1. Normalise the decay out of the state inside a chunk by substituting
   :math:`S_t = \\mathrm{Diag}(G_t)\\tilde S_t` with
   :math:`G_t = \\prod_{l\\le t}\\alpha_l`:
   :math:`\\tilde S_i = \\tilde S_{i-1} + \\beta_i d_i
   (v_i^{\\top} - e_i^{\\top}\\tilde S_{i-1})` with the two channel-wise factors
   :math:`d_i = k_i / G_i` (erase-left *and* write-left) and
   :math:`e_i = k_i G_i` (erase-right).
2. The only sequentially coupled quantities are
   :math:`y_i = \\beta_i (v_i^{\\top} - e_i^{\\top}\\tilde S_{i-1})`, and they
   satisfy the triangular system
   :math:`(I + M) Y = \\mathrm{diag}(\\beta)(V - E \\tilde S_0)` with
   :math:`M_{ij} = \\beta_i\\, e_i^{\\top} d_j` for ``j < i`` -- one triangular
   solve per chunk produces all of them.
3. State and outputs follow directly:
   :math:`\\tilde S_C = \\tilde S_0 + D^{\\top} Y`,
   :math:`S_C = \\mathrm{Diag}(G_C)\\tilde S_C`, and
   :math:`o_i = \\tilde q_i^{\\top}\\tilde S_0 +
   \\sum_{j\\le i}(\\tilde q_i^{\\top} d_j) y_j` with
   :math:`\\tilde q_i = q_i G_i`.

Setting :math:`M = 0` and dropping the read term recovers the SSD (Mamba-2) chunk
algorithm, which is what ``kind="ssd"`` does.

Numerics: :math:`d_i = k_i/G_i` grows like :math:`\\exp(-\\mathrm{gc}_i)`, which
can reach ``1e13`` for a 16-token chunk with the maximum decay used in the tests.
The final multiplication by :math:`G_C` shrinks it back, so the result is accurate
in float64, but this is exactly why production kernels *bound* the log-decay
("lower-bounded decay" in KDA) and accumulate in FP32.  ``decay_clamp`` exposes
the bound, and the validation suite runs in float64 so that algorithmic
equivalence is not confounded by conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["LinearAttentionConfig", "LinearAttentionLayer", "STATE_KINDS"]

STATE_KINDS = ("ssd", "gdn", "kda", "gdn2")


@dataclass
class LinearAttentionConfig:
    d_model: int = 64
    n_heads: int = 4
    d_state: int = 32          # key dimension of the recurrent state
    expand_v: float = 1.0      # value dimension = head_dim * expand_v
    chunk: int = 32
    kind: str = "gdn"          # ssd | gdn | kda | gdn2
    decay_clamp: float = 4.0   # log-decay lower bound (per step, per channel)
    init_gate: float = 0.5     # sigmoid offset for the decay gate

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_value(self) -> int:
        return max(1, int(self.head_dim * self.expand_v))


class LinearAttentionLayer(torch.nn.Module):
    """One linear-attention / SSM layer with the recurrence above."""

    def __init__(self, cfg: LinearAttentionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.kind not in STATE_KINDS:
            raise ValueError(f"unknown kind {cfg.kind!r}; expected one of {STATE_KINDS}")
        h, dk, dv = cfg.n_heads, cfg.d_state, cfg.d_value
        self.q_proj = torch.nn.Linear(cfg.d_model, h * dk, bias=False)
        self.k_proj = torch.nn.Linear(cfg.d_model, h * dk, bias=False)
        self.v_proj = torch.nn.Linear(cfg.d_model, h * dv, bias=False)
        self.o_proj = torch.nn.Linear(h * dv, cfg.d_model, bias=False)
        # decay gate: log(alpha) <= 0 ; per-channel for kda, per-head otherwise
        decay_out = h * dk if cfg.kind in ("kda", "gdn2") else h
        self.decay_proj = torch.nn.Linear(cfg.d_model, decay_out, bias=False)
        self.beta_proj = torch.nn.Linear(cfg.d_model, h, bias=False)
        # GDN-2 decouples the two roles of the scalar gate: a channel-wise erase gate
        # ("how much old content to remove on the key side") and a channel-wise write
        # gate ("how much new content to commit").  KDA is both gates collapsed.
        if cfg.kind == "gdn2":
            self.beta_erase_proj = torch.nn.Linear(cfg.d_model, h * dk, bias=False)
            self.beta_write_proj = torch.nn.Linear(cfg.d_model, h * dk, bias=False)
        # Mamba-2 style per-head scalar A, initialised log-uniformly
        self.A_log = torch.nn.Parameter(torch.log(torch.linspace(1.0, 16.0, h)))

    # ------------------------------------------------------------------ helpers
    def _project(self, x: torch.Tensor):
        """``x``: (B, T, d_model) -> q,k,v,beta,log_decay,shapes (B,H,T,.)``"""
        b, t, _ = x.shape
        h, dk, dv = self.cfg.n_heads, self.cfg.d_state, self.cfg.d_value
        q = self.q_proj(x).view(b, t, h, dk).transpose(1, 2)
        k = self.k_proj(x).view(b, t, h, dk).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, dv).transpose(1, 2)
        # The delta rule is only stable for keys with bounded norm: the erase operator
        # is (I - beta k k^T), whose eigenvalue is 1 - beta||k||^2, so any beta||k||^2 > 2
        # diverges.  Every DeltaNet/GDN/KDA implementation therefore L2-normalises the
        # keys (the write strength is carried entirely by beta).  Leaving this out
        # produces a layer that looks correct in a two-step unit test and explodes
        # after ~60 steps of training.
        k = torch.nn.functional.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)              # (B,H,T)
        raw = self.decay_proj(x)
        if self.cfg.kind in ("kda", "gdn2"):
            g = -F.softplus(raw.view(b, t, h, dk)).transpose(1, 2)           # (B,H,T,dk)
        else:
            g = -F.softplus(raw.view(b, t, h)).transpose(1, 2)               # (B,H,T)
            g = g * torch.nn.functional.softplus(self.A_log).view(1, h, 1)
        g = g.clamp(min=-self.cfg.decay_clamp)
        if self.cfg.kind == "gdn2":
            erase = torch.sigmoid(self.beta_erase_proj(x)).view(b, t, h, dk).transpose(1, 2)
            write = torch.sigmoid(self.beta_write_proj(x)).view(b, t, h, dk).transpose(1, 2)
        else:
            erase = write = beta
        return q, k, v, erase, write, g

    # ------------------------------------------------------------------ reference
    def forward_recurrent(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Literal per-token recurrence.  Returns ``(y, state)``."""
        q, k, v, erase, write, g = self._project(x)
        b, h, t, dk = k.shape
        dv = v.shape[-1]
        state = torch.zeros(b, h, dk, dv, dtype=x.dtype, device=x.device) if state is None else state
        outs = []
        for i in range(t):
            alpha = torch.exp(g[..., i, :] if g.dim() == 4 else g[..., i].unsqueeze(-1))
            ki, vi, qi = k[:, :, i], v[:, :, i], q[:, :, i]
            be, bw = erase[:, :, i], write[:, :, i]
            if be.dim() == 2:                     # scalar gates -> per-channel view
                be, bw = be[..., None], bw[..., None]
            state = state * alpha.unsqueeze(-1)
            if self.cfg.kind != "ssd":
                # erase: S <- S - (b * k) (k^T S)  (rank-1 removal along k)
                proj = torch.einsum("bhd,bhde->bhe", ki, state)
                state = state - torch.einsum("bhd,bhe->bhde", be * ki, proj)
            state = state + torch.einsum("bhd,bhe->bhde", bw * ki, vi)
            outs.append(torch.einsum("bhd,bhde->bhe", qi, state))
        y = torch.stack(outs, dim=2)
        return self.o_proj(y.transpose(1, 2).reshape(x.shape[0], t, -1)), state

    # ------------------------------------------------------------------ chunked
    def forward_chunked(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Chunkwise algorithm: matmuls plus one triangular solve per chunk."""
        q, k, v, erase, write, g = self._project(x)
        b, h, t, dk = k.shape
        dv = v.shape[-1]
        c = self.cfg.chunk
        state = torch.zeros(b, h, dk, dv, dtype=x.dtype, device=x.device) if state is None else state
        outs = []
        for start in range(0, t, c):
            stop = min(start + c, t)
            qc, kc, vc = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
            ec, wc = erase[:, :, start:stop], write[:, :, start:stop]
            gc = g[:, :, start:stop]
            if gc.dim() == 3:
                gc = gc.unsqueeze(-1)
            gc = gc.expand(b, h, stop - start, dk).cumsum(dim=2)         # (B,H,C,dk) <= 0
            out, state = self._chunk(qc, kc, vc, ec, wc, gc, state)
            outs.append(out)
        y = torch.cat(outs, dim=2)
        return self.o_proj(y.transpose(1, 2).reshape(x.shape[0], t, -1)), state

    def _chunk(self, q, k, v, erase, write, gc, state):
        """Dispatch to the symmetric (scalar-gate) or asymmetric chunk algorithm."""
        if self.cfg.kind == "gdn2":
            return self._chunk_asymmetric(q, k, v, erase, write, gc, state)
        beta = erase                                  # scalar-gate kinds: erase == write
        return self._chunk_symmetric(q, k, v, beta, gc, state)

    def _chunk_symmetric(self, q, k, v, beta, gc, state):
        """One chunk of the chunkwise algorithm (see module docstring)."""
        b, h, c, dk = k.shape
        dv = v.shape[-1]
        G = torch.exp(gc)
        d = k / G                                                        # erase-left / write-left
        e = k * G                                                        # erase-right
        qt = q * G                                                       # decayed query
        tril = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(-1)
        tril0 = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(0)

        # M[i, j] = beta_i * (e_i . d_j) for j < i
        read = torch.einsum("bhcd,bhde->bhce", e, state)                 # e_i^T S_0
        if self.cfg.kind == "ssd":
            m = torch.zeros(b, h, c, c, dtype=k.dtype, device=k.device)
            rhs = beta.unsqueeze(-1) * v
        else:
            m = torch.einsum("bhcd,bhjd->bhcj", e, d) * tril * beta.unsqueeze(-1)
            rhs = beta.unsqueeze(-1) * (v - read)
        eye = torch.eye(c, dtype=k.dtype, device=k.device)
        y = torch.linalg.solve_triangular(eye + m, rhs, upper=False)     # (B,H,C,dv)

        # state: S~_C = S~_0 + sum_j d_j y_j, then scale back by the chunk's decay
        normalised = state + torch.einsum("bhcd,bhce->bhde", d, y)
        new_state = torch.exp(gc[:, :, -1, :]).unsqueeze(-1) * normalised

        # outputs: o_i = q~_i S~_0 + sum_{j<=i} (q~_i . d_j) y_j
        qd = torch.einsum("bhcd,bhjd->bhcj", qt, d) * tril0
        out = torch.einsum("bhcd,bhde->bhce", qt, state) + torch.einsum("bhcj,bhje->bhce", qd, y)
        return out, new_state

    def _chunk_asymmetric(self, q, k, v, erase, write, gc, state):
        """Chunk algorithm with distinct channel-wise erase and write gates (GDN-2).

        With :math:`a_i = b_i k_i / G_i` (erase), :math:`c_i = w_i k_i / G_i`
        (write) and :math:`e_i = k_i G_i` (read), the normalised recurrence is

        .. math:: \\tilde S_i = \\tilde S_{i-1} - a_i s_i + c_i v_i^{\\top},
                  \\qquad s_i = e_i^{\\top}\\tilde S_{i-1}

        which gives one triangular system for all ``s_i``,
        :math:`(I + E_a) S = E\\tilde S_0 + E_c V` with
        :math:`E_a[i,j] = e_i\\cdot a_j`, :math:`E_c[i,j] = e_i\\cdot c_j`
        (``j < i``), and then
        :math:`\\tilde S_C = \\tilde S_0 - A^{\\top}S + C^{\\top}V`,
        :math:`o_i = \\tilde q_i^{\\top}\\tilde S_0 -
        \\sum_{j\\le i}(\\tilde q_i\\cdot a_j)s_j + \\sum_{j\\le i}(\\tilde q_i\\cdot c_j)v_j`.

        Setting ``b = w = beta`` collapses this to :meth:`_chunk_symmetric`, which
        ``tests/test_recurrent.py`` checks numerically so the two paths cannot drift.
        """
        b, h, c, dk = k.shape
        dv = v.shape[-1]
        G = torch.exp(gc)
        e = k * G                                                        # erase-right / read
        a = erase * k / G                                                # erase-left
        cw = write * k / G                                               # write-left
        qt = q * G
        tril = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(-1)
        tril0 = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(0)

        ea = torch.einsum("bhcd,bhjd->bhcj", e, a) * tril
        ec = torch.einsum("bhcd,bhjd->bhcj", e, cw) * tril
        rhs = torch.einsum("bhcd,bhde->bhce", e, state) + torch.einsum("bhcj,bhje->bhce", ec, v)
        rows = torch.linalg.solve_triangular(torch.eye(c, dtype=k.dtype, device=k.device) + ea, rhs, upper=False)

        normalised = state - torch.einsum("bhcd,bhce->bhde", a, rows) + torch.einsum("bhcd,bhce->bhde", cw, v)
        new_state = torch.exp(gc[:, :, -1, :]).unsqueeze(-1) * normalised

        qa = torch.einsum("bhcd,bhjd->bhcj", qt, a) * tril0
        qc = torch.einsum("bhcd,bhjd->bhcj", qt, cw) * tril0
        out = (torch.einsum("bhcd,bhde->bhce", qt, state)
               - torch.einsum("bhcj,bhje->bhce", qa, rows)
               + torch.einsum("bhcj,bhje->bhce", qc, v))
        return out, new_state

    def forward(self, x, state=None, mode: str = "chunked"):
        if mode == "chunked":
            return self.forward_chunked(x, state)
        if mode == "recurrent":
            return self.forward_recurrent(x, state)
        raise ValueError(f"unknown mode {mode!r}")

    # ------------------------------------------------------------------ accounting
    def state_bytes(self, batch: int = 1) -> int:
        return 4 * batch * self.cfg.n_heads * self.cfg.d_state * self.cfg.d_value

    def decode_flops_per_token(self, batch: int = 1) -> int:
        h, dk, dv = self.cfg.n_heads, self.cfg.d_state, self.cfg.d_value
        return int(2 * batch * (h * dk * dv * 3 + 3 * self.cfg.d_model * h * dk))
