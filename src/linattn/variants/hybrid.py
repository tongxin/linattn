"""Hybrid stacks: interleaving full-attention layers with fixed-state layers.

Every frontier open-weight model that adopted compressed state did so *hybrid*:
MiniMax-01 (lightning attention every 8th layer), Qwen3-Next, Kimi Linear / K3
(KDA with a 3:1 pattern and, in K3, one full-attention layer every four), and
DeepSeek-V4 (compressed sparse attention with a causal encoder-decoder).  The
reason is the *recall* problem: a fixed-size state cannot do exact retrieval, so
the few full-attention layers carry the retrieval heads while the linear layers
carry the bulk of the sequence mixing cheaply.

Rather than pretend to reproduce a frontier model, this module builds small
hybrids and evaluates them on the task family where the trade-off is sharpest:
multi-query associative recall (MQAR), the standard synthetic probe for
associative memory.  ``experiments/hybrid_recall.py`` reports the curves that go
into ``docs/variants/compressed_state.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from linattn.variants.recurrent import LinearAttentionConfig, LinearAttentionLayer

__all__ = ["HybridConfig", "HybridLM", "make_mqar_batch", "train_mqar"]


@dataclass
class HybridConfig:
    vocab_size: int = 64
    d_model: int = 96
    n_layers: int = 6
    n_heads: int = 4
    d_state: int = 32
    kind: str = "gdn"                 # linear-layer kind: ssd | gdn | kda
    attn_period: int = 4              # one attention layer every `attn_period` layers (0 = none)
    attn_kind: str = "dense"          # dense | sliding | gdn
    window: int = 64
    seq_len: int = 128
    chunk: int = 16                   # chunkwise-scan chunk for the linear layers
    decay_clamp: float = 0.25         # log-decay lower bound (see docs/kernels.md)
    use_linear_layers: bool = True
    seed: int = 0


class _AttentionLayer(torch.nn.Module):
    """Minimal MHA used inside the hybrids (optionally windowed)."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = torch.nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, state: Optional[Dict] = None, offset: int = 0):
        b, t, _ = x.shape
        h, d = self.n_heads, self.d_head
        qkv = self.qkv(x).view(b, t, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if state is not None:
            k = torch.cat([state["k"], k], dim=-2)
            v = torch.cat([state["v"], v], dim=-2)
            state["k"], state["v"] = k, v
        t_k = k.shape[-2]
        qi = torch.arange(t, device=x.device).unsqueeze(1) + offset
        kj = torch.arange(t_k, device=x.device).unsqueeze(0)
        allow = kj <= qi
        if self.cfg.attn_kind == "sliding":
            allow = allow & ((qi - kj) < self.cfg.window)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d**0.5)
        scores = scores.masked_fill(~allow, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out(out.transpose(1, 2).reshape(b, t, -1))


class _LinearLayer(torch.nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.mixer = LinearAttentionLayer(
            LinearAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_state=cfg.d_state,
                kind=cfg.kind,
                chunk=min(cfg.chunk, cfg.seq_len),
                decay_clamp=cfg.decay_clamp,
            )
        )

    def forward(self, x, state=None, offset: int = 0):
        y, new_state = self.mixer.forward_chunked(x, state)
        return y, new_state

    def state_bytes(self, batch: int = 1) -> int:
        return self.mixer.state_bytes(batch)


class _Block(torch.nn.Module):
    def __init__(self, cfg: HybridConfig, layer_kind: str):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(cfg.d_model)
        self.mixer = _AttentionLayer(cfg) if layer_kind == "attn" else _LinearLayer(cfg)
        self.norm2 = torch.nn.LayerNorm(cfg.d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(cfg.d_model, 2 * cfg.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(2 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, state=None, offset: int = 0):
        if isinstance(self.mixer, _AttentionLayer):
            h = self.norm1(x)
            y = self.mixer(h, state, offset)
        else:
            y, state = self.mixer(self.norm1(x), state, offset)
        x = x + y
        return x + self.mlp(self.norm2(x)), state


class HybridLM(torch.nn.Module):
    """A small LM whose layer kinds follow the interleaving pattern."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        kinds = []
        for layer in range(cfg.n_layers):
            use_attn = (not cfg.use_linear_layers) or (
                cfg.attn_period > 0 and layer % cfg.attn_period == 0
            )
            kinds.append("attn" if use_attn else "linear")
        self.kinds = kinds
        self.blocks = torch.nn.ModuleList([_Block(cfg, k) for k in kinds])
        self.norm = torch.nn.LayerNorm(cfg.d_model)
        self.head = torch.nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens, states=None, offset: int = 0):
        x = self.embed(tokens)
        states = states or [None] * len(self.blocks)
        new_states = []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state, offset)
            new_states.append(state)
        return self.head(self.norm(x)), new_states

    def state_bytes(self, seq_len: int = 1, batch: int = 1) -> int:
        """Persistent state for a sequence of ``seq_len`` tokens.

        The two families scale differently, which is the whole reason hybrids exist:
        an attention layer costs ``2 * n_heads * d_head`` numbers *per token* (linear
        in sequence length), while a linear layer costs ``n_heads * d_state * d_value``
        numbers *per sequence* (constant).  Note the constant: a Gated DeltaNet layer
        with ``d_state=32`` and ``d_value=24`` holds 768 numbers per head, which is
        the equivalent of ``d_state * d_value / (2 * d_head) = 16`` tokens of KV cache
        per head.  That is the trade-off in one number.
        """
        total = 0
        for kind, block in zip(self.kinds, self.blocks):
            if kind == "attn":
                d_head = self.cfg.d_model // self.cfg.n_heads
                total += 2 * batch * self.cfg.n_heads * d_head * 4 * seq_len
            else:
                total += block.mixer.state_bytes(batch)
        return total


# --------------------------------------------------------------------------------------
# multi-query associative recall (MQAR)
# --------------------------------------------------------------------------------------
def make_mqar_batch(
    batch: int,
    seq_len: int,
    vocab_size: int,
    *,
    n_pairs: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Build the standard multi-query associative recall batch.

    Layout: ``k_1 v_1 ... k_n v_n | q_1 a_1 ... q_n a_n`` -- the whole key-value list
    comes first, then queries whose answers are the values bound to those keys
    earlier in the sequence.  Only the answer slots carry loss.

    Getting the layout wrong makes the task trivial in a way that is easy to miss:
    with ``k v k v`` per pair the answer slot is *adjacent to its own query and to a
    copy of its own value*, so a model can solve it by copying the previous token, and
    every architecture scores 100% (we measured exactly that before fixing it here).
    """
    gen = torch.Generator().manual_seed(seed)
    half = vocab_size // 2
    used = 4 * n_pairs
    if used > seq_len:
        raise ValueError(f"n_pairs={n_pairs} needs {used} tokens > seq_len={seq_len}")
    if half < n_pairs:
        raise ValueError(f"vocab_size={vocab_size} is too small for {n_pairs} unique keys")
    keys = torch.stack([torch.randperm(half, generator=gen)[:n_pairs] for _ in range(batch)])
    vals = torch.stack([torch.randperm(vocab_size - half, generator=gen)[:n_pairs] + half
                        for _ in range(batch)])
    seq = torch.zeros(batch, used, dtype=torch.long)
    target = torch.full((batch, used), -100, dtype=torch.long)
    # prefix: the key-value list
    pair_order = torch.stack([torch.randperm(n_pairs, generator=gen) for _ in range(batch)])
    prefix = (pair_order.unsqueeze(-1) * 2 + torch.arange(2)).reshape(batch, 2 * n_pairs)
    kv = torch.stack([keys, vals], dim=-1).reshape(batch, 2 * n_pairs)
    seq[:, : 2 * n_pairs] = torch.gather(kv, 1, prefix)
    # suffix: queries (a permutation of the same keys) followed by their answers
    query_order = torch.stack([torch.randperm(n_pairs, generator=gen) for _ in range(batch)])
    q_keys = torch.gather(keys, 1, query_order)
    q_vals = torch.gather(vals, 1, query_order)
    seq[:, 2 * n_pairs :: 2] = q_keys
    seq[:, 2 * n_pairs + 1 :: 2] = q_vals
    target[:, 2 * n_pairs + 1 :: 2] = q_vals
    pad = seq_len - used
    if pad:
        seq = torch.cat([seq, torch.zeros(batch, pad, dtype=torch.long)], dim=1)
        target = torch.cat([target, torch.full((batch, pad), -100, dtype=torch.long)], dim=1)
    return seq, target


def train_mqar(
    cfg: HybridConfig,
    *,
    steps: int = 400,
    batch: int = 16,
    lr: float = 3e-3,
    eval_every: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train a hybrid on MQAR and return accuracy/state-size statistics."""
    torch.manual_seed(seed)
    model = HybridLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: List[float] = []
    for step in range(steps):
        seq, tgt = make_mqar_batch(batch, cfg.seq_len, cfg.vocab_size, seed=seed + step)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if eval_every and (step + 1) % eval_every == 0:
            acc = evaluate_mqar(model, cfg, seed=seed + 10_000)
            history.append(acc)
            if verbose:
                print(f"step {step + 1:4d} loss {float(loss):.4f} acc {acc:.3f}")
    final = evaluate_mqar(model, cfg, seed=seed + 20_000)
    return {
        "accuracy": final,
        "loss": float(loss),
        "state_bytes": float(model.state_bytes(1)),
        "history": history,  # type: ignore[dict-item]
    }


@torch.no_grad()
def evaluate_mqar(model: HybridLM, cfg: HybridConfig, *, batch: int = 32, seed: int = 0) -> float:
    seq, tgt = make_mqar_batch(batch, cfg.seq_len, cfg.vocab_size, seed=seed)
    logits, _ = model(seq)
    mask = tgt != -100
    pred = logits.argmax(-1)
    return float((pred[mask] == tgt[mask]).float().mean())
