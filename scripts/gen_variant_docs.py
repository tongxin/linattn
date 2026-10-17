"""Generate one documentation page per variant from the code itself.

Why generate rather than hand-write: the numbers in the cost tables come from the
variant's own ``memory_bytes``/``read_budget``/``flops`` methods, and the
mechanism text comes from the class docstring.  A doc page therefore cannot drift
from the implementation without ``tests/test_docs.py`` noticing (`rel_err` and
the cost table are recomputed on demand, not pasted).

The parts that are *not* derivable -- caveats, usage snippets and the "when to use
it" judgement -- live in ``NOTES`` below, and are the hand-written content of this
script.

Run::

    python scripts/gen_variant_docs.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linattn.core import AttnConfig  # noqa: E402
from linattn.metrics import efficiency_table  # noqa: E402
from linattn.registry import MIXERS, VARIANTS, build, build_mixer, describe, describe_mixer  # noqa: E402

OUT = ROOT / "docs" / "variants"

# --------------------------------------------------------------------------------------
# hand-written content: what the code cannot say about itself
# --------------------------------------------------------------------------------------
USAGE: Dict[str, str] = {
    "dense": """
        from linattn.registry import build
        from linattn.core import AttnConfig

        attn = build("dense", AttnConfig(n_heads=32, n_kv_heads=8, head_dim=128))
        out, state = attn.forward(q, k, v)              # prefill
        out, state = attn.forward(q_t, k_t, v_t, state=state)   # one decode step
    """,
    "sliding_window": """
        attn = build("sliding_window", AttnConfig(window_size=4096, sink_size=4))
        out, state = attn.forward(q, k, v)
    """,
    "streaming_sink": """
        attn = build("streaming_sink", AttnConfig(window_size=1024, sink_size=4))
    """,
    "random_eviction": """
        attn = build("random_eviction", AttnConfig(cache_budget=2048, window_size=128, seed=0))
    """,
    "quest": """
        attn = build("quest", AttnConfig(block_size=16, topk_blocks=8, extras={"page_size": 16}))
        # keep the first two layers dense, as the paper does
        attn = build("quest", AttnConfig(block_size=16, topk_blocks=8,
                                         extras={"page_size": 16, "dense_layers": 2, "layer_idx": 6}))
    """,
    "minference": """
        # one pattern per head (the paper searches offline); "auto" searches per call
        attn = build("minference", AttnConfig(block_size=64, topk_blocks=8,
                                              extras={"pattern": "vertical_slash", "last_q": 64,
                                                      "n_vertical": 4, "n_slash": 4}))
    """,
    "flexprefill": """
        attn = build("flexprefill", AttnConfig(block_size=64, window_size=128,
                                               extras={"alpha_query_aware": 0.9, "alpha_structured": 0.95,
                                                       "tau_js": 0.05}))
    """,
    "xattention": """
        attn = build("xattention", AttnConfig(block_size=64, threshold=0.9,
                                              extras={"xattn_block": 8, "antidiagonal_stride": 4}))
        # per-head thresholds found offline by the paper's DP procedure can be supplied:
        attn = build("xattention", AttnConfig(block_size=64,
                                              extras={"threshold_per_head": [0.9] * 32, "xattn_block": 8}))
    """,
    "moba": """
        attn = build("moba", AttnConfig(block_size=4096, topk_blocks=12))
    """,
    "dsa": """
        attn = build("dsa", AttnConfig(block_size=64, topk_tokens=2048,
                                       extras={"indexer_heads": 64, "indexer_dim": 128, "compress_ratio": 1}))
        attn.fit(q_calib, k_calib, v_calib, steps=300)     # the indexer must be trained
        # compressed variant (CSA-style): index pooled keys, expand selected groups
        csa = build("dsa", AttnConfig(topk_tokens=2048, extras={"compress_ratio": 8}))
    """,
    "seerattention_r": """
        attn = build("seerattention_r", AttnConfig(block_size=64,
                                                   extras={"pool_dim": 16, "seer_topn": 8}))
        attn.fit_gate(q_calib, k_calib, v_calib, steps=300)
    """,
    "nsa": """
        attn = build("nsa", AttnConfig(block_size=64, window_size=512, topk_blocks=16,
                                       compress_ratio=16, extras={"selected_blocks": 16,
                                                                  "gate_mode": "learned"}))
        attn.fit_gates(q_calib, k_calib, v_calib, steps=200)
    """,
    "mla": """
        # the cache stores the latent; K/V are materialised inside the kernel
        attn = build("mla", AttnConfig(n_heads=128, n_kv_heads=1, head_dim=128,
                                       extras={"latent_dim": 512}))
        print(attn.compression_ratio())     # ~9x for DeepSeek-V2-style shapes
    """,
}

NOTES: Dict[str, str] = {
    "dense": """
        Keep this as the oracle. Every other page's numbers are only interpretable
        relative to it, and the test-suite validates every variant against it at an
        unrestricted budget.
    """,
    "sliding_window": """
        The cheapest thing that works, and the branch that every production sparse
        configuration keeps unconditionally. Beware of measuring it on retrieval
        tasks: a sliding window *cannot* retrieve, so its score on RULER is a
        measurement of how much of that benchmark is local.
    """,
    "streaming_sink": """
        Included because "StreamingLLM-style" is the vocabulary used in serving
        stacks for *permanently dropping* evicted tokens, which is subtly different
        from masking reads at attention time. The mask-based variant here has the
        same arithmetic as the sliding window; the difference is the cache contract.
    """,
    "random_eviction": """
        This is the most useful page in the eviction family. It keeps the prompt and
        the recent window and randomises everything else, and on reasoning workloads
        it matches tuned selectors (arXiv:2609.03430). Any eviction method you are
        considering should be compared against this baseline; if it does not beat it
        clearly, the scoring machinery is not earning its keep.
    """,
    "quest": """
        Quest's bound is not an estimate: channel-wise min/max give a true upper bound
        on the page's best logit, which is why it rarely misses the top token. Two
        caveats are reproduced in the implementation because they matter in practice:
        the first layers should stay dense (their sparsity is low), and under GQA the
        per-head page selections must be unioned, so the effective budget can inflate
        by the group ratio unless the kernel handles head divergence explicitly.
    """,
    "minference": """
        The pattern search is the interesting part: the paper assigns each head a
        pattern offline (`pattern="a_shape" | "vertical_slash" | "block_sparse"`) and
        then builds dynamic indices per prompt. `pattern="auto"` here performs the
        search online, on the first chunk, by mass recall at equal read budget.
        `reports/suite_*.md` shows what happens when you do that: the warm-up chunk is
        short, so its attention is local, so the search picks a local pattern and
        keeps it for the rest of the sequence. The paper's offline per-head search
        over long representative prompts is therefore load-bearing rather than a
        detail -- an online version needs a long probe (or a per-head prior) to behave
        the same way. Note too the 1x64 block size for vertical-slash: a "vertical
        line" is one token wide, which is exactly the kind of pattern that
        block-granular kernels handle badly.
    """,
    "flexprefill": """
        The two thresholds are the method's character: 0.9 for heads whose pattern is
        genuinely query-dependent, 0.95 for heads where one estimate serves every
        query. In practice most heads in most models resolve to the structured regime,
        which is consistent with MInference's finding that static patterns cover a lot
        of mass -- and it is a warning against assuming per-query adaptivity is paying
        for itself.
    """,
    "xattention": """
        The antidiagonal argument is the most transferable idea in the block-selection
        family: a strided sum of 1/stride of the block's entries is a good proxy
        because every vertical/slash/dense pattern intersects antidiagonals. The
        paper's per-head thresholds come from an offline DP; if you skip that step you
        get a uniformly-thresholded variant, which is what the config knob here
        defaults to. Its strongest evidence is video, where block structure is
        stronger than in text -- do not extrapolate the 13.5x number to language
        prefill.
    """,
    "moba": """
        MoBA's selling point is conceptual: sliding-window and sink attention fall out
        as special cases of the router, so the model can learn them instead of being
        told. The released inference recipe (block size 4096, top-12 blocks, 100B tokens
        of continual pretraining with the pattern active) is the configuration that
        reports parity with full attention at 1M context; a training-free MoBA-style
        router on a dense model will not reproduce that.
    """,
    "dsa": """
        The lightning indexer is cheap per key but touches every key, so its cost grows
        quadratically while the attention it gates grows linearly: at long context the
        indexer becomes the bottleneck (the 256 GB intermediate in the naive CSA
        implementation is the extreme version of this). Three mitigations are in the
        literature and matter more than the selection rule itself: cross-layer index
        reuse, hierarchical/compressed indexing, and streaming top-k without
        materialising scores. `fit` exists because an unfitted indexer is meaningless:
        run it on your own calibration data before believing any selection-recall
        number.
    """,
    "seerattention_r": """
        The gate is learned, and this page is the place to see why that matters: a
        random gate selects essentially at random, while a fitted one can approach the
        dense block distribution. In production this family needs a fine-tuning stage,
        which is why it appears in the "can you afford to retrain?" branch of the
        [frontier gap analysis](../frontier-gap.md) rather than the plug-and-play one.
    """,
    "nsa": """
        Read the three branches as three different bets: the compressed branch buys
        global coverage cheaply, the selected branch buys precision, the window branch
        buys locality and stops the other two from being shortcut. The gates are
        learned, so `gate_mode="uniform"` is an inference-time proxy and
        `fit_gates` is how the branch weights can be recovered without a full
        pretraining run. NSA's reported quality wins over full attention are the
        strongest evidence in this repository's taxonomy that *native* sparsity can
        help rather than merely preserve quality.
    """,
    "mla": """
        MLA is not sparsity -- and that is the point to take away. Compressing the
        feature axis (one latent per token) is the cheapest large win available and it
        composes with everything else: DSA sits on top of MLA, and V4.1-Flash adds FP4
        and cross-layer reuse on top of the compressed latent. The `compression_ratio`
        here reports the KV-footprint reduction for DeepSeek-V2-style shapes.
    """,
    "csa2": """
        This is the **per-layer** CSA2 mechanism: pooled groups, an FP4/FP8 cache with
        per-group scales, indexer top-k over groups, and a raw local window kept disjoint
        from the pooled groups so no token is counted twice. It is *not* the released
        layer: there is no trained indexer, no published layer schedule, and the FP4 path
        is a simulated codec (packed byte accounting, FP32 arithmetic after
        dequantisation). Cross-layer reuse lives outside the variant, in
        `kv/reuse.py` + `kv/csa2.py`, because it is a property of the layer *stack*.
        `reports/csa2_model.md` shows the footprint decomposition (the reported 890
        B/token is not what one-copy-per-four-layers yields) and the divergence that
        sharing introduces as neighbouring layers disagree.
    """,
}


def clean(doc: str) -> str:
    return textwrap.dedent(doc).strip()


def cost_table(name: str) -> str:
    cfg = AttnConfig(n_heads=32, n_kv_heads=8, head_dim=128, block_size=64, window_size=512,
                     cache_budget=4096, topk_blocks=16, topk_tokens=2048, compress_ratio=16)
    variant = build(name, cfg)
    rows = ["| T | state (MiB) | attn FLOPs (G) | KV reads/query |", "| --- | --- | --- | --- |"]
    for t in (32768, 131072, 1048576):
        rows.append(f"| {t:,} | {variant.memory_bytes(t) / 2**20:.1f} | {variant.flops(t) / 1e9:,.0f} "
                    f"| {variant.read_budget(t):,.0f} |")
    return "\n".join(rows)


def mixers_page() -> str:
    """One page for the fixed-size-state mixers (SSD / GDN / KDA / GDN-2).

    They are documented separately from the attention variants because their
    interface differs: they take ``x`` of shape ``(B, T, d_model)`` and own their
    projections, rather than consuming pre-projected ``(q, k, v)``.
    """
    lines = [
        "# State mixers (compressed state)",
        "",
        "These are the models that replaced the growing KV cache with a fixed-size",
        "recurrent state. They live in `src/linattn/variants/recurrent.py` and are",
        "**not** `AttentionVariant`s: they take `x` of shape `(B, T, d_model)` and own",
        "their q/k/v projections, so they cannot be driven through the `(q, k, v)`",
        "harness. `python -m linattn.cli variants` lists them alongside the attention",
        "variants, and `python -m linattn.cli mixers` prints their state size and",
        "per-token cost.",
        "",
        "All four are one recurrence with different gates:",
        "",
        "```",
        "S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T",
        "o_t = q_t^T S_t",
        "```",
        "",
        "with the decay `a_t` scalar or channel-wise and the gates `b_t`/`w_t`",
        "collapsed or separate. That is why one chunkwise kernel with a flag covers",
        "the whole family -- and why GDN-2 is the general case rather than a new",
        "mechanism.",
        "",
        "| kind | decay | gates | recurrence | citation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind in MIXERS:
        meta = describe_mixer(kind)
        cite = f"[{meta['citation'][0]}]({meta['citation'][1]})" if meta["citation"] else ""
        lines.append(f"| `{kind}` | {meta['decay']} | {meta['gates']} | `{meta['recurrence']}` | {cite} |")

    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "from linattn.registry import build_mixer",
        "",
        "layer = build_mixer(\"kda\", d_model=2048, n_heads=16, d_state=128, chunk=64)",
        "y_chunked, state = layer.forward_chunked(x)     # the fast path (matmuls + one solve)",
        "y_exact, state2 = layer.forward_recurrent(x)    # the literal per-token recurrence",
        "```",
        "",
        "`forward_chunked` and `forward_recurrent` are asserted equal in float64 (and in",
        "gradients) by `tests/test_recurrent.py`, so the chunked path is a verified",
        "re-association of the recurrence rather than an approximation.",
        "",
        "## State and cost",
        "",
        "| kind | state floats (d_state=16, d_v=64) | state floats (d_state=128, d_v=128) |",
        "| --- | --- | --- |",
    ]
    for kind in MIXERS:
        small = build_mixer(kind, d_model=256, n_heads=4, d_state=16, expand_v=4.0)
        large = build_mixer(kind, d_model=2048, n_heads=16, d_state=128)
        s_small = small.cfg.n_heads * small.cfg.d_state * small.cfg.d_value
        s_large = large.cfg.n_heads * large.cfg.d_state * large.cfg.d_value
        lines.append(f"| `{kind}` | {s_small:,} | {s_large:,} |")

    lines += [
        "",
        "The state size and the per-token FLOPs are **identical across kinds** -- only",
        "the update rule (and therefore quality) differs. What changes is capacity and",
        "expressivity: `reports/state_capacity.md` measures how many associations a",
        "state of a given size can hold before recall collapses.",
        "",
        "## Layer interleaving",
        "",
        "`variants/hybrid.py` builds `HybridLM`, which interleaves these layers with",
        "full-attention layers (the Kimi Linear 3:1 pattern). `experiments/hybrid_recall.py`",
        "compares dense, hybrid and linear-only stacks on MQAR, and states honestly that",
        "at laptop scale MQAR does not separate the families.",
        "",
        "---",
        "",
        "[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md)",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index_rows = ["| variant | family | training | mechanism |", "| --- | --- | --- | --- |"]
    for name in VARIANTS:
        meta = describe(name)
        cls = VARIANTS[name]
        doc = clean(cls.__doc__ or "")
        first_para = doc.split("\n\n")[0].replace("\n", " ")
        citation = meta["citation"]
        cite_line = f"{citation[0]} -- {citation[1]}" if citation else "_(no citation recorded)_"
        body = [
            f"# `{name}`",
            "",
            f"**Family:** `{meta['family']}` &nbsp;|&nbsp; **Training required:** "
            f"{'yes' if meta['needs_training'] else 'no'} &nbsp;|&nbsp; **Class:** `{meta['class']}`",
            "",
            cite_line,
            "",
            "## Mechanism",
            "",
            doc or "_(no docstring)_",
            "",
            "## Usage",
            "",
            "```python",
            clean(USAGE.get(name, "from linattn.registry import build\nattn = build(\"" + name + "\")")),
            "```",
            "",
            "## Cost model",
            "",
            "Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, "
            "block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:",
            "",
            cost_table(name),
            "",
            "## What to watch out for",
            "",
            clean(NOTES.get(name, "_(no notes yet)_")),
            "",
            "## Validation",
            "",
            "```bash",
            f"python -m pytest tests -q -k {name}",
            f"python -m linattn.cli select --variant {name} --seq 1024 --structured",
            "```",
            "",
            "Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or "
            "`tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` "
            "(the accelerated path for this family).",
            "",
            "---",
            "",
            "[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; "
            "[evaluation evidence](../evaluation-evidence.md)",
            "",
        ]
        (OUT / f"{name}.md").write_text("\n".join(body), encoding="utf-8")
        index_rows.append(f"| [`{name}`]({name}.md) | {meta['family']} | "
                          f"{'yes' if meta['needs_training'] else 'no'} | {first_para[:150]} |")

    header = [
        "# Variant index",
        "",
        "One page per implemented variant, generated from the code by "
        "`scripts/gen_variant_docs.py`. The taxonomy behind the grouping is in "
        "[the survey](../survey.md#3-taxonomy).",
        "",
        "The fixed-size-state models are documented separately, because they take "
        "`x` instead of `(q, k, v)`: **[state mixers](state_mixers.md)** "
        "(SSD / Gated DeltaNet / KDA / Gated DeltaNet-2).",
        "",
    ]
    (OUT / "index.md").write_text("\n".join(header + index_rows + [""]), encoding="utf-8")
    (OUT / "state_mixers.md").write_text(mixers_page(), encoding="utf-8")
    print(f"wrote {len(VARIANTS)} variant pages + index to {OUT}")
    print(f"wrote {OUT / 'state_mixers.md'} ({len(MIXERS)} state mixers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
