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
    "linformer": """
        attn = build("linformer", AttnConfig(extras={"linformer_k": 64}))
        # non-causal by design (the projection mixes positions)
    """,
    "performer": """
        attn = build("performer", AttnConfig(extras={"features": 64}))   # causal, linear time
    """,
    "bigbird": """
        attn = build("bigbird", AttnConfig(window_size=256,
                                           extras={"window": 256, "n_random": 3, "n_global": 2}))
    """,
    "reformer_lsh": """
        attn = build("reformer_lsh", AttnConfig(extras={"buckets": 16}))
    """,
    "free_solo": """
        attn = build("free_solo", AttnConfig(window_size=128,
                                             extras={"cell": 64, "topk_cells": 4, "window": 128}))
    """,
    "namoh": """
        attn = build("namoh", AttnConfig(window_size=64))   # heads own token subsets
    """,
    "hla_mixing": """
        attn = build("hla_mixing", AttnConfig(extras={"gate_tau": 1.0}))
    """,
    "mass_oracle": """
        # analysis instrument (peeks at attention mass); not deployable
        attn = build("mass_oracle", AttnConfig(extras={"topk_blocks": 8, "sinks": 2}))
    """,
    "h2o": """
        attn = build("h2o", AttnConfig(cache_budget=512, window_size=64))
    """,
    "snapkv": """
        attn = build("snapkv", AttnConfig(cache_budget=512, obs_window=32,
                                          extras={"pool_kernel": 7}))
    """,
    "tova": """
        attn = build("tova", AttnConfig(cache_budget=512, window_size=64))
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
    "saga": """
        attn = build("saga", AttnConfig(n_kv_heads=8, topk_tokens=2048,
                                        extras={"key_heads": 2}))
    """,
    "pisa": """
        attn = build("pisa", AttnConfig(topk_tokens=1024,
                                        extras={"branch": 2, "topk_tokens": 1024}))
    """,
    "jssa": """
        attn = build("jssa", AttnConfig(topk_tokens=2048, extras={"rank": 16}))
    """,
    "masa": """
        # analysis oracle: uses the exact profile attention map
        attn = build("masa", AttnConfig(block_size=64, topk_blocks=16,
                                        extras={"block": 64, "topk_blocks": 16}))
    """,
    "oval": """
        attn = build("oval", AttnConfig(block_size=16, topk_blocks=8,
                                        extras={"page_size": 16, "rank": 8, "eta": 0.5}))
    """,
    "sfa": """
        attn = build("sfa", AttnConfig(head_dim=128,
                                       extras={"feature_keep": 32}))   # every token is kept
    """,
    "dropkv": """
        attn = build("dropkv", AttnConfig(cache_budget=512, obs_window=8,
                                          extras={"pool_kernel": 11}))
    """,
    "restkv": """
        attn = build("restkv", AttnConfig(cache_budget=512, obs_window=8,
                                          extras={"pool_kernel": 7}))
    """,
    "beaconkv": """
        attn = build("beaconkv", AttnConfig(cache_budget=512, window_size=64,
                                             extras={"n_beacons": 4}))
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
    "linformer": """
        The low-rank hypothesis in one line: attention is approximately rank-``E``,
        so project K/V to ``E`` dimensions and the cost stops growing with context.
        Non-causal, as published; it is the ancestor of every feature-axis compression
        (MLA included), which is why it stays in the catalogue despite never shipping.
    """,
    "performer": """
        The random-feature estimator is unbiased but high-variance: on random inputs we
        measured ~0.5 relative error at 16 features improving to ~0.46 at 1024. That is
        the honest picture of kernel approximations -- they converge slowly, and their
        advantage is asymptotic, which is why production chose *exact* sparse
        selection over approximation. Implemented with positive features to avoid the
        ``exp(+||x||^2/2)`` blow-up.
    """,
    "bigbird": """
        Window + random + global is the sparsity pattern with a universality proof. It
        is also the pattern most production systems rejected: random blocks are
        hardware-hostile and, unlike a learned selector, cannot adapt to content. The
        example of an academic result that is *right* and still not adopted.
    """,
    "reformer_lsh": """
        Hashing is the cheapest way to get sub-quadratic attention, and the least
        accurate on i.i.d. data (we measured relative error growing from 0 at one
        bucket to ~1.9 at 32). Its lasting contribution is the insight that *bucketing*
        can be a routing mechanism, which is what later learned indexers (DSA) do
        better.
    """,
    "free_solo": """
        2026 line: a window plus *keypoint cells* selected by score, with length
        extrapolation as the goal rather than raw benchmark parity. The cell-level
        selection is what makes it linear; the paper's learned prefix pruning and
        balanced-tree search are not reproduced here.
    """,
    "namoh": """
        2026 line: sparsity along the *head* axis rather than the token axis -- each
        head owns a token subset and attends causally inside it. With one head it is
        exact (tested), so the mechanism is a strict generalisation; what it argues
        against is the assumption that all heads must see the same tokens.
    """,
    "hla_mixing": """
        2026 line: a *data-dependent, per-chunk* gate between exact chunk-local
        attention and a linear state read. Same family as Infini-attention, but the
        gate is dynamic rather than a learned scalar -- the paper's claim is that the
        mixing policy, not the state, is what limits expressivity.
    """,
    "mass_oracle": """
        Not a method: the COBS oracle. It selects blocks by their exact attention mass
        (computed by running dense attention first), so it bounds what *any* selector
        could achieve at the same budget. Use it to calibrate a real selector -- if a
        deployable method is far below the oracle at equal reads, the gap is in the
        selector, not in the idea of sparsity.
    """,
    "h2o": """
        Re-introduced as academic lineage, not as a recommendation. Accumulated
        attention mass is a past statistic used to predict future usefulness; Random
        Attention (2026) showed the score barely beats random once the prompt and
        recent window are protected. Kept because new selectors need a baseline.
    """,
    "snapkv": """
        The prefill-time variant: score once from an observation window, pool the
        scores, keep the top keys plus the window. Its cost is paid once rather than
        per decode step -- the opposite trade-off from Quest/DSA -- which is why it is
        still a reasonable baseline for compression-at-prefill.
    """,
    "tova": """
        The direct rebuttal to H2O: the query that matters is the *current* one, so
        score with the last query only. Cheap, and a useful control for whether
        accumulated mass is buying anything.
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
    "saga": """
        2026 correction to the "just read fewer tokens" story: once the selected
        probability-value product is small, QK scoring and key metadata dominate, so
        allocating fewer heads to keys and more to values can be the right trade.
        This reference uses exact top-N for the inference semantics; the paper trains
        Atop-N and reports a conversion fine-tune for pretrained GQA models. The
        memory table is therefore the point: the key-side heads are configurable
        independently of the value-side heads.
    """,
    "pisa": """
        PISA's contribution is not another block score; it is removing the
        ``T x T`` *routing* term. The hierarchy is narrow at the top and only expands
        selected children, giving log-linear selection under a fixed token budget.
        This implementation uses mean-pooled node keys and exact attention over the
        final leaves. The paper's LogSumExp node score and fused Triton kernel are
        the next step; do not compare its wall clock to this Python reference.
    """,
    "jssa": """
        2026 line: token-level selection is finer than pages, but a low-dimensional
        ranking score can lose the structure that made the dense inner product useful.
        The reference takes the leading key-covariance subspace; the paper learns a
        shared subspace with a score-preservation objective and adds a gather-free
        kernel. At full rank and full token budget this reduces to dense attention,
        which is the correctness check.
    """,
    "masa": """
        The important negative result is conceptual: selecting the largest attention
        entries is the identity-Gram special case of preserving ``A V``, not the
        general objective. MASA's score is exact but expensive here -- it deliberately
        peeks at the profile attention map. Use it as an upper-bound/calibration
        instrument for a cheap selector, not as a deployable kernel.
    """,
    "oval": """
        2026 line in the KV-retrieval layer: page summaries should preserve the
        directions that matter to the *output*, not only key energy. The ``eta=0``
        endpoint is key-PCA; ``eta=1`` is the output-sensitive basis. The reference
        sets ``G=I`` and builds summaries from K/V alone; a production conversion
        would use the head's output projector and a calibrated ``eta``.
    """,
    "sfa": """
        This is the complementary axis to every mask in the repository: all tokens
        remain readable, but each query/key carries a sparse feature code. At
        ``feature_keep == head_dim`` the implementation is exactly dense attention.
        The learned codebook and FlashSFA overlap kernel are not reproduced; the
        top-magnitude code is a transparent inference-time stand-in.
    """,
    "dropkv": """
        This is the 2026 answer to the claim that eviction scores are exhausted: rank
        by the *output perturbation* of removing a token, not by attention mass.
        The score uses the observation window and values, which is why it can beat
        attention-only baselines on RULER. The paper's Triton scorer is the deployment
        path; this page keeps the reference score and the protected prompt/window
        invariant, so the comparison to `random_eviction` remains apples-to-apples.
    """,
    "restkv": """
        ReST-KV is the reconstruction/smoothing branch of the same 2026 revival.
        Its score is a single-token output-discrepancy proxy, then an EMA and a local
        max-pool make it robust to layer- and position-specific noise. The layer-wise
        reconstruction solve is simplified here; treat this as a reproducible baseline
        for the smoother, not as the authors' full system.
    """,
    "beaconkv": """
        BeaconKV targets the failure mode that made "recency is enough" look false on
        long reasoning traces: a later thought can revisit an early plan. Keeping
        representatives of past query clusters makes that revisit visible without
        retaining the full query history. The reference uses farthest-point
        representatives and attention mass; the paper learns the clusters and tunes
        them on long generation traces.
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
    """One page for the fixed-size-state mixers (SSD through Q-Delta/KDN/SwiLA).

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
        "The first four are one recurrence with different gates:",
        "",
        "```",
        "S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T",
        "o_t = q_t^T S_t",
        "```",
        "",
        "with the decay `a_t` scalar or channel-wise and the gates `b_t`/`w_t`",
        "collapsed or separate. That is why one chunkwise kernel with a flag covers",
        "the whole family -- and why GDN-2 is the general case rather than a new",
        "mechanism. Q-Delta adds query feedback, KDN makes the write gain come from",
        "tracked uncertainty, and SwiLA routes a mixture of delta-rule states.",
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
        "For SSD/GDN/KDA/GDN-2, `forward_chunked` and `forward_recurrent` are asserted",
        "equal in float64 (and in gradients) by `tests/test_recurrent.py`, so the",
        "chunked path is a verified re-association of the recurrence. Q-Delta, KDN and",
        "SwiLA currently expose the exact reference recurrence; their papers' WY /",
        "covariance-scan kernels are cited but not reproduced. `forward_chunked` is an",
        "explicit alias for those three rather than a claim of a serial-to-parallel",
        "transformation.",
        "",
        "## State and cost",
        "",
        "| kind | state floats (d_state=16, d_v=64) | state floats (d_state=128, d_v=128) |",
        "| --- | --- | --- |",
    ]
    for kind in MIXERS:
        small = build_mixer(kind, d_model=256, n_heads=4, d_state=16, expand_v=4.0)
        large = build_mixer(kind, d_model=2048, n_heads=16, d_state=128)
        # Use the layer's own accounting: KDN has an auxiliary uncertainty channel
        # and SwiLA stores one matrix per mixture expert.
        s_small = small.state_bytes(1) // 4
        s_large = large.state_bytes(1) // 4
        lines.append(f"| `{kind}` | {s_small:,} | {s_large:,} |")

    lines += [
        "",
        "SSD/GDN/KDA/GDN-2 and Q-Delta use the same matrix state. Diagonal KDN adds one",
        "uncertainty value per key channel; SwiLA multiplies the state by its number of",
        "mixture experts. The state table reports those auxiliary costs rather than",
        "hiding them in a fixed-size claim.",
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
        "(SSD / GDN / KDA / GDN-2 / Q-Delta / KDN / SwiLA).",
        "",
    ]
    (OUT / "index.md").write_text("\n".join(header + index_rows + [""]), encoding="utf-8")
    (OUT / "state_mixers.md").write_text(mixers_page(), encoding="utf-8")
    print(f"wrote {len(VARIANTS)} variant pages + index to {OUT}")
    print(f"wrote {OUT / 'state_mixers.md'} ({len(MIXERS)} state mixers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
