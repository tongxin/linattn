# Approximation theory map

This map attaches each relevant result to the object it actually constrains. The
key distinction is between a bound on the full attention operator, a bound on a
sparse support, a probabilistic estimator, and a statement about expressivity or
optimization. A theorem in one row does not automatically transfer to another.

Status is marked as **peer-reviewed**, **preprint**, or **foundational**. Dates
are the arXiv dates unless a venue is named.

## 1. Hard limits and sampling/streaming bounds

| work | status | object | result and assumptions | metric | key limitation |
| --- | --- | --- | --- | --- | --- |
| *Efficiently Approximating Attention Is Hard* (arXiv:2609.37261) | preprint, Sep 2026 | any truly subquadratic approximation over arbitrary inputs | Under standard complexity assumptions, no truly subquadratic algorithm gives a nontrivial uniform additive or relative approximation; even polynomial KV preprocessing cannot guarantee a small selected set in the worst case | worst-case operator/output error | Rules out a universal guarantee, not good performance on structured language inputs |
| *Attention via Black-Box Vector Search* (arXiv:2610.10135) | preprint, Oct 2026 | attention output from a MIPS oracle | One index requires Θ(√n/ε) retrieved keys; with Θ(log n) indices O(log n + 1/ε²) suffices and is near optimal; query/key augmentation can use O(1/ε²) | probabilistic ε-output error | Relies on a black-box MIPS oracle and a sampling/priority-sampling model |
| *Streaming Attention Approximation via Discrepancy Theory* (arXiv:2502.07861) | theory preprint | streaming attention approximation | BalanceKV constructs a balanced core-set with discrepancy-theoretic ε-approximation and space lower bounds | output/operator error and space | Input assumptions and the particular kernels matter; not a statement that every KV-cache heuristic is certified |
| *Towards Tight Bounds for Streaming Attention* (arXiv:2606.07205) | preprint, Jun 2026 | streaming KV-cache approximation | Combines discrepancy, polynomial and space-partition methods for nearly tight upper/lower space bounds | output approximation vs space | The bound is for the modeled attention primitive, not a full trained LM |
| *Certified Sparse Attention via Perturbed Retrieval* (arXiv:2610.24991) | preprint, Oct 2026 | sparse decode output with per-query certificates | Randomly perturbed retrieval yields nested samples and simultaneous confidence bounds; O(ε⁻² log(1/δ)) reads can meet a target error | probabilistic output error / certificate | Requires the sampling assumptions and calibrated perturbation; certificates are probabilistic |

The practical message is already strong: **there cannot be a universal cheap
approximation guarantee for arbitrary K/V.** A useful theory must condition on
input structure, a sampling model, or a task distribution.

## 2. Sparse support and dropped-mass bounds

| work | status | object | result and assumptions | metric | key limitation |
| --- | --- | --- | --- | --- | --- |
| *How Sparse Attention Approximates Exact Attention?* (arXiv:2404.02690) | preprint | sparse attention matrices | Characterizes an `n^C`-sparse regime and shows that stable `o(log n)` sparsity generally leaves `O(1)` error under its framework | attention/output error vs support size | Distributional assumptions and the definition of stability are load-bearing |
| SSA: *Sparse Sparse Attention* (arXiv:2511.20102) | preprint | sparse vs full attention outputs | Proves approximation error scales linearly with dropped attention mass and uses output alignment to reduce that mass | dropped mass / output error | It gives a useful sufficient quantity, not a claim that mass alone determines task quality |
| COBS (arXiv:2607.09052) | preprint | block selection | Reduces exact block selection to ranking blocks by attention mass; exact-mass selection matches dense, and a second-order proxy improves over first-order selection | block mass / output error | Requires a compact estimator for mass; block granularity and value geometry remain |
| CEDAR (arXiv:2609.07237) | preprint | coarse-to-fine search | Derives an output-error bound depending on within-chunk key/value dispersion and spends refinement where the error estimate is high | output error / dispersion | Uses a learned or heuristic coarse summary; the bound is model-dependent |
| *On the Trade-off Between Information Loss and Generalization in Sparse Attention* (arXiv:2610.04424) | preprint | sparse attention distribution and generalization | Gives a JS-divergence expression under i.i.d. sub-Gaussian score assumptions and Rademacher/mutual-information generalization bounds | JS divergence and generalization | The closed form is not valid for arbitrary correlated attention maps |

These results justify the metric hierarchy: dropped mass and JS divergence are
diagnostics, but the value-weighted output and hidden-state propagation are the
objects that ultimately connect to model behavior.

## 3. Low-rank, kernel and nearest-neighbor approximations

| work | status | object | result and assumptions | metric | key limitation |
| --- | --- | --- | --- | --- | --- |
| Linformer (ICML 2020) | foundational | sequence-axis low-rank attention | Johnson–Lindenstrauss-style projection argument for low-rank attention | operator/output error | Approximate low-rank and usually non-causal; projection mixes positions |
| Performer/FAVOR+ (ICML 2021) | foundational | softmax kernel via random features | Unbiased estimator and variance/concentration results for positive random features | estimator variance / expected error | High variance; unbiased does not imply a tight per-query error |
| Nyströmformer (NeurIPS 2021) | foundational | landmark matrix approximation | Reconstructs attention from a landmark sketch under matrix-approximation assumptions | operator approximation | Numerical conditioning and landmark coverage; the repository's small-scale implementation was unstable |
| `k`NN Attention Demystified (arXiv:2411.04013) | preprint | kNN attention and gradients | Reformulates attention as an expectation and uses lazy Gumbel sampling for approximate gradients | probabilistic output/gradient error | Not a deterministic guarantee for hard top-k inference |
| PolySketchFormer (arXiv:2310.01655) | preprint/theory | polynomial-kernel sketch | Sketching polynomial kernels to obtain near-linear attention | kernel/operator approximation | Polynomial degree, feature dimension and numerical error |

The shared lesson is that **low-rank, kernel and sampling methods have different
notions of approximation**. A low-rank projection error cannot be compared directly
with a sparse-support miss probability without a metric bridge.

## 4. Quantization and compressed-state theory

| work | status | object | result and assumptions | metric | key limitation |
| --- | --- | --- | --- | --- | --- |
| *Bridging KV-Cache Quantization and Linear Attention* (arXiv:2610.11214) | preprint | per-entry quantized KV vs recurrent aggregation | Shows soft address assignments can locally approximate full-attention similarity and support recurrent aggregation, enabling weight migration | local similarity / migrated accuracy | “Locally approximates” is weaker than a global softmax guarantee |
| RoLaQ (arXiv:2610.20686) | preprint | 2-bit MLA cache | Learns rotations/ranges under a frozen model and evaluates prediction/attention reconstruction | reconstruction and output error | Model-specific calibration; not a universal bound |
| *How Linear Attention Remembers* (arXiv:2609.33093) | preprint, Sep 2026 | fixed-state recurrent memory | Finds fact-specific writes/reads and cross-fact interference in GLA/GDN through causal interventions | interference / recall / editability | Architectural/empirical study, not a general approximation theorem |
| *Anatomy of Associative Recall in Fixed-State Recurrences* (arXiv:2609.16183) | preprint, Sep 2026 | associative recall capacity | Matched-state decomposition; interference and training curriculum determine whether a state budget is usable | recall vs state load | Controlled task, not full language modeling |

These papers explain why a scalar “state size” or “bits per value” is not enough:
the same state budget can support selective recall or fail through interference,
depending on the write/read rule and training.

## 5. Universal approximation results that are not efficiency guarantees

Two useful but easily misread results:

- *Learning Theory of Transformers: Local-to-Global Approximation via Softmax
  Partition of Unity* (arXiv:2605.08811) proves approximation and generalization
  rates for **dense, shallow, wide** Transformers. It does not bound a sparse or
  compressed implementation of that Transformer.
- *Approximation Theory for Lipschitz Continuous Transformers* (arXiv:2602.15503)
  proves universal approximation for a specially constructed Lipschitz class.
  It is a result about the function class, not about the runtime approximation of
  an existing attention layer.

These should be cited as context for what Transformers can represent, not as
evidence that a lower-cost implementation preserves the representation.

## 6. What the map implies for our metrics

The theory supports a four-level measurement program:

1. **support/distribution:** mass recall, TV, JS, Hellinger;
2. **operator/output:** value-weighted relative error and cosine distance;
3. **state:** hidden-state drift under the same input and layer intervention;
4. **model/task:** predictive distribution and task behavior, reported only with
   the first three levels.

No single metric is sufficient. Uniform worst-case approximation is impossible
under standard assumptions; structured-input and probabilistic guarantees are
possible. The next subproblem must therefore test whether the metrics distinguish
the mechanisms on controlled inputs rather than assuming they do.
