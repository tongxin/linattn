# Direction-lens coverage across attention families

The direction lens asks whether an approximation changes the output mainly by
changing its magnitude or by rotating it. We now report it across the mechanism
families instead of applying it only to sparse selection.

For a reference output `O` and approximation `O_hat`:

```text
error = O − O_hat
parallel = projection of error onto O
orthogonal = error − parallel
```

The report records:

- output cosine distance;
- cosine of error to the reference and estimate;
- parallel-error fraction;
- orthogonal-error fraction;
- estimate/reference norm ratio.

The fractions are descriptive: a large orthogonal fraction means the
approximation rotates the output, while a large parallel fraction means it mostly
shrinks or expands it.

## Coverage by family

| family | direction assessment | current evidence |
| --- | --- | --- |
| sparse selection | support mass + omitted/selected value centroids + output-error direction | `approx-validation`, `theory-checks` |
| kernel/random features | output direction and parallel/orthogonal error decomposition | `approx-validation`, `theory-checks` |
| low-rank projection | output direction; noncausal reference for Linformer | `approx-validation` |
| KV precision | output direction plus key- vs value-induced component cosine | `theory-checks` |
| fixed-size state | retrieval cosine plus error direction and parallel/orthogonal split | `state-capacity` |
| learned selector | output direction before/after fitting for NSA; support metrics for DSA/SeerAttention-R | `selector-capacity` |

## Observed patterns

### Sparse selection

On the structured-input metric matrix, the sparse selectors' error was mostly
orthogonal to the reference. For example, Quest/ MInference/ MoBA had orthogonal
error fractions around `0.96–0.999` in the tested rows. The sparse output was not
simply a scaled dense output; it changed direction.

The centroid certificate explains why: omitted and selected value centroids were
nearly orthogonal (`c ≈ 0`). The missing value direction is not recoverable by
renormalising the selected support.

### Kernel and low-rank methods

Performer and Linformer expose no causal sparse support, so their direction
assessment is at the output level rather than the centroid level.

- Performer's orthogonal error fraction was about `0.77` on random inputs and
  `0.37` on structured inputs.
- Linformer's orthogonal fraction was about `0.94` on random inputs and `0.72`
  on structured inputs (noncausal reference).

This is a different failure geometry from sparse selection: a kernel estimator
can alter output direction even when it is not making a hard support decision.

### Quantized KV

The quantisation output error had an orthogonal fraction around `0.999` on the
tested inputs. The key-induced and value-induced component errors were also
nearly orthogonal (`component_cos ≈ −0.02–0.02`). The directional component norm
was about `0.70–0.75` of the triangle bound. This validates the value of reporting
components instead of only a total norm.

### Fixed-state recurrence

At low associative load, state retrieval errors were almost entirely orthogonal
to the value direction. At high load, the error acquired an increasing parallel
component. The mechanism is visible in the decomposition

```text
retrieved = α · v_true + u,     u approximately orthogonal to v_true
error = (α−1) v_true + u
```

The parallel error fraction is proportional to `|α−1|`; the orthogonal fraction
comes from the interference `u`. At low load, cross-key interference is small
and random-like, while `α` remains close to one. At high load, the fixed state
saturates: keys collide, the surviving self coefficient `α` shrinks, and
`(α−1)v_true` dominates the error. The measured components are:

| associations | d_state | orthogonal fraction | parallel fraction | retrieval cosine |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 128 | 0.980 | 0.166 | 0.985 |
| 64 | 128 | 0.876 | 0.428 | 0.811 |
| 256 | 128 | 0.689 | 0.679 | 0.438 |
| 1024 | 128 | 0.566 | 0.806 | 0.122 |

This is a different mechanism from sparse omission: as a fixed state saturates,
the recovery failure increasingly looks like a magnitude/parallel loss, not only
a rotation. The direction lens therefore reveals capacity saturation rather than
just retrieval error.

The current experiment is a **decayed additive state**, not the full delta-rule
erase used by GDN/DeltaNet/KDA. Its capacity curve establishes the direction
mechanism for this surrogate; a GDN/KDA-specific claim needs the corresponding
erase recurrence.

### Learned selectors

For the fitted NSA branch gate, output direction improved only slightly
(`cosine distance 0.292 -> 0.284`) while the orthogonal error fraction changed
substantially (`0.505 -> 0.706`). This is a warning that “better” in one scalar
does not imply a cleaner mechanism; selector fitting can change the geometry of
the remaining error.

## Interpretation

The direction lens is useful, but it should not be applied as a universal scalar:

- For sparse selection, it explains the cost of omitting a value direction.
- For kernel/low-rank methods, it distinguishes scale error from rotation.
- For quantisation, it exposes cancellation between key and value perturbations.
- For fixed-state recurrence, it distinguishes low-load directional noise from
  high-load capacity collapse.
- For learned selectors, it shows whether fitting changes the mechanism or only
  the aggregate error.

The next step is to strengthen each family separately—value subspaces for sparse
selection, covariance/feature direction for kernels, key/value component geometry
for precision, and interference direction for compressed state—rather than force
one certificate across all of them.
