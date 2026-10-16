/*
 * Paged decode attention: the kernel every sparse-attention paper is ultimately
 * arguing about.
 *
 * Decoding one token means reading a handful of KV vectors for every query and
 * doing ~2*dim FLOPs per byte read, so the kernel is *memory bound*: what matters
 * is how many bytes it touches and whether it touches each byte once.  Two
 * implementations are provided so the difference is measurable on a CPU (no GPU
 * required):
 *
 *   paged_decode_naive  -- walks the selected pages key by key, computing the
 *                          score and then updating the accumulator.  Simple, and
 *                          the reference for correctness.
 *   paged_decode_tiled  -- loads a BLOCK x dim tile of K (and later of V) into a
 *                          stack buffer once, computes BLOCK scores from the
 *                          buffer, and updates the accumulator once.  This is the
 *                          IO-aware organisation FlashAttention is built on: one
 *                          read of each byte, arithmetic intensity kept in
 *                          registers.
 *
 * Both consume the same "selection": an array of page ids, which is what Quest,
 * MInference, XAttention, MoBA, NSA and DSA all produce.  The only difference
 * between those methods, from this kernel's point of view, is which page ids it
 * receives -- the argument of the survey in one sentence.
 *
 * Build: cc -O3 -ffast-math -shared -fPIC -o decode_attn.dylib decode_attn.c
 */

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define BLOCK 64

static inline float maxf_local(float a, float b) { return a > b ? a : b; }

/* Single-query attention over a list of pages, one key at a time. */
void paged_decode_naive(const float *q, const float *k, const float *v,
                        const int32_t *page_ids, int n_pages, int page_size,
                        int dim, int n_tokens, float scale, float *out) {
    float m = -INFINITY;
    float l = 0.0f;
    for (int d = 0; d < dim; ++d) out[d] = 0.0f;

    for (int p = 0; p < n_pages; ++p) {
        int start = page_ids[p] * page_size;
        for (int j = 0; j < page_size; ++j) {
            int idx = start + j;
            if (idx >= n_tokens) break;
            const float *krow = k + (size_t)idx * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) s += q[d] * krow[d];
            s *= scale;
            float m_new = maxf_local(m, s);
            float alpha = expf(m - m_new);
            float p_ij = expf(s - m_new);
            l = l * alpha + p_ij;
            const float *vrow = v + (size_t)idx * dim;
            for (int d = 0; d < dim; ++d) out[d] = out[d] * alpha + p_ij * vrow[d];
            m = m_new;
        }
    }
    for (int d = 0; d < dim; ++d) out[d] /= (l > 0.0f ? l : 1.0f);
}

/*
 * Tiled variant: one load of K and one load of V per byte, scores computed from
 * the buffered tile.  BLOCK is a compile-time constant so the inner loops
 * unroll and the tile stays in registers/L1.
 */
void paged_decode_tiled(const float *q, const float *k, const float *v,
                        const int32_t *page_ids, int n_pages, int page_size,
                        int dim, int n_tokens, float scale, float *out) {
    float m = -INFINITY;
    float l = 0.0f;
    if (dim > 512) return; /* guarded by the Python wrapper */
    /* heap tiles: BLOCK x dim for K and V, so the tile size follows the head dim */
    float *acc = (float *)calloc((size_t)dim, sizeof(float));
    float *kbuf = (float *)malloc(sizeof(float) * (size_t)BLOCK * (size_t)dim);
    float *vbuf = (float *)malloc(sizeof(float) * (size_t)BLOCK * (size_t)dim);
    float scores[BLOCK];
    if (!acc || !kbuf || !vbuf) {
        free(acc); free(kbuf); free(vbuf);
        return;
    }

    for (int d = 0; d < dim; ++d) acc[d] = 0.0f;

    for (int p = 0; p < n_pages; ++p) {
        int start = page_ids[p] * page_size;
        for (int j0 = 0; j0 < page_size; j0 += BLOCK) {
            int rows = page_size - j0;
            if (rows > BLOCK) rows = BLOCK;
            if (start + j0 + rows > n_tokens) rows = n_tokens - start - j0;
            if (rows <= 0) break;

            /* load the K tile once */
            for (int r = 0; r < rows; ++r) {
                memcpy(kbuf + (size_t)r * dim, k + (size_t)(start + j0 + r) * dim,
                       sizeof(float) * (size_t)dim);
            }
            float m_new = m;
            for (int r = 0; r < rows; ++r) {
                const float *krow = kbuf + (size_t)r * dim;
                float s = 0.0f;
                for (int d = 0; d < dim; ++d) s += q[d] * krow[d];
                scores[r] = s * scale;
                m_new = maxf_local(m_new, scores[r]);
            }
            /* load the V tile once and fold it into the accumulator */
            for (int r = 0; r < rows; ++r) {
                memcpy(vbuf + (size_t)r * dim, v + (size_t)(start + j0 + r) * dim,
                       sizeof(float) * (size_t)dim);
            }
            float alpha = expf(m - m_new);
            for (int d = 0; d < dim; ++d) acc[d] *= alpha;
            float new_l = l * alpha;
            for (int r = 0; r < rows; ++r) {
                float p_ij = expf(scores[r] - m_new);
                new_l += p_ij;
                const float *vrow = vbuf + (size_t)r * dim;
                for (int d = 0; d < dim; ++d) acc[d] += p_ij * vrow[d];
            }
            l = new_l;
            m = m_new;
        }
    }
    for (int d = 0; d < dim; ++d) out[d] = acc[d] / (l > 0.0f ? l : 1.0f);
    free(acc);
    free(kbuf);
    free(vbuf);
}

/* Dense reference for the same query (reads everything). */
void dense_decode_reference(const float *q, const float *k, const float *v,
                            int n_tokens, int dim, float scale, float *out) {
    float m = -INFINITY, l = 0.0f;
    for (int d = 0; d < dim; ++d) out[d] = 0.0f;
    for (int j = 0; j < n_tokens; ++j) {
        const float *krow = k + (size_t)j * dim;
        float s = 0.0f;
        for (int d = 0; d < dim; ++d) s += q[d] * krow[d];
        s *= scale;
        float m_new = maxf_local(m, s);
        float alpha = expf(m - m_new);
        float p_ij = expf(s - m_new);
        l = l * alpha + p_ij;
        const float *vrow = v + (size_t)j * dim;
        for (int d = 0; d < dim; ++d) out[d] = out[d] * alpha + p_ij * vrow[d];
        m = m_new;
    }
    for (int d = 0; d < dim; ++d) out[d] /= (l > 0.0f ? l : 1.0f);
}
