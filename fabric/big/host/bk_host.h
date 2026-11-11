/* bk_host: the Big-Kev vector path (everything of a token that is not a GEMV) in plain C11.
 *
 * Bit-identical to fabric/big/ref/fabric_ref.py (whose module docstring is the normative
 * spec) for finite data: RMSNorm, RoPE, KV-cache quantise, attention, softmax, SiLU,
 * residuals, BFP10 quantise of the GEMV inputs, argmax. The 253 GEMVs of a token are NOT
 * done here: bk_step() hands each quantised input to a callback (the PL on the board, the
 * spec's ExactBfpGemv in the gate).
 *
 * BUILD (mandatory flags; -ffp-contract=off because gcc on aarch64 fuses a*b+c into an FMA
 * by default, which changes results):
 *
 *     gcc -O3 -ffp-contract=off -fno-fast-math -pthread -shared -fPIC -o libbk_host.so bk_host.c
 *
 * Any optimisation level is correct (run_host.py checks -O0, -O2, -O3 and -O3 -march=native
 * builds against each other); -O3 is what bk_host.py uses and what the timings refer to.
 *
 * TWO SETS OF KERNELS, same results bit for bit:
 *   - the REFERENCE kernels: plain scalar C, one statement per spec operation (what the
 *     first version of this library was). They are the readable statement of the spec.
 *   - the FAST kernels (default when the target has NEON or SSE2): the same fp32 operations
 *     in the same per-lane order, written 4 lanes x 4 independent chains at a time so that an
 *     in-order core (Cortex-A53) is not stalled by one long dependency chain, with the
 *     attention restructured so the accumulators stay in registers (see bk_host.c).
 *   bk_config.kernels / $BK_KERNELS=ref|fast select; run_host.py (stage kern) proves fast
 *   == ref on adversarial inputs and both == the spec; on a target without the gate's torch
 *   stack: python3 -m fabric.big.host.bk_host --kern-check --threads-check <replay.npz>.
 *   Inputs a fast kernel does not handle exactly are handed to the reference kernel: NaN in
 *   exp, block exponents below -126, the float64 quantiser path, odd group sizes, more than
 *   8 query heads per kv head. silu of a NaN is a NaN in both; its payload is not specified.
 *
 * THREADS. A persistent pthread pool (bk_config.n_threads, else $BK_THREADS, else
 * min(4, online CPUs); 1 = no pool, everything on the caller's thread). Two parallel regions
 * per layer: the attention (one unit per kv head: q/k norms, RoPE, cache quantise, scores,
 * softmax, context, quantise of the context) and the FFN (silu * up and its quantise, in
 * chunks). Units are independent in the spec, each unit is computed by one thread in the
 * spec's order, so the result does not depend on the thread count. The callback is only
 * ever called from the thread that called bk_step. Idle workers spin for
 * bk_config.spin_us / $BK_SPIN_US microseconds (default 200) and then sleep; waking a
 * sleeping worker costs tens of microseconds per region, spinning costs idle CPU.
 * (MEASURED on the KV260's A53, 8B shape, 4 threads, 500 us between host sections as when a
 * GEMV runs in the fabric: 11.1 ms/token at Tk 20 with the default, 8.6 ms with
 * spin_us = 3000, i.e. sleeping costs about 1 ms per token and spinning keeps three cores
 * busy through every GEMV. The default favours the other processes on the board.)
 *
 * THE A53 (what the fast kernels are written for; measured on the KV260, 1.33 GHz): it
 * issues in order, at most one 128-bit NEON operation per cycle, and waits when an operand
 * is not ready; a 128-bit load costs two cycles; integer loads / stores issue next to a NEON
 * operation. So the kernels are rows of four independent operations, values stay in
 * registers, and BK_PINV in bk_host.c pins the order against gcc's scheduler. Measured
 * there: 99 cycles per 64 score products (4 heads x 16 channels; 82 by instruction count),
 * 63 per 64 context products, 14 cycles per silu element (reference: 50).
 *
 * No dependency beyond libc + pthreads, and no libm call in the data path: every transcendental is a
 * table passed in by the caller (built by bk_host.py with the spec's numpy code), and
 * frexp / floor / round are done on the bits or with exactly specified IEEE operations.
 * bk_selftest() (also run by bk_init) refuses to work if the build contracted a*b+c, is
 * not round-to-nearest-even, or evaluates float in a wider format.
 *
 * GEMV numbering (the order bk_step calls them in, the spec's order):
 *
 *     mat_id = 7*L + kind,  L = layer 0..n_layer-1,
 *              kind: 0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6 down      (BK_Q .. BK_DOWN)
 *     mat_id = 7*n_layer    lm_head (skipped when want_logits == 0)
 *
 *     kind      input (n_in)        rows (output)
 *     q         d                   n_head*head_dim
 *     k, v      d                   n_kv*head_dim
 *     o         n_head*head_dim     d
 *     gate, up  d                   ffn
 *     down      ffn                 d
 *     lm_head   d                   vocab
 *
 * LIMITS (where the spec itself is open, the library refuses instead of guessing):
 *   - a block exponent must fit int8 (the cache and the callback carry int8): a group whose
 *     largest |x| is non-zero but below ~2^-119 returns BK_ERR_EXP_RANGE;
 *   - inf / NaN reaching a quantiser returns BK_ERR_NONFINITE.
 * One bk_state is one decode stream and is not thread-safe (it runs its own worker threads;
 * call it from one thread at a time); separate states are independent.
 * KV cache memory: n_layer * n_kv * (3*head_dim + head_dim/16 + head_dim/128) bytes per
 * position (Bonsai-8B: 113 kB), grown by doubling up to max_ctx.
 *
 * INPUT SHARING. q/k/v get the same input, and so do gate/up. bk_step passes the SAME xq
 * (and xe) pointer for those calls, and guarantees more: across consecutive callback
 * calls (also across bk_step calls on the same state) the xq pointer equals the previous
 * call's pointer IF AND ONLY IF the input data are the previous call's data. A callee may
 * therefore skip reloading the activation whenever xq == the pointer it loaded last.
 */
#ifndef BK_HOST_H
#define BK_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BK_API_VERSION 2
#define BK_P        16      /* reduction lanes */
#define BK_EXP_K    4096    /* exp table: BK_EXP_K + 1 entries */
#define BK_RS_N     512     /* rsqrt tables: 512 + 512 entries */
#define BK_QK       128     /* GEMV input / V cache / embedding group */
#define BK_KGRP     16      /* K cache group */
#define BK_GEMV_BITS 10
#define BK_K_BITS    10
#define BK_V_BITS    8

enum { BK_Q = 0, BK_K = 1, BK_V = 2, BK_O = 3, BK_GATE = 4, BK_UP = 5, BK_DOWN = 6, BK_KINDS = 7 };

/* return codes (bk_init reports through *err, bk_step returns them) */
enum {
    BK_OK            = 0,
    BK_ERR_ARG       = -1,   /* bad config / NULL pointer / token out of range */
    BK_ERR_CTX_FULL  = -2,   /* position >= max_ctx (or >= max_pos of the RoPE tables) */
    BK_ERR_NOMEM     = -3,
    BK_ERR_GEMV      = -4,   /* the callback returned non-zero (value: bk_last_gemv_rc) */
    BK_ERR_EXP_RANGE = -5,   /* a block exponent does not fit int8 (|x| < ~2^-119: never on real data) */
    BK_ERR_NONFINITE = -6,   /* inf / NaN reached a quantiser (the spec is unspecified there) */
    BK_ERR_FPENV     = -7    /* bk_selftest failed: wrong compiler flags or rounding mode */
};

/* y[rows] = W[mat_id] @ (xq[i] * 2^xe[i/128]),  xq int16 BFP10 mantissas in [-511, 511]
 * (groups*128 of them), xe one exponent per 128. Return 0 on success. */
typedef int (*bk_gemv_fn)(void *ctx, int mat_id, const int16_t *xq, const int8_t *xe,
                          int groups, float *y, int rows);

typedef struct {
    int32_t n_layer, d, ffn, n_head, n_kv, head_dim, vocab;
    int32_t max_ctx;        /* most positions the KV cache may grow to (<= max_pos) */
    int32_t max_pos;        /* rows of cos_tab / sin_tab */
    int32_t init_ctx;       /* positions allocated up front (0 = 64); grows by doubling */
    float   eps;            /* fp32(cfg.eps) */
    float   inv_sqrt_hd;    /* fp32(1 / sqrt(head_dim)) */
    /* --- API 2 (all 0 = defaults) --- */
    int32_t n_threads;      /* 0: $BK_THREADS, else min(4, online CPUs); 1: no pool; max BK_MAX_THREADS */
    int32_t spin_us;        /* idle workers spin this long before sleeping; 0: $BK_SPIN_US, else 200; < 0: never spin */
    int32_t kernels;        /* BK_KERN_AUTO ($BK_KERNELS=ref|fast, else fast), BK_KERN_FAST, BK_KERN_REF */
    int32_t _pad;
} bk_config;

#define BK_MAX_THREADS 16
enum { BK_KERN_AUTO = 0, BK_KERN_FAST = 1, BK_KERN_REF = 2 };

/* Everything here is BORROWED: the caller keeps it alive and unchanged while the state lives. */
typedef struct {
    const float *attn_norm;         /* [n_layer][d]         RMSNorm gains */
    const float *q_norm;            /* [n_layer][head_dim] */
    const float *k_norm;            /* [n_layer][head_dim] */
    const float *ffn_norm;          /* [n_layer][d] */
    const float *out_norm;          /* [d] */
    /* embedding, Q1 format: row r, group g (128 weights): 16 sign bytes at
     * emb_signs + (r*(d/128) + g)*emb_sign_stride, weight j = bit (j%8) of byte (j/8),
     * 1 = +scale; fp16 (little-endian, may be unaligned) scale at
     * emb_scales + (r*(d/128) + g)*emb_scale_stride.
     * A GGUF Q1_0 tensor mapped as is: emb_scales = base, emb_signs = base + 2, strides 18. */
    const uint8_t *emb_signs;
    const uint8_t *emb_scales;
    int64_t emb_sign_stride, emb_scale_stride;
    const float *exp_tab;           /* [4097]  fp32(2^(k/4096)) */
    const float *rs_lo;             /* [512]   fp32(1/sqrt(0.5 + (i+0.5)/1024)) */
    const float *rs_hi;             /* [512]   fp32(1/sqrt(1.0 + (i+0.5)/512)) */
    const float *cos_tab;           /* [max_pos][head_dim/2] */
    const float *sin_tab;           /* [max_pos][head_dim/2] */
} bk_weights;

/* Optional taps: any pointer may be NULL (that tap is off). Filled by one bk_step call.
 * xq / xe / y are the concatenation over mat_id in call order, at the offsets returned by
 * bk_mat_info(); the lm_head slots are written only when want_logits != 0. */
typedef struct {
    int16_t *xq;        /* every GEMV input, mantissas             [bk_tap_sizes: n_xq] */
    int8_t  *xe;        /* every GEMV input, exponents             [n_xe] */
    float   *y;         /* every GEMV output                       [n_y] */
    float   *x_in;      /* [n_layer][d]  residual entering the layer (x_in[0] = embedding row) */
    float   *x_mid;     /* [n_layer][d]  after the attention residual */
    float   *x_out;     /* [n_layer][d]  after the FFN residual */
    float   *q_rope;    /* [n_layer][n_head*head_dim] */
    float   *k_rope;    /* [n_layer][n_kv*head_dim]   (before the K-cache quantiser) */
    float   *att_s;     /* [n_layer][n_head][att_cap] scores, first Tk valid */
    float   *att_p;     /* [n_layer][n_head][att_cap] probabilities, first Tk valid */
    float   *ctx;       /* [n_layer][n_head*head_dim] */
    float   *silu;      /* [n_layer][ffn]  silu(gate) */
    float   *act;       /* [n_layer][ffn]  silu(gate) * up */
    float   *h_out;     /* [d]  out_norm output (only when want_logits != 0) */
    int32_t  att_cap;   /* capacity (positions) of att_s / att_p rows */
    int32_t  _pad;
} bk_taps;

/* timing buckets, nanoseconds (CLOCK_MONOTONIC), accumulated until bk_reset_stats.
 * ns[] is wall time on the calling thread and sums to the time spent in bk_step: the wall
 * time of a parallel region is split over its buckets in proportion to the thread time
 * spent in each. cpu_ns[] is the thread time itself (== ns[] with one thread). */
enum {
    BK_T_GEMV = 0,      /* inside the callback */
    BK_T_EMBED,         /* embedding row dequant */
    BK_T_RMSNORM,       /* all RMSNorms (layer, q/k per head, final) */
    BK_T_QUANT,         /* BFP10 quantise of the GEMV inputs */
    BK_T_ROPE,
    BK_T_KVQ,           /* quantise K (BFP10 g16) and V (BFP8 g128) into the cache */
    BK_T_ATT_SCORE,     /* K dequant + q.k lane sums */
    BK_T_ATT_SOFTMAX,   /* max, exp_fab, Z, p */
    BK_T_ATT_CTX,       /* V dequant + p*v lane sums */
    BK_T_SILU,          /* silu(gate) * up */
    BK_T_RESID,         /* residual adds */
    BK_T_TAPS,          /* tap copies (zero when taps == NULL) */
    BK_T_OTHER,
    BK_T_COUNT
};

typedef struct {
    uint64_t ns[BK_T_COUNT];
    uint64_t steps;         /* bk_step calls completed */
    uint64_t gemv_calls;
    uint64_t att_pos;       /* sum over steps of the history length Tk */
    /* --- API 2 --- */
    uint64_t cpu_ns[BK_T_COUNT];
    uint64_t regions;       /* parallel regions run */
    int32_t  threads;       /* threads in use (caller included) */
    int32_t  kernels;       /* BK_KERN_FAST or BK_KERN_REF, as resolved */
    uint64_t worker_units;  /* region units run by worker threads (the rest ran on the caller) */
} bk_stats;

typedef struct bk_state bk_state;

/* ---- model / state ---- */
bk_state *bk_init(const bk_config *cfg, const bk_weights *w, int *err);
void      bk_free(bk_state *s);
void      bk_reset(bk_state *s);                 /* position 0, cache kept allocated */
int       bk_pos(const bk_state *s);             /* next position = tokens in the cache */
int       bk_truncate(bk_state *s, int pos);     /* drop the cache back to `pos` tokens */
int       bk_capacity(const bk_state *s);        /* positions currently allocated */

/* One token through all layers at position bk_pos(s); the K/V of this position are appended
 * to the cache. want_logits == 0 (prefill) skips out_norm and the lm_head call; otherwise
 * `logits` [vocab] receives the lm_head output. taps may be NULL. On error the position is
 * not advanced (the call can be repeated). */
int bk_step(bk_state *s, int32_t token, int want_logits, bk_gemv_fn gemv, void *gemv_ctx,
            float *logits, const bk_taps *taps);
int bk_last_gemv_rc(const bk_state *s);

/* the LOWEST index holding the maximum (the spec's tie rule); -1 if n <= 0 */
int bk_argmax(const float *x, int n);

/* ---- introspection ---- */
int  bk_n_mats(const bk_state *s);               /* 7*n_layer + 1 */
/* n_in / rows of a GEMV and its offsets in the xq / xe / y taps; 0 on success */
int  bk_mat_info(const bk_state *s, int mat_id, int *n_in, int *rows,
                 int64_t *xq_off, int64_t *xe_off, int64_t *y_off);
void bk_tap_sizes(const bk_state *s, int64_t *n_xq, int64_t *n_xe, int64_t *n_y);
/* the cache rows as stored, layer L, kv head g: kq int16 [pos][head_dim], ke int8
 * [pos][head_dim/16], vq int8 [pos][head_dim], ve int8 [pos][head_dim/128]. The pointers
 * move when the cache grows. */
int  bk_kv_rows(const bk_state *s, int layer, int kv_head, const int16_t **kq,
                const int8_t **ke, const int8_t **vq, const int8_t **ve);
void bk_get_stats(const bk_state *s, bk_stats *out);
void bk_reset_stats(bk_state *s);
int  bk_selftest(void);                          /* 0 = this build computes IEEE fp32 as the spec needs */
int  bk_api_version(void);
/* sizeof of the public structs, for binding sanity checks: which = 0 config, 1 weights, 2 taps, 3 stats */
size_t bk_sizeof(int which);

/* ---- the spec's primitives, exported for unit gates ---- */
float bk_fsum(const float *x, int n);
float bk_rsqrt_fab(const float *rs_lo, const float *rs_hi, float v);
float bk_exp_fab(const float *exp_tab, float x);
void  bk_exp_fab_n(const float *exp_tab, const float *x, float *y, int n);
void  bk_silu(const float *exp_tab, const float *x, float *y, int n);
void  bk_rmsnorm(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                 const float *w, int n, float *y);
void  bk_rope(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y);
/* x[n] -> mantissas + one exponent per `group` (n % group == 0). bits <= 16 (q16) / <= 8 (q8).
 * Returns BK_OK, BK_ERR_EXP_RANGE or BK_ERR_NONFINITE. */
int   bk_bfp_quant16(const float *x, int n, int group, int bits, int16_t *q, int8_t *e);
int   bk_bfp_quant8(const float *x, int n, int group, int bits, int8_t *q, int8_t *e);
void  bk_bfp_dequant16(const int16_t *q, const int8_t *e, int n, int group, float *x);
void  bk_bfp_dequant8(const int8_t *q, const int8_t *e, int n, int group, float *x);
/* one embedding row (see bk_weights for the layout) */
void  bk_embed_row(const uint8_t *signs, int64_t sign_stride, const uint8_t *scales,
                   int64_t scale_stride, int d, int64_t row, float *x);

/* ---- reference kernels and kernel-level test entry points ---- */
/* The public primitives above run the fast kernels when the build has them; these always
 * run the scalar reference kernels. */
float bk_ref_fsum(const float *x, int n);
void  bk_ref_exp_fab_n(const float *exp_tab, const float *x, float *y, int n);
void  bk_ref_silu(const float *exp_tab, const float *x, float *y, int n);
void  bk_ref_rmsnorm(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                     const float *w, int n, float *y);
void  bk_ref_rope(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y);
int   bk_ref_bfp_quant16(const float *x, int n, int group, int bits, int16_t *q, int8_t *e);
int   bk_ref_bfp_quant8(const float *x, int n, int group, int bits, int8_t *q, int8_t *e);
/* silu(gate) * up in one pass, as the FFN does it: sil (may be NULL, may alias gate) gets
 * silu(gate), act gets silu(gate) * up. kernels = BK_KERN_FAST / BK_KERN_REF. */
void  bk_silu_mul(int kernels, const float *exp_tab, const float *gate, const float *up,
                  float *sil, float *act, int n);
/* The attention of ONE kv head: `rep` query heads q [rep][hd] against Tk cached rows
 * (kq int16 [Tk][hd], ke [Tk][hd/16], vq int8 [Tk][hd], ve [Tk][hd/128]).
 * s_out / p_out [rep][Tk] (may be NULL) get the scores / probabilities, ctx [rep][hd] the
 * context. kernels = BK_KERN_FAST / BK_KERN_REF. Returns BK_OK / BK_ERR_ARG / BK_ERR_NOMEM. */
int   bk_attend_kv(int kernels, const float *exp_tab, int hd, int rep, int Tk, float inv_sqrt_hd,
                   const float *q, const int16_t *kq, const int8_t *ke, const int8_t *vq,
                   const int8_t *ve, float *s_out, float *p_out, float *ctx);
/* 0 = this build has no fast kernels (they then alias the reference ones); else 1 NEON, 2 SSE2 */
int   bk_simd(void);
int   bk_threads(const bk_state *s);
int   bk_kernels(const bk_state *s);

/* Deterministic stand-in GEMV for benchmarks (NOT the model): fills y with a fixed
 * pseudo-random pattern of plausible magnitude that depends on mat_id and the input.
 * ctx = NULL, or a `const int32_t *` holding a number of microseconds to busy-wait before
 * returning (models the time the fabric takes, during which the worker threads idle).
 * Lets the host cost be timed on a machine that has no PL and no weights. */
int bk_bench_gemv(void *ctx, int mat_id, const int16_t *xq, const int8_t *xe, int groups,
                  float *y, int rows);

#ifdef __cplusplus
}
#endif
#endif
