/* bk_host.c: the Big-Kev vector path, bit-identical to fabric/big/ref/fabric_ref.py.
 * See bk_host.h for the API, the build flags and the GEMV numbering.
 *
 * How bit-identity is kept (read before touching anything here):
 *   - every fp32 operation of the spec is one C float operation, in the spec's order. No
 *     a*b+c contraction (flag + pragma + selftest), no reassociation (never -ffast-math).
 *   - reductions are written with the spec's 16 explicit lanes, so a vectoriser can use
 *     SIMD without reordering anything: lanes are independent, each lane is sequential.
 *   - "x + 0" for the zero padding of the spec is skipped: a lane accumulator starts at +0
 *     and can never become -0 (IEEE: only (-0) + (-0) gives -0), so adding +0 is a no-op.
 *   - 2^n is built from the exponent bits; frexp / floor / round-half-even are done on the
 *     bits, with an exact int conversion, or with the 1.5*2^23 trick (exact RNE for
 *     |t| < 2^22 in round-to-nearest-even mode). No libm.
 *   - the spec's bfp_quant works in float64. Here: fp32 x * 2^-e is exact unless it
 *     underflows, and an underflowed value (< 2^-126) rounds to 0 either way; exponents the
 *     fp32 path cannot represent take a float64 path. Same integers in every case.
 *
 * -DBK_BUG_* plant deliberate spec violations; fabric/big/run_host.py uses them to show the
 * gate has teeth. Never define them in a real build.
 */
#define _POSIX_C_SOURCE 199309L
#include "bk_host.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__FAST_MATH__)
#error "bk_host must not be built with -ffast-math (it reorders fp32 sums)"
#endif
#if !defined(BK_ALLOW_CONTRACT)
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#endif

#define P BK_P
#define RND32 12582912.0f               /* 1.5 * 2^23 */
#define RND64 6755399441055744.0        /* 1.5 * 2^52 */
/* fp32(1.4426950408889634) = np.float32(LOG2E) of the spec, bits 0x3FB8AA3B */
#define BK_LOG2E 1.44269502162933349609375f

/* ------------------------------------------------------------------ bit helpers */
static inline uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float f32_from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
/* exact 2^n, n in [-126, 127] */
static inline float pow2f(int n) { return f32_from_bits((uint32_t)(n + 127) << 23); }
/* exact 2^n as a double, n in [-1022, 1023] */
static inline double pow2d(int n)
{
    uint64_t u = (uint64_t)(n + 1023) << 52;
    double d;
    memcpy(&d, &u, 8);
    return d;
}

/* frexp of a positive finite non-zero fp32 given as bits (sign cleared):
 * value = (mant / 2^24) * 2^E with mant in [2^23, 2^24), i.e. m = mant/2^24 in [0.5, 1). */
static inline void fdecomp(uint32_t u, int *E, uint32_t *mant)
{
    int ex = (int)(u >> 23);
    uint32_t fr = u & 0x7fffffu;
    if (ex == 0) {                      /* subnormal: normalise */
        ex = 1;
        while (!(fr & 0x800000u)) { fr <<= 1; ex--; }
        *mant = fr;
    } else {
        *mant = fr | 0x800000u;
    }
    *E = ex - 126;
}

static inline float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t ex = (h >> 10) & 0x1fu, fr = h & 0x3ffu, u;
    if (ex == 0) {
        if (fr == 0) {
            u = sign;
        } else {                        /* fp16 subnormal = fr * 2^-24 */
            int sh = 0;
            while (!(fr & 0x400u)) { fr <<= 1; sh++; }
            u = sign | ((uint32_t)(113 - sh) << 23) | ((fr & 0x3ffu) << 13);
        }
    } else if (ex == 31) {
        u = sign | 0x7f800000u | (fr << 13);
    } else {
        u = sign | ((ex + 112u) << 23) | (fr << 13);
    }
    return f32_from_bits(u);
}

static inline uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* ------------------------------------------------------------------ fsum */
/* pairwise tree over the 16 lanes: [0:8] + [8:16], [0:4] + [4:8], ... */
static inline float tree16(float *a)
{
#if defined(BK_BUG_TREE)                /* planted bug: adjacent pairs */
    for (int w = P; w > 1; w /= 2)
        for (int i = 0; i < w / 2; i++) a[i] = a[2 * i] + a[2 * i + 1];
    return a[0];
#else
    for (int i = 0; i < 8; i++) a[i] = a[i] + a[i + 8];
    for (int i = 0; i < 4; i++) a[i] = a[i] + a[i + 4];
    for (int i = 0; i < 2; i++) a[i] = a[i] + a[i + 2];
    return a[0] + a[1];
#endif
}

float bk_fsum(const float *x, int n)
{
    float acc[P] = {0};
    int i = 0;
    for (; i + P <= n; i += P)
        for (int l = 0; l < P; l++) acc[l] = acc[l] + x[i + l];
    for (int l = 0; i < n; i++, l++) acc[l] = acc[l] + x[i];
    return tree16(acc);
}

/* fsum(x*x) */
static float sumsq(const float *restrict x, int n)
{
#if defined(BK_BUG_REASSOC)             /* planted bug: one sequential accumulator */
    float a = 0.0f;
    for (int i = 0; i < n; i++) a = a + x[i] * x[i];
    return a;
#else
    float acc[P] = {0};
    int i = 0;
    for (; i + P <= n; i += P)
        for (int l = 0; l < P; l++) {
#if defined(BK_BUG_FMA)                 /* planted bug: what a contracting compiler emits */
            acc[l] = __builtin_fmaf(x[i + l], x[i + l], acc[l]);
#else
            acc[l] = acc[l] + x[i + l] * x[i + l];
#endif
        }
    for (int l = 0; i < n; i++, l++) acc[l] = acc[l] + x[i] * x[i];
    return tree16(acc);
#endif
}

/* fsum(a*b) over hd dims (hd % 16 == 0): the attention score before the 1/sqrt(hd) */
static inline float dot_lanes(const float *restrict a, const float *restrict b, int hd)
{
    float acc[P] = {0};
    for (int j = 0; j < hd; j += P)
        for (int l = 0; l < P; l++) {
#if defined(BK_BUG_FMA)
            acc[l] = __builtin_fmaf(a[j + l], b[j + l], acc[l]);
#else
            acc[l] = acc[l] + a[j + l] * b[j + l];
#endif
        }
    return tree16(acc);
}

/* ------------------------------------------------------------------ transcendentals */
float bk_rsqrt_fab(const float *rs_lo, const float *rs_hi, float v)
{
    uint32_t u = f32_bits(v), a = u & 0x7fffffffu;
    float y;
    if (a == 0) {
        y = rs_lo[0];                   /* torch.frexp(0) = (0, 0): index clamps to 0 */
    } else {
        int E;
        uint32_t mant;
        fdecomp(a, &E, &mant);          /* (inf / NaN: unspecified in the spec) */
        int odd = E & 1;
        int E2 = odd ? E - 1 : E;       /* even */
        /* m in [0.5,1): floor((m - 0.5)*1024) = frac >> 14; 2m in [1,2): floor((2m - 1)*512)
         * = frac >> 14 as well. A negative v clamps the index to 0, as the spec's clamp does. */
        int idx = (u >> 31) ? 0 : (int)((mant & 0x7fffffu) >> 14);
        y = (odd ? rs_hi[idx] : rs_lo[idx]) * pow2f(-(E2 / 2));
    }
    float h = v * 0.5f;
#if defined(BK_BUG_NEWTON1)
    y = y * (1.5f - h * (y * y));
#else
    y = y * (1.5f - h * (y * y));
    y = y * (1.5f - h * (y * y));
#endif
    return y;
}

/* exp_fab over a block (n <= BK_CH; x may alias e):
 *     y = x*LOG2E; n = floor(y); f = y - n; k = RNE(f*4096); result = TAB[k] * 2^n,
 *     n < -126 -> +0, n > 127 -> +inf            (NaN in -> NaN out; the spec leaves it open)
 * arranged so the arithmetic vectorises: (A) index and scale, SIMD; (B) the table lookups,
 * scalar; (C) the 0 / inf / NaN selects, SIMD. n < -126 <=> y < -126 and n > 127 <=> y >= 128,
 * so (C) decides on y; for those y the masked intermediate of (A) is discarded. */
#define BK_CH 256
static void exp_block(const float *restrict tab, const float *x, float *e, int n)
{
    float ys[BK_CH];
    int32_t kk[BK_CH];
    uint32_t sb[BK_CH];
    for (int i = 0; i < n; i++) {
        float y = x[i] * BK_LOG2E;
        ys[i] = y;
        /* |y| >= 128 (and NaN) -> 0 so the int conversion stays defined; pass (C) overrides
         * those results. A bit mask, not a select: gcc vectorises this form at -O2 and -O3. */
        uint32_t yb = f32_bits(y);
        float yc = f32_from_bits(yb & (uint32_t) - (int32_t)((yb & 0x7fffffffu) < 0x43000000u));
        int nn = (int)yc;                           /* trunc */
        float fn = (float)nn;
        nn -= (fn > yc);                            /* floor */
        fn = (float)nn;
        float f = yc - fn;
        float t = f * 4096.0f;
#if defined(BK_BUG_EXP_TRUNC)
        int k = (int)t;
#else
        int k = (int)((t + RND32) - RND32);
#endif
        k = k < 0 ? 0 : k;
        k = k > BK_EXP_K ? BK_EXP_K : k;
        nn = nn < -126 ? -126 : nn;
        nn = nn > 127 ? 127 : nn;
        kk[i] = k;
        sb[i] = (uint32_t)(nn + 127) << 23;         /* bits of 2^n */
    }
    for (int i = 0; i < n; i++) e[i] = tab[kk[i]] * f32_from_bits(sb[i]);
    const float inf = f32_from_bits(0x7f800000u);
    for (int i = 0; i < n; i++) {
        float y = ys[i], r = e[i];
        r = y < -126.0f ? 0.0f : r;
        r = y >= 128.0f ? inf : r;
        r = y != y ? y : r;
        e[i] = r;
    }
}

float bk_exp_fab(const float *exp_tab, float x)
{
    float e;
    exp_block(exp_tab, &x, &e, 1);
    return e;
}

void bk_exp_fab_n(const float *exp_tab, const float *x, float *y, int n)
{
    for (int i = 0; i < n; i += BK_CH) exp_block(exp_tab, x + i, y + i, n - i < BK_CH ? n - i : BK_CH);
}

void bk_silu(const float *exp_tab, const float *x, float *y, int n)
{
    float nx[BK_CH], e[BK_CH];
    for (int i0 = 0; i0 < n; i0 += BK_CH) {
        int m = n - i0 < BK_CH ? n - i0 : BK_CH;
        const float *xi = x + i0;
        float *yi = y + i0;
        for (int i = 0; i < m; i++) nx[i] = -xi[i];
        exp_block(exp_tab, nx, e, m);
        for (int i = 0; i < m; i++) yi[i] = xi[i] * (1.0f / (1.0f + e[i]));
    }
}

static inline int ilog2(int n)          /* floor(log2(n)), n >= 1 (the spec: int(math.log2(n))) */
{
    int k = 0;
    while ((n >> (k + 1)) != 0) k++;
    return k;
}

void bk_rmsnorm(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                const float *w, int n, float *y)
{
    float ss = sumsq(x, n);
    float mean = ss * pow2f(-ilog2(n));
    float r = bk_rsqrt_fab(rs_lo, rs_hi, mean + eps);
    for (int i = 0; i < n; i++) y[i] = w[i] * (x[i] * r);
}

void bk_rope(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y)
{
    int h2 = head_dim / 2;
    const float *restrict x1 = x, *restrict x2 = x + h2;
    float *restrict y1 = y, *restrict y2 = y + h2;
    for (int i = 0; i < h2; i++) {
        float c = cos_row[i], s = sin_row[i];
        y1[i] = x1[i] * c - x2[i] * s;
        y2[i] = x2[i] * c + x1[i] * s;
    }
}

/* ------------------------------------------------------------------ BFP */
/* the block exponent of the spec's bfp_quant, from the bits of the group's max |x| */
static inline int bfp_exp(uint32_t am_bits, int bits)
{
    if (am_bits == 0) return 0;
    int E;
    uint32_t mant;
    fdecomp(am_bits, &E, &mant);
    /* m * 2^(bits-1) <= qmax  <=>  mant <= qmax * 2^(25-bits) */
    uint32_t thr = (uint32_t)((1u << (bits - 1)) - 1u) << (25 - bits);
    return (mant <= thr) ? E - (bits - 1) : E - (bits - 2);
}

static inline uint32_t absmax_bits(const float *restrict x, int n)
{
    int32_t am = 0;                     /* |x| bit patterns order like the values (NaN on top) */
    for (int i = 0; i < n; i++) {
        int32_t a = (int32_t)(f32_bits(x[i]) & 0x7fffffffu);
        am = a > am ? a : am;
    }
    return (uint32_t)am;
}

/* q = clamp(RNE(x * 2^-e), -qmax, qmax) written as float integers into r[n] */
static inline void bfp_round_group(const float *restrict x, int n, int e, float qmax,
                                   float *restrict r)
{
    if (-e <= 127) {                    /* -e >= -120 always: 2^-e is an fp32 */
        float sc = pow2f(-e);
        for (int i = 0; i < n; i++) {
            float t = x[i] * sc;
#if defined(BK_BUG_ROUND_AWAY)          /* planted bug: round half away from zero */
            float v = (float)(int)(t + (t < 0.0f ? -0.5f : 0.5f));
#else
            float v = (t + RND32) - RND32;
#endif
            v = v > qmax ? qmax : v;
            v = v < -qmax ? -qmax : v;
            r[i] = v;
        }
    } else {                            /* group max below 2^-118: float64, as the spec */
        double sc = pow2d(-e);
        for (int i = 0; i < n; i++) {
            double t = (double)x[i] * sc;
            double v = (t + RND64) - RND64;
            v = v > qmax ? qmax : v;
            v = v < -qmax ? -qmax : v;
            r[i] = (float)v;
        }
    }
}

#define BK_QUANT_FN(NAME, QT)                                                              \
    int NAME(const float *x, int n, int group, int bits, QT *q, int8_t *e)                 \
    {                                                                                      \
        const float qmax = (float)((1 << (bits - 1)) - 1);                                 \
        float r[BK_QK];                                                                    \
        for (int g = 0; g < n / group; g++) {                                              \
            const float *xg = x + (size_t)g * group;                                       \
            QT *qg = q + (size_t)g * group;                                                \
            uint32_t am = absmax_bits(xg, group);                                          \
            if (am >= 0x7f800000u) return BK_ERR_NONFINITE;                                \
            int ex = bfp_exp(am, bits);                                                    \
            if (ex < -128 || ex > 127) return BK_ERR_EXP_RANGE;                            \
            e[g] = (int8_t)ex;                                                             \
            for (int i0 = 0; i0 < group; i0 += BK_QK) {                                    \
                int m = group - i0 < BK_QK ? group - i0 : BK_QK;                           \
                bfp_round_group(xg + i0, m, ex, qmax, r);                                  \
                for (int i = 0; i < m; i++) qg[i0 + i] = (QT)(int32_t)r[i];                \
            }                                                                              \
        }                                                                                  \
        return BK_OK;                                                                      \
    }

BK_QUANT_FN(bk_bfp_quant16, int16_t)
BK_QUANT_FN(bk_bfp_quant8, int8_t)

/* exact q * 2^e (the spec: float64 product, then to fp32; q*2^e is always an fp32) */
#define BK_DEQUANT_FN(NAME, QT)                                                            \
    void NAME(const QT *q, const int8_t *e, int n, int group, float *x)                    \
    {                                                                                      \
        for (int g = 0; g < n / group; g++) {                                              \
            const QT *restrict qg = q + (size_t)g * group;                                 \
            float *restrict xg = x + (size_t)g * group;                                    \
            int ex = e[g];                                                                 \
            if (ex >= -126) {                                                              \
                float sc = pow2f(ex);                                                      \
                for (int i = 0; i < group; i++) xg[i] = (float)qg[i] * sc;                 \
            } else {                    /* 2^ex is not a normal fp32: two exact steps */   \
                float s2 = pow2f(ex + 126);                                                \
                for (int i = 0; i < group; i++) xg[i] = ((float)qg[i] * pow2f(-126)) * s2; \
            }                                                                              \
        }                                                                                  \
    }

BK_DEQUANT_FN(bk_bfp_dequant16, int16_t)
BK_DEQUANT_FN(bk_bfp_dequant8, int8_t)

/* ------------------------------------------------------------------ embedding / argmax */
void bk_embed_row(const uint8_t *signs, int64_t sign_stride, const uint8_t *scales,
                  int64_t scale_stride, int d, int64_t row, float *x)
{
    int G = d / BK_QK;
    for (int g = 0; g < G; g++) {
        int64_t blk = row * G + g;
        const uint8_t *sp = scales + blk * scale_stride;
        const uint8_t *bp = signs + blk * sign_stride;
        float sc = half_to_float((uint16_t)(sp[0] | ((uint16_t)sp[1] << 8)));
        float ns = -sc;                 /* -1 * scale */
        float *xg = x + (size_t)g * BK_QK;
        for (int b = 0; b < BK_QK / 8; b++) {
            unsigned byte = bp[b];
            for (int k = 0; k < 8; k++) xg[b * 8 + k] = ((byte >> k) & 1u) ? sc : ns;
        }
    }
}

int bk_argmax(const float *x, int n)
{
    if (n <= 0) return -1;
    int best = 0;
    float m = x[0];
    for (int i = 1; i < n; i++) {
#if defined(BK_BUG_ARGMAX_LAST)
        if (x[i] >= m) { m = x[i]; best = i; }
#else
        if (x[i] > m) { m = x[i]; best = i; }
#endif
    }
    return best;
}

/* ------------------------------------------------------------------ state */
typedef struct {
    int16_t *kq;        /* [cap][hd]      K mantissas (BFP10) */
    int8_t  *ke;        /* [cap][hd/16]   K exponents */
    int8_t  *vq;        /* [cap][hd]      V mantissas (BFP8) */
    int8_t  *ve;        /* [cap][hd/128]  V exponents */
} kv_head;

#define BK_MAX_ALLOCS 48

struct bk_state {
    bk_config c;
    bk_weights w;
    int rep, hd, hd2, nq, nkv;          /* nq = n_head*hd, nkv = n_kv*hd */
    int pos, cap;
    int n_mats;
    kv_head *kv;                        /* [n_layer*n_kv] */
    /* per-token work vectors */
    float *x, *h, *yq, *yk, *yv, *tq, *qr, *kr, *ctx, *yo, *gate, *up, *act, *yd;
    float *kf;                          /* [hd] one dequantised cache row */
    float *cacc;                        /* [rep][P][hd] */
    float *sc;                          /* [rep][cap] scores -> exp -> p */
    /* GEMV inputs: one buffer per role, so pointer identity <=> same data (see bk_host.h) */
    int16_t *xq_attn, *xq_ctx, *xq_ffn, *xq_act, *xq_out;
    int8_t *xe_attn, *xe_ctx, *xe_ffn, *xe_act, *xe_out;
    int64_t *off_xq, *off_xe, *off_y;   /* [n_mats + 1] tap offsets */
    void *allocs[BK_MAX_ALLOCS];
    int n_allocs;
    bk_stats st;
    uint64_t tlast;
    int gemv_rc;
};

static inline void lap(bk_state *s, int cat)
{
    uint64_t t = now_ns();
    s->st.ns[cat] += t - s->tlast;
    s->tlast = t;
}

static void *xalloc(bk_state *s, size_t n, size_t sz, int *fail)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p || s->n_allocs >= BK_MAX_ALLOCS) {
        free(p);
        *fail = 1;
        return NULL;
    }
    s->allocs[s->n_allocs++] = p;
    return p;
}

static void mat_shape(const bk_state *s, int mat, int *n_in, int *rows)
{
    const bk_config *c = &s->c;
    if (mat == 7 * c->n_layer) { *n_in = c->d; *rows = c->vocab; return; }
    switch (mat % 7) {
    case BK_Q:    *n_in = c->d;   *rows = s->nq;  break;
    case BK_K:
    case BK_V:    *n_in = c->d;   *rows = s->nkv; break;
    case BK_O:    *n_in = s->nq;  *rows = c->d;   break;
    case BK_GATE:
    case BK_UP:   *n_in = c->d;   *rows = c->ffn; break;
    default:      *n_in = c->ffn; *rows = c->d;   break;
    }
}

static int ensure_cap(bk_state *s, int need)
{
    if (need <= s->cap) return BK_OK;
    const bk_config *c = &s->c;
    int nc = s->cap ? s->cap : (c->init_ctx > 0 ? c->init_ctx : 64);
    while (nc < need) nc *= 2;
    if (nc > c->max_ctx) nc = c->max_ctx;
    if (nc < need) return BK_ERR_CTX_FULL;
    size_t hd = (size_t)s->hd;
    for (int i = 0; i < c->n_layer * c->n_kv; i++) {
        kv_head *k = &s->kv[i];
        void *p;
        if (!(p = realloc(k->kq, (size_t)nc * hd * sizeof(int16_t)))) return BK_ERR_NOMEM;
        k->kq = p;
        if (!(p = realloc(k->ke, (size_t)nc * (hd / BK_KGRP)))) return BK_ERR_NOMEM;
        k->ke = p;
        if (!(p = realloc(k->vq, (size_t)nc * hd))) return BK_ERR_NOMEM;
        k->vq = p;
        if (!(p = realloc(k->ve, (size_t)nc * (hd / BK_QK)))) return BK_ERR_NOMEM;
        k->ve = p;
    }
    void *p = realloc(s->sc, (size_t)s->rep * (size_t)nc * sizeof(float));
    if (!p) return BK_ERR_NOMEM;
    s->sc = p;
    s->cap = nc;
    return BK_OK;
}

/* a*b + c through volatile operands: if this TU was compiled with contraction, the result
 * keeps the low product bits and differs from the two-rounding IEEE result. */
static float st_muladd(float a, float b, float c) { return a * b + c; }

int bk_selftest(void)
{
    volatile float a = 1.0f + 1.0f / 4096.0f;       /* 1 + 2^-12 */
    volatile float c = -(1.0f + 1.0f / 2048.0f);    /* -(1 + 2^-11) */
    volatile float one = 1.0f, t24 = 1.0f / 16777216.0f, t23 = 1.0f / 8388608.0f;
    volatile float mn = 1.17549435e-38f, half = 0.5f;
    volatile float v25 = 2.5f, v35 = 3.5f, n25 = -2.5f;
    volatile double d25 = 2.5, d35 = 3.5;
    int bad = 0;
    /* (1+2^-12)^2 = 1 + 2^-11 + 2^-24 -> RNE to fp32 = 1 + 2^-11; + c = 0 exactly.
     * FMA or wider evaluation gives 2^-24. */
    if (st_muladd(a, a, c) != 0.0f) bad |= 1;
    /* round-to-nearest-even: 1 + 2^-24 ties down to 1; (1 + 2^-23) + 2^-24 ties up */
    float r1 = one + t24, r2 = (one + t23) + t24;
    if (r1 != 1.0f || r2 != 1.0f + 2.0f * t23) bad |= 2;
    /* gradual underflow on (no flush-to-zero / denormals-are-zero) */
    float sub = mn * half;
    if (sub == 0.0f || sub + sub != mn) bad |= 4;
    /* the rounding tricks */
    float q1 = (v25 + RND32) - RND32, q2 = (v35 + RND32) - RND32, q3 = (n25 + RND32) - RND32;
    double q4 = (d25 + RND64) - RND64, q5 = (d35 + RND64) - RND64;
    if (q1 != 2.0f || q2 != 4.0f || q3 != -2.0f || q4 != 2.0 || q5 != 4.0) bad |= 8;
    if (f32_bits(BK_LOG2E) != 0x3FB8AA3Bu || f32_bits(RND32) != 0x4B400000u) bad |= 16;
    if (sizeof(float) != 4 || sizeof(double) != 8 || (-1 >> 1) != -1) bad |= 32;
    return bad;
}

int bk_api_version(void) { return BK_API_VERSION; }

size_t bk_sizeof(int which)
{
    switch (which) {
    case 0: return sizeof(bk_config);
    case 1: return sizeof(bk_weights);
    case 2: return sizeof(bk_taps);
    case 3: return sizeof(bk_stats);
    default: return 0;
    }
}

void bk_free(bk_state *s)
{
    if (!s) return;
    if (s->kv)
        for (int i = 0; i < s->c.n_layer * s->c.n_kv; i++) {
            free(s->kv[i].kq);
            free(s->kv[i].ke);
            free(s->kv[i].vq);
            free(s->kv[i].ve);
        }
    free(s->sc);
    for (int i = 0; i < s->n_allocs; i++) free(s->allocs[i]);
    free(s);
}

bk_state *bk_init(const bk_config *cfg, const bk_weights *w, int *err)
{
    int e = BK_OK, dummy;
    if (!err) err = &dummy;
    *err = BK_OK;
    if (!cfg || !w) { *err = BK_ERR_ARG; return NULL; }
#if !defined(BK_SKIP_SELFTEST)
    if (bk_selftest() != 0) { *err = BK_ERR_FPENV; return NULL; }
#endif
    const bk_config *c = cfg;
    if (c->n_layer < 1 || c->d < BK_QK || c->d % BK_QK || c->ffn < BK_QK || c->ffn % BK_QK ||
        c->n_head < 1 || c->n_kv < 1 || c->n_head % c->n_kv || c->head_dim < BK_QK ||
        c->head_dim % BK_QK || c->vocab < 1 || c->max_ctx < 1 || c->max_pos < c->max_ctx ||
        c->init_ctx < 0)
        e = BK_ERR_ARG;
    if (!w->attn_norm || !w->q_norm || !w->k_norm || !w->ffn_norm || !w->out_norm ||
        !w->emb_signs || !w->emb_scales || !w->exp_tab || !w->rs_lo || !w->rs_hi ||
        !w->cos_tab || !w->sin_tab)
        e = BK_ERR_ARG;
    if (e) { *err = e; return NULL; }

    bk_state *s = calloc(1, sizeof(*s));
    if (!s) { *err = BK_ERR_NOMEM; return NULL; }
    s->c = *c;
    s->w = *w;
    s->hd = c->head_dim;
    s->hd2 = c->head_dim / 2;
    s->rep = c->n_head / c->n_kv;
    s->nq = c->n_head * c->head_dim;
    s->nkv = c->n_kv * c->head_dim;
    s->n_mats = 7 * c->n_layer + 1;

    int fail = 0;
    size_t d = (size_t)c->d, ffn = (size_t)c->ffn, nq = (size_t)s->nq, nkv = (size_t)s->nkv;
    s->kv = xalloc(s, (size_t)c->n_layer * c->n_kv, sizeof(kv_head), &fail);
    s->x = xalloc(s, d, sizeof(float), &fail);
    s->h = xalloc(s, d, sizeof(float), &fail);
    s->yq = xalloc(s, nq, sizeof(float), &fail);
    s->yk = xalloc(s, nkv, sizeof(float), &fail);
    s->yv = xalloc(s, nkv, sizeof(float), &fail);
    s->tq = xalloc(s, nq, sizeof(float), &fail);
    s->qr = xalloc(s, nq, sizeof(float), &fail);
    s->kr = xalloc(s, nkv, sizeof(float), &fail);
    s->ctx = xalloc(s, nq, sizeof(float), &fail);
    s->yo = xalloc(s, d, sizeof(float), &fail);
    s->gate = xalloc(s, ffn, sizeof(float), &fail);
    s->up = xalloc(s, ffn, sizeof(float), &fail);
    s->act = xalloc(s, ffn, sizeof(float), &fail);
    s->yd = xalloc(s, d, sizeof(float), &fail);
    s->kf = xalloc(s, (size_t)s->hd, sizeof(float), &fail);
    s->cacc = xalloc(s, (size_t)s->rep * P * (size_t)s->hd, sizeof(float), &fail);
    s->xq_attn = xalloc(s, d, sizeof(int16_t), &fail);
    s->xq_ctx = xalloc(s, nq, sizeof(int16_t), &fail);
    s->xq_ffn = xalloc(s, d, sizeof(int16_t), &fail);
    s->xq_act = xalloc(s, ffn, sizeof(int16_t), &fail);
    s->xq_out = xalloc(s, d, sizeof(int16_t), &fail);
    s->xe_attn = xalloc(s, d / BK_QK, 1, &fail);
    s->xe_ctx = xalloc(s, nq / BK_QK, 1, &fail);
    s->xe_ffn = xalloc(s, d / BK_QK, 1, &fail);
    s->xe_act = xalloc(s, ffn / BK_QK, 1, &fail);
    s->xe_out = xalloc(s, d / BK_QK, 1, &fail);
    s->off_xq = xalloc(s, (size_t)s->n_mats + 1, sizeof(int64_t), &fail);
    s->off_xe = xalloc(s, (size_t)s->n_mats + 1, sizeof(int64_t), &fail);
    s->off_y = xalloc(s, (size_t)s->n_mats + 1, sizeof(int64_t), &fail);
    if (fail) { bk_free(s); *err = BK_ERR_NOMEM; return NULL; }
    for (int m = 0; m < s->n_mats; m++) {
        int n_in, rows;
        mat_shape(s, m, &n_in, &rows);
        s->off_xq[m + 1] = s->off_xq[m] + n_in;
        s->off_xe[m + 1] = s->off_xe[m] + n_in / BK_QK;
        s->off_y[m + 1] = s->off_y[m] + rows;
    }
    int init = c->init_ctx > 0 ? c->init_ctx : 64;
    if (init > c->max_ctx) init = c->max_ctx;
    if ((e = ensure_cap(s, init)) != BK_OK) { bk_free(s); *err = e; return NULL; }
    return s;
}

void bk_reset(bk_state *s) { s->pos = 0; }
int bk_pos(const bk_state *s) { return s->pos; }
int bk_capacity(const bk_state *s) { return s->cap; }
int bk_last_gemv_rc(const bk_state *s) { return s->gemv_rc; }
int bk_n_mats(const bk_state *s) { return s->n_mats; }

int bk_truncate(bk_state *s, int pos)
{
    if (pos < 0 || pos > s->pos) return BK_ERR_ARG;
    s->pos = pos;
    return BK_OK;
}

int bk_mat_info(const bk_state *s, int mat_id, int *n_in, int *rows, int64_t *xq_off,
                int64_t *xe_off, int64_t *y_off)
{
    if (mat_id < 0 || mat_id >= s->n_mats) return BK_ERR_ARG;
    int a, b;
    mat_shape(s, mat_id, &a, &b);
    if (n_in) *n_in = a;
    if (rows) *rows = b;
    if (xq_off) *xq_off = s->off_xq[mat_id];
    if (xe_off) *xe_off = s->off_xe[mat_id];
    if (y_off) *y_off = s->off_y[mat_id];
    return BK_OK;
}

void bk_tap_sizes(const bk_state *s, int64_t *n_xq, int64_t *n_xe, int64_t *n_y)
{
    if (n_xq) *n_xq = s->off_xq[s->n_mats];
    if (n_xe) *n_xe = s->off_xe[s->n_mats];
    if (n_y) *n_y = s->off_y[s->n_mats];
}

int bk_kv_rows(const bk_state *s, int layer, int kv_head_i, const int16_t **kq,
               const int8_t **ke, const int8_t **vq, const int8_t **ve)
{
    if (layer < 0 || layer >= s->c.n_layer || kv_head_i < 0 || kv_head_i >= s->c.n_kv)
        return BK_ERR_ARG;
    const kv_head *k = &s->kv[layer * s->c.n_kv + kv_head_i];
    if (kq) *kq = k->kq;
    if (ke) *ke = k->ke;
    if (vq) *vq = k->vq;
    if (ve) *ve = k->ve;
    return BK_OK;
}

void bk_get_stats(const bk_state *s, bk_stats *out) { *out = s->st; }
void bk_reset_stats(bk_state *s) { memset(&s->st, 0, sizeof(s->st)); }

/* ------------------------------------------------------------------ attention */
/* One query position (all heads) of layer L against the Tk cached positions.
 * q [n_head][hd] -> ctx [n_head][hd]. Head h reads kv head h / rep. */
static void attend(bk_state *s, int L, int Tk, const float *q, float *ctx, const bk_taps *tp)
{
    const bk_config *c = &s->c;
    const int hd = s->hd, rep = s->rep, cap = s->cap;
    const float isq = c->inv_sqrt_hd;
    const float *tab = s->w.exp_tab;
    float *restrict kf = s->kf;

    for (int g = 0; g < c->n_kv; g++) {
        const kv_head *kv = &s->kv[L * c->n_kv + g];
        const float *qg = q + (size_t)g * rep * hd;

        /* scores: s_t = fsum(q * k_t) * fp32(1/sqrt(hd)) */
        for (int t = 0; t < Tk; t++) {
            bk_bfp_dequant16(kv->kq + (size_t)t * hd, kv->ke + (size_t)t * (hd / BK_KGRP), hd,
                             BK_KGRP, kf);
            for (int r = 0; r < rep; r++)
                s->sc[(size_t)r * cap + t] = dot_lanes(qg + (size_t)r * hd, kf, hd) * isq;
        }
        lap(s, BK_T_ATT_SCORE);

        /* softmax: m = max; e = exp_fab(s - m); Z = fsum(e); p = e * (1/Z) */
        for (int r = 0; r < rep; r++) {
            float *restrict sc = s->sc + (size_t)r * cap;
            int h = g * rep + r;
            if (tp && tp->att_s) {
                int n = Tk < tp->att_cap ? Tk : tp->att_cap;
                memcpy(tp->att_s + ((size_t)L * c->n_head + h) * tp->att_cap, sc, (size_t)n * 4);
            }
            float m = sc[0];
            for (int t = 1; t < Tk; t++) m = sc[t] > m ? sc[t] : m;
            for (int t = 0; t < Tk; t++) sc[t] = sc[t] - m;
            bk_exp_fab_n(tab, sc, sc, Tk);
            float Z = bk_fsum(sc, Tk);
            float rz = 1.0f / Z;
            for (int t = 0; t < Tk; t++) sc[t] = sc[t] * rz;
            if (tp && tp->att_p) {
                int n = Tk < tp->att_cap ? Tk : tp->att_cap;
                memcpy(tp->att_p + ((size_t)L * c->n_head + h) * tp->att_cap, sc, (size_t)n * 4);
            }
        }
        lap(s, BK_T_ATT_SOFTMAX);

        /* ctx_d = fsum over t (lane = t % 16, sequential per lane) of p_t * v_t,d */
        float *restrict cacc = s->cacc;
        memset(cacc, 0, (size_t)rep * P * hd * sizeof(float));
        for (int t = 0; t < Tk; t++) {
            bk_bfp_dequant8(kv->vq + (size_t)t * hd, kv->ve + (size_t)t * (hd / BK_QK), hd, BK_QK,
                            kf);
#if defined(BK_BUG_CTX_LANE)            /* planted bug: every position in lane 0 */
            int lane = 0;
#else
            int lane = t % P;
#endif
            for (int r = 0; r < rep; r++) {
                float pt = s->sc[(size_t)r * cap + t];
                float *restrict a = cacc + ((size_t)r * P + lane) * hd;
                for (int dd = 0; dd < hd; dd++) a[dd] = a[dd] + pt * kf[dd];
            }
        }
        for (int r = 0; r < rep; r++) {
            float *restrict a = cacc + (size_t)r * P * hd;
            float *restrict out = ctx + ((size_t)g * rep + r) * hd;
#if defined(BK_BUG_TREE)
            for (int w = P; w > 1; w /= 2)
                for (int i = 0; i < w / 2; i++)
                    for (int dd = 0; dd < hd; dd++)
                        a[(size_t)i * hd + dd] = a[(size_t)(2 * i) * hd + dd] + a[(size_t)(2 * i + 1) * hd + dd];
#else
            for (int w = P / 2; w >= 1; w /= 2)
                for (int i = 0; i < w; i++) {
                    float *restrict lo = a + (size_t)i * hd;
                    const float *restrict hi = a + (size_t)(i + w) * hd;
                    for (int dd = 0; dd < hd; dd++) lo[dd] = lo[dd] + hi[dd];
                }
#endif
            memcpy(out, a, (size_t)hd * sizeof(float));
        }
        lap(s, BK_T_ATT_CTX);
    }
}

/* ------------------------------------------------------------------ one token */
static int call_gemv(bk_state *s, bk_gemv_fn fn, void *fctx, int mat, const int16_t *xq,
                     const int8_t *xe, float *y, const bk_taps *tp)
{
    int n_in, rows;
    mat_shape(s, mat, &n_in, &rows);
    if (tp && (tp->xq || tp->xe)) {
        if (tp->xq) memcpy(tp->xq + s->off_xq[mat], xq, (size_t)n_in * sizeof(int16_t));
        if (tp->xe) memcpy(tp->xe + s->off_xe[mat], xe, (size_t)(n_in / BK_QK));
        lap(s, BK_T_TAPS);
    }
    int rc = fn(fctx, mat, xq, xe, n_in / BK_QK, y, rows);
    s->st.gemv_calls++;
    lap(s, BK_T_GEMV);
    if (rc != 0) {
        s->gemv_rc = rc;
        return BK_ERR_GEMV;
    }
    if (tp && tp->y) {
        memcpy(tp->y + s->off_y[mat], y, (size_t)rows * sizeof(float));
        lap(s, BK_T_TAPS);
    }
    return BK_OK;
}

static inline void tapf(bk_state *s, float *dst, size_t off, const float *src, size_t n)
{
    if (dst) {
        memcpy(dst + off, src, n * sizeof(float));
        lap(s, BK_T_TAPS);
    }
}

#define BK_TRY(expr)                    \
    do {                                \
        int rc_ = (expr);               \
        if (rc_ != BK_OK) return rc_;   \
    } while (0)

int bk_step(bk_state *s, int32_t token, int want_logits, bk_gemv_fn gemv, void *gemv_ctx,
            float *logits, const bk_taps *tp)
{
    if (!s || !gemv || (want_logits && !logits)) return BK_ERR_ARG;
    const bk_config *c = &s->c;
    const bk_weights *w = &s->w;
    if (token < 0 || token >= c->vocab) return BK_ERR_ARG;
    const int pos = s->pos, Tk = pos + 1;
    if (pos >= c->max_ctx || pos >= c->max_pos) return BK_ERR_CTX_FULL;
    BK_TRY(ensure_cap(s, Tk));
    const int d = c->d, ffn = c->ffn, hd = s->hd, nq = s->nq, nkv = s->nkv;
    const float *cosr = w->cos_tab + (size_t)pos * s->hd2;
    const float *sinr = w->sin_tab + (size_t)pos * s->hd2;
    float *restrict x = s->x;
    s->tlast = now_ns();

    bk_embed_row(w->emb_signs, w->emb_sign_stride, w->emb_scales, w->emb_scale_stride, d, token, x);
    lap(s, BK_T_EMBED);

    for (int L = 0; L < c->n_layer; L++) {
        const int m0 = 7 * L;
        if (tp) tapf(s, tp->x_in, (size_t)L * d, x, (size_t)d);

        /* --- attention --- */
        bk_rmsnorm(w->rs_lo, w->rs_hi, c->eps, x, w->attn_norm + (size_t)L * d, d, s->h);
        lap(s, BK_T_RMSNORM);
        BK_TRY(bk_bfp_quant16(s->h, d, BK_QK, BK_GEMV_BITS, s->xq_attn, s->xe_attn));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_Q, s->xq_attn, s->xe_attn, s->yq, tp));
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_K, s->xq_attn, s->xe_attn, s->yk, tp));
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_V, s->xq_attn, s->xe_attn, s->yv, tp));

        const float *qn = w->q_norm + (size_t)L * hd, *kn = w->k_norm + (size_t)L * hd;
        for (int h = 0; h < c->n_head; h++)
            bk_rmsnorm(w->rs_lo, w->rs_hi, c->eps, s->yq + (size_t)h * hd, qn, hd,
                       s->tq + (size_t)h * hd);
        lap(s, BK_T_RMSNORM);
        for (int h = 0; h < c->n_head; h++)
            bk_rope(cosr, sinr, s->tq + (size_t)h * hd, hd, s->qr + (size_t)h * hd);
        lap(s, BK_T_ROPE);
        for (int g = 0; g < c->n_kv; g++)
            bk_rmsnorm(w->rs_lo, w->rs_hi, c->eps, s->yk + (size_t)g * hd, kn, hd,
                       s->tq + (size_t)g * hd);
        lap(s, BK_T_RMSNORM);
        for (int g = 0; g < c->n_kv; g++)
            bk_rope(cosr, sinr, s->tq + (size_t)g * hd, hd, s->kr + (size_t)g * hd);
        lap(s, BK_T_ROPE);
        if (tp) {
            tapf(s, tp->q_rope, (size_t)L * nq, s->qr, (size_t)nq);
            tapf(s, tp->k_rope, (size_t)L * nkv, s->kr, (size_t)nkv);
        }

        /* K (after RoPE): BFP10, exponent per 16 channels. V: BFP8, exponent per 128. */
        for (int g = 0; g < c->n_kv; g++) {
            kv_head *kv = &s->kv[L * c->n_kv + g];
            BK_TRY(bk_bfp_quant16(s->kr + (size_t)g * hd, hd, BK_KGRP, BK_K_BITS,
                                  kv->kq + (size_t)pos * hd, kv->ke + (size_t)pos * (hd / BK_KGRP)));
            BK_TRY(bk_bfp_quant8(s->yv + (size_t)g * hd, hd, BK_QK, BK_V_BITS,
                                 kv->vq + (size_t)pos * hd, kv->ve + (size_t)pos * (hd / BK_QK)));
        }
        lap(s, BK_T_KVQ);

        attend(s, L, Tk, s->qr, s->ctx, tp);
        if (tp) tapf(s, tp->ctx, (size_t)L * nq, s->ctx, (size_t)nq);

        BK_TRY(bk_bfp_quant16(s->ctx, nq, BK_QK, BK_GEMV_BITS, s->xq_ctx, s->xe_ctx));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_O, s->xq_ctx, s->xe_ctx, s->yo, tp));
        for (int i = 0; i < d; i++) x[i] = x[i] + s->yo[i];
        lap(s, BK_T_RESID);
        if (tp) tapf(s, tp->x_mid, (size_t)L * d, x, (size_t)d);

        /* --- FFN --- */
        bk_rmsnorm(w->rs_lo, w->rs_hi, c->eps, x, w->ffn_norm + (size_t)L * d, d, s->h);
        lap(s, BK_T_RMSNORM);
        BK_TRY(bk_bfp_quant16(s->h, d, BK_QK, BK_GEMV_BITS, s->xq_ffn, s->xe_ffn));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_GATE, s->xq_ffn, s->xe_ffn, s->gate, tp));
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_UP, s->xq_ffn, s->xe_ffn, s->up, tp));
        bk_silu(w->exp_tab, s->gate, s->gate, ffn);         /* gate <- silu(gate) */
        for (int i = 0; i < ffn; i++) s->act[i] = s->gate[i] * s->up[i];
        lap(s, BK_T_SILU);
        if (tp) {
            tapf(s, tp->silu, (size_t)L * ffn, s->gate, (size_t)ffn);
            tapf(s, tp->act, (size_t)L * ffn, s->act, (size_t)ffn);
        }
        BK_TRY(bk_bfp_quant16(s->act, ffn, BK_QK, BK_GEMV_BITS, s->xq_act, s->xe_act));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_DOWN, s->xq_act, s->xe_act, s->yd, tp));
        for (int i = 0; i < d; i++) x[i] = x[i] + s->yd[i];
        lap(s, BK_T_RESID);
        if (tp) tapf(s, tp->x_out, (size_t)L * d, x, (size_t)d);
    }

    if (want_logits) {
        bk_rmsnorm(w->rs_lo, w->rs_hi, c->eps, x, w->out_norm, d, s->h);
        lap(s, BK_T_RMSNORM);
        if (tp) tapf(s, tp->h_out, 0, s->h, (size_t)d);
        BK_TRY(bk_bfp_quant16(s->h, d, BK_QK, BK_GEMV_BITS, s->xq_out, s->xe_out));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, 7 * c->n_layer, s->xq_out, s->xe_out, logits, tp));
    }
    s->pos = pos + 1;
    s->st.steps++;
    s->st.att_pos += (uint64_t)Tk;
    lap(s, BK_T_OTHER);
    return BK_OK;
}

/* ------------------------------------------------------------------ bench stand-in */
int bk_bench_gemv(void *ctx, int mat_id, const int16_t *xq, const int8_t *xe, int groups,
                  float *y, int rows)
{
    (void)ctx;
    (void)xe;
    uint32_t st = 0x9E3779B9u * (uint32_t)(mat_id + 1);
    for (int i = 0; i < groups * BK_QK; i += 61) st = st * 1664525u + (uint32_t)(uint16_t)xq[i] + 1013904223u;
    for (int i = 0; i < rows; i++) {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        if (st == 0) st = 0x1234567u;
        /* 24 random bits -> [-2, 2) */
        y[i] = ((float)(int32_t)(st >> 8) - 8388608.0f) * (1.0f / 4194304.0f);
    }
    return 0;
}
