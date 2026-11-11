/* bk_host.c: the Big-Kev vector path, bit-identical to fabric/big/ref/fabric_ref.py.
 * See bk_host.h for the API, the build flags and the GEMV numbering.
 *
 * How bit-identity is kept (read before touching anything here):
 *   - every fp32 operation of the spec is one C float operation, in the spec's order. No
 *     a*b+c contraction (flag + pragma + selftest), no reassociation (never -ffast-math).
 *   - reductions are written with the spec's 16 explicit lanes, so SIMD can be used without
 *     reordering anything: lanes are independent, each lane is sequential.
 *   - "x + 0" for the zero padding of the spec is skipped: a lane accumulator starts at +0
 *     and can never become -0 (IEEE: only (-0) + (-0) gives -0), so adding +0 is a no-op.
 *   - 2^n is built from the exponent bits; frexp / floor / round-half-even are done on the
 *     bits, with an exact int conversion, or with the 1.5*2^23 trick (exact RNE for
 *     |t| < 2^22 in round-to-nearest-even mode). No libm.
 *   - the spec's bfp_quant works in float64. Here: fp32 x * 2^-e is exact unless it
 *     underflows, and an underflowed value (< 2^-126) rounds to 0 either way; exponents the
 *     fp32 path cannot represent take a float64 path. Same integers in every case.
 *
 * Two sets of kernels (bk_host.h): the scalar REFERENCE kernels (*_ref, the readable
 * statement of the spec) and the FAST kernels (*_fast). The fast ones do the same fp32
 * operations on the same operands; what changes is only the order in which INDEPENDENT
 * operations are issued and where intermediate values live:
 *   - elementwise maps and the 16-lane sums are written 16 elements (4 vectors of 4) per
 *     iteration, operation by operation. A Cortex-A53 issues in order and stalls on an
 *     operand that is not ready, so "mul; add; store" per vector runs at the latency of the
 *     chain; four vectors side by side run at the throughput of the pipes.
 *   - floor / round-half-even use the exact instructions (frintm, fcvtns / cvtps2dq)
 *     instead of the bit tricks; the results are the same integers.
 *   - attention scores: the K row is dequantised in registers, one exponent group (16
 *     channels = the 16 lanes of the sum) at a time, and the lane accumulators of the `rep`
 *     query heads that share the kv head stay in registers.
 *   - attention context: computed lane by lane (lane = t % 16). For one lane and one group
 *     of 16 channels the accumulators of the `rep` heads stay in registers while the lane's
 *     rows are walked in order; V is dequantised in registers. The 16 lane results are then
 *     added with the spec's tree. Per lane the additions are the spec's, in the spec's order.
 * Anything a fast kernel does not handle exactly (NaN in exp, exponents below -126, the
 * float64 quantiser path, odd sizes) is handed to the reference kernel for that block.
 *
 * -DBK_BUG_* plant deliberate spec violations; fabric/big/run_host.py uses them to show the
 * gate has teeth. Never define them in a real build.
 */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "bk_host.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* SIMD tier of the fast kernels: 1 NEON (aarch64), 2 SSE2, 0 none (reference kernels only) */
#if !defined(BK_NO_SIMD) && defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#define BK_SIMD 1
#elif !defined(BK_NO_SIMD) && defined(__SSE2__) && (defined(__GNUC__) || defined(__clang__))
#include <emmintrin.h>
#define BK_SIMD 2
#else
#define BK_SIMD 0
#endif

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
#define BK_INLINE static inline __attribute__((always_inline))
/* gcc's pre-register-allocation scheduler (on by default on aarch64) hoists every load of a
 * basic block to its top. In the attention kernels that makes more values live than there
 * are NEON registers and spills accumulators to the stack. An empty volatile asm is a
 * scheduling barrier (and emits nothing): the kernels use it to keep the loads of at most
 * two heads in one scheduling region. */
#if defined(__GNUC__)
#define BK_BARRIER __asm__ volatile("")
#else
#define BK_BARRIER (void)0
#endif
/* BK_PINV(v) says "v is in a NEON register here" and emits nothing. gcc otherwise moves a
 * value that is used once down to its use (and its scheduler then sorts what is left), which
 * turns code written operation-by-operation across four vectors back into four dependent
 * chains. The pins fix the issue order on the A53; on other targets they expand to nothing. */
#if defined(__GNUC__) && defined(__aarch64__) && !defined(BK_NO_PIN)
#define BK_PINV(v) __asm__ volatile("" : "+w"(v))
#define BK_PING(g) __asm__ volatile("" : "+r"(g))
#define BK_PINVG(v, g) __asm__ volatile("" : "+w"(v), "+r"(g))
#else
#define BK_PINV(v) (void)0
#define BK_PING(g) (void)0
#define BK_PINVG(v, g) (void)0
#endif
/* the kernels are not inlined into their callers: nothing to gain, and it keeps their
 * register allocation (and their assembly, when it has to be read) to themselves */
#define BK_KERN __attribute__((noinline))

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

/* ================================================================== reference kernels */
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

/* the tail of a lane sum: elements i..n-1 go to lanes 0.. (i is a multiple of 16) */
static inline float lanes_tail(float *acc, const float *x, int i, int n, int sq)
{
    for (int l = 0; l < P && i < n; i++, l++) acc[l] = acc[l] + (sq ? x[i] * x[i] : x[i]);
    return tree16(acc);
}

static float fsum_ref(const float *x, int n)
{
    float acc[P] = {0};
    int i = 0;
    for (; i + P <= n; i += P)
        for (int l = 0; l < P; l++) acc[l] = acc[l] + x[i + l];
    return lanes_tail(acc, x, i, n, 0);
}

/* fsum(x*x) */
static float sumsq_ref(const float *restrict x, int n)
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
    return lanes_tail(acc, x, i, n, 1);
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
 * (A) index and scale; (B) the table lookups; (C) the 0 / inf / NaN selects.
 * n < -126 <=> y < -126 and n > 127 <=> y >= 128, so (C) decides on y; for those y the masked
 * intermediate of (A) is discarded. */
#define BK_CH 256
static void exp_block_ref(const float *restrict tab, const float *x, float *e, int n)
{
    float ys[BK_CH];
    int32_t kk[BK_CH];
    uint32_t sb[BK_CH];
    for (int i = 0; i < n; i++) {
        float y = x[i] * BK_LOG2E;
        ys[i] = y;
        /* |y| >= 128 (and NaN) -> 0 so the int conversion stays defined; pass (C) overrides
         * those results. */
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

static void exp_ref(const float *tab, const float *x, float *y, int n)
{
    for (int i = 0; i < n; i += BK_CH) exp_block_ref(tab, x + i, y + i, n - i < BK_CH ? n - i : BK_CH);
}

static void silu_ref(const float *tab, const float *x, float *y, int n)
{
    float nx[BK_CH], e[BK_CH];
    for (int i0 = 0; i0 < n; i0 += BK_CH) {
        int m = n - i0 < BK_CH ? n - i0 : BK_CH;
        const float *xi = x + i0;
        float *yi = y + i0;
        for (int i = 0; i < m; i++) nx[i] = -xi[i];
        exp_block_ref(tab, nx, e, m);
        for (int i = 0; i < m; i++) yi[i] = xi[i] * (1.0f / (1.0f + e[i]));
    }
}

static inline int ilog2(int n)          /* floor(log2(n)), n >= 1 (the spec: int(math.log2(n))) */
{
    int k = 0;
    while ((n >> (k + 1)) != 0) k++;
    return k;
}

static void rmsnorm_ref(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                        const float *w, int n, float *y)
{
    float ss = sumsq_ref(x, n);
    float mean = ss * pow2f(-ilog2(n));
    float r = bk_rsqrt_fab(rs_lo, rs_hi, mean + eps);
    for (int i = 0; i < n; i++) y[i] = w[i] * (x[i] * r);
}

static void rope_ref(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y)
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

/* q = clamp(RNE(x * 2^-e), -qmax, qmax) written as float integers into r[n].
 * (The clamp never engages: the exponent rule keeps max|x| * 2^-e <= qmax.) */
static inline void bfp_round_group(const float *restrict x, int n, int e, float qmax,
                                   float *restrict r)
{
    if (-e <= 127) {                    /* -e >= -122 always: 2^-e is an fp32 */
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

#define BK_ROUND_REF(QT, xg, qg, group, ex, qmax, r)                                       \
    for (int i0 = 0; i0 < (group); i0 += BK_QK) {                                          \
        int m = (group) - i0 < BK_QK ? (group) - i0 : BK_QK;                               \
        bfp_round_group((xg) + i0, m, (ex), (qmax), (r));                                  \
        for (int i = 0; i < m; i++) (qg)[i0 + i] = (QT)(int32_t)(r)[i];                    \
    }

#define BK_QUANT_REF_FN(NAME, QT)                                                          \
    static int NAME(const float *x, int n, int group, int bits, QT *q, int8_t *e)          \
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
            BK_ROUND_REF(QT, xg, qg, group, ex, qmax, r)                                   \
        }                                                                                  \
        return BK_OK;                                                                      \
    }

BK_QUANT_REF_FN(quant_ref16, int16_t)
BK_QUANT_REF_FN(quant_ref8, int8_t)

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

static inline int any_tiny_exp(const int8_t *e, int n)     /* an exponent 2^e is not a normal fp32 */
{
    int m = 0;
    for (int i = 0; i < n; i++) m |= (e[i] < -126);
    return m;
}

/* ================================================================== fast kernels */
#if BK_SIMD
typedef float v4f __attribute__((vector_size(16)));
typedef int32_t v4i __attribute__((vector_size(16)));

/* unaligned loads / stores. (With memcpy gcc moves some of them through the integer
 * registers on aarch64, which costs several cycles each on the A53; the intrinsics do not.) */
#if BK_SIMD == 1
BK_INLINE v4f ld4(const float *p) { return (v4f)vld1q_f32(p); }
BK_INLINE void st4(float *p, v4f v) { vst1q_f32(p, (float32x4_t)v); }
BK_INLINE v4i ld4i(const void *p) { return (v4i)vld1q_s32((const int32_t *)p); }
BK_INLINE void st4i(void *p, v4i v) { vst1q_s32((int32_t *)p, (int32x4_t)v); }
#elif BK_SIMD == 2
BK_INLINE v4f ld4(const float *p) { return (v4f)_mm_loadu_ps(p); }
BK_INLINE void st4(float *p, v4f v) { _mm_storeu_ps(p, (__m128)v); }
BK_INLINE v4i ld4i(const void *p) { return (v4i)_mm_loadu_si128((const __m128i *)p); }
BK_INLINE void st4i(void *p, v4i v) { _mm_storeu_si128((__m128i *)p, (__m128i)v); }
#else
BK_INLINE v4f ld4(const float *p) { v4f v; memcpy(&v, p, 16); return v; }
BK_INLINE void st4(float *p, v4f v) { memcpy(p, &v, 16); }
BK_INLINE v4i ld4i(const void *p) { v4i v; memcpy(&v, p, 16); return v; }
BK_INLINE void st4i(void *p, v4i v) { memcpy(p, &v, 16); }
#endif
BK_INLINE v4f v4_set1(float x) { return (v4f){x, x, x, x}; }
BK_INLINE v4i v4i_set1(int32_t x) { return (v4i){x, x, x, x}; }

#if BK_SIMD == 1                        /* ---- NEON ---- */
BK_INLINE v4f v4_floor(v4f y) { return (v4f)vrndmq_f32((float32x4_t)y); }
BK_INLINE v4i v4_trunc(v4f y) { return (v4i)vcvtq_s32_f32((float32x4_t)y); }
BK_INLINE v4i v4_rne(v4f t) { return (v4i)vcvtnq_s32_f32((float32x4_t)t); }      /* round half even */
/* lo <= result <= hi for a finite y; a NaN y gives a NaN or lo (either is only ever fed to
 * conversions that are defined for it, and the caller redoes NaN blocks in the reference) */
BK_INLINE v4f v4_clamp(v4f y, float lo, float hi)
{
    return (v4f)vminq_f32(vmaxq_f32((float32x4_t)y, vdupq_n_f32(lo)), vdupq_n_f32(hi));
}
BK_INLINE v4f v4_lo(v4f y, float lo) { return (v4f)vmaxq_f32((float32x4_t)y, vdupq_n_f32(lo)); }
BK_INLINE v4f v4_hi(v4f y, float hi) { return (v4f)vminq_f32((float32x4_t)y, vdupq_n_f32(hi)); }
/* v * p[lane] (one instruction: fmul by element) */
#define v4_mul_lane(v, p, lane) ((v4f)vmulq_laneq_f32((float32x4_t)(v), (float32x4_t)(p), (lane)))
BK_INLINE v4i v4i_max(v4i a, v4i b) { return (v4i)vmaxq_s32((int32x4_t)a, (int32x4_t)b); }
BK_INLINE int32_t v4i_hmax(v4i a) { return vmaxvq_s32((int32x4_t)a); }
BK_INLINE void ld_i16_f(const int16_t *p, v4f *o)         /* 8 int16 -> 2 x 4 float, exact */
{
    int16x8_t v = vld1q_s16(p);
    o[0] = (v4f)vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
    o[1] = (v4f)vcvtq_f32_s32(vmovl_high_s16(v));
}
BK_INLINE void ld_i8_f(const int8_t *p, v4f *o)           /* 16 int8 -> 4 x 4 float, exact */
{
    int8x16_t v = vld1q_s8(p);
    int16x8_t a = vmovl_s8(vget_low_s8(v)), b = vmovl_high_s8(v);
    o[0] = (v4f)vcvtq_f32_s32(vmovl_s16(vget_low_s16(a)));
    o[1] = (v4f)vcvtq_f32_s32(vmovl_high_s16(a));
    o[2] = (v4f)vcvtq_f32_s32(vmovl_s16(vget_low_s16(b)));
    o[3] = (v4f)vcvtq_f32_s32(vmovl_high_s16(b));
}
BK_INLINE void st_q16(int16_t *p, v4i a, v4i b, v4i c, v4i d)     /* 16 int32 -> int16 */
{
    vst1q_s16(p, vcombine_s16(vqmovn_s32((int32x4_t)a), vqmovn_s32((int32x4_t)b)));
    vst1q_s16(p + 8, vcombine_s16(vqmovn_s32((int32x4_t)c), vqmovn_s32((int32x4_t)d)));
}
BK_INLINE void st_q8(int8_t *p, v4i a, v4i b, v4i c, v4i d)       /* 16 int32 -> int8 */
{
    int16x8_t lo = vcombine_s16(vqmovn_s32((int32x4_t)a), vqmovn_s32((int32x4_t)b));
    int16x8_t hi = vcombine_s16(vqmovn_s32((int32x4_t)c), vqmovn_s32((int32x4_t)d));
    vst1q_s8(p, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
}
#else                                   /* ---- SSE2 ---- */
BK_INLINE v4f v4_floor(v4f y)                              /* |y| < 2^31 */
{
    __m128 f = _mm_cvtepi32_ps(_mm_cvttps_epi32((__m128)y));
    return (v4f)_mm_sub_ps(f, _mm_and_ps(_mm_cmpgt_ps(f, (__m128)y), _mm_set1_ps(1.0f)));
}
BK_INLINE v4i v4_trunc(v4f y) { return (v4i)_mm_cvttps_epi32((__m128)y); }
BK_INLINE v4i v4_rne(v4f t) { return (v4i)_mm_cvtps_epi32((__m128)t); }           /* MXCSR: RNE */
BK_INLINE v4f v4_clamp(v4f y, float lo, float hi)          /* NaN y -> lo */
{
    return (v4f)_mm_min_ps(_mm_max_ps((__m128)y, _mm_set1_ps(lo)), _mm_set1_ps(hi));
}
BK_INLINE v4f v4_lo(v4f y, float lo) { return (v4f)_mm_max_ps((__m128)y, _mm_set1_ps(lo)); }
BK_INLINE v4f v4_hi(v4f y, float hi) { return (v4f)_mm_min_ps((__m128)y, _mm_set1_ps(hi)); }
#define v4_mul_lane(v, p, lane) \
    ((v4f)_mm_mul_ps((__m128)(v), _mm_shuffle_ps((__m128)(p), (__m128)(p), _MM_SHUFFLE((lane), (lane), (lane), (lane)))))
BK_INLINE v4i v4i_max(v4i a, v4i b)
{
    v4i m = a > b;
    return (a & m) | (b & ~m);
}
BK_INLINE int32_t v4i_hmax(v4i a)
{
    int32_t m = a[0];
    for (int i = 1; i < 4; i++) m = a[i] > m ? a[i] : m;
    return m;
}
BK_INLINE void ld_i16_f(const int16_t *p, v4f *o)
{
    __m128i v = _mm_loadu_si128((const __m128i *)(const void *)p), sg = _mm_srai_epi16(v, 15);
    o[0] = (v4f)_mm_cvtepi32_ps(_mm_unpacklo_epi16(v, sg));
    o[1] = (v4f)_mm_cvtepi32_ps(_mm_unpackhi_epi16(v, sg));
}
BK_INLINE void ld_i8_f(const int8_t *p, v4f *o)
{
    __m128i v = _mm_loadu_si128((const __m128i *)(const void *)p);
    __m128i sg = _mm_cmpgt_epi8(_mm_setzero_si128(), v);
    __m128i a = _mm_unpacklo_epi8(v, sg), b = _mm_unpackhi_epi8(v, sg);
    __m128i sa = _mm_srai_epi16(a, 15), sb = _mm_srai_epi16(b, 15);
    o[0] = (v4f)_mm_cvtepi32_ps(_mm_unpacklo_epi16(a, sa));
    o[1] = (v4f)_mm_cvtepi32_ps(_mm_unpackhi_epi16(a, sa));
    o[2] = (v4f)_mm_cvtepi32_ps(_mm_unpacklo_epi16(b, sb));
    o[3] = (v4f)_mm_cvtepi32_ps(_mm_unpackhi_epi16(b, sb));
}
BK_INLINE void st_q16(int16_t *p, v4i a, v4i b, v4i c, v4i d)
{
    _mm_storeu_si128((__m128i *)(void *)p, _mm_packs_epi32((__m128i)a, (__m128i)b));
    _mm_storeu_si128((__m128i *)(void *)(p + 8), _mm_packs_epi32((__m128i)c, (__m128i)d));
}
BK_INLINE void st_q8(int8_t *p, v4i a, v4i b, v4i c, v4i d)
{
    __m128i lo = _mm_packs_epi32((__m128i)a, (__m128i)b), hi = _mm_packs_epi32((__m128i)c, (__m128i)d);
    _mm_storeu_si128((__m128i *)(void *)p, _mm_packs_epi16(lo, hi));
}
#endif

/* the tree16 of four vectors holding lanes 0..15 */
BK_INLINE float tree_v(v4f a0, v4f a1, v4f a2, v4f a3)
{
#if defined(BK_BUG_TREE)
    float a[P];
    st4(a, a0); st4(a + 4, a1); st4(a + 8, a2); st4(a + 12, a3);
    return tree16(a);
#else
    v4f b0 = a0 + a2, b1 = a1 + a3;     /* [0:8] + [8:16] */
    v4f c = b0 + b1;                    /* [0:4] + [4:8] */
    float d0 = c[0] + c[2], d1 = c[1] + c[3];
    return d0 + d1;
#endif
}

BK_KERN static float fsum_fast(const float *x, int n)
{
    v4f a0 = v4_set1(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + P <= n; i += P) {
        a0 = a0 + ld4(x + i);
        a1 = a1 + ld4(x + i + 4);
        a2 = a2 + ld4(x + i + 8);
        a3 = a3 + ld4(x + i + 12);
    }
    if (i == n) return tree_v(a0, a1, a2, a3);
    float acc[P];
    st4(acc, a0); st4(acc + 4, a1); st4(acc + 8, a2); st4(acc + 12, a3);
    return lanes_tail(acc, x, i, n, 0);
}

BK_KERN static float sumsq_fast(const float *x, int n)
{
#if defined(BK_BUG_REASSOC) || defined(BK_BUG_FMA)
    return sumsq_ref(x, n);
#else
    v4f a0 = v4_set1(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + P <= n; i += P) {
        v4f x0 = ld4(x + i), x1 = ld4(x + i + 4), x2 = ld4(x + i + 8), x3 = ld4(x + i + 12);
        x0 = x0 * x0; x1 = x1 * x1; x2 = x2 * x2; x3 = x3 * x3;
        a0 = a0 + x0; a1 = a1 + x1; a2 = a2 + x2; a3 = a3 + x3;
    }
    if (i == n) return tree_v(a0, a1, a2, a3);
    float acc[P];
    st4(acc, a0); st4(acc + 4, a1); st4(acc + 8, a2); st4(acc + 12, a3);
    return lanes_tail(acc, x, i, n, 1);
#endif
}

BK_KERN static void rmsnorm_fast(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                         const float *w, int n, float *y)
{
    float ss = sumsq_fast(x, n);
    float mean = ss * pow2f(-ilog2(n));
    float r = bk_rsqrt_fab(rs_lo, rs_hi, mean + eps);
    const v4f rv = v4_set1(r);
    int i = 0;
    for (; i + P <= n; i += P) {
        v4f t0 = ld4(x + i) * rv, t1 = ld4(x + i + 4) * rv, t2 = ld4(x + i + 8) * rv, t3 = ld4(x + i + 12) * rv;
        t0 = ld4(w + i) * t0; t1 = ld4(w + i + 4) * t1; t2 = ld4(w + i + 8) * t2; t3 = ld4(w + i + 12) * t3;
        st4(y + i, t0); st4(y + i + 4, t1); st4(y + i + 8, t2); st4(y + i + 12, t3);
    }
    for (; i < n; i++) y[i] = w[i] * (x[i] * r);
}

BK_KERN static void rope_fast(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y)
{
    int h2 = head_dim / 2, i = 0;
    const float *x1 = x, *x2 = x + h2;
    float *y1 = y, *y2 = y + h2;
    for (; i + 8 <= h2; i += 8) {
        v4f a0 = ld4(x1 + i), b0 = ld4(x2 + i), c0 = ld4(cos_row + i), s0 = ld4(sin_row + i);
        v4f a1 = ld4(x1 + i + 4), b1 = ld4(x2 + i + 4), c1 = ld4(cos_row + i + 4), s1 = ld4(sin_row + i + 4);
        v4f ac0 = a0 * c0, bs0 = b0 * s0, bc0 = b0 * c0, as0 = a0 * s0;
        v4f ac1 = a1 * c1, bs1 = b1 * s1, bc1 = b1 * c1, as1 = a1 * s1;
        st4(y1 + i, ac0 - bs0); st4(y2 + i, bc0 + as0);
        st4(y1 + i + 4, ac1 - bs1); st4(y2 + i + 4, bc1 + as1);
    }
    for (; i < h2; i++) {
        float c = cos_row[i], s = sin_row[i], a = x1[i], b = x2[i];
        y1[i] = a * c - b * s;
        y2[i] = b * c + a * s;
    }
}

/* x += y */
BK_KERN static void add_fast(float *x, const float *y, int n)
{
    int i = 0;
    for (; i + P <= n; i += P) {
        v4f a0 = ld4(x + i) + ld4(y + i), a1 = ld4(x + i + 4) + ld4(y + i + 4);
        v4f a2 = ld4(x + i + 8) + ld4(y + i + 8), a3 = ld4(x + i + 12) + ld4(y + i + 12);
        st4(x + i, a0); st4(x + i + 4, a1); st4(x + i + 8, a2); st4(x + i + 12, a3);
    }
    for (; i < n; i++) x[i] = x[i] + y[i];
}

/* Passes (A) and (B) of exp_fab(mult/LOG2E * x) over m16 elements (a multiple of 16):
 * tv[i] = TAB[k_i], sb[i] = bits of 2^n_i. y is clamped to [-127, 128] first, which makes the
 * spec's two range rules fall out of the arithmetic: n = -127 gives scale bits 0 (+0.0,
 * and TAB*0 = +0), n = 128 gives the bits of +inf (TAB*inf = +inf), and at the clamp points
 * f = 0, k = 0. Returns 0 if a NaN was seen (the caller redoes the block in the reference). */
BK_INLINE int exp_prep(const float *restrict tab, const float *xs, float mult, int m16,
                       float *restrict tv, int32_t *restrict sb)
{
    int32_t kk[BK_CH];
    const v4f mv = v4_set1(mult), k4096 = v4_set1(4096.0f);
    const v4i c127 = v4i_set1(127);
    v4i ok = v4i_set1(-1);
    for (int i = 0; i < m16; i += P) {
        v4f y0 = ld4(xs + i) * mv, y1 = ld4(xs + i + 4) * mv, y2 = ld4(xs + i + 8) * mv, y3 = ld4(xs + i + 12) * mv;
        ok &= (y0 == y0) & (y1 == y1) & (y2 == y2) & (y3 == y3);
        y0 = v4_clamp(y0, -127.0f, 128.0f); y1 = v4_clamp(y1, -127.0f, 128.0f);
        y2 = v4_clamp(y2, -127.0f, 128.0f); y3 = v4_clamp(y3, -127.0f, 128.0f);
        v4f f0 = v4_floor(y0), f1 = v4_floor(y1), f2 = v4_floor(y2), f3 = v4_floor(y3);
        v4i n0 = v4_trunc(f0), n1 = v4_trunc(f1), n2 = v4_trunc(f2), n3 = v4_trunc(f3);
        y0 = (y0 - f0) * k4096; y1 = (y1 - f1) * k4096; y2 = (y2 - f2) * k4096; y3 = (y3 - f3) * k4096;
#if defined(BK_BUG_EXP_TRUNC)
        v4i k0 = v4_trunc(y0), k1 = v4_trunc(y1), k2 = v4_trunc(y2), k3 = v4_trunc(y3);
#else
        v4i k0 = v4_rne(y0), k1 = v4_rne(y1), k2 = v4_rne(y2), k3 = v4_rne(y3);
#endif
        n0 = (n0 + c127) << 23; n1 = (n1 + c127) << 23; n2 = (n2 + c127) << 23; n3 = (n3 + c127) << 23;
        st4i(kk + i, k0); st4i(kk + i + 4, k1); st4i(kk + i + 8, k2); st4i(kk + i + 12, k3);
        st4i(sb + i, n0); st4i(sb + i + 4, n1); st4i(sb + i + 8, n2); st4i(sb + i + 12, n3);
    }
    if ((ok[0] & ok[1] & ok[2] & ok[3]) != -1) return 0;
    for (int i = 0; i < m16; i += 4) {
        float a = tab[kk[i]], b = tab[kk[i + 1]], c = tab[kk[i + 2]], d = tab[kk[i + 3]];
        tv[i] = a; tv[i + 1] = b; tv[i + 2] = c; tv[i + 3] = d;
    }
    return 1;
}

BK_KERN static void exp_fast(const float *tab, const float *x, float *e, int n)
{
    int32_t sb[BK_CH];
    float tv[BK_CH], xb[BK_CH];
    for (int i0 = 0; i0 < n; i0 += BK_CH) {
        const int m = n - i0 < BK_CH ? n - i0 : BK_CH, m16 = (m + P - 1) & ~(P - 1);
        const float *xs = x + i0;
        float *eo = e + i0;
        if (m16 != m) {
            memcpy(xb, xs, (size_t)m * 4);
            memset(xb + m, 0, (size_t)(m16 - m) * 4);
            xs = xb;
            eo = xb;
        }
        if (!exp_prep(tab, xs, BK_LOG2E, m16, tv, sb)) {
            exp_block_ref(tab, x + i0, e + i0, m);
            continue;
        }
        for (int i = 0; i < m16; i += P) {
            v4f e0 = ld4(tv + i) * (v4f)ld4i(sb + i), e1 = ld4(tv + i + 4) * (v4f)ld4i(sb + i + 4);
            v4f e2 = ld4(tv + i + 8) * (v4f)ld4i(sb + i + 8), e3 = ld4(tv + i + 12) * (v4f)ld4i(sb + i + 12);
            st4(eo + i, e0); st4(eo + i + 4, e1); st4(eo + i + 8, e2); st4(eo + i + 12, e3);
        }
        if (m16 != m) memcpy(e + i0, xb, (size_t)m * 4);
    }
}

/* s = x * (1 / (1 + exp_fab(-x))); sil (may be NULL, may alias x) <- s; act (may be NULL) <- s * up.
 *
 * Elementwise the operations are exp_fast's and the reference's. The loop is a three-stage
 * software pipeline over groups of 16, because of what the A53 is (measured): one 128-bit
 * NEON operation per cycle, a 128-bit divide that takes 10 cycles but lets other operations
 * issue meanwhile, integer loads that pair with NEON operations, and no reordering. So in
 * one iteration: the four divides of group g are issued one at a time, each followed by one
 * vector of pass (A) of group g+2 (index and scale -> kk, sb) and four table lookups of
 * group g+1 (integer side); then the divides are consumed (group g is finished) and
 * 1 + e of group g+1 is formed for the next iteration.
 * A NaN input gives a NaN output (through x * ...; its payload is not specified). */
#define BK_SCH 512
#if defined(BK_BUG_FAST_CLAMP)
#define BK_SILU_HI 127.0f
#else
#define BK_SILU_HI 128.0f
#endif
#if defined(BK_BUG_EXP_TRUNC)
#define BK_EXP_RNE(y) v4_trunc(y)
#else
#define BK_EXP_RNE(y) v4_rne(y)
#endif
#define SILU_P(o, j)                                                                       \
    {                                                                                      \
        v4f y_ = v4_clamp(ld4(xs + (o) + 4 * (j)) * mv, -127.0f, 128.0f);                  \
        v4f f_ = v4_floor(y_);                                                             \
        v4i n_ = v4_trunc(f_);                                                             \
        y_ = (y_ - f_) * k4096;                                                            \
        st4i(kk + (o) + 4 * (j), BK_EXP_RNE(y_));                                          \
        st4i(sb + (o) + 4 * (j), (n_ + c127) << 23);                                       \
    }
#define SILU_G(o, j)                                                                       \
    {                                                                                      \
        const int32_t *k_ = kk + (o) + 4 * (j);                                            \
        float a_ = tab[k_[0]], b_ = tab[k_[1]], c_ = tab[k_[2]], d_ = tab[k_[3]];          \
        float *t_ = tv + (o) + 4 * (j);                                                    \
        t_[0] = a_; t_[1] = b_; t_[2] = c_; t_[3] = d_;                                    \
    }
#define SILU_DEN(o)                                                                        \
    den0 = one + ld4(tv + (o)) * (v4f)ld4i(sb + (o));                                      \
    den1 = one + ld4(tv + (o) + 4) * (v4f)ld4i(sb + (o) + 4);                              \
    den2 = one + ld4(tv + (o) + 8) * (v4f)ld4i(sb + (o) + 8);                              \
    den3 = one + ld4(tv + (o) + 12) * (v4f)ld4i(sb + (o) + 12);
/* The main loop, written in the order the A53 should issue it: rows of four independent
 * operations (one per vector of the group of 16), so that no operation waits for the one
 * before it. Per iteration g:
 *   - SILU_F4: the outputs of group g-1, from the quotients of the previous iteration;
 *   - SILU_PGD: stage P of group g+2, with the gather of group g+1 (integer loads and
 *     stores, which issue next to a NEON operation) zipped in;
 *   - SILU_DEN4: the denominators of group g+1;
 *   - the four divides of group g, spread evenly over the iteration (one before each of
 *     F4, the two halves of PGD and DEN4): the A53 has one iterative divider, and a second
 *     divide issued before the first is done stalls everything behind it, while other
 *     operations issue freely meanwhile.
 * BK_PIN* after every step keeps gcc from undoing the order (see BK_PINV). */
#define SILU_PGD(xsrc, o2, o1)                                                                     \
    {                                                                                              \
        const float *xs2_ = (xsrc); int32_t *kk2_ = kk + (o2), *sb2_ = sb + (o2);                  \
        const int32_t *kk1_ = kk + (o1); float *tv1_ = tv + (o1);                                  \
        rn1 = one / den1; BK_PINV(rn1);                                                            \
        v4f y0_ = ld4(xs2_ + 0); int32_t i0_ = kk1_[0]; BK_PINVG(y0_, i0_);                        \
        v4f y1_ = ld4(xs2_ + 4); int32_t i1_ = kk1_[1]; BK_PINVG(y1_, i1_);                        \
        v4f y2_ = ld4(xs2_ + 8); int32_t i2_ = kk1_[2]; BK_PINVG(y2_, i2_);                        \
        v4f y3_ = ld4(xs2_ + 12); int32_t i3_ = kk1_[3]; BK_PINVG(y3_, i3_);                       \
        y0_ = y0_ * mv; uint32_t t0_; memcpy(&t0_, tab + i0_, 4); BK_PINVG(y0_, t0_);              \
        y1_ = y1_ * mv; uint32_t t1_; memcpy(&t1_, tab + i1_, 4); BK_PINVG(y1_, t1_);              \
        y2_ = y2_ * mv; uint32_t t2_; memcpy(&t2_, tab + i2_, 4); BK_PINVG(y2_, t2_);              \
        y3_ = y3_ * mv; uint32_t t3_; memcpy(&t3_, tab + i3_, 4); BK_PINVG(y3_, t3_);              \
        y0_ = v4_lo(y0_, -127.0f); memcpy(tv1_ + 0, &t0_, 4); BK_PINV(y0_);                        \
        y1_ = v4_lo(y1_, -127.0f); memcpy(tv1_ + 1, &t1_, 4); BK_PINV(y1_);                        \
        y2_ = v4_lo(y2_, -127.0f); memcpy(tv1_ + 2, &t2_, 4); BK_PINV(y2_);                        \
        y3_ = v4_lo(y3_, -127.0f); memcpy(tv1_ + 3, &t3_, 4); BK_PINV(y3_);                        \
        y0_ = v4_hi(y0_, BK_SILU_HI); int32_t i4_ = kk1_[4]; BK_PINVG(y0_, i4_);                       \
        y1_ = v4_hi(y1_, BK_SILU_HI); int32_t i5_ = kk1_[5]; BK_PINVG(y1_, i5_);                       \
        y2_ = v4_hi(y2_, BK_SILU_HI); int32_t i6_ = kk1_[6]; BK_PINVG(y2_, i6_);                       \
        y3_ = v4_hi(y3_, BK_SILU_HI); int32_t i7_ = kk1_[7]; BK_PINVG(y3_, i7_);                       \
        v4f f0_ = v4_floor(y0_); uint32_t t4_; memcpy(&t4_, tab + i4_, 4); BK_PINVG(f0_, t4_);     \
        v4f f1_ = v4_floor(y1_); uint32_t t5_; memcpy(&t5_, tab + i5_, 4); BK_PINVG(f1_, t5_);     \
        v4f f2_ = v4_floor(y2_); uint32_t t6_; memcpy(&t6_, tab + i6_, 4); BK_PINVG(f2_, t6_);     \
        v4f f3_ = v4_floor(y3_); uint32_t t7_; memcpy(&t7_, tab + i7_, 4); BK_PINVG(f3_, t7_);     \
        y0_ = y0_ - f0_; memcpy(tv1_ + 4, &t4_, 4); BK_PINV(y0_);                                  \
        y1_ = y1_ - f1_; memcpy(tv1_ + 5, &t5_, 4); BK_PINV(y1_);                                  \
        y2_ = y2_ - f2_; memcpy(tv1_ + 6, &t6_, 4); BK_PINV(y2_);                                  \
        y3_ = y3_ - f3_; memcpy(tv1_ + 7, &t7_, 4); BK_PINV(y3_);                                  \
        rn2 = one / den2; BK_PINV(rn2);                                                            \
        v4i n0_ = v4_trunc(f0_); int32_t i8_ = kk1_[8]; BK_PINVG(n0_, i8_);                        \
        v4i n1_ = v4_trunc(f1_); int32_t i9_ = kk1_[9]; BK_PINVG(n1_, i9_);                        \
        v4i n2_ = v4_trunc(f2_); int32_t i10_ = kk1_[10]; BK_PINVG(n2_, i10_);                     \
        v4i n3_ = v4_trunc(f3_); int32_t i11_ = kk1_[11]; BK_PINVG(n3_, i11_);                     \
        y0_ = y0_ * k4096; uint32_t t8_; memcpy(&t8_, tab + i8_, 4); BK_PINVG(y0_, t8_);           \
        y1_ = y1_ * k4096; uint32_t t9_; memcpy(&t9_, tab + i9_, 4); BK_PINVG(y1_, t9_);           \
        y2_ = y2_ * k4096; uint32_t t10_; memcpy(&t10_, tab + i10_, 4); BK_PINVG(y2_, t10_);       \
        y3_ = y3_ * k4096; uint32_t t11_; memcpy(&t11_, tab + i11_, 4); BK_PINVG(y3_, t11_);       \
        n0_ = n0_ + c127; memcpy(tv1_ + 8, &t8_, 4); BK_PINV(n0_);                                 \
        n1_ = n1_ + c127; memcpy(tv1_ + 9, &t9_, 4); BK_PINV(n1_);                                 \
        n2_ = n2_ + c127; memcpy(tv1_ + 10, &t10_, 4); BK_PINV(n2_);                               \
        n3_ = n3_ + c127; memcpy(tv1_ + 11, &t11_, 4); BK_PINV(n3_);                               \
        v4i q0_ = BK_EXP_RNE(y0_); int32_t i12_ = kk1_[12]; BK_PINVG(q0_, i12_);                   \
        v4i q1_ = BK_EXP_RNE(y1_); int32_t i13_ = kk1_[13]; BK_PINVG(q1_, i13_);                   \
        v4i q2_ = BK_EXP_RNE(y2_); int32_t i14_ = kk1_[14]; BK_PINVG(q2_, i14_);                   \
        v4i q3_ = BK_EXP_RNE(y3_); int32_t i15_ = kk1_[15]; BK_PINVG(q3_, i15_);                   \
        n0_ = n0_ << 23; uint32_t t12_; memcpy(&t12_, tab + i12_, 4); BK_PINVG(n0_, t12_);         \
        n1_ = n1_ << 23; uint32_t t13_; memcpy(&t13_, tab + i13_, 4); BK_PINVG(n1_, t13_);         \
        n2_ = n2_ << 23; uint32_t t14_; memcpy(&t14_, tab + i14_, 4); BK_PINVG(n2_, t14_);         \
        n3_ = n3_ << 23; uint32_t t15_; memcpy(&t15_, tab + i15_, 4); BK_PINVG(n3_, t15_);         \
        st4i(kk2_ + 0, q0_); memcpy(tv1_ + 12, &t12_, 4); BK_BARRIER;                              \
        st4i(kk2_ + 4, q1_); memcpy(tv1_ + 13, &t13_, 4); BK_BARRIER;                              \
        st4i(kk2_ + 8, q2_); memcpy(tv1_ + 14, &t14_, 4); BK_BARRIER;                              \
        st4i(kk2_ + 12, q3_); memcpy(tv1_ + 15, &t15_, 4); BK_BARRIER;                             \
        st4i(sb2_ + 0, n0_); BK_BARRIER;                                                           \
        st4i(sb2_ + 4, n1_); BK_BARRIER;                                                           \
        st4i(sb2_ + 8, n2_); BK_BARRIER;                                                           \
        st4i(sb2_ + 12, n3_); BK_BARRIER;                                                          \
        rn3 = one / den3; BK_PINV(rn3);                                                            \
    }
#define SILU_F4(o, HAS_S, HAS_A)                                                                   \
    {                                                                                              \
        v4f x0_ = ld4(xs + (o) + 0); BK_PINV(x0_);                                                 \
        v4f x1_ = ld4(xs + (o) + 4); BK_PINV(x1_);                                                 \
        v4f x2_ = ld4(xs + (o) + 8); BK_PINV(x2_);                                                 \
        v4f x3_ = ld4(xs + (o) + 12); BK_PINV(x3_);                                                \
        v4f u0_ = (HAS_A) ? ld4(us + (o) + 0) : one; BK_PINV(u0_);                                 \
        v4f u1_ = (HAS_A) ? ld4(us + (o) + 4) : one; BK_PINV(u1_);                                 \
        v4f u2_ = (HAS_A) ? ld4(us + (o) + 8) : one; BK_PINV(u2_);                                 \
        v4f u3_ = (HAS_A) ? ld4(us + (o) + 12) : one; BK_PINV(u3_);                                \
        x0_ = x0_ * r0; BK_PINV(x0_);                                                              \
        x1_ = x1_ * r1; BK_PINV(x1_);                                                              \
        x2_ = x2_ * r2; BK_PINV(x2_);                                                              \
        x3_ = x3_ * r3; BK_PINV(x3_);                                                              \
        if (HAS_S) st4(sw + (o) + 0, x0_);                                                         \
        if (HAS_S) st4(sw + (o) + 4, x1_);                                                         \
        if (HAS_S) st4(sw + (o) + 8, x2_);                                                         \
        if (HAS_S) st4(sw + (o) + 12, x3_);                                                        \
        BK_BARRIER;                                                                                \
        if (HAS_A) { x0_ = x0_ * u0_; BK_PINV(x0_); }                                              \
        if (HAS_A) { x1_ = x1_ * u1_; BK_PINV(x1_); }                                              \
        if (HAS_A) { x2_ = x2_ * u2_; BK_PINV(x2_); }                                              \
        if (HAS_A) { x3_ = x3_ * u3_; BK_PINV(x3_); }                                              \
        if (HAS_A) st4(aw + (o) + 0, x0_);                                                         \
        if (HAS_A) st4(aw + (o) + 4, x1_);                                                         \
        if (HAS_A) st4(aw + (o) + 8, x2_);                                                         \
        if (HAS_A) st4(aw + (o) + 12, x3_);                                                        \
    }
#define SILU_DEN4(o)                                                                               \
    {                                                                                              \
        v4f t0_ = ld4(tv + (o) + 0); BK_PINV(t0_);                                                 \
        v4f t1_ = ld4(tv + (o) + 4); BK_PINV(t1_);                                                 \
        v4f t2_ = ld4(tv + (o) + 8); BK_PINV(t2_);                                                 \
        v4f t3_ = ld4(tv + (o) + 12); BK_PINV(t3_);                                                \
        v4f s0_ = (v4f)ld4i(sb + (o) + 0); BK_PINV(s0_);                                           \
        v4f s1_ = (v4f)ld4i(sb + (o) + 4); BK_PINV(s1_);                                           \
        v4f s2_ = (v4f)ld4i(sb + (o) + 8); BK_PINV(s2_);                                           \
        v4f s3_ = (v4f)ld4i(sb + (o) + 12); BK_PINV(s3_);                                          \
        t0_ = t0_ * s0_; BK_PINV(t0_);                                                             \
        t1_ = t1_ * s1_; BK_PINV(t1_);                                                             \
        t2_ = t2_ * s2_; BK_PINV(t2_);                                                             \
        t3_ = t3_ * s3_; BK_PINV(t3_);                                                             \
        den0 = one + t0_; BK_PINV(den0);                                                           \
        den1 = one + t1_; BK_PINV(den1);                                                           \
        den2 = one + t2_; BK_PINV(den2);                                                           \
        den3 = one + t3_; BK_PINV(den3);                                                           \
    }
#define SILU_CORE(NAME, HAS_S, HAS_A)                                                              \
BK_KERN static void NAME(const float *restrict tab, const float *x, const float *up, float *sil,  \
                         float *act, int n)                                                        \
{                                                                                                  \
    int32_t kk[BK_SCH], sb[BK_SCH];                                                                \
    float tv[BK_SCH], xb[BK_SCH], ub[BK_SCH], so[BK_SCH], ao[BK_SCH];                              \
    const v4f one = v4_set1(1.0f), mv = v4_set1(-BK_LOG2E), k4096 = v4_set1(4096.0f);              \
    const v4i c127 = v4i_set1(127);                                                                \
    for (int i0 = 0; i0 < n; i0 += BK_SCH) {                                                       \
        const int m = n - i0 < BK_SCH ? n - i0 : BK_SCH, m16 = (m + P - 1) & ~(P - 1);             \
        const int tail = m16 != m, G = m16 / P;                                                    \
        const float *xs = x + i0, *us = (HAS_A) ? up + i0 : NULL;                                  \
        float *sd = (HAS_S) ? sil + i0 : NULL, *ad = (HAS_A) ? act + i0 : NULL;                    \
        if (tail) {                                                                                \
            memcpy(xb, xs, (size_t)m * 4);                                                         \
            memset(xb + m, 0, (size_t)(m16 - m) * 4);                                              \
            xs = xb;                                                                               \
            if (HAS_A) {                                                                           \
                memcpy(ub, us, (size_t)m * 4);                                                     \
                memset(ub + m, 0, (size_t)(m16 - m) * 4);                                          \
                us = ub;                                                                           \
            }                                                                                      \
        }                                                                                          \
        float *const sw = (HAS_S) ? (tail ? so : sd) : NULL;                                       \
        float *const aw = (HAS_A) ? (tail ? ao : ad) : NULL;                                       \
        v4f den0, den1, den2, den3, r0 = one, r1 = one, r2 = one, r3 = one;                        \
        /* (-x)*LOG2E == x*(-LOG2E) exactly */                                                     \
        SILU_P(0, 0) SILU_P(0, 1) SILU_P(0, 2) SILU_P(0, 3)                                        \
        SILU_G(0, 0) SILU_G(0, 1) SILU_G(0, 2) SILU_G(0, 3)                                        \
        if (G > 1) { SILU_P(P, 0) SILU_P(P, 1) SILU_P(P, 2) SILU_P(P, 3) }                         \
        SILU_DEN(0)                                                                                \
        int g = 0;                                                                                 \
        for (; g + 2 < G; g++) {                                                                   \
            const int o = g * P;                                                                   \
            v4f rn0 = one / den0, rn1, rn2, rn3; BK_PINV(rn0);                                     \
            if (g) SILU_F4(o - P, HAS_S, HAS_A)                                                    \
            SILU_PGD(xs + o + 2 * P, o + 2 * P, o + P)                                             \
            SILU_DEN4(o + P)                                                                       \
            r0 = rn0; r1 = rn1; r2 = rn2; r3 = rn3;                                                \
        }                                                                                          \
        if (g) SILU_F4(g * P - P, HAS_S, HAS_A)                                                    \
        for (; g < G; g++) {                                                                       \
            const int o = g * P;                                                                   \
            r0 = one / den0; r1 = one / den1; r2 = one / den2; r3 = one / den3;                    \
            if (g + 1 < G) {                                                                       \
                SILU_G(o + P, 0) SILU_G(o + P, 1) SILU_G(o + P, 2) SILU_G(o + P, 3)                \
                SILU_DEN(o + P)                                                                    \
            }                                                                                      \
            SILU_F4(o, HAS_S, HAS_A)                                                               \
        }                                                                                          \
        if (tail) {                                                                                \
            if (HAS_S) memcpy(sd, so, (size_t)m * 4);                                              \
            if (HAS_A) memcpy(ad, ao, (size_t)m * 4);                                              \
        }                                                                                          \
    }                                                                                              \
}
SILU_CORE(silu_fast_s, 1, 0)
SILU_CORE(silu_fast_a, 0, 1)
SILU_CORE(silu_fast_sa, 1, 1)

/* sil and / or act may be NULL (up is only read with act) */
static void silu_fast(const float *tab, const float *x, const float *up, float *sil, float *act, int n)
{
    if (sil && act) silu_fast_sa(tab, x, up, sil, act, n);
    else if (act) silu_fast_a(tab, x, up, sil, act, n);
    else if (sil) silu_fast_s(tab, x, up, sil, act, n);
}

/* bfp_quant: the absmax scan and the scale / round-half-even / narrow, 16 at a time. The
 * float64 path (group max below 2^-118) and groups that are not a multiple of 16 go to the
 * reference. No clamp: |RNE(x * 2^-e)| <= qmax by the exponent rule (see bfp_round_group). */
#if defined(BK_BUG_ROUND_AWAY)
#define BK_QUANT_SLOW 1
#else
#define BK_QUANT_SLOW 0
#endif
#define BK_QUANT_FAST_FN(NAME, REFNAME, QT, STORE)                                         \
    BK_KERN static int NAME(const float *x, int n, int group, int bits, QT *q, int8_t *e)          \
    {                                                                                      \
        if (group % P) return REFNAME(x, n, group, bits, q, e);                            \
        const float qmax = (float)((1 << (bits - 1)) - 1);                                 \
        const v4i absm = v4i_set1(0x7fffffff);                                             \
        float r[BK_QK];                                                                    \
        for (int g = 0; g < n / group; g++) {                                              \
            const float *xg = x + (size_t)g * group;                                       \
            QT *qg = q + (size_t)g * group;                                                \
            v4i m0 = v4i_set1(0), m1 = m0, m2 = m0, m3 = m0;                               \
            for (int i = 0; i < group; i += P) {                                           \
                v4i b0 = (v4i)ld4(xg + i) & absm, b1 = (v4i)ld4(xg + i + 4) & absm;        \
                v4i b2 = (v4i)ld4(xg + i + 8) & absm, b3 = (v4i)ld4(xg + i + 12) & absm;   \
                m0 = v4i_max(m0, b0); m1 = v4i_max(m1, b1);                                \
                m2 = v4i_max(m2, b2); m3 = v4i_max(m3, b3);                                \
            }                                                                              \
            uint32_t am = (uint32_t)v4i_hmax(v4i_max(v4i_max(m0, m1), v4i_max(m2, m3)));   \
            if (am >= 0x7f800000u) return BK_ERR_NONFINITE;                                \
            int ex = bfp_exp(am, bits);                                                    \
            if (ex < -128 || ex > 127) return BK_ERR_EXP_RANGE;                            \
            e[g] = (int8_t)ex;                                                             \
            if (BK_QUANT_SLOW || -ex > 127) {                                              \
                BK_ROUND_REF(QT, xg, qg, group, ex, qmax, r)                               \
            } else {                                                                       \
                const v4f sc = v4_set1(pow2f(-ex));                                        \
                for (int i = 0; i < group; i += P) {                                       \
                    v4f t0 = ld4(xg + i) * sc, t1 = ld4(xg + i + 4) * sc;                  \
                    v4f t2 = ld4(xg + i + 8) * sc, t3 = ld4(xg + i + 12) * sc;             \
                    STORE(qg + i, v4_rne(t0), v4_rne(t1), v4_rne(t2), v4_rne(t3));         \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
        return BK_OK;                                                                      \
    }

BK_QUANT_FAST_FN(quant_fast16, quant_ref16, int16_t, st_q16)
BK_QUANT_FAST_FN(quant_fast8, quant_ref8, int8_t, st_q8)

#else                                   /* no SIMD tier: the fast names are the reference */
#define fsum_fast fsum_ref
#define rmsnorm_fast rmsnorm_ref
#define rope_fast rope_ref
#define exp_fast exp_ref
#define quant_fast16 quant_ref16
#define quant_fast8 quant_ref8
static void add_fast(float *x, const float *y, int n)
{
    for (int i = 0; i < n; i++) x[i] = x[i] + y[i];
}
#endif

/* silu(gate) (-> sil, optional, may alias gate) and silu(gate) * up (-> act, optional) */
static void silu_mul_ref(const float *tab, const float *gate, const float *up, float *sil,
                         float *act, int n)
{
    float so[BK_CH];
    for (int i0 = 0; i0 < n; i0 += BK_CH) {
        int m = n - i0 < BK_CH ? n - i0 : BK_CH;
        silu_ref(tab, gate + i0, so, m);
        if (act) for (int i = 0; i < m; i++) act[i0 + i] = so[i] * up[i0 + i];
        if (sil) memcpy(sil + i0, so, (size_t)m * 4);
    }
}
#if !BK_SIMD
#define silu_fast silu_mul_ref
#endif

/* ------------------------------------------------------------------ exported primitives */
int bk_simd(void) { return BK_SIMD; }
float bk_fsum(const float *x, int n) { return fsum_fast(x, n); }
float bk_ref_fsum(const float *x, int n) { return fsum_ref(x, n); }
float bk_exp_fab(const float *exp_tab, float x)
{
    float e;
    exp_block_ref(exp_tab, &x, &e, 1);
    return e;
}
void bk_exp_fab_n(const float *exp_tab, const float *x, float *y, int n) { exp_fast(exp_tab, x, y, n); }
void bk_ref_exp_fab_n(const float *exp_tab, const float *x, float *y, int n) { exp_ref(exp_tab, x, y, n); }
void bk_silu(const float *exp_tab, const float *x, float *y, int n) { silu_fast(exp_tab, x, NULL, y, NULL, n); }
void bk_ref_silu(const float *exp_tab, const float *x, float *y, int n) { silu_ref(exp_tab, x, y, n); }
void bk_silu_mul(int kernels, const float *exp_tab, const float *gate, const float *up,
                 float *sil, float *act, int n)
{
    if (kernels == BK_KERN_REF) silu_mul_ref(exp_tab, gate, up, sil, act, n);
    else silu_fast(exp_tab, gate, up, sil, act, n);
}
void bk_rmsnorm(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                const float *w, int n, float *y)
{
    rmsnorm_fast(rs_lo, rs_hi, eps, x, w, n, y);
}
void bk_ref_rmsnorm(const float *rs_lo, const float *rs_hi, float eps, const float *x,
                    const float *w, int n, float *y)
{
    rmsnorm_ref(rs_lo, rs_hi, eps, x, w, n, y);
}
void bk_rope(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y)
{
    rope_fast(cos_row, sin_row, x, head_dim, y);
}
void bk_ref_rope(const float *cos_row, const float *sin_row, const float *x, int head_dim, float *y)
{
    rope_ref(cos_row, sin_row, x, head_dim, y);
}
int bk_bfp_quant16(const float *x, int n, int group, int bits, int16_t *q, int8_t *e)
{
    return quant_fast16(x, n, group, bits, q, e);
}
int bk_bfp_quant8(const float *x, int n, int group, int bits, int8_t *q, int8_t *e)
{
    return quant_fast8(x, n, group, bits, q, e);
}
int bk_ref_bfp_quant16(const float *x, int n, int group, int bits, int16_t *q, int8_t *e)
{
    return quant_ref16(x, n, group, bits, q, e);
}
int bk_ref_bfp_quant8(const float *x, int n, int group, int bits, int8_t *q, int8_t *e)
{
    return quant_ref8(x, n, group, bits, q, e);
}

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

/* ================================================================== thread pool */
/* A persistent pool for "run fn(arg, unit, tid) for unit = 0..n-1" jobs. The caller is
 * thread 0 and works too. One atomic word holds the job: generation, unit count and the
 * next unclaimed unit; a unit is claimed by a compare-and-swap on that word, so a worker
 * that wakes up late (after the job, or several jobs, finished) can never run a unit of a
 * job it did not see whole. The job is over when `done` reaches the unit count; the caller
 * never waits for a worker that is asleep, only for units that are being worked on. */
typedef void (*bk_job_fn)(void *arg, int unit, int tid);

typedef struct bk_pool bk_pool;
typedef struct { bk_pool *p; int tid; } bk_worker_arg;

struct bk_pool {
    int n_workers;                      /* threads besides the caller */
    pthread_t th[BK_MAX_THREADS];
    bk_worker_arg wa[BK_MAX_THREADS];
    bk_job_fn fn;
    void *arg;
    _Atomic uint64_t ctl;               /* generation:24 | units:20 | next:20 */
    _Atomic int done;                   /* units finished in the current job */
    _Atomic int sleepers;
    _Atomic int stop;
    _Atomic uint64_t worker_units;      /* units run by the workers (not the caller) since pool_new */
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int64_t spin_ns;
};

#define CTL_UNITS(w) ((int)(((w) >> 20) & 0xfffffu))
#define CTL_NEXT(w)  ((int)((w) & 0xfffffu))
#define BK_MAX_UNITS 0xfffff

static inline void cpu_relax(void)
{
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static void *pool_worker(void *a)
{
    bk_worker_arg *wa = a;
    bk_pool *p = wa->p;
    const int tid = wa->tid;
    uint64_t w = atomic_load(&p->ctl);
    for (;;) {
        if (CTL_NEXT(w) < CTL_UNITS(w)) {
            if (atomic_compare_exchange_weak(&p->ctl, &w, w + 1)) {
                /* the claim succeeded on a job that is still open, so fn / arg are its own */
                p->fn(p->arg, CTL_NEXT(w), tid);
                atomic_fetch_add(&p->worker_units, 1);
                atomic_fetch_add(&p->done, 1);
                w = atomic_load(&p->ctl);
            }
            continue;
        }
        if (atomic_load(&p->stop)) break;
        /* nothing to claim: spin until the word changes, then sleep */
        const uint64_t seen = w;
        uint64_t t0 = 0;
        unsigned k = 0;
        for (;;) {
            w = atomic_load(&p->ctl);
            if (w != seen || atomic_load(&p->stop)) break;
            int go_sleep = p->spin_ns <= 0;
            if (!go_sleep) {
                cpu_relax();
                if ((++k & 63u) == 0) {
                    uint64_t t = now_ns();
                    if (!t0) t0 = t;
                    else if ((int64_t)(t - t0) >= p->spin_ns) go_sleep = 1;
                }
            }
            if (go_sleep) {
                pthread_mutex_lock(&p->mu);
                atomic_fetch_add(&p->sleepers, 1);
                while (atomic_load(&p->ctl) == seen && !atomic_load(&p->stop))
                    pthread_cond_wait(&p->cv, &p->mu);
                atomic_fetch_sub(&p->sleepers, 1);
                pthread_mutex_unlock(&p->mu);
            }
        }
    }
    return NULL;
}

static void pool_run(bk_pool *p, bk_job_fn fn, void *arg, int n)
{
    p->fn = fn;
    p->arg = arg;
    atomic_store(&p->done, 0);
    uint64_t w = atomic_load(&p->ctl);
    w = ((((w >> 40) + 1) & 0xffffffu) << 40) | ((uint64_t)n << 20);
    atomic_store(&p->ctl, w);           /* publishes fn / arg / done */
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    while (CTL_NEXT(w) < n) {
        if (atomic_compare_exchange_weak(&p->ctl, &w, w + 1)) {
            fn(arg, CTL_NEXT(w), 0);
            atomic_fetch_add(&p->done, 1);
            w = atomic_load(&p->ctl);
        }
    }
    unsigned k = 0;
    while (atomic_load(&p->done) < n) {
        cpu_relax();
        if ((++k & 1023u) == 0) sched_yield();
    }
}

static void pool_free(bk_pool *p)
{
    if (!p) return;
    atomic_store(&p->stop, 1);
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_workers; i++) pthread_join(p->th[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p);
}

/* n_threads >= 2. Returns NULL if no worker could be started (the caller then runs alone). */
static bk_pool *pool_new(int n_threads, int64_t spin_ns)
{
    bk_pool *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    p->spin_ns = spin_ns;
    for (int i = 0; i < n_threads - 1; i++) {
        p->wa[i].p = p;
        p->wa[i].tid = p->n_workers + 1;
        if (pthread_create(&p->th[p->n_workers], NULL, pool_worker, &p->wa[i]) != 0) break;
        p->n_workers++;
    }
    if (p->n_workers == 0) {
        pthread_mutex_destroy(&p->mu);
        pthread_cond_destroy(&p->cv);
        free(p);
        return NULL;
    }
    return p;
}

/* ================================================================== state */
typedef struct {
    int16_t *kq;        /* [cap][hd]      K mantissas (BFP10) */
    int8_t  *ke;        /* [cap][hd/16]   K exponents */
    int8_t  *vq;        /* [cap][hd]      V mantissas (BFP8) */
    int8_t  *ve;        /* [cap][hd/128]  V exponents */
} kv_head;

/* per-thread scratch of the attention */
typedef struct {
    float *sc;          /* [rep][cap]  scores -> exp -> p */
    float *ps;          /* [cap][16]  p and the V scales of a row, rows in lane order (ctx_rows) */
    float *la;          /* [rep][P][hd]  lane accumulators */
    float *kf;          /* [hd]  one dequantised cache row (reference kernels) */
    float *tq;          /* [rep + 1][hd]  q / k after their norms */
    uint64_t ns[BK_T_COUNT];
    char pad[64];
} bk_tls;

#define BK_MAX_ALLOCS 128
#define BK_FFN_CHUNK 512                /* elements per FFN unit (a multiple of BK_QK) */

struct bk_state {
    bk_config c;
    bk_weights w;
    int rep, hd, hd2, nq, nkv;          /* nq = n_head*hd, nkv = n_kv*hd */
    int pos, cap;
    int n_mats;
    int fast;                           /* fast kernels in use */
    int n_threads;
    bk_pool *pool;                      /* NULL with one thread */
    bk_tls *tls;                        /* [n_threads] */
    int *unit_rc;                       /* [max units of a region] */
    kv_head *kv;                        /* [n_layer*n_kv] */
    /* per-token work vectors */
    float *x, *h, *yq, *yk, *yv, *qr, *kr, *ctx, *yo, *gate, *up, *act, *yd;
    /* GEMV inputs: one buffer per role, so pointer identity <=> same data (see bk_host.h) */
    int16_t *xq_attn, *xq_ctx, *xq_ffn, *xq_act, *xq_out;
    int8_t *xe_attn, *xe_ctx, *xe_ffn, *xe_act, *xe_out;
    int64_t *off_xq, *off_xe, *off_y;   /* [n_mats + 1] tap offsets */
    void *allocs[BK_MAX_ALLOCS];
    int n_allocs;
    bk_stats st;
    uint64_t wu0;                       /* pool->worker_units at the last bk_reset_stats */
    uint64_t tlast;
    int gemv_rc;
    /* the layer being computed, for the region units */
    int cur_L, cur_pos;
    const bk_taps *cur_tp;
};

/* time since the last lap goes to bucket `cat` (serial sections: wall == thread time) */
static inline void lap(bk_state *s, int cat)
{
    uint64_t t = now_ns();
    s->st.ns[cat] += t - s->tlast;
    s->st.cpu_ns[cat] += t - s->tlast;
    s->tlast = t;
}

/* the same inside a region unit, into the thread's own counters */
#define TLAP(T, t0, cat)                \
    do {                                \
        uint64_t t1_ = now_ns();        \
        (T)->ns[cat] += t1_ - (t0);     \
        (t0) = t1_;                     \
    } while (0)

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
    for (int t = 0; t < s->n_threads; t++) {
        bk_tls *T = &s->tls[t];
        void *p;
        if (!(p = realloc(T->sc, (size_t)s->rep * (size_t)nc * sizeof(float)))) return BK_ERR_NOMEM;
        T->sc = p;
        if (!(p = realloc(T->ps, (size_t)nc * 16 * sizeof(float)))) return BK_ERR_NOMEM;
        T->ps = p;
    }
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
    pool_free(s->pool);
    if (s->kv)
        for (int i = 0; i < s->c.n_layer * s->c.n_kv; i++) {
            free(s->kv[i].kq);
            free(s->kv[i].ke);
            free(s->kv[i].vq);
            free(s->kv[i].ve);
        }
    if (s->tls)
        for (int t = 0; t < s->n_threads; t++) {
            free(s->tls[t].sc);
            free(s->tls[t].ps);
        }
    for (int i = 0; i < s->n_allocs; i++) free(s->allocs[i]);
    free(s);
}

static int env_int(const char *name, int dflt)
{
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    return atoi(v);
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
        c->init_ctx < 0 || c->n_threads < 0 || c->n_threads > BK_MAX_THREADS ||
        c->kernels < BK_KERN_AUTO || c->kernels > BK_KERN_REF)
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

    /* kernels */
    int kern = c->kernels;
    if (kern == BK_KERN_AUTO) {
        const char *v = getenv("BK_KERNELS");
        kern = (v && (v[0] == 'r' || v[0] == 'R' || v[0] == '2')) ? BK_KERN_REF : BK_KERN_FAST;
    }
    s->fast = (kern == BK_KERN_FAST) && BK_SIMD != 0;
    /* threads */
    int nt = c->n_threads;
    if (nt <= 0) nt = env_int("BK_THREADS", 0);
    if (nt <= 0) {
        long on = sysconf(_SC_NPROCESSORS_ONLN);
        nt = on < 1 ? 1 : (on > 4 ? 4 : (int)on);
    }
    if (nt > BK_MAX_THREADS) nt = BK_MAX_THREADS;
    int spin = c->spin_us;
    if (spin == 0) spin = env_int("BK_SPIN_US", 200);
    if (spin < 0) spin = 0;
    if (nt > 1) {
        s->pool = pool_new(nt, (int64_t)spin * 1000);
        nt = s->pool ? s->pool->n_workers + 1 : 1;
    }
    s->n_threads = nt;

    int fail = 0;
    size_t d = (size_t)c->d, ffn = (size_t)c->ffn, nq = (size_t)s->nq, nkv = (size_t)s->nkv;
    int n_units = (c->ffn + BK_FFN_CHUNK - 1) / BK_FFN_CHUNK;
    if (n_units < c->n_kv) n_units = c->n_kv;
    s->kv = xalloc(s, (size_t)c->n_layer * c->n_kv, sizeof(kv_head), &fail);
    s->tls = xalloc(s, (size_t)nt, sizeof(bk_tls), &fail);
    s->unit_rc = xalloc(s, (size_t)n_units, sizeof(int), &fail);
    s->x = xalloc(s, d, sizeof(float), &fail);
    s->h = xalloc(s, d, sizeof(float), &fail);
    s->yq = xalloc(s, nq, sizeof(float), &fail);
    s->yk = xalloc(s, nkv, sizeof(float), &fail);
    s->yv = xalloc(s, nkv, sizeof(float), &fail);
    s->qr = xalloc(s, nq, sizeof(float), &fail);
    s->kr = xalloc(s, nkv, sizeof(float), &fail);
    s->ctx = xalloc(s, nq, sizeof(float), &fail);
    s->yo = xalloc(s, d, sizeof(float), &fail);
    s->gate = xalloc(s, ffn, sizeof(float), &fail);
    s->up = xalloc(s, ffn, sizeof(float), &fail);
    s->act = xalloc(s, ffn, sizeof(float), &fail);
    s->yd = xalloc(s, d, sizeof(float), &fail);
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
    for (int t = 0; t < nt && !fail; t++) {
        bk_tls *T = &s->tls[t];
        T->la = xalloc(s, (size_t)s->rep * P * (size_t)s->hd, sizeof(float), &fail);
        T->kf = xalloc(s, (size_t)s->hd, sizeof(float), &fail);
        T->tq = xalloc(s, (size_t)(s->rep + 1) * (size_t)s->hd, sizeof(float), &fail);
    }
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
int bk_threads(const bk_state *s) { return s->n_threads; }
int bk_kernels(const bk_state *s) { return s->fast ? BK_KERN_FAST : BK_KERN_REF; }

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

void bk_get_stats(const bk_state *s, bk_stats *out)
{
    *out = s->st;
    out->worker_units = (s->pool ? atomic_load(&s->pool->worker_units) : 0) - s->wu0;
    out->threads = s->n_threads;
    out->kernels = s->fast ? BK_KERN_FAST : BK_KERN_REF;
}
void bk_reset_stats(bk_state *s)
{
    memset(&s->st, 0, sizeof(s->st));
    s->wu0 = s->pool ? atomic_load(&s->pool->worker_units) : 0;
}

/* ================================================================== attention */
/* The attention of one kv head: `rep` query heads q [rep][hd] against Tk cached rows. */
typedef struct {
    int hd, rep, cap;                   /* cap: row stride of the score scratch */
    float isq;
    const float *tab;
    const int16_t *kq;
    const int8_t *ke, *vq, *ve;
} att_in;

/* ---- reference: the first version's loops ---- */
static void score_ref(const att_in *a, int Tk, const float *q, bk_tls *T)
{
    const int hd = a->hd;
    for (int t = 0; t < Tk; t++) {
        bk_bfp_dequant16(a->kq + (size_t)t * hd, a->ke + (size_t)t * (hd / BK_KGRP), hd, BK_KGRP, T->kf);
        for (int r = 0; r < a->rep; r++)
            T->sc[(size_t)r * a->cap + t] = dot_lanes(q + (size_t)r * hd, T->kf, hd) * a->isq;
    }
}

/* scores -> probabilities in place: m = max; e = exp_fab(s - m); Z = fsum(e); p = e * (1/Z) */
static void softmax_ref(const float *tab, float *restrict sc, int Tk)
{
    float m = sc[0];
    for (int t = 1; t < Tk; t++) m = sc[t] > m ? sc[t] : m;
    for (int t = 0; t < Tk; t++) sc[t] = sc[t] - m;
    exp_ref(tab, sc, sc, Tk);
    float Z = fsum_ref(sc, Tk);
    float rz = 1.0f / Z;
    for (int t = 0; t < Tk; t++) sc[t] = sc[t] * rz;
}

/* ctx_d = fsum over t (lane = t % 16, sequential per lane) of p_t * v_t,d */
static void ctx_ref(const att_in *a, int Tk, bk_tls *T, float *ctx)
{
    const int hd = a->hd, rep = a->rep;
    float *restrict cacc = T->la, *restrict kf = T->kf;
    memset(cacc, 0, (size_t)rep * P * hd * sizeof(float));
    for (int t = 0; t < Tk; t++) {
        bk_bfp_dequant8(a->vq + (size_t)t * hd, a->ve + (size_t)t * (hd / BK_QK), hd, BK_QK, kf);
#if defined(BK_BUG_CTX_LANE)            /* planted bug: every position in lane 0 */
        int lane = 0;
#else
        int lane = t % P;
#endif
        for (int r = 0; r < rep; r++) {
            float pt = T->sc[(size_t)r * a->cap + t];
            float *restrict acc = cacc + ((size_t)r * P + lane) * hd;
            for (int dd = 0; dd < hd; dd++) acc[dd] = acc[dd] + pt * kf[dd];
        }
    }
    for (int r = 0; r < rep; r++) {
        float *restrict acc = cacc + (size_t)r * P * hd;
#if defined(BK_BUG_TREE)
        for (int w = P; w > 1; w /= 2)
            for (int i = 0; i < w / 2; i++)
                for (int dd = 0; dd < hd; dd++)
                    acc[(size_t)i * hd + dd] = acc[(size_t)(2 * i) * hd + dd] + acc[(size_t)(2 * i + 1) * hd + dd];
#else
        for (int w = P / 2; w >= 1; w /= 2)
            for (int i = 0; i < w; i++) {
                float *restrict lo = acc + (size_t)i * hd;
                const float *restrict hi = acc + (size_t)(i + w) * hd;
                for (int dd = 0; dd < hd; dd++) lo[dd] = lo[dd] + hi[dd];
            }
#endif
        memcpy(ctx + (size_t)r * hd, acc, (size_t)hd * sizeof(float));
    }
}

#if BK_SIMD
#define BK_REP_FAST 8                   /* most query heads per kv head the fast kernels take */
#define BK_NV_FAST 8                    /* most V exponents per row (head_dim <= 1024) */
#define BK_PS 16                        /* floats per row of the p / V-scale scratch: p[0..7], scale[8..15] */
#if defined(BK_BUG_FMA)
#define BK_SCORE_SLOW 1                 /* the planted FMA lives in dot_lanes */
#else
#define BK_SCORE_SLOW 0
#endif
#if defined(BK_BUG_FAST_TINY)
#define BK_SCORE_TINY(er, ng) 0
#else
#define BK_SCORE_TINY(er, ng) any_tiny_exp(er, ng)
#endif
#define BK_HEADS_1(M) M(0)
#define BK_HEADS_2(M) M(0) M(1)
#define BK_HEADS_4(M) M(0) M(1) BK_BARRIER; M(2) M(3) BK_BARRIER;
#define BK_ACC_DECL(r) v4f a##r##0 = zero, a##r##1 = zero, a##r##2 = zero, a##r##3 = zero;

/* Scores of all rows. One K row at a time; its exponent groups are 16 channels = the 16
 * lanes of the sum, so one group is: dequantise 4 vectors in registers, then for each head
 * 4 loads of q, 4 products and 4 lane additions. The rep*4 lane accumulators are plain
 * locals (registers). Also starts pulling the V rows towards the cache for the context pass.
 * (Measured on the A53: a 128-bit load costs two cycles and every 128-bit NEON operation
 * one, so this is 80 cycles per 64 products, 40 % of it the loads of q.) */
#define BK_SC_MAC(r)                                                                               \
    {                                                                                              \
        const float *qc = q + (size_t)(r) * hd + c * P;                                            \
        v4f p0 = ld4(qc) * k0, p1 = ld4(qc + 4) * k1, p2 = ld4(qc + 8) * k2, p3 = ld4(qc + 12) * k3; \
        a##r##0 = a##r##0 + p0; a##r##1 = a##r##1 + p1;                                            \
        a##r##2 = a##r##2 + p2; a##r##3 = a##r##3 + p3;                                            \
    }
#define BK_SC_OUT(r) sc[(size_t)(r) * cap + t] = tree_v(a##r##0, a##r##1, a##r##2, a##r##3) * isq;
#define BK_SCORE_FN(NAME, HEADS, REP)                                                              \
    BK_KERN static void NAME(const int hd, const int Tk, const float isq, const float *restrict q,        \
                     const int16_t *restrict kq, const int8_t *restrict ke, const int8_t *vq,     \
                     float *restrict sc, const int cap, float *restrict kf)                       \
    {                                                                                              \
        const int ng = hd / BK_KGRP;                                                               \
        const v4f zero = v4_set1(0.0f);                                                            \
        for (int t = 0; t < Tk; t++) {                                                             \
            const int16_t *kr = kq + (size_t)t * hd;                                               \
            const int8_t *er = ke + (size_t)t * ng;                                                \
            __builtin_prefetch(vq + (size_t)(t + 8) * hd);                                         \
            __builtin_prefetch(vq + (size_t)(t + 8) * hd + 64);                                    \
            if (BK_SCORE_SLOW || BK_SCORE_TINY(er, ng)) {                                          \
                bk_bfp_dequant16(kr, er, hd, BK_KGRP, kf);                                         \
                for (int r = 0; r < (REP); r++)                                                    \
                    sc[(size_t)r * cap + t] = dot_lanes(q + (size_t)r * hd, kf, hd) * isq;         \
                continue;                                                                          \
            }                                                                                      \
            HEADS(BK_ACC_DECL)                                                                     \
            for (int c = 0; c < ng; c++) {                                                         \
                const v4f sv = v4_set1(pow2f(er[c]));                                              \
                v4f kk[4];                                                                         \
                ld_i16_f(kr + c * P, kk);                                                          \
                ld_i16_f(kr + c * P + 8, kk + 2);                                                  \
                const v4f k0 = kk[0] * sv, k1 = kk[1] * sv, k2 = kk[2] * sv, k3 = kk[3] * sv;      \
                HEADS(BK_SC_MAC)                                                                   \
            }                                                                                      \
            HEADS(BK_SC_OUT)                                                                       \
        }                                                                                          \
    }
BK_SCORE_FN(score_fast_1, BK_HEADS_1, 1)
BK_SCORE_FN(score_fast_2, BK_HEADS_2, 2)
BK_SCORE_FN(score_fast_4, BK_HEADS_4, 4)

/* any other rep <= BK_REP_FAST: the same with the accumulators in an array */
BK_KERN static void score_fast_n(const int rep, const int hd, const int Tk, const float isq,
                         const float *restrict q, const int16_t *restrict kq,
                         const int8_t *restrict ke, const int8_t *vq, float *restrict sc,
                         const int cap, float *restrict kf)
{
    const int ng = hd / BK_KGRP;
    (void)vq;
    for (int t = 0; t < Tk; t++) {
        const int16_t *kr = kq + (size_t)t * hd;
        const int8_t *er = ke + (size_t)t * ng;
        if (BK_SCORE_SLOW || any_tiny_exp(er, ng)) {
            bk_bfp_dequant16(kr, er, hd, BK_KGRP, kf);
            for (int r = 0; r < rep; r++)
                sc[(size_t)r * cap + t] = dot_lanes(q + (size_t)r * hd, kf, hd) * isq;
            continue;
        }
        v4f acc[BK_REP_FAST][4];
        for (int r = 0; r < rep; r++)
            for (int j = 0; j < 4; j++) acc[r][j] = v4_set1(0.0f);
        for (int c = 0; c < ng; c++) {
            const v4f sv = v4_set1(pow2f(er[c]));
            v4f k[4];
            ld_i16_f(kr + c * P, k);
            ld_i16_f(kr + c * P + 8, k + 2);
            k[0] = k[0] * sv; k[1] = k[1] * sv; k[2] = k[2] * sv; k[3] = k[3] * sv;
            for (int r = 0; r < rep; r++) {
                const float *qc = q + (size_t)r * hd + c * P;
                v4f p0 = ld4(qc) * k[0], p1 = ld4(qc + 4) * k[1], p2 = ld4(qc + 8) * k[2], p3 = ld4(qc + 12) * k[3];
                acc[r][0] = acc[r][0] + p0; acc[r][1] = acc[r][1] + p1;
                acc[r][2] = acc[r][2] + p2; acc[r][3] = acc[r][3] + p3;
            }
        }
        for (int r = 0; r < rep; r++)
            sc[(size_t)r * cap + t] = tree_v(acc[r][0], acc[r][1], acc[r][2], acc[r][3]) * isq;
    }
}

BK_KERN static void softmax_fast(const float *tab, float *sc, int Tk)
{
    int t = 0;
    float m = sc[0];
    if (Tk >= P) {
        v4f m0 = ld4(sc), m1 = ld4(sc + 4), m2 = ld4(sc + 8), m3 = ld4(sc + 12);
        for (t = P; t + P <= Tk; t += P) {
            v4f x0 = ld4(sc + t), x1 = ld4(sc + t + 4), x2 = ld4(sc + t + 8), x3 = ld4(sc + t + 12);
            v4i g0 = x0 > m0, g1 = x1 > m1, g2 = x2 > m2, g3 = x3 > m3;
            m0 = (v4f)(((v4i)x0 & g0) | ((v4i)m0 & ~g0)); m1 = (v4f)(((v4i)x1 & g1) | ((v4i)m1 & ~g1));
            m2 = (v4f)(((v4i)x2 & g2) | ((v4i)m2 & ~g2)); m3 = (v4f)(((v4i)x3 & g3) | ((v4i)m3 & ~g3));
        }
        float mm[P];
        st4(mm, m0); st4(mm + 4, m1); st4(mm + 8, m2); st4(mm + 12, m3);
        m = mm[0];
        for (int i = 1; i < P; i++) m = mm[i] > m ? mm[i] : m;
    }
    for (; t < Tk; t++) m = sc[t] > m ? sc[t] : m;
    /* (which of two equal zeros the max is does not matter: s - m is then a zero of either
     * sign, and exp_fab of both is exactly 1) */
    const v4f mv = v4_set1(m);
    for (t = 0; t + P <= Tk; t += P) {
        v4f x0 = ld4(sc + t) - mv, x1 = ld4(sc + t + 4) - mv, x2 = ld4(sc + t + 8) - mv, x3 = ld4(sc + t + 12) - mv;
        st4(sc + t, x0); st4(sc + t + 4, x1); st4(sc + t + 8, x2); st4(sc + t + 12, x3);
    }
    for (; t < Tk; t++) sc[t] = sc[t] - m;
    exp_fast(tab, sc, sc, Tk);
    float Z = fsum_fast(sc, Tk);
    float rz = 1.0f / Z;
    const v4f rv = v4_set1(rz);
    for (t = 0; t + P <= Tk; t += P) {
        v4f x0 = ld4(sc + t) * rv, x1 = ld4(sc + t + 4) * rv, x2 = ld4(sc + t + 8) * rv, x3 = ld4(sc + t + 12) * rv;
        st4(sc + t, x0); st4(sc + t + 4, x1); st4(sc + t + 8, x2); st4(sc + t + 12, x3);
    }
    for (; t < Tk; t++) sc[t] = sc[t] * rz;
}

/* lanes in use and the row stride of a lane (the planted lane bug: one lane holds every row) */
#if defined(BK_BUG_CTX_LANE)
#define BK_CTX_NL(Tk) 1
#define BK_CTX_STRIDE 1
#else
#define BK_CTX_NL(Tk) ((Tk) < P ? (Tk) : P)
#define BK_CTX_STRIDE P
#endif

/* The context scratch: one row of BK_PS floats per cached position, rows in lane order
 * (lane 0's rows, then lane 1's, ...): p of the rep heads at [0..], the V scales 2^ve at
 * [8..]. fold[lane] = 1 when, for every row of the lane (and one V exponent per row), the
 * scale could be multiplied into p exactly; the row then holds p * 2^ve and the kernel skips
 * the scale: p * (v * 2^ve) and (p * 2^ve) * v are the same real number rounded once, as
 * long as p * 2^ve is exact, i.e. p is zero or the product is a normal number. */
static void ctx_rows(int rep, int nv, int Tk, const float *sc, int cap, const int8_t *ve,
                     float *restrict ps, uint8_t *fold)
{
    const int nl = BK_CTX_NL(Tk);
    float *row = ps;
    for (int l = 0; l < nl; l++) {
        float *row0 = row;
        int ok = (nv == 1);
        for (int t = l; t < Tk; t += BK_CTX_STRIDE, row += BK_PS) {
            for (int r = 0; r < rep; r++) row[r] = sc[(size_t)r * cap + t];
            for (int r = rep; r < 8; r++) row[r] = 0.0f;
            for (int k = 0; k < nv; k++) row[8 + k] = pow2f(ve[(size_t)t * nv + k]);
            if (ok) {
                const float sv = row[8];
                for (int r = 0; r < rep; r++) {
                    float pe = row[r] * sv;
                    /* exact: p is zero, or the product is a normal number. (A product that
                     * underflows to zero from a nonzero p is NOT exact.) */
#if defined(BK_BUG_FAST_FOLD)
                    ok &= (pe == 0.0f) | ((f32_bits(pe) & 0x7fffffffu) >= 0x00800000u);
#else
                    ok &= (row[r] == 0.0f) | ((f32_bits(pe) & 0x7fffffffu) >= 0x00800000u);
#endif
                }
            }
        }
        fold[l] = (uint8_t)ok;
        if (ok)
            for (float *rw = row0; rw < row; rw += BK_PS)
                for (int r = 0; r < rep; r++) rw[r] = rw[r] * rw[8];
    }
}

/* The lane accumulators la[r][lane][hd]. For a lane and a group of 16 channels: walk the
 * lane's rows in order; per row dequantise 16 V values in registers and add p * v into the
 * rep*4 register accumulators. The lane's rows (Tk/16 * hd bytes) stay in L1 over the hd/16
 * passes. FOLD: the rows hold p * 2^ve, no scale multiply. */
#define BK_CX_MAC(r)                                                                               \
    {                                                                                              \
        v4f p0 = v4_mul_lane(v0, pv, r), p1 = v4_mul_lane(v1, pv, r);                              \
        v4f p2 = v4_mul_lane(v2, pv, r), p3 = v4_mul_lane(v3, pv, r);                              \
        a##r##0 = a##r##0 + p0; a##r##1 = a##r##1 + p1;                                            \
        a##r##2 = a##r##2 + p2; a##r##3 = a##r##3 + p3;                                            \
    }
#define BK_CX_OUT(r)                                                                               \
    {                                                                                              \
        float *o = la + ((size_t)(r) * P + l) * hd + c * P;                                        \
        st4(o, a##r##0); st4(o + 4, a##r##1); st4(o + 8, a##r##2); st4(o + 12, a##r##3);           \
    }
#define BK_CX_LANE(HEADS, FOLD)                                                                    \
    for (int c = 0; c < hd / P; c++) {                                                             \
        const int vi = 8 + c / (BK_QK / P);                                                        \
        HEADS(BK_ACC_DECL)                                                                         \
        const int8_t *vr = vq + (size_t)l * hd + c * P;                                            \
        const float *pr = pl;                                                                      \
        for (int i = 0; i < nr; i++, vr += (size_t)stride * hd, pr += BK_PS) {                     \
            if (c == 0) {                                                                          \
                __builtin_prefetch(vr + (size_t)(2 * stride) * hd);                                \
                __builtin_prefetch(vr + (size_t)(2 * stride) * hd + 64);                           \
            }                                                                                      \
            v4f vv[4];                                                                             \
            ld_i8_f(vr, vv);                                                                       \
            const v4f pv = ld4(pr);                                                                \
            v4f v0 = vv[0], v1 = vv[1], v2 = vv[2], v3 = vv[3];                                    \
            if (!(FOLD)) {                                                                         \
                const v4f sv = v4_set1(pr[vi]);                                                    \
                v0 = v0 * sv; v1 = v1 * sv; v2 = v2 * sv; v3 = v3 * sv;                            \
            }                                                                                      \
            HEADS(BK_CX_MAC)                                                                       \
        }                                                                                          \
        HEADS(BK_CX_OUT)                                                                           \
    }
#define BK_CTX_FN(NAME, HEADS)                                                                     \
    BK_KERN static void NAME(const int hd, const int Tk, const float *restrict ps, const uint8_t *fold,   \
                     const int8_t *restrict vq, float *restrict la)                               \
    {                                                                                              \
        const int nl = BK_CTX_NL(Tk), stride = BK_CTX_STRIDE;                                      \
        const v4f zero = v4_set1(0.0f);                                                            \
        const float *pl = ps;                                                                      \
        for (int l = 0; l < nl; l++) {                                                             \
            const int nr = (Tk - l + stride - 1) / stride;                                         \
            if (fold[l]) {                                                                         \
                BK_CX_LANE(HEADS, 1)                                                               \
            } else {                                                                               \
                BK_CX_LANE(HEADS, 0)                                                               \
            }                                                                                      \
            pl += (size_t)nr * BK_PS;                                                              \
        }                                                                                          \
    }
BK_CTX_FN(ctx_fast_1, BK_HEADS_1)
BK_CTX_FN(ctx_fast_2, BK_HEADS_2)
BK_CTX_FN(ctx_fast_4, BK_HEADS_4)

/* any other rep <= BK_REP_FAST: accumulators in an array, scalar p loads */
BK_KERN static void ctx_fast_n(const int rep, const int hd, const int Tk, const float *restrict ps,
                       const uint8_t *fold, const int8_t *restrict vq, float *restrict la)
{
    const int nl = BK_CTX_NL(Tk), stride = BK_CTX_STRIDE;
    const float *pl = ps;
    for (int l = 0; l < nl; l++) {
        const int nr = (Tk - l + stride - 1) / stride;
        for (int c = 0; c < hd / P; c++) {
            const int vi = 8 + c / (BK_QK / P);
            v4f acc[BK_REP_FAST][4];
            for (int r = 0; r < rep; r++)
                for (int j = 0; j < 4; j++) acc[r][j] = v4_set1(0.0f);
            const int8_t *vr = vq + (size_t)l * hd + c * P;
            const float *pr = pl;
            for (int i = 0; i < nr; i++, vr += (size_t)stride * hd, pr += BK_PS) {
                v4f v[4];
                ld_i8_f(vr, v);
                if (!fold[l]) {
                    const v4f sv = v4_set1(pr[vi]);
                    v[0] = v[0] * sv; v[1] = v[1] * sv; v[2] = v[2] * sv; v[3] = v[3] * sv;
                }
                for (int r = 0; r < rep; r++) {
                    const v4f pb = v4_set1(pr[r]);
                    v4f p0 = pb * v[0], p1 = pb * v[1], p2 = pb * v[2], p3 = pb * v[3];
                    acc[r][0] = acc[r][0] + p0; acc[r][1] = acc[r][1] + p1;
                    acc[r][2] = acc[r][2] + p2; acc[r][3] = acc[r][3] + p3;
                }
            }
            for (int r = 0; r < rep; r++) {
                float *o = la + ((size_t)r * P + l) * hd + c * P;
                st4(o, acc[r][0]); st4(o + 4, acc[r][1]); st4(o + 8, acc[r][2]); st4(o + 12, acc[r][3]);
            }
        }
        pl += (size_t)nr * BK_PS;
    }
}

/* out[hd] = the spec's tree over the lanes a[P][hd], of which the first nl are in use.
 * A lane that is not in use is +0, and x + (+0) = x for every x a lane sum can be (never
 * -0), so those additions are skipped. */
BK_KERN static void lanes_tree_fast(float *restrict a, int nl, int hd, float *restrict out)
{
#if defined(BK_BUG_TREE)
    memset(a + (size_t)nl * hd, 0, (size_t)(P - nl) * hd * sizeof(float));
    for (int w = P; w > 1; w /= 2)
        for (int i = 0; i < w / 2; i++)
            for (int dd = 0; dd < hd; dd++)
                a[(size_t)i * hd + dd] = a[(size_t)(2 * i) * hd + dd] + a[(size_t)(2 * i + 1) * hd + dd];
    memcpy(out, a, (size_t)hd * sizeof(float));
#else
    if (nl == P) {
        for (int dd = 0; dd < hd; dd += 4) {
            v4f v[P];
            for (int l = 0; l < P; l++) v[l] = ld4(a + (size_t)l * hd + dd);
            for (int i = 0; i < 8; i++) v[i] = v[i] + v[i + 8];
            for (int i = 0; i < 4; i++) v[i] = v[i] + v[i + 4];
            for (int i = 0; i < 2; i++) v[i] = v[i] + v[i + 2];
            st4(out + dd, v[0] + v[1]);
        }
        return;
    }
    for (int w = P / 2; w >= 1; w /= 2)
        for (int i = 0; i < w && i + w < nl; i++) {
            float *restrict lo = a + (size_t)i * hd;
            const float *restrict hi = a + (size_t)(i + w) * hd;
            for (int dd = 0; dd < hd; dd++) lo[dd] = lo[dd] + hi[dd];
        }
    memcpy(out, a, (size_t)hd * sizeof(float));
#endif
}
#endif                                  /* BK_SIMD */

/* scores -> (tap) -> softmax -> (tap) -> context, for one kv head. t0: the thread's lap clock. */
static void attend_core(int fast, const att_in *a, int Tk, const float *q, bk_tls *T, float *ctx,
                        float *tap_s, float *tap_p, size_t tap_stride, int tap_n, uint64_t *t0)
{
    const int hd = a->hd, rep = a->rep;
#if BK_SIMD
    fast = fast && rep <= BK_REP_FAST;
    if (fast) {
        switch (rep) {
        case 1: score_fast_1(hd, Tk, a->isq, q, a->kq, a->ke, a->vq, T->sc, a->cap, T->kf); break;
        case 2: score_fast_2(hd, Tk, a->isq, q, a->kq, a->ke, a->vq, T->sc, a->cap, T->kf); break;
        case 4: score_fast_4(hd, Tk, a->isq, q, a->kq, a->ke, a->vq, T->sc, a->cap, T->kf); break;
        default: score_fast_n(rep, hd, Tk, a->isq, q, a->kq, a->ke, a->vq, T->sc, a->cap, T->kf); break;
        }
    } else
#else
    (void)fast;
    (void)hd;
#endif
        score_ref(a, Tk, q, T);
    TLAP(T, *t0, BK_T_ATT_SCORE);

    for (int r = 0; r < rep; r++) {
        float *sc = T->sc + (size_t)r * a->cap;
        if (tap_s) memcpy(tap_s + (size_t)r * tap_stride, sc, (size_t)tap_n * 4);
#if BK_SIMD
        if (fast) softmax_fast(a->tab, sc, Tk);
        else
#endif
            softmax_ref(a->tab, sc, Tk);
        if (tap_p) memcpy(tap_p + (size_t)r * tap_stride, sc, (size_t)tap_n * 4);
    }
    TLAP(T, *t0, BK_T_ATT_SOFTMAX);

#if BK_SIMD
    const int nv = hd / BK_QK;
    if (fast && nv <= BK_NV_FAST && !any_tiny_exp(a->ve, Tk * nv)) {
        uint8_t fold[P];
        ctx_rows(rep, nv, Tk, T->sc, a->cap, a->ve, T->ps, fold);
        switch (rep) {
        case 1: ctx_fast_1(hd, Tk, T->ps, fold, a->vq, T->la); break;
        case 2: ctx_fast_2(hd, Tk, T->ps, fold, a->vq, T->la); break;
        case 4: ctx_fast_4(hd, Tk, T->ps, fold, a->vq, T->la); break;
        default: ctx_fast_n(rep, hd, Tk, T->ps, fold, a->vq, T->la); break;
        }
        for (int r = 0; r < rep; r++)
            lanes_tree_fast(T->la + (size_t)r * P * hd, BK_CTX_NL(Tk), hd, ctx + (size_t)r * hd);
    } else
#endif
        ctx_ref(a, Tk, T, ctx);
    TLAP(T, *t0, BK_T_ATT_CTX);
}

int bk_attend_kv(int kernels, const float *exp_tab, int hd, int rep, int Tk, float inv_sqrt_hd,
                 const float *q, const int16_t *kq, const int8_t *ke, const int8_t *vq,
                 const int8_t *ve, float *s_out, float *p_out, float *ctx)
{
    if (!exp_tab || !q || !kq || !ke || !vq || !ve || !ctx || hd < BK_QK || hd % BK_QK || rep < 1 || Tk < 1)
        return BK_ERR_ARG;
    bk_tls T;
    memset(&T, 0, sizeof(T));
    T.sc = malloc((size_t)rep * Tk * sizeof(float));
    T.ps = malloc((size_t)Tk * 16 * sizeof(float));
    T.la = malloc((size_t)rep * P * hd * sizeof(float));
    T.kf = malloc((size_t)hd * sizeof(float));
    int rc = BK_ERR_NOMEM;
    if (T.sc && T.ps && T.la && T.kf) {
        att_in a = {hd, rep, Tk, inv_sqrt_hd, exp_tab, kq, ke, vq, ve};
        uint64_t t0 = now_ns();
        attend_core(kernels != BK_KERN_REF, &a, Tk, q, &T, ctx, s_out, p_out, (size_t)Tk, Tk, &t0);
        rc = BK_OK;
    }
    free(T.sc);
    free(T.ps);
    free(T.la);
    free(T.kf);
    return rc;
}

/* ================================================================== one token */
#define K_RMSNORM(s) ((s)->fast ? rmsnorm_fast : rmsnorm_ref)
#define K_ROPE(s) ((s)->fast ? rope_fast : rope_ref)
#define K_QUANT16(s) ((s)->fast ? quant_fast16 : quant_ref16)
#define K_QUANT8(s) ((s)->fast ? quant_fast8 : quant_ref8)

/* Region unit: everything of the attention that belongs to kv head g (the spec computes the
 * heads independently): q / k norms and RoPE, the cache rows of this position, scores,
 * softmax, context, and the BFP quantise of the context of its `rep` heads. */
static void att_unit(void *arg, int g, int tid)
{
    bk_state *s = arg;
    bk_tls *T = &s->tls[tid];
    const bk_config *c = &s->c;
    const bk_weights *w = &s->w;
    const bk_taps *tp = s->cur_tp;
    const int L = s->cur_L, pos = s->cur_pos, Tk = pos + 1, hd = s->hd, rep = s->rep;
    const int h0 = g * rep;
    const float *cosr = w->cos_tab + (size_t)pos * s->hd2, *sinr = w->sin_tab + (size_t)pos * s->hd2;
    const float *qn = w->q_norm + (size_t)L * hd, *kn = w->k_norm + (size_t)L * hd;
    float *qr = s->qr + (size_t)h0 * hd, *kr = s->kr + (size_t)g * hd;
    uint64_t t0 = now_ns();
    int rc;

    for (int r = 0; r < rep; r++)
        K_RMSNORM(s)(w->rs_lo, w->rs_hi, c->eps, s->yq + (size_t)(h0 + r) * hd, qn, hd, T->tq + (size_t)r * hd);
    K_RMSNORM(s)(w->rs_lo, w->rs_hi, c->eps, s->yk + (size_t)g * hd, kn, hd, T->tq + (size_t)rep * hd);
    TLAP(T, t0, BK_T_RMSNORM);
    for (int r = 0; r < rep; r++) K_ROPE(s)(cosr, sinr, T->tq + (size_t)r * hd, hd, qr + (size_t)r * hd);
    K_ROPE(s)(cosr, sinr, T->tq + (size_t)rep * hd, hd, kr);
    TLAP(T, t0, BK_T_ROPE);

    /* K (after RoPE): BFP10, exponent per 16 channels. V: BFP8, exponent per 128. */
    kv_head *kv = &s->kv[L * c->n_kv + g];
    rc = K_QUANT16(s)(kr, hd, BK_KGRP, BK_K_BITS, kv->kq + (size_t)pos * hd, kv->ke + (size_t)pos * (hd / BK_KGRP));
    if (rc == BK_OK)
        rc = K_QUANT8(s)(s->yv + (size_t)g * hd, hd, BK_QK, BK_V_BITS, kv->vq + (size_t)pos * hd,
                         kv->ve + (size_t)pos * (hd / BK_QK));
    TLAP(T, t0, BK_T_KVQ);
    if (rc != BK_OK) { s->unit_rc[g] = rc; return; }

    att_in a = {hd, rep, s->cap, c->inv_sqrt_hd, w->exp_tab, kv->kq, kv->ke, kv->vq, kv->ve};
    float *tap_s = NULL, *tap_p = NULL;
    int tap_n = 0;
    if (tp && tp->att_cap > 0) {
        tap_n = Tk < tp->att_cap ? Tk : tp->att_cap;
        if (tp->att_s) tap_s = tp->att_s + ((size_t)L * c->n_head + h0) * tp->att_cap;
        if (tp->att_p) tap_p = tp->att_p + ((size_t)L * c->n_head + h0) * tp->att_cap;
    }
    float *ctx = s->ctx + (size_t)h0 * hd;
    attend_core(s->fast, &a, Tk, qr, T, ctx, tap_s, tap_p, tp ? (size_t)tp->att_cap : 0, tap_n, &t0);

    rc = K_QUANT16(s)(ctx, rep * hd, BK_QK, BK_GEMV_BITS, s->xq_ctx + (size_t)h0 * hd,
                      s->xe_ctx + (size_t)h0 * hd / BK_QK);
    TLAP(T, t0, BK_T_QUANT);
    if (rc != BK_OK) s->unit_rc[g] = rc;
}

/* Region unit: a chunk of the FFN activation: act = silu(gate) * up and its BFP quantise. */
static void ffn_unit(void *arg, int u, int tid)
{
    bk_state *s = arg;
    bk_tls *T = &s->tls[tid];
    const int i0 = u * BK_FFN_CHUNK;
    const int n = s->c.ffn - i0 < BK_FFN_CHUNK ? s->c.ffn - i0 : BK_FFN_CHUNK;
    float *gate = s->gate + i0;
    float *sil = (s->cur_tp && s->cur_tp->silu) ? gate : NULL;      /* gate <- silu(gate) for the tap */
    uint64_t t0 = now_ns();
    if (s->fast) silu_fast(s->w.exp_tab, gate, s->up + i0, sil, s->act + i0, n);
    else silu_mul_ref(s->w.exp_tab, gate, s->up + i0, sil, s->act + i0, n);
    TLAP(T, t0, BK_T_SILU);
    int rc = K_QUANT16(s)(s->act + i0, n, BK_QK, BK_GEMV_BITS, s->xq_act + i0, s->xe_act + i0 / BK_QK);
    TLAP(T, t0, BK_T_QUANT);
    if (rc != BK_OK) s->unit_rc[u] = rc;
}

/* Run n units on the pool. The region's wall time (from the last lap) is split over the
 * buckets in proportion to the thread time spent in each. Returns the first unit error. */
static int run_region(bk_state *s, bk_job_fn fn, int n)
{
    for (int t = 0; t < s->n_threads; t++) memset(s->tls[t].ns, 0, sizeof(s->tls[t].ns));
    memset(s->unit_rc, 0, (size_t)n * sizeof(int));
    if (s->pool && n > 1) pool_run(s->pool, fn, s, n);
    else
        for (int u = 0; u < n; u++) fn(s, u, 0);
    uint64_t t1 = now_ns(), wall = t1 - s->tlast, cpu[BK_T_COUNT], tot = 0, given = 0;
    int big = BK_T_OTHER;
    for (int b = 0; b < BK_T_COUNT; b++) {
        cpu[b] = 0;
        for (int t = 0; t < s->n_threads; t++) cpu[b] += s->tls[t].ns[b];
        tot += cpu[b];
        if (cpu[b] > cpu[big]) big = b;
    }
    for (int b = 0; b < BK_T_COUNT; b++) {
        uint64_t share = tot ? (uint64_t)((double)wall * (double)cpu[b] / (double)tot) : 0;
        if (given + share > wall) share = wall - given;
        s->st.ns[b] += share;
        s->st.cpu_ns[b] += cpu[b];
        given += share;
    }
    s->st.ns[big] += wall - given;      /* rounding */
    s->st.regions++;
    s->tlast = t1;
    for (int u = 0; u < n; u++)
        if (s->unit_rc[u] != BK_OK) return s->unit_rc[u];
    return BK_OK;
}

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

static inline void resid(bk_state *s, float *x, const float *y, int n)
{
    if (s->fast) add_fast(x, y, n);
    else
        for (int i = 0; i < n; i++) x[i] = x[i] + y[i];
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
    const int d = c->d, ffn = c->ffn, nq = s->nq, nkv = s->nkv;
    const int ffn_units = (ffn + BK_FFN_CHUNK - 1) / BK_FFN_CHUNK;
    float *restrict x = s->x;
    s->cur_pos = pos;
    s->cur_tp = tp;
    s->tlast = now_ns();

    bk_embed_row(w->emb_signs, w->emb_sign_stride, w->emb_scales, w->emb_scale_stride, d, token, x);
    lap(s, BK_T_EMBED);

    for (int L = 0; L < c->n_layer; L++) {
        const int m0 = 7 * L;
        s->cur_L = L;
        if (tp) tapf(s, tp->x_in, (size_t)L * d, x, (size_t)d);

        /* --- attention --- */
        K_RMSNORM(s)(w->rs_lo, w->rs_hi, c->eps, x, w->attn_norm + (size_t)L * d, d, s->h);
        lap(s, BK_T_RMSNORM);
        BK_TRY(K_QUANT16(s)(s->h, d, BK_QK, BK_GEMV_BITS, s->xq_attn, s->xe_attn));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_Q, s->xq_attn, s->xe_attn, s->yq, tp));
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_K, s->xq_attn, s->xe_attn, s->yk, tp));
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_V, s->xq_attn, s->xe_attn, s->yv, tp));

        /* per kv head: norms, RoPE, cache rows, scores, softmax, context, quantise */
        BK_TRY(run_region(s, att_unit, c->n_kv));
        if (tp) {
            tapf(s, tp->q_rope, (size_t)L * nq, s->qr, (size_t)nq);
            tapf(s, tp->k_rope, (size_t)L * nkv, s->kr, (size_t)nkv);
            tapf(s, tp->ctx, (size_t)L * nq, s->ctx, (size_t)nq);
        }
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_O, s->xq_ctx, s->xe_ctx, s->yo, tp));
        resid(s, x, s->yo, d);
        lap(s, BK_T_RESID);
        if (tp) tapf(s, tp->x_mid, (size_t)L * d, x, (size_t)d);

        /* --- FFN --- */
        K_RMSNORM(s)(w->rs_lo, w->rs_hi, c->eps, x, w->ffn_norm + (size_t)L * d, d, s->h);
        lap(s, BK_T_RMSNORM);
        BK_TRY(K_QUANT16(s)(s->h, d, BK_QK, BK_GEMV_BITS, s->xq_ffn, s->xe_ffn));
        lap(s, BK_T_QUANT);
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_GATE, s->xq_ffn, s->xe_ffn, s->gate, tp));
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_UP, s->xq_ffn, s->xe_ffn, s->up, tp));
        /* act = silu(gate) * up, quantised, in chunks */
        BK_TRY(run_region(s, ffn_unit, ffn_units));
        if (tp) {
            tapf(s, tp->silu, (size_t)L * ffn, s->gate, (size_t)ffn);
            tapf(s, tp->act, (size_t)L * ffn, s->act, (size_t)ffn);
        }
        BK_TRY(call_gemv(s, gemv, gemv_ctx, m0 + BK_DOWN, s->xq_act, s->xe_act, s->yd, tp));
        resid(s, x, s->yd, d);
        lap(s, BK_T_RESID);
        if (tp) tapf(s, tp->x_out, (size_t)L * d, x, (size_t)d);
    }

    if (want_logits) {
        K_RMSNORM(s)(w->rs_lo, w->rs_hi, c->eps, x, w->out_norm, d, s->h);
        lap(s, BK_T_RMSNORM);
        if (tp) tapf(s, tp->h_out, 0, s->h, (size_t)d);
        BK_TRY(K_QUANT16(s)(s->h, d, BK_QK, BK_GEMV_BITS, s->xq_out, s->xe_out));
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
    if (ctx) {                          /* stand in for the fabric's time: busy-wait */
        int32_t us = *(const int32_t *)ctx;
        if (us > 0) {
            uint64_t t0 = now_ns();
            while (now_ns() - t0 < (uint64_t)us * 1000u) cpu_relax();
        }
    }
    return 0;
}
