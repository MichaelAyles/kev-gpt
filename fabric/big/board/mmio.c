/* Plain 32-bit accesses to the gemv_axi register window (AXI-Lite behind /dev/mem).
 *
 * A library memcpy must not be used on device memory: it issues 16-byte and unaligned
 * accesses (numpy 2.x's copy took a bus error on the result window). Every access here is one
 * aligned 32-bit load or store.
 *
 *   gcc -O2 -shared -fPIC -o libbk_mmio.so mmio.c
 */
#include <stddef.h>
#include <stdint.h>

void mmio_rd32(const volatile uint32_t *src, uint32_t *dst, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = src[i];
}

void mmio_wr32(volatile uint32_t *dst, const uint32_t *src, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = src[i];
}

/* Load `groups` activation groups: words = groups x nwords mantissa words, exps = int8 per
 * group. Register offsets are in bytes from `regs`. */
void mmio_load_act(volatile uint32_t *regs, const uint32_t *words, const int8_t *exps, int groups,
                   int nwords, uint32_t off_group, uint32_t off_exp, uint32_t off_mant, uint32_t off_commit)
{
    for (int g = 0; g < groups; g++) {
        regs[off_group >> 2] = (uint32_t)g;
        regs[off_exp >> 2] = (uint32_t)(int32_t)exps[g];
        for (int i = 0; i < nwords; i++)
            regs[(off_mant >> 2) + i] = words[(size_t)g * nwords + i];
        regs[off_commit >> 2] = 1;
    }
}

/* Poll a status register until (value & mask) != 0 or `spins` reads have gone by.
 * Returns the last value read. */
uint32_t mmio_wait(const volatile uint32_t *reg, uint32_t mask, uint64_t spins)
{
    uint32_t v = *reg;
    while (!(v & mask) && spins--)
        v = *reg;
    return v;
}

/* ------------------------------------------------------------------------------------------
 * gemv_axi version 2: the GEMV callback of host/bk_host.c with the activation and the results
 * going through DDR (aux DMA) and the LM-head argmax taken in the fabric. See
 * gemv_axi_regs.py for the register map and the record formats.
 *
 * Cache coherency is ours: the PL reads and writes DDR behind the A53's caches.
 *   activation: written through the cache, then every line is cleaned to DDR (DC CVAC)
 *   results:    after DONE every line is invalidated (DC CIVAC) before it is read; the host
 *               never writes that buffer, so the invalidate never writes stale data back.
 * ---------------------------------------------------------------------------------------- */
#include <math.h>
#include <string.h>
#include <time.h>

enum {
    R_CTRL, R_STATUS, R_ROWS, R_GROUPS, R_PAD, R_BURSTS, R_CHUNK, R_STEP, R_MAXOUT, R_RES_START,
    R_BASE_LO, R_BASE_HI, R_MODE, R_PORT_OFF, R_ACT_ADDR_LO, R_ACT_ADDR_HI, R_RES_ADDR_LO,
    R_RES_ADDR_HI, R_CYC_LO, R_CYC_HI, R_ROWS_DONE, R_TUPLES, R_STALL_IN, R_ERRCNT, R_ARGMAX_IDX,
    R_ARGMAX_VAL, R_CYC_ACT, R_CYC_RES, R_PORT_BASE, R_PORT_STRIDE, R_P_BEATS, R_P_XSUM, R_BATCH, R_ARGMAX_B, R_N
};

typedef struct {
    uint64_t pa;
    uint32_t rows, groups, pad, bursts, tuples, beats;
    uint32_t xsum[8][4];
} bkpl_mat;

typedef struct {
    volatile uint32_t *regs;
    uint32_t off[R_N];
    int32_t nports, chunk, maxout, n_mats;
    uint32_t step, port_off[8];
    const bkpl_mat *mats;                 /* in mat_id order; the last one is the LM head */
    uint8_t *act_va, *res_va;
    uint64_t act_pa, res_pa;
    int32_t head_argmax;                  /* 1: the LM head returns only its maximum (rest = -inf) */
    int32_t check_xsum;                   /* 1: compare every port's 128-bit xor with the image */
    uint64_t spins;                       /* poll limit per run */
    /* results / counters (the caller clears them) */
    uint64_t cyc, cyc_act, cyc_res, stall_in, runs, act_loads;
    uint64_t ns_act, ns_regs, ns_wait, ns_read;
    int32_t last_argmax, err_mat;
    uint32_t last_status, err_code;
    const int16_t *last_xq;
    /* register map VERSION 3 (batch): nb_max = NB of the bitstream, 0 on version 2 */
    int32_t nb_max, last_nb;
    int32_t argmax_b[8];                  /* per element, after an argmax-only LM head */
    uint64_t res_cap;                     /* bytes available at res_va */
} bkpl;

static inline uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static inline size_t line_size(void)
{
#ifdef __aarch64__
    uint64_t ctr;
    __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
    return (size_t)4 << ((ctr >> 16) & 0xf);
#else
    return 64;
#endif
}

void bkpl_cache_clean(const void *p, size_t n)       /* push dirty lines to DDR */
{
#ifdef __aarch64__
    size_t ls = line_size();
    uintptr_t a = (uintptr_t)p & ~(ls - 1), e = (uintptr_t)p + n;
    for (; a < e; a += ls)
        __asm__ volatile("dc cvac, %0" : : "r"(a) : "memory");
    __asm__ volatile("dsb ish" : : : "memory");
#else
    (void)p; (void)n;
#endif
}

void bkpl_cache_invalidate(const void *p, size_t n)  /* drop (clean and) cached lines */
{
#ifdef __aarch64__
    size_t ls = line_size();
    uintptr_t a = (uintptr_t)p & ~(ls - 1), e = (uintptr_t)p + n;
    for (; a < e; a += ls)
        __asm__ volatile("dc civac, %0" : : "r"(a) : "memory");
    __asm__ volatile("dsb ish" : : : "memory");
#else
    (void)p; (void)n;
#endif
}

/* groups x 256-byte records: 128 ten-bit mantissas (element j at bits [10j+9:10j]) in bytes
 * 0..159, the exponent in byte 160. Bytes 161..255 are never written (zeroed at set-up). */
void bkpl_pack_act(uint8_t *dst, const int16_t *xq, const int8_t *xe, int groups)
{
    for (int g = 0; g < groups; g++, dst += 256, xq += 128) {
        uint8_t *o = dst;
        for (int j = 0; j < 128; j += 4, o += 5) {     /* 4 mantissas = 40 bits = 5 bytes */
            uint64_t v = (uint64_t)((uint16_t)xq[j] & 0x3ff) | (uint64_t)((uint16_t)xq[j + 1] & 0x3ff) << 10 |
                         (uint64_t)((uint16_t)xq[j + 2] & 0x3ff) << 20 | (uint64_t)((uint16_t)xq[j + 3] & 0x3ff) << 30;
            memcpy(o, &v, 8);                          /* little-endian; the 3 spare bytes are zero and
                                                          are overwritten by the next store (the last one
                                                          zeroes bytes 160..162, byte 160 is set below) */
        }
        dst[160] = (uint8_t)xe[g];
    }
}

#define RG(c, r) ((c)->regs[(c)->off[r] >> 2])

int bkpl_gemv(void *vctx, int mat_id, const int16_t *xq, const int8_t *xe, int groups, float *y, int rows)
{
    bkpl *c = (bkpl *)vctx;
    if (mat_id < 0 || mat_id >= c->n_mats)
        return -10;
    const bkpl_mat *m = &c->mats[mat_id];
    if ((uint32_t)rows != m->rows || (uint32_t)groups != m->groups)
        return -11;
    int head = mat_id == c->n_mats - 1 && c->head_argmax;
    uint32_t mode = head ? 4u : 6u;                    /* RES_NOCAP, + RES_DMA unless argmax only */
    if (c->nb_max > 0 && c->last_nb != 1) {            /* after a batch run: back to one element */
        RG(c, R_BATCH) = 1;
        c->last_nb = 1;
        c->last_xq = 0;
    }
    uint64_t t0 = now_ns();
    if (xq != c->last_xq) {                            /* q/k/v and gate/up share one input */
        bkpl_pack_act(c->act_va, xq, xe, groups);
        bkpl_cache_clean(c->act_va, (size_t)groups * 256);
        c->last_xq = xq;
        c->act_loads++;
        mode |= 1u;                                    /* ACT_DMA */
    }
    uint64_t t1 = now_ns();
    RG(c, R_ROWS) = m->rows;
    RG(c, R_GROUPS) = m->groups;
    RG(c, R_PAD) = m->pad;
    RG(c, R_BURSTS) = m->bursts;
    RG(c, R_BASE_LO) = (uint32_t)m->pa;
    RG(c, R_BASE_HI) = (uint32_t)(m->pa >> 32);
    RG(c, R_MODE) = mode;
    RG(c, R_CTRL) = 1;
    uint64_t t2 = now_ns();
    uint32_t st = mmio_wait(&RG(c, R_STATUS), 2, c->spins);
    uint64_t t3 = now_ns();
    c->last_status = st;
    c->ns_act += t1 - t0;
    c->ns_regs += t2 - t1;
    c->ns_wait += t3 - t2;
    if (st != 2u || RG(c, R_ERRCNT) != 0 || RG(c, R_ROWS_DONE) != m->rows || RG(c, R_TUPLES) != m->tuples) {
        c->err_mat = mat_id;
        c->err_code = st;
        return -12;
    }
    if (c->check_xsum) {
        for (int k = 0; k < c->nports; k++) {
            uint32_t pb = c->off[R_PORT_BASE] + c->off[R_PORT_STRIDE] * (uint32_t)k;
            if (c->regs[(pb + c->off[R_P_BEATS]) >> 2] != m->beats) {
                c->err_mat = mat_id;
                return -13;
            }
            for (int i = 0; i < 4; i++)
                if (c->regs[(pb + c->off[R_P_XSUM] + 4u * (uint32_t)i) >> 2] != m->xsum[k][i]) {
                    c->err_mat = mat_id;
                    return -14;                        /* the beats read are not the packed image */
                }
        }
    }
    c->cyc += RG(c, R_CYC_LO) | ((uint64_t)RG(c, R_CYC_HI) << 32);
    c->cyc_act += RG(c, R_CYC_ACT);
    c->cyc_res += RG(c, R_CYC_RES);
    c->stall_in += RG(c, R_STALL_IN);
    c->runs++;
    if (head) {
        uint32_t idx = RG(c, R_ARGMAX_IDX), raw = RG(c, R_ARGMAX_VAL);
        float v;
        memcpy(&v, &raw, 4);
        for (int i = 0; i < rows; i++)
            y[i] = -INFINITY;
        if (idx >= (uint32_t)rows) {
            c->err_mat = mat_id;
            return -15;
        }
        y[idx] = v;
        c->last_argmax = (int32_t)idx;
    } else {
        bkpl_cache_invalidate(c->res_va, (size_t)rows * 4);
        memcpy(y, c->res_va, (size_t)rows * 4);
    }
    c->ns_read += now_ns() - t3;
    return 0;
}

/* registers that do not change between runs */
/* Batch form (register map VERSION 3): nb activation vectors through one pass over the matrix.
 * xq[b] / xe[b] / y[b] are element b's arrays, laid out exactly as bkpl_gemv's. reuse != 0: the
 * nb activations are the ones of the previous call (same nb), so the banks are not reloaded.
 * The argmax-only LM head (head_argmax) fills y[b] with -inf except the maximum when y[b] is
 * not NULL, and always sets argmax_b[b]. */
int bkpl_gemv_batch(void *vctx, int mat_id, int nb, const int16_t *const *xq, const int8_t *const *xe, int groups,
                    float *const *y, int rows, int reuse)
{
    bkpl *c = (bkpl *)vctx;
    if (mat_id < 0 || mat_id >= c->n_mats)
        return -10;
    const bkpl_mat *m = &c->mats[mat_id];
    if ((uint32_t)rows != m->rows || (uint32_t)groups != m->groups)
        return -11;
    if (nb < 1 || nb > c->nb_max || nb > 8)
        return -16;
    int head = mat_id == c->n_mats - 1 && c->head_argmax;
    size_t res_bytes = ((size_t)rows * (size_t)nb * 4 + 15) & ~(size_t)15;
    if (!head && res_bytes > c->res_cap)
        return -17;
    uint32_t mode = head ? 4u : 6u;
    uint64_t t0 = now_ns();
    if (!reuse || nb != c->last_nb) {
        for (int b = 0; b < nb; b++)
            bkpl_pack_act(c->act_va + (size_t)256 * (size_t)b * (size_t)groups, xq[b], xe[b], groups);
        bkpl_cache_clean(c->act_va, (size_t)nb * (size_t)groups * 256);
        c->act_loads += (uint64_t)nb;
        mode |= 1u;
    }
    c->last_xq = 0;                                    /* the single-vector reuse test no longer holds */
    c->last_nb = nb;
    uint64_t t1 = now_ns();
    RG(c, R_BATCH) = (uint32_t)nb;
    RG(c, R_ROWS) = m->rows;
    RG(c, R_GROUPS) = m->groups;
    RG(c, R_PAD) = m->pad;
    RG(c, R_BURSTS) = m->bursts;
    RG(c, R_BASE_LO) = (uint32_t)m->pa;
    RG(c, R_BASE_HI) = (uint32_t)(m->pa >> 32);
    RG(c, R_MODE) = mode;
    RG(c, R_CTRL) = 1;
    uint64_t t2 = now_ns();
    uint32_t st = mmio_wait(&RG(c, R_STATUS), 2, c->spins);
    uint64_t t3 = now_ns();
    c->last_status = st;
    c->ns_act += t1 - t0;
    c->ns_regs += t2 - t1;
    c->ns_wait += t3 - t2;
    if (st != 2u || RG(c, R_ERRCNT) != 0 || RG(c, R_ROWS_DONE) != m->rows || RG(c, R_TUPLES) != m->tuples) {
        c->err_mat = mat_id;
        c->err_code = st;
        return -12;
    }
    if (c->check_xsum) {
        for (int k = 0; k < c->nports; k++) {
            uint32_t pb = c->off[R_PORT_BASE] + c->off[R_PORT_STRIDE] * (uint32_t)k;
            if (c->regs[(pb + c->off[R_P_BEATS]) >> 2] != m->beats) {
                c->err_mat = mat_id;
                return -13;
            }
            for (int i = 0; i < 4; i++)
                if (c->regs[(pb + c->off[R_P_XSUM] + 4u * (uint32_t)i) >> 2] != m->xsum[k][i]) {
                    c->err_mat = mat_id;
                    return -14;
                }
        }
    }
    c->cyc += RG(c, R_CYC_LO) | ((uint64_t)RG(c, R_CYC_HI) << 32);
    c->cyc_act += RG(c, R_CYC_ACT);
    c->cyc_res += RG(c, R_CYC_RES);
    c->stall_in += RG(c, R_STALL_IN);
    c->runs++;
    if (head) {
        for (int b = 0; b < nb; b++) {
            uint32_t a = c->off[R_ARGMAX_B] + 8u * (uint32_t)b;
            uint32_t idx = c->regs[a >> 2], raw = c->regs[(a + 4u) >> 2];
            if (idx >= (uint32_t)rows) {
                c->err_mat = mat_id;
                return -15;
            }
            c->argmax_b[b] = (int32_t)idx;
            if (y[b]) {
                float v;
                memcpy(&v, &raw, 4);
                for (int i = 0; i < rows; i++)
                    y[b][i] = -INFINITY;
                y[b][idx] = v;
            }
        }
        c->last_argmax = c->argmax_b[nb - 1];
    } else {
        bkpl_cache_invalidate(c->res_va, res_bytes);
        const float *s = (const float *)(const void *)c->res_va;   /* row r, element b at r * nb + b */
        if (nb == 1)
            memcpy(y[0], s, (size_t)rows * 4);
        else
            for (int b = 0; b < nb; b++) {
                float *d = y[b];
                const float *q = s + b;
                for (int r = 0; r < rows; r++, q += nb)
                    d[r] = *q;
            }
    }
    c->ns_read += now_ns() - t3;
    return 0;
}

void bkpl_setup(bkpl *c)
{
    RG(c, R_CHUNK) = (uint32_t)c->chunk;
    RG(c, R_STEP) = c->step;
    RG(c, R_MAXOUT) = (uint32_t)c->maxout;
    RG(c, R_RES_START) = 0;
    for (int k = 0; k < c->nports; k++)
        c->regs[(c->off[R_PORT_OFF] + 4u * (uint32_t)k) >> 2] = c->port_off[k];
    RG(c, R_ACT_ADDR_LO) = (uint32_t)c->act_pa;
    RG(c, R_ACT_ADDR_HI) = (uint32_t)(c->act_pa >> 32);
    RG(c, R_RES_ADDR_LO) = (uint32_t)c->res_pa;
    RG(c, R_RES_ADDR_HI) = (uint32_t)(c->res_pa >> 32);
    c->last_xq = 0;
    if (c->nb_max > 0)
        RG(c, R_BATCH) = 1;
    c->last_nb = 1;
}
