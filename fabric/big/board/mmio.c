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
