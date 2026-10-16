"""On-chip sizing for the Genesys2 (Kintex-7 xc7k325t) port.

kev-gpt/fabric/spec.py is the KV260 (Zynq UltraScale+) contract: its
ONCHIP_WEIGHT_BUDGET_BYTES (~3 MB) and weight_params() (linears only, no head
GEMV, no KV cache) describe that board, not this one. This module adds what
the Genesys2 port needs on top, without touching spec.py's KV260 numbers:

  1. The head GEMV (vocab x n_embd) that spec.py's weight_params() excludes,
     but that sequencer_vec.sv streams as GW_HEAD and every onchip-residency
     budget must therefore count.
  2. The KV cache footprint (kv_bank.sv's persistent K8/V8 code + header
     banks), which the KV260 budget also folds in but spec.py never modelled
     explicitly since KV260's ~3 MB ceiling had enough slack not to need it.
  3. Genesys2's actual free BRAM pool, which is a different device family
     (36Kb dual-port BRAM, no URAM at all) with a much smaller number.

Formulas verified against two known-good worked examples (see
`_self_check()`): the deployed KV260 config's own stated "~1.5MB weights",
and kv_bank.sv's own header-comment example ("TMAX=256, 4 layers, 4 heads ->
codes 512KB + hdr 48KB").
"""

from __future__ import annotations

from dataclasses import dataclass

from fabric.spec import ModelDims, WEIGHT_BITS

# --------------------------------------------------------------------------- #
# Target device — Genesys2 (Kintex-7 xc7k325tffg900-2)
# --------------------------------------------------------------------------- #
# No URAM on this part at all (UltraScale+-only primitive) -- everything is
# ordinary 36Kb-slice dual-port Block RAM.
XC7K325T_RAMB36_TOTAL = 445
RAMB36_BYTES = 36 * 1024 // 8  # 4608 bytes per 36Kb-slice block

# Measured (not assumed): Genesys2AiChatbot's own placed Vivado utilisation
# report for X-HEEP + cv32e40px alone (no accelerator) on the real part --
# see genesys2-vivado/.../xilinx_core_v_mini_mcu_wrapper_minimal_utilization_placed.rpt.
XHEEP_SOC_RAMB36_USED = 72
GENESYS2_FREE_RAMB36 = XC7K325T_RAMB36_TOTAL - XHEEP_SOC_RAMB36_USED
GENESYS2_FREE_BYTES = GENESYS2_FREE_RAMB36 * RAMB36_BYTES  # ~1.64 MB

# Recommended conservative target: leave real margin below the free pool for
# GEMV/dequant/GELU/Gumbel scratch BRAMs already present in sequencer_vec.sv
# (qkv_bank, head_bank, LUTs) and for Vivado packing overhead when many small
# RAMB18/36s replace the wider URAM288 blocks the KV260 build relied on.
GENESYS2_BUDGET_BYTES = int(1.3 * 1024 * 1024)

# kv_bank.sv pins K/V codes at 8 bits (K8/V8) -- not a free parameter.
KV_CODE_BITS = 8
# Per-position header bytes/head, from kv_bank.sv's own layout (verified by
# _self_check() against its header-comment worked example).
KV_HDR_BYTES_PER_HEAD_POS = 12


def head_weight_bytes(dims: ModelDims, weight_bits: int = WEIGHT_BITS) -> int:
    """The head GEMV (vocab x n_embd), GW_HEAD in sequencer_vec.sv -- excluded
    from spec.ModelDims.weight_params() (linears-only) but resident on-chip."""
    return dims.vocab_size * dims.n_embd * weight_bits // 8


def total_weight_bytes(dims: ModelDims, weight_bits: int = WEIGHT_BITS) -> int:
    """Every INT4 weight byte sequencer_vec.sv keeps resident: the four
    per-layer linears (spec.weight_params()) plus the head GEMV."""
    return dims.weight_params() * weight_bits // 8 + head_weight_bytes(dims, weight_bits)


@dataclass(frozen=True)
class KVCacheBytes:
    code_bytes: int
    hdr_bytes: int

    @property
    def total(self) -> int:
        return self.code_bytes + self.hdr_bytes


def kv_cache_bytes(dims: ModelDims) -> KVCacheBytes:
    """kv_bank.sv's persistent K8/V8 cache footprint for a full context window
    (dims.block_size == kv_bank's TMAX). code_bytes holds the quantised K/V
    codes themselves; hdr_bytes holds the per-position/per-head scale headers."""
    code_bytes = dims.n_layer * dims.n_embd * KV_CODE_BITS * dims.block_size // 4
    hdr_bytes = KV_HDR_BYTES_PER_HEAD_POS * dims.n_layer * dims.n_head * dims.block_size
    return KVCacheBytes(code_bytes=code_bytes, hdr_bytes=hdr_bytes)


def onchip_fits_genesys2(
    dims: ModelDims, weight_bits: int = WEIGHT_BITS, budget_bytes: int = GENESYS2_BUDGET_BYTES
) -> tuple[int, bool]:
    """Return (total_bytes, fits) for weights + full KV cache against the
    Genesys2 conservative budget -- the load-bearing check for this port,
    analogous to spec.onchip_fits() for the KV260."""
    total = total_weight_bytes(dims, weight_bits) + kv_cache_bytes(dims).total
    return total, total <= budget_bytes


# --------------------------------------------------------------------------- #
# Sizing options, as presented in the port plan
# --------------------------------------------------------------------------- #
# n_head varies, head_dim is held FIXED at 64 (the deployed KV260 value) for
# all three options -- not n_head=4 for every option. This is a deliberate
# port decision, not an oversight: sequencer_vec.sv's attention-score scale
# (1/sqrt(head_dim)) and the shared Python golden reference (seq_ref.ISQRT /
# vec_attn_w.sv's SCORE_SH) are hardcoded exact integer shifts valid only for
# head_dim=64 specifically (see fabric/genesys2/PORT-NOTES.md for the full
# story of why this surfaced and what generalizing head_dim itself would
# have cost). vocab_size=57 matches this repo's actual trained tokenizer
# (data/TinyStories-valid.kevin.txt -> model.data), not the KV260 self-check's
# vocab_size=193 (sequencer_vec.sv's own default, from a different/larger
# corpus snapshot -- see _self_check() below, which correctly keeps 193).
OPTION_A_SAFE_BRINGUP = ModelDims(n_layer=2, n_head=2, n_embd=128, block_size=128, vocab_size=57)
# NOT n_embd=192: layernorm_vec.sv's mean/var divide (see PORT-NOTES.md) is
# ALSO an exact shift, valid only for a power-of-2 D -- independent of the
# head_dim=64-fixed constraint above. With head_dim pinned at 64, D=n_head*64
# is only a power of 2 at n_head in {1,2,4,...}, i.e. D in {64,128,256,...};
# there is no valid D between 128 and 256 under both constraints at once. So
# "Option B" grows via n_layer/context instead of D, keeping D=128 (n_head=2)
# -- 2x Option A's depth and context.
OPTION_B_RECOMMENDED = ModelDims(n_layer=4, n_head=2, n_embd=128, block_size=256, vocab_size=57)
OPTION_C_REJECTED = ModelDims(n_layer=4, n_head=4, n_embd=256, block_size=48, vocab_size=57)


def _self_check() -> None:
    # 1. KV260 deployed config (d=256, n_layer=4, vocab=193): weights alone
    #    (linears + head) should land at ~1.5 MB, matching the project's own
    #    stated figure (fabric/stage3/PL-ARCHITECTURE.md).
    kv260 = ModelDims(n_layer=4, n_head=4, n_embd=256, block_size=256, vocab_size=193)
    w = total_weight_bytes(kv260)
    assert abs(w - 1_597_568) < 4, f"KV260 weight-size check failed: {w} bytes"

    # 2. kv_bank.sv's own header-comment example: "TMAX=256, 4 layers, 4 heads
    #    -> codes 512KB + hdr 48KB".
    example = ModelDims(n_layer=4, n_head=4, n_embd=256, block_size=256, vocab_size=193)
    kv = kv_cache_bytes(example)
    assert kv.code_bytes == 512 * 1024, f"KV code-bytes check failed: {kv.code_bytes}"
    assert kv.hdr_bytes == 48 * 1024, f"KV header-bytes check failed: {kv.hdr_bytes}"


def _print_option(name: str, dims: ModelDims) -> None:
    w = total_weight_bytes(dims)
    kv = kv_cache_bytes(dims)
    total = w + kv.total
    pct = 100.0 * total / GENESYS2_FREE_BYTES
    print(
        f"{name:28s} d={dims.n_embd:3d} L={dims.n_layer} H={dims.n_head} "
        f"ctx={dims.block_size:3d}  weights={w/1024:7.1f}KB  kv={kv.total/1024:6.1f}KB "
        f"(code={kv.code_bytes/1024:.1f}KB hdr={kv.hdr_bytes/1024:.1f}KB)  "
        f"total={total/1024:7.1f}KB  ({pct:4.1f}% of {GENESYS2_FREE_BYTES/1024:.0f}KB free)"
    )


if __name__ == "__main__":
    _self_check()
    print("self-check OK (KV260 weight size, kv_bank.sv worked example)\n")
    print(f"xc7k325t total RAMB36        {XC7K325T_RAMB36_TOTAL}")
    print(f"X-HEEP SoC RAMB36 (measured) {XHEEP_SOC_RAMB36_USED}")
    print(f"free RAMB36 / bytes          {GENESYS2_FREE_RAMB36} / {GENESYS2_FREE_BYTES/1024:.1f} KB")
    print(f"recommended budget           {GENESYS2_BUDGET_BYTES/1024:.0f} KB\n")
    _print_option("A - safe bring-up", OPTION_A_SAFE_BRINGUP)
    _print_option("B - recommended final", OPTION_B_RECOMMENDED)
    _print_option("C - original size, ctx cut (rejected)", OPTION_C_REJECTED)
