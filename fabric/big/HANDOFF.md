# Big Kev — handoff (2026-11-10, updated after the R1 leftovers and R2a sim)

**Goal.** Run a 1-bit 8B LLM (PrismML **Bonsai-8B**, Qwen3-8B architecture) on a $250 Xilinx
Kria KV260, streaming its weights from DDR as fast as the bus allows. Single-stream decode is
memory-bandwidth bound, so **tok/s ≈ DDR bandwidth ÷ bytes per token**. This work is the
DDR-resident counterpart to the repo's on-chip "Kevin" model. Optimistic target **50 tok/s**: about
14 from line-rate streaming, then sub-1-bit weight compression (R3) and speculative decoding
with an on-chip draft model (R4).

Branch `big-kev` (off `main`, not pushed). Everything lives under `fabric/big/`. The design
docs (`docs/10-big-kevin.md`, `docs/big-kev-r0.md`, `docs/big-kev-r1.md`) are **local-only**
because `docs/` is gitignored, so this file carries the context those docs hold.

Rules of the house (from CLAUDE.md, and they apply here):
- **Bit-honest before fast.** Every block is gated bit-exact against its Python reference
  before any speed number counts.
- Tag numbers **MEASURED** (silicon or a committed tool) / **DERIVED** (arithmetic) /
  **PROJECTED** (needs a build or run). Say where the approach loses.

## 1. Where we are

| rung | status | result |
|---|---|---|
| **R0** DDR bandwidth probe | **DONE, MEASURED on silicon** | PL→DDR **15.54 GB/s = 91.1%** of the DDR4-2133 peak (17.07 GB/s), using HP0+HP1+HP3 at 333 MHz with a **port-interleaved layout**. Data-checked. |
| **R1a** float reference | **PASS** | from-scratch Qwen3 forward == HF transformers (max Δlogit 1.7e-5, greedy identical) |
| **R1b** GEMV activation format | **PASS** | **BFP10, one power-of-two exponent per 128-group**: 99.48% top-1 vs float |
| **R1c** bit-true fabric spec | **PASS** | **99.41% top-1**, ΔNLL +0.0004, cos 0.99999 vs float over 1,020 tokens (K = BFP10 g16); step()==forward_seq() bit-identical |
| **R1 long context** | **PASS** (after a spec change) | top-1 **0.9961 at T=1024, 0.9946 at T=2048**, no trend across 256-token buckets. First run FAILED at 0.9886/0.9885 with 8-bit K; root cause the K mantissa width → **K = BFP10** |
| **R1 golden vectors** | **PASS** | trace taps in the spec + `.mem` exporter; 8B dump (layers 0 and 35) round-trips bit-exact, `GOLDEN_VERDICT PASS 42/42` |
| **R1 free-run** | **PASS** | 3 prompts × 80 greedy tokens: fabric output identical to float greedy at every position |
| **R1d** ternary-4B vs binary-8B | **DONE, no flag** | pooled NLL 2.598 vs 2.516 (binary wins by ~2σ, tied on neutral text); bytes equal as stored, ternary 18.5% smaller at 5 trits/byte. Ternary-8B is the quality ceiling (2.438) at 1.53× the bytes. R2 stays on binary-8B |
| **R2a (sim)** packer + streaming GEMV RTL | **PASS in iverilog** | `GEMV_STREAM_VERDICT PASS` (136 sims, bit-exact vs `ExactBfpGemv`), 2.637 groups/cycle at 3 ports, 1,064.1 MB/token packed. **OOC place+route at 3.0 ns (333 MHz): WNS +0.197 ns, 9.3k LUT (8%), 13.3k FF, 6 DSP, 0 BRAM** (PROJECTED, core only) |
| **R2** streaming engine on silicon | not started | roofline **~14.0 tok/s** (DERIVED: 1,064 MB weights + ~44 MB KV per token at ctx 512) |

Key facts you would otherwise rediscover:
- **KV260 DDR is DDR4-2133** (read from the DPLL/DDR_CTRL/MSTR registers), so the peak is
  17.07 GB/s, not 19.2.
- **HP1 and HP2 share a DDR controller port.** Use HP0+HP1+HP3 at 333 MHz (15.5 GB/s), or HP0–3
  at 250 MHz (14.6). The HPC ports only hurt. Far-apart per-port regions thrash DRAM rows
  (~10.8 GB/s); interleave instead.
- fclk0 cannot make 300 MHz. It snaps to 333.3 / 250 / 200.
- **Bonsai Q1_0 format** (ggml type 41): per 128 weights along a row, one fp16 scale + 16 bytes
  of signs; weight j = bit (j%8) of byte (j//8), 1 = +scale. Every matrix is 1-bit, including the
  embedding and LM head. Pinned against PrismML's unpacked release.
- Qwen3 here uses **YaRN RoPE ×4** (original context 16384 on the 8B, 8192 on the 1.7B), applied
  at every length.
- **K cache = BFP10 with an exponent per 16 channels.** Qwen's post-RoPE K has outlier
  channels; BFP8 with one exponent per 128 dropped agreement to 96.3% (R1c), and BFP8 per 16
  still flipped near-ties on number-heavy text at long context (0.9886). V stays BFP8 per 128.
- **Agreement numbers are noisy at the 0.3–0.5% level** (a few hundred to a few thousand
  tokens, and most disagreements are float near-ties). Use ablations, not single runs.

## 2. Code map (`fabric/big/`)

| path | what |
|---|---|
| `rtl/axi_rd_probe.v`, `gen_bwprobe_top.py` → `rtl/bwprobe_axi.v` | R0 probe: 6 AXI4 read masters + AXI-Lite regs (generated top) |
| `tb/`, `run_bwprobe.py` | R0 iverilog gate (`BWPROBE_VERDICT`) |
| `tcl/build_bwprobe.tcl` | Vivado BD + synth/impl/bitstream (Vivado 2026.1 at `~/tools/Xilinx/2026.1`) |
| `board/pl_bwprobe.py` | on-Kria driver: `--check` (data honesty) `--sweep` `--ilv` |
| `ref/gguf_q1.py` | minimal GGUF reader + Q1_0 dequant |
| `ref/qwen3_ref.py` | **float reference** (packed 1-bit weights on GPU, `step()` + `forward_seq()`, YaRN) |
| `ref/fabric_ref.py` | **the bit-true spec the RTL copies.** Its module docstring is normative. |
| `ref/actquant.py`, `ref/r1b_sweep.py` | R1b activation-format sweep |
| `ref/gate_float.py` | R1a gate (vs HF, 1.7B, CPU) |
| `ref/r1c_gate.py` | R1c gate (fabric spec vs float, 8B, GPU) → `results/r1c_gate.json` |
| `ref/r1_longctx.py` | long-context gate (chunked prefill, 256-token buckets) → `results/r1_longctx.json`; `r1_longctx_k8.*`, `_ablate.*`, `_ksweep.*` are the failing run and its root-cause |
| `ref/golden.py`, `tests/test_big_golden.py` | golden-vector exporter (`.mem` + `MANIFEST.json`) with `--verify` round-trip; tiny-model test |
| `ref/r1_freerun.py` | free-running greedy check → `results/r1_freerun.json` |
| `ref/r1d_ternary.py` | ternary vs binary: Q2_0 pin, bytes/token table, NLL eval → `results/r1d_ternary.json`, `r1d_pin.json` |
| `pack_weights.py` | **R2a weight packer**: Q1 matrices → port-interleaved 128-bit beat stream (docstring is the stream-format spec) |
| `rtl/gemv_stream.v`, `tb/tb_gemv_stream.sv`, `run_gemv_stream.py` | **R2a streaming GEMV** + iverilog gate (`GEMV_STREAM_VERDICT`) |
| `tcl/ooc_gemv_stream.tcl` | OOC synth + place + route of the GEMV core (~7 min) → `results/ooc_gemv_stream.txt` |
| `results/` | raw R0 sweeps, R1b sweep, R1c gate, the R1 leftover results, cached float chat text (`Bonsai-8B-Q1_0_chatgen_T256.json`, which took 35 min to make) |

The fabric spec in one breath: the GEMV input goes to BFP10 per 128; per group an exact integer
Σ sign·q (17 b) is multiplied by the fp16 scale split as an 11-bit integer × 2^e, then shifted
into a 64-bit accumulator with LSB 2^−40 (peak use 56 bits), with one RNE to fp32 at the end. The
vector path is IEEE fp32 with **lane-strided reductions** (lane = i % 16, sequential per lane,
then a pairwise tree), table exp (4097 entries) and rsqrt (1024 entries + 2 Newton steps),
streamed RoPE tables, and KV as block floating point (K: BFP10, exponent per 16 channels;
V: BFP8, exponent per 128).

## 3. Environment

```bash
cd kev-gpt && git checkout big-kev
# python: the repo .venv (torch 2.4 cu121). Add the HF deps if missing:
uv pip install --python .venv/bin/python "transformers>=4.51,<5" safetensors
# weights (data/ is gitignored):
.venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("prism-ml/Bonsai-8B-gguf", "Bonsai-8B-Q1_0.gguf", local_dir="data/big")
hf_hub_download("prism-ml/Bonsai-1.7B-gguf", "Bonsai-1.7B-Q1_0.gguf", local_dir="data/big")
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    hf_hub_download("prism-ml/Bonsai-8B-unpacked", f, local_dir="data/big/Bonsai-8B-meta")
snapshot_download("prism-ml/Bonsai-1.7B-unpacked", local_dir="data/big/Bonsai-1.7B-unpacked")  # R1a only
EOF
# eval texts also read data/TinyStories-valid.txt (gitignored; any ~40 KB of plain English works)
```

**Hardware notes.**
- The dev laptop has a **4 GB RTX 3050 Ti**. One 8B job at a time (peak ~2.6 GB).
- Keep logits on the CPU (vocab 151,669 × T × 8 bytes adds up fast).
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Always run Python with `-u` and don't pipe through `grep` while it runs, or you see no
  progress.
- On CPU-only machines the exact-integer spec is slow on the 8B: develop on
  `Bonsai-1.7B-Q1_0.gguf` (same architecture), and run 8B checks only where needed.

- **Vivado licence** is node-locked to the laptop's home-network MAC (or the dock ethernet).
  On a hotspot NetworkManager randomises the MAC and synthesis cannot check out a licence:
  `nmcli connection modify "<profile>" wifi.cloned-mac-address permanent`, then reconnect.
- Ternary GGUFs for R1d live in `data/big/` (`Ternary-Bonsai-4B-Q2_0.gguf`, `-1.7B-`); the
  tokenizer is identical to Bonsai-8B's.

Sanity check that the setup works (about 3 min on the laptop GPU):
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m fabric.big.ref.r1c_gate`
→ `R1C_VERDICT PASS`.

## 4. Open tasks

The four R1 leftovers are done (table in §1; write-up in `docs/big-kev-r1.md` §8–§13, local
only). What is open now, in order:

1. ~~OOC synth of `gemv_stream`~~ **done:** post-route WNS +0.197 ns at 3.0 ns, 9.3k LUT, 13.3k FF,
   6 DSP (`results/ooc_gemv_stream.txt`). Worst path = fan-out of the sign-beat steering select.
   The margin is thin and core-only: re-time in context once the AXI masters are attached.
2. **R2a on silicon** (needs the board, §6): AXI read masters + address generator around
   `gemv_stream` (reuse `rtl/axi_rd_probe.v`'s burst logic), a packed image in reserved DDR,
   a driver that loads activations and reads fp32 rows back. **Accept:** bit-exact against the
   golden vectors (`ref/golden.py`) on real 8B matrices, then a MEASURED GB/s and the DERIVED
   tok/s it implies.
3. **Stream layout decision:** the packer's dense 3-slot stripe has no silicon number; R0's
   15.54 GB/s used a 4-slot stripe with HP2's slot unread (`--slots 4 --port-slots 0,1,3`, 4/3
   the image size). Measure both with the R0 probe before fixing the format.
4. ~~Ternary-8B quality ceiling~~ **done:** pooled NLL 2.438 vs binary-8B 2.516, but 1,633 MB per
   token at best packing vs 1,064 (9.5 vs 14.6 tok/s weights-only, DERIVED).
5. **R6 entry experiment** (contextual sparsity, `docs/10-big-kevin.md`): threshold sweep on
   the FFN `down` input, skipped fraction vs top-1 / ΔNLL, all layers (~15 min GPU). If R6 is
   taken up, `down` wants a column-major stream: decide before item 2 hardens the format.

GPU jobs must run one at a time on a 4 GB card.

## 5. R2 plan

- **R2a:** a weight packer that emits R0's port-interleaved layout, plus a streaming 1-bit ×
  BFP10 GEMV RTL across HP0/1/3. **Packer and RTL are done and gated bit-exact in iverilog**
  (own test cases plus real 8B rows, not yet the golden-vector files). Remaining: synth, then
  run it on silicon at line rate (§4 items 1–3). First Big-Kev silicon number.
  - Risk: ~330 weights/cycle at 333 MHz is aggressive for timing. Fallback is HP0–3 at 250 MHz
    (14.6 GB/s).
- **R2b:** fp32 vector unit, attention with KV in DDR, layer sequencing, board driver.

## 6. Board / production safety (only if you touch the Kria)

The KV260 also serves a live public chat demo.
- Before loading any experimental bitstream, stop the model rotation on the serving box
  (`systemctl --user stop kevin-rotation.service` on `precision`). It re-activates the
  production model over ssh every 5 min.
- Restore afterwards with `systemctl --user start kevin-rotation.service`, which reloads the
  production bitstream and reconnects the chat server. Then verify the public chat end to end.
- Never `pgrep -f a53_daemon` inside an ssh command line; it matches and kills your own shell.
- Get the owner's go-ahead before any of this. **None of tasks 1–4 needs the board.**
