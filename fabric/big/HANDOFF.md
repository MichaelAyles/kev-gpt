# Big Kev — handoff (2026-11-10)

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
| **R1c** bit-true fabric spec | **PASS** | **99.41% top-1**, ΔNLL −0.0012, cos 0.99997 vs float over 1,020 tokens; step()==forward_seq() bit-identical |
| R1 leftovers (§3) | open | long context, golden vectors, free-run check, ternary comparison |
| **R2** streaming engine on silicon | not started | roofline **~14.0 tok/s** (DERIVED: 1,064 MB weights + ~39 MB KV per token at ctx 512) |

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
- **K cache needs an exponent per 16 channels.** Qwen's post-RoPE K has outlier channels; BFP8
  with one exponent per 128 dropped agreement to 96.3%.

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
| `results/` | raw R0 sweeps, R1b sweep, R1c gate, cached float chat text (`Bonsai-8B-Q1_0_chatgen_T256.json`, which took 35 min to make) |

The fabric spec in one breath: the GEMV input goes to BFP10 per 128; per group an exact integer
Σ sign·q (17 b) is multiplied by the fp16 scale split as an 11-bit integer × 2^e, then shifted
into a 64-bit accumulator with LSB 2^−40 (peak use 56 bits), with one RNE to fp32 at the end. The
vector path is IEEE fp32 with **lane-strided reductions** (lane = i % 16, sequential per lane,
then a pairwise tree), table exp (4097 entries) and rsqrt (1024 entries + 2 Newton steps),
streamed RoPE tables, and KV as BFP8 (K: exponent per 16 channels; V: per 128).

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

Sanity check that the setup works (about 3 min on the laptop GPU):
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m fabric.big.ref.r1c_gate`
→ `R1C_VERDICT PASS`.

## 4. Open tasks (R1 leftovers), with acceptance criteria

GPU tasks (1, 3, and the eval part of 4) must run one after another on a 4 GB card. Task 2 is
mostly code and runs alongside anything. Task 4's downloads run alongside anything.

### Task 1 — Long-context check (~1–1.5 h, GPU)
The R1c gate ran at T = 256. KV quantisation and softmax error grow with context.
- Add **chunked prefill** to `FabricRef`: `_layers()` already accepts `kv_hist` and a `valid`
  mask, so feed queries in chunks of ~128 with the growing K/V history. The full-T attention
  temporaries (`[32, T, T, 16]` fp32) don't fit 4 GB at T = 1024.
- Compare against the float reference (`Qwen3Ref.forward_seq` is fine to T ≈ 2048) on 2–3 long
  texts at T = 1024 and 2048. Report top-1 / ΔNLL / cos **by position bucket** (0–256,
  256–512, …).
- **Accept:** pooled top-1 ≥ 0.99 at each length, and no downward trend across buckets. If it
  trends down, ablate K/V format first (`FabricRef.K_FMT` / `V_FMT`), then softmax.
- Also confirm chunked prefill == `forward_seq` bit-identical at T = 256 (same invariant as
  step == seq).
- Write results to `results/r1_longctx.json`.

### Task 2 — Golden vectors for RTL (~2 h, mostly code)
The bridge to R2. The analogue of `fabric/stage3/seq_ref.py: block0_phase_signals()`.
- Add a trace option to `FabricRef._layers` that records, per layer and per position, every
  intermediate the RTL will produce:
  - embed row;
  - attn_norm out;
  - **BFP10 quantised GEMV input (q mantissas + exponents)**;
  - q/k/v GEMV outputs;
  - q_norm/k_norm;
  - roped q/k;
  - quantised K/V (mantissas + exponents);
  - scores;
  - softmax p;
  - ctx;
  - o_proj out;
  - residual;
  - ffn_norm;
  - gate / up;
  - silu·up (and its BFP10 quantisation);
  - down out;
  - residual;
  - final norm;
  - logits / argmax.
- Also dump per-GEMV **per-group integer partials** (inner sums and shifted terms) for at least
  one matrix, so the RTL accumulator can be debugged term by term.
- Export as `.mem` hex in a directory layout one RTL testbench can `$readmemh` (one file per
  signal, a fixed bit-width per signal documented in a `MANIFEST.json`). fp32 values as raw IEEE
  bits.
- Default scope: Bonsai-8B, a short chat prompt (~16 tokens), layers 0 and 35 in full, plus
  final logits. Make layers and positions arguments.
- Mind the iverilog traps in CLAUDE.md: flat ROMs + ranged `$readmemh`, never a 2D-array row.
- **Accept:** a Python round-trip (re-read the `.mem` files, recompute layer 0 from them with the
  spec functions) reproduces the recorded outputs bit-exactly. Commit the generator; don't
  commit large vector dumps (put them under `/tmp/kevbuild/big_vectors/` or similar).

### Task 3 — Free-running sanity (~30 min, GPU)
Teacher forcing can hide drift.
- Greedy-generate 64–100 tokens with `FabricRef` (per-token `step()` is ~2.3 s/token on the
  laptop GPU) on 3 chat prompts.
- Find the first divergence from float greedy cheaply: take the fabric output, run
  `Qwen3Ref.forward_seq` on it once, and see where float's argmax differs.
- Accept if the text stays coherent. Report divergence positions and a sample in
  `results/r1_freerun.json`.

### Task 4 — R1d: ternary vs binary at equal bytes (~2–3 h; downloads + CPU work + ~20 min GPU)
Decision-affecting. Ternary (−1, 0, +1) costs ~1.6–2.1 bits/weight but is often better per bit
than binary. At equal bytes/token (≈ equal tok/s) does **Ternary-Bonsai-4B** beat
**binary Bonsai-8B**?
- Download `prism-ml/Ternary-Bonsai-4B-gguf` (and `-8B-gguf` for the quality ceiling), plus
  `Ternary-Bonsai-4B-unpacked` to pin the GGUF block layout, the same way Q1_0 was pinned
  (signs/zeros agree 100%, values equal after bf16 rounding).
- Extend `gguf_q1.py` with a dequant for their ternary type, and a packed-ternary matrix class
  for `qwen3_ref.py` (or dequant to dense on load if memory allows).
- Evaluate **float-reference NLL** for binary-8B, ternary-4B and ternary-8B on the same texts
  (`r1b_sweep.eval_texts` + the cached chat). Report NLL, bytes/token (exact, from tensor
  sizes, excluding the embedding), and DERIVED tok/s at 15.54 GB/s.
- Note the bits/weight as stored and the best achievable packing (5 trits/byte = 1.6 bpw).
- Output `results/r1d_ternary.json`. **Decision rule:** if ternary-4B NLL < binary-8B NLL at
  ≤ the same bytes/token, flag it. The model choice for R2 is the human's call.

## 5. After R1: R2 plan (not started)

- **R2a:** a weight packer that emits R0's port-interleaved layout, plus a streaming 1-bit ×
  BFP10 GEMV RTL across HP0/1/3. Gate it bit-exact in iverilog against task 2's vectors, then
  run it on silicon at line rate. First Big-Kev silicon number.
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
