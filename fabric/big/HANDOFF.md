# Big Kev — handoff (2026-11-11, updated after R2a on silicon and the R3 study)

**Goal.** Run a 1-bit 8B LLM (PrismML **Bonsai-8B**, Qwen3-8B architecture) on a $250 Xilinx
Kria KV260, streaming its weights from DDR as fast as the bus allows. Single-stream decode is
memory-bandwidth bound, so **tok/s ≈ DDR bandwidth ÷ bytes per token**. This work is the
DDR-resident counterpart to the repo's on-chip "Kevin" model. Optimistic target **50 tok/s**: about
14 from line-rate streaming, then sub-1-bit weight compression (R3) and speculative decoding
with an on-chip draft model (R4).

Branch `big-kev` (off `main`, pushed to origin; not merged). Everything lives under `fabric/big/`. The design
docs (`docs/10-big-kevin.md`, `docs/big-kev-r0.md`, `-r1.md`, `-r2.md`, `-r3.md`) are **local-only**
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
| **R2a on silicon** weight stream + GEMV | **DONE, MEASURED, bit-exact** | One real token's 253 GEMVs (36 layers × q,k,v,o,gate,up,down + LM head) streamed from DDR at 333 MHz: **253/253 bit-exact vs the spec in 9 of 9 runs; 1,064.1 MB in 68.9–69.3 ms = 15.40 GB/s (median) = 14.47 tok/s weights-only (14.43–14.51)**. 250 MHz: 11.86 GB/s, 11.15 tok/s, bit-exact. Engine starved 3.7% of cycles (DDR is the limit, 96% of 3 beats/cycle). Dense 3-slot layout == R0's 4-slot stripe (15.41 vs 15.39 GB/s on 6 layers), so the format stays dense. Build: WNS −0.096 ns at 3.0 ns (3 endpoints, in the result-readback RAM cascade), 13.2k LUT, 19.7k FF, 37 BRAM tiles, 6 DSP. Host loads activations / reads results over AXI-Lite; that time is not in the number. |
| **R2b stage 1** hybrid (fabric GEMVs + A53 vector path) | **DONE, MEASURED, tokens identical** | `board/pl_token.py`: the torch spec on the A53 with the GEMV swapped for the PL. **240/240 generated tokens identical to the spec free-run** (3 prompts × 80), 253/253 GEMV inputs+outputs identical at the golden position. End to end **0.15–0.25 tok/s, host-bound** (Python + AXI-Lite); fabric 70–72 ms/token. A correctness baseline, not a speed result. |
| **R2b stage 2** hybrid with a C host | **DONE, MEASURED, tokens identical** | `host/bk_host.c` (spec's vector path in C; `HOST_VERDICT PASS` on x86, `HOST_REPLAY_VERDICT PASS` on the A53) + `board/pl_token_c.py`: **240/240 tokens identical, 1.16–1.21 tok/s** at context 23..103 with the v1 bitstream. Per token: fabric 73 ms, C host ~100 ms single-threaded (35 ms + 0.89 ms per cached position), AXI-Lite ~570 ms (1 µs per register write, 130 ns per word read), LM-head re-streams 51 ms. |
| **R2b stage 3** `gemv_axi` v2 on silicon | **DONE, MEASURED, tokens identical** | Activations fetched from DDR and results written to DDR by a small aux master on HP2, LM-head argmax in fabric. `GEMV_DMA_VERDICT PASS` (253/253 bit-exact, 4/4 runs, 333 MHz). Hybrid with the single-thread C host: **240/240 tokens identical, 5.24–5.53 tok/s end to end** at context 23..103 (fabric 71–76 ms, C host 92–98 ms, activation pack 12 ms, readback 4 ms); 3.4–4.3 tok/s at context ~130–240 because the host grows ~0.9 ms per cached position. Build: 14.2k LUT, WNS +0.229 ns **at 3.333 ns** (see §3: the BD builds are constrained at 300 MHz, the board runs them at 333). `results/r2b_hybrid_dma.json`, `results/impl_gemv_axi_v2.txt`. |
| **R2b** next | in progress | (1) C host tuning on the A53 (threads, attention kernel), (2) batch-K GEMV: several positions per weight pass (`gemv_stream` NB, register map VERSION 3), (3) the vector unit in fabric. Whole-token roofline **~14.0 tok/s** (DERIVED: 1,064 MB weights + ~44 MB KV per token at ctx 512). |
| **R4 entry** prompt-lookup drafting | **MEASURED (replay)** | `ref/r4_draft.py` over board-generated greedy transcripts (`results/r4_transcripts.json`): plain chat **1.05–1.12 tokens per pass** (959 tokens), copy-heavy answers **1.45–1.91** (446 tokens). The lookup drafter is worth ~1.1× on chat; R4's multiplier needs a real drafter. |
| **R6 entry** contextual sparsity | **MEASURED, parked** | `ref/r6_sparsity.py`: keeping 75% of FFN units per token = −12% bytes at ΔNLL +0.011 (top-1 95.0%); 50% = −24% at +0.091 (83.5%). Not sparse enough for a cheap 2× (`results/r6_sparsity.json`, table in doc 10 R6). |
| **R3a** scale compression (study) | **DONE, not in RTL** | `bfp3` (3-bit scales, 16-bit header per 32 groups): 1,064.1 → 975.4 MB/token (**−8.3%**, 15.93 tok/s weights-only DERIVED), ΔNLL +0.0019 ± 0.0018 on the R1c set, +0.0047 ± 0.0013 pooled with a second set, 97.6–98.0% argmax kept. Lossless dictionary: −4.8% but needs scale codes packed across rows. Never clamp the largest scales (+3.4 nats). |
| **R3b** sub-1-bit signs (study) | **KILL** | signs are incompressible (0.9979 bits/sign lossless); at ≤0.875 total bpw the best code costs +0.42 nats vs a 0.05 bar. Would need retraining with the code in the loop. So realistic R3 = R3a only, and R4's arithmetic is ~15.25 × speculative gain, not 50. |

Key facts you would otherwise rediscover:
- **KV260 DDR is DDR4-2133** (read from the DPLL/DDR_CTRL/MSTR registers), so the peak is
  17.07 GB/s, not 19.2.
- **HP1 and HP2 share a DDR controller port.** Use HP0+HP1+HP3 at 333 MHz (15.5 GB/s), or HP0–3
  at 250 MHz (14.6). The HPC ports only hurt. Far-apart per-port regions thrash DRAM rows
  (~10.8 GB/s); interleave instead.
- fclk0 cannot make 300 MHz. It snaps to 333.3 / 250 / 200 (and 500: at 500 MHz the DDR
  transport still checks out at ~16.0 GB/s but the GEMV arithmetic fails, as expected).
- **Weight image without a reboot:** 2 MB hugepages in a hugetlbfs file (`/dev/hugepages/bigkev`),
  physical addresses from `/proc/self/pagemap`, each matrix bump-allocated in a physically
  contiguous run (`board/physmem.py`). Packing the 8B on the Kria takes 95 s; later runs re-attach
  in ~20 s. Gotchas: physical page 0 is real RAM on ZynqMP; writing `nr_hugepages` below the
  in-use count marks pages *surplus* and makes later "grow by N" allocate nothing (`hp_settle`);
  a small pool comes back as scattered single pages, so ask for ≥ 256 pages and free the old
  pool before building a new one.
- **Hugepages MOVE.** They are movable allocations on arm64; ones the kernel placed in the CMA
  area (1000 MB at 0x37400000) are migrated by any CMA allocation, and `fpgautil -b` is one.
  `HugePool` rejects every page a trial long-term pin (io_uring fixed buffers) refuses, replaces
  it, and pins the rest. Keep the per-port image checksum check in any driver.
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
| `gemv_axi_regs.py` | **register map** of the board wrapper (single source for RTL, sim, driver) |
| `rtl/gemv_axi.v`, `rtl/axi_rd_stream.v`, `tb/tb_gemv_axi.sv`, `tb/axi_rd_mem_slave.sv`, `run_gemv_axi.py` | **R2a board wrapper**: AXI-Lite + NPORTS read masters with credit FIFOs (RREADY always 1) + result capture RAM + counters; iverilog gate (`GEMV_AXI_VERDICT`) |
| `tcl/build_gemv_axi.tcl` | BD + synth/impl/bitstream: `-tclargs <NPORTS> <MHz> <dir> [GMAX]` (~40 min) → `results/impl_gemv_axi.txt` |
| `ref/golden_token.py` | one whole token as a replay file: input (BFP10) and fp32 output of all 253 GEMVs |
| `board/physmem.py` | physically contiguous memory from hugepages (see the gotchas above) |
| `board/pl_gemv.py` | **R2a board driver**: packs the image on the Kria, replays the golden token, checks every result word + per-port image checksums, reports GB/s and tok/s (`GEMV_BOARD_VERDICT`) → `results/r2a_board.json` |
| `host/bk_host.c`, `host/bk_host.h`, `host/bk_host.py`, `run_host.py` | **C host kernel**: the spec's vector path, bit-identical on x86 and aarch64, GEMV through a callback; gate `HOST_VERDICT`; `python -m fabric.big.host.bk_host --selftest --replay ... / --bench` on the board. Ship the tables (`host_tables_8B.npz`): torch's fp32 pow for the RoPE frequencies is not reproducible from first principles. |
| `board/pl_token_c.py`, `board/mmio.c` | **the C-host hybrid** (`PL_TOKEN_C_VERDICT`) → `results/r2b_hybrid_c.json`; `mmio.c` = 32-bit-only register access (never memcpy on the AXI-Lite window) |
| `board/pl_token.py` | **the hybrid**: generate text with the fabric doing the GEMVs and the A53 running the spec's vector path (`PL_TOKEN_VERDICT`) → `results/r2b_hybrid.json`. Needs torch on the board (`~/bigkev/venv`). |
| `ref/r3a_scales.py`, `ref/r3b_signs.py` | R3 compression study → `results/r3a_scales.*`, `results/r3b_signs.*` |
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
  Without touching the connection (hotel wifi): run Vivado in a network namespace that holds a
  dummy interface with the wifi card's permanent address (`ethtool -P <wifi>`). FlexLM only sees
  an interface from `/proc/net/dev` there, and Vivado needs the system libudev preloaded:
  `sudo ip netns add vlic; sudo ip netns exec vlic ip link add eth0 type dummy;`
  `sudo ip netns exec vlic ip link set eth0 address <permanent MAC> up;`
  `sudo ip netns exec vlic sudo -u $USER -H env LD_PRELOAD=/lib64/libudev.so.1 bash -c 'source .../settings64.sh && vivado ...'`.
- **The BD clock is not what you asked for.** `build_gemv_axi.tcl` with FREQMHZ 333 gets
  300.03 MHz from the PS clock generator and Vivado constrains 3.333 ns. The board driver forces
  333.33 MHz, so silicon runs ~10% past the constraint (bit-exact so far). Pass the 5th tcl
  argument (e.g. `3.000`) to constrain implementation at the real period (added, not yet tried).
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
2. ~~R2a on silicon~~ **done** (§1): bit-exact on all 253 GEMVs of a real token, 15.40 GB/s,
   14.47 tok/s weights-only MEASURED. Reproduce: build with `tcl/build_gemv_axi.tcl`, copy the
   `.bin` to `~/kevbit/gemv_axi.bit.bin`, `sudo fpgautil -b` it, then on the Kria (tree at
   `~/bigkev`, GGUF + `Bonsai-8B-Q1_0_token.npz` in `~/bigkev/data`)
   `sudo python3 -m fabric.big.board.pl_gemv --fclk 333.333e6 --reps 6`.
3. ~~Stream layout decision~~ **done:** dense 3-slot == 4-slot stripe on silicon; format stays dense.
4. ~~Ternary-8B quality ceiling~~ **done:** pooled NLL 2.438 vs binary-8B 2.516, but 1,633 MB per
   token at best packing vs 1,064 (9.5 vs 14.6 tok/s weights-only, DERIVED).
5. ~~R6 entry experiment~~ **done, parked:** see §1 and doc 10 R6. If reopened, per-layer
   budgets first; `down` would want a column-major stream.

6. **R2b** (next): the rest of a token on the board. fp32 vector unit (RMSNorm, SiLU, residual,
   BFP10 requantise), attention with the KV cache in DDR (K BFP10 g16, V BFP8), RoPE, softmax,
   layer sequencer, argmax. Accept: tokens identical to `fabric_ref` free-run, then MEASURED
   whole-token tok/s. The R2a wrapper's activation RAM and result FIFO are the seams.
7. **R3a in RTL** if wanted: `bfp3` scale decode in `gemv_stream` (shift, 10-bit subtract,
   64-entry ROM) + packer change; the reference then becomes "Bonsai-8B with bfp3 scales".
   Still open on the study side: `bfp3` through the bit-true spec, free-run and long context.

GPU jobs must run one at a time on a 4 GB card.

## 5. R2 plan

- **R2a: done on silicon** (§1). The weight stream runs at the R0 ceiling; nothing left to gain
  there except bytes per token (R3a) and fewer full passes per token (R4).
- **R2b:** fp32 vector unit, attention with KV in DDR, layer sequencing, board driver.

## 6. Board / production safety (only if you touch the Kria)

The KV260 also serves a live public chat demo.
- Before loading any experimental bitstream, stop the model rotation on the serving box
  (`systemctl --user stop kevin-rotation.service` on `precision`). It re-activates the
  production model over ssh every 5 min.
- Restore afterwards with `systemctl --user start kevin-rotation.service`, which reloads the
  production bitstream and reconnects the chat server. Then verify the public chat end to end.
- Never `pgrep -f a53_daemon` inside an ssh command line; it matches and kills your own shell.
- `~/bigkev/borrow_pl.sh` on the Kria parks the demo daemon by PID and loads the Big Kev
  bitstream; `bash ~/activate_model.sh kevin` gives the board back (health probe must say OK).
- Check `ps` for `unattended-upgrade` before trusting a timing: it ran for over an hour on
  2026-11-11 and cost the weight stream 3–12%.
- Get the owner's go-ahead before any of this.
- When the serving box is down (as on 2026-11-11) the rotation is not running: kill the
  daemon by explicit PID before `fpgautil`, and restore with `ssh kria bash ~/activate_model.sh
  kevin` (reloads the production bitstream at 200 MHz, restarts the daemon, runs the health
  probe). The hugepage weight image survives that and costs the Kria 1.2 GB of RAM until freed
  (`pl_gemv --free`, or delete `/dev/hugepages/bigkev` and shrink `nr_hugepages`).
