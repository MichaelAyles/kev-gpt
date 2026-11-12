# Kevin on Kria

> *Why waste BRAM say lot word when few word do trick.*

A tiny language model that runs entirely inside the FPGA fabric of a Xilinx Kria
KV260: weights resident in on-chip BRAM/URAM so they never touch DDR, trained on
telegraphic text so it talks like Kevin Malone. The joke is the thesis: a model
small enough to live on-chip is necessarily dumb, and on-chip residency is the
*only* thing that beats the board's Arm cores. **Being dumb and being fast are
the same property.**

Talk to it live at [chat.mikeayles.com](https://chat.mikeayles.com):

![The live chat, straight from the fabric](webchat/kevchat.png)

**Record: 59,965.5 tok/s** (16 streams @ 200 MHz, MEASURED, 16/16 bit-exact, 3/3 runs).
**Faithful single stream: 19,242 tok/s** of fabric, about 27× a laptop GPU at the same batch size.

## The two facts it hangs on

1. **The bandwidth wall.** Single-stream decode is memory-bandwidth bound. On the
   KV260 the A53 cores and the PL fabric share one DDR4 controller (17 GB/s peak),
   so a DDR-resident model gets no uplift from the fabric. The only escape is
   keeping the whole model (~1–2 MB, INT4) in on-chip memory at hundreds of GB/s
   to TB/s.
2. **The fusion.** Telegraphic output = fewer tokens = less weight streamed and a
   smaller KV cache = more fits on-chip = faster. The compression is the comedy
   *and* the optimisation. The Keviniser strips the *training corpus*, not the
   input: the model learns to talk this way.

## Code

| dir | what |
|---|---|
| [`keviniser/`](keviniser/) | the Keviniser preprocessor, corpus harness and TinyStories fetcher ([README](keviniser/README.md)) |
| [`model/`](model/) | a 2–4M-param nanoGPT-scale model: trainer, sampler, INT4 QAT, and the integer reference ladder the RTL copies ([README](model/README.md)) |
| [`fabric/`](fabric/) | the RTL, the bit-exact gate harnesses, Vivado scripts, board drivers and engineering logs |
| [`webchat/`](webchat/) | the serving stack: A53 daemon, WebSocket backend, speculative-typing client, load dashboard ([README](webchat/demo/README.md)) |
| [`bench/`](bench/) | CPU/GPU reference benchmarks (ONNX Runtime, CUDA) for the comparison numbers |
| [`tests/`](tests/) | pytest suite (Keviniser, chat protocol, daemon) |

## The speed ladder

![Speed ladder](fabric/progress.png)

Every rung is **MEASURED on silicon**: 3/3 runs, token stream bit-exact against
the integer reference.

| What | tok/s |
|---|---|
| Single pass N=8: one weight pass serves all 8 streams (69,172 cyc / 8 tokens @ 166.7 MHz) | 19,275.6 |
| N=16: 12 DSP-packed banks + shared LN/attention (110,494 cyc / 16 tokens @ 166.7 MHz) | 24,134.0 |
| + softmax latency cut (103,582 cyc @ 166.7 MHz): **the stream ceiling** | 25,744.5 |
| **Split-brain** N=14: two cohorts on the dual-ported URAM (63,113 cyc / 14 tokens @ 166.7 MHz) | 36,970.7 |
| N=16 @ **200 MHz**: first 200-clean build | 46,604.4 |
| + schedule pipelining (56,876 cyc @ 200 MHz) | 56,262.7 |
| + TMAX=16 architectural wave (53,364 cyc @ 200 MHz) | **59,965.5** |

Past 25.7k, **N=16 is the stream ceiling**: three INT4×INT8 MACs per DSP is
provably impossible (27-bit port vs 28 needed;
[`dsp3_pack_proof.py`](fabric/stage3/research/dsp3_pack_proof.py), 1.2M trials). So
the levers became cycles and clock, not streams: split-brain (two N=8 cohorts on the
true-dual-port URAM) plus a worst-path retirement campaign. The 100k target was
chased and honestly disowned: the cycle floor and a 250 MHz hard wall (the fabric
hangs) put this architecture's ceiling on this chip at 62k–78k.

References (same model, batch 1, greedy): A53 = 11 tok/s, XPS 15 ONNX Runtime CPU =
1,273, RTX 3050 Ti = 719. The 16-stream aggregate beats the laptop GPU about 83×;
the fair single-stream comparison is the faithful N=1 build at 19,242 tok/s, about
27× the GPU.

## The two ceilings: fabric vs round trip

![The ceiling stack](fabric/ceilings.png)

What the fabric does and what a chat user feels are different numbers.

- **Fabric peak:** 59,965.5 tok/s (N=16 @ 200 MHz), pure PL cycles.
- **Faithful single stream** (N=1, the deployed chat build): 19,242 tok/s by counted
  cycles. A 2,000-connection bench holds ~21,300 tok/s across the sweep (peak 21,479,
  ~12,000 replies, zero errors).
- **Round trip:** bound by serving, not silicon. The biggest tax was sampling: the
  host read 193 logits per token back over `/dev/mem`. Moving it **on-chip via the
  Gumbel-max trick** (sampling `softmax(logit/T)` is exactly `argmax(logit + T·g)`,
  `g ~ Gumbel`, so the existing argmax hardware does it from one seed register per
  request) lifted the localhost round trip ~5.6× (1k → 5,600 tok/s) with the fabric
  still bit-exact. The public figure (~1,658 tok/s through the Cloudflare tunnel) is
  WAN and tunnel latency.

## What's in it

- **Keviniser:** strips on part-of-speech tags, not a stopword list, so it keeps
  main-verb "do" and drops auxiliary "do". On the full TinyStories train split
  (2,119,718 stories) it keeps **70.1%** of words (371.7M → 260.5M), ~67% of tokens:
  the compression the speed story rests on.
- **Model:** a 3.16M-param char-level GPT that goes from noise to coherent
  telegraphic Kevin in ~4 min on an RTX 3050 Ti, plus a Brevitas INT4-weight /
  INT8-activation QAT path that matches the FP model's loss within ~0.01 nats.
- **The reference ladder** (`model/goformer_*.py`, `fabric/stage3/seq_ref.py`): full
  integer forward → KV-cached decode → pinned fixed-point formats → fabric-precision
  non-linears → a per-phase sequencer reference that exposes every intermediate. It
  is the bit-true spec every RTL block is gated against.
- **Stage 3 on silicon:** the full forward (4 blocks + final LN + head + argmax) runs
  inside the PL with zero DRAM in the token loop: weights in URAM, activations and KV
  in BRAM. The live design is split-brain N=16; the 16 streams double as
  keystroke-speculative completions (every keypress forks a stream; Enter shows the
  precomputed answer). Engineering log:
  [`fabric/stage3/WIDE-WORD-DATAPATH-LOG.md`](fabric/stage3/WIDE-WORD-DATAPATH-LOG.md).
- **KV to DDR (simulation only, bit-exact):** `kv_dma` + `kv_prefetch` move the KV
  cache off-chip with double-buffered burst prefetch that hides DDR latency, to
  restore context beyond the on-chip window.

## Reproducing it

Three tiers, in increasing hardware cost. **Tiers 1–2 need no FPGA**, and tier 2 is
where anyone can check the bit-honest claim in software.

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Tier 1: the data tool and the model (CPU, or a GPU to go big)

```
# Kevinise the bundled sample
python -m keviniser.harness samples/canonical.txt
#  -> why us waste time say lot word when few word do trick

# Fetch TinyStories, Kevinise it, train a tiny model, sample it
python -m keviniser.fetch_tinystories                       # validation split (~20 MB)
python -m keviniser.harness data/TinyStories-valid.txt \
    -o data/TinyStories-valid.kevin.txt --marker "<|endoftext|>"
python -m model.train --max-iters 4000                      # ~37 min M1, ~4 min RTX 3050 Ti
python -m model.sample data/ckpt.pt --prompt "once upon time"

# Optional: INT4 QAT fine-tune from the FP checkpoint (pip install brevitas)
python -m model.train --qat --init-from data/ckpt.pt --max-iters 2000 --out data/ckpt.qat.pt
```

Training on the full corpus belongs on a CUDA GPU; see
[`model/SETUP-DELL.md`](model/SETUP-DELL.md). The Kevinised train corpus
(394 MB gzipped) is a release asset:
`gh release download corpus-v1 -R michaelayles/kev-gpt -D data/`.

### Tier 2: the bit-exact fabric gates (needs `iverilog`, no board)

Every RTL block is proven bit-identical (or cosine > 0.9999 for the transcendental
lookup tables) to the Python integer reference in
[Icarus Verilog](https://steveicarus.github.io/iverilog/) **before** any silicon or
speed number is trusted. Each `run_*.py` harness writes the weight images, compiles
the RTL with `iverilog -g2012`, runs it, diffs every intermediate against
`seq_ref.py`, and prints a one-line `*_VERDICT`.

```
# Debian/Ubuntu: sudo apt install iverilog   |   macOS: brew install icarus-verilog
python -m fabric.stage3.run_gelu             # GELU lookup table vs exact GELU
python -m fabric.stage3.run_layernorm        # fabric-precision LayerNorm
python -m fabric.stage3.run_softmax          # fabric-precision softmax
python -m fabric.stage3.run_banked           # the systolic INT4 GEMV, across PE widths
python -m fabric.stage3.run_vec_seq          # a full single-token forward through the sequencer
```

Scratch files go to `<system-temp>/kevbuild`; override with `--dir <path>` where a
harness takes it, or `KEV_SIM_DIR` for all of them.

### Tier 3: on the board (Kria KV260 + Vivado 2025.2)

Synthesis, implementation, bitstream and the on-board clock sweep need the hardware
and the Xilinx toolchain; the commands are in
[`fabric/stage3/WIDE-WORD-DATAPATH-LOG.md`](fabric/stage3/WIDE-WORD-DATAPATH-LOG.md).
Without the ~$250 board this tier isn't reproducible: the silicon numbers are ours,
but the method that makes them trustworthy (tier 2) runs on your laptop.

## How to read the Verilog (for software engineers)

The fabric is hand- and LLM-written RTL (me and Claude Code, no HLS). Verilog
describes **hardware that all exists at once**, not a sequence of instructions.
Three ideas get you most of the way, and the modules in
[`fabric/stage3/rtl/`](fabric/stage3/rtl/) are small enough to read.

**1. A module is a function whose arguments are wires of a fixed bit-width.** The
whole GELU activation is an 8192-entry lookup table
([`rtl/gelu_lut.sv`](fabric/stage3/rtl/gelu_lut.sv)):

```systemverilog
module gelu_lut (
    input  wire                clk,
    input  wire signed [15:0]  x,      // Q4.12 fixed-point in
    output reg  signed [15:0]  y       // Q4.12 fixed-point out
);
```

`[15:0]` is a 16-bit bus: no ints, floats or growable types, just piles of bits you
size yourself. `Q4.12` is fixed-point (4 integer bits, 12 fractional). Floats cost too
much fabric, so the whole model runs in scaled integers.

**2. `always @(posedge clk)` means "on every rising clock edge".** It is the only
notion of time. A `reg` updated inside one is a register that holds its value between
ticks; a `wire`/`assign` is combinational logic that settles continuously. The GELU
is a 3-stage pipeline: a new `x` enters every cycle and its `y` comes out 3 cycles
later, with three inputs in flight at once:

```systemverilog
always @(posedge clk) u  <= x + 16'h8000;          // stage 0: recenter the index
always @(posedge clk) begin                         // stage 1: two registered table reads
    l0 <= lut[idx];  l1 <= lut[idx1];  f1 <= frac;
end
always @(posedge clk) y <= l0 + (step >>> 3);       // stage 2: linear interpolation
```

`<=` schedules every register in the block to update *together* at the edge, so order
inside the block doesn't matter. `>>>` is an arithmetic shift (a cheap divide by 8).

**3. Parallelism is spatial: `generate`/`for` stamps out copies of hardware.** One
stream's 128 multiply-accumulate lanes
([`rtl/gemm_banked_resident_vec.sv`](fabric/stage3/rtl/gemm_banked_resident_vec.sv)):

```systemverilog
generate
    for (gl = 0; gl < LANES; gl = gl + 1) begin : g_l
        reg signed [ABITS-1:0] a;                   // this lane's accumulator
        always @(posedge clk) begin
            if (clr)      a <= 0;
            else if (en)  a <= a + $signed(w[gl*4 +: 4]) * x;   // a += w[gl] * x
        end
        assign acc[gl*ABITS +: ABITS] = a;
    end
endgenerate
```

That `for` is **not** a loop that runs 128 times; it lays down 128 physical
multiply-accumulate units that all fire on the same clock edge. `w[gl*4 +: 4]` means
"4 bits starting at bit `gl*4`": one INT4 weight sliced out of a packed bus. Speed on
the fabric comes from doing more of these per cycle and clocking them faster, which is
the whole speed ladder.

**Where it stops feeling like software: memory layout is physical.** A 2-D array
indexed by a runtime value (`buf[lane][row]`) synthesises to a giant per-lane mux tree
that overflows the chip; the same data as one wide word per row
(`reg [P*32-1:0] bank [0:ROWS-1]`, lane `l` at bits `[l*32 +: 32]`) makes the row a
memory *address* instead of a mux, and fits. That refactor, same maths in a different
physical shape, recurs throughout the repo (see
[`rtl/layernorm_vec.sv`](fabric/stage3/rtl/layernorm_vec.sv)). Keep `seq_ref.py` open
while reading the rest.

## Side quest: Big Kev

The same board running an 8B model the other way round: weights streamed from DDR
instead of held on-chip. PrismML's 1-bit
[Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) generates at **11.3 tok/s**
end to end, every token identical to a bit-true spec, with the fabric streaming the
1,064 MB of weights per token at 99% of the measured DDR ceiling. It lives on the
[`big-kev`](https://github.com/MichaelAyles/kev-gpt/tree/big-kev) branch; start at
[`fabric/big/HANDOFF.md`](https://github.com/MichaelAyles/kev-gpt/blob/big-kev/fabric/big/HANDOFF.md).

## Conventions

- **Mischief in the title, dry in the body.** The output prose is deliberately bad;
  the speed numbers carry the story.
- **Honest-first.** Every claim states where the approach loses. The roofline
  crossover (where the fabric stops winning) is a result, not a caveat.
- **Bit-honest before fast.** No speed number counts until the block behind it is
  bit-exact against the reference.

## License

[MIT](LICENSE) for the code in this repository. Third-party data and models keep
their own licences: the Kevinised corpus release is derived from TinyStories, and
the Bonsai-8B model used on the `big-kev` branch is PrismML's (Apache-2.0) and is
not included here.
