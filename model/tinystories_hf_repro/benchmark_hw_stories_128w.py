"""Captures ~128-word real-hardware stories from the Genesys2 board over
UART, for the kev-gpt-vs-SauravP97/tiny-stories-19M length/quality
benchmark (see build_benchmark_128w_report.py).

REQUIRES a benchmark firmware build: kevgpt_interactive's normal chat
config caps replies at MAX_GEN_LEN=60 word tokens (~45-55 words) -- too
short for this benchmark. Both constants are `#ifndef`-guarded in
kevgpt_interactive/main.c specifically for this (search "BENCHMARK
OVERRIDE"); bump them, rebuild, resend weights, run this script, then
revert and repeat before returning the board to normal chat use:

    # 1. bump the two constants in
    #    ~/RVchatbot/kevgpt-genesys2-soc/sw/applications/kevgpt_interactive/main.c:
    #      #define MAX_GEN_LEN 124u     (was 60u)
    #      #define STORY_SENTENCES 20u  (was 8u)
    #    (124 leaves headroom under KEVGPT_TMAX=128 for a short prompt;
    #    20 sentences is generous enough that the token budget, not the
    #    sentence count, is the binding stop condition.)
    cd ~/RVchatbot/kevgpt-genesys2-soc
    make app PROJECT=kevgpt_interactive TARGET=genesys2

    # 2. start the weight+tokenizer listener FIRST, wait for it to print
    #    "waiting for KEVGPT_UART_READY", THEN trigger the GDB reload --
    #    reversing this order causes SEND_WEIGHTS_FAIL. If
    #    tokenizer_ddr_v16384.bin doesn't exist, regenerate it first:
    #      python -m fabric.genesys2.gen_chat_fw \\
    #        --npz fabric/export_word16384/goformer.npz --meta data/word_v16384/meta.json \\
    #        --prompt "once upon a time" --ngen 4 --lanes 64 --p 8 \\
    #        --out /tmp/scratch_weights_header.h --tokenizer-ddr \\
    #        --tokenizer-blob-out ~/RVchatbot/kevgpt-genesys2-soc/tokenizer_ddr_v16384.bin
    python -m fabric.genesys2.send_weights \\
        --port /dev/ttyUSB0 --npz fabric/export_stepC_d128_v16384/goformer.npz \\
        --tokenizer-blob ~/RVchatbot/kevgpt-genesys2-soc/tokenizer_ddr_v16384.bin
    # (in a separate terminal, once the listener above is confirmed ready)
    riscv32-corev-elf-gdb -batch -ex "target remote :3333" \\
        -ex "monitor reset halt" -ex "load" -ex "monitor resume" -ex "detach" \\
        ~/RVchatbot/kevgpt-genesys2-soc/hw/vendor/esl_epfl_x_heep/sw/build/main.elf

    # 3. run this capture
    python model/tinystories_hf_repro/benchmark_hw_stories_128w.py \\
        --out model/tinystories_hf_repro/benchmark_128w_hw.json

    # 4. revert MAX_GEN_LEN/STORY_SENTENCES to 60u/8u in main.c, rebuild,
    #    resend weights, reload -- same two steps as above -- to restore
    #    the board to its normal-chat configuration before real use.

Methodology matches this directory's own established convention
(hw_vs_sw_report.html / sweep_tiny_stories_hf.py): 5 prompts x 5 repeats
= 25 samples, normal on-chip Gumbel-max sampling (no seed control --
each turn's real captured seed is recorded), objective detector
model.filter_synth_corpus.is_degenerate.

min_bigram_recurrence defaults to 6, not the 3 used for ~60-word
samples elsewhere in this directory -- the bigram-recurrence detector
saturates at ~2x the length (more tokens = more chances for ANY
bigram, even an ordinary one, to recur 3+ times by chance, independent
of real degeneracy). 6 was picked by inspecting this benchmark's own
actual max-bigram-count distribution: hardware samples cleanly split
into a 2-4 bulk (incidental reuse) and a 6/8/10/12 tail (genuine
repetition loops) -- 6 is the natural cut between them. See
build_benchmark_128w_report.py's own caveat text for the full
rationale and the reference-model distribution that confirms it.
"""
import argparse
import json
import re
import sys
import time

sys.path.insert(0, "/home/tparng/kev-gpt")
import serial
from fabric.genesys2.chat_over_uart import send_prompt
from model.filter_synth_corpus import is_degenerate

PORT_DEFAULT = "/dev/ttyUSB0"
BAUD = 115200
PROMPTS = [
    "once upon a time",
    "the sun was",
    "the dog ran",
    "a little girl",
    "she found a",
]
SEED_RE = re.compile(r"KEVGPT_DEBUG_SEED,(0x[0-9a-f]+),cyc=0x[0-9a-f]+\n?")


def capture(port, repeats, quiet_secs, timeout, min_bigram_recurrence=6):
    results = []
    with serial.Serial(port, BAUD, timeout=0.3) as ser:
        for rep in range(1, repeats + 1):
            for prompt in PROMPTS:
                raw, elapsed = send_prompt(ser, prompt, quiet_secs, timeout)

                m = SEED_RE.search(raw)
                real_seed = m.group(1) if m else None
                body = raw[m.end():] if m else raw
                body = body.split("KEVGPT_WEIGHT_CRC", 1)[0].lstrip("\r\n")

                # print_word_token() (kevgpt_interactive/main.c) suppresses
                # the leading space on a reply's very first token
                # unconditionally -- reinsert one unless the first char is
                # punctuation that wouldn't have gotten a leading space
                # anyway (same is_punct/apostrophe exemption it uses).
                if body and body[0] not in ".,!?;:\"-'":
                    full_text = (prompt + " " + body).strip()
                else:
                    full_text = (prompt + body).strip()

                n_words = len(full_text.split())
                gen_words = max(n_words - len(prompt.split()), 0)
                rate = round(gen_words / elapsed, 1) if elapsed > 0 else 0.0
                reason = is_degenerate(full_text, min_bigram_recurrence=min_bigram_recurrence)

                sample = {
                    "seed": rep, "real_seed": real_seed, "prompt": prompt,
                    "text": full_text, "reason": reason,
                    "elapsed_s": round(elapsed, 3), "rate_tok_s": rate,
                    "word_count": n_words,
                }
                results.append(sample)
                print(f"=== rep={rep} prompt={prompt!r} seed={real_seed} "
                      f"words={n_words} reason={reason!r} ===")
                print(full_text[:200])
                sys.stdout.flush()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quiet-secs", type=float, default=4.0,
                     help="longer than the default chat_over_uart 3.0s -- "
                          "~124-token replies take longer to finish streaming")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--min-bigram-recurrence", type=int, default=6,
                     help="see module docstring for why this isn't the 3 used "
                          "for shorter (~60-word) samples elsewhere")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    results = capture(a.port, a.repeats, a.quiet_secs, a.timeout, a.min_bigram_recurrence)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    flagged = sum(1 for s in results if s["reason"])
    avg_words = sum(s["word_count"] for s in results) / len(results)
    print(f"\nDONE,total={len(results)},flagged={flagged},avg_words={avg_words:.1f}")
    print(f"WROTE,{a.out}")


if __name__ == "__main__":
    main()
