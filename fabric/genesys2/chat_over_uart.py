"""Sends one or more prompts to kevgpt_interactive's real-hardware chat
loop over the console UART, captures the generated reply, and reports
per-prompt wall-clock timing / word count / tokens-per-second.

Completion detection: `main.c`'s `chat_turn()` streams each generated
word out as it's produced (no fixed-length reply, no explicit "done"
marker), so this waits for a QUIET PERIOD on the wire (no new bytes for
`--quiet-secs`) rather than string-matching the trailing `"> "` prompt
literal. A naive "wait for `> `" check was tried first
(fabric/genesys2/PORT-NOTES.md "five more stories + measured token
rate") and silently never fired -- every run just idled out at the
outer deadline instead of detecting real completion, which is why this
script uses the quiet-period heuristic instead: robust to exact
byte-pattern/decoding edge cases, and confirmed correct against full
byte-timestamp instrumentation on a real run (the whole reply streams
continuously with no gaps, then genuinely stops).

Assumes the board is already programmed and `kevgpt_interactive` has
already loaded its weight image (+ tokenizer blob, for DDR3-resident-
tokenizer builds) and reached `KEVGPT_INTERACTIVE_READY` -- see
PORT-NOTES.md's "VOCAB=16384: clean repeatable sequence" for the full
bring-up chain this assumes already happened. Baud is 115200 (this
project's own `UART_BAUDRATE`, NOT the 9600 some other X-HEEP configs
on this board use).

    python -m fabric.genesys2.chat_over_uart --port /dev/ttyUSB0 \\
        --prompt "once upon a time" --prompt "the dog ran"
"""
from __future__ import annotations

import argparse
import time


BAUDRATE = 115200


def send_prompt(ser, prompt: str, quiet_secs: float, hard_timeout_s: float):
    """Sends one prompt + newline, reads until `quiet_secs` of silence on
    the wire, returns (raw_text, total_elapsed_s)."""
    ser.reset_input_buffer()
    t0 = time.time()
    ser.write((prompt + "\n").encode())
    ser.flush()

    buf = b""
    last_byte_time = t0
    hard_deadline = t0 + hard_timeout_s
    while time.time() < hard_deadline:
        chunk = ser.read(ser.in_waiting or 1)
        now = time.time()
        if chunk:
            buf += chunk
            last_byte_time = now
        elif buf and (now - last_byte_time) > quiet_secs:
            break
    return buf.decode(errors="replace"), last_byte_time - t0


def count_reply_words(text: str, prompt: str) -> int:
    """Strips the echoed prompt and the trailing "> " turn marker, counts
    the remaining whitespace-separated words -- an approximation of
    generated TOKEN count (this is the word-level tokenizer's own unit,
    so word count IS token count for this deployment)."""
    reply = text[len(prompt):] if text.startswith(prompt) else text
    reply = reply.rsplit(">", 1)[0]
    return len(reply.split())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.genesys2.chat_over_uart")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=BAUDRATE)
    ap.add_argument("--prompt", action="append", required=True,
                     help="a prompt to send; repeat --prompt for multiple, sent "
                          "sequentially over one connection (one chat_turn() each)")
    ap.add_argument("--quiet-secs", type=float, default=3.0,
                     help="consider a reply complete after this many seconds "
                          "with no new bytes on the wire")
    ap.add_argument("--timeout", type=float, default=60.0,
                     help="hard per-prompt deadline (safety net if the board "
                          "never goes quiet)")
    a = ap.parse_args(argv)

    import serial  # deferred: only needed for this real-hardware tool

    results = []
    with serial.Serial(a.port, a.baud, timeout=0.3) as ser:
        for prompt in a.prompt:
            text, elapsed = send_prompt(ser, prompt, a.quiet_secs, a.timeout)
            n_words = count_reply_words(text, prompt)
            rate = n_words / elapsed if elapsed > 0 else 0.0
            results.append((prompt, elapsed, n_words, rate, text))
            print(f"=== prompt={prompt!r} elapsed={elapsed:.3f}s "
                  f"words={n_words} rate={rate:.1f} tok/s ===")
            print(text.strip())
            print()

    print("--- SUMMARY ---")
    for prompt, elapsed, n_words, rate, _ in results:
        print(f"{prompt!r}: {elapsed:.3f}s, {n_words} words -> {rate:.1f} tok/s")
    if results:
        avg_words = sum(r[2] for r in results) / len(results)
        avg_time = sum(r[1] for r in results) / len(results)
        avg_rate = sum(r[3] for r in results) / len(results)
        print(f"average: {avg_words:.1f} words / {avg_time:.3f}s = {avg_rate:.1f} tok/s")


if __name__ == "__main__":
    main()
