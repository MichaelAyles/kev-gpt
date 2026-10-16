"""Sends a trained checkpoint's weight image to kevgpt_interactive's UART
loader (fabric/genesys2/PORT-NOTES.md "interactive UART chat" pass) --
replaces the GDB `restore ... binary` step every earlier real-hardware
bring-up this port used, so a chat session only needs a serial cable, not
a live JTAG/OpenOCD/GDB session driving every run.

Protocol (matches sw/applications/kevgpt_interactive/main.c's receive
loop exactly, both sides hand-written since there's no existing UART
bulk-transfer protocol anywhere in this project to reuse):
  1. Host waits for firmware's own "KEVGPT_UART_READY" marker line (same
     KEVGPT_* marker convention every other firmware app already prints)
     before sending anything -- avoids a race where the host starts
     transmitting before the board has booted far enough to read it.
  2. Host sends a 4-byte little-endian word count, then that many 4-byte
     little-endian words -- the exact same wrom_to_words() packing
     (low-chunk-first, one 32-bit chunk per WBITS/32 split) every other
     weight-loading path in this project already uses (kevgpt_ddr_stage,
     kevgpt_wld_bringup, kevgpt_optionB_bringup, kevgpt_chat's own
     firmware-embedded array) -- reused, not re-derived.
  3. Host waits for "KEVGPT_UART_LOAD_DONE" to confirm the DMA reload
     (weight_loader_ddr, triggered by the firmware after the transfer)
     completed before declaring success.

    python -m fabric.genesys2.send_weights \\
        --npz fabric/export_optionB/goformer.npz --lanes 64 --p 8 \\
        --port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
import time

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_wideword
from fabric.genesys2.gen_chat_fw import wrom_to_words

BAUDRATE = 115200  # matches x-heep.h's UART_BAUDRATE -- every other real-hardware
                    # bring-up this session used the same rate, not a new choice.
READY_MARKER = b"KEVGPT_UART_READY"
DONE_MARKER = b"KEVGPT_UART_LOAD_DONE"
# DDR3-resident tokenizer table (PORT-NOTES.md "VOCAB=16384 runbook"): a
# second blob transfer, same wire protocol, distinct markers so the host can
# tell which phase main.c is in -- see main.c's uart_load_blob() call sites.
TOK_READY_MARKER = b"KEVGPT_UART_TOK_READY"
TOK_DONE_MARKER = b"KEVGPT_UART_TOK_DONE"


def weight_words_from_checkpoint(npz, lanes, p):
    goformer_p, cfg = seq_ref.build(npz)
    nlayer = len(goformer_p["blocks"])
    iseq = seq_ref.IntSequencer(goformer_p, cfg)
    with tempfile.TemporaryDirectory() as td:
        write_mems_wideword(td, iseq, lanes, nlayer, p)
        return wrom_to_words(os.path.join(td, "wrom.mem"), lanes)


def _wait_for_marker(ser, marker, timeout_s, pre=b"", echo=True):
    """Waits for `marker` to appear on the wire. `pre` seeds the search buffer
    with bytes already read by a PRIOR call (see the leftover-bytes bug this
    fixes, below) -- returns (found, leftover), where `leftover` is whatever
    arrived AFTER the marker in the same read, to be threaded into the NEXT
    call's own `pre` so it isn't silently dropped.

    Real bug this fixes (PORT-NOTES.md "VOCAB=16384 runbook", DDR3 tokenizer
    real-hardware bring-up): main.c prints KEVGPT_UART_LOAD_DONE immediately
    followed by KEVGPT_UART_TOK_READY with no work in between, so both often
    land in the SAME ser.read() chunk. The original version's `buf` was local
    to each _wait_for_marker call and discarded on return -- bytes after the
    found marker (i.e. the NEXT marker, already on the wire) were echoed for
    visibility but never handed to the next wait, which started from an empty
    buffer and timed out waiting for a marker that had already arrived and
    been silently dropped. First real-hardware run of the two-blob (weights +
    tokenizer) protocol hit this exactly: SEND_WEIGHTS_FAIL,timeout_waiting_
    for_KEVGPT_UART_TOK_READY even though the marker was visibly already in
    the echoed output, one call earlier."""
    deadline = time.time() + timeout_s
    buf = pre
    if marker in buf:
        idx = buf.index(marker) + len(marker)
        return True, buf[idx:]
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        if echo:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        buf += chunk
        if marker in buf:
            idx = buf.index(marker) + len(marker)
            return True, buf[idx:]
        # keep the tail bounded -- markers are short, no need to accumulate forever
        if len(buf) > 4096:
            buf = buf[-len(marker):]
    return False, buf


def _send_blob(ser, words, ready_marker, done_marker, ready_timeout_s, done_timeout_s, pre=b""):
    print(f"waiting for {ready_marker.decode()} ...")
    found, pre = _wait_for_marker(ser, ready_marker, ready_timeout_s, pre=pre)
    if not found:
        print(f"SEND_WEIGHTS_FAIL,timeout_waiting_for_{ready_marker.decode()}")
        return False, pre

    print(f"\nsending {len(words)} words ({len(words) * 4} bytes) ...")
    ser.write(struct.pack("<I", len(words)))
    payload = b"".join(struct.pack("<I", w) for w in words)
    ser.write(payload)
    ser.flush()

    print(f"waiting for {done_marker.decode()} ...")
    found, pre = _wait_for_marker(ser, done_marker, done_timeout_s, pre=pre)
    if not found:
        print(f"SEND_WEIGHTS_FAIL,timeout_waiting_for_{done_marker.decode()}")
        return False, pre
    return True, pre


def send(port, weight_words, ready_timeout_s=30.0, done_timeout_s=30.0, tokenizer_words=None):
    """Sends the weight image, then -- if `tokenizer_words` is given (PORT-
    NOTES.md "VOCAB=16384 runbook", DDR3-resident tokenizer table) -- a
    second blob over the same connection, using distinct READY/DONE markers
    (main.c calls uart_load_blob() twice at boot, weights then tokenizer).
    The leftover bytes after each found marker are threaded into the next
    _send_blob call (see _wait_for_marker's own docstring) so a marker that
    arrives packed together with the previous one is never silently missed."""
    import serial  # deferred: only needed for this real-hardware tool

    with serial.Serial(port, BAUDRATE, timeout=0.2) as ser:
        ok, pre = _send_blob(ser, weight_words, READY_MARKER, DONE_MARKER,
                              ready_timeout_s, done_timeout_s)
        if not ok:
            return False

        if tokenizer_words is not None:
            ok, pre = _send_blob(ser, tokenizer_words, TOK_READY_MARKER, TOK_DONE_MARKER,
                                  ready_timeout_s, done_timeout_s, pre=pre)
            if not ok:
                return False

        print("\nSEND_WEIGHTS_PASS")
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.genesys2.send_weights")
    ap.add_argument("--npz", default="fabric/export_optionB/goformer.npz")
    ap.add_argument("--lanes", type=int, default=64)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--port", required=True)
    ap.add_argument("--ready-timeout", type=float, default=30.0)
    ap.add_argument("--done-timeout", type=float, default=30.0)
    ap.add_argument("--tokenizer-blob", default=None,
                     help="path to a DDR3 tokenizer blob (fabric.genesys2.gen_chat_fw "
                          "--tokenizer-ddr --tokenizer-blob-out) to send after the "
                          "weight image, word-level DDR3-tokenizer builds only")
    a = ap.parse_args(argv)

    weight_words = weight_words_from_checkpoint(a.npz, a.lanes, a.p)
    tokenizer_words = None
    if a.tokenizer_blob:
        with open(a.tokenizer_blob, "rb") as f:
            blob = f.read()
        assert len(blob) % 4 == 0, "tokenizer blob must be a whole number of 32-bit words"
        tokenizer_words = list(struct.unpack(f"<{len(blob) // 4}I", blob))
    ok = send(a.port, weight_words, a.ready_timeout, a.done_timeout, tokenizer_words)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
