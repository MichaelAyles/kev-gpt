"""Generate the embedded weight image + expected-token self-check data for
sw/applications/kevgpt_chat/ (the Genesys2 port's Phase 5 firmware).

There is no host/DDR path on this board (unlike the KV260's A53/Linux daemon),
so the trained weight image has to be baked into the firmware image itself and
streamed into xheep_kevgpt_peripheral's W_DATA register at boot, exactly the
way fabric/stage3/tb/tb_seq_vec_kv.sv streams wrom.mem into wl_we/wl_data in
simulation (see that testbench's main initial block): one wl_rst pulse, then
every wide word split into WBITS/32 32-bit chunks, LOW chunk first
(`wword[s*32 +: 32]` for s=0..SUBW-1).

This script reuses the exact same code path fabric.stage3.run_vec_kv gates
against (seq_ref.build + write_mems_wideword + IntKVQSequencer.generate_greedy
at kbits=8/vbits=8/rotate=False/divfree=True) so the embedded expected-token
array is the identical golden reference the RTL gate is bit-exact-verified
against -- not a separately-computed value that could silently drift from it.

    python -m fabric.genesys2.gen_chat_fw \\
        --npz fabric/export_optionA/goformer.npz --meta data/char_optionA/meta.json \\
        --prompt once --ngen 6 --lanes 128 --p 8 \\
        --out ~/RVchatbot/kevgpt-genesys2-soc/sw/applications/kevgpt_chat/kevgpt_weights.h
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_wideword
from model.goformer_kvq import IntKVQSequencer

# Byte offset into DDR3 (ext_slaves-window-relative -- the same address space
# WEIGHTS_DDR_BASE/KV_DDR_BASE describe from the accelerator's own side) where
# the DDR3-resident tokenizer table (build_tokenizer_ddr_blob() below) is
# staged. PORT-NOTES.md "VOCAB=16384 runbook": VOCAB=16384's ~182KB firmware-
# baked tokenizer string table didn't fit in on-chip BRAM alongside that
# VOCAB's own much bigger weight/KV-cache BRAM need (RAMB36E1 overflow at
# `place_design`, no SoC-memory resize closes the gap) -- moved the table to
# DDR3 instead, read back through the same already-verified cpu_ddr_bridge/
# ext_slaves CPU-DDR3 path uart_load_weights() already uses for the weight
# image. 16MB clears WEIGHTS_DDR_BASE=0's own image (10,682,368 bytes at
# VOCAB=16384) and KV_DDR_BASE=12,582,912's own region
# (12,582,912+589,824=13,172,736) with ~2.8MB of margin -- MUST be rechecked
# against those two on every VOCAB/NLAYER change, same "recompute from the
# real staged image size" discipline KV_DDR_BASE's own history already
# established.
TOKENIZER_DDR_BASE = 16 * 1024 * 1024


def _encode(meta_path, text):
    m = json.load(open(meta_path))
    if m.get("tokenizer") == "word":
        from model.word_data import tokenize, UNK
        toks = tokenize(text.lower()) or [UNK]
        return [m["stoi"].get(t, m["stoi"][UNK]) for t in toks]
    return [m["stoi"].get(c, 0) for c in text]


def wrom_to_words(wrom_path: str, lanes: int) -> list[int]:
    """Each wrom.mem line is `lanes` hex chars (lanes*4-bit wide word). Split
    into 32-bit chunks, low chunk first -- matches tb_seq_vec_kv.sv's
    `wword[s*32 +: 32]` streaming order exactly."""
    subw = (lanes * 4) // 32
    words = []
    with open(wrom_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            wide = int(line, 16)
            for s in range(subw):
                words.append((wide >> (32 * s)) & 0xFFFFFFFF)
    return words


def emit_tokenizer_header_word(out_path, meta_path):
    """Word-level counterpart to emit_tokenizer_header() below (PORT-NOTES.md
    "word-level vocabulary"): itos is an array of C string literals (not a
    char[] -- each token can be a whole word) and there is no separate stoi
    table at all. model.word_data.build_vocab() guarantees ids 2..VOCAB-1 are
    alphabetically sorted among themselves (id 0=<unk>, id 1=<eos> are the
    only two exceptions, both handled specially by the firmware, never looked
    up by string) -- so kevgpt_word_lookup() in main.c binary-searches
    kevgpt_itos[2..] directly by strcmp instead of needing its own index.
    """
    m = json.load(open(meta_path))
    itos = {int(k): v for k, v in m["itos"].items()}
    vocab = m["vocab_size"]
    assert m.get("tokenizer") == "word"

    # Derived by scanning itos for the UNK/EOS strings, NOT hardcoded 0/1:
    # a checkpoint trained before model.word_data.build_vocab()'s EOS-
    # exclusion fix can have EOS at a different id (id=1 left dead/unused) --
    # see that function's own docstring. This works correctly either way.
    from model.word_data import UNK, EOS
    unk_id = next(i for i, w in itos.items() if w == UNK)
    eos_id = next(i for i, w in itos.items() if w == EOS)
    # id=1 may be genuinely absent from itos (the pre-fix bug above) -- the
    # emitted array below must still cover every id 0..vocab-1 contiguously
    # (a plain C array, no holes allowed), so fabricate a placeholder string
    # for any missing slot rather than crashing on the KeyError.
    missing = [i for i in range(vocab) if i not in itos]

    stop_words = (".", "!", "?")
    stop_ids = [i for i in range(vocab) if i not in (unk_id, eos_id) and itos.get(i) in stop_words]

    def _c_str_literal(s):
        out = []
        for ch in s:
            assert ord(ch) < 128, (
                f"itos contains non-ASCII token {s!r} (char U+{ord(ch):04x}) -- "
                "kevgpt_itos entries are plain C string literals and can't "
                "represent it; clean the vocab instead of guessing here"
            )
            if ch == '"' or ch == '\\':
                out.append("\\" + ch)
            else:
                out.append(ch)
        return '"' + "".join(out) + '"'

    with open(out_path, "w") as f:
        f.write("/* Auto-generated by fabric/genesys2/gen_chat_fw.py "
                "(emit_tokenizer_header_word) -- do not hand-edit.\n")
        f.write(" * Regenerate after retraining/re-exporting the word-vocab checkpoint. */\n")
        f.write("#ifndef KEVGPT_TOKENIZER_H\n#define KEVGPT_TOKENIZER_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define KEVGPT_TOKENIZER_WORD 1\n")
        f.write(f"#define KEVGPT_VOCAB_SIZE {vocab}u\n")
        f.write(f"#define KEVGPT_UNK_ID {unk_id}u\n")
        f.write(f"#define KEVGPT_EOS_ID {eos_id}u\n\n")

        f.write("/* ids 2..VOCAB_SIZE-1 are alphabetically sorted (build_vocab's own\n"
                 " * invariant) -- kevgpt_word_lookup() in main.c relies on this. Any id\n"
                 " * absent from itos (a checkpoint trained before the EOS-exclusion fix\n"
                 " * in model.word_data.build_vocab can have a dead, never-trained slot\n"
                 " * where EOS's OWN reserved id used to be -- see that function's\n"
                 " * docstring) gets a harmless empty-string placeholder; it is never\n"
                 " * reachable by typed input or generation either way. */\n")
        f.write(f"static const char *const kevgpt_itos[KEVGPT_VOCAB_SIZE] = {{\n    ")
        f.write(",\n    ".join(_c_str_literal(itos.get(i, "")) for i in range(vocab)))
        f.write("\n};\n\n")
        if missing:
            f.write(f"/* NOTE: ids {missing} had no itos entry in this checkpoint's "
                     f"meta.json (dead slots, see above) -- filled with \"\". */\n\n")

        f.write("/* token ids whose word IS a sentence-ending '.', '!', or '?' --\n"
                 " * KEVGPT_EOS_ID is ALSO always a stop condition (the story-boundary\n"
                 " * token) but is handled separately in main.c, not listed here. */\n")
        f.write(f"#define KEVGPT_STOP_COUNT {len(stop_ids)}u\n")
        f.write(f"static const uint16_t kevgpt_stop_ids[KEVGPT_STOP_COUNT] = {{"
                + ", ".join(str(i) for i in stop_ids) + "};\n\n")
        f.write("#endif\n")


def build_tokenizer_ddr_blob(meta_path) -> bytes:
    """DDR3-resident tokenizer table (PORT-NOTES.md "VOCAB=16384 runbook",
    tokenizer-off-BRAM fix): a per-id uint32 LE offset table (VOCAB entries,
    byte offset of that id's null-terminated string, relative to the START
    OF THE STRINGS SECTION which immediately follows the offset table -- so
    `KEVGPT_TOKENIZER_STRINGS_OFFSET = VOCAB*4` needs no separate stored
    field, main.c/this function both compute it the same way), followed by
    every id's string 0..VOCAB-1 in order, null-terminated, concatenated.
    Same "every id needs an entry, missing ones get a harmless empty-string
    placeholder" handling as emit_tokenizer_header_word()'s kevgpt_itos[]
    (a checkpoint trained before model.word_data.build_vocab()'s EOS-
    exclusion fix can have a dead id=1 slot -- see that function's own
    docstring). Read back through cpu_ddr_bridge's ext_slaves window by
    plain C pointer dereference (no data cache on this core, no special
    DDR3-aware read path needed) -- see main.c's kevgpt_itos_ddr()."""
    m = json.load(open(meta_path))
    itos = {int(k): v for k, v in m["itos"].items()}
    vocab = m["vocab_size"]
    assert m.get("tokenizer") == "word"

    strings = []
    for i in range(vocab):
        s = itos.get(i, "")
        assert all(ord(ch) < 128 for ch in s), (
            f"itos[{i}]={s!r} has non-ASCII characters -- DDR3 strings are "
            "plain null-terminated C strings and can't represent it"
        )
        strings.append(s.encode("ascii"))

    offsets = []
    off = 0
    for s in strings:
        offsets.append(off)
        off += len(s) + 1  # +1 for the null terminator
    offset_table = b"".join(struct.pack("<I", o) for o in offsets)
    strings_blob = b"".join(s + b"\x00" for s in strings)
    blob = offset_table + strings_blob
    # send_weights.py's wire protocol moves whole 32-bit words -- the offset
    # table is already word-aligned (vocab*4 bytes) but the strings section's
    # total length is arbitrary (sum of variable string lengths + null
    # terminators), so pad the WHOLE blob up to a word boundary or the last
    # partial word silently drops on the wire (struct.unpack's own
    # len(blob)//4 truncates, not an error -- caught by round-tripping this
    # blob back through _decode and diffing against meta.json before ever
    # trusting it on real hardware).
    pad = (-len(blob)) % 4
    return blob + b"\x00" * pad


def emit_tokenizer_header_word_ddr(out_path, meta_path):
    """DDR3-resident counterpart to emit_tokenizer_header_word() above: emits
    ONLY the small constants + kevgpt_stop_ids (still tiny, stays in on-chip
    SRAM) -- NOT kevgpt_itos[] itself, which at VOCAB=16384 is ~182KB, too
    big to bake into firmware .rodata alongside VOCAB=16384's own much
    bigger weight/KV-cache BRAM need (RAMB36E1 over-utilized at
    `place_design`, PORT-NOTES.md "VOCAB=16384 runbook" -- no SoC-memory
    resize closes that gap). The actual strings + offset table are
    build_tokenizer_ddr_blob()'s job, streamed into DDR3 by
    fabric.genesys2.send_weights alongside the weight image."""
    m = json.load(open(meta_path))
    itos = {int(k): v for k, v in m["itos"].items()}
    vocab = m["vocab_size"]
    assert m.get("tokenizer") == "word"

    from model.word_data import UNK, EOS
    unk_id = next(i for i, w in itos.items() if w == UNK)
    eos_id = next(i for i, w in itos.items() if w == EOS)

    stop_words = (".", "!", "?")
    stop_ids = [i for i in range(vocab) if i not in (unk_id, eos_id) and itos.get(i) in stop_words]

    with open(out_path, "w") as f:
        f.write("/* Auto-generated by fabric/genesys2/gen_chat_fw.py "
                "(emit_tokenizer_header_word_ddr) -- do not hand-edit.\n")
        f.write(" * Regenerate after retraining/re-exporting the word-vocab checkpoint.\n")
        f.write(" * kevgpt_itos strings live in DDR3, not here -- see\n")
        f.write(" * build_tokenizer_ddr_blob() / fabric.genesys2.send_weights. */\n")
        f.write("#ifndef KEVGPT_TOKENIZER_H\n#define KEVGPT_TOKENIZER_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define KEVGPT_TOKENIZER_WORD 1\n")
        f.write("#define KEVGPT_TOKENIZER_DDR 1\n")
        f.write(f"#define KEVGPT_VOCAB_SIZE {vocab}u\n")
        f.write(f"#define KEVGPT_UNK_ID {unk_id}u\n")
        f.write(f"#define KEVGPT_EOS_ID {eos_id}u\n")
        f.write(f"#define KEVGPT_TOKENIZER_DDR_BASE {TOKENIZER_DDR_BASE}u\n")
        f.write(f"#define KEVGPT_TOKENIZER_STRINGS_OFFSET ({vocab}u*4u)\n\n")

        f.write("/* token ids whose word IS a sentence-ending '.', '!', or '?' --\n"
                 " * KEVGPT_EOS_ID is ALSO always a stop condition (the story-boundary\n"
                 " * token) but is handled separately in main.c, not listed here. */\n")
        f.write(f"#define KEVGPT_STOP_COUNT {len(stop_ids)}u\n")
        f.write(f"static const uint16_t kevgpt_stop_ids[KEVGPT_STOP_COUNT] = {{"
                + ", ".join(str(i) for i in stop_ids) + "};\n\n")
        f.write("#endif\n")


def emit_tokenizer_header(out_path, meta_path):
    """Emit the C-side stoi/itos/stop-id tables `kevgpt_interactive` needs to
    encode typed input and decode generated tokens -- there is no such table
    anywhere in sw/ today (every prior firmware app only ever baked in ONE
    fixed prompt's pre-encoded ids, never a general encoder). Sourced from
    the same meta.json _encode() already reads, so this can't drift from
    what every other gate/tool in this project treats as the vocab.

    kevgpt_stoi[] is ASCII-code-indexed (0..127), -1 for anything meta.json's
    stoi has no entry for -- unlike Python's `stoi.get(c, 0)` (which maps
    unknown chars to id 0, a real character), the firmware caller decides
    the fallback explicitly at the call site, since -1 out of a lookup table
    is a clearer "not found" signal in C than overloading a valid id.

    kevgpt_stop_ids[] is computed via itos (decode), matching
    fabric/stage3/board/pl_kv256.py's own `_stop_ids` exactly and for the
    same reason: encoding the literal strings "."/"!"/"\\n" would risk
    nothing here (this vocab has no encode-side surprises), but computing
    it the same way as the already-proven KV260 reference keeps the two
    implementations directly comparable rather than coincidentally similar.
    """
    m = json.load(open(meta_path))
    stoi = m["stoi"]
    itos = {int(k): v for k, v in m["itos"].items()}
    vocab = len(itos)

    ascii_to_id = [-1] * 128
    for ch, idx in stoi.items():
        if len(ch) == 1 and ord(ch) < 128:
            ascii_to_id[ord(ch)] = idx

    stop_ids = [i for i in range(vocab) if itos.get(i) in (".", "!", "?", "\n")]

    def _c_char_literal(ch):
        # kevgpt_itos is a plain 8-bit `char[]` -- it can only hold ASCII.
        # Assert rather than silently substituting a placeholder: which
        # character (if any) is "safe" to stand in for an out-of-range
        # codepoint depends on what THIS vocab actually contains (a prior
        # version of this function hardcoded '?' on the assumption that a
        # specific checkpoint's vocab had no real '?' token -- true then,
        # false for the story-teller vocab, which does). Emits as a numeric
        # escape so the header is valid C regardless of source encoding.
        assert ord(ch) < 128, (
            f"itos contains non-ASCII char {ch!r} (U+{ord(ch):04x}) -- "
            "kevgpt_itos is a char[] and can't represent it; clean the "
            "training corpus instead of guessing a placeholder here"
        )
        return "\\x%02x" % ord(ch)

    with open(out_path, "w") as f:
        f.write("/* Auto-generated by fabric/genesys2/gen_chat_fw.py -- do not hand-edit.\n")
        f.write(" * Regenerate after retraining/re-exporting the checkpoint. */\n")
        f.write("#ifndef KEVGPT_TOKENIZER_H\n#define KEVGPT_TOKENIZER_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define KEVGPT_VOCAB_SIZE {vocab}u\n\n")

        f.write(f"static const char kevgpt_itos[KEVGPT_VOCAB_SIZE] = {{\n    ")
        f.write(", ".join(f"'{_c_char_literal(itos[i])}'" for i in range(vocab)))
        f.write("\n};\n\n")

        f.write("/* ASCII code -> token id, -1 if this vocab has no such char. */\n")
        f.write("static const int8_t kevgpt_stoi[128] = {\n    ")
        f.write(", ".join(str(v) for v in ascii_to_id))
        f.write("\n};\n\n")

        f.write("/* token ids whose character is a sentence-ending '.', '!', '?', or '\\n' --\n")
        f.write(" * computed via itos (decode), matching fabric/stage3/board/pl_kv256.py's\n")
        f.write(" * own _stop_ids exactly, not hardcoded to any one vocab's specific chars. */\n")
        f.write(f"#define KEVGPT_STOP_COUNT {len(stop_ids)}u\n")
        f.write(f"static const uint8_t kevgpt_stop_ids[KEVGPT_STOP_COUNT] = {{"
                + ", ".join(str(i) for i in stop_ids) + "};\n\n")
        f.write("#endif\n")


def emit_header(out_path, weight_words, prompt_ids, expected_gen, plen, ngen):
    npass = plen + ngen - 1
    with open(out_path, "w") as f:
        f.write("/* Auto-generated by fabric/genesys2/gen_chat_fw.py -- do not hand-edit.\n")
        f.write(" * Regenerate after retraining/re-exporting the Option A checkpoint. */\n")
        f.write("#ifndef KEVGPT_WEIGHTS_H\n#define KEVGPT_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define KEVGPT_PLEN {plen}\n")
        f.write(f"#define KEVGPT_NGEN {ngen}\n")
        f.write(f"#define KEVGPT_NPASS {npass}\n\n")

        f.write(f"static const uint32_t kevgpt_weight_words[{len(weight_words)}] = {{\n")
        for i in range(0, len(weight_words), 8):
            chunk = weight_words[i:i + 8]
            f.write("    " + ", ".join(f"0x{w:08x}u" for w in chunk) + ",\n")
        f.write("};\n")
        f.write(f"#define KEVGPT_WEIGHT_WORDS_COUNT {len(weight_words)}u\n\n")

        f.write(f"static const uint16_t kevgpt_prompt_ids[KEVGPT_PLEN] = {{"
                + ", ".join(str(int(t)) for t in prompt_ids) + "};\n")
        f.write(f"static const uint16_t kevgpt_expected_gen[KEVGPT_NGEN] = {{"
                + ", ".join(str(int(t)) for t in expected_gen) + "};\n\n")
        f.write("#endif\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.genesys2.gen_chat_fw")
    ap.add_argument("--npz", default="fabric/export_optionA/goformer.npz")
    ap.add_argument("--meta", default="data/char_optionA/meta.json")
    ap.add_argument("--prompt", default="once")
    ap.add_argument("--ngen", type=int, default=6)
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-out", default=None,
                     help="also emit the stoi/itos/stop-id header kevgpt_interactive needs "
                          "(e.g. .../kevgpt_interactive/kevgpt_tokenizer.h)")
    ap.add_argument("--tokenizer-ddr", action="store_true",
                     help="emit the DDR3-resident tokenizer variant (constants only, no "
                          "kevgpt_itos[] array -- PORT-NOTES.md VOCAB=16384 runbook) instead "
                          "of the fully-baked-in header. Word-level tokenizer only.")
    ap.add_argument("--tokenizer-blob-out", default=None,
                     help="with --tokenizer-ddr: also write the DDR3 blob "
                          "(build_tokenizer_ddr_blob()) to this path, for "
                          "fabric.genesys2.send_weights to stream over UART")
    a = ap.parse_args(argv)

    prompt_ids = _encode(a.meta, a.prompt)
    plen = len(prompt_ids)

    p, cfg = seq_ref.build(a.npz)
    gold_seq = IntKVQSequencer(p, cfg, kbits=8, vbits=8, rotate=False, divfree=True)
    expected_gen = gold_seq.generate_greedy(list(prompt_ids), a.ngen)[plen:]

    nlayer = len(p["blocks"])
    iseq = seq_ref.IntSequencer(p, cfg)
    with tempfile.TemporaryDirectory() as td:
        write_mems_wideword(td, iseq, a.lanes, nlayer, a.p)
        weight_words = wrom_to_words(os.path.join(td, "wrom.mem"), a.lanes)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    emit_header(a.out, weight_words, prompt_ids, expected_gen, plen, a.ngen)
    print(f"wrote {a.out}: {len(weight_words)} weight words "
          f"({len(weight_words) * 4} bytes), prompt={a.prompt!r} "
          f"plen={plen} ngen={a.ngen} expected_gen={expected_gen}")

    if a.tokenizer_out:
        os.makedirs(os.path.dirname(a.tokenizer_out), exist_ok=True)
        meta = json.load(open(a.meta))
        if a.tokenizer_ddr:
            assert meta.get("tokenizer") == "word", "--tokenizer-ddr is word-level only"
            emit_tokenizer_header_word_ddr(a.tokenizer_out, a.meta)
        elif meta.get("tokenizer") == "word":
            emit_tokenizer_header_word(a.tokenizer_out, a.meta)
        else:
            emit_tokenizer_header(a.tokenizer_out, a.meta)
        print(f"wrote {a.tokenizer_out}")

    if a.tokenizer_blob_out:
        assert a.tokenizer_ddr, "--tokenizer-blob-out only makes sense with --tokenizer-ddr"
        blob = build_tokenizer_ddr_blob(a.meta)
        os.makedirs(os.path.dirname(a.tokenizer_blob_out) or ".", exist_ok=True)
        with open(a.tokenizer_blob_out, "wb") as f:
            f.write(blob)
        print(f"wrote {a.tokenizer_blob_out}: {len(blob)} bytes "
              f"({len(blob) // 4} words, DDR3 base 0x{TOKENIZER_DDR_BASE:x})")


if __name__ == "__main__":
    main()
