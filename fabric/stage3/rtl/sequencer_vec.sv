// -----------------------------------------------------------------------------
// sequencer_vec — the P-WIDE vector-datapath sequencer (the road to ~10k tok/s).
// WIDE-WORD banking + SYNCHRONOUS-READ (BRAM) edition.
//
// Every scratch buffer and every big ROM is ONE row-addressed memory holding P
// elements per word (lane l in bits [l*W +: W]) — and every read is REGISTERED:
// address presented on cycle k, data consumed on k+1. One read register per
// memory, addressed via a small state mux, so Vivado infers BLOCK RAM instead of
// LUTRAM. Why: the P=8 LUT budget was 109% (~31k distributed-RAM LUTs); BRAM
// frees this. Bonus: iverilog sees the same 1-cycle read latency as silicon, so
// async-read divergences (the §6 board bug class) are out by construction.
//
// FSM loops pipeline (one cycle skew, not 2x):
//   fr/ci/gfr/ar  : read-address counter (held at limit when done)
//   frd/cid/...   : address delayed 1 cycle  -> consume stage
//   frv/civ/...   : valid bit (entered loop, address in range)
// Producer states (G_RB, S_ACL) keep their plain-reg staging words.
//
// iverilog-2012 safe: variable +: part-selects only index PLAIN REGS; flat ROMs
// for $readmemh; reads of unpacked arrays only by element index.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module sequencer_vec #(
    parameter integer D     = 256,
    parameter integer D3    = 768,
    parameter integer D_MLP = 1024,
    parameter integer P     = 8,
    parameter integer LANES = 16,
    parameter integer VOCAB = 193,
    parameter integer TMAX  = 256,
    parameter integer WWORDS = 262144,
    parameter integer NLAYER = 4,
    // gamma_w capacity: 2 LayerNorms (ln1+ln2) per block + 1 final LN_f.
    // dqm_w/dqe_w capacity: one dequant channel per GEMV output row across
    // every block's qkv/proj/mlp_fc/mlp_proj, plus the head. Both were
    // HARDCODED (9 / 9409, sized for NLAYER=4/D=256/D3=768/D_MLP=1024/
    // VOCAB=193's original KV260 shape) until this fix -- silently
    // truncating gamma_w.mem/dqm_w.mem/dqe_w.mem at $readmemh load time for
    // ANY caller with a different real need, with NO error or warning
    // beyond an easily-missed "not enough words" message, corrupting
    // LayerNorm/dequant for the truncated tail (in practice: the last
    // block(s)) and propagating as X/garbage through everything
    // downstream. Found via a real NLAYER=5 hang-free-but-X-output bug
    // (GAMMA_N=9=2*4+1 fit NLAYER<=4 exactly, silently broke at NLAYER=5)
    // -- see fabric/genesys2/PORT-NOTES.md. Now derived from the actual
    // shape parameters instead of a guessed constant, matching gdone's own
    // earlier width fix -- bit-identical to the old hardcoded defaults for
    // every existing NLAYER=4 config (2*4+1=9, exactly), so no existing
    // build's behavior changes.
    parameter integer GAMMA_N = 2*NLAYER + 1,
    parameter integer DQ_N    = NLAYER*(D3+D+D_MLP+D) + VOCAB,
    // inv_sact capacity: g_asel = blk*4 + {0,1,2,3} (qkv/proj/mlp_fc/mlp_proj
    // dequant-frac select per block) ranges 0..4*(NLAYER-1)+3, PLUS the
    // final g_asel=4*NLAYER for the head -- 4*NLAYER+1 entries total. Same
    // hardcoded-for-NLAYER=4 bug as GAMMA_N/DQ_N above (17=4*4+1 exactly):
    // found via bisecting the SAME NLAYER=8 X-output symptom down to an
    // exact blk==4 boundary (blocks 0-3 fine, 4-7 corrupted) AFTER the
    // GAMMA_N/DQ_N fix alone didn't resolve it -- g_asel=4*4=16 (block 4's
    // qkv access) was still the last VALID index into the old 17-deep
    // array, but block 4's proj/mlp_fc/mlp_proj accesses (g_asel=17,18,19)
    // read past the end. See fabric/genesys2/PORT-NOTES.md.
    parameter integer NSACT   = 4*NLAYER + 1,
    parameter integer NHEAD = 4,
    parameter integer HEAD_DIM = 64,
    parameter integer RESID_FRAC  = 25,
    parameter integer LN_OUT_FRAC = 22,
    parameter integer VFRAC       = 16,
    parameter integer GELU_FRAC   = 12,
    parameter integer ISH         = 40,
    // Genesys2 (Kintex-7) port: no URAM primitive exists on that part, so the
    // resident weight image and KV-cache code bank must fall back to ordinary
    // BRAM TDP. Passed straight through to gemv_banked_resident_vec/kv_bank;
    // default "ultra" leaves the KV260 build/gates untouched.
    parameter               MEM_PRIMITIVE = "ultra",
    // DDR3-backed KV cache (fabric/genesys2/PORT-NOTES.md "Phase 2
    // architecture"): 0 (default) keeps every existing build -- KV260 and
    // Genesys2 Option A alike -- byte-for-byte on the resident kv_bank.sv
    // path, untouched. 1 selects kv_bank_ddr.sv instead (same wq_*/rd_*
    // external contract, verified bit-exact against kv_bank.sv in
    // fabric/genesys2/tb/tb_kv_bank_ddr.sv), routing the KV cache through
    // the kv_wr_*/kv_rd_* DMA ports below instead of on-chip BRAM. The two
    // are mutually exclusive at elaboration time (generate), not a runtime
    // mux -- KV260's build never even sees kv_bank_ddr's logic.
    parameter               KV_DDR_BACKED = 0,
    // KV_DDR_BASE: byte offset of the KV cache's own DDR3 region, distinct
    // from WEIGHTS_DDR_BASE below. Found the hard way (real-hardware-only,
    // reproducible corruption starting at generate-token 8 under per-layer
    // weight streaming): with both bases defaulting to 0, kv_bank_ddr's
    // preallocated (layer,kv,head,pos)-indexed region and weight_loader_
    // ddr's staged weight image are the SAME physical DDR3 bytes -- every
    // KV cache write silently overwrites staged weight bytes. Simulation
    // never caught this because no gate has ever run KV_DDR_BACKED=1 and
    // WEIGHT_STREAM_PER_LAYER=1 together (tb_seq_vec_kv_stream.sv keeps KV
    // resident, "unrelated to this gate"). Default stays 0 for every
    // existing KV_DDR_BACKED=1-only build (kv_bank_ddr is the sole DDR3
    // consumer there, so 0 is fine); any build that ALSO sets WEIGHT_
    // STREAM_PER_LAYER=1 MUST override this to a value >= the staged
    // weight image's total byte size (NLAYER*GW_BLK+GW_HEAD+GW_EMB, in
    // WBYTES_STRM-sized words) -- see xilinx_core_v_mini_mcu_wrapper_
    // kevgpt.sv for the real deployed value.
    parameter integer       KV_DDR_BASE = 0,
    // DDR3-backed weight-window loader (PORT-NOTES.md "weight_loader_ddr
    // wired to top level"): 0 (default) leaves every existing build
    // byte-for-byte untouched -- wld_* outputs tied idle, gemv_banked_
    // resident_vec's boot-load port driven ONLY by the existing firmware
    // wl_rst/wl_we/wl_data path, exactly as today. 1 additionally
    // instantiates weight_loader_ddr.sv, ORing its wb_ld_rst/wb_w_we/
    // wb_w_data into that SAME boot-load port alongside wl_rst/wl_we/
    // wl_data -- additive, not a generate-selected replacement like
    // KV_DDR_BACKED, since firmware's boot-time stream and the DMA
    // reloader are both valid sources of the same port at different
    // times (never driven together; that's a firmware-sequencing
    // invariant, not enforced in hardware here).
    parameter               WEIGHT_DDR_BACKED = 0,
    // Per-layer DDR3 weight streaming (fabric/genesys2/PORT-NOTES.md
    // "per-layer weight streaming"): 0 (default) leaves every existing
    // build byte-for-byte untouched -- g_wbase keeps its block-absolute
    // blk*GW_BLK+WB_XXX addressing into a FULLY resident weight_bank_tdp,
    // exactly as today. 1 additionally: (a) drops the blk*GW_BLK term
    // from every g_wbase assignment (QKV/PROJ/FC/MP/HEAD), since under
    // streaming only ONE block's (or the head's) window is EVER resident
    // at a time, always starting at on-chip address 0 right after its own
    // fresh reload; (b) inserts a new S_STRW state between L_COLL's exit
    // and the real next state (S_QKVRET or S_HEADSET) that triggers
    // weight_loader_ddr INTERNALLY (mid-inference, once per block plus
    // once for the head, not just once at firmware boot) and stalls until
    // it completes. Requires WEIGHT_DDR_BACKED=1 (reuses that same
    // weight_loader_ddr instance) and WWORDS sized to >= max(GW_BLK,
    // GW_HEAD), not the whole NLAYER-scaled image -- the actual BRAM win:
    // weight-bank BRAM cost becomes independent of NLAYER. Real per-layer
    // reload cost is a measured ~9 cycles/wide-word (a real weight_bank_
    // tdp write-port-width ceiling, not a DDR3 latency artifact -- see
    // PORT-NOTES.md's feasibility measurement), not free -- an explicit,
    // accepted throughput/capacity tradeoff, not a bug.
    parameter               WEIGHT_STREAM_PER_LAYER = 0,
    // Byte address within weight_loader_ddr's own DDR3 aperture where the
    // full weight image starts -- matches send_weights.py/uart_load_
    // weights()'s existing staging point (ddr_addr=0 in the boot-time
    // kevgpt_wld_load() call this mode replaces), not a new staging
    // location.
    parameter integer       WEIGHTS_DDR_BASE = 0
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,
    // tok_id/tok_out width: was hardcoded [8:0] (max 511), fine for every
    // char-level VOCAB (<=193) this project ever deployed but a REAL bug at
    // word-level VOCAB=1900 (PORT-NOTES.md "word-level vocabulary") -- ANY
    // prompt token with id>511 silently truncated on input, corrupting the
    // whole forward pass from the very first embed lookup. $clog2(VOCAB)
    // computed here, not guessed; every caller (xheep_kevgpt_peripheral.sv's
    // own tok_id/core_tok_out regs, both testbenches' tok/tok_out) needs the
    // same fix -- a port-width change, not an internal-signal-only one.
    input  wire [$clog2(VOCAB)-1:0] tok_id,
    input  wire [8:0]  pos,
    output reg         done,
    output reg [$clog2(VOCAB)-1:0] tok_out,  // argmax token id (after the full forward + head)
    // readback: rd_sel picks the bank, rd_addr the element (2-cyc registered). 64-bit so
    // the Q.22 LN/gelu values fit; 32-bit values are sign-extended in their bank.
    input  wire [3:0]  rd_sel,
    input  wire [10:0] rd_addr,
    output reg signed [63:0] rd_data,
    // runtime weight load (stream the whole transposed image once)
    input  wire        wl_rst,
    input  wire        wl_we,
    input  wire [31:0] wl_data,
    // DEBUG halt points (banks then hold a block-0 snapshot for board readback vs
    // seq_ref.block0_phase_signals): 1=after embed (xres=x_in), 2=after LN2 (xres=x_res1,
    // lnout2=ln2), 3=after block0 (xres=x_out). 0 = no stop (normal forward).
    input  wire [1:0]  dbg_stop,
    // on-chip Gumbel-max sampling: seed_we loads the persistent xorshift state and
    // enables sampling (state != 0). The argmax over the VOCAB head logits then adds
    // a per-logit Gumbel noise (precomputed into gumbel_bank during the head GEMV).
    // seed == 0 (or never written) => greedy argmax, bit-exact to the old behaviour.
    input  wire [31:0] seed,
    input  wire        seed_we,

    // ---- DDR-backed KV cache DMA ports (KV_DDR_BACKED=1 only; idle-tied
    // when 0 -- see KV_DDR_BACKED parameter comment above). Wire straight to
    // a fabric/genesys2/rtl/kevgpt_ddr_bundle.sv instance's kv_wr_*/kv_rd_*
    // ports at the top level. -----------------------------------------------
    output wire                 kv_wr_pkt_valid,
    input  wire                 kv_wr_pkt_ready,
    output wire [28:0]          kv_wr_pkt_addr,
    output wire [255:0]         kv_wr_pkt_data,
    output wire [31:0]          kv_wr_pkt_mask,
    input  wire                 kv_wr_ack_valid,
    output wire                 kv_wr_ack_ready,
    output wire                 kv_rd_req_valid,
    input  wire                 kv_rd_req_ready,
    output wire [28:0]          kv_rd_req_addr,
    input  wire                 kv_rd_ret_valid,
    output wire                 kv_rd_ret_ready,
    input  wire [255:0]         kv_rd_ret_data,

    // ---- DDR-backed weight-window loader control + DMA ports
    // (WEIGHT_DDR_BACKED=1 only; idle-tied when 0). Control ports mirror
    // weight_loader_ddr.sv's own ld_start/ld_ddr_addr/ld_words/ld_done
    // (firmware-triggered, one window per pulse); DMA ports are named to
    // match fabric/genesys2/rtl/kevgpt_ddr_bundle.sv's existing wl_rd_*
    // pass-through ports for direct top-level wiring. -----------------------
    input  wire                 wld_ld_start,
    input  wire [28:0]          wld_ld_ddr_addr,
    input  wire [31:0]          wld_ld_words,
    output wire                 wld_ld_done,
    output wire                 wl_rd_req_valid,
    input  wire                 wl_rd_req_ready,
    output wire [28:0]          wl_rd_req_addr,
    input  wire                 wl_rd_ret_valid,
    output wire                 wl_rd_ret_ready,
    input  wire [255:0]         wl_rd_ret_data,

    // FIXATION-WORD-CDC-INVESTIGATION.md Sec8 item 1: running CRC32 over
    // every word weight_loader_ddr's REAL DMA path writes into
    // weight_bank_tdp (wld_ldb_we/wld_ldb_data below -- not the CPU-manual
    // wl_we/wl_data boot-load path). Free-running since reset; firmware
    // reads it at a checkpoint of its choosing and compares against a
    // host/simulation-computed expected value for the identical run --
    // the whole point is testing weight_loader_ddr -> CDC -> mux ->
    // arbiter -> MIG -> CDC end to end, which no prior diagnostic covered
    // (KEVGPT_DIAG_DUMP_HEAD reads DDR3 via a plain CPU load, bypassing
    // this entire path).
    output wire [31:0]          weight_stream_crc
);
    localparam integer ROWS  = D    / P;
    localparam integer ROWS3 = D3   / P;
    localparam integer ROWSM = D_MLP/ P;
    localparam integer LSH   = $clog2(P);
    localparam integer GRPSH = $clog2(LANES/P);   // P-rows per ymem group word (shift)
    localparam integer DQROWS = (DQ_N + P - 1) / P;
    // weight bases (LANES=16) and dequant channel-row bases (/P)
    localparam integer GW_QKV  = ((D3   + LANES-1)/LANES) * D;   // 12288
    localparam integer GW_PROJ = ((D    + LANES-1)/LANES) * D;   // 4096
    localparam integer GW_FC   = ((D_MLP+ LANES-1)/LANES) * D;   // 16384
    localparam integer WB_QKV = 0, WB_PROJ = GW_QKV,
                       WB_FC = GW_QKV+GW_PROJ, WB_MP = GW_QKV+GW_PROJ+GW_FC;
    localparam integer DR_QKV = 0, DR_PROJ = D3/P, DR_FC = (D3+D)/P, DR_MP = (D3+D+D_MLP)/P;
    localparam integer GW_MP   = ((D + LANES-1)/LANES) * D_MLP;          // 16384
    localparam integer GW_HEAD = ((VOCAB + LANES-1)/LANES) * D;          // 3328
    localparam integer GW_BLK  = GW_QKV+GW_PROJ+GW_FC+GW_MP;             // 49152 (words/block)
    localparam integer DQ_BLK  = D3+D+D_MLP+D;                           // 2304 (channels/block)
    localparam integer DQB_P   = DQ_BLK/P;                               // 288 (rows/block, /P)
    localparam integer WB_HEAD = NLAYER*GW_BLK;                          // head weight base
    localparam integer DR_HEAD = (NLAYER*DQ_BLK)/P;                      // head dequant row base
    localparam integer ARROWS  = (VOCAB + P - 1)/P;                      // argmax rows
    // dor (G_RB's readback/dequant row counter, declared below) drives
    // EVERY g_dst destination's readback -- qkv/proj/mlp/head -- and was
    // sized ONLY off ROWSM (D_MLP/P, the biggest BLOCK-side row count) on
    // the assumption VOCAB would always stay small. Real bug, found via a
    // real hang: at word-vocab sizes (VOCAB=1900, PORT-NOTES.md "word-
    // level vocabulary"), the head's own row count (ARROWS=238) exceeds
    // ROWSM=64, so dor (7 bits, max 127) wrapped before ever reaching
    // dor==ARROWS-1=237 -- G_RB's own exit condition (see below) never
    // fires, an infinite loop, not a slow simulation. MAXROWS is sized
    // from the actual larger of the two, same "compute from shape
    // params, don't guess a width" discipline as GAMMA_N/NSACT/l_gbase.
    localparam integer MAXROWS = (ROWSM > ARROWS) ? ROWSM : ARROWS;
    // width for internal registers that hold an actual VOCAB index (argmax
    // candidates/winner: gj/gj_d, best_idx, hidx, wm_idx, am_idx, pi0-3,
    // ia/ib below) -- same "found via a real word-vocab bug" story as
    // MAXROWS/tok_id/tok_out above. A padded row's out-of-range index
    // (ia/ib >= VOCAB) is forced to NEG_INF34 before it can ever become a
    // winner (see S_ARGMAX), so VOCAB itself (not ARROWS*P) is the correct
    // bound -- a stored winning index is always < VOCAB.
    localparam integer VIDXW = $clog2(VOCAB);
    // width for the whole "row index / raw dimension" register family that
    // was hardcoded [BUSW-1:0] (11 bits, max 2047) throughout this FSM -- ci,
    // cid/cid1-3, rb0-2, qkv_wrow, gv_m/gv_k/gv_rdaddr, g_m/g_k, and every
    // *_ra read-address wire (xres/lnout1/lnout2/ctxv/mlpbuf/qkv/attn/mlp/
    // head_ra, gam_fr) -- found the same way MAXROWS/VIDXW were: `dor` got
    // fixed once already for the VOCAB=57->1900 jump (see MAXROWS's own
    // comment above), but that fix wasn't propagated to these SIBLING
    // registers, which silently assumed 11 bits was always enough. It
    // wasn't, at VOCAB=16384: ARROWS=2048 needs MAXROWS's own counter to
    // reach the value 2048 (a loop-exit sentinel, RIDXW=$clog2(MAXROWS+1)
    // bits), and g_m/g_k store VOCAB itself as a raw dimension value (up
    // to 16384, DIMW=$clog2(VOCAB+1) bits) when driving the head GEMV --
    // both exceed 11 bits, the first as a real `iverilog` elaboration
    // error (negative concatenation-repeat count), the second SILENTLY
    // (`g_m<=VOCAB[BUSW-1:0]` truncated 16384 to 0, corrupting every head
    // logit) -- only caught by actually trying VOCAB=16384, not by any
    // compile-time check. BUSW covers both needs with one consistent
    // width, floored at 11 to keep every existing smaller-VOCAB shape's
    // own already-verified bit-exactness byte-for-byte unchanged.
    localparam integer MAXDIM  = (D3 > D_MLP) ? ((D3 > VOCAB) ? D3 : VOCAB)
                                              : ((D_MLP > VOCAB) ? D_MLP : VOCAB);
    localparam integer DIMW    = $clog2(MAXDIM + 1);
    localparam integer RIDXW   = $clog2(MAXROWS + 1);
    localparam integer BUSW_RAW = (DIMW > RIDXW) ? DIMW : RIDXW;
    localparam integer BUSW    = (BUSW_RAW > 11) ? BUSW_RAW : 11;
    // per-layer weight streaming (WEIGHT_STREAM_PER_LAYER=1 only): DDR3
    // byte stride per wide word and 32-bit "loader word" count per wide
    // word -- same WBITS/SUBW relationship weight_bank_tdp.sv/
    // weight_loader_ddr.sv already use internally, needed here too since
    // the DDR3 reload address/word-count arithmetic now lives in the FSM
    // instead of only in firmware.
    localparam integer WBITS_STRM  = LANES*4;
    localparam integer SUBW_STRM   = WBITS_STRM/32;
    localparam integer WBYTES_STRM = WBITS_STRM/8;
    // cycles S_STRW idles after wld_ld_done before its first read -- see
    // that state's own comment (weight_bank_tdp's WRITE_MODE="no_change"
    // true-dual-port BRAM write/read collision timing, real-hardware-only,
    // never exercised by the fully-resident design's "load once at boot"
    // pattern). Cheap relative to a single reload's own thousands of
    // cycles; not tuned tight against real hardware yet.
    localparam integer STRW_SETTLE_CYCLES = 32;
    localparam signed [33:0] NEG_INF34 = {1'b1, 33'b0};                  // -2^33 (argmax -inf)
    localparam integer EROWS   = D / P;                                  // emb/gamma rows per set
    // emb_row_w's multiplicand is tok_id OR pos (whichever this read is
    // for), so it must cover the larger of VOCAB/TMAX, not VOCAB alone --
    // same "found via a real word-vocab bug" story as VIDXW above.
    localparam integer EMBROWW = $clog2(((VOCAB>TMAX)?VOCAB:TMAX)*EROWS + EROWS);

    // ---- embed image in the resident weight URAM's SPARE DEPTH (log §36 plan 2) -
    // The tok/pos embed ROMs (~92 BRAM tiles) are APPENDED to the wrom weight image
    // by write_mems_wideword and read through the GEMV bank's port B (phase-disjoint
    // with GEMV reads). EPW embed rows (P*32b each) pack into one LANES*4-bit word:
    // word = {row_{EPW-1}, .., row1, row0}; row r of a table lives at word
    // EMB_*_BASE + r/EPW. Bases are EVEN-ALIGNED so the DP=1 column-parity pair
    // read returns RPP = 2*EPW consecutive rows; slot select within the pair is
    // r % RPP. Requires EPW >= 1 (LANES >= 8*P, e.g. LANES=256 at P=8): smaller
    // LANES no longer carry embeds in the wrom image.
    localparam integer EPW   = (LANES*4)/(P*32);            // embed rows per wrom word
    localparam integer EPWS  = (EPW > 1) ? $clog2(EPW) : 0; // row -> word shift
    localparam integer RPP   = (EPW > 0) ? 2*EPW : 1;       // rows per pair read
    localparam [2:0]   RSELM = RPP - 1;                     // pair-slot mask (RPP<=8)
    localparam integer EMB_TOK0     = WB_HEAD + GW_HEAD;    // first word past the image
    localparam integer EMB_TOK_BASE = EMB_TOK0 + (EMB_TOK0 % 2);
    localparam integer EMB_TOKW     = (EPW > 0) ? (VOCAB*EROWS + EPW - 1)/EPW : 0;
    localparam integer EMB_POS0     = EMB_TOK_BASE + EMB_TOKW;
    localparam integer EMB_POS_BASE = EMB_POS0 + (EMB_POS0 % 2);
    localparam integer EMB_POSW     = (EPW > 0) ? (TMAX*EROWS + EPW - 1)/EPW : 0;
    // GW_EMB: the FULL combined tok+pos+alignment-pad span on DDR3 (from
    // EMB_TOK_BASE through the end of the wrom.mem image) -- this is the
    // real DDR3 image size (used by WEIGHT_IMAGE_BYTES / send_weights.py's
    // own byte count) and by EMB_TOK_BASE/EMB_POS_BASE below to locate a
    // GIVEN tok_id/pos row within that off-chip table. It does NOT size
    // any on-chip BRAM under streaming (see EMB_ROWG_W).
    localparam integer GW_EMB = (EMB_POS_BASE + EMB_POSW) - EMB_TOK_BASE;

    // ---- streaming embed reload (row-at-a-time, not the whole table) ----------
    // Fix for the VOCAB/BRAM ceiling documented in PORT-NOTES.md's "streaming
    // embed lookups" proposal: S_EMB only ever consumes ONE tok_id row and ONE
    // pos row per token (EROWS sub-rows each) -- reloading the ENTIRE embed
    // table (GW_EMB words, VOCAB-linear) into weight_bank_tdp every token to
    // read just those two rows was the actual VOCAB/BRAM-ceiling driver, not
    // anything S_EMB's own read pattern needed. Under WEIGHT_STREAM_PER_LAYER,
    // S_STRW now reloads just the CURRENT tok_id's row (via a tok_id-scaled
    // DDR3 address, EMB_ROWG_W words), then S_EMB reads+holds it in `tok_hold`
    // (weight_loader_ddr's boot-load write port has no destination-address
    // input -- ld_rst always resets weight_bank_tdp's internal write pointer
    // to 0, so the tok row MUST be consumed into registers before the pos
    // reload overwrites the same on-chip words), then S_STRW reloads the
    // CURRENT pos row (same on-chip words, same EMB_ROWG_W size), and S_EMB
    // reads it and commits tok_hold+pos_row to xres_bank exactly as the old
    // single-shot design did. Both phases land at on-chip word 0 (there is
    // no separate on-chip base for tok vs pos any more, unlike the old
    // EMB_TOK_BASE_STRM/EMB_POS_BASE_STRM scheme this replaces), so
    // EMB_ROWG_W -- not GW_EMB -- is what WWORDS actually needs to cover for
    // the embed path once this is in place: VOCAB-independent, same
    // (EROWS+EPW-1)/EPW packing EMB_TOKW/EMB_POSW already use, just sized
    // for ONE row-group instead of VOCAB/TMAX of them.
    localparam integer EMB_ROWG_W = (EPW > 0) ? (EROWS + EPW - 1)/EPW : 0;

    // ---- FSM -------------------------------------------------------------------
    localparam [4:0]
      S_IDLE=0, S_EMB=1,
      L_GAM=2, L_FEED=3, L_COLL=4,                  // callable LN (L_GAM = start-only)
      G_AQ=5, G_RUN=6, G_WAIT=7, G_RB=8,            // callable GEMV (RB = fused rb+dq+gelu)
      S_QKVRET=10, S_AST=11, S_ALD=12, S_ACL=13,    // attention
      S_RES1=14, S_LN2=15, S_FCRET=16,              // proj/res1/LN2-call
      S_MPSET=17, S_RES2=20, S_FIN=21,              // mlp_proj setup (GELU folded into G_RB)
      S_HEADSET=22, S_ARGMAX=23,                    // final LN_f -> head -> argmax
      S_KVW_S=24, S_KVW_F=25, S_KVW_W=26,           // KV quant-write (doc-7 R1)
      S_CDR=27,                                     // ctx drain from the head catcher
      S_STRW=28;                                    // per-layer weight-stream reload+wait
                                                      // (WEIGHT_STREAM_PER_LAYER=1 only)
    reg [4:0] st;
    reg [3:0] blk;                           // transformer block 0..NLAYER-1
    // per-layer weight streaming (WEIGHT_STREAM_PER_LAYER=1 only): strw_ret
    // holds the REAL destination state (S_QKVRET for a block reload,
    // S_HEADSET for the head reload) S_STRW jumps to once the reload
    // completes -- also doubles as the block-vs-head discriminator inside
    // S_STRW itself, so no separate flag is needed. strw_armed marks
    // "already pulsed wldi_start, now waiting for wld_ld_done" within one
    // S_STRW visit. wldi_start/addr/words are the FSM's own internal
    // trigger into weight_loader_ddr, muxed in at the u_wld instantiation
    // alongside the existing firmware-facing wld_ld_start/etc. ports.
    reg [4:0]  strw_ret;
    reg        strw_armed;
    reg [5:0]  strw_settle;   // post-reload BRAM write-pipeline settle counter
    reg        wldi_start;
    reg [28:0] wldi_addr;
    reg [31:0] wldi_words;
    reg [BUSW-1:0] ci;
    reg [$clog2(ROWSM+1)-1:0] fr, orow;
    reg [$clog2(MAXROWS+1)-1:0] dor;   // widened for VOCAB-scale ARROWS -- see MAXROWS above
    // read-pipeline delayed addresses + valids (consume stage of each FSM loop)
    reg [BUSW-1:0] cid;  reg civ;
    reg [$clog2(ROWSM+1)-1:0] frd;  reg frv;
    reg [$clog2(ARROWS+1)-1:0] ard;  reg arv;

    // ---- S_EMB embed fetch through the weight bank's embed port -----------------
    // NON-STREAMING (WEIGHT_STREAM_PER_LAYER=0, whole table resident at boot):
    // issue alternates tok (etp=0) / pos (etp=1) row fetches, one address/cycle;
    // the pair lands on emb_pair 1 cycle later (eb_* are the arrival-stage regs).
    // tok row = tok_id*EROWS+fr, pos row = pos*EROWS+fr; word = BASE + row/EPW;
    // pair slot = row % RPP (bases even-aligned). ~2*EROWS+2 cycles per token.
    // STREAMING (WEIGHT_STREAM_PER_LAYER=1): only ONE EROWS-sized row-group is
    // ever resident on-chip at a time (see EMB_ROWG_W above) -- emb_ph (0=the
    // just-reloaded TOK row, 1=the just-reloaded POS row) selects the whole
    // read PASS, not the cycle; there is no per-cycle etp toggle under
    // streaming, since interleaving would need both rows resident at once.
    reg        etp;                          // non-streaming fetch phase: 0=tok,1=pos
    reg        emb_ph;                       // streaming reload/read phase: 0=tok,1=pos
    reg        eb_v, eb_tp;                  // arrival valid + phase
    reg [2:0]  eb_sel;                       // arrival pair-slot (row % RPP)
    reg [$clog2(ROWSM+1)-1:0] eb_row;        // arrival xres destination row
    reg [P*32-1:0]    tacc;                  // tok row held for the tok+pos sum (non-streaming)
    reg [P*32-1:0]    tok_hold [0:ROWS-1];   // tok row held per-sub-row (streaming only --
                                              // the on-chip words get overwritten by the
                                              // pos-row reload before all ROWS are read)
    reg [LANES*8-1:0] epr;                   // plain-reg pair copy (safe part-select)
    reg [P*32-1:0]    erw;                   // selected embed row
    // TIMING (5ns cone weight-bank BRAM -> xres LUTRAM): the pair-slot select is
    // REGISTERED (erw_r + eb2_* tags); the tacc hold / tok+pos sum / xres commit
    // run one cycle behind the arrival. Same numbers, +1 cycle per token.
    reg        eb2_v, eb2_tp;                // select-stage valid + phase
    reg [$clog2(ROWSM+1)-1:0] eb2_row;       // select-stage xres destination row
    reg [P*32-1:0]    erw_r;                 // REGISTERED selected embed row
    // non-streaming only -- absolute row index into the resident whole table
    wire [EMBROWW-1:0] emb_row_w = (etp ? pos : tok_id) * EROWS
                            + {{(EMBROWW-$clog2(ROWSM+1)){1'b0}}, fr};
    // 32-bit param + 14-bit row word offset, truncated to the address width.
    // Non-streaming: absolute row within the resident whole table (both bases
    // + the largest offset are < WWORDS by the spare-depth budget). Streaming:
    // `fr` alone, local to the just-reloaded EROWS-sized row-group -- BOTH
    // phases land at on-chip word 0 (weight_loader_ddr's boot-load write port
    // has no destination-address input; ld_rst always resets weight_bank_tdp's
    // write pointer to 0), so there is no separate tok/pos on-chip base any
    // more, unlike the EMB_TOK_BASE_STRM/EMB_POS_BASE_STRM scheme this replaces.
    wire [$clog2(WWORDS)-1:0] emb_addr_w =
        WEIGHT_STREAM_PER_LAYER
            ? (fr >> EPWS)
            : (etp ? EMB_POS_BASE : EMB_TOK_BASE) + (emb_row_w >> EPWS);
    wire emb_sel_w = (st == S_EMB);

    // ---- wide-word ROMs ($readmemh: one P-packed word per line) -----------------
    // All sync-read (registered) so they infer BLOCK RAM, not LUTs.
    (* rom_style = "block" *) reg [P*32-1:0] gamma_w   [0:GAMMA_N*EROWS-1];// Q4.20
    reg signed [63:0] inv_sact [0:NSACT-1];                       // 17-deep: stays LUT
    (* rom_style = "block" *) reg [P*24-1:0] dqm_w [0:DQROWS-1];           // P mant / word
    (* rom_style = "block" *) reg [P*8-1:0]  dqe_w [0:DQROWS-1];           // P exp  / word
    initial begin
        $readmemh("gamma_w.mem",   gamma_w);
        $readmemh("inv_sact.mem",  inv_sact);
        $readmemh("dqm_w.mem",     dqm_w);
        $readmemh("dqe_w.mem",     dqe_w);
    end

    // ---- wide-word scratch (one row-addressed memory per gateable intermediate) -
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank  [0:ROWS-1];    // residual Q6.25
    (* ram_style = "block" *) reg [P*64-1:0] lnout1_bank[0:ROWS-1];    // LN1 / LN_f out Q.22
    (* ram_style = "block" *) reg [P*64-1:0] lnout2_bank[0:ROWS-1];    // LN2 out Q.22
    (* ram_style = "block" *) reg [P*32-1:0] qkv_bank   [0:ROWS3-1];   // q|k|v Q.16
    (* ram_style = "block" *) reg [P*32-1:0] ctxv_bank  [0:ROWS-1];    // attention ctx Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] attn_bank  [0:ROWS-1];    // proj out Q6.25
    (* ram_style = "block" *) reg [P*64-1:0] mlpbuf_bank[0:ROWSM-1];   // mlp hidden Q4.12 / Q.22
    (* ram_style = "block" *) reg [P*32-1:0] mlp_bank   [0:ROWS-1];    // mlp_proj out Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] head_bank  [0:ARROWS-1];  // head logits Q6.25

    // ---- Gumbel-max on-chip sampling -------------------------------------------
    // gumbel_lut: 1024-deep signed Q.25 noise ROM (ONE read port), gumbel_lut[idx] =
    // round(temp*(-log(-log((idx+0.5)/1024)))*2^25). gumbel_bank: the per-logit noise
    // (one wide P-packed word per argmax row) filled during the head GEMV using that
    // single LUT port, so S_ARGMAX adds noise with NO extra read ports.
    (* rom_style = "block" *) reg signed [31:0] gumbel_lut  [0:1023];
    (* ram_style = "block" *) reg [P*32-1:0]    gumbel_bank [0:ARROWS-1];
    initial $readmemh("gumbel_lut.mem", gumbel_lut);
    reg [31:0]        rng_state;            // persistent xorshift32 state (loaded by seed_we)
    reg               smp_en;               // sampling enabled (rng_state != 0)
    reg signed [31:0] gumbel_lut_r;         // registered LUT read (1-cyc latency)
    reg [P*32-1:0]    gpre_r;

    // ---- synchronous-read registers (BRAM output stage; declared before use) ---
    reg [P*32-1:0] xres_r, qkv_r, ctxv_r, attn_r, mlp_r, head_r;
    reg [P*64-1:0] lnout1_r, lnout2_r, mlpbuf_r;
    reg [P*32-1:0] gam_r;
    reg [P*32-1:0] gumbel_r;                 // registered gumbel_bank read (S_ARGMAX)

    // ---- layernorm_vec ---------------------------------------------------------
    reg                ln_start, ln_vin;
    reg  [P*32-1:0]    ln_x, ln_g;
    wire               ln_yv, ln_done;
    wire [P*64-1:0]    ln_y;
    layernorm_vec #(.P(P), .D(D)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yv), .y_out(ln_y), .done(ln_done));

    // ---- GEMV (resident, P-wide boundary) ---------------------------------------
    // act feed: P INT8 lanes/cycle; readback: P INT32 outputs/cycle (rd_addr is a
    // P-group index). The MAC core is the proven gemv_banked_resident.
    reg                gv_ldrst, gv_xwe, gv_start;
    reg  [P*8-1:0]     gv_xdata;
    reg [BUSW-1:0]         gv_m, gv_k;
    reg [$clog2(WWORDS)-1:0] gv_wbase;
    wire               gv_done;
    // committed-group count (RB overlap) -- width MUST match gemv_banked_
    // resident_vec's own derived GDONE_W (ceil(MMAX/LANES) groups, MMAX
    // fixed at the u_gemv instantiation below) or the port connection
    // silently truncates. A hardcoded [3:0] here deadlocked G_RB for any
    // GEMV needing >=16 groups (D_MLP=1024 at LANES=64 needs exactly 16) --
    // see gdone's own comment in gemv_banked_resident_vec.sv. That fix
    // itself was another hardcoded guess (1024, sized for THAT one case),
    // not derived -- and VOCAB=16384 (the HEAD call's own M dimension)
    // blew past it the same way, deadlocking G_RB again (this time at
    // blk=NLAYER-1/head, `gv_gdone` never reaching a high enough count
    // since the underlying gemv_banked_resident_vec's own MMAX=1024
    // couldn't count that many groups at all). MMAX now derived from
    // MAXDIM (the actual widest M any call in this design ever uses --
    // D3/D_MLP for the block GEMVs, VOCAB for the head) instead of another
    // hardcoded number, so this doesn't need rediscovering a third time.
    localparam integer GEMV_MMAX    = MAXDIM;
    localparam integer GEMV_GROUPS  = (GEMV_MMAX + LANES - 1) / LANES;
    localparam integer GDONE_W      = $clog2(GEMV_GROUPS + 1);
    wire [GDONE_W-1:0] gv_gdone;
    reg [BUSW-1:0]         gv_rdaddr;
    wire [P*32-1:0]    gv_yout;
    wire [LANES*8-1:0] emb_pair;            // pair-read data (RPP embed rows)
    // ---- optional DDR-backed weight-window loader (WEIGHT_DDR_BACKED=1
    // only) -- see the parameter's own comment above. Additive: ORs onto
    // the SAME boot-load port firmware's wl_rst/wl_we/wl_data already
    // drives, never a generate-selected replacement (unlike kv_bank vs.
    // kv_bank_ddr), since both are valid sources at different times.
    wire        wld_ldb_rst, wld_ldb_we;
    wire [31:0] wld_ldb_data;
    generate
    if (WEIGHT_DDR_BACKED) begin : g_wld
        // WEIGHT_STREAM_PER_LAYER=1: the FSM's own S_STRW state drives the
        // loader internally (once per block plus once for the head, every
        // token), so the firmware-facing wld_ld_start/etc. ports are
        // unused in that mode -- elaboration-time select (WEIGHT_STREAM_
        // PER_LAYER is a parameter), not a runtime mux.
        weight_loader_ddr #(.ADDR_W(29), .DATA_W(256)) u_wld (
            .clk(clk), .rst(rst),
            .ld_start(WEIGHT_STREAM_PER_LAYER ? wldi_start : wld_ld_start),
            .ld_ddr_addr(WEIGHT_STREAM_PER_LAYER ? wldi_addr : wld_ld_ddr_addr),
            .ld_words(WEIGHT_STREAM_PER_LAYER ? wldi_words : wld_ld_words),
            .ld_done(wld_ld_done),
            .wb_ld_rst(wld_ldb_rst), .wb_w_we(wld_ldb_we), .wb_w_data(wld_ldb_data),
            .rd_req_valid(wl_rd_req_valid), .rd_req_ready(wl_rd_req_ready),
            .rd_req_addr(wl_rd_req_addr),
            .rd_ret_valid(wl_rd_ret_valid), .rd_ret_ready(wl_rd_ret_ready),
            .rd_ret_data(wl_rd_ret_data));
    end else begin : g_wld_off
        assign wld_ldb_rst     = 1'b0;
        assign wld_ldb_we      = 1'b0;
        assign wld_ldb_data    = 32'd0;
        assign wld_ld_done     = 1'b0;
        assign wl_rd_req_valid = 1'b0;
        assign wl_rd_req_addr  = 29'd0;
        assign wl_rd_ret_ready = 1'b1;
    end
    endgenerate

    // FIXATION-WORD-CDC-INVESTIGATION.md Sec8 item 1: taps the REAL
    // DMA-streamed write port (wld_ldb_we/wld_ldb_data, driven by
    // weight_loader_ddr's own u_wld instance above through its full
    // CDC/mux/arbiter/MIG round trip when WEIGHT_DDR_BACKED=1), never the
    // CPU-manual boot-load path (wl_we/wl_data). Accepts one word/cycle
    // unconditionally -- weight_loader_ddr's own back-to-back sub-word
    // writes (unpacking one DDR beat) never risk a dropped/missed word
    // here, unlike a slower multi-cycle-per-word CRC design would.
    crc32_word u_weight_stream_crc (
        .clk(clk), .rst(rst),
        .word_valid(wld_ldb_we), .word_data(wld_ldb_data),
        .crc_out(weight_stream_crc)
    );

`ifndef SYNTHESIS
    // KV cache / weight-image DDR3 region overlap check -- see KV_DDR_BASE's
    // own parameter comment for the real-hardware bug this is guarding
    // against (kv_bank_ddr and weight_loader_ddr silently aliasing onto the
    // SAME physical DDR3 bytes when both default to base 0). Mirrors kv_
    // bank_ddr.sv's own row/beat sizing formula (KBITS=8 matches the fixed
    // .KBITS(8) passed to u_kvb above) so this check tracks that module
    // without needing to read its internals at elaboration time.
    localparam integer KVDBG_BEAT_BYTES = 256/8;
    localparam integer KVDBG_CODE_BEATS = (HEAD_DIM*8 + 255)/256;
    localparam integer KVDBG_ROW_BYTES  = (KVDBG_CODE_BEATS+1)*KVDBG_BEAT_BYTES;
    localparam integer KVDBG_HROWS      = NLAYER*2*NHEAD*TMAX;
    localparam integer KV_IMAGE_BYTES   = KVDBG_HROWS*KVDBG_ROW_BYTES;
    localparam integer WEIGHT_IMAGE_BYTES = (NLAYER*GW_BLK+GW_HEAD+GW_EMB)*WBYTES_STRM;
    initial begin
        if (WEIGHT_STREAM_PER_LAYER && !WEIGHT_DDR_BACKED)
            $display("sequencer_vec: WARNING WEIGHT_STREAM_PER_LAYER=1 requires WEIGHT_DDR_BACKED=1 -- S_STRW will hang forever waiting for wld_ld_done, which g_wld_off ties permanently low");
        if (WEIGHT_STREAM_PER_LAYER && (WWORDS < GW_BLK || WWORDS < GW_HEAD || WWORDS < EMB_ROWG_W))
            $display("sequencer_vec: WARNING WEIGHT_STREAM_PER_LAYER=1 needs WWORDS >= max(GW_BLK=%0d, GW_HEAD=%0d, EMB_ROWG_W=%0d), got WWORDS=%0d -- g_wbase/emb_addr_w will silently truncate/wrap into weight_bank_tdp (EMB_ROWG_W, not GW_EMB, is the embed path's real WWORDS driver now -- see EMB_ROWG_W's own comment)", GW_BLK, GW_HEAD, EMB_ROWG_W, WWORDS);
        if (KV_DDR_BACKED && WEIGHT_STREAM_PER_LAYER &&
            (KV_DDR_BASE < WEIGHTS_DDR_BASE + WEIGHT_IMAGE_BYTES) &&
            (WEIGHTS_DDR_BASE < KV_DDR_BASE + KV_IMAGE_BYTES))
            $display("sequencer_vec: WARNING KV_DDR_BASE=%0d..%0d overlaps WEIGHTS_DDR_BASE=%0d..%0d in DDR3 -- kv_bank_ddr writes will silently corrupt staged weight bytes (this is the real-hardware bug that caused reproducible generate-token-8 corruption before KV_DDR_BASE was separated)", KV_DDR_BASE, KV_DDR_BASE+KV_IMAGE_BYTES, WEIGHTS_DDR_BASE, WEIGHTS_DDR_BASE+WEIGHT_IMAGE_BYTES);
    end
`endif

    gemv_banked_resident_vec #(.LANES(LANES), .P(P), .MMAX(GEMV_MMAX), .KMAX(1024), .RLAT(2),
                  .WWORDS(WWORDS), .K2(1), .MEM_PRIMITIVE(MEM_PRIMITIVE)) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ldrst | wl_rst | wld_ldb_rst),
        .w_we(wl_we | wld_ldb_we),
        .w_data(wl_we ? wl_data : wld_ldb_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .gdone(gv_gdone),
        .rd_addr(gv_rdaddr[$clog2(GEMV_MMAX/P)-1:0]), .y_out(gv_yout),
        .emb_sel(emb_sel_w), .emb_addr(emb_addr_w), .emb_pair(emb_pair));

    // ---- vec_dequant (P lanes, runtime frac) -----------------------------------
    reg               dq_vin;
    reg signed [6:0]  dq_frac;
    reg  [P*32-1:0]   dq_gemvy;
    reg  [P*24-1:0]   dq_mant;
    reg  [P*8-1:0]    dq_exp;
    wire              dq_vout;
    wire [P*32-1:0]   dq_out;
    vec_dequant #(.P(P)) u_dq (
        .clk(clk), .rst(rst), .in_valid(dq_vin), .frac(dq_frac),
        .gemvy(dq_gemvy), .mant(dq_mant), .exp(dq_exp),
        .out_valid(dq_vout), .dq_out(dq_out));

    // ---- vec_gelu (P lanes, Q4.12 in/out, 3-cyc latency) -----------------------
    reg             gl_vin;
    reg [P*16-1:0]  gl_x;
    wire            gl_vout;
    wire [P*16-1:0] gl_y;
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(gl_vin), .x(gl_x),
        .out_valid(gl_vout), .y(gl_y));

    // ---- SINGLE vec_attn_w engine (Genesys2 port): was twin engines (A+B)
    // running a head PAIR concurrently on kv_bank's two read ports -- found to
    // cost 247 DSP48E1 + ~44K LUTs PER ENGINE at synth (494 DSPs = 61% of the
    // xc7k325t's 840, LUTs alone put the whole design at 123.75% before even
    // reaching place&route), independent of LANES (which only sizes the GEMV
    // weight bank, 0 DSPs). Collapsed to one engine processing all NHEAD heads
    // serially -- a real throughput cost (no more pair-concurrency) in exchange
    // for roughly halving attention's DSP/LUT footprint. kv_bank's second read
    // port is tied permanently idle at the instantiation below (kv_bank.sv
    // itself is UNCHANGED -- shared with the KV260 gate). See PORT-NOTES.md.
    reg               at_startA, at_ldvA;
    reg [8:0]         at_tcount;
    wire              at_kdoneA, at_ctxvA, at_doneA;
    wire [6:0]        at_ctxidxA;
    wire [P*32-1:0]   at_ctxdataA;
    reg [1:0]  hh;                          // the single engine's current head, 0..NHEAD-1
    reg        adone_s;                     // sticky engine-done flag for the current head
    reg [8:0]  wi;                          // load-address counter (runs ahead)
    reg [8:0]  wic;                         // accepted-word counter (consume stage)
    reg        wiv;                         // (legacy)
    localparam integer HR = HEAD_DIM / P;
    wire [BUSW-1:0] aw_src = hh*HR + wi;
    reg  [P*32-1:0] q_data_q;                    // registered WITH at_ldv
    reg             ldv0;                        // addr-stage valid (1 ahead of at_ldv)
    // ctx strobes land in a catch buffer, drained into ctxv_bank in S_CDR (HR
    // cycles/head). A direct ctxv_bank write from the catcher made the bank
    // 2W1R -> synthesis fell back to REGISTERS (take-5); the buffer keeps it 1W1R.
    reg [P*32-1:0] ctxbufA [0:HR-1];
    reg [4:0]  cdr;                         // ctx drain counter (0..HR-1)
    // KV-write FEEDER (R4f, extended to ALL writes): every (head, K/V) of the
    // new position quantises into kv_bank through this mini-FSM, STARTED AT THE
    // QKV DISPATCH so the K writes hide under the qkv GEMV+readback. Ordering
    // (general NHEAD -- Genesys2 port, single engine): K h0..h(NHEAD-1) then
    // V h0..h(NHEAD-1), plain ascending — K-first so attention can start on
    // kvp_done counts (head h needs kvp_done >= h+1, S_AST). The original
    // NHEAD=4 twin-engine build wrote V in a hand-tuned shuffled order
    // (V1,V0,V3,V2) so each engine's V-write landed in a gap in the OTHER
    // engine's read schedule; with a single engine that micro-optimization
    // has no target to hide under anyway, so ascending order is simply
    // correct — see PORT-NOTES.md.
    // Data-ready gate: row fr of the current vector may be consumed only once
    // the qkv readback has written it (kvw_src < qkv_wrow). The qkv_bank read
    // port is arbitrated: S_ALD's q stream wins; the feeder pauses (addr
    // presented only when granted, the in-flight beat completes regardless).
    // V reads are interlocked: at_kdone latches a pending flag, kb_rstart
    // fires once kvp_done covers this head's V write (vneedA).
    localparam [1:0] KF_IDLE=2'd0, KF_S=2'd1, KF_F=2'd2, KF_W=2'd3;
    reg [1:0] kvf_st;
    reg       kvf_active;
    reg [3:0] kvp_done;                     // completed (head,K/V) writes 0..2*NHEAD
    reg [BUSW-1:0] qkv_wrow;                    // qkv rows committed by G_RB so far
    reg       vpA;                          // V-read pending (kdone seen, write not)
    wire      kvf_grant = (st != S_ALD) && (st != S_KVW_F);
    // kvp_done value once V for the current head has committed: V for head h
    // is the (NHEAD+h+1)th write in the ascending K-then-V order above.
    wire [3:0] vneedA = NHEAD[3:0] + {2'b0, hh} + 4'd1;

    // ---- kv_bank: the on-chip K8/V8 cache making decode faithful ---------------
    reg         kb_wstart, kb_wvalid, kb_rstart, kb_rkv;
    reg  [P*32-1:0] kb_wdata_q;                  // registered with kb_wvalid (L_FEED idiom)
    reg  [1:0]  kvw_h;                           // KV-write head loop
    reg         kvw_kv;                          // 0 = K, 1 = V
    wire        kb_wdone, kb_rvalid, kb_rdone;
    wire [HEAD_DIM*32-1:0] kb_rdata;             // one dequantised position per beat
    reg  [1:0]  alds;                            // (legacy, unused post single-engine collapse)
    wire [BUSW-1:0] kvw_src = (kvw_kv ? 2*D/P : D/P) + kvw_h*HR
                          + {{(11-$clog2(ROWSM+1)){1'b0}}, fr};
    // kv_bank's second read port is a fixed interface (shared, unmodified file
    // -- see below) permanently idled here: rd2_start tied low so Vivado can
    // constant-propagate/eliminate the second port's logic during synthesis.
    //
    // KV_DDR_BACKED selects kv_bank (on-chip, default) or kv_bank_ddr
    // (DDR3-streamed) at ELABORATION time -- see the parameter's own comment.
    // Both branches drive the SAME internal signals (kb_wstart..kb_rdone);
    // sequencer_vec's own FSM is unmodified either way (confirmed event-
    // driven on wq_done/rd_done throughout, not cycle-counted, before this
    // generate was added -- see PORT-NOTES.md).
    generate
    if (KV_DDR_BACKED) begin : g_kvb_ddr
        kv_bank_ddr #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER),
                      .TMAX(TMAX), .KBITS(8), .ADDR_W(29), .DATA_W(256),
                      .KV_DDR_BASE(KV_DDR_BASE)) u_kvb (
            .clk(clk), .rst(rst),
            .wq_start(kb_wstart), .wq_layer(blk), .wq_kv(kvw_kv), .wq_head(kvw_h),
            .wq_pos(pos), .wq_valid(kb_wvalid), .wq_data(kb_wdata_q), .wq_done(kb_wdone),
            .wr_pkt_valid(kv_wr_pkt_valid), .wr_pkt_ready(kv_wr_pkt_ready),
            .wr_pkt_addr(kv_wr_pkt_addr), .wr_pkt_data(kv_wr_pkt_data), .wr_pkt_mask(kv_wr_pkt_mask),
            .wr_ack_valid(kv_wr_ack_valid), .wr_ack_ready(kv_wr_ack_ready),
            .rd_start(kb_rstart), .rd_layer(blk), .rd_kv(kb_rkv), .rd_head(hh),
            .rd_tcount(pos + 9'd1),
            .rd_valid(kb_rvalid), .rd_data(kb_rdata), .rd_done(kb_rdone),
            .rd_req_valid(kv_rd_req_valid), .rd_req_ready(kv_rd_req_ready), .rd_req_addr(kv_rd_req_addr),
            .rd_ret_valid(kv_rd_ret_valid), .rd_ret_ready(kv_rd_ret_ready), .rd_ret_data(kv_rd_ret_data));
    end else begin : g_kvb_resident
        kv_bank #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER),
                  .TMAX(TMAX), .KBITS(8), .MEM_PRIMITIVE(MEM_PRIMITIVE)) u_kvb (
            .clk(clk), .rst(rst),
            .wq_start(kb_wstart), .wq_layer(blk), .wq_kv(kvw_kv), .wq_head(kvw_h),
            .wq_pos(pos), .wq_valid(kb_wvalid), .wq_data(kb_wdata_q), .wq_done(kb_wdone),
            .rd_start(kb_rstart), .rd_layer(blk), .rd_kv(kb_rkv), .rd_head(hh),
            .rd_tcount(pos + 9'd1),
            .rd_valid(kb_rvalid), .rd_data(kb_rdata), .rd_done(kb_rdone),
            .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head(2'd0),
            .rd2_tcount(9'd0),
            .rd2_valid(), .rd2_data(), .rd2_done());
        // KV_DDR_BACKED=0: DMA ports are unused, tied to inert/idle values
        // (never asserts a request, always accepts an ack/return it will
        // never actually receive) so the module elaborates cleanly with no
        // dangling/undriven top-level outputs.
        assign kv_wr_pkt_valid = 1'b0;
        assign kv_wr_pkt_addr  = 29'd0;
        assign kv_wr_pkt_data  = 256'd0;
        assign kv_wr_pkt_mask  = 32'hFFFFFFFF;
        assign kv_wr_ack_ready = 1'b1;
        assign kv_rd_req_valid = 1'b0;
        assign kv_rd_req_addr  = 29'd0;
        assign kv_rd_ret_ready = 1'b1;
    end
    endgenerate

    vec_attn_w #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(TMAX)) u_attnA (
        .clk(clk), .rst(rst), .start(at_startA), .tcount(at_tcount),
        .q_valid(at_ldvA), .q_data(q_data_q),
        .kv_valid(kb_rvalid), .kv_data(kb_rdata),
        .k_done(at_kdoneA),
        .ctx_valid(at_ctxvA), .ctx_idx(at_ctxidxA), .ctx_data(at_ctxdataA),
        .done(at_doneA));

    // ---- callable GEMV / LN parameter registers --------------------------------
    reg [19:0] g_wbase;            // weight base
    reg [BUSW-1:0] g_m, g_k;           // dims
    reg [1:0]  g_asrc;             // act source: 0 lnout1, 1 ctxv(>>3), 2 lnout2, 3 mlpbuf
    reg [5:0]  g_asel;             // inv_sact index
    reg signed [6:0] g_frac;       // dequant frac
    reg [11:0] g_dqrow;            // dequant channel-row base (up to NLAYER*DQ_BLK/P = 1152)
    reg [2:0]  g_dst;              // dest: 0 qkv,1 attn,2 mlpbuf(sat16),3 mlp,4 head
    reg [4:0]  g_ret;             // return state after the GEMV
    // LN gamma set (0=ln1.0,1=ln2.0,...,2*NLAYER=ln_f). Was `reg [3:0]` (max 15) --
    // a REAL bug, same class as the earlier GAMMA_N/NSACT "hardcoded for NLAYER=4"
    // fixes but MISSED then: at NLAYER=8, `l_gbase<=NLAYER*2`=16 silently wrapped to
    // 0, so the FINAL LayerNorm (ln_f) read BLOCK 0's ln1 gamma instead of ln_f's
    // own gains -- invisible at NLAYER<=7 (2*NLAYER<=14 fits 4 bits), and invisible
    // under GREEDY decode even at NLAYER=8 (wrong-but-still-reasonable per-channel
    // gamma distorts head-logit MAGNITUDES channel-by-channel but rarely flips which
    // channel is largest), but fatal to on-chip Gumbel sampling (found via real-
    // hardware garbled chat -> simulation-gate reproduction -> per-channel dequant
    // trace -> gamma_w content check (correct!) -> this register's width). Sized
    // from GAMMA_N (same "compute from shape params, not a guessed constant"
    // convention as GAMMA_N/NSACT themselves) instead of another hardcoded width.
    reg [$clog2(GAMMA_N)-1:0] l_gbase;
    reg        l_dst;              // LN dest: 0 lnout1, 1 lnout2
    reg [4:0]  l_ret;             // return state after the LN

    // ---- temporaries (PLAIN regs — safe targets for variable part-selects) -----
    reg signed [63:0]  lntmp;
    reg signed [95:0]  aq_prod, aq_sh;
    reg signed [31:0]  aq_int;
    reg signed [95:0]  aq_prod_r [0:P-1];     // G_AQ stage-1 product registers
    reg                aq_neg_r  [0:P-1];
    reg signed [63:0]  lnt_r     [0:P-1];     // G_AQ stage-0 source registers
    // TIMING (5ns cone g_asel -> aq_prod_r): the 17-deep inv_sact LUT read was
    // muxed combinationally INSIDE the multiply cycle. g_asel is constant for a
    // whole GEMV call (set at dispatch, >=2 cycles before the first civ1 multiply),
    // so a free-running registered read is always settled in time.
    reg signed [63:0]  isact_r;
    reg [BUSW-1:0]         cid1, cid2;  reg civ1, civ2;
    // TIMING (5ns cone aq_prod_r DSP-out -> gv_xdata): the 96-bit conditional
    // negate+round+shift and the clip+pack were one cycle. Stage 2a registers the
    // rounded value (only [31:0] is ever consumed downstream — identical
    // semantics), stage 2b clips+packs. +1 cycle per AQ phase.
    reg signed [31:0]  aq_sh_r [0:P-1];
    reg [BUSW-1:0]         cid3;  reg civ3;
    reg signed [31:0]  cb, dqv, hv;
    reg [P*32-1:0]  ww, sw, dword, hw;
    reg [P*8-1:0]   aqw;                     // P-wide act-quant word
    reg [P*16-1:0]  mword;                   // P-wide sat16 word (GELU feed)
    reg [P*64-1:0]  gsw;
    reg [P*24-1:0]  mwr;
    reg [P*8-1:0]   ewr;
    reg signed [63:0] gl_sh;
    reg [BUSW-1:0] rb0, rb1, rb2; reg rv0, rv1, rv2;
    reg [$clog2(ROWSM+1)-1:0] gor;
    integer pp;

    // free-running registered inv_sact read (see isact_r note above)
    always @(posedge clk) isact_r <= inv_sact[g_asel];

    // ---- gumbel precompute FSM (fills gumbel_bank during the head GEMV) ---------
    // One LUT read/cycle: advance the xorshift, present (next_state>>22) as the LUT
    // address, place the registered LUT value one cycle later into lane gj%P. After
    // P lanes a wide word is written to gumbel_bank[row]. VOCAB advances total — the
    // bit-exact match to gumbel.GumbelRng (state advances BEFORE each logit's noise).
    reg                gpre_active, gpre_done;
    // gj/gj_d need to hold the SENTINEL value VOCAB itself, not just valid
    // indices 0..VOCAB-1 -- VIDXW=$clog2(VOCAB) only guarantees the latter.
    // At VOCAB=16384 (an exact power of 2) that's a real, silent bug: the
    // advance-stage exit check below used to compare against
    // `VOCAB[VIDXW-1:0]`, which truncates 16384 to 0 in 14 bits -- since gj
    // starts at 0, `gj != 0` is false on the very FIRST check, the advance
    // body never runs even once, gumbel_bank never fills, gpre_done never
    // fires, and S_ARGMAX (gated on gpre_done) hangs forever. Only power-
    // of-2 VOCAB values hit this (VOCAB=1900's own VIDXW=11 bits already
    // holds 1900 exactly, no truncation) -- found via a real sampling-mode
    // hang (greedy/seed=0 never exercises this precompute path at all,
    // which is why the seed=0 bit-exact gate never caught it). Widened by
    // one bit (matching this file's own "+1 for a sentinel" convention,
    // e.g. RIDXW=$clog2(MAXROWS+1)) so gj can represent VOCAB itself
    // without wrapping after the last valid index increments past
    // VOCAB-1.
    reg [VIDXW:0]      gj;                  // logit counter 0..VOCAB (advance stage)
    reg [VIDXW:0]      gj_d;                // delayed counter (place stage, LUT-read aligned)
    reg                gj_dv;               // place-stage valid
    reg [P*32-1:0]     gpre_word;           // P-wide staging word being assembled
    integer            gl_lane;
    // combinational xorshift32 of the live state (== gumbel.xorshift32)
    wire [31:0] xs1   = rng_state ^ (rng_state << 13);
    wire [31:0] xs2   = xs1 ^ (xs1 >> 17);
    wire [31:0] xs_ns = xs2 ^ (xs2 << 5);            // next state
    wire [9:0]  gpre_idx_w = xs_ns[31:22];           // LUT index = next_state >> 22

    reg signed [33:0] best_val; reg [VIDXW-1:0] best_idx, hidx;  // widened: logit32 + gumbel32
    reg [$clog2(ARROWS+1)-1:0] ar;           // argmax row counter
    // Compare values are WIDENED to signed 34-bit: a head logit (signed int32) plus a
    // gumbel noise value (signed int32) can exceed int32; first-index-wins ties kept.
    reg signed [33:0] wm_val;  reg [VIDXW-1:0] wm_idx;        // word-max tree temporaries
    reg signed [33:0] am_val;  reg [VIDXW-1:0] am_idx;        // stage-1 registers
    reg [$clog2(ARROWS+1)-1:0] amd;  reg amv;
    // argmax 3-stage pipeline (the head_bank -> compare chain was the OOC critical path):
    // stage A registers the row + its gumbel noise, stage B halves P -> P/2 (with noise
    // added), stage C reduces + running best.
    reg [P*32-1:0] hw_r;
    reg [P*32-1:0] gw_r;                      // stage-A registered gumbel word for the row
    reg signed [33:0] pv0, pv1, pv2, pv3;     // P/2 pair maxima
    reg [VIDXW-1:0]   pi0, pi1, pi2, pi3;
    reg [$clog2(ARROWS+1)-1:0] ad1;  reg av1;
    reg signed [33:0] va, vb;
    reg [VIDXW-1:0]   ia, ib;


    // ---- synchronous reads (one read register per memory, address muxed) -------
    wire [BUSW-1:0] rbr = rd_addr >> LSH;            // board readback row (idle only)
    // fr/ar/eb_row below are all narrower than BUSW (fr/eb_row <=
    // $clog2(ROWSM+1) bits, ar <= $clog2(ARROWS+1) bits, both <= BUSW by
    // construction) -- plain assignment zero-extends correctly, no manual
    // padding needed (the old hardcoded-11 padding formulas here computed
    // a negative concatenation-repeat count and failed to compile once
    // ARROWS+1 needed more than 11 bits, at VOCAB=16384 -- see BUSW's own
    // comment above).
    wire [BUSW-1:0] xres_ra   = (st==L_FEED) ? fr :
                            (st==S_RES1 || st==S_RES2) ? ci : rbr;
    // G_AQ counts P-rows directly (one wide word per cycle into the GEMV boundary)
    wire [BUSW-1:0] lnout1_ra = (st==G_AQ) ? ci : rbr;
    wire [BUSW-1:0] lnout2_ra = (st==G_AQ) ? ci : rbr;
    wire [BUSW-1:0] ctxv_ra   = (st==G_AQ) ? ci : rbr;
    wire [BUSW-1:0] mlpbuf_ra = (st==G_AQ) ? ci : rbr;
    wire [BUSW-1:0] qkv_ra    = (st==S_ALD)   ? aw_src :
                            (st==S_KVW_F) ? kvw_src :
                            (kvf_active && kvf_st==KF_F) ? kvw_src : rbr;
    wire [BUSW-1:0] attn_ra   = (st==S_RES1) ? ci : rbr;
    wire [BUSW-1:0] mlp_ra    = (st==S_RES2) ? ci : rbr;
    wire [BUSW-1:0] head_ra   = (st==S_ARGMAX) ? ar : rbr;
    // LN FEED FUSION: every LN call is fed DURING the state that produces its
    // input rows (S_EMB commit / S_RES1 / S_RES2), so the gamma read tracks the
    // producer's row counter (eb_row in S_EMB — issued one cycle ahead of the
    // commit stage; ci elsewhere). L_GAM/L_FEED are retired.
    wire [BUSW-1:0] gam_fr    = (st==S_EMB) ? eb_row : ci;

    always @(posedge clk) begin
        xres_r   <= xres_bank  [xres_ra];
        lnout1_r <= lnout1_bank[lnout1_ra];
        lnout2_r <= lnout2_bank[lnout2_ra];
        qkv_r    <= qkv_bank   [qkv_ra];
        ctxv_r   <= ctxv_bank  [ctxv_ra];
        attn_r   <= attn_bank  [attn_ra];
        mlpbuf_r <= mlpbuf_bank[mlpbuf_ra];
        mlp_r    <= mlp_bank   [mlp_ra];
        head_r   <= head_bank  [head_ra];
        gam_r    <= gamma_w  [l_gbase*EROWS + gam_fr];
        // gumbel_bank read tracks head_bank in S_ARGMAX (same row); LUT read for the
        // precompute is addressed by the new xorshift state (advance stage).
        gumbel_r     <= gumbel_bank[head_ra];
        gumbel_lut_r <= gumbel_lut[gpre_idx_w];
    end

    always @(posedge clk) begin
        ln_start<=1'b0; ln_vin<=1'b0; gv_ldrst<=1'b0; gv_xwe<=1'b0; gv_start<=1'b0;
        dq_vin<=1'b0; gl_vin<=1'b0; done<=1'b0;
        at_startA<=1'b0; at_ldvA<=1'b0;
        kb_wstart<=1'b0; kb_wvalid<=1'b0; kb_rstart<=1'b0;
        if (rst) begin
            st<=S_IDLE; ci<=0; fr<=0; orow<=0; dor<=0; gor<=0;
            rv0<=0; rv1<=0; rv2<=0;
            civ<=0; frv<=0; arv<=0; wiv<=0; etp<=1'b0; eb_v<=1'b0; eb2_v<=1'b0;
            kvf_st<=KF_IDLE; kvf_active<=1'b0;
            kvp_done<=4'd0; qkv_wrow<=11'd0; vpA<=1'b0;
            rng_state<=32'd0; smp_en<=1'b0;
            gpre_active<=1'b0; gpre_done<=1'b0; gj<=9'd0; gj_dv<=1'b0;
            strw_armed<=1'b0; strw_settle<=6'd0; wldi_start<=1'b0;
        end else begin
            // SEED write: load the persistent xorshift state + enable sampling. Lives
            // OUTSIDE the FSM case (host writes it once between GOs, core idle). seed==0
            // => greedy. Persists across gen GOs (the rng advances during each head GEMV).
            if (seed_we) begin rng_state <= seed; smp_en <= (seed != 32'd0); end
            case (st)
                S_IDLE: if (go) begin
                    fr<=0; frv<=0; etp<=1'b0; emb_ph<=1'b0; eb_v<=1'b0; eb2_v<=1'b0; blk<=4'd0;
                    // block-0 LN1 is FED during the S_EMB commits (LN fusion)
                    l_gbase<=4'd0; l_dst<=1'b0;
                    ln_start<=1'b1;
                    // Under streaming, NEITHER the embed tables NOR block 0's
                    // weights are resident just because the LAST token's
                    // forward pass ended somewhere else (head, or mid-block on
                    // an interrupted run) -- EVERY token needs a fresh embed
                    // reload before S_EMB can run at all. l_ret is armed here
                    // for the SECOND reload (block 0's weights, needed after
                    // S_EMB->L_COLL) but strw_ret for THAT hop is set fresh at
                    // S_EMB's own completion, not here -- this cycle's
                    // strw_ret is needed IMMEDIATELY for the embed reload
                    // instead (S_STRW is entered directly, not via L_COLL,
                    // for this specific hop) and the two would otherwise
                    // collide in the same register on the same cycle.
                    if (WEIGHT_STREAM_PER_LAYER) begin
                        l_ret<=S_STRW;
                        strw_ret<=S_EMB;
                        st<=S_STRW;
                    end else begin
                        l_ret<=S_QKVRET;
                        st<=S_EMB;
                    end
                end
                // ---- embed -> xres via the weight bank's embed port (log §36 plan 2):
                // tok row then pos row per fr. Non-streaming: serial interleave,
                // 1 fetch/cycle, ~2*EROWS+2 cyc (whole table already resident).
                // Streaming: one row-group reload+read PASS per phase (emb_ph),
                // ~EROWS+2 cyc/pass, plus the S_STRW reload latency between passes
                // -- see EMB_ROWG_W's comment above for why this can't interleave.
                S_EMB: if (WEIGHT_STREAM_PER_LAYER) begin
                    // issue stage: emb_addr_w is combinational from fr this cycle
                    // (LOCAL row index within the just-reloaded row-group)
                    eb_v   <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    eb_sel <= fr[2:0] & RSELM;
                    eb_row <= fr;
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    // select stage: register the pair-slot mux of last cycle's pair
                    if (eb_v) begin
                        epr = emb_pair;                         // plain-reg copy first
                        erw_r <= epr[eb_sel*(P*32) +: P*32];
                    end
                    eb2_v <= eb_v; eb2_row <= eb_row;
                    // commit stage, one cycle behind. Phase 0 (tok row): hold it in
                    // tok_hold -- the pos row it sums with isn't reloaded yet, and
                    // xres/LN can't fire until the sum is ready. Phase 1 (pos row):
                    // sum with the HELD tok row and commit, same as the
                    // non-streaming path's tok+pos sum. LN FUSION unchanged: each
                    // committed row also feeds the LN (block-0 LN1 loads during
                    // the embed, L_FEED dies) -- gated on phase 1 only, since
                    // that's the only phase that ever produces an xres commit.
                    ln_vin <= eb2_v && emb_ph;
                    ln_g   <= gam_r;
                    if (eb2_v) begin
                        if (!emb_ph) begin
                            tok_hold[eb2_row] <= erw_r;
                        end else begin
                            for (pp=0; pp<P; pp=pp+1)
                                ww[pp*32 +: 32] = $signed(tok_hold[eb2_row][pp*32 +: 32])
                                                + $signed(erw_r[pp*32 +: 32]);
                            xres_bank[eb2_row] <= ww;
                            ln_x <= ww;
                        end
                        if (eb2_row==ROWS-1) begin
                            fr<=0; frv<=0; eb_v<=1'b0; eb2_v<=1'b0;
                            if (!emb_ph) begin
                                // tok row fully held on-chip -- now reload+read
                                // the pos row (same on-chip words, overwriting
                                // the tok row weight_loader_ddr just wrote --
                                // already safely captured in tok_hold above)
                                emb_ph   <= 1'b1;
                                strw_ret <= S_EMB;
                                st       <= S_STRW;
                            end else begin
                                emb_ph <= 1'b0;   // reset for next token
                                if (dbg_stop==2'd1) st<=S_FIN;   // DEBUG: stop after embed
                                else begin
                                    // arm strw_ret fresh for the L_COLL->S_STRW
                                    // hop l_ret already points at (set back in
                                    // S_IDLE) -- this is ALWAYS the block-0
                                    // weight reload (S_EMB only ever precedes
                                    // block 0), never the head case.
                                    strw_ret<=S_QKVRET;
                                    orow<=0; st<=L_COLL;
                                end
                            end
                        end
                    end
                end else begin
                    // issue stage: emb_addr_w is combinational from etp/fr this cycle
                    eb_v   <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    eb_tp  <= etp;
                    eb_sel <= emb_row_w[2:0] & RSELM;
                    eb_row <= fr;
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) begin
                        if (etp) begin etp <= 1'b0; fr <= fr + 1'b1; end
                        else etp <= 1'b1;
                    end
                    // select stage: register the pair-slot mux of last cycle's pair
                    if (eb_v) begin
                        epr = emb_pair;                         // plain-reg copy first
                        erw_r <= epr[eb_sel*(P*32) +: P*32];
                    end
                    eb2_v <= eb_v; eb2_tp <= eb_tp; eb2_row <= eb_row;
                    // commit stage: tacc hold / tok+pos sum, one cycle behind.
                    // LN FUSION: each committed row is ALSO fed to the LN (ln_x
                    // = the same sum, ln_g = the gamma row issued at the eb
                    // stage) — block-0 LN1 loads during the embed, L_FEED dies.
                    ln_vin <= eb2_v && eb2_tp;
                    ln_g   <= gam_r;
                    if (eb2_v) begin
                        if (!eb2_tp) tacc <= erw_r;             // hold the tok row
                        else begin                              // pos row: sum + commit
                            for (pp=0; pp<P; pp=pp+1)
                                ww[pp*32 +: 32] = $signed(tacc[pp*32 +: 32])
                                                + $signed(erw_r[pp*32 +: 32]);
                            xres_bank[eb2_row] <= ww;
                            ln_x <= ww;
                            if (eb2_row==ROWS-1) begin
                                fr<=0; frv<=0; eb_v<=1'b0; eb2_v<=1'b0; etp<=1'b0;
                                if (dbg_stop==2'd1) st<=S_FIN;       // DEBUG: stop after embed
                                else begin
                                    // arm strw_ret fresh for the L_COLL->S_STRW
                                    // hop l_ret already points at (set back in
                                    // S_IDLE) -- this is ALWAYS the block-0
                                    // weight reload (S_EMB only ever precedes
                                    // block 0), never the head case.
                                    if (WEIGHT_STREAM_PER_LAYER) strw_ret<=S_QKVRET;
                                    orow<=0; st<=L_COLL;
                                end
                            end
                        end
                    end
                end
                // ================= callable LayerNorm =========================
                L_GAM: begin ln_start<=1'b1; fr<=0; frv<=0; st<=L_FEED; end
                L_FEED: begin
                    frd <= fr; frv <= (fr != ROWS[$clog2(ROWSM+1)-1:0]);
                    if (fr != ROWS[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    ln_vin <= frv;
                    ln_x   <= xres_r;
                    ln_g   <= gam_r;
                    if (frv && frd==ROWS-1) begin orow<=0; fr<=0; frv<=0; st<=L_COLL; end
                end
                L_COLL: begin
                    if (ln_yv) begin
                        if (l_dst==1'b0) lnout1_bank[orow] <= ln_y;
                        else             lnout2_bank[orow] <= ln_y;
                        if (orow==ROWS-1) st<=l_ret; else orow<=orow+1'b1;
                    end
                end
                // qkv GEMV setup (after LN1) and the GEMV call ------------------
                // The KV FEEDER is armed HERE: its K writes ride the qkv
                // readback (data-ready gated), so attention starts as soon as
                // the pair's K writes commit (R1 contract kept per-read via
                // the kvp_done gates + V interlocks, not a serial S_KVW block).
                S_QKVRET: begin
                    g_wbase<=WEIGHT_STREAM_PER_LAYER ? WB_QKV : (blk*GW_BLK + WB_QKV);
                    g_m<=D3[BUSW-1:0]; g_k<=D[BUSW-1:0]; g_asrc<=2'd0;
                    g_asel<=blk*4 + 6'd0; g_frac<=7'd16; g_dqrow<=blk*DQB_P + DR_QKV; g_dst<=3'd0;
                    g_ret<=S_AST; ci<=0; civ<=0;
                    hh<=2'd0;
                    kvw_h<=2'd0; kvw_kv<=1'b0;
                    kvf_active<=1'b1; kvf_st<=KF_S; kvp_done<=4'd0; qkv_wrow<=11'd0;
                    vpA<=1'b0;
                    gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ============ KV quant-write: k then v, per head ==============
                S_KVW_S: begin
                    kb_wstart<=1'b1; fr<=0; frv<=0; st<=S_KVW_F;
                end
                S_KVW_F: begin
                    frd <= fr; frv <= (fr != HR[$clog2(ROWSM+1)-1:0]);
                    if (fr != HR[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    kb_wvalid  <= frv;
                    kb_wdata_q <= qkv_r;
                    if (frv && frd==HR-1) begin fr<=0; frv<=0; st<=S_KVW_W; end
                end
                // main FSM writes heads 0..1 only; heads 2..3 go to the FEEDER,
                // which runs during pair-0's attention (R4f).
                S_KVW_W: if (kb_wdone) begin
                    if (!kvw_kv) begin kvw_kv<=1'b1; st<=S_KVW_S; end
                    else if (kvw_h == 2'd0) begin kvw_h<=2'd1; kvw_kv<=1'b0; st<=S_KVW_S; end
                    else begin
                        kvw_h<=2'd2; kvw_kv<=1'b0; kvf_active<=1'b1; kvf_st<=KF_S;
                        hh<=2'd0; st<=S_AST;
                    end
                end
                // ================= callable GEMV ==============================
                G_AQ: begin    // act-quant: P lanes/cycle, 3-STAGE (mux | mult | round+sat).
                    // Source mux + ctx-rounding registered BEFORE the 64x40 multiply —
                    // the BRAM-read -> mux -> DSP chain was the post-pipeline critical path.
                    cid <= ci; civ <= (ci != (g_k >> LSH));
                    cid1 <= cid; civ1 <= civ;
                    cid2 <= cid1; civ2 <= civ1;
                    cid3 <= cid2; civ3 <= civ2;
                    if (ci != (g_k >> LSH)) ci <= ci + 1'b1;
                    if (civ) begin                 // stage 0: source mux -> lnt_r
                        for (pp=0; pp<P; pp=pp+1) begin
                            case (g_asrc)
                                2'd0: lntmp = $signed(lnout1_r[pp*64 +: 64]);
                                2'd1: begin
                                    cb = ctxv_r[pp*32 +: 32];
                                    // ctx Q6.25 -> Q.22 rsh_round (matches _proj_after_attn)
                                    if ($signed({{32{cb[31]}}, cb}) >= 0)
                                        lntmp = ($signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC);
                                    else
                                        lntmp = -((-$signed({{32{cb[31]}}, cb}) + 64'sd4) >>> (RESID_FRAC-LN_OUT_FRAC));
                                end
                                2'd2: lntmp = $signed(lnout2_r[pp*64 +: 64]);
                                default: lntmp = $signed(mlpbuf_r[pp*64 +: 64]);
                            endcase
                            lnt_r[pp] <= lntmp;
                        end
                    end
                    if (civ1) begin                // stage 1: multiply (DSP-retimed)
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_prod_r[pp] <= $signed(lnt_r[pp]) * $signed(isact_r);
                            aq_neg_r[pp]  <= (lnt_r[pp] < 0);
                        end
                    end
                    if (civ2) begin                // stage 2a: round (negate/add/shift)
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_prod = aq_prod_r[pp];
                            if (!aq_neg_r[pp])
                                aq_sh = (aq_prod + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH);
                            else
                                aq_sh = -(((-aq_prod) + (96'sd1 <<< (LN_OUT_FRAC+ISH-1))) >>> (LN_OUT_FRAC+ISH));
                            aq_sh_r[pp] <= aq_sh[31:0];
                        end
                    end
                    if (civ3) begin                // stage 2b: clip + pack -> INT8 lane
                        for (pp=0; pp<P; pp=pp+1) begin
                            aq_int = aq_sh_r[pp];
                            if (aq_int>127) aq_int=127; if (aq_int<-128) aq_int=-128;
                            aqw[pp*8 +: 8] = aq_int[7:0];
                        end
                        gv_xwe<=1'b1; gv_xdata<=aqw;
                        // AQ/RUN OVERLAP (the §22/§24-proven SB lever, now here):
                        // start the GEMV WITH the first act-row write. xmem row r
                        // is written at edge firstxwe+1+r; the MAC (K2=1) first
                        // reads row r at edge firstxwe+2+4r — the feed stays >=1
                        // row ahead for all r, so the values the MAC consumes are
                        // identical and the AQ feed hides under the MAC compute.
                        if (cid3 == 11'd0) begin
                            gv_m<=g_m; gv_k<=g_k;
                            gv_wbase<=g_wbase[$clog2(WWORDS)-1:0];
                            gv_start<=1'b1;
                        end
                        if (cid3==(g_k >> LSH)-1) begin
                            ci<=0; civ<=0; civ1<=0; civ2<=0; civ3<=0; st<=G_WAIT;
                        end
                    end
                end
                // G_RUN retired: gv_start now fires inside G_AQ (AQ/RUN overlap)
                G_WAIT: if (gv_gdone != 4'd0) begin
                    // groupwise RB overlap: drain begins on the FIRST group commit,
                    // not gv_done — rows of committed groups are final (the MAC
                    // writes ymem[g] exactly once, at group g's commit).
                    ci<=0; gv_rdaddr<=0; rv0<=0; rv1<=0; rv2<=0; dor<=0; st<=G_RB;
                end
                G_RB: begin
                    // FUSED readback -> dequant -> dest. ymem readback is 2-cycle
                    // (rd_addr -> rd_word -> y_out): gv_yout for address rb0 lands at rv2.
                    // dqm/dqe BRAM reads at rb1 land in the same cycle -> stream the
                    // dequant unit at 1 word/cycle. For g_dst==2 (mlp hidden), dequant
                    // output feeds vec_gelu in flight; exit when GELU drains.
                    // Row issue is gated on the row's GROUP being committed
                    // (ci>>GRPSH < gv_gdone): the drain rides the MAC, stalling
                    // (rv0=0 bubbles) when it catches the in-flight group.
                    if (ci < ((g_m + P-1) >> LSH)) begin
                        if ({4'b0, ci} >> GRPSH < {11'b0, gv_gdone}) begin
                            gv_rdaddr<=ci; rb0<=ci; ci<=ci+1'b1;
                        end
                    end else gv_rdaddr <= ((g_m + P-1) >> LSH) - 1'b1;
                    rb1<=rb0; rb2<=rb1;
                    rv0<=(ci < ((g_m + P-1) >> LSH))
                         && ({4'b0, ci} >> GRPSH < {11'b0, gv_gdone});
                    rv1<=rv0; rv2<=rv1;
                    mwr <= dqm_w[g_dqrow + rb1];
                    ewr <= dqe_w[g_dqrow + rb1];
                    if (rv2) begin
                        dq_vin<=1'b1; dq_frac<=g_frac;
                        dq_gemvy <= gv_yout;
                        dq_mant  <= mwr;
                        dq_exp   <= ewr;
                    end
                    if (dq_vout) begin
                        for (pp=0; pp<P; pp=pp+1) begin
                            dqv = dq_out[pp*32 +: 32];
                            dword[pp*32 +: 32] = dqv;
                            if      ($signed(dqv) >  32'sd32767)  mword[pp*16 +: 16] = 16'sd32767;
                            else if ($signed(dqv) < -32'sd32768)  mword[pp*16 +: 16] = -16'sd32768;
                            else    mword[pp*16 +: 16] = dqv[15:0];
                        end
                        // feeder data-ready horizon: rows 0..dor are committed
                        if (g_dst == 3'd0)
                            qkv_wrow <= dor + 1'b1;
                        case (g_dst)
                            3'd0: qkv_bank [dor] <= dword;
                            3'd1: attn_bank[dor] <= dword;
                            3'd2: begin gl_vin<=1'b1; gl_x<=mword; end // Q4.12 sat16 -> GELU
                            3'd3: mlp_bank [dor] <= dword;
                            default: head_bank[dor] <= dword;  // 4: head logits Q6.25
                        endcase
                        // ci must re-enter g_ret as 0 (S_RES1/S_RES2 stream ci=0..ROWS-1).
                        // On silicon the 6-bit address WRAPS (the §6 board bug); keep clean.
                        if (g_dst != 3'd2 && dor==((g_m + P-1) >> LSH)-1) begin
                            ci<=0; civ<=0; st<=g_ret;
                            // LN fusion: arm the LN for the residual-state feed
                            if (g_ret==S_RES1 || g_ret==S_RES2) ln_start<=1'b1;
                        end else dor<=dor+1'b1;
                    end
                    if (gl_vout) begin                     // collect GELU -> mlpbuf (Q.22)
                        for (pp=0; pp<P; pp=pp+1) begin
                            gl_sh = $signed(gl_y[pp*16 +: 16]) <<< (LN_OUT_FRAC-GELU_FRAC);
                            gsw[pp*64 +: 64] = gl_sh;
                        end
                        mlpbuf_bank[gor] <= gsw;
                        if (gor==ROWSM-1) begin ci<=0; civ<=0; st<=g_ret; end
                        else gor<=gor+1'b1;
                    end
                end
                // ============ attention (Genesys2 port: single engine, ========
                // ============ heads processed serially 0..NHEAD-1) =============
                // head hh starts once ITS K write has committed (write hh+1);
                // the V write is covered by the read interlock (vneedA) below.
                S_AST: if (kvp_done >= {2'b0, hh} + 4'd1) begin
                    at_startA<=1'b1; at_tcount<=pos + 9'd1;
                    wi<=9'd0; wic<=9'd0; wiv<=1'b0; ldv0<=1'b0;
                    adone_s<=1'b0; st<=S_ALD;
                end
                S_ALD: begin
                    // stream q to the engine; its K stream starts the moment its
                    // q is in. V-start and ctx collection are the catchers below.
                    ldv0     <= (wi != HR[8:0]);
                    q_data_q <= qkv_r;
                    at_ldvA  <= ldv0;
                    if (wi != HR[8:0]) wi <= wi + 1'b1;
                    if (at_ldvA) begin
                        if (wic == HR-1) begin
                            kb_rstart <= 1'b1; kb_rkv <= 1'b0;   // K for head hh
                            st <= S_ACL;
                        end else wic <= wic + 1'b1;
                    end
                end
                S_ACL: if (adone_s) begin
                    cdr <= 5'd0; st <= S_CDR;
                end
                // drain the ctx catcher into ctxv_bank (the bank's ONLY writer)
                S_CDR: begin
                    ctxv_bank[hh*HR + {6'b0, cdr[2:0]}] <= ctxbufA[cdr[2:0]];
                    // Genesys2 port: was pair-indexed (2 heads/iteration via twin
                    // engines); single engine now loops one head at a time,
                    // hh 0..NHEAD-1, HR drain cycles/head instead of 2*HR/pair.
                    if (cdr == HR[4:0]-1'b1) begin
                        if (hh == NHEAD[1:0]-2'd1) begin           // -> proj GEMV
                            g_wbase<=WEIGHT_STREAM_PER_LAYER ? WB_PROJ : (blk*GW_BLK + WB_PROJ);
                            g_m<=D[BUSW-1:0]; g_k<=D[BUSW-1:0]; g_asrc<=2'd1;
                            g_asel<=blk*4 + 6'd1; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_PROJ; g_dst<=3'd1;
                            l_gbase<=blk*2 + 4'd1; l_dst<=1'b1; l_ret<=S_FCRET;  // LN2 (fed in S_RES1)
                            g_ret<=S_RES1; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                        end else begin
                            hh<=hh+2'd1; st<=S_AST;
                        end
                    end else cdr <= cdr + 5'd1;
                end
                // ---- res1: xres += attn_out (sync read, 1-cyc skew) ----
                // LN FUSION: LN2 is fed the residual sum AS IT IS WRITTEN
                // (ln_start fired at the proj G_RB exit; l_gbase/l_dst/l_ret
                // were set at the proj dispatch) -> straight to L_COLL.
                S_RES1: begin
                    cid <= ci; civ <= (ci != ROWS[BUSW-1:0]);
                    if (ci != ROWS[BUSW-1:0]) ci <= ci + 1'b1;
                    ln_vin <= civ;
                    ln_g   <= gam_r;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(attn_r[pp*32 +: 32]);
                        xres_bank[cid] <= sw;
                        ln_x <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0; orow<=0; st<=L_COLL;
                        end
                    end
                end
                // mlp_fc GEMV setup (after LN2) --------------------------------
                S_FCRET: if (dbg_stop==2'd2 && blk==4'd0) st<=S_FIN;  // DEBUG: stop after LN2
                  else begin
                    g_wbase<=WEIGHT_STREAM_PER_LAYER ? WB_FC : (blk*GW_BLK + WB_FC);
                    g_m<=D_MLP[BUSW-1:0]; g_k<=D[BUSW-1:0]; g_asrc<=2'd2;
                    g_asel<=blk*4 + 6'd2; g_frac<=7'd12; g_dqrow<=blk*DQB_P + DR_FC; g_dst<=3'd2;
                    g_ret<=S_MPSET; ci<=0; civ<=0; gor<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // mlp_proj GEMV setup (GELU already streamed inside G_RB) -------
                S_MPSET: begin
                    g_wbase<=WEIGHT_STREAM_PER_LAYER ? WB_MP : (blk*GW_BLK + WB_MP);
                    g_m<=D[BUSW-1:0]; g_k<=D_MLP[BUSW-1:0]; g_asrc<=2'd3;
                    g_asel<=blk*4 + 6'd3; g_frac<=7'd25; g_dqrow<=blk*DQB_P + DR_MP; g_dst<=3'd3;
                    // next LN (fed in S_RES2): block blk+1's LN1, or LN_f after the last block.
                    // Under streaming, L_COLL's own l_ret is redirected through S_STRW (the
                    // reload+wait state) instead of jumping straight to S_QKVRET/S_HEADSET --
                    // strw_ret carries the REAL destination for S_STRW to dispatch to once
                    // the reload completes.
                    if (blk == NLAYER-1) begin
                        l_gbase<=NLAYER*2;     l_dst<=1'b0;
                        strw_ret<=S_HEADSET; l_ret<=WEIGHT_STREAM_PER_LAYER ? S_STRW : S_HEADSET;
                    end else begin
                        l_gbase<=(blk+4'd1)*2; l_dst<=1'b0;
                        strw_ret<=S_QKVRET; l_ret<=WEIGHT_STREAM_PER_LAYER ? S_STRW : S_QKVRET;
                    end
                    g_ret<=S_RES2; ci<=0; civ<=0; gv_ldrst<=1'b1; st<=G_AQ;
                end
                // ---- res2: xres += mlp_out ; next block LN1 or final LN_f ----
                // LN FUSION: the next LN (set up at S_MPSET) is fed the residual
                // sum as it is written -> straight to L_COLL.
                S_RES2: begin
                    cid <= ci; civ <= (ci != ROWS[BUSW-1:0]);
                    if (ci != ROWS[BUSW-1:0]) ci <= ci + 1'b1;
                    ln_vin <= civ;
                    ln_g   <= gam_r;
                    if (civ) begin
                        for (pp=0; pp<P; pp=pp+1)
                            sw[pp*32 +: 32] = $signed(xres_r[pp*32 +: 32])
                                            + $signed(mlp_r[pp*32 +: 32]);
                        xres_bank[cid] <= sw;
                        ln_x <= sw;
                        if (cid==ROWS-1) begin
                            ci<=0; civ<=0;
                            if (dbg_stop==2'd3 && blk==4'd0) st<=S_FIN; // DEBUG: stop after block 0
                            else begin
                                if (blk != NLAYER-1) blk<=blk+1'b1;
                                orow<=0; st<=L_COLL;
                            end
                        end
                    end
                end
                // ---- head GEMV (act = LN_f out in lnout1) -> head_bank -------
                S_HEADSET: begin
                    // under streaming, the head's window is a fresh reload
                    // landing at on-chip address 0, same as every block --
                    // WB_HEAD only means something in the fully-resident
                    // (non-streaming) addressing scheme.
                    g_wbase<=WEIGHT_STREAM_PER_LAYER ? 20'd0 : WB_HEAD;
                    g_m<=VOCAB[BUSW-1:0]; g_k<=D[BUSW-1:0]; g_asrc<=2'd0;
                    g_asel<=4*NLAYER; g_frac<=7'd25; g_dqrow<=DR_HEAD[11:0]; g_dst<=3'd4;
                    g_ret<=S_ARGMAX; ci<=0; civ<=0; gv_ldrst<=1'b1;
                    best_val<=NEG_INF34; best_idx<=9'd0; ar<=0; arv<=0; av1<=0; amv<=0; st<=G_AQ;
                    // arm the gumbel precompute (rides the head GEMV). Greedy => skip.
                    if (smp_en) begin
                        gpre_active<=1'b1; gpre_done<=1'b0; gj<=9'd0; gj_dv<=1'b0;
                        gpre_word = {(P*32){1'b0}};   // blocking: matches the precompute block
                    end else begin
                        gpre_active<=1'b0; gpre_done<=1'b1;
                    end
                end
                // ---- per-layer weight-stream reload+wait (WEIGHT_STREAM_
                // PER_LAYER=1 only; never entered otherwise). Arms exactly
                // one weight_loader_ddr load on the first cycle (the embed
                // tables, block blk's window, or the head's -- discriminated
                // by strw_ret, which also IS the real destination), then
                // stalls until wld_ld_done before jumping there. blk is
                // ALREADY the correct next-block index by the time the
                // block-reload branch runs: S_RES2 increments it before the
                // L_COLL->S_STRW transition that reaches this state. -------
                S_STRW: begin
                    wldi_start <= 1'b0;
                    // settle phase FIRST (checked ahead of !strw_armed, since
                    // strw_armed is cleared the same cycle wld_ld_done fires --
                    // real-hardware-only finding, PORT-NOTES.md "per-layer
                    // weight streaming": weight_bank_tdp's WRITE_MODE_A/B=
                    // "no_change" true-dual-port XPM BRAM has documented
                    // same-cycle/adjacent-cycle cross-port write/read
                    // ambiguity Xilinx's own guide flags, which the fully-
                    // resident design (load once at boot, read much later)
                    // never exercised but streaming's "read immediately after
                    // every fresh reload" pattern does, every block/embed/
                    // head reload, every token. A few idle cycles between the
                    // last write commit and the first read gives the BRAM's
                    // real write pipeline time to fully settle; simulation
                    // (a behavioral memory model, not the real XPM primitive)
                    // never needed this and stayed bit-exact without it.
                    if (strw_settle != 6'd0) begin
                        if (strw_settle == STRW_SETTLE_CYCLES) begin
                            strw_settle <= 6'd0;
                            st <= strw_ret;
                        end else begin
                            strw_settle <= strw_settle + 1'b1;
                        end
                    end else if (!strw_armed) begin
                        if (strw_ret == S_EMB) begin
                            // ONE row-group at a time (tok_id's row when
                            // emb_ph==0, pos's row when emb_ph==1 -- S_EMB's own
                            // completion re-arms this same dispatch with
                            // emb_ph<=1 for the second reload, see its comment),
                            // not the whole table. (tok_id or pos)*EROWS locates
                            // that row within the DDR3-resident tok/pos table;
                            // >>EPWS converts row->word exactly like EMB_TOKW/
                            // EMB_POSW's own packing.
                            wldi_addr  <= WEIGHTS_DDR_BASE + (emb_ph
                                            ? EMB_POS_BASE + ((pos    * EROWS) >> EPWS)
                                            : EMB_TOK_BASE + ((tok_id * EROWS) >> EPWS)
                                          ) * WBYTES_STRM;
                            wldi_words <= EMB_ROWG_W*SUBW_STRM;
                        end else if (strw_ret == S_HEADSET) begin
                            wldi_addr  <= WEIGHTS_DDR_BASE + WB_HEAD*WBYTES_STRM;
                            wldi_words <= GW_HEAD*SUBW_STRM;
                        end else begin
                            wldi_addr  <= WEIGHTS_DDR_BASE + blk*GW_BLK*WBYTES_STRM;
                            wldi_words <= GW_BLK*SUBW_STRM;
                        end
                        wldi_start  <= 1'b1;
                        strw_armed  <= 1'b1;
                    end else if (wld_ld_done) begin
                        strw_armed  <= 1'b0;
                        strw_settle <= 6'd1;
                    end
                end
                // ---- P-wide argmax over the VOCAB logits (Q6.25, sync read) --
                // 3-stage pipeline (head_bank -> compare chain was the critical path):
                // A: register the row; B: P -> P/2 pairwise maxima; C: reduce + best.
                // First index wins ties (strict >), matching the scalar reference.
                // Gumbel-max sampling folds in here: each logit gets gumbel_bank's
                // precomputed noise (0 when greedy) added BEFORE the pair-max compare,
                // on the widened signed-34 path. The winner IS the sample (Gumbel-max).
                S_ARGMAX: if (gpre_done) begin
                    ard <= ar; arv <= (ar != ARROWS[$clog2(ARROWS+1)-1:0]);
                    if (ar != ARROWS[$clog2(ARROWS+1)-1:0]) ar <= ar + 1'b1;
                    hw_r <= head_r; gw_r <= gumbel_r; ad1 <= ard; av1 <= arv;  // stage A
                    if (av1) begin                                   // stage B: 4 pair maxima
                        for (pp=0; pp<4; pp=pp+1) begin
                            va = $signed(hw_r[pp*32 +: 32])
                                 + (smp_en ? $signed(gw_r[pp*32 +: 32]) : 34'sd0);
                            vb = $signed(hw_r[(pp+4)*32 +: 32])
                                 + (smp_en ? $signed(gw_r[(pp+4)*32 +: 32]) : 34'sd0);
                            ia = ad1*P + pp;  ib = ad1*P + pp + 4;
                            if (ia >= VOCAB) va = NEG_INF34;
                            if (ib >= VOCAB) vb = NEG_INF34;
                            case (pp)
                                0: begin pv0 <= (vb > va) ? vb : va; pi0 <= (vb > va) ? ib : ia; end
                                1: begin pv1 <= (vb > va) ? vb : va; pi1 <= (vb > va) ? ib : ia; end
                                2: begin pv2 <= (vb > va) ? vb : va; pi2 <= (vb > va) ? ib : ia; end
                                default: begin pv3 <= (vb > va) ? vb : va; pi3 <= (vb > va) ? ib : ia; end
                            endcase
                        end
                    end
                    amv <= av1; amd <= ad1;
                    if (amv) begin                                   // stage C: reduce + best
                        wm_val = pv0; wm_idx = pi0;
                        if (pv1 > wm_val) begin wm_val = pv1; wm_idx = pi1; end
                        if (pv2 > wm_val) begin wm_val = pv2; wm_idx = pi2; end
                        if (pv3 > wm_val) begin wm_val = pv3; wm_idx = pi3; end
                        if (wm_val > best_val) begin
                            best_val <= wm_val; best_idx <= wm_idx;
                        end
                        if (amd==ARROWS-1) begin
                            tok_out <= (wm_val > best_val) ? wm_idx : best_idx;
                            st <= S_FIN;
                        end
                    end
                end
                S_FIN: begin done<=1'b1; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
            // ---- R4e catcher: fires regardless of FSM state --------------------
            // V stream starts when the engine's probs are ready AND this head's
            // V write has committed (kvp_done >= vneed); a not-yet-ready kdone
            // latches a pending flag and fires on the write's completion.
            if (at_kdoneA) vpA <= 1'b1;
            if ((at_kdoneA || vpA) && (kvp_done >= vneedA)) begin
                kb_rstart  <= 1'b1; kb_rkv  <= 1'b1; vpA <= 1'b0;
            end
            if (at_ctxvA) ctxbufA[at_ctxidxA[2:0]] <= at_ctxdataA;
            if (at_doneA) adone_s <= 1'b1;
            // ---- KV feeder: ALL (head,K/V) writes, armed at the qkv dispatch --
            // K h0..h(NHEAD-1) ride the qkv readback (data-ready gated row by
            // row), then V h0..h(NHEAD-1) (plain ascending -- Genesys2 port).
            // Originally (KV260 / twin-engine builds) V was written in a
            // hand-shuffled order so each of the two concurrent engines' V
            // writes landed in a gap in the OTHER engine's read schedule -- a
            // latency-hiding micro-optimization with no meaning once there is
            // only one engine processing one head at a time (see the
            // single-engine collapse above, PORT-NOTES.md): plain ascending
            // order is simply correct here, not a fallback. vneedA (declared
            // above) tracks this order; S_AST's per-head gate
            // (`kvp_done >= hh+1`) needs no further conditioning since
            // K-writes are already ascending.
            case (kvf_st)
                KF_S: if (kvf_active) begin
                    kb_wstart<=1'b1; fr<=0; frv<=0; kvf_st<=KF_F;
                end
                KF_F: begin
                    if (kvf_grant && ((fr == HR[$clog2(ROWSM+1)-1:0])
                                      || (kvw_src < qkv_wrow))) begin
                        frd <= fr; frv <= (fr != HR[$clog2(ROWSM+1)-1:0]);
                        if (fr != HR[$clog2(ROWSM+1)-1:0]) fr <= fr + 1'b1;
                    end else frv <= 1'b0;
                    kb_wvalid  <= frv;
                    kb_wdata_q <= qkv_r;
                    if (frv && frd==HR-1) begin fr<=0; frv<=0; kvf_st<=KF_W; end
                end
                KF_W: if (kb_wdone) begin
                    kvp_done <= kvp_done + 4'd1;
                    if (!kvw_kv) begin                       // K phase: h0->h1->...->h(NHEAD-1)
                        if (kvw_h != NHEAD[1:0]-2'd1) begin kvw_h<=kvw_h+2'd1; kvf_st<=KF_S; end
                        else begin kvw_kv<=1'b1; kvw_h<=2'd0; kvf_st<=KF_S; end  // -> V phase, start h0
                    end else begin                           // V phase: h0->h1->...->h(NHEAD-1)
                        if (kvw_h != NHEAD[1:0]-2'd1) begin kvw_h<=kvw_h+2'd1; kvf_st<=KF_S; end
                        else begin kvf_active<=1'b0; kvf_st<=KF_IDLE; end
                    end
                end
                default: ;
            endcase
            // ---- gumbel precompute: ONE xorshift advance + ONE LUT read per cycle,
            // filling gumbel_bank during the head GEMV (armed in S_HEADSET; greedy
            // skips). Advance/place are 1 cycle apart (the LUT read is registered).
            // ADVANCE stage: while gj < VOCAB, step the xorshift (rng_state <= xs_ns);
            // the LUT address gpre_idx_w = xs_ns>>22 is read in the sync block, value
            // lands next cycle in gumbel_lut_r. gj_d/gj_dv carry the lane to place at.
            if (gpre_active) begin
                if (gj != VOCAB[VIDXW:0]) begin
                    rng_state <= xs_ns;          // persistent state advances once/logit
                    gj_d  <= gj; gj_dv <= 1'b1;
                    gj    <= gj + 9'd1;
                end else gj_dv <= 1'b0;
                // PLACE stage: gumbel_lut_r holds the noise for logit gj_d. Drop it in
                // lane gj_d%P; flush the wide word to gumbel_bank[gj_d/P] when lane==P-1
                // OR at the final logit (a partial last row).
                if (gj_dv) begin
                    gpre_word[(gj_d % P)*32 +: 32] = gumbel_lut_r;
                    if ((gj_d % P)==P-1 || gj_d==VOCAB-1) begin
                        gumbel_bank[gj_d / P] <= gpre_word;
                        gpre_word = {(P*32){1'b0}};   // zero unused lanes of the next row
                        if (gj_d==VOCAB-1) begin gpre_active<=1'b0; gpre_done<=1'b1; end
                    end
                end
            end
        end
    end

    // ---- readback (2-cyc): bank read registers above are stage 1 (address = rbr
    // while idle); stage 2 muxes the selected word + extracts the lane.
    reg [LSH-1:0] rd_lane;
    reg [P*64-1:0] rw64; reg [P*32-1:0] rw32; reg is64;
    always @* begin
        is64 = 1'b0; rw64 = {(P*64){1'b0}}; rw32 = {(P*32){1'b0}};
        case (rd_sel)
            4'd0: begin rw64 = lnout1_r; is64 = 1'b1; end   // ln1   (Q.22)
            4'd1: rw32 = qkv_r;                             // qkv   (Q.16)
            4'd2: rw32 = ctxv_r;                            // ctx   (Q6.25)
            4'd3: rw32 = attn_r;                            // attn_out (Q6.25)
            4'd4: begin rw64 = lnout2_r; is64 = 1'b1; end   // ln2   (Q.22)
            4'd5: begin rw64 = mlpbuf_r; is64 = 1'b1; end   // gelu  (Q.22)
            4'd6: rw32 = mlp_r;                             // mlp_out (Q6.25)
            4'd7: rw32 = xres_r;                            // x4 residual (Q6.25)
            default: rw32 = head_r;                         // 8: head logits (Q6.25)
        endcase
    end
    always @(posedge clk) begin
        rd_lane <= rd_addr[LSH-1:0];
        if (is64) rd_data <= $signed(rw64[rd_lane*64 +: 64]);
        else      rd_data <= {{32{rw32[rd_lane*32 + 31]}}, rw32[rd_lane*32 +: 32]};
    end
endmodule
