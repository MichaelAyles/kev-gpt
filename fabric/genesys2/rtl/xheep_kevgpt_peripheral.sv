// -----------------------------------------------------------------------------
// xheep_kevgpt_peripheral — X-HEEP ext_peripheral register-interface shell
// around the UNMODIFIED sequencer_vec core, replacing gemv_axi_seq_vec.v's
// AXI4-Lite shell for the Genesys2 (X-HEEP + cv32e40px) port.
//
// Same register map, register semantics, and IDCODE ("SQRV" 0x53515256) as
// gemv_axi_seq_vec.v (fabric/stage3/rtl/gemv_axi_seq_vec.v) -- only the outer
// bus protocol changes. X-HEEP's reg_req_t/reg_rsp_t (reg_pkg.sv) is a flat,
// single-cycle valid/ready register interface with no separate address/data
// phases (unlike AXI4-Lite's AW/W/B/AR/R channels), so one request per cycle
// carries address + write-data + write-enable together, and this peripheral
// always completes it the same cycle (reg_rsp_o.ready == reg_req_i.valid) --
// there is no internal wait-state condition here to gate on, unlike a
// multi-cycle CSR block might need.
//
// Register map (addr[7:2], the same 6-bit word index the AXI shell used):
//  0x00 CTRL (b0 go, b1 wl_rst, b2 soft_reset, b4:3 dbg_stop)
//  0x04 STATUS (b0 done, b1 busy, b2 wld_done -- see below)
//  0x08 TOK_ID   0x0C POS    0x10 W_DATA (wl_we) 0x14 RD_SEL  0x18 RD_ADDR
//  0x1C RD_DATA_LO   0x20 RD_DATA_HI   0x24 TOK_OUT   0x28 CYCLES   0x2C IDCODE
//  0x30 SEED (write-only): nonzero => load xorshift state + enable on-chip
//       Gumbel-max sampling (persists across GOs); 0/never-written => greedy
//       argmax (bit-exact) -- identical semantics to the AXI shell.
//
// 0x00-0x30 above match gemv_axi_seq_vec.v's own map exactly (KV260 parity,
// unmodified). Registers below 0x34 are Genesys2/DDR-only (WEIGHT_DDR_BACKED,
// no KV260 equivalent), added rather than reusing CTRL bits so that parity
// statement stays true:
//  0x34 WLD_ADDR  (write-only): wld_ld_ddr_addr[28:0], DDR3 byte address
//       (beat-aligned) of the weight window to stream, sampled on the SAME
//       write that would follow -- write this and WLD_WORDS before WLD_CTRL.
//  0x38 WLD_WORDS (write-only): wld_ld_words[31:0], 32-bit chunks to stream
//       (must be a multiple of 8 -- weight_loader_ddr.sv's own DATA_W/32).
//  0x3C WLD_CTRL  (write-only, b0 wld_ld_start): pulses weight_loader_ddr's
//       ld_start, same one-cycle-pulse semantics as CTRL's go/wl_rst above.
//       STATUS.b2 (wld_done) latches high when the load completes, clearing
//       on the next WLD_CTRL write (mirrors STATUS.b0's done_latched vs.
//       go_pulse clearing behaviour) or on soft_reset/rst_ni.
//
// Registers below 0x40 are also Genesys2/DDR-only, added for
// fabric/genesys2/FIXATION-WORD-CDC-INVESTIGATION.md Sec8 item 7's
// permanent hardware health monitor (ddr_health_monitor.sv, instantiated
// at the top-level wrapper, watching mig_read_mux2's and
// mig_dual_master_arbiter's owner-FIFO architectural invariants):
//  0x40 DDR_HEALTH (read-only): sticky violation bits, one per invariant
//       (b0 rdmux_owner_mismatch, b1 rdmux_push_not_ready, b2
//       rdmux_pop_when_empty, b3 dualarb_rd_mismatch, b4
//       dualarb_wr_mismatch, b5 dualarb_rd_push_not_ready, b6
//       dualarb_wr_push_not_ready, b7 dualarb_rd_pop_when_empty, b8 =
//       combined OR of all 8 -- "any violation since the last clear").
//       All zero is the expected, steady-state reading; any bit set means
//       a real owner-FIFO invariant was violated at least once since the
//       last DDR_HEALTH_CLR write (or reset) -- worth an ILA re-run (item
//       6) to see it happen live, not just that it happened.
//  0x44 DDR_ERR_COUNT (read-only): 8-bit saturating count of total
//       violation *events* (any one or more of the 8 flags true on a
//       given ui_clk cycle counts as one event) since the last clear.
//       Saturates at 0xFF rather than wrapping -- read "255" as "at least
//       255", not literally 255.
//  0x48 DDR_HEALTH_CLR (write-only, b0): a HELD LEVEL, not a one-cycle
//       pulse like every other *_CTRL register above -- write 1, wait a
//       few cycles for the clear to reach and settle in ddr_health_
//       monitor's own ui_clk domain (it's synchronized, not a same-cycle
//       clear), then write 0 to resume normal monitoring. Deliberately
//       NOT auto-clearing: a pulse can be missed crossing clock domains,
//       a held level cannot.
//  0x4C WEIGHT_STREAM_CRC (read-only): a free-running CRC32 (IEEE 802.3,
//       matches Python zlib.crc32()) over every word weight_loader_ddr's
//       REAL DMA path (weight_loader_ddr -> CDC -> mux -> arbiter -> MIG
//       -> CDC) writes into weight_bank_tdp -- Sec8 item 1's own gap,
//       never actually checked before this. Never resets on its own
//       (free-running since power-on/reset); read it at a checkpoint of
//       your choosing and compare against a host- or simulation-computed
//       expected value for the identical run.
//  0x50 WBDIAG_ADDR (write-only): weight_bank_tdp row address (raddr_a,
//       $clog2(WWORDS) bits) -- FIXATION-WORD-POSTMORTEM.md item 6. Held,
//       not a pulse: write once, then read WBDIAG_DATA0-7 as many times as
//       needed: weight_bank_tdp's port A is otherwise completely unused
//       for reading (see gemv_banked_resident_vec.sv's own comment), so
//       this is a free, always-safe tap -- correct whether the sequencer
//       is idle or actively computing, since nothing else ever reads port
//       A. To dump a specific vocab row's real resident weight content,
//       compute row = group*k_count + k for k=0..D-1, where group =
//       vocab_id/LANES and w_base(=0 for the head GEMV under
//       WEIGHT_STREAM_PER_LAYER=1) is folded in already; each row holds
//       LANES nibbles, the target vocab id's value sits at nibble
//       (vocab_id % LANES). Read promptly after the head GEMV completes,
//       before the next token's forward pass starts reloading QKV weights
//       into the same (single-buffered) resident image.
//  0x54-0x70 WBDIAG_DATA0-7 (read-only): the 256-bit row at WBDIAG_ADDR,
//       as 8 sequential 32-bit words (DATA0 = bits [31:0] ... DATA7 =
//       bits [255:224]). Valid one cycle after WBDIAG_ADDR is written --
//       always true by the time firmware's own register-bus round trip
//       reaches a subsequent read, same as every other bank readback here.
//  0x74 DBG_STOP_BLOCK (write-only, held not pulsed): which transformer
//       block (0..NLAYER-1) CTRL's own dbg_stop==2/3 halt points apply
//       to -- FIXATION-WORD-CDC-INVESTIGATION.md Sec8 item 10's own
//       follow-up ("extend dbg_stop to check layer 1"). 0 (the reset
//       value) reproduces every prior dbg_stop use exactly (block 0
//       only); write e.g. 1 before a GO with dbg_stop=2/3 to halt after
//       block 1 instead, then rd_sel/rd_addr/WBDIAG read back that
//       block's own activations/weights the same way item 8-10's own
//       block-0 diagnostics already do. dbg_stop==1 ("after embed") is
//       NOT affected by this register -- it's block-independent by
//       construction; a later block's own "x_in" is simply the
//       previous block's own x_out, already reachable via dbg_stop=3 at
//       block N-1. Added as a new register rather than extending CTRL's
//       own dbg_stop field, matching this file's own established
//       convention (see the header note on 0x00-0x30 above) of keeping
//       CTRL's bit layout identical to the KV260 AXI shell's.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module xheep_kevgpt_peripheral #(
    // P=16 (this instance's old, never-actually-verified default) produces
    // X-state tok_out at the full-sequencer level -- found while gating
    // Phase 5's real weight image (fabric.stage3.run_vec_kv --p 16 --lanes 128,
    // VEC_KV_VERDICT match=False, hw gen=[None]*6). Every previously-green
    // run_vec_kv result in this project (including the Option A and KV260
    // regression checks in PORT-NOTES.md) used P=8; P=16 is not exercised by
    // any gate in the repo (run_banked.sv sweeps `lanes`, not `P`) and is
    // apparently untested/broken at the sequencer_vec level, not just unlucky
    // parameterization. Not investigated further -- out of scope for this
    // phase, flagged rather than silently worked around. Defaulting to the
    // one P value this project has ever gated green.
    parameter integer P      = 8,
    // LANES=128 (inherited unmodified from the KV260 config) drove the
    // systolic GEMV/attention datapath to 252,194 LUTs (123.75% of the
    // xc7k325t's 203,800) and 804/840 DSP48E1 (95.7%) at synth alone --
    // found via a synth-only Vivado checkpoint, doesn't fit this part at
    // all. LANES=64 (gated bit-exact against Option A, avg_cyc 2498->4038
    // per pass -- a real, accepted throughput cost) is the deployed value.
    parameter integer LANES  = 64,
    // Defaults are sizing.py's Option A shape (fabric/genesys2/PORT-NOTES.md's
    // "sizing decision"): d=128, n_layer=2, n_head=2, ctx=128, vocab=57 (this
    // port's trained Kevin-speak char vocab). Was silently left at
    // sequencer_vec's own KV260-deployed defaults (D=256/NHEAD=4/VOCAB=193)
    // here -- this instance never actually overrode them even though NLAYER/
    // WWORDS/TMAX below look like they were port-tuned; caught when wiring up
    // Phase 5's real weight image. HEAD_DIM stays fixed at sequencer_vec's own
    // default (64) per that same sizing decision -- only D/NHEAD/VOCAB vary.
    parameter integer D      = 128,
    parameter integer D3     = 384,
    parameter integer D_MLP  = 512,
    parameter integer NHEAD  = 2,
    parameter integer VOCAB  = 57,
    parameter integer NLAYER = 2,
    // sequencer_vec's own default (262144 wide words = 16MB) is a generic
    // upper bound, never tightened per-model -- Option A's real footprint is
    // ~4680 wide words (74,880 32-bit W_DATA writes / SUBW=16), found to be
    // eating 256/445 RAMB36 (57.5%) at the untightened default during a
    // synth-only Vivado checkpoint. 8192 (power-of-2 headroom over the real
    // need) is the deployed value here.
    // WWORDS counts LANES*4-bit wide words, so halving LANES (above) halves
    // the bytes/word too -- doubled from 8192 to keep the same real byte
    // capacity margin over Option A's actual need.
    parameter integer WWORDS = 16384,
    parameter integer TMAX   = 128,
    // Genesys2 has no URAM at all -- see weight_bank_tdp.sv/kv_bank.sv.
    parameter               MEM_PRIMITIVE = "block",
    // DDR3-backed KV cache (fabric/genesys2/PORT-NOTES.md "Phase 2
    // architecture") -- 0 (default) keeps this peripheral's existing,
    // fully-resident build untouched. 1 threads kv_wr_*/kv_rd_* straight
    // through to sequencer_vec's own KV_DDR_BACKED=1 path; see that
    // parameter's comment in sequencer_vec.sv for the full picture.
    parameter               KV_DDR_BACKED = 0,
    // KV cache's own DDR3 region base -- see sequencer_vec.sv's KV_DDR_BASE
    // parameter comment (real-hardware bug: this MUST be kept non-
    // overlapping with WEIGHTS_DDR_BASE below whenever KV_DDR_BACKED=1 and
    // WEIGHT_STREAM_PER_LAYER=1 are both set).
    parameter integer       KV_DDR_BASE = 0,
    // DDR3-backed weight-window loader (PORT-NOTES.md "weight_loader_ddr
    // wired to top level") -- 0 (default) leaves this peripheral's existing
    // build untouched (WLD_* registers still exist but drive an idle
    // sequencer_vec.wld_* port set). 1 threads them through to sequencer_vec's
    // own WEIGHT_DDR_BACKED=1 path; see that parameter's comment.
    parameter               WEIGHT_DDR_BACKED = 0,
    // Per-layer DDR3 weight streaming (PORT-NOTES.md "per-layer weight
    // streaming") -- 0 (default) leaves this peripheral's existing build
    // untouched. 1 threads straight through to sequencer_vec's own
    // WEIGHT_STREAM_PER_LAYER=1 path (requires WEIGHT_DDR_BACKED=1 too);
    // see that parameter's comment in sequencer_vec.sv for the full
    // picture. When 1, WWORDS above should be sized to this checkpoint's
    // own max(GW_BLK, GW_HEAD, GW_EMB), not the whole NLAYER-scaled image
    // -- NLAYER-independent, the actual point of this mode.
    parameter               WEIGHT_STREAM_PER_LAYER = 0,
    parameter integer       WEIGHTS_DDR_BASE = 0,
    parameter type reg_req_t = reg_pkg::reg_req_t,
    parameter type reg_rsp_t = reg_pkg::reg_rsp_t
) (
    input  logic     clk_i,
    input  logic     rst_ni,
    input  reg_req_t reg_req_i,
    output reg_rsp_t reg_rsp_o,

    // ---- DDR-backed KV cache DMA ports (KV_DDR_BACKED=1 only; idle-tied by
    // sequencer_vec itself when 0, see that module) -- wire straight to a
    // fabric/genesys2/rtl/kevgpt_ddr_bundle.sv instance's kv_wr_*/kv_rd_*
    // ports at the top level. ---------------------------------------------
    output logic                 kv_wr_pkt_valid,
    input  logic                 kv_wr_pkt_ready,
    output logic [28:0]          kv_wr_pkt_addr,
    output logic [255:0]         kv_wr_pkt_data,
    output logic [31:0]          kv_wr_pkt_mask,
    input  logic                 kv_wr_ack_valid,
    output logic                 kv_wr_ack_ready,
    output logic                 kv_rd_req_valid,
    input  logic                 kv_rd_req_ready,
    output logic [28:0]          kv_rd_req_addr,
    input  logic                 kv_rd_ret_valid,
    output logic                 kv_rd_ret_ready,
    input  logic [255:0]         kv_rd_ret_data,

    // ---- DDR-backed weight-window loader DMA ports (WEIGHT_DDR_BACKED=1
    // only; idle-tied by sequencer_vec itself when 0) -- wire straight to a
    // kevgpt_ddr_bundle.sv instance's wl_rd_req_*/wl_rd_ret_* ports at the
    // top level (its CDC-facing side, once that port grows real CDC -- see
    // kevgpt_ddr_bundle.sv's own header note). ------------------------------
    output logic                 wl_rd_req_valid,
    input  logic                 wl_rd_req_ready,
    output logic [28:0]          wl_rd_req_addr,
    input  logic                 wl_rd_ret_valid,
    output logic                 wl_rd_ret_ready,
    input  logic [255:0]         wl_rd_ret_data,

    // ---- DDR health monitor (Sec8 item 7) -- ddr_health_monitor.sv at the
    // top level already does all the sticky-latch/CDC/counter work; this
    // peripheral just exposes its gen_clk-domain outputs as registers and
    // passes the clear level through. ----------------------------------------
    input  logic [8:0]           health_sticky_i,
    input  logic [7:0]           health_count_i,
    output logic                 health_clear_o
);
    wire clk = clk_i;
    wire [31:0] weight_stream_crc;  // Sec8 item 1 -- see u_seq's own port comment
    // FIXATION-WORD-POSTMORTEM.md item 6 -- see u_seq's own wbdiag_addr/
    // wbdiag_pair port comment (gemv_banked_resident_vec.sv has the full
    // rationale: this is weight_bank_tdp's otherwise-completely-unused port
    // A, and why it carries BOTH DP=1 column-parity halves). The row's own
    // LSB (wbdiag_addr_r[0], below) selects which half WBDIAG_DATA0-7
    // expose -- an earlier version of this tap exposed only the even half
    // unconditionally, silently returning the wrong row's data for every
    // odd address; caught by item 6's own simulation gate before real
    // hardware, see FIXATION-WORD-POSTMORTEM.md item 6's verification note.
    // WBDIAG_DATA0-7 hardcode 8x 32-bit slices (256 bits, one half) --
    // correct for the deployed LANES=64 (LANES*4=256), NOT generically
    // parameterized; the check below flags it loudly if that ever changes.
    wire [LANES*8-1:0] wbdiag_pair;
    wire [LANES*4-1:0] wbdiag_data = wbdiag_addr_r[0] ? wbdiag_pair[LANES*8-1:LANES*4]
                                                        : wbdiag_pair[LANES*4-1:0];
`ifndef SYNTHESIS
    initial if (LANES*4 != 256)
        $display("xheep_kevgpt_peripheral: WARNING LANES=%0d makes wbdiag_data %0d bits, but WBDIAG_DATA0-7 only expose the low 256 -- add more WBDIAG_DATA registers before trusting this diagnostic", LANES, LANES*4);
`endif
    wire [5:0] windex = reg_req_i.addr[7:2];
    // tok_id/core_tok_out width: was hardcoded [8:0] (max 511), fine for
    // every char-level VOCAB (<=193) this project ever deployed but a real
    // bug at word-level VOCAB=1900 (PORT-NOTES.md "word-level vocabulary")
    // -- any prompt token with id>511 silently truncated on write, and any
    // generated id>511 truncated on readback. Matches sequencer_vec.sv's
    // own VIDXW=$clog2(VOCAB) fix.
    localparam integer VIDXW = $clog2(VOCAB);

    reg          go_pulse, wl_rst, soft_reset, wl_we;
    reg [1:0]    dbg_stop;
    reg [3:0]    dbg_stop_block;
    reg [31:0]   wl_data;
    reg [VIDXW-1:0] tok_id;
    reg [8:0]    pos;
    reg [3:0]    rd_sel;
    reg [10:0]   rd_addr;
    reg [31:0]   seed;
    reg          seed_we;
    reg          wld_ld_start_r;
    reg [28:0]   wld_ld_ddr_addr_r;
    reg [31:0]   wld_ld_words_r;
    reg [$clog2(WWORDS)-1:0] wbdiag_addr_r;

    // ---- write side: one register-file update per accepted write request ----
    // (matches the AXI shell's pulse semantics for go/wl_we/seed_we -- each is
    // high for exactly the one cycle after the write request lands, which is
    // what sequencer_vec's start-pulse inputs expect).
    always @(posedge clk) begin
        if (!rst_ni) begin
            go_pulse<=0; wl_rst<=0; soft_reset<=0; wl_we<=0; wl_data<=0;
            tok_id<=0; pos<=0; rd_sel<=0; rd_addr<=0; dbg_stop<=0; dbg_stop_block<=0;
            seed<=0; seed_we<=0;
            wld_ld_start_r<=0; wld_ld_ddr_addr_r<=0; wld_ld_words_r<=0;
            wbdiag_addr_r<=0;
        end else if (reg_req_i.valid && reg_req_i.write) begin
            go_pulse<=0; wl_rst<=0; wl_we<=0; seed_we<=0; wld_ld_start_r<=0;   // 1-cycle pulses default low
            case (windex)
                6'h0: begin go_pulse<=reg_req_i.wdata[0]; wl_rst<=reg_req_i.wdata[1];
                            soft_reset<=reg_req_i.wdata[2]; dbg_stop<=reg_req_i.wdata[4:3]; end
                6'h2: tok_id  <= reg_req_i.wdata[VIDXW-1:0];
                6'h3: pos     <= reg_req_i.wdata[8:0];
                6'h4: begin wl_we<=1; wl_data<=reg_req_i.wdata; end
                6'h5: rd_sel  <= reg_req_i.wdata[3:0];
                6'h6: rd_addr <= reg_req_i.wdata[10:0];
                6'hC: begin seed <= reg_req_i.wdata; seed_we <= 1'b1; end   // 0x30 SEED
                6'hD: wld_ld_ddr_addr_r <= reg_req_i.wdata[28:0];          // 0x34 WLD_ADDR
                6'hE: wld_ld_words_r    <= reg_req_i.wdata;                // 0x38 WLD_WORDS
                6'hF: wld_ld_start_r    <= reg_req_i.wdata[0];             // 0x3C WLD_CTRL
                6'h14: wbdiag_addr_r    <= reg_req_i.wdata[$clog2(WWORDS)-1:0]; // 0x50 WBDIAG_ADDR
                6'h1D: dbg_stop_block   <= reg_req_i.wdata[3:0];           // 0x74 DBG_STOP_BLOCK
                default: ;
            endcase
        end else begin
            go_pulse<=0; wl_rst<=0; wl_we<=0; seed_we<=0; wld_ld_start_r<=0;
        end
    end

    // wld_done latches on weight_loader_ddr's ld_done pulse, clears on the
    // NEXT WLD_CTRL write (mirrors done_latched vs. go_pulse below) or reset.
    reg wld_done_latched;
    always @(posedge clk) begin
        if (core_rst) wld_done_latched <= 1'b0;
        else if (wld_ld_start_r) wld_done_latched <= 1'b0;
        else if (core_wld_done) wld_done_latched <= 1'b1;
    end

    // 0x48 DDR_HEALTH_CLR: a HELD LEVEL (see this module's own header
    // comment for why), NOT one of the one-cycle pulses in the shared
    // write-side always block above -- kept in its own always block so it
    // is never swept up in that block's "default low every other cycle"
    // pulse semantics.
    reg health_clear_r;
    always @(posedge clk) begin
        if (!rst_ni) health_clear_r <= 1'b0;
        else if (reg_req_i.valid && reg_req_i.write && windex == 6'h12)
            health_clear_r <= reg_req_i.wdata[0];
    end
    assign health_clear_o = health_clear_r;

    // ---- read side: combinational mux, same word map as the AXI shell ----
    reg [31:0] rdata_mux;
    always @(*) begin
        case (windex)
            6'h1: rdata_mux = {29'b0, wld_done_latched, core_busy, done_latched};
            6'h7: rdata_mux = core_rd_data[31:0];
            6'h8: rdata_mux = core_rd_data[63:32];
            6'h9: rdata_mux = {{(32-VIDXW){1'b0}}, core_tok_out};
            6'hA: rdata_mux = cycles_latched;
            6'hB: rdata_mux = 32'h5351_5256;              // "SQRV"
            6'h10: rdata_mux = {23'b0, health_sticky_i};  // 0x40 DDR_HEALTH
            6'h11: rdata_mux = {24'b0, health_count_i};   // 0x44 DDR_ERR_COUNT
            6'h13: rdata_mux = weight_stream_crc;         // 0x4C WEIGHT_STREAM_CRC
            6'h15: rdata_mux = wbdiag_data[31:0];          // 0x54 WBDIAG_DATA0
            6'h16: rdata_mux = wbdiag_data[63:32];         // 0x58 WBDIAG_DATA1
            6'h17: rdata_mux = wbdiag_data[95:64];         // 0x5C WBDIAG_DATA2
            6'h18: rdata_mux = wbdiag_data[127:96];        // 0x60 WBDIAG_DATA3
            6'h19: rdata_mux = wbdiag_data[159:128];       // 0x64 WBDIAG_DATA4
            6'h1A: rdata_mux = wbdiag_data[191:160];       // 0x68 WBDIAG_DATA5
            6'h1B: rdata_mux = wbdiag_data[223:192];       // 0x6C WBDIAG_DATA6
            6'h1C: rdata_mux = wbdiag_data[255:224];       // 0x70 WBDIAG_DATA7
            default: rdata_mux = 32'b0;
        endcase
    end

    // X-HEEP's register_interface is one-cycle decoupled; `ready` means the
    // transaction (read or write) completes this cycle -- no wait states.
    assign reg_rsp_o.ready = reg_req_i.valid;
    assign reg_rsp_o.error = 1'b0;
    assign reg_rsp_o.rdata = rdata_mux;

    // ---- the unmodified sequencer core + its run/cycle-count bookkeeping,
    //      identical to gemv_axi_seq_vec.v ----
    wire               core_done_w;
    wire               core_wld_done;
    wire [VIDXW-1:0]   core_tok_out;
    wire signed [63:0] core_rd_data;
    reg                core_busy, done_latched;
    reg  [31:0]        cycles_run, cycles_latched;
    wire               core_rst = ~rst_ni | soft_reset;

    always @(posedge clk) begin
        if (core_rst) begin core_busy<=0; cycles_run<=0; cycles_latched<=0; done_latched<=0; end
        else if (go_pulse) begin core_busy<=1; cycles_run<=0; done_latched<=0; end
        else if (core_busy) begin
            cycles_run<=cycles_run+1'b1;
            if (core_done_w) begin core_busy<=0; cycles_latched<=cycles_run; done_latched<=1; end
        end
    end

    sequencer_vec #(.P(P), .LANES(LANES), .D(D), .D3(D3), .D_MLP(D_MLP), .NHEAD(NHEAD),
                     .VOCAB(VOCAB), .NLAYER(NLAYER), .WWORDS(WWORDS), .TMAX(TMAX),
                     .MEM_PRIMITIVE(MEM_PRIMITIVE), .KV_DDR_BACKED(KV_DDR_BACKED),
                     .KV_DDR_BASE(KV_DDR_BASE),
                     .WEIGHT_DDR_BACKED(WEIGHT_DDR_BACKED),
                     .WEIGHT_STREAM_PER_LAYER(WEIGHT_STREAM_PER_LAYER),
                     .WEIGHTS_DDR_BASE(WEIGHTS_DDR_BASE)) u_seq (
        .clk(clk), .rst(core_rst), .go(go_pulse),
        .tok_id(tok_id), .pos(pos), .done(core_done_w), .tok_out(core_tok_out),
        .rd_sel(rd_sel), .rd_addr(rd_addr), .rd_data(core_rd_data),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data), .dbg_stop(dbg_stop),
        .dbg_stop_block(dbg_stop_block),
        .seed(seed), .seed_we(seed_we),
        .kv_wr_pkt_valid(kv_wr_pkt_valid), .kv_wr_pkt_ready(kv_wr_pkt_ready),
        .kv_wr_pkt_addr(kv_wr_pkt_addr), .kv_wr_pkt_data(kv_wr_pkt_data), .kv_wr_pkt_mask(kv_wr_pkt_mask),
        .kv_wr_ack_valid(kv_wr_ack_valid), .kv_wr_ack_ready(kv_wr_ack_ready),
        .kv_rd_req_valid(kv_rd_req_valid), .kv_rd_req_ready(kv_rd_req_ready), .kv_rd_req_addr(kv_rd_req_addr),
        .kv_rd_ret_valid(kv_rd_ret_valid), .kv_rd_ret_ready(kv_rd_ret_ready), .kv_rd_ret_data(kv_rd_ret_data),
        .wld_ld_start(wld_ld_start_r), .wld_ld_ddr_addr(wld_ld_ddr_addr_r),
        .wld_ld_words(wld_ld_words_r), .wld_ld_done(core_wld_done),
        .wl_rd_req_valid(wl_rd_req_valid), .wl_rd_req_ready(wl_rd_req_ready), .wl_rd_req_addr(wl_rd_req_addr),
        .wl_rd_ret_valid(wl_rd_ret_valid), .wl_rd_ret_ready(wl_rd_ret_ready), .wl_rd_ret_data(wl_rd_ret_data),
        .weight_stream_crc(weight_stream_crc),
        .wbdiag_addr(wbdiag_addr_r), .wbdiag_pair(wbdiag_pair)
    );
endmodule
