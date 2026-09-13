// -----------------------------------------------------------------------------
// kv_bank_ddr — the DDR3-backed KV cache (PORT-NOTES.md "Phase 2
// architecture"). Same quantise-at-write / dequantise-at-read contract as
// fabric/stage3/rtl/kv_bank.sv (doc 7 R1's K8/V8 asymmetric-per-(head,pos)
// scheme) — the write side's collect/scale/quantise pipeline (W_IDLE..W_QNT)
// and the read side's dequant math are both byte-for-byte copies of
// kv_bank.sv's own logic, unchanged, so this module is guaranteed to produce
// IDENTICAL results to the already-verified reference for identical inputs
// (this is what makes a bit-exact gate against kv_bank.sv possible, instead
// of re-deriving and re-trusting the divide-free magic-multiply scale math,
// or the dequant reconstruction, a second time).
//
// WRITE: instead of a same-cycle on-chip TDP BRAM write (kv_bank.sv's
// W_CWR), the write FSM issues a sequence of DMA write packets on a
// mig_write_engine-shaped pkt_*/ack_* port (ROW_BEATS beats: CODE_BEATS beats
// of quantised codes + 1 beat of {scale,lo} header).
//
// READ: instead of a 1-cycle-latency on-chip TDP BRAM read stream
// (kv_bank.sv's R_RUN), the read FSM walks positions 0..tcount-1 of a
// (layer,kv,head) selector one at a time (matching kv_bank.sv's own
// contract: no direct "read position p" input, a sequential scan from 0),
// and for each position issues ROW_BEATS sequential DMA read requests on a
// mig_read_engine-shaped req_*/ret_* port, then dequantises the captured
// beats the same way kv_bank.sv's deq_word combinational block does before
// pulsing rd_valid/rd_data (and rd_done on the last position). Requests are
// issued strictly one-at-a-time (never more than one outstanding) rather
// than pipelined ahead -- simpler and still correctness-first; Phase 0
// already established DDR bandwidth/latency headroom is ample at the 50
// tok/s target, so there is no throughput pressure to pipeline this yet.
//
// Both sides address a row ((layer,kv,head,pos)'s ROW_BEATS-beat record) as
// a flat byte offset into a KV_DDR_BASE-relative region of DDR3 —
// kv_bank.sv's existing row-index arithmetic (((layer*2+kv)*NHEAD+head)
// *TMAX+pos) is reused verbatim, just multiplied by ROW_BEATS*BEAT_BYTES
// instead of used as a BRAM row address directly.
//
// Address convention matches this codebase's existing DDR-facing modules
// (cpu_ddr_bridge.sv's header, v22_streamer_acc's ddr_beat_addr()): a BYTE
// address, beat-aligned (low $clog2(DATA_W/8) bits always 0), never shifted.
// wr_pkt_mask polarity matches Xilinx UG586's MIG7 native-UI convention:
// active-low (mask bit = 1 means that byte is NOT written).
//
// No rd2_*/second read port: kv_bank.sv's rd2_* (the "URAM/BRAM second
// port," doc-7 R4e) exists for a twin-engine attention scheduler this port's
// single-engine sequencer_vec.sv redesign no longer uses (rd2_start is tied
// 1'b0 permanently there) -- matching actual usage rather than kv_bank.sv's
// full historical interface.
//
// KBITS is assumed a multiple of 8 (byte-aligned codes) for the beat/mask
// packing below to be well-formed -- true for every configuration this
// project has ever used (KBITS=8, "K8/V8" is a pinned contract per
// kv_bank.sv's own header), asserted in sim, not re-derived generally.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_bank_ddr #(
    parameter integer P          = 8,
    parameter integer HEAD_DIM   = 64,
    parameter integer NHEAD      = 4,
    parameter integer NLAYER     = 4,
    parameter integer TMAX       = 256,
    parameter integer KBITS      = 8,
    parameter integer INV_SH     = 24,
    parameter integer ADDR_W     = 29,
    parameter integer DATA_W     = 256,
    // Byte address, beat-aligned (low $clog2(DATA_W/8) bits must be 0) --
    // base of this instance's KV region in the shared DDR3 address space.
    parameter integer KV_DDR_BASE = 0
) (
    input  wire        clk,
    input  wire        rst,

    // ---- quantise-write port: SAME external contract as kv_bank.sv's wq_* --
    input  wire        wq_start,            // pulse; selectors sampled here
    input  wire [3:0]  wq_layer,
    input  wire        wq_kv,               // 0 = K, 1 = V
    input  wire [1:0]  wq_head,
    input  wire [8:0]  wq_pos,
    input  wire        wq_valid,            // HR beats of P Q.16 lanes follow
    input  wire [P*32-1:0] wq_data,
    output reg          wq_done,            // pulses when all ROW_BEATS acked

    // ---- DMA write-request port: wires straight to a mig_write_engine's
    // pkt_*/ack_* ports (fabric/genesys2/tb/mig_behav_model.sv stands in for
    // the real MIG app port in simulation) ------------------------------------
    output wire                    wr_pkt_valid,
    input  wire                    wr_pkt_ready,
    output wire [ADDR_W-1:0]       wr_pkt_addr,
    output wire [DATA_W-1:0]       wr_pkt_data,
    output wire [DATA_W/8-1:0]     wr_pkt_mask,
    input  wire                    wr_ack_valid,
    output wire                    wr_ack_ready,

    // ---- dequant read-stream port: SAME external contract as kv_bank.sv's
    // rd_* (stream A only -- see header re: no rd2_*) ------------------------
    input  wire        rd_start,            // pulse; selectors sampled here
    input  wire [3:0]  rd_layer,
    input  wire        rd_kv,
    input  wire [1:0]  rd_head,
    input  wire [8:0]  rd_tcount,           // positions to stream (1..TMAX)
    output reg         rd_valid,            // a dequantised wide row is on rd_data
    output reg  [HEAD_DIM*32-1:0] rd_data,  // ONE POSITION's head row, dequantised
    output reg         rd_done,             // pulses after the last position

    // ---- DMA read-request port: wires straight to a mig_read_engine's
    // req_*/ret_* ports --------------------------------------------------------
    output wire                    rd_req_valid,
    input  wire                    rd_req_ready,
    output wire [ADDR_W-1:0]       rd_req_addr,
    input  wire                    rd_ret_valid,
    output wire                    rd_ret_ready,
    input  wire [DATA_W-1:0]       rd_ret_data
);
    localparam integer HR    = HEAD_DIM / P;
    localparam integer QMAX  = (1 << KBITS) - 1;
    localparam integer NHSEL = NLAYER * 2 * NHEAD;          // (layer,kv,head) combos
    localparam integer HROWS = NHSEL * TMAX;                // one row per (sel, pos)
    localparam integer PBW   = $clog2(HROWS);
    localparam integer DIVW  = INV_SH + 2;                  // divider width (26)

    // ---- DDR beat/row packing --------------------------------------------------
    localparam integer BEAT_BYTES  = DATA_W / 8;
    localparam integer CODE_BITS   = HEAD_DIM * KBITS;
    localparam integer CODE_BEATS  = (CODE_BITS + DATA_W - 1) / DATA_W;
    localparam integer CODE_BYTES  = CODE_BITS / 8;
    localparam integer LAST_VALID_BYTES = CODE_BYTES - (CODE_BEATS - 1) * BEAT_BYTES;
    localparam integer HDR_BYTES   = 6;                     // {scale16, lo32}
    localparam integer ROW_BEATS   = CODE_BEATS + 1;         // codes + 1 hdr beat
    localparam integer ROW_BYTES   = ROW_BEATS * BEAT_BYTES;
    localparam integer BI_W        = $clog2(ROW_BEATS + 1);

    localparam [BEAT_BYTES-1:0] FULL_MASK = {BEAT_BYTES{1'b0}};             // write every byte
    localparam [BEAT_BYTES-1:0] LAST_CODE_MASK =
        {{(BEAT_BYTES-LAST_VALID_BYTES){1'b1}}, {LAST_VALID_BYTES{1'b0}}};
    localparam [BEAT_BYTES-1:0] HDR_MASK =
        {{(BEAT_BYTES-HDR_BYTES){1'b1}}, {HDR_BYTES{1'b0}}};

    // ---- write FSM ---------------------------------------------------------------
    localparam [3:0] W_IDLE=4'd0, W_COLL=4'd1, W_SCALE=4'd2, W_INVL=4'd3,
                     W_INVR=4'd4, W_SCALE2=4'd5, W_QNT=4'd6, W_DMA=4'd7,
                     W_DACK=4'd8, W_MMD=4'd9;
    reg [3:0] wst;

    // ---- write-side state (verbatim from kv_bank.sv) ---------------------------
    reg [3:0]  w_layer; reg w_kv; reg [1:0] w_head; reg [8:0] w_pos;
    reg [PBW-1:0] w_pbase;
    reg [HEAD_DIM*32-1:0] vecbuf;            // plain reg: the head vector, collected
    reg signed [31:0] minv, maxv;
    reg [$clog2(HR+1)-1:0] w_vi;             // collect beat counter
    reg [$clog2(HR+1)-1:0] w_qi;             // quantise row counter
    reg signed [31:0] w_lo;
    reg [15:0] w_scale;
    reg [INV_SH:0] w_inv;                    // inv <= 2^INV_SH

    // doc-7 R4a divide-free scale/inv (verbatim from kv_bank.sv -- see that
    // file's header for the derivation/proof note; not re-derived here).
    localparam integer INVD = 16512;
    (* rom_style = "block" *) reg [INV_SH:0] inv_lut_lo [0:4095];
    (* rom_style = "block" *) reg [12:0]     inv_lut_hi [0:INVD-4097];
    initial begin
        $readmemh("inv_lut_lo.mem", inv_lut_lo);
        $readmemh("inv_lut_hi.mem", inv_lut_hi);
    end
    reg [22:0] w_span;
    reg [53:0] w_magic;
    reg [INV_SH:0] inv_lo_rd;
    reg [12:0]     inv_hi_rd;
    reg            inv_hi_sel;
    wire [14:0] sc_now = (w_magic[53:39] == 0) ? 15'd1 : w_magic[53:39];
    wire [14:0] lut_sc = (wst == W_INVL) ? sc_now : w_scale[14:0];
    always @(posedge clk) begin
        inv_lo_rd  <= inv_lut_lo[lut_sc[11:0]];
        inv_hi_rd  <= inv_lut_hi[lut_sc - 15'd4096];
        inv_hi_sel <= (lut_sc >= 15'd4096);
    end
    wire [INV_SH:0] inv_rd = inv_hi_sel ? {12'b0, inv_hi_rd} : inv_lo_rd;

    // min/max collection pipeline (verbatim from kv_bank.sv)
    integer mp;
    reg signed [31:0] lane_v, quad_min [0:1], quad_max [0:1];
    always @* begin
        quad_min[0] = 32'sh7FFFFFFF; quad_max[0] = 32'sh80000000;
        quad_min[1] = 32'sh7FFFFFFF; quad_max[1] = 32'sh80000000;
        for (mp = 0; mp < P; mp = mp + 1) begin
            lane_v = $signed(wq_data[mp*32 +: 32]);
            if (lane_v < quad_min[mp/(P/2)]) quad_min[mp/(P/2)] = lane_v;
            if (lane_v > quad_max[mp/(P/2)]) quad_max[mp/(P/2)] = lane_v;
        end
    end
    reg signed [31:0] qm_min0, qm_min1, qm_max0, qm_max1;
    reg               qm_v;
    wire signed [31:0] qmn = (qm_min0 < qm_min1) ? qm_min0 : qm_min1;
    wire signed [31:0] qmx = (qm_max0 > qm_max1) ? qm_max0 : qm_max1;

    // quantise pipeline (verbatim from kv_bank.sv); wstage padded up to a
    // whole number of DATA_W beats (CODE_BEATS*DATA_W bits) instead of
    // exactly HEAD_DIM*KBITS, so the DMA beat part-select below is always
    // in-range -- the pad bits (above CODE_BITS) are masked out on write by
    // LAST_CODE_MASK and never reach DDR3.
    reg [P*32-1:0] qd_r;
    reg [$clog2(HR+1)-1:0] w_qi_d;
    reg            qd_v;
    reg [P*32-1:0] qd_now;
    reg [$clog2(HR+1)-1:0] q_sel;
    reg signed [32:0] q_diff;
    integer qp2;
    always @* begin
        q_sel  = (w_qi >= HR) ? {($clog2(HR+1)){1'b0}} : w_qi;
        qd_now = {(P*32){1'b0}};
        for (qp2 = 0; qp2 < P; qp2 = qp2 + 1) begin
            q_diff = $signed(vecbuf[(q_sel*P + qp2)*32 +: 32]) - w_lo;
            qd_now[qp2*32 +: 32] = q_diff[31:0];
        end
    end

    reg [P*KBITS-1:0] q_codes;
    reg [31:0]  q_u;
    reg [31+INV_SH+1:0] q_prod;
    reg [31:0]  q_code;
    integer qp;
    always @* begin
        q_codes = {(P*KBITS){1'b0}};
        for (qp = 0; qp < P; qp = qp + 1) begin
            q_u    = qd_r[qp*32 +: 32];
            q_prod = q_u * w_inv + (1 << (INV_SH-1));
            q_code = q_prod >> INV_SH;
            if (q_code > QMAX) q_code = QMAX;
            q_codes[qp*KBITS +: KBITS] = q_code[KBITS-1:0];
        end
    end

    reg [CODE_BEATS*DATA_W-1:0] wstage;      // staged code beats (padded)
    reg [BI_W-1:0] dma_bi;                   // beat index within this row (0..ROW_BEATS-1)

    wire [BI_W-1:0] hdr_bi     = CODE_BEATS[BI_W-1:0];
    wire            is_hdr_beat  = (dma_bi == hdr_bi);
    wire            is_last_code = (dma_bi == (CODE_BEATS-1));

    wire [ADDR_W-1:0] row_base = KV_DDR_BASE[ADDR_W-1:0]
                                  + ({{(ADDR_W-PBW){1'b0}}, w_pbase} * ROW_BYTES);

    assign wr_pkt_valid = (wst == W_DMA);
    assign wr_pkt_addr  = row_base + ({{(ADDR_W-BI_W){1'b0}}, dma_bi} * BEAT_BYTES);
    assign wr_pkt_data  = is_hdr_beat ? {{(DATA_W-48){1'b0}}, w_scale, w_lo}
                                       : wstage[dma_bi*DATA_W +: DATA_W];
    assign wr_pkt_mask  = is_hdr_beat ? HDR_MASK
                         : is_last_code ? LAST_CODE_MASK
                                        : FULL_MASK;
    assign wr_ack_ready = (wst == W_DACK);

    always @(posedge clk) begin
        wq_done <= 1'b0;
        if (rst) begin
            wst <= W_IDLE; qd_v <= 1'b0;
        end else begin
            case (wst)
                W_IDLE: if (wq_start) begin
                    w_layer <= wq_layer; w_kv <= wq_kv; w_head <= wq_head; w_pos <= wq_pos;
                    w_pbase <= ((wq_layer*2 + {3'b0,wq_kv})*NHEAD + {2'b0,wq_head})*TMAX
                                + {3'b0,wq_pos};
                    minv <= 32'sh7FFFFFFF; maxv <= 32'sh80000000;
                    qm_v <= 1'b0;
                    w_vi <= 0; wst <= W_COLL;
                    wstage <= {(CODE_BEATS*DATA_W){1'b0}};
                end
                W_COLL: begin
                    if (qm_v) begin
                        if (qmn < minv) minv <= qmn;
                        if (qmx > maxv) maxv <= qmx;
                    end
                    qm_v <= wq_valid;
                    if (wq_valid) begin
                        vecbuf[w_vi*P*32 +: P*32] <= wq_data;
                        qm_min0 <= quad_min[0]; qm_min1 <= quad_min[1];
                        qm_max0 <= quad_max[0]; qm_max1 <= quad_max[1];
                        if (w_vi == HR-1) begin
                            wst <= W_MMD;
                        end else w_vi <= w_vi + 1'b1;
                    end
                end
                W_MMD: begin
                    if (qmn < minv) minv <= qmn;
                    if (qmx > maxv) maxv <= qmx;
                    qm_v <= 1'b0;
                    wst <= W_SCALE;
                end
                W_SCALE: begin
                    w_lo   <= minv;
                    w_span <= (maxv - minv + (QMAX >> 1));
                    wst <= W_SCALE2;
                end
                W_SCALE2: begin
                    w_magic <= {31'd0, w_span} * 54'd2155905153;
                    wst <= W_INVL;
                end
                W_INVL: begin
                    w_scale <= {1'b0, sc_now};
                    wst <= W_INVR;
                end
                W_INVR: begin
                    w_inv <= inv_rd;
                    w_qi  <= 0;
                    qd_v  <= 1'b0;
                    wst   <= W_QNT;
                end
                W_QNT: begin
                    qd_r   <= qd_now;
                    w_qi_d <= w_qi;
                    qd_v   <= (w_qi != HR);
                    if (qd_v) wstage[w_qi_d*P*KBITS +: P*KBITS] <= q_codes;
                    if (w_qi == HR) begin
                        dma_bi <= 0;
                        wst <= W_DMA;
                    end else w_qi <= w_qi + 1'b1;
                end
                W_DMA: begin
                    // wr_pkt_valid is 1 combinationally (wst==W_DMA); move on
                    // once the write engine has accepted this beat.
                    if (wr_pkt_ready) wst <= W_DACK;
                end
                W_DACK: begin
                    // wr_ack_ready is 1 combinationally (wst==W_DACK).
                    if (wr_ack_valid) begin
                        if (dma_bi == hdr_bi) begin
                            wq_done <= 1'b1;
                            wst <= W_IDLE;
                        end else begin
                            dma_bi <= dma_bi + 1'b1;
                            wst <= W_DMA;
                        end
                    end
                end
                default: wst <= W_IDLE;
            endcase
        end
    end

    // ---- read FSM ------------------------------------------------------------
    // RR_REQ issues one beat's DMA read request; RR_WAIT captures its return
    // (one outstanding request at a time, see header); once ROW_BEATS beats
    // for the current position have landed, RR_EMIT dequantises and pulses
    // rd_valid/rd_data (and rd_done on the last position), then either loops
    // back to RR_REQ for the next position or returns to RR_IDLE.
    localparam [2:0] RR_IDLE=3'd0, RR_REQ=3'd1, RR_WAIT=3'd2, RR_EMIT=3'd3;
    reg [2:0] rrst;

    reg [PBW-1:0] r_pbase;             // base row of the (layer,kv,head) selector
    reg [8:0]     r_rowi;              // position counter (0..tcount-1)
    reg [8:0]     r_nrows;             // tcount
    reg [BI_W-1:0] r_bi;               // beat index within the current row

    reg [CODE_BEATS*DATA_W-1:0] r_codebuf;  // captured code beats (padded, same shape as wstage)
    reg [DATA_W-1:0]            r_hdrbeat;  // captured header beat ({scale,lo} in low 48 bits)

    wire [PBW-1:0]    r_cur_row  = r_pbase + {{(PBW-9){1'b0}}, r_rowi};
    wire [ADDR_W-1:0] r_row_base = KV_DDR_BASE[ADDR_W-1:0]
                                    + ({{(ADDR_W-PBW){1'b0}}, r_cur_row} * ROW_BYTES);

    assign rd_req_valid = (rrst == RR_REQ);
    assign rd_req_addr  = r_row_base + ({{(ADDR_W-BI_W){1'b0}}, r_bi} * BEAT_BYTES);
    assign rd_ret_ready = (rrst == RR_WAIT);

    // dequant math verbatim from kv_bank.sv's deq_word combinational block:
    // x_hat = code * scale + lo, exact integer.
    reg [HEAD_DIM*32-1:0] deq_word_r;
    reg [KBITS-1:0] rd_code;
    reg [KBITS+16-1:0] rd_prod;
    reg signed [33:0] rd_full;
    integer rdp;
    always @* begin
        deq_word_r = {(HEAD_DIM*32){1'b0}};
        for (rdp = 0; rdp < HEAD_DIM; rdp = rdp + 1) begin
            rd_code = r_codebuf[rdp*KBITS +: KBITS];
            rd_prod = rd_code * r_hdrbeat[47:32];
            rd_full = $signed({1'b0, rd_prod}) + $signed(r_hdrbeat[31:0]);
            deq_word_r[rdp*32 +: 32] = rd_full[31:0];
        end
    end

    always @(posedge clk) begin
        rd_valid <= 1'b0;
        rd_done  <= 1'b0;
        if (rst) begin
            rrst <= RR_IDLE;
        end else begin
            case (rrst)
                RR_IDLE: if (rd_start) begin
                    r_nrows <= rd_tcount;
                    r_rowi  <= 9'd0;
                    r_pbase <= ((rd_layer*2 + {3'b0,rd_kv})*NHEAD + {2'b0,rd_head})*TMAX;
                    r_bi    <= {(BI_W){1'b0}};
                    rrst    <= RR_REQ;
                end
                RR_REQ: begin
                    // rd_req_valid is 1 combinationally (rrst==RR_REQ); move
                    // on once the read engine has accepted this request.
                    if (rd_req_ready) rrst <= RR_WAIT;
                end
                RR_WAIT: begin
                    // rd_ret_ready is 1 combinationally (rrst==RR_WAIT).
                    if (rd_ret_valid) begin
                        if (r_bi == hdr_bi) begin
                            r_hdrbeat <= rd_ret_data;
                            rrst      <= RR_EMIT;
                        end else begin
                            r_codebuf[r_bi*DATA_W +: DATA_W] <= rd_ret_data;
                            r_bi <= r_bi + 1'b1;
                            rrst <= RR_REQ;
                        end
                    end
                end
                RR_EMIT: begin
                    rd_valid <= 1'b1;
                    rd_data  <= deq_word_r;
                    if (r_rowi == r_nrows - 9'd1) begin
                        rd_done <= 1'b1;
                        rrst    <= RR_IDLE;
                    end else begin
                        r_rowi <= r_rowi + 9'd1;
                        r_bi   <= {(BI_W){1'b0}};
                        rrst   <= RR_REQ;
                    end
                end
                default: rrst <= RR_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        if ((KBITS % 8) != 0)
            $display("kv_bank_ddr: WARNING KBITS=%0d is not byte-aligned -- the beat/mask packing above assumes CODE_BITS=HEAD_DIM*KBITS is a whole number of bytes", KBITS);
        if ((KV_DDR_BASE % BEAT_BYTES) != 0)
            $display("kv_bank_ddr: WARNING KV_DDR_BASE=%0d is not beat-aligned (must be a multiple of %0d bytes)", KV_DDR_BASE, BEAT_BYTES);
    end
`endif
endmodule
