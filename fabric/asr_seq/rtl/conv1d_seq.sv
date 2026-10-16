// -----------------------------------------------------------------------------
// conv1d_seq -- Stage 1 conv front-end's conv1d engine (gen2asr/
// ASR-ACCELERATOR-OP-SEQUENCE.md Stage 1 table: conv1/conv2/conv3, "No conv
// engine anywhere in kevgpt_seq -- every existing compute block
// (gemv_banked_resident_vec) is a GEMV (dense matmul), not a sliding-window
// op. This is the single biggest new hardware component."). This module
// reframes conv1d as a SEQUENCE of TOUT ordinary GEMV calls against the SAME
// resident weight image, each fed a gathered strided window -- no new MAC
// hardware, gemv_banked_resident_vec.sv (already proven, fabric/stage3/) is
// reused unmodified, same G_XRESET/G_XFEED/G_XSTART/G_WAIT/G_DRAIN dispatcher
// idiom as decoder_block_seq.sv/encoder_block_seq.sv.
//
// Window/weight layout (own choice, not PyTorch's): real conv weight is
// [COUT][CIN_real][KW] (PyTorch's own nn.Conv1d layout); this module wants a
// K-major/channel-minor window so it can be built out of PLAIN ROW COPIES of
// the t-major/channel-minor input storage every other block in this project
// already uses (groupnorm1_vec.sv's own xbank, encoder_block_seq.sv's own
// xres_bank) -- so the packer transposes the weight's last two axes at
// export time (W.transpose(0,2,1), trivial in numpy) to get
// W'[cout, k*CIN+ci] = W[cout, ci, k], and this module's own xtbank is
// [TIN][CIN/P] (row r = t*CGRP+cg). The payoff: for output position t_out,
// the whole KW*CIN-element window is exactly KMAX/P=KW*CGRP CONSECUTIVE
// xtbank rows starting at row (t_out*STRIDE)*CGRP -- a straight linear
// address, no gather/mux logic at all (t_out*STRIDE is always a multiple of
// CGRP's own row unit since STRIDE only ever appears as a row-COUNT
// multiplier here, not a bit position).
//
// CIN is the EFFECTIVE (possibly padded) channel count and MUST be a
// multiple of P: conv1's real input has 1 channel (raw audio), which does
// NOT divide P=8 -- rather than special-case that shape, this module always
// pads CIN up to P (CIN=8 for conv1, with the packer zeroing the weight's 7
// extra per-tap lanes so the extra MACs contribute exactly 0). This wastes
// 8x the K-dimension work for conv1 specifically (127 real taps -> 1016
// padded ones) -- a known, bounded, and cheap cost (conv1's real work is
// tiny next to conv2/conv3's own GEMVs or any of this project's LLM linear
// layers) traded for ONE shape-generic RTL module instead of two. Same
// "correct now, a real later resource-optimization option if needed, not a
// correctness requirement" idiom this project already uses for vec_tanh.sv/
// vec_silu.sv's own unpaired-lane tradeoff.
//
// ---- Per-row (per-output-channel) dequant -- vec_dequant.sv, real, reused
// unmodified (checkpoint C's own proven mantissa*2^exponent-per-row scheme,
// the SAME machinery output_head_seq.sv already uses for its own lm_head
// GEMV). Originally this module used a single shared dq_shift (one shift
// for the WHOLE weight matrix); found, while chasing gelu1->conv3's own
// real quantization noise (see conv_front_end_seq.sv's own header), that a
// single shared weight scale under-serves output channels whose own
// dynamic range is much smaller than the matrix's own max -- a real ~2.4x
// error on at least one real element even after fixing the separate
// wasted-INT8-headroom issue. Per-row weight quantization (one wshift per
// OUTPUT channel, `conv1d_ref.quantize_weight_per_row`) fixes that, but a
// per-row WEIGHT scale needs a per-row DEQUANT to match -- a single global
// dq_shift can't correctly undo per-row-varying weight scales. Preload:
// dq_we for MROWS cycles (mant/exp per output row, P lanes/cycle, same
// auto-incrementing-pointer convention as xt_we/b_we below -- NOT
// output_head_seq.sv's own externally-addressed dq_waddr port, to match
// THIS module's own simpler established style).
//
// Output format: ALWAYS wide Q6.25 (P*32b/lane, optional per-output-channel
// bias added in the SAME domain post-dequant) -- the SAME "real activation"
// format layernorm_vec_gendiv.sv/groupnorm1_vec.sv's own x_in expects,
// regardless of what real activation (tanh/gelu) comes next. Format
// conversion to whatever THAT LUT needs (Q4.12) and re-quantization back to
// INT8 for the NEXT conv's own x_we feed are a top-level-FSM/glue concern
// (conv_front_end_seq.sv), not this module's -- keeps this module bit-
// format-agnostic and reusable as-is for conv1/conv2/conv3 via 3 separate
// parameterizations. vec_dequant.sv's own target fraction (its `frac` port)
// is wired to a fixed 0 here: the per-row (mant,exp) table is chosen so
// mant*2^exp ALREADY targets Q4.12 directly (matching output_head_seq.sv's
// own DQ_FRAC=0 convention -- "no further scale" needed once the scale
// itself is computed to land exactly where it should).
//
// Protocol: preload weights (gv_ld_we/gv_ld_data, passthrough to the
// internal gemv core), per-row dequant table (dq_we/dq_wmant/dq_wexp,
// MROWS=COUT/P rows), and, if HAS_BIAS, bias (b_we/b_data, MROWS rows)
// ONCE; preload the whole input tensor (xt_we/xt_data, XTROWS=TIN*CGRP
// rows, t-major/channel-minor) ONCE; pulse `go` for the WHOLE conv (all TOUT
// output positions in one call); y_valid/y_data stream MROWS rows per
// output position, TOUT*MROWS rows total, same t-major/channel-minor order
// as the input (so this module's own output can feed directly into the
// next stage's input bank without reshuffling).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module conv1d_seq #(
    parameter integer P        = 8,
    parameter integer WBW      = 8,
    parameter integer CIN      = 8,        // effective (possibly padded) input channels, multiple of P
    parameter integer COUT     = 288,      // output channels, multiple of P
    parameter integer KW       = 127,      // kernel width (real taps; CIN padding does not affect this)
    parameter integer STRIDE   = 64,
    parameter integer TIN      = 3000,     // input timesteps
    parameter integer HAS_BIAS = 0,
    parameter integer LANES    = 128,
    parameter integer WWORDS   = 1 << 20
) (
    input  wire        clk,
    input  wire        rst,
    // one-time weight preload (passthrough to the internal gemv core)
    input  wire        gv_ld_rst,
    input  wire        gv_ld_we,
    input  wire [31:0] gv_ld_data,
    // one-time per-row dequant table preload (MROWS rows, auto-incrementing
    // pointer, P lanes/row -- matches vec_dequant.sv's own packed-bus convention)
    input  wire             dq_we,
    input  wire [P*24-1:0]  dq_wmant,
    input  wire [P*8-1:0]   dq_wexp,
    // one-time bias preload (only consumed if HAS_BIAS; MROWS rows, Q6.25, added post-dequant)
    input  wire            b_we,
    input  wire [P*32-1:0] b_data,
    // one-time input-tensor preload (XTROWS rows, INT8, t-major/channel-minor)
    input  wire           xt_we,
    input  wire [P*8-1:0] xt_data,
    // run: one pulse computes ALL TOUT output positions
    input  wire  go,
    output reg   done,
    output reg          y_valid,
    output reg [P*32-1:0] y_data
);
    localparam integer TOUT   = (TIN - KW) / STRIDE + 1;
    localparam integer CGRP   = CIN / P;          // input row-groups per timestep
    localparam integer MROWS  = COUT / P;         // output rows per timestep
    localparam integer KMAX   = KW * CIN;
    localparam integer MMAX   = COUT;
    localparam integer XTROWS = TIN * CGRP;
    localparam integer KROWS  = KMAX / P;         // feed cycles per output position

    // ---- input tensor storage: one wide row per (t,cg), t-major/channel-minor
    (* ram_style = "distributed" *) reg [P*8-1:0] xtbank [0:XTROWS-1];
    reg [$clog2(XTROWS+1)-1:0] xt_wptr;
    always @(posedge clk) begin
        if (rst) xt_wptr <= 0;
        else if (xt_we) begin xtbank[xt_wptr] <= xt_data; xt_wptr <= xt_wptr + 1'b1; end
    end

    // ---- bias storage (only meaningful if HAS_BIAS) ---------------------------
    (* ram_style = "distributed" *) reg [P*32-1:0] b_bank [0:(MROWS>0?MROWS-1:0)];
    reg [$clog2(MROWS+1)-1:0] b_wptr;
    always @(posedge clk) begin
        if (rst) b_wptr <= 0;
        else if (b_we) begin b_bank[b_wptr] <= b_data; b_wptr <= b_wptr + 1'b1; end
    end

    // ---- per-row dequant (mant,exp) table --------------------------------------
    (* ram_style = "distributed" *) reg [P*24-1:0] mant_bank [0:(MROWS>0?MROWS-1:0)];
    (* ram_style = "distributed" *) reg [P*8-1:0]  exp_bank  [0:(MROWS>0?MROWS-1:0)];
    reg [$clog2(MROWS+1)-1:0] dq_wptr;
    always @(posedge clk) begin
        if (rst) dq_wptr <= 0;
        else if (dq_we) begin
            mant_bank[dq_wptr] <= dq_wmant;
            exp_bank[dq_wptr]  <= dq_wexp;
            dq_wptr <= dq_wptr + 1'b1;
        end
    end

    // ---- gemv_banked_resident_vec (real, WBW=8) --------------------------------
    reg                          gv_start;
    reg                          gv_xrst;
    wire                         gv_done;
    reg  [$clog2(MMAX+1)-1:0]    gv_m;
    reg  [$clog2(KMAX+1)-1:0]    gv_k;
    reg  [$clog2(WWORDS)-1:0]    gv_wbase;
    reg                          gv_xwe;
    reg  [P*8-1:0]               gv_xdata;
    wire [$clog2((MMAX+LANES-1)/LANES+1)-1:0] gv_gdone;
    wire [$clog2(MMAX/P + 2)-1:0] gv_rdaddr = gi;   // combinational -- see decoder_block_seq.sv's own gv_rdaddr comment
    wire [P*32-1:0]              gv_yout;
    gemv_banked_resident_vec #(.LANES(LANES), .WBW(WBW), .P(P), .MMAX(MMAX), .KMAX(KMAX),
                                .WWORDS(WWORDS), .RLAT(2), .K2(0), .MEM_PRIMITIVE("block")) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ld_rst || gv_xrst), .w_we(gv_ld_we), .w_data(gv_ld_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .gdone(gv_gdone),
        .rd_addr(gv_rdaddr[$clog2(MMAX/P)-1:0]), .y_out(gv_yout),
        .emb_sel(1'b0), .emb_addr({$clog2(WWORDS){1'b0}}), .emb_pair(),
        .wbdiag_addr({$clog2(WWORDS){1'b0}}), .wbdiag_pair()
    );

    // ---- vec_dequant (real, unmodified) -- per-row mant/exp dequant ------------
    reg                    vdq_in_valid;
    reg  [P*32-1:0]        vdq_gemvy;
    reg  [P*24-1:0]        vdq_mant;
    reg  [P*8-1:0]         vdq_exp;
    wire                   vdq_out_valid;
    wire [P*32-1:0]        vdq_dq_out;
    vec_dequant #(.P(P)) u_dequant (
        .clk(clk), .rst(rst), .in_valid(vdq_in_valid), .frac(7'sd0),
        .gemvy(vdq_gemvy), .mant(vdq_mant), .exp(vdq_exp),
        .out_valid(vdq_out_valid), .dq_out(vdq_dq_out)
    );

    // ---- FSM --------------------------------------------------------------------
    localparam [3:0] S_IDLE=0, S_TSTART=1, G_XRESET=2, G_XFEED=3, G_XSTART=4,
                      G_WAIT=5, G_DRAIN=6, S_TNEXT=7, S_DONE=8;
    reg [3:0] state;
    reg [$clog2(TOUT+1)-1:0]      t_out;
    reg [$clog2(XTROWS+1)-1:0]    t_base;
    reg [$clog2(KROWS+1)-1:0]     ri_g;
    reg [$clog2(MROWS+3)-1:0]     gi, gi2;
    reg [$clog2(MROWS+1)-1:0]     ro;            // OUTPUT drain counter -- separate from gi
    // (input feed / GEMV-readback counter) since vec_dequant.sv's own
    // 3-cycle pipeline means the ro-th output row arrives several cycles
    // after the ro-th input row was fed, same "gi vs ri_out" separation
    // output_head_seq.sv's own S_LMDRAIN already established.
    integer biap;
    reg signed [31:0] bias_lane;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; y_valid <= 1'b0;
            t_out <= 0; ri_g <= 0; gi <= 0; ro <= 0;
            gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0; vdq_in_valid <= 1'b0;
        end else begin
            done    <= 1'b0;
            y_valid <= 1'b0;
            gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0; vdq_in_valid <= 1'b0;
            case (state)
                S_IDLE: if (go) begin t_out <= 0; state <= S_TSTART; end
                S_TSTART: begin
                    t_base <= t_out * STRIDE * CGRP;
                    ri_g   <= 0;
                    gv_m   <= MMAX[$clog2(MMAX+1)-1:0];
                    gv_k   <= KMAX[$clog2(KMAX+1)-1:0];
                    gv_wbase <= {$clog2(WWORDS){1'b0}};
                    state  <= G_XRESET;
                end
                G_XRESET: begin gv_xrst <= 1'b1; state <= G_XFEED; end
                G_XFEED: begin
                    gv_xdata <= xtbank[t_base + ri_g];
                    gv_xwe   <= 1'b1;
                    if (ri_g != KROWS-1) ri_g <= ri_g + 1'b1;
                    else begin ri_g <= 0; state <= G_XSTART; end
                end
                G_XSTART: begin gv_start <= 1'b1; state <= G_WAIT; end
                G_WAIT: if (gv_done) begin gi <= 0; ro <= 0; state <= G_DRAIN; end
                // ---- combined GEMV readback + vec_dequant feed + bias-add
                // drain (same "input counter != output counter, both
                // advancing concurrently" idiom as output_head_seq.sv's own
                // S_LMDRAIN -- vec_dequant.sv is a plain 3-cycle-latency
                // streaming pipeline, no start/done handshake, so most
                // vdq_out_valid pulses land WHILE gi is still advancing, not
                // after). ----
                G_DRAIN: begin
                    if (gi >= 2) begin
                        gi2 = gi - 2;
                        vdq_in_valid <= 1'b1;
                        vdq_gemvy <= gv_yout;
                        vdq_mant  <= mant_bank[gi2[$clog2(MROWS+1)-1:0]];
                        vdq_exp   <= exp_bank[gi2[$clog2(MROWS+1)-1:0]];
                    end
                    if (gi != MROWS+1) gi <= gi + 1'b1;

                    if (vdq_out_valid) begin
                        for (biap = 0; biap < P; biap = biap + 1) begin
                            bias_lane = (HAS_BIAS != 0) ? $signed(b_bank[ro][biap*32 +: 32]) : 32'sd0;
                            y_data[biap*32 +: 32] <= $signed(vdq_dq_out[biap*32 +: 32]) + bias_lane;
                        end
                        y_valid <= 1'b1;
                        if (ro == MROWS-1) state <= S_TNEXT;
                        else ro <= ro + 1'b1;
                    end
                end
                S_TNEXT: begin
                    if (t_out == TOUT-1) state <= S_DONE;
                    else begin t_out <= t_out + 1'b1; state <= S_TSTART; end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule // conv1d_seq
