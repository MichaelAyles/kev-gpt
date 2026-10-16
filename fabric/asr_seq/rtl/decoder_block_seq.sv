// -----------------------------------------------------------------------------
// decoder_block_seq -- the real, sized, FUNCTIONALLY GATED top-level FSM for
// ASR's decoder block (Stage 3b of gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md).
// Chains real, already-gated sub-modules (kv_bank.sv x2, vec_attn_w.sv,
// layernorm_vec_gendiv.sv, gemv_banked_resident_vec.sv WBW=8,
// rope_apply_vec.sv, vec_silu.sv) into one full decoder-layer forward pass:
//   LN1 -> self Q/K/V -> RoPE(self) -> self KV-write -> self-attn(causal)
//     -> O -> RES1 -> LN2 -> cross Q (POST_SCALE, no RoPE) -> cross-attn
//     (static full-attend) -> Oc -> RES2 -> LN3 -> FC1(+bias) -> SwiGLU
//     (SiLU-gate * value) -> FC2(+bias) -> RES3
//
// gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md's "Decoder-block FSM sketch, turned
// into a real, sized state machine" section documents the design and its
// real, flagged open items; a first revision closed the functional-gate gap
// that pass left open (fabric/asr_seq/pack_decoder_block.py +
// tb_decoder_block_seq.sv, real moonshine-tiny layer-0 weights, T2=6, decode
// step 0): BIT-EXACT. The gate now covers real MULTI-STEP decoding (step
// 0..N_STEPS-1, self-attn KV cache growing causally across steps, RoPE
// position advancing) too -- `step`/`blk` were ALREADY runtime ports before
// this, not new; only the gate itself (+ its own fixed, all-steps-profiled
// ACT_*/GF_* quantization constants) needed building. See that gate's own
// verdict line for the current status. Layer-looping (blk=1..5, chaining
// multiple decoder layers) needed two real RTL changes, both the same
// shape: wb_q/wb_k/.../wb_fc2 (weight-image offsets) AND gf_q/gf_k/.../
// gf_fc2 (dequant shifts) were compile-time WB_*/GF_* parameters in the
// single-layer gate -- both converted to runtime ports here so the caller
// can drive each layer's own values before its own `go` pulse, the same
// way gam_wdata/bias_wdata/xkv_wdata are already caller-driven per layer.
// GF_* needed converting (WSHIFT, baked into g_frac, is genuinely
// per-layer -- each layer's own weight matrix gets its own optimal
// scale); ACT_* did NOT (one profiled ACT_RSHIFT per call site, verified
// to clip nowhere across all 6 layers x 3 steps -- LayerNorm keeps the
// activation side's scale comparable across layers even though the
// weight side isn't). Per-layer gamma/bias reload (through the existing
// gam_we/bias_we ports) and self-attn's own per-layer kv_bank storage
// (u_self_kv, via `blk`) were already correct. u_cross_kv was NOT: it was
// sized NLAYER=1 with wq_layer/rd_layer hardwired to 4'd0 (fine for the
// single-layer gate, silently wrong for real multi-layer use -- every
// layer's own cross-K/V preload landed in the SAME storage slot, so only
// the LAST layer's own data survived once more than one layer was loaded;
// found as every layer's own decode pass mismatching, even layer 0's, the
// moment a real 6-layer gate first exercised this). Fixed by sizing/wiring
// u_cross_kv exactly like u_self_kv already was.
//
// GEMV QUANTIZATION SCHEME (a real, explicit, documented simplification --
// the production INT8 export/quantization scheme is still genuinely
// undecided upstream, per the op-sequence doc's own note; this is NOT that
// scheme, it is a self-consistent, bit-exactly-reproducible one built for
// THIS gate, matching pack_decoder_block.py's linear_q()/gdequant()/
// act_quantize() exactly):
//   - weights: per-matrix single-scale INT8, w_int8 = round(w_real*2^WSHIFT)
//     (WSHIFT chosen offline in Python so no weight clips; loaded verbatim
//     via gv_ld_*)
//   - activations: per-call single-shift INT8, x_int8 = sat(x_fixed >>>
//     ACT_RSHIFT, -128, 127) -- a plain arithmetic-shift floor, no rounding
//   - dequant: y_fixed = raw >>> g_frac (g_frac = FRAC_IN - ACT_RSHIFT +
//     WSHIFT - FRAC_OUT, computed offline, one combined shift per call)
// ACT_RSHIFT/g_frac are real, DATA-PROFILED constants (see
// pack_decoder_block.py's choose_act_rshift(), which also asserts no
// clipping occurs) -- ACT_Q/ACT_K/... below are compile-time parameters
// (one value covers every real layer's own weights); gf_q/gf_k/... are
// runtime ports instead (see the multi-layer update above for why).
//
// PER-HEAD BANK LAYOUT (fixes the first draft's real indexing-overflow bug):
// q_bank/k_bank/v_bank/q_rope_bank/k_rope_bank/ctx_bank are stored at
// ATTN_P=4-wide row granularity (NHEAD*HR_ATTN = 72 rows over D=288),
// matching kv_bank.sv/vec_attn_w.sv's own native P -- NOT at the GEMV's
// P=8-wide granularity (36 rows), which does not divide evenly by
// HEAD_DIM=36 (36 mod 8 = 4, a real per-head misalignment) and was the
// first draft's actual bug (row-index overflow, not just "elided"). The
// P=8<->ATTN_P=4 boundary crossing (GEMV's own P=8 activation/readback
// width) is a clean 1:2 split/combine EVERYWHERE, since P=8 = 2*ATTN_P=4
// exactly, regardless of where within D it happens.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module decoder_block_seq #(
    parameter integer P         = 8,
    parameter integer D         = 288,
    parameter integer FFN       = 1152,
    parameter integer DFFN2     = 2*FFN,    // 2304
    parameter integer NHEAD     = 8,
    parameter integer HEAD_DIM  = 36,
    parameter integer ATTN_P    = 4,
    parameter integer ATTN_TMAX = 8,        // self-attn cache depth (>= DECODE_STEPS)
    parameter integer T2        = 6,        // cross-attn K/V length (this gate's real T2)
    parameter integer ROT_PAIRS = 16,
    parameter integer ROPE_TMAX = 128,
    // ---- real, data-profiled quantization constants (pack_decoder_block.py) --
    parameter integer POST_SCALE_Q16 = 75674,
    // ACT_* stay compile-time parameters (unlike GF_*, below) -- one fixed
    // ACT_RSHIFT per call site, profiled to safely cover every layer AND
    // every step, verified to clip nowhere (LayerNorm re-normalizes the
    // residual stream's scale at each layer boundary, so the activation
    // side genuinely doesn't need a per-layer value; see
    // pack_decoder_block.py's own header).
    parameter signed [6:0] ACT_Q=17,  ACT_K=17,  ACT_V=17,  ACT_O=15,
                           ACT_CQ=19, ACT_OC=20, ACT_FC1=17, ACT_FC2=11
) (
    input  wire clk,
    input  wire rst,

    input  wire        go,             // pulse: run ONE decoder layer, ONE decode step
    input  wire [3:0]  blk,            // self kv_bank's layer index
    input  wire [8:0]  step,           // decode step (self-attn tcount=step+1)
    output reg         done,

    // real word offsets into the resident weight image (pack_banked_
    // resident_vec.build_resident()'s own layer_meta[i]["w_base"] --
    // depends on each matrix's (M,K) shape, NOT a simple 0..7 layer index).
    // Runtime ports, NOT compile-time parameters (the first draft's own
    // choice) -- multi-LAYER decoding needs a DIFFERENT weight offset per
    // layer for the SAME call site (layer 1's own Q-projection lives at a
    // different resident-image address than layer 0's), and `blk` alone
    // can't index into a parameter. The caller is expected to drive these
    // to the CURRENT layer's own offsets before each `go` pulse, the same
    // way gam_wdata/bias_wdata/xkv_wdata are already caller-driven per
    // layer, not baked in.
    input  wire [19:0] wb_q, wb_k, wb_v, wb_o,
    input  wire [19:0] wb_cq, wb_co, wb_fc1, wb_fc2,

    // dequant shift per call site -- ALSO runtime ports, not GF_Q/GF_K/...
    // parameters: g_frac = FRAC_IN - ACT_RSHIFT + WSHIFT - FRAC_OUT, and
    // WSHIFT is genuinely per-layer (each layer's own weight matrix gets
    // its own optimal scale, chosen independently so its own INT8 image
    // doesn't clip) -- unlike ACT_RSHIFT (above), a single compile-time
    // GF_* could NOT stay fixed across layers without forcing every
    // layer's weights to share one, more conservative, WSHIFT (a real
    // precision trade-off this file doesn't take). Found by the two-pass
    // profiling gate's own g_frac/act_rshift-must-be-fixed assertion
    // tripping on WSHIFT drift between layer 0 and layer 1, not ACT_RSHIFT.
    input  wire signed [7:0] gf_q, gf_k, gf_v, gf_o,
    input  wire signed [7:0] gf_cq, gf_co, gf_fc1, gf_fc2,

    // residual stream in/out -- P*32-bit packed rows, D/P rows, Q6.25
    input  wire                   xres_wr,
    input  wire [$clog2(D/P)-1:0] xres_waddr,
    input  wire [P*32-1:0]        xres_wdata,
    output wire [P*32-1:0]        xres_rdata_dbg,

    // ---- GEMV weight load (passthrough to gemv_banked_resident_vec) --------
    input  wire        gv_ld_rst,
    input  wire        gv_ld_we,
    input  wire [31:0] gv_ld_data,

    // ---- LN gamma load: sel picks LN1/LN2/LN3 (0/1/2) ----------------------
    input  wire         gam_we,
    input  wire [1:0]   gam_sel,
    input  wire [$clog2(D/P)-1:0] gam_waddr,
    input  wire [P*32-1:0]        gam_wdata,

    // ---- bias load: sel picks fc1/fc2 (0/1) ---------------------------------
    input  wire         bias_we,
    input  wire         bias_sel,
    input  wire [$clog2(DFFN2/P)-1:0] bias_waddr,
    input  wire [P*32-1:0]            bias_wdata,

    // ---- cross-attn K/V preload passthrough (Stage 3a, out of this file's
    // own scope -- see header) ------------------------------------------------
    input  wire        xkv_wstart,
    input  wire        xkv_wkv,
    input  wire [$clog2(NHEAD)-1:0] xkv_whead,
    input  wire [8:0]  xkv_wpos,
    input  wire        xkv_wvalid,
    input  wire [ATTN_P*32-1:0] xkv_wdata,
    output wire         xkv_wdone
);
    localparam integer ROWS_D     = D / P;              // 36
    localparam integer ROWS_FFN2  = DFFN2 / P;           // 288
    localparam integer ROWS_FFN   = FFN / P;             // 144
    localparam integer HR_ATTN    = HEAD_DIM / ATTN_P;   // 9
    localparam integer ROWS_AD    = NHEAD * HR_ATTN;     // 72 (D at ATTN_P width)

    // =========================================================================
    // ---- residual + LN output + generic GEMV dest banks (P=8-wide) --------
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank   [0:ROWS_D-1];      // Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] ln_out_bank [0:ROWS_D-1];      // Q.22
    (* ram_style = "block" *) reg [P*32-1:0] gout_bank   [0:ROWS_FFN2-1];   // generic GEMV dest
    (* ram_style = "block" *) reg [P*32-1:0] combined_bank [0:ROWS_FFN-1]; // Q4.12, post SiLU*gate

    assign xres_rdata_dbg = xres_bank[xres_waddr];
    always @(posedge clk) if (xres_wr) xres_bank[xres_waddr] <= xres_wdata;

    // ---- gamma tables (3 x D, Q4.20) ----
    (* ram_style = "block" *) reg [P*32-1:0] gamma_bank [0:2][0:ROWS_D-1];
    always @(posedge clk) if (gam_we) gamma_bank[gam_sel][gam_waddr] <= gam_wdata;

    // ---- bias tables (fc1: DFFN2, Q.12; fc2: D, Q.25) ----
    (* ram_style = "block" *) reg [P*32-1:0] bias_fc1_bank [0:ROWS_FFN2-1];
    (* ram_style = "block" *) reg [P*32-1:0] bias_fc2_bank [0:ROWS_D-1];
    always @(posedge clk) if (bias_we) begin
        if (!bias_sel) bias_fc1_bank[bias_waddr[$clog2(ROWS_FFN2)-1:0]] <= bias_wdata;
        else           bias_fc2_bank[bias_waddr[$clog2(ROWS_D)-1:0]]    <= bias_wdata;
    end

    // ---- per-head banks (ATTN_P=4-wide, 72 rows) ----
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] q_bank      [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] k_bank      [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] v_bank      [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] q_rope_bank [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] k_rope_bank [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] ctx_bank    [0:ROWS_AD-1];  // Q.25

    // =========================================================================
    // ---- rope_apply_vec (real, unmodified) -- POST_SCALE_Q16 corrects
    // vec_attn_w.sv's HEAD_DIM=64-specific SCORE_SH for ASR's HEAD_DIM=36
    // (same correction as every prior ASR attention gate). --------------------
    reg                       rope_start;
    reg  [$clog2(ROPE_TMAX)-1:0] rope_pos;
    reg  [HEAD_DIM*32-1:0]    rope_head_in;
    wire                      rope_done;
    wire [HEAD_DIM*32-1:0]    rope_head_out;
    rope_apply_vec #(.HEAD_DIM(HEAD_DIM), .ROT_DIM(32), .ROT_PAIRS(ROT_PAIRS),
                      .TMAX(ROPE_TMAX), .POST_SCALE_Q16(POST_SCALE_Q16)) u_rope (
        .clk(clk), .start(rope_start), .position(rope_pos),
        .head_in(rope_head_in), .done(rope_done), .head_out(rope_head_out)
    );

    // ---- gather head hh's HR_ATTN rows into one flat HEAD_DIM*32 bus, and
    // the reverse scatter -- plain concatenation (72-row banks make every
    // head's HR_ATTN=9 rows contiguous, no cross-alignment). ------------------
    integer gi_r;
    integer sc_i;   // scatter loop var (S_ROPE_Q1/S_ROPE_K1) -- separate
    // from gi_r (a DIFFERENT always block; sharing a for-loop var across two
    // procedural blocks is a real multi-driver risk in Verilog, not just style).
    reg [HEAD_DIM*32-1:0] gather_q, gather_k;
    always @* begin
        gather_q = {(HEAD_DIM*32){1'b0}};
        gather_k = {(HEAD_DIM*32){1'b0}};
        for (gi_r = 0; gi_r < HR_ATTN; gi_r = gi_r + 1) begin
            gather_q[gi_r*ATTN_P*32 +: ATTN_P*32] = q_bank[hh*HR_ATTN + gi_r];
            gather_k[gi_r*ATTN_P*32 +: ATTN_P*32] = k_bank[hh*HR_ATTN + gi_r];
        end
    end

    // =========================================================================
    // ---- self-attn kv_bank (real, unmodified) -------------------------------
    reg         sk_wstart, sk_wvalid, sk_rstart;
    reg  [3:0]  sk_wlayer;  reg sk_wkv;  reg [$clog2(NHEAD)-1:0] sk_whead;  reg [8:0] sk_wpos;
    reg  [ATTN_P*32-1:0] sk_wdata;
    wire        sk_wdone;
    reg  [3:0]  sk_rlayer;  reg sk_rkv;  reg [$clog2(NHEAD)-1:0] sk_rhead;  reg [8:0] sk_rtcount;
    wire        sk_rvalid, sk_rdone;
    wire [HEAD_DIM*32-1:0] sk_rdata;
    kv_bank #(.P(ATTN_P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(16), .TMAX(ATTN_TMAX),
              .KBITS(8), .MEM_PRIMITIVE("block")) u_self_kv (
        .clk(clk), .rst(rst),
        .wq_start(sk_wstart), .wq_layer(sk_wlayer), .wq_kv(sk_wkv), .wq_head(sk_whead),
        .wq_pos(sk_wpos), .wq_valid(sk_wvalid), .wq_data(sk_wdata), .wq_done(sk_wdone),
        .rd_start(sk_rstart), .rd_layer(sk_rlayer), .rd_kv(sk_rkv), .rd_head(sk_rhead),
        .rd_tcount(sk_rtcount), .rd_valid(sk_rvalid), .rd_data(sk_rdata), .rd_done(sk_rdone),
        .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head({$clog2(NHEAD){1'b0}}),
        .rd2_tcount(9'd0), .rd2_valid(), .rd2_data(), .rd2_done()
    );

    // ---- cross-attn kv_bank (real, unmodified) -- write port exposed via
    // xkv_* passthrough (Stage 3a's own precompute is out of this file's
    // scope; the OUTER testbench/integration drives these). -------------------
    // NLAYER(1)/wq_layer(4'd0)/rd_layer(4'd0) (the original code here) only
    // ever worked for the single-layer gate: ALL layers' cross-K/V preload
    // landed in the SAME layer-0 storage slot, so loading layer 1's cross
    // K/V silently overwrote layer 0's -- by the time all NLAYER layers'
    // preload finished, only the LAST layer's own data survived. Sized and
    // wired to `blk` now, mirroring u_self_kv (above) exactly -- found via
    // every layer's own decode pass mismatching, even layer 0's, the
    // moment a real multi-layer gate first exercised more than one layer's
    // worth of cross-attn K/V.
    reg  [3:0]  xk_rlayer;
    reg         xk_rstart;  reg xk_rkv;  reg [$clog2(NHEAD)-1:0] xk_rhead;  reg [8:0] xk_rtcount;
    wire        xk_rvalid, xk_rdone;
    wire [HEAD_DIM*32-1:0] xk_rdata;
    kv_bank #(.P(ATTN_P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(16), .TMAX(T2),
              .KBITS(8), .MEM_PRIMITIVE("block")) u_cross_kv (
        .clk(clk), .rst(rst),
        .wq_start(xkv_wstart), .wq_layer(blk), .wq_kv(xkv_wkv), .wq_head(xkv_whead),
        .wq_pos(xkv_wpos), .wq_valid(xkv_wvalid), .wq_data(xkv_wdata), .wq_done(xkv_wdone),
        .rd_start(xk_rstart), .rd_layer(xk_rlayer), .rd_kv(xk_rkv), .rd_head(xk_rhead),
        .rd_tcount(xk_rtcount), .rd_valid(xk_rvalid), .rd_data(xk_rdata), .rd_done(xk_rdone),
        .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head({$clog2(NHEAD){1'b0}}),
        .rd2_tcount(9'd0), .rd2_valid(), .rd2_data(), .rd2_done()
    );

    // =========================================================================
    // ---- vec_attn_w (real, unmodified) -- ONE engine, self then cross -------
    reg                    at_start;
    reg  [8:0]             at_tcount;
    reg                    at_qvalid;
    reg  [ATTN_P*32-1:0]   at_qdata;
    reg                    at_kvvalid;
    reg  [HEAD_DIM*32-1:0] at_kvdata;
    wire                   at_kdone, at_ctxvalid, at_done;
    wire [6:0]             at_ctxidx;
    wire [ATTN_P*32-1:0]   at_ctxdata;
    vec_attn_w #(.P(ATTN_P), .HEAD_DIM(HEAD_DIM), .TMAX(ATTN_TMAX > T2 ? ATTN_TMAX : T2)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .q_valid(at_qvalid), .q_data(at_qdata),
        .kv_valid(at_kvvalid), .kv_data(at_kvdata),
        .k_done(at_kdone), .ctx_valid(at_ctxvalid), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata),
        .done(at_done)
    );

    // =========================================================================
    // ---- layernorm_vec_gendiv (real, this session's D=288 fix) -------------
    reg               ln_start, ln_vin;
    reg [P*32-1:0]    ln_x, ln_g;
    wire              ln_yvalid, ln_done;
    wire [P*64-1:0]   ln_yout;
    layernorm_vec_gendiv #(.P(P), .D(D)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yvalid), .y_out(ln_yout), .done(ln_done)
    );
    // ln_yout packs each of its P lanes into 64 bits (sign-extended Q.22,
    // `y_out[lp*64+:64]` -- see layernorm_vec_gendiv.sv's own header), NOT
    // P*32 -- taking ln_yout[P*32-1:0] directly (the original code here) grabs
    // only lanes 0..3's FULL 64-bit slots and misreads them as 8 packed
    // 32-bit lanes, landing each real value's low 32 bits at EVEN "lane"
    // positions with its high 32 bits (mostly 0) landing at the ODD
    // positions in between -- e.g. real [v0,v1,v2,v3] read back as
    // [v0,0,v1,0,v2,0,v3,0], and lanes 4..7 never read at all. Re-pack by
    // taking each lane's own low 32 bits explicitly. Found by comparing
    // ln_out_bank[0] against pack_decoder_block.py's own xn1 -- the exact
    // [v0,0,v1,0,...] pattern was the tell.
    integer lnp;
    reg [P*32-1:0] ln_out_word;
    always @* begin
        ln_out_word = {(P*32){1'b0}};
        for (lnp = 0; lnp < P; lnp = lnp + 1)
            ln_out_word[lnp*32 +: 32] = ln_yout[lnp*64 +: 32];
    end

    // =========================================================================
    // ---- gemv_banked_resident_vec (real, WBW=8) -----------------------------
    localparam integer GEMV_MMAX = DFFN2;
    localparam integer GEMV_KMAX = DFFN2;
    reg                          gv_start;
    reg                          gv_xrst;   // internal per-call xptr rewind
    wire                         gv_done;
    reg  [$clog2(GEMV_MMAX+1)-1:0] gv_m;
    reg  [$clog2(GEMV_KMAX+1)-1:0] gv_k;
    reg  [19:0]                  gv_wbase;
    reg                          gv_xwe;
    reg  [P*8-1:0]               gv_xdata;
    wire [$clog2((GEMV_MMAX+127)/128+1)-1:0] gv_gdone;
    // combinational, tied directly to `gi` -- NOT a registered `gv_rdaddr<=gi`
    // (which lags gi by a cycle). G_DRAIN needs rd_addr=gi[N] valid the SAME
    // cycle gi==N, so that 2 cycles later (gemv_banked_resident_vec.sv's own
    // read->y_out latency) the write at gi==N+2 (gi2=N) sees the CORRECT
    // group-N data. A registered version drove rd_addr from an UNDEFINED reg
    // for G_DRAIN's very first cycle (X on the very first GEMV call of the
    // whole sim; a stale, valid-looking but WRONG address on every later
    // call, since nothing ever reset it between calls) -- found by tracing
    // q_bank[0]/[1] (RoPE's own head-0 rows 0-1) reading X all the way back
    // to gv_yout/gv_rdaddr at G_DRAIN's first cycle. Seeding the register to
    // 0 one cycle early (tried first) only traded the X for an off-by-one:
    // an extra valid-but-duplicate rd_addr=0 cycle shifted every row from
    // index 1 onward to the PREVIOUS row's data. Tying it straight to gi
    // removes the lag instead of patching around it.
    wire [$clog2(GEMV_MMAX/P)-1:0] gv_rdaddr = gi;
    wire [P*32-1:0]              gv_yout;
    gemv_banked_resident_vec #(.LANES(128), .WBW(8), .P(P), .MMAX(GEMV_MMAX), .KMAX(GEMV_KMAX),
                                .WWORDS(1<<20), .RLAT(2), .K2(0), .MEM_PRIMITIVE("block")) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        // gv_xrst: internal per-call activation-pointer rewind (see
        // G_XRESET below) -- gemv_banked_resident_vec.sv's xptr auto-
        // increments on every x_we and never resets on its own; ld_rst
        // is the ONLY way to rewind it, and the established protocol
        // (fabric/stage3/run_resident_banked_vec.py) pulses it before
        // EVERY GEMV call for exactly this reason. Sharing it with
        // gv_ld_rst (OR'd) is safe: re-pulsing it after the initial
        // bulk weight load only resets weight_bank_tdp's transient
        // write-staging registers (wword/wsub/wbuf), never the stored
        // weight content itself, and no more w_we pulses ever happen.
        .ld_rst(gv_ld_rst || gv_xrst), .w_we(gv_ld_we), .w_data(gv_ld_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .gdone(gv_gdone),
        .rd_addr(gv_rdaddr), .y_out(gv_yout),
        .emb_sel(1'b0), .emb_addr({$clog2(1<<20){1'b0}}), .emb_pair(),
        .wbdiag_addr({$clog2(1<<20){1'b0}}), .wbdiag_pair()
    );

    // real dequant: y = raw >>> g_frac (g_frac may be negative -> left shift)
    function automatic signed [31:0] gdequant;
        input signed [31:0] raw;
        input signed [7:0]  frac;
        begin
            gdequant = (frac >= 0) ? (raw >>> frac) : (raw <<< (-frac));
        end
    endfunction
    // real act-quant: x_int8 = sat(x_fixed >>> shift, -128, 127)
    function automatic signed [7:0] actquant;
        input signed [31:0] x;
        input signed [6:0]  shift;
        reg signed [31:0] q;
        begin
            q = (shift >= 0) ? (x >>> shift) : (x <<< (-shift));
            if (q > 32'sd127) actquant = 8'sd127;
            else if (q < -32'sd128) actquant = -8'sd128;
            else actquant = q[7:0];
        end
    endfunction
    // round-half-away-from-zero shift -- copied verbatim from
    // rope_apply_vec.sv's own rsh_round (this project's established
    // convention, matching fabric.stage3.seq_ref.rsh_round /
    // pack_decoder_block.py's own use of it for BOTH the POST_SCALE_Q16
    // cross-Q correction and the SiLU gate*value combine). A plain `>>>`
    // (floor shift) at either site is NOT the same function -- it matches
    // python only when the shifted-out bits happen to be zero, so it looked
    // right on plenty of lanes and wrong (off by exactly 1) on the rest.
    // Found via scattered +-1 mismatches surviving after the RES1/RES2/RES3
    // carry-bleed fix (res_add_word) had already made those two exact.
    function automatic signed [63:0] rsh_round64;
        input signed [63:0] v;
        input integer s;
        reg signed [63:0] half;
        begin
            if (s <= 0) rsh_round64 = v <<< (-s);
            else begin
                half = (64'sd1 <<< (s-1));
                if (v >= 0) rsh_round64 = (v + half) >>> s;
                else        rsh_round64 = -(((-v) + half) >>> s);
            end
        end
    endfunction
    // gate saturate: pack_decoder_block.py's own `np.clip(gate, -32768,
    // 32767)` before silu_q -- NO shift, straight saturate of the raw
    // (already-dequantized) Q4.12 gate value into a 16-bit lane.
    function automatic signed [15:0] sat16;
        input signed [31:0] x;
        begin
            if (x > 32'sd32767) sat16 = 16'sd32767;
            else if (x < -32'sd32768) sat16 = -16'sd32768;
            else sat16 = x[15:0];
        end
    endfunction

    // =========================================================================
    // ---- vec_silu (real) ----------------------------------------------------
    reg                     su_vin;
    reg signed [16*P-1:0]   su_x;
    wire                    su_vout;
    wire signed [16*P-1:0]  su_y;
    // gout_bank's row is P LANES OF 32 BITS (Q4.12 gate values fit in 32b
    // storage like every other gout_bank row), NOT P lanes of 16 -- taking
    // gout_bank[...][16*P-1:0] directly (the original code) grabbed only
    // lanes 0..3's full 32-bit values and misread them as 8 packed 16-bit
    // lanes (the SAME "wide lane, narrow reinterpretation" bug already fixed
    // once for ln_out_bank/ln_yout -- see u_ln's own comment above). Each
    // lane must be saturated to 16 bits (pack_decoder_block.py's own
    // np.clip(gate,-32768,32767)), not truncated. Found by comparing
    // combined_bank[0] against pack_decoder_block.py's own `combined` array
    // -- the same [v0,0,v1,0,...]-shaped tell as the LN bug.
    integer sup;
    reg signed [16*P-1:0] su_x_word;
    always @* begin
        su_x_word = {(16*P){1'b0}};
        for (sup = 0; sup < P; sup = sup + 1)
            su_x_word[sup*16 +: 16] = sat16($signed(gout_bank[ROWS_FFN + ri_f][sup*32 +: 32]));
    end
    vec_silu #(.P(P)) u_silu (
        .clk(clk), .in_valid(su_vin), .x(su_x), .out_valid(su_vout), .y(su_y)
    );

    // =========================================================================
    // ---- state encoding -----------------------------------------------------
    localparam [5:0]
        S_IDLE=0,
        S_LN1SET=1,  L_FEED=2,  L_WAIT=3,
        S_QSET=4,    G_XFEED=5, G_WAIT=6, G_DRAIN=7,
        S_KSET=8,    S_VSET=9,
        S_ROPE_Q0=10, S_ROPE_Q1=11, S_ROPE_K0=12, S_ROPE_K1=13,
        S_KVW_K0=14, S_KVW_K1=15, S_KVW_V0=16, S_KVW_V1=17,
        S_ASELF_Q=18, S_ASELF_K0=19, S_ASELF_V0=20, S_ASELF_DRAIN=21,
        S_OSET=22,
        S_RES1=23,
        S_LN2SET=24,
        S_CQSET=25, S_CQSCALE=26,
        S_ACROSS_Q=27, S_ACROSS_K0=28, S_ACROSS_V0=29, S_ACROSS_DRAIN=30,
        S_COSET=31,
        S_RES2=32,
        S_LN3SET=33,
        S_FC1SET=34,
        S_SILU0=35, S_SILU1=36,
        S_FC2SET=37,
        S_RES3=38,
        S_DONE=39,
        // start-alone states (fix: start/first-valid-beat overlap bug --
        // layernorm_vec_gendiv.sv/vec_attn_w.sv both ignore valid_in/q_valid
        // during their OWN start-sampling cycle, so the first beat must land
        // one cycle AFTER start, not the same cycle; found via a real hang,
        // not assumed -- see this file's own commit message).
        L_START=40, S_ASELF_START=41, S_ACROSS_START=42, S_SILU_INIT=43,
        G_XSTART=44, G_XRESET=45;
    reg [5:0] st;

    reg [5:0] l_ret, g_ret;
    reg [1:0] l_gbase;
    reg [2:0] g_src;                         // 0=ln_out, 1=ctx(combine), 2=combined
    reg [2:0] g_dst;                         // 0..3: q/k/v/gout (see G_DRAIN)
    reg signed [7:0] g_frac;
    reg signed [6:0] g_actshift;
    reg g_bias_en; reg g_bias_sel;           // 0=fc1 bank, 1=fc2 bank

    reg [$clog2(NHEAD)-1:0] hh;
    reg [$clog2(HR_ATTN+1)-1:0] wi;

    // sized for ROWS_AD (72), the larger of its two uses (LN row counter,
    // 0..ROWS_D-1=35; S_CQSCALE's per-row scale loop, 0..ROWS_AD-1=71) --
    // ROWS_D alone ($clog2(36)=6 bits, max 63) would silently overflow the
    // second use.
    reg [$clog2(ROWS_AD)-1:0]    ri_d;
    reg [$clog2(ROWS_FFN2)-1:0] ri_g;
    reg [$clog2(ROWS_FFN)-1:0]  ri_f;
    reg [$clog2(GEMV_MMAX/P)-1:0] gi;
    reg [$clog2(GEMV_MMAX/P)-1:0] gi2;   // plain-reg copy of gi-2, for G_DRAIN's part-selects

    // ---- generic GEMV activation source mux (P=8-wide) ----------------------
    // g_src=1 (ctx) COMBINES 2 consecutive ATTN_P=4 rows into one P=8 row --
    // the clean inverse of G_DRAIN's split (8=2*4, always aligned).
    reg [P*32-1:0] g_src_row;
    always @* begin
        case (g_src)
            3'd0: g_src_row = ln_out_bank[ri_g[$clog2(ROWS_D)-1:0]];
            3'd1: g_src_row = {ctx_bank[2*ri_g[$clog2(ROWS_AD/2)-1:0]+1],
                                ctx_bank[2*ri_g[$clog2(ROWS_AD/2)-1:0]]};
            3'd2: g_src_row = combined_bank[ri_g[$clog2(ROWS_FFN)-1:0]];
            default: g_src_row = {(P*32){1'b0}};
        endcase
    end

    integer bp;
    reg [P*8-1:0] act_word;
    // explicit wide product temporaries -- Verilog does NOT auto-widen
    // `*` beyond the assignment target's own width, so these two real
    // multiplies (POST_SCALE_Q16 x Q.16 value, up to ~49 bits; Q4.12 x
    // Q4.12 gate product, up to ~32 bits) get explicit >=64-bit scratch
    // regs rather than risking silent truncation in a bare 32-bit context.
    reg signed [63:0] scale_prod, gate_prod;
    integer bp2;   // separate from `bp` (that one belongs to the always @*
    // act_word block) -- same multi-driver-avoidance reasoning as sc_i above.
    always @* begin
        act_word = {(P*8){1'b0}};
        for (bp = 0; bp < P; bp = bp + 1)
            act_word[bp*8 +: 8] = actquant($signed(g_src_row[bp*32 +: 32]), g_actshift);
    end

    integer dp;
    reg [P*32-1:0] deq_word;
    always @* begin
        deq_word = {(P*32){1'b0}};
        for (dp = 0; dp < P; dp = dp + 1)
            deq_word[dp*32 +: 32] = gdequant($signed(gv_yout[dp*32 +: 32]), g_frac);
    end

    // residual add: MUST be P independent 32-bit signed lane additions, not
    // one flat 256-bit vector `+` (xres_bank/gout_bank are plain `reg
    // [P*32-1:0]`, unsigned, undelimited -- a bare `xres_bank[ri_d] +
    // gout_bank[ri_d]` adds the whole 256-bit word as ONE binary number, so
    // a carry out of lane N's own 32-bit sum silently bleeds into lane N+1.
    // At these magnitudes (Q6.25, values up to ~2^21) that carry fires
    // often enough to be a real, not theoretical, off-by-a-few-ULP bug --
    // found via xres_bank[0] mismatching pack_decoder_block.py's xres1 by
    // exactly +-1..3 in scattered lanes even though gout_bank[0] (O's own
    // GEMV output) and the pre-add xres_bank[0] both verified bit-exact
    // individually.
    integer rap;
    reg [P*32-1:0] res_add_word;
    always @* begin
        res_add_word = {(P*32){1'b0}};
        for (rap = 0; rap < P; rap = rap + 1)
            res_add_word[rap*32 +: 32] = $signed(xres_bank[ri_d][rap*32 +: 32]) +
                                          $signed(gout_bank[ri_d][rap*32 +: 32]);
    end

    // same flat-vector-add bug as res_add_word, in G_DRAIN's own FC1/FC2
    // bias add (`deq_word + bias_..._bank[...]`, the original code) -- per
    // lane, not one 256-bit add. Computed INLINE in G_DRAIN itself (below),
    // not as a separate always@* here: gi2 is blocking-assigned in G_DRAIN
    // the same cycle it's consumed, and a combinational block sensitized to
    // a blocking-assigned variable from a DIFFERENT always block is a real
    // scheduling race in Verilog (tried first -- produced worse mismatches,
    // 288/288, than the flat-add bug it was meant to fix).
    integer biap;
    reg signed [31:0] bias_lane;

    always @(posedge clk) begin
        done <= 1'b0;
        rope_start <= 1'b0;
        sk_wstart <= 1'b0; sk_wvalid <= 1'b0; sk_rstart <= 1'b0;
        xk_rstart <= 1'b0;
        at_start <= 1'b0; at_qvalid <= 1'b0; at_kvvalid <= 1'b0;
        ln_start <= 1'b0; ln_vin <= 1'b0;
        gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0;
        su_vin <= 1'b0;

        if (rst) begin
            st <= S_IDLE;
        end else begin
            case (st)
                S_IDLE: if (go) begin
                    ri_d <= 0; st <= S_LN1SET;
                end

                // ---- LN1 ----
                S_LN1SET: begin l_gbase<=2'd0; l_ret<=S_QSET; ri_d<=0; st<=L_START; end
                L_START: begin ln_start <= 1'b1; ri_d <= 0; st <= L_FEED; end
                L_FEED: begin
                    ln_x <= xres_bank[ri_d]; ln_g <= gamma_bank[l_gbase][ri_d];
                    ln_vin <= 1'b1;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin ri_d <= 0; st <= L_WAIT; end
                end
                L_WAIT: begin
                    if (ln_yvalid) begin
                        ln_out_bank[ri_d] <= ln_out_word;
                        if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    end
                    if (ln_done) st <= l_ret;
                end

                // ---- Q/K/V GEMVs ----
                S_QSET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_q;
                    g_src<=3'd0; g_dst<=3'd0; g_frac<=gf_q; g_actshift<=ACT_Q;
                    g_bias_en<=1'b0; g_ret<=S_KSET;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_KSET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_k;
                    g_src<=3'd0; g_dst<=3'd1; g_frac<=gf_k; g_actshift<=ACT_K;
                    g_bias_en<=1'b0; g_ret<=S_VSET;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_VSET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_v;
                    g_src<=3'd0; g_dst<=3'd2; g_frac<=gf_v; g_actshift<=ACT_V;
                    g_bias_en<=1'b0; g_ret<=S_ROPE_Q0;
                    // CRITICAL: hh is never otherwise reset before this, its
                    // very first use anywhere in the per-layer flow. On the
                    // first pass hh is X; Verilog's `if(X)` takes the ELSE
                    // branch, so S_ROPE_Q1 would silently treat hh==X as
                    // "last head", write q_rope_bank[X*HR_ATTN+...] (a no-op,
                    // X-indexed) exactly ONCE, and jump straight to
                    // S_ROPE_K0 -- Q-RoPE would never actually run for any
                    // of the 8 real heads, and q_rope_bank would stay
                    // garbage. Found by auditing every hh/ri_d/ri_f/ri_g use
                    // after two real reset-ordering bugs already turned up
                    // this exact class of mistake -- not caught by
                    // simulation alone, since an X-taking-the-else-branch
                    // doesn't hang, it just silently computes nothing.
                    hh<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                // ---- shared GEMV dispatch ----
                // rewind gemv_banked_resident_vec.sv's activation
                // write-pointer before EVERY GEMV call's own feed --
                // found via a real bug (FC2, the 8th and last call,
                // reading entirely-X activation data: xptr was never
                // rewound across the prior 7 calls' own 252 x_we
                // pulses, so FC2's 144 beats landed at xmem[252..395],
                // overflowing xmem's 288-row depth, while the MAC read
                // xmem[0..143] -- stale data from four earlier,
                // unrelated calls, not FC2's own activations at all).
                G_XRESET: begin gv_xrst <= 1'b1; st <= G_XFEED; end
                G_XFEED: begin
                    gv_xdata <= act_word;
                    gv_xwe   <= 1'b1;
                    if (ri_g != (gv_k >> $clog2(P)) - 1) ri_g <= ri_g + 1'b1;
                    else begin ri_g <= 0; st <= G_XSTART; end
                end
                // one-cycle gap between the LAST x_we beat and start, matching
                // the established, proven gemv_banked_resident_vec.sv protocol
                // exactly (fabric/stage3/run_resident_banked_vec.py's own
                // testbench: x_we for the last beat, @(posedge clk), THEN
                // x_we=0, THEN start=1 on the NEXT edge -- never the same
                // cycle). Found while root-causing FC2's own GEMV call (the
                // only one of 8 with K=1152, not K=288 like the rest)
                // producing entirely X output despite valid weights/
                // activations confirmed on both sides of the call.
                G_XSTART: begin gv_start <= 1'b1; st <= G_WAIT; end
                G_WAIT: if (gv_done) begin gi <= 0; st <= G_DRAIN; end
                G_DRAIN: begin
                    if (gi >= 2) begin
                        case (g_dst)
                            // q/k/v: SPLIT the P=8 dequant row into 2 ATTN_P=4 rows
                            3'd0: begin
                                q_bank[2*(gi-2)]   <= deq_word[ATTN_P*32-1:0];
                                q_bank[2*(gi-2)+1] <= deq_word[P*32-1:ATTN_P*32];
                            end
                            3'd1: begin
                                k_bank[2*(gi-2)]   <= deq_word[ATTN_P*32-1:0];
                                k_bank[2*(gi-2)+1] <= deq_word[P*32-1:ATTN_P*32];
                            end
                            3'd2: begin
                                v_bank[2*(gi-2)]   <= deq_word[ATTN_P*32-1:0];
                                v_bank[2*(gi-2)+1] <= deq_word[P*32-1:ATTN_P*32];
                            end
                            default: begin   // 3: gout_bank (O/Oc/FC1/FC2 -- optional bias)
                                gi2 = gi - 2;   // plain-reg copy: part-selects need a
                                                 // vector, not the bare expression (gi-2)
                                if (g_bias_en) begin
                                    for (biap = 0; biap < P; biap = biap + 1) begin
                                        bias_lane = g_bias_sel
                                            ? $signed(bias_fc2_bank[gi2[$clog2(ROWS_D)-1:0]][biap*32 +: 32])
                                            : $signed(bias_fc1_bank[gi2][biap*32 +: 32]);
                                        gout_bank[gi2][biap*32 +: 32] <=
                                            $signed(deq_word[biap*32 +: 32]) + bias_lane;
                                    end
                                end else
                                    gout_bank[gi2] <= deq_word;
                            end
                        endcase
                    end
                    if (gi == (gv_m + P - 1) / P + 1) st <= g_ret;
                    else gi <= gi + 1'b1;
                end

                // ---- RoPE: per head, Q then K (V never RoPE'd). Gather is
                // combinational (gather_q/gather_k above); one head/iter. ----
                S_ROPE_Q0: begin
                    rope_head_in <= gather_q; rope_pos <= step[$clog2(ROPE_TMAX)-1:0];
                    rope_start <= 1'b1; st <= S_ROPE_Q1;
                end
                S_ROPE_Q1: if (rope_done) begin
                    for (sc_i = 0; sc_i < HR_ATTN; sc_i = sc_i + 1)
                        q_rope_bank[hh*HR_ATTN + sc_i] <= rope_head_out[sc_i*ATTN_P*32 +: ATTN_P*32];
                    if (hh != NHEAD-1) begin hh <= hh + 1'b1; st <= S_ROPE_Q0; end
                    else begin hh <= 0; st <= S_ROPE_K0; end
                end
                S_ROPE_K0: begin
                    rope_head_in <= gather_k; rope_pos <= step[$clog2(ROPE_TMAX)-1:0];
                    rope_start <= 1'b1; st <= S_ROPE_K1;
                end
                S_ROPE_K1: if (rope_done) begin
                    for (sc_i = 0; sc_i < HR_ATTN; sc_i = sc_i + 1)
                        k_rope_bank[hh*HR_ATTN + sc_i] <= rope_head_out[sc_i*ATTN_P*32 +: ATTN_P*32];
                    if (hh != NHEAD-1) begin hh <= hh + 1'b1; st <= S_ROPE_K0; end
                    else begin hh <= 0; wi <= 0; st <= S_KVW_K0; end
                end

                // ---- self kv_bank write: K (RoPE'd), then V (raw), per head --
                S_KVW_K0: begin
                    sk_wlayer<=blk; sk_wkv<=1'b0; sk_whead<=hh[$clog2(NHEAD)-1:0]; sk_wpos<=step[8:0];
                    sk_wstart<=1'b1; wi<=0; st<=S_KVW_K1;
                end
                S_KVW_K1: begin
                    sk_wdata <= k_rope_bank[hh*HR_ATTN + wi];
                    sk_wvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else if (sk_wdone) begin wi<=0; st<=S_KVW_V0; end
                end
                S_KVW_V0: begin
                    sk_wlayer<=blk; sk_wkv<=1'b1; sk_whead<=hh[$clog2(NHEAD)-1:0]; sk_wpos<=step[8:0];
                    sk_wstart<=1'b1; wi<=0; st<=S_KVW_V1;
                end
                S_KVW_V1: begin
                    sk_wdata <= v_bank[hh*HR_ATTN + wi];
                    sk_wvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else if (sk_wdone) begin
                        if (hh != NHEAD-1) begin hh<=hh+1'b1; wi<=0; st<=S_KVW_K0; end
                        else begin hh<=0; st<=S_ASELF_START; end
                    end
                end

                // ---- self-attn (causal, tcount=step+1) ----
                S_ASELF_START: begin
                    at_start <= 1'b1; at_tcount <= step + 9'd1;  // tcount must be
                    // valid THE SAME cycle start pulses -- vec_attn_w.sv samples it
                    // combinationally in W_IDLE's `if(start) T<=tcount;`, one cycle
                    // too late = a stale/X T, found via a real hang (the FSM never
                    // left W_K since scnt never matched an X-valued T-1).
                    wi <= 0; st <= S_ASELF_Q;
                end
                S_ASELF_Q: begin
                    at_qdata <= q_rope_bank[hh*HR_ATTN + wi];
                    at_qvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else begin wi<=0; sk_rlayer<=blk; sk_rkv<=1'b0; sk_rhead<=hh[$clog2(NHEAD)-1:0];
                               sk_rtcount<=step+9'd1; sk_rstart<=1'b1; st<=S_ASELF_K0; end
                end
                S_ASELF_K0: begin
                    at_kvvalid <= sk_rvalid; at_kvdata <= sk_rdata;
                    if (at_kdone) begin
                        sk_rlayer<=blk; sk_rkv<=1'b1; sk_rhead<=hh[$clog2(NHEAD)-1:0];
                        sk_rtcount<=step+9'd1; sk_rstart<=1'b1; st<=S_ASELF_V0;
                    end
                end
                S_ASELF_V0: begin
                    at_kvvalid <= sk_rvalid; at_kvdata <= sk_rdata;
                    // ctx_bank's row-0 write must happen HERE too, not only
                    // in S_ASELF_DRAIN -- at_ctxvalid's FIRST pulse (idx=0)
                    // is what triggers this very state transition, so by the
                    // time S_ASELF_DRAIN's own `if(at_ctxvalid)` runs, idx
                    // has already moved to 1 and row 0's write is dropped,
                    // silently leaving ctx_bank[hh*HR_ATTN+0] at its
                    // undriven-reg X forever. Found via ctx_bank[0]/[1]
                    // (O's own g_src=1 combine read) showing X while
                    // ctx_bank[1..8] were all valid.
                    if (at_ctxvalid) ctx_bank[hh*HR_ATTN + at_ctxidx[$clog2(HR_ATTN)-1:0]] <= at_ctxdata;
                    if (at_ctxvalid || at_done) st <= S_ASELF_DRAIN;
                end
                S_ASELF_DRAIN: begin
                    at_kvvalid <= sk_rvalid; at_kvdata <= sk_rdata;
                    if (at_ctxvalid) ctx_bank[hh*HR_ATTN + at_ctxidx[$clog2(HR_ATTN)-1:0]] <= at_ctxdata;
                    if (at_done) begin
                        if (hh != NHEAD-1) begin hh<=hh+1'b1; st<=S_ASELF_START; end
                        else begin hh<=0; st<=S_OSET; end
                    end
                end

                // ---- O ----
                S_OSET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_o;
                    g_src<=3'd1; g_dst<=3'd3; g_frac<=gf_o; g_actshift<=ACT_O;
                    g_bias_en<=1'b0; g_ret<=S_RES1;
                    // ri_d must be reset here: S_RES1 reuses it, but it was
                    // last left at ROWS_D-1 by LN1's own L_WAIT drain loop --
                    // without this, S_RES1's first cycle sees ri_d==ROWS_D-1
                    // immediately and only ever processes ONE row (a real,
                    // silent correctness bug, not a hang -- found by auditing
                    // every ri_d use after the ri_f/ri_g uninitialized-reset
                    // bug turned up the same class of mistake).
                    ri_d<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_RES1: begin
                    xres_bank[ri_d] <= res_add_word;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin ri_d<=0; st<=S_LN2SET; end
                end

                // ---- LN2 ----
                S_LN2SET: begin l_gbase<=2'd1; l_ret<=S_CQSET; ri_d<=0; st<=L_START; end

                // ---- cross Q GEMV (no RoPE -- POST_SCALE applied after) ----
                S_CQSET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_cq;
                    g_src<=3'd0; g_dst<=3'd0; g_frac<=gf_cq; g_actshift<=ACT_CQ;
                    g_bias_en<=1'b0; g_ret<=S_CQSCALE;
                    // same fix as S_OSET/S_COSET/S_FC2SET -- ri_d was left at
                    // ROWS_D-1=35 by LN2's own L_WAIT drain loop; without this,
                    // S_CQSCALE's 72-row loop would start at 35 and only cover
                    // rows 35..71, silently skipping the POST_SCALE_Q16
                    // correction for roughly half of q_bank (heads 0-3ish).
                    ri_d<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                // scale q_bank's cross-Q by POST_SCALE_Q16 in place (matches
                // rope_apply_vec.sv's own pass-through-lane formula:
                // rsh_round(v*POST_SCALE_Q16,16) -- but plain floor here to
                // stay consistent with this gate's own no-rounding act-quant
                // convention -- a real, documented simplification, checked
                // against the Python reference which uses the SAME formula).
                S_CQSCALE: begin
                    ri_d <= (ri_d == ROWS_AD-1) ? 0 : ri_d + 1'b1;
                    for (bp2 = 0; bp2 < ATTN_P; bp2 = bp2 + 1) begin
                        scale_prod = $signed(q_bank[ri_d][bp2*32 +: 32]) * $signed({32'd0, POST_SCALE_Q16});
                        q_bank[ri_d][bp2*32 +: 32] <= rsh_round64(scale_prod, 16);
                    end
                    if (ri_d == ROWS_AD-1) st <= S_ACROSS_START;
                end

                // ---- cross-attn (static full-attend, tcount=T2) ----
                S_ACROSS_START: begin
                    at_start <= 1'b1; at_tcount <= T2[8:0];  // same fix as S_ASELF_START
                    wi <= 0; st <= S_ACROSS_Q;
                end
                S_ACROSS_Q: begin
                    at_qdata <= q_bank[hh*HR_ATTN + wi];
                    at_qvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else begin wi<=0; xk_rlayer<=blk; xk_rkv<=1'b0; xk_rhead<=hh[$clog2(NHEAD)-1:0];
                               xk_rtcount<=T2[8:0]; xk_rstart<=1'b1; st<=S_ACROSS_K0; end
                end
                S_ACROSS_K0: begin
                    at_kvvalid <= xk_rvalid; at_kvdata <= xk_rdata;
                    if (at_kdone) begin
                        xk_rlayer<=blk; xk_rkv<=1'b1; xk_rhead<=hh[$clog2(NHEAD)-1:0];
                        xk_rtcount<=T2[8:0]; xk_rstart<=1'b1; st<=S_ACROSS_V0;
                    end
                end
                S_ACROSS_V0: begin
                    at_kvvalid <= xk_rvalid; at_kvdata <= xk_rdata;
                    // same dropped-row-0 fix as S_ASELF_V0 -- see its comment.
                    if (at_ctxvalid) ctx_bank[hh*HR_ATTN + at_ctxidx[$clog2(HR_ATTN)-1:0]] <= at_ctxdata;
                    if (at_ctxvalid || at_done) st <= S_ACROSS_DRAIN;
                end
                S_ACROSS_DRAIN: begin
                    at_kvvalid <= xk_rvalid; at_kvdata <= xk_rdata;
                    if (at_ctxvalid) ctx_bank[hh*HR_ATTN + at_ctxidx[$clog2(HR_ATTN)-1:0]] <= at_ctxdata;
                    if (at_done) begin
                        if (hh != NHEAD-1) begin hh<=hh+1'b1; st<=S_ACROSS_START; end
                        else begin hh<=0; st<=S_COSET; end
                    end
                end

                S_COSET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_co;
                    g_src<=3'd1; g_dst<=3'd3; g_frac<=gf_co; g_actshift<=ACT_OC;
                    g_bias_en<=1'b0; g_ret<=S_RES2;
                    // same fix as S_OSET -- ri_d was left at ROWS_AD-1 (71) by
                    // S_CQSCALE's own loop, which would make S_RES2 index
                    // xres_bank/gout_bank OUT OF BOUNDS (both declared only
                    // [0:ROWS_D-1]=[0:35]).
                    ri_d<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_RES2: begin
                    xres_bank[ri_d] <= res_add_word;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin ri_d<=0; st<=S_LN3SET; end
                end

                // ---- LN3 ----
                S_LN3SET: begin l_gbase<=2'd2; l_ret<=S_FC1SET; ri_d<=0; st<=L_START; end

                // ---- FC1: D -> DFFN2, +bias ----
                S_FC1SET: begin
                    gv_m<=DFFN2[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_fc1;
                    g_src<=3'd0; g_dst<=3'd3; g_frac<=gf_fc1; g_actshift<=ACT_FC1;
                    g_bias_en<=1'b1; g_bias_sel<=1'b0; g_ret<=S_SILU_INIT;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end

                // reset ri_f/ri_g to 0 before S_SILU0 reuses them (ri_g was
                // just left at G_DRAIN's own final readback-row value, ri_f
                // was never set at all on the FIRST entry -- both real bugs,
                // found via a real hang where ri_f read back X forever).
                S_SILU_INIT: begin ri_f <= 0; ri_g <= 0; st <= S_SILU0; end

                // ---- SwiGLU: gate=silu(h1[FFN:]), combined=h1[:FFN]*gate ----
                // ONE combined feed+drain state (not two sequential phases):
                // vec_silu.sv is a plain 3-cycle-latency streaming pipeline
                // with no start/done handshake (unlike LN/GEMV), so most
                // su_vout pulses land WHILE ri_f is still feeding su_vin, not
                // after -- feeding all 144 beats first then only THEN
                // watching for su_vout (the original two-state split) misses
                // every one of those, found via a real hang: ri_g (output
                // side) is the real exit condition, tracked independently of
                // ri_f (input side), same idiom as G_DRAIN/G_XFEED running
                // concurrently.
                S_SILU0: begin
                    su_x <= su_x_word;
                    su_vin <= (ri_f != ROWS_FFN);
                    if (ri_f != ROWS_FFN) ri_f <= ri_f + 1'b1;
                    if (su_vout) begin
                        for (bp2 = 0; bp2 < P; bp2 = bp2 + 1) begin
                            gate_prod = $signed(gout_bank[ri_g[$clog2(ROWS_FFN)-1:0]][bp2*32 +: 32]) *
                                        $signed({{16{su_y[bp2*16+15]}}, su_y[bp2*16 +: 16]});
                            combined_bank[ri_g[$clog2(ROWS_FFN)-1:0]][bp2*32 +: 32] <= rsh_round64(gate_prod, 12);
                        end
                        if (ri_g == ROWS_FFN-1) begin ri_f<=0; ri_g<=0; st<=S_FC2SET; end
                        else ri_g <= ri_g + 1'b1;
                    end
                end

                // ---- FC2: FFN -> D, +bias ----
                S_FC2SET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=FFN[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_fc2;
                    g_src<=3'd2; g_dst<=3'd3; g_frac<=gf_fc2; g_actshift<=ACT_FC2;
                    g_bias_en<=1'b1; g_bias_sel<=1'b1; g_ret<=S_RES3;
                    // same fix as S_OSET -- ri_d was left at ROWS_D-1 by LN3's
                    // own L_WAIT drain loop.
                    ri_d<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_RES3: begin
                    xres_bank[ri_d] <= res_add_word;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin ri_d<=0; st<=S_DONE; end
                end

                S_DONE: begin done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
