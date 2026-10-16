// -----------------------------------------------------------------------------
// encoder_block_seq -- the real, sized, MULTI-LAYER top-level FSM for ASR's
// encoder block (Stage 2 of gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md). Chains
// real, already-gated sub-modules (kv_bank.sv, vec_attn_w.sv,
// layernorm_vec_gendiv.sv, gemv_banked_resident_vec.sv WBW=8,
// rope_apply_vec.sv, vec_gelu.sv) into one full encoder-layer forward pass
// over ALL T2 real positions at once (the encoder is NOT autoregressive --
// no `step` port, no causal KV growth; every position is known up front):
//   for pos=0..T2-1: LN1 -> Q/K/V -> RoPE(Q,K) -> self KV-write
//   for pos=0..T2-1: self-attn(bidirectional, tcount=T2 always) -> O ->
//     RES1 -> LN2 -> fc1(+bias) -> GELU -> fc2(+bias) -> RES2
// (two phases, not interleaved: self-attention's own "static full-attend"
// access pattern -- proven by pack_encoder_self_attn.py's own isolated
// gate -- needs EVERY position's own K/V written before ANY position's own
// query can read them, so all T2 positions' Q/K/V/RoPE/KV-write happen
// first, then all T2 positions' attend+MLP happen second.)
//
// Modeled directly on decoder_block_seq.sv's own idiom (shared GEMV
// dispatcher G_XFEED/G_WAIT/G_DRAIN/G_XSTART/G_XRESET, shared LayerNorm
// dispatcher L_FEED/L_WAIT/L_START, the same P=8<->ATTN_P=4 per-head bank
// layout, the same wb_*/gf_* runtime-port quantization scheme) -- built
// AFTER decoder_block_seq.sv's own step-loop and layer-loop gates, so the
// lessons from both are already folded in here from the start rather than
// rediscovered: wb_*/gf_* are runtime ports (not WB_*/GF_* parameters) from
// day one, u_self_kv is sized NLAYER and wired to `blk` from day one (the
// bug decoder_block_seq.sv's own u_cross_kv had, avoided here by
// construction, not by a later fix).
//
// Real structural differences from decoder_block_seq.sv, not oversights:
// - No cross-attention at all (the encoder attends only to itself) --
//   no CQ/CO GEMVs, no second kv_bank instance, no xkv_* ports.
// - Only 2 LayerNorms per layer (ln1, ln2), not 3 -- gamma_bank is [0:1].
// - Self-attention is BIDIRECTIONAL, not causal: tcount=T2 for every query,
//   always (no step-driven growth) -- kv_bank's read port called T2 times
//   per position with the SAME fixed tcount, on data written once per
//   layer and never rewritten (pack_encoder_self_attn.py's own proof this
//   access pattern needs no new storage RTL).
// - Plain MLP (fc1 -> GELU -> fc2), not SwiGLU -- no gate*value multiply,
//   vec_gelu.sv (same P-lane-parallel shape as vec_silu.sv, activation
//   swapped) applied directly to fc1's own full FFN-wide output.
// - xres_bank and q_rope_bank are T2-indexed (persist across the phase-A/
//   phase-B boundary within one `go`); every other scratch bank
//   (ln_out_bank, gout_bank, gelu_bank, q_bank/k_bank/v_bank/k_rope_bank,
//   ctx_bank) stays single-position, reused sequentially across positions
//   exactly like decoder_block_seq.sv's own banks are reused across GEMV
//   call sites.
//
// GEMV QUANTIZATION SCHEME -- identical convention to decoder_block_seq.sv
// (see that file's own header for the full rationale): per-matrix
// single-scale INT8 weights, per-call single-shift INT8 activations, one
// combined dequant shift. ACT_* stay compile-time parameters (one value
// per call site, profiled across every real layer AND every position);
// wb_*/gf_* are runtime ports (weight-image offset and dequant shift both
// genuinely vary per layer -- WSHIFT is chosen per layer's own weight
// distribution).
//
// STATUS: gated bit-exact against pack_encoder_block.py's own reference,
// all 6 real layers x all T2=6 real positions, second attempt (see that
// gate's own verdict line). One real, new-to-this-file bug found and
// fixed on the way there: `rope_pos <= pos[$clog2(ROPE_TMAX)-1:0]`
// (mirroring decoder_block_seq.sv's own `rope_pos <= step[...]`) reads
// bits beyond `pos`'s own declared width ($clog2(T2)=3 bits here vs the
// 7 ROPE_TMAX needs) and returns X for every out-of-range bit --
// decoder_block_seq.sv never hit this because `step` is a 9-bit port,
// wide enough on its own. Fixed by assigning `rope_pos <= pos;` directly
// (Verilog zero-extends correctly on assignment; only the explicit
// bit-select was wrong) -- found via q_rope_bank ending up entirely X
// while q_bank (the pre-RoPE GEMV output) was valid, traced back to
// rope_pos itself reading X at rope_apply_vec.sv's own start.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module encoder_block_seq #(
    parameter integer P         = 8,
    parameter integer D         = 288,
    parameter integer FFN       = 1152,
    parameter integer NHEAD     = 8,
    parameter integer HEAD_DIM  = 36,
    parameter integer ATTN_P    = 4,
    parameter integer T2        = 6,        // real positions/layer (this gate's real T2)
    parameter integer ROT_PAIRS = 16,
    parameter integer ROPE_TMAX = 128,
    parameter integer POST_SCALE_Q16 = 75674,
    // ACT_* stay compile-time parameters -- one fixed ACT_RSHIFT per call
    // site, profiled to safely cover every layer AND every position (see
    // decoder_block_seq.sv's own header for why this split -- ACT_* fixed,
    // GF_*/WB_* runtime -- holds).
    parameter signed [6:0] ACT_Q=17, ACT_K=17, ACT_V=17, ACT_O=15,
                           ACT_FC1=17, ACT_FC2=11
) (
    input  wire clk,
    input  wire rst,

    input  wire        go,             // pulse: run ONE encoder layer, ALL T2 positions
    input  wire [3:0]  blk,            // self kv_bank's layer index
    output reg         done,

    // real word offsets into the resident weight image, runtime ports (see
    // decoder_block_seq.sv's own header for why: each layer needs its own
    // offset for the "same" call site). Only 6 call sites (no cq/co --
    // no cross-attention).
    input  wire [19:0] wb_q, wb_k, wb_v, wb_o, wb_fc1, wb_fc2,
    // dequant shift per call site, runtime ports (WSHIFT, baked into
    // g_frac, is genuinely per-layer -- see decoder_block_seq.sv's header).
    input  wire signed [7:0] gf_q, gf_k, gf_v, gf_o, gf_fc1, gf_fc2,

    // residual stream in/out -- P*32-bit packed rows, T2 positions x D/P
    // rows each, Q6.25. xres_wpos selects the position (NOT present on
    // decoder_block_seq.sv -- the decoder processes one token/go, the
    // encoder processes T2 tokens/go).
    input  wire                     xres_wr,
    input  wire [$clog2(T2)-1:0]    xres_wpos,
    input  wire [$clog2(D/P)-1:0]   xres_waddr,
    input  wire [P*32-1:0]          xres_wdata,
    input  wire [$clog2(T2)-1:0]    xres_rpos_dbg,
    output wire [P*32-1:0]          xres_rdata_dbg,

    // ---- GEMV weight load (passthrough to gemv_banked_resident_vec) --------
    input  wire        gv_ld_rst,
    input  wire        gv_ld_we,
    input  wire [31:0] gv_ld_data,

    // ---- LN gamma load: sel picks LN1/LN2 (0/1) -----------------------------
    input  wire         gam_we,
    input  wire         gam_sel,
    input  wire [$clog2(D/P)-1:0] gam_waddr,
    input  wire [P*32-1:0]        gam_wdata,

    // ---- bias load: sel picks fc1/fc2 (0/1) ---------------------------------
    input  wire         bias_we,
    input  wire         bias_sel,
    input  wire [$clog2(FFN/P)-1:0] bias_waddr,
    input  wire [P*32-1:0]          bias_wdata
);
    localparam integer ROWS_D     = D / P;              // 36
    localparam integer ROWS_FFN   = FFN / P;             // 144
    localparam integer HR_ATTN    = HEAD_DIM / ATTN_P;   // 9
    localparam integer ROWS_AD    = NHEAD * HR_ATTN;     // 72 (D at ATTN_P width)

    // =========================================================================
    // ---- residual (T2-indexed, persists phase A -> phase B) + LN output/
    // generic GEMV dest/GELU banks (single-position scratch, P=8-wide) -------
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank   [0:T2-1][0:ROWS_D-1];  // Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] ln_out_bank [0:ROWS_D-1];          // Q.22
    (* ram_style = "block" *) reg [P*32-1:0] gout_bank   [0:ROWS_FFN-1];        // generic GEMV dest
    (* ram_style = "block" *) reg [P*32-1:0] gelu_bank   [0:ROWS_FFN-1];       // Q4.12, post-GELU

    assign xres_rdata_dbg = xres_bank[xres_rpos_dbg][xres_waddr];
    always @(posedge clk) if (xres_wr) xres_bank[xres_wpos][xres_waddr] <= xres_wdata;

    // ---- gamma tables (2 x D, Q4.20) ----
    (* ram_style = "block" *) reg [P*32-1:0] gamma_bank [0:1][0:ROWS_D-1];
    always @(posedge clk) if (gam_we) gamma_bank[gam_sel][gam_waddr] <= gam_wdata;

    // ---- bias tables (fc1: FFN, Q.12; fc2: D, Q.25) ----
    (* ram_style = "block" *) reg [P*32-1:0] bias_fc1_bank [0:ROWS_FFN-1];
    (* ram_style = "block" *) reg [P*32-1:0] bias_fc2_bank [0:ROWS_D-1];
    always @(posedge clk) if (bias_we) begin
        if (!bias_sel) bias_fc1_bank[bias_waddr] <= bias_wdata;
        else           bias_fc2_bank[bias_waddr[$clog2(ROWS_D)-1:0]] <= bias_wdata;
    end

    // ---- per-head banks (ATTN_P=4-wide, 72 rows), single-position scratch
    // except q_rope_bank (T2-indexed: RoPE'd in phase A, consumed in
    // phase B, after phase A has moved on to later positions). -------------
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] q_bank      [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] k_bank      [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] v_bank      [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] q_rope_bank [0:T2-1][0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] k_rope_bank [0:ROWS_AD-1];  // Q.16
    (* ram_style = "block" *) reg [ATTN_P*32-1:0] ctx_bank    [0:ROWS_AD-1];  // Q.25

    // =========================================================================
    // ---- rope_apply_vec (real, unmodified) ----------------------------------
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

    integer gi_r;
    integer sc_i;
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
    // ---- self-attn kv_bank (real, unmodified) -- sized/wired for real
    // multi-layer use FROM THE START (see this file's own header). ----------
    reg         sk_wstart, sk_wvalid, sk_rstart;
    reg  [3:0]  sk_wlayer;  reg sk_wkv;  reg [$clog2(NHEAD)-1:0] sk_whead;  reg [8:0] sk_wpos;
    reg  [ATTN_P*32-1:0] sk_wdata;
    wire        sk_wdone;
    reg  [3:0]  sk_rlayer;  reg sk_rkv;  reg [$clog2(NHEAD)-1:0] sk_rhead;  reg [8:0] sk_rtcount;
    wire        sk_rvalid, sk_rdone;
    wire [HEAD_DIM*32-1:0] sk_rdata;
    kv_bank #(.P(ATTN_P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(16), .TMAX(T2),
              .KBITS(8), .MEM_PRIMITIVE("block")) u_self_kv (
        .clk(clk), .rst(rst),
        .wq_start(sk_wstart), .wq_layer(blk), .wq_kv(sk_wkv), .wq_head(sk_whead),
        .wq_pos(sk_wpos), .wq_valid(sk_wvalid), .wq_data(sk_wdata), .wq_done(sk_wdone),
        .rd_start(sk_rstart), .rd_layer(sk_rlayer), .rd_kv(sk_rkv), .rd_head(sk_rhead),
        .rd_tcount(sk_rtcount), .rd_valid(sk_rvalid), .rd_data(sk_rdata), .rd_done(sk_rdone),
        .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head({$clog2(NHEAD){1'b0}}),
        .rd2_tcount(9'd0), .rd2_valid(), .rd2_data(), .rd2_done()
    );

    // =========================================================================
    // ---- vec_attn_w (real, unmodified) -- bidirectional, tcount=T2 always --
    reg                    at_start;
    reg  [8:0]             at_tcount;
    reg                    at_qvalid;
    reg  [ATTN_P*32-1:0]   at_qdata;
    reg                    at_kvvalid;
    reg  [HEAD_DIM*32-1:0] at_kvdata;
    wire                   at_kdone, at_ctxvalid, at_done;
    wire [6:0]             at_ctxidx;
    wire [ATTN_P*32-1:0]   at_ctxdata;
    vec_attn_w #(.P(ATTN_P), .HEAD_DIM(HEAD_DIM), .TMAX(T2)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .q_valid(at_qvalid), .q_data(at_qdata),
        .kv_valid(at_kvvalid), .kv_data(at_kvdata),
        .k_done(at_kdone), .ctx_valid(at_ctxvalid), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata),
        .done(at_done)
    );

    // =========================================================================
    // ---- layernorm_vec_gendiv (real, unmodified) ----------------------------
    reg               ln_start, ln_vin;
    reg [P*32-1:0]    ln_x, ln_g;
    wire              ln_yvalid, ln_done;
    wire [P*64-1:0]   ln_yout;
    layernorm_vec_gendiv #(.P(P), .D(D)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .valid_in(ln_vin),
        .x_in(ln_x), .gamma_in(ln_g), .y_valid(ln_yvalid), .y_out(ln_yout), .done(ln_done)
    );
    // ln_yout packs P lanes at 64 bits each -- see decoder_block_seq.sv's own
    // u_ln comment for why the low-256-bits-direct read is wrong.
    integer lnp;
    reg [P*32-1:0] ln_out_word;
    always @* begin
        ln_out_word = {(P*32){1'b0}};
        for (lnp = 0; lnp < P; lnp = lnp + 1)
            ln_out_word[lnp*32 +: 32] = ln_yout[lnp*64 +: 32];
    end

    // =========================================================================
    // ---- gemv_banked_resident_vec (real, WBW=8) -----------------------------
    // GEMV_MMAX/KMAX = FFN (1152, the largest single-call M or K: fc1 has
    // M=FFN, fc2 has K=FFN; every other call is D=288) -- NOT DFFN2 (no
    // SwiGLU doubling on the encoder side).
    localparam integer GEMV_MMAX = FFN;
    localparam integer GEMV_KMAX = FFN;
    reg                          gv_start;
    reg                          gv_xrst;
    wire                         gv_done;
    reg  [$clog2(GEMV_MMAX+1)-1:0] gv_m;
    reg  [$clog2(GEMV_KMAX+1)-1:0] gv_k;
    reg  [19:0]                  gv_wbase;
    reg                          gv_xwe;
    reg  [P*8-1:0]               gv_xdata;
    wire [$clog2((GEMV_MMAX+127)/128+1)-1:0] gv_gdone;
    wire [$clog2(GEMV_MMAX/P)-1:0] gv_rdaddr = gi;   // combinational -- see decoder_block_seq.sv
    wire [P*32-1:0]              gv_yout;
    gemv_banked_resident_vec #(.LANES(128), .WBW(8), .P(P), .MMAX(GEMV_MMAX), .KMAX(GEMV_KMAX),
                                .WWORDS(1<<20), .RLAT(2), .K2(0), .MEM_PRIMITIVE("block")) u_gemv (
        .clk(clk), .rst(rst), .m_count(gv_m), .k_count(gv_k), .w_base(gv_wbase),
        .ld_rst(gv_ld_rst || gv_xrst), .w_we(gv_ld_we), .w_data(gv_ld_data),
        .x_we(gv_xwe), .x_data(gv_xdata),
        .start(gv_start), .done(gv_done), .gdone(gv_gdone),
        .rd_addr(gv_rdaddr), .y_out(gv_yout),
        .emb_sel(1'b0), .emb_addr({$clog2(1<<20){1'b0}}), .emb_pair(),
        .wbdiag_addr({$clog2(1<<20){1'b0}}), .wbdiag_pair()
    );

    function automatic signed [31:0] gdequant;
        input signed [31:0] raw;
        input signed [7:0]  frac;
        begin
            gdequant = (frac >= 0) ? (raw >>> frac) : (raw <<< (-frac));
        end
    endfunction
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
    // gate saturate: pack_encoder_block.py's own np.clip(h1,-32768,32767)
    // before gelu_q -- straight saturate of the raw (already-dequantized)
    // Q4.12 FC1 value into a 16-bit lane, matching decoder_block_seq.sv's
    // own sat16() (same lane-width class of issue, gelu_bank's own row is
    // P lanes of 32 bits, vec_gelu wants P lanes of 16).
    function automatic signed [15:0] sat16;
        input signed [31:0] x;
        begin
            if (x > 32'sd32767) sat16 = 16'sd32767;
            else if (x < -32'sd32768) sat16 = -16'sd32768;
            else sat16 = x[15:0];
        end
    endfunction

    // =========================================================================
    // ---- vec_gelu (real) -- same P-lane shape as decoder's vec_silu, no
    // gate multiply needed (plain MLP, not SwiGLU): GELU applied directly
    // to fc1's own full FFN-wide output. ---------------------------------
    reg                     ge_vin;
    reg signed [16*P-1:0]   ge_x;
    wire                    ge_vout;
    wire signed [16*P-1:0]  ge_y;
    integer gep;
    reg signed [16*P-1:0] ge_x_word;
    always @* begin
        ge_x_word = {(16*P){1'b0}};
        for (gep = 0; gep < P; gep = gep + 1)
            ge_x_word[gep*16 +: 16] = sat16($signed(gout_bank[ri_f][gep*32 +: 32]));
    end
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(ge_vin), .x(ge_x), .out_valid(ge_vout), .y(ge_y)
    );

    // =========================================================================
    // ---- state encoding -----------------------------------------------------
    localparam [5:0]
        S_IDLE=0,
        S_PHA_START=1,
        S_LN1SET=2,  L_FEED=3,  L_WAIT=4,
        S_QSET=5,    G_XFEED=6, G_WAIT=7, G_DRAIN=8,
        S_KSET=9,    S_VSET=10,
        S_ROPE_Q0=11, S_ROPE_Q1=12, S_ROPE_K0=13, S_ROPE_K1=14,
        S_KVW_K0=15, S_KVW_K1=16, S_KVW_V0=17, S_KVW_V1=18,
        S_PHA_NEXT=19,
        S_PHB_START=20,
        S_ASELF_Q=21, S_ASELF_K0=22, S_ASELF_V0=23, S_ASELF_DRAIN=24,
        S_OSET=25,
        S_RES1=26,
        S_LN2SET=27,
        S_FC1SET=28,
        S_GELU_INIT=29, S_GELU0=30,
        S_FC2SET=31,
        S_RES2=32,
        S_DONE=33,
        L_START=34, S_ASELF_START=35, G_XSTART=36, G_XRESET=37;
    reg [5:0] st;

    reg [5:0] l_ret, g_ret;
    reg       l_gbase;
    reg [2:0] g_src;                         // 0=ln_out, 1=ctx(combine), 2=gelu
    reg [2:0] g_dst;                         // 0..2: q/k/v, 3: gout (see G_DRAIN)
    reg signed [7:0] g_frac;
    reg signed [6:0] g_actshift;
    reg g_bias_en; reg g_bias_sel;

    reg [$clog2(NHEAD)-1:0] hh;
    reg [$clog2(HR_ATTN+1)-1:0] wi;
    reg [$clog2(T2)-1:0] pos;

    reg [$clog2(ROWS_AD)-1:0]    ri_d;
    reg [$clog2(ROWS_FFN)-1:0]  ri_g;
    reg [$clog2(ROWS_FFN)-1:0]  ri_f;
    reg [$clog2(GEMV_MMAX/P)-1:0] gi;
    reg [$clog2(GEMV_MMAX/P)-1:0] gi2;

    // ---- generic GEMV activation source mux (P=8-wide) ----------------------
    reg [P*32-1:0] g_src_row;
    always @* begin
        case (g_src)
            3'd0: g_src_row = ln_out_bank[ri_g[$clog2(ROWS_D)-1:0]];
            3'd1: g_src_row = {ctx_bank[2*ri_g[$clog2(ROWS_AD/2)-1:0]+1],
                                ctx_bank[2*ri_g[$clog2(ROWS_AD/2)-1:0]]};
            3'd2: g_src_row = gelu_bank[ri_g];
            default: g_src_row = {(P*32){1'b0}};
        endcase
    end

    integer bp;
    reg [P*8-1:0] act_word;
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

    // residual add: P independent 32-bit signed lane additions, not one flat
    // 256-bit vector `+` -- see decoder_block_seq.sv's own res_add_word
    // comment for why (a real carry-bleed bug, not theoretical).
    integer rap;
    reg [P*32-1:0] res_add_word;
    always @* begin
        res_add_word = {(P*32){1'b0}};
        for (rap = 0; rap < P; rap = rap + 1)
            res_add_word[rap*32 +: 32] = $signed(xres_bank[pos][ri_d][rap*32 +: 32]) +
                                          $signed(gout_bank[ri_d][rap*32 +: 32]);
    end

    integer biap;
    reg signed [31:0] bias_lane;

    always @(posedge clk) begin
        done <= 1'b0;
        rope_start <= 1'b0;
        sk_wstart <= 1'b0; sk_wvalid <= 1'b0; sk_rstart <= 1'b0;
        at_start <= 1'b0; at_qvalid <= 1'b0; at_kvvalid <= 1'b0;
        ln_start <= 1'b0; ln_vin <= 1'b0;
        gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0;
        ge_vin <= 1'b0;

        if (rst) begin
            st <= S_IDLE;
        end else begin
            case (st)
                S_IDLE: if (go) begin
                    pos <= 0; st <= S_PHA_START;
                end

                // ================= PHASE A: for pos=0..T2-1, write K/V =====
                S_PHA_START: begin ri_d <= 0; st <= S_LN1SET; end

                // ---- LN1 ----
                S_LN1SET: begin l_gbase<=1'b0; l_ret<=S_QSET; ri_d<=0; st<=L_START; end
                L_START: begin ln_start <= 1'b1; ri_d <= 0; st <= L_FEED; end
                L_FEED: begin
                    ln_x <= xres_bank[pos][ri_d]; ln_g <= gamma_bank[l_gbase][ri_d];
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
                    // hh reset before its own first use, same real bug class
                    // as decoder_block_seq.sv's own S_VSET comment.
                    hh<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                // ---- shared GEMV dispatch (identical to decoder_block_seq.sv) --
                G_XRESET: begin gv_xrst <= 1'b1; st <= G_XFEED; end
                G_XFEED: begin
                    gv_xdata <= act_word;
                    gv_xwe   <= 1'b1;
                    if (ri_g != (gv_k >> $clog2(P)) - 1) ri_g <= ri_g + 1'b1;
                    else begin ri_g <= 0; st <= G_XSTART; end
                end
                G_XSTART: begin gv_start <= 1'b1; st <= G_WAIT; end
                G_WAIT: if (gv_done) begin gi <= 0; st <= G_DRAIN; end
                G_DRAIN: begin
                    if (gi >= 2) begin
                        case (g_dst)
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
                            default: begin   // 3: gout_bank (O/FC1/FC2 -- optional bias)
                                gi2 = gi - 2;
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

                // ---- RoPE: per head, Q then K (V never RoPE'd) -- position =
                // `pos`, the CURRENT phase-A position, not a decode step. ----
                S_ROPE_Q0: begin
                    rope_head_in <= gather_q; rope_pos <= pos;   // direct assign -- Verilog zero-extends correctly on
                    // assignment; a bit-select pos[$clog2(ROPE_TMAX)-1:0] (the
                    // original code here) reads bits beyond pos's own declared
                    // width ($clog2(T2)=3 bits here, ROPE_TMAX needs 7) and returns
                    // X for every out-of-range bit -- found via q_rope_bank ending
                    // up entirely X while q_bank (pre-RoPE) was valid, traced to
                    // rope_pos itself reading X at rope_apply_vec.sv's own start.
                    rope_start <= 1'b1; st <= S_ROPE_Q1;
                end
                S_ROPE_Q1: if (rope_done) begin
                    for (sc_i = 0; sc_i < HR_ATTN; sc_i = sc_i + 1)
                        q_rope_bank[pos][hh*HR_ATTN + sc_i] <= rope_head_out[sc_i*ATTN_P*32 +: ATTN_P*32];
                    if (hh != NHEAD-1) begin hh <= hh + 1'b1; st <= S_ROPE_Q0; end
                    else begin hh <= 0; st <= S_ROPE_K0; end
                end
                S_ROPE_K0: begin
                    rope_head_in <= gather_k; rope_pos <= pos;   // direct assign -- Verilog zero-extends correctly on
                    // assignment; a bit-select pos[$clog2(ROPE_TMAX)-1:0] (the
                    // original code here) reads bits beyond pos's own declared
                    // width ($clog2(T2)=3 bits here, ROPE_TMAX needs 7) and returns
                    // X for every out-of-range bit -- found via q_rope_bank ending
                    // up entirely X while q_bank (pre-RoPE) was valid, traced to
                    // rope_pos itself reading X at rope_apply_vec.sv's own start.
                    rope_start <= 1'b1; st <= S_ROPE_K1;
                end
                S_ROPE_K1: if (rope_done) begin
                    for (sc_i = 0; sc_i < HR_ATTN; sc_i = sc_i + 1)
                        k_rope_bank[hh*HR_ATTN + sc_i] <= rope_head_out[sc_i*ATTN_P*32 +: ATTN_P*32];
                    if (hh != NHEAD-1) begin hh <= hh + 1'b1; st <= S_ROPE_K0; end
                    else begin hh <= 0; wi <= 0; st <= S_KVW_K0; end
                end

                // ---- self kv_bank write: K (RoPE'd), then V (raw), per head,
                // at position `pos` -- write-once, never revisited within
                // this `go` (bidirectional full-attend, not incremental). --
                S_KVW_K0: begin
                    sk_wlayer<=blk; sk_wkv<=1'b0; sk_whead<=hh[$clog2(NHEAD)-1:0]; sk_wpos<=pos;
                    sk_wstart<=1'b1; wi<=0; st<=S_KVW_K1;
                end
                S_KVW_K1: begin
                    sk_wdata <= k_rope_bank[hh*HR_ATTN + wi];
                    sk_wvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else if (sk_wdone) begin wi<=0; st<=S_KVW_V0; end
                end
                S_KVW_V0: begin
                    sk_wlayer<=blk; sk_wkv<=1'b1; sk_whead<=hh[$clog2(NHEAD)-1:0]; sk_wpos<=pos;
                    sk_wstart<=1'b1; wi<=0; st<=S_KVW_V1;
                end
                S_KVW_V1: begin
                    sk_wdata <= v_bank[hh*HR_ATTN + wi];
                    sk_wvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else if (sk_wdone) begin
                        if (hh != NHEAD-1) begin hh<=hh+1'b1; wi<=0; st<=S_KVW_K0; end
                        else begin hh<=0; st<=S_PHA_NEXT; end
                    end
                end
                S_PHA_NEXT: begin
                    if (pos != T2-1) begin pos <= pos + 1'b1; st <= S_PHA_START; end
                    else begin pos <= 0; st <= S_PHB_START; end
                end

                // ================= PHASE B: for pos=0..T2-1, attend + MLP ==
                S_PHB_START: begin ri_d <= 0; st <= S_ASELF_START; end

                // ---- self-attn (bidirectional, tcount=T2 always) ----
                S_ASELF_START: begin
                    at_start <= 1'b1; at_tcount <= T2[8:0];  // valid the SAME
                    // cycle start pulses -- see decoder_block_seq.sv's own
                    // S_ASELF_START comment for why (a real hang otherwise).
                    wi <= 0; st <= S_ASELF_Q;
                end
                S_ASELF_Q: begin
                    at_qdata <= q_rope_bank[pos][hh*HR_ATTN + wi];
                    at_qvalid <= 1'b1;
                    if (wi != HR_ATTN-1) wi <= wi + 1'b1;
                    else begin wi<=0; sk_rlayer<=blk; sk_rkv<=1'b0; sk_rhead<=hh[$clog2(NHEAD)-1:0];
                               sk_rtcount<=T2[8:0]; sk_rstart<=1'b1; st<=S_ASELF_K0; end
                end
                S_ASELF_K0: begin
                    at_kvvalid <= sk_rvalid; at_kvdata <= sk_rdata;
                    if (at_kdone) begin
                        sk_rlayer<=blk; sk_rkv<=1'b1; sk_rhead<=hh[$clog2(NHEAD)-1:0];
                        sk_rtcount<=T2[8:0]; sk_rstart<=1'b1; st<=S_ASELF_V0;
                    end
                end
                S_ASELF_V0: begin
                    at_kvvalid <= sk_rvalid; at_kvdata <= sk_rdata;
                    // row-0 write must happen HERE too -- see decoder_block_
                    // seq.sv's own S_ASELF_V0 comment for why (a real bug,
                    // not defensive).
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
                    ri_d<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_RES1: begin
                    xres_bank[pos][ri_d] <= res_add_word;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin ri_d<=0; st<=S_LN2SET; end
                end

                // ---- LN2 ----
                S_LN2SET: begin l_gbase<=1'b1; l_ret<=S_FC1SET; ri_d<=0; st<=L_START; end

                // ---- FC1: D -> FFN, +bias ----
                S_FC1SET: begin
                    gv_m<=FFN[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_fc1;
                    g_src<=3'd0; g_dst<=3'd3; g_frac<=gf_fc1; g_actshift<=ACT_FC1;
                    g_bias_en<=1'b1; g_bias_sel<=1'b0; g_ret<=S_GELU_INIT;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_GELU_INIT: begin ri_f <= 0; ri_g <= 0; st <= S_GELU0; end
                // ---- GELU: one combined feed+drain state (vec_gelu.sv is a
                // plain 3-cycle-latency streaming pipeline, no start/done
                // handshake) -- same idiom as decoder_block_seq.sv's own
                // S_SILU0, but no gate*value multiply: gelu_bank just
                // captures GELU's own output directly. ----
                S_GELU0: begin
                    ge_x <= ge_x_word;
                    ge_vin <= (ri_f != ROWS_FFN);
                    if (ri_f != ROWS_FFN) ri_f <= ri_f + 1'b1;
                    if (ge_vout) begin
                        for (bp = 0; bp < P; bp = bp + 1)
                            gelu_bank[ri_g][bp*32 +: 32] <=
                                {{16{ge_y[bp*16+15]}}, ge_y[bp*16 +: 16]};
                        if (ri_g == ROWS_FFN-1) begin ri_f<=0; ri_g<=0; st<=S_FC2SET; end
                        else ri_g <= ri_g + 1'b1;
                    end
                end

                // ---- FC2: FFN -> D, +bias ----
                S_FC2SET: begin
                    gv_m<=D[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=FFN[$clog2(GEMV_KMAX+1)-1:0]; gv_wbase<=wb_fc2;
                    g_src<=3'd2; g_dst<=3'd3; g_frac<=gf_fc2; g_actshift<=ACT_FC2;
                    g_bias_en<=1'b1; g_bias_sel<=1'b1; g_ret<=S_RES2;
                    ri_d<=0;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                S_RES2: begin
                    xres_bank[pos][ri_d] <= res_add_word;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin
                        ri_d<=0;
                        if (pos != T2-1) begin pos <= pos + 1'b1; st <= S_PHB_START; end
                        else st <= S_DONE;
                    end
                end

                S_DONE: begin done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
