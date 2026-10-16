// -----------------------------------------------------------------------------
// output_head_seq -- the real, sized top-level FSM for ASR's output head
// (Stage 4 of gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md): final decoder
// LayerNorm -> tied lm_head GEMV (VOCAB=32768 x D=288, INT8) -> argmax.
// Runs once per real decode step (fed the decoder's own final-layer hidden
// state each time -- no `blk`/layer concept here, this component exists
// exactly once, not per-layer).
//
// Real, deliberate difference from decoder_block_seq.sv/encoder_block_seq.sv's
// OWN dequant scheme: this file uses vec_dequant.sv (checkpoint C's real,
// already-deployed PER-ROW mantissa/exponent dequant), not the per-matrix
// single-g_frac simplification those two files use. With VOCAB=32768 very
// different output rows feeding directly into an argmax, one shared shift
// for the whole matrix risks real precision loss that could flip which
// token wins -- the real firmware (model/c_port/ops.c's own
// linear_lmhead_i8) already uses a per-row float scale for exactly this
// reason; vec_dequant.sv is the RTL equivalent (mant*2^exp per row,
// matching fabric.stage3.seq_ref.quantize_scale_24() exactly). The lm_head
// weight matrix is quantized per-ROW too (not per-matrix), for the same
// reason on the weight side.
//
// GEMV activation quantization stays this project's own established
// per-call single-shift scheme (ACT_LM, a compile-time parameter, the
// SAME activation feeds every one of the 32768 output rows so one shift
// is the right granularity there, unlike the weight/dequant side).
//
// Only ONE GEMV call and ONE LayerNorm exist in this whole module (unlike
// decoder/encoder blocks' shared multi-call dispatchers) -- no g_dst mux,
// no g_ret return-state field needed; the shared G_XFEED/G_XSTART/G_WAIT
// feed idiom is still reused (same established, proven protocol), but the
// drain is a dedicated state (S_LMDRAIN) feeding vec_dequant.sv + argmax
// directly, not the generic gdequant()-into-a-bank drain those two files
// use (this file has nowhere to park 32768 dequantized rows, and doesn't
// need to -- argmax only ever needs the running max, not the full vector).
//
// STATUS: gated bit-exact against pack_output_head.py's own reference (real
// tied lm_head weight, real final-decoder-LayerNorm gamma, real per-step
// decoder hidden state) -- see that gate's own verdict line. Two real bugs
// found on the way there, both genuinely new (VOCAB=32768 is far larger
// than anything gated in this project before, both in row count and in the
// real hidden-state magnitudes feeding it):
// 1. `gi` (the drain-side readback/feed counter) needs to count up to
//    ROWS_VOCAB+1 (4097 at VOCAB=32768), but was declared
//    `$clog2(GEMV_MMAX/P)` wide -- copied verbatim from decoder_block_seq.sv/
//    encoder_block_seq.sv's own `gi`, where GEMV_MMAX (FFN=1152/DFFN2=2304)
//    never got close to that width's own max. Here it silently wrapped at
//    4096 (exactly the missing bit's own range) and the drain loop never
//    reached its terminal condition -- found via a debug tap re-firing on
//    a ~4096-cycle period, far too soon for a real 256-group GEMV to have
//    actually finished and restarted. Fixed by widening `gi` with explicit
//    margin (`$clog2(GEMV_MMAX/P + 2)`), not by copying a width formula
//    that happened to work at smaller scale.
// 2. The REAL decoder hidden state's own magnitude (confirmed up to ~240
//    in real units) already exceeds Q6.25's 32-bit range for some of its
//    288 elements even before any layer-to-layer accumulation --
//    pack_output_head.py's own xres0/xres35 matched bit-for-bit (those two
//    elements happened to be in range) while the LayerNorm's own internal
//    sum (over all 288 elements) still diverged, since other, unchecked
//    elements were not. Same class of fix as decoder_block_seq.sv/
//    encoder_block_seq.sv's own wrap32() (xres_bank truncates to 32
//    bits/lane on store, the Python reference must match), but triggered
//    by a fresh, real input's own magnitude rather than accumulated
//    residual growth.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module output_head_seq #(
    parameter integer P     = 8,
    parameter integer D     = 288,
    parameter integer VOCAB = 32768,
    // ACT_LM stays a compile-time parameter -- one fixed ACT_RSHIFT for the
    // single activation vector every output row's own dot product shares
    // (see this file's own header for why the WEIGHT side, in contrast,
    // needs a genuinely per-row scheme).
    parameter signed [6:0] ACT_LM = 17,
    // vec_dequant's own target output fraction -- 0 (plain integer
    // "logit") is the natural choice: argmax only cares about relative
    // ordering, and every row goes through the SAME frac, so the specific
    // value doesn't change which token wins, only the absolute magnitude.
    parameter signed [6:0] DQ_FRAC = 0
) (
    input  wire clk,
    input  wire rst,

    input  wire        go,             // pulse: run the output head on the CURRENT xres_bank
    output reg          done,
    output reg  [$clog2(VOCAB)-1:0] argmax_idx,
    output reg  signed [31:0]       argmax_val,

    // real word offset into the resident weight image, runtime port (no
    // per-layer concept here, but kept a port rather than a parameter for
    // consistency with decoder/encoder's own convention, and because a
    // real deployment may still want the lm_head image relocatable).
    input  wire [19:0] wb_lm,

    // residual stream in -- the decoder's own final-layer hidden state,
    // P*32-bit packed rows, D/P rows, Q6.25 (SAME format decoder_block_
    // seq.sv's own xres_bank uses, so it can be fed directly).
    input  wire                   xres_wr,
    input  wire [$clog2(D/P)-1:0] xres_waddr,
    input  wire [P*32-1:0]        xres_wdata,

    // ---- GEMV weight load (passthrough to gemv_banked_resident_vec) --------
    input  wire        gv_ld_rst,
    input  wire        gv_ld_we,
    input  wire [31:0] gv_ld_data,

    // ---- final-LN gamma load -------------------------------------------------
    input  wire         gam_we,
    input  wire [$clog2(D/P)-1:0] gam_waddr,
    input  wire [P*32-1:0]        gam_wdata,

    // ---- per-row dequant (mant,exp) table load -- ROWS_VOCAB=VOCAB/P rows,
    // P lanes/row, matching vec_dequant.sv's own packed-bus convention. ----
    input  wire         dq_we,
    input  wire [$clog2(VOCAB/P)-1:0] dq_waddr,
    input  wire [P*24-1:0]            dq_wmant,
    input  wire [P*8-1:0]             dq_wexp
);
    localparam integer ROWS_D     = D / P;                // 36
    localparam integer ROWS_VOCAB = VOCAB / P;             // 4096

    // =========================================================================
    (* ram_style = "block" *) reg [P*32-1:0] xres_bank   [0:ROWS_D-1];   // Q6.25
    (* ram_style = "block" *) reg [P*32-1:0] ln_out_bank [0:ROWS_D-1];   // Q.22
    always @(posedge clk) if (xres_wr) xres_bank[xres_waddr] <= xres_wdata;

    (* ram_style = "block" *) reg [P*32-1:0] gamma_bank [0:ROWS_D-1];
    always @(posedge clk) if (gam_we) gamma_bank[gam_waddr] <= gam_wdata;

    (* ram_style = "block" *) reg [P*24-1:0] mant_bank [0:ROWS_VOCAB-1];
    (* ram_style = "block" *) reg [P*8-1:0]  exp_bank  [0:ROWS_VOCAB-1];
    always @(posedge clk) if (dq_we) begin
        mant_bank[dq_waddr] <= dq_wmant;
        exp_bank[dq_waddr]  <= dq_wexp;
    end

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
    // ln_yout packs P lanes at 64 bits each -- see decoder_block_seq.sv's
    // own u_ln comment for why the low-256-bits-direct read is wrong.
    integer lnp;
    reg [P*32-1:0] ln_out_word;
    always @* begin
        ln_out_word = {(P*32){1'b0}};
        for (lnp = 0; lnp < P; lnp = lnp + 1)
            ln_out_word[lnp*32 +: 32] = ln_yout[lnp*64 +: 32];
    end

    // =========================================================================
    // ---- gemv_banked_resident_vec (real, WBW=8), M=VOCAB, K=D ---------------
    localparam integer GEMV_MMAX = VOCAB;
    localparam integer GEMV_KMAX = D;
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

    integer bp;
    reg [P*8-1:0] act_word;
    always @* begin
        act_word = {(P*8){1'b0}};
        for (bp = 0; bp < P; bp = bp + 1)
            act_word[bp*8 +: 8] = actquant($signed(ln_out_bank[ri_g][bp*32 +: 32]), ACT_LM);
    end

    // =========================================================================
    // ---- vec_dequant (real, unmodified) -- per-row mant/exp dequant --------
    reg                   vdq_in_valid;
    reg [P*32-1:0]        vdq_gemvy;
    reg [P*24-1:0]        vdq_mant;
    reg [P*8-1:0]         vdq_exp;
    wire                  vdq_out_valid;
    wire [P*32-1:0]       vdq_dq_out;
    vec_dequant #(.P(P)) u_dequant (
        .clk(clk), .rst(rst), .in_valid(vdq_in_valid), .frac(DQ_FRAC),
        .gemvy(vdq_gemvy), .mant(vdq_mant), .exp(vdq_exp),
        .out_valid(vdq_out_valid), .dq_out(vdq_dq_out)
    );

    // =========================================================================
    // ---- argmax: find the max of vdq_dq_out's P lanes THIS cycle (comb),
    // then compare against the running max (registered) -- avoids a
    // multi-writer race across the P lanes updating the same running-max
    // register independently. Strict `>` (not `>=`), matching the real
    // C reference's own argmax_f: first occurrence wins on a tie, and rows
    // are processed in strictly increasing index order here. ------------
    integer amp;
    reg signed [31:0] local_max_val;
    reg [$clog2(P)-1:0] local_max_lane;
    always @* begin
        local_max_val = $signed(vdq_dq_out[31:0]);
        local_max_lane = 0;
        for (amp = 1; amp < P; amp = amp + 1)
            if ($signed(vdq_dq_out[amp*32 +: 32]) > local_max_val) begin
                local_max_val = $signed(vdq_dq_out[amp*32 +: 32]);
                local_max_lane = amp[$clog2(P)-1:0];
            end
    end

    // =========================================================================
    // ---- state encoding -----------------------------------------------------
    localparam [4:0]
        S_IDLE=0,
        S_LNSET=1, L_FEED=2, L_WAIT=3, L_START=4,
        S_GSET=5, G_XRESET=6, G_XFEED=7, G_XSTART=8, G_WAIT=9,
        S_LMDRAIN=10,
        S_DONE=11;
    reg [4:0] st;

    reg [$clog2(ROWS_D)-1:0]     ri_d;
    reg [$clog2(ROWS_D)-1:0]     ri_g;      // feed-side row counter (LN output rows, 0..35)
    // gi must count up to gcount+1 = ROWS_VOCAB+1 (4097 at VOCAB=32768), NOT
    // just ROWS_VOCAB/P-1=4095 -- $clog2(GEMV_MMAX/P) (the width decoder_
    // block_seq.sv/encoder_block_seq.sv's own `gi` use, copied here at first)
    // gives exactly enough bits to hold ROWS_VOCAB-1, one short of what THIS
    // module's own terminal value needs. Harmless at their much smaller
    // GEMV_MMAX (FFN=1152/DFFN2=2304, where the terminal value never gets
    // close to the width's own max); real here: gi silently wrapped at 4096
    // (0..4095, exactly the missing bit's own range) and the drain loop
    // never reached its terminal condition, re-entering the same low gi
    // values on a ~4096-cycle period instead of running to completion --
    // found via a "gi==2" debug tap firing twice ~4096 cycles apart, far too
    // soon for a real 256-group GEMV to have finished and restarted.
    reg [$clog2(GEMV_MMAX/P + 2)-1:0] gi;
    reg [$clog2(ROWS_VOCAB)-1:0] gi2;
    reg [$clog2(ROWS_VOCAB+1)-1:0] ri_out;  // drain-side output-group counter, 0..ROWS_VOCAB

    always @(posedge clk) begin
        done <= 1'b0;
        ln_start <= 1'b0; ln_vin <= 1'b0;
        gv_start <= 1'b0; gv_xwe <= 1'b0; gv_xrst <= 1'b0;
        vdq_in_valid <= 1'b0;

        if (rst) begin
            st <= S_IDLE;
        end else begin
            case (st)
                S_IDLE: if (go) begin ri_d <= 0; st <= S_LNSET; end

                // ---- final LayerNorm ----
                S_LNSET: begin ri_d <= 0; st <= L_START; end
                L_START: begin ln_start <= 1'b1; ri_d <= 0; st <= L_FEED; end
                L_FEED: begin
                    ln_x <= xres_bank[ri_d]; ln_g <= gamma_bank[ri_d];
                    ln_vin <= 1'b1;
                    if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    else begin ri_d <= 0; st <= L_WAIT; end
                end
                L_WAIT: begin
                    if (ln_yvalid) begin
                        ln_out_bank[ri_d] <= ln_out_word;
                        if (ri_d != ROWS_D-1) ri_d <= ri_d + 1'b1;
                    end
                    if (ln_done) st <= S_GSET;
                end

                // ---- lm_head GEMV dispatch (shared feed idiom, ONE call site) --
                S_GSET: begin
                    gv_m<=VOCAB[$clog2(GEMV_MMAX+1)-1:0]; gv_k<=D[$clog2(GEMV_KMAX+1)-1:0];
                    gv_wbase<=wb_lm;
                    ri_g<=0; gi<=0; st<=G_XRESET;
                end
                G_XRESET: begin gv_xrst <= 1'b1; st <= G_XFEED; end
                G_XFEED: begin
                    gv_xdata <= act_word;
                    gv_xwe   <= 1'b1;
                    if (ri_g != ROWS_D-1) ri_g <= ri_g + 1'b1;
                    else begin ri_g <= 0; st <= G_XSTART; end
                end
                G_XSTART: begin gv_start <= 1'b1; st <= G_WAIT; end
                G_WAIT: if (gv_done) begin
                    gi <= 0; ri_out <= 0;
                    argmax_val <= 32'sh80000000;   // -2^31, any real logit beats it
                    argmax_idx <= 0;
                    st <= S_LMDRAIN;
                end

                // ---- combined readback + vec_dequant feed + argmax drain
                // (same "input counter != output counter, both advancing
                // concurrently" idiom as decoder_block_seq.sv's own S_SILU0/
                // encoder_block_seq.sv's own S_GELU0 -- vec_dequant.sv is a
                // plain 3-cycle-latency streaming pipeline, no start/done
                // handshake, so most out_valid pulses land WHILE gi is still
                // advancing, not after). ----
                S_LMDRAIN: begin
                    if (gi >= 2) begin
                        gi2 = gi - 2;
                        vdq_in_valid <= 1'b1;
                        vdq_gemvy <= gv_yout;
                        vdq_mant  <= mant_bank[gi2];
                        vdq_exp   <= exp_bank[gi2];
                    end
                    if (gi != (gv_m + P - 1) / P + 1) gi <= gi + 1'b1;

                    if (vdq_out_valid) begin
                        if (local_max_val > argmax_val) begin
                            argmax_val <= local_max_val;
                            argmax_idx <= ri_out * P + local_max_lane;
                        end
                        if (ri_out == (gv_m + P - 1) / P - 1) st <= S_DONE;
                        else ri_out <= ri_out + 1'b1;
                    end
                end

                S_DONE: begin done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
