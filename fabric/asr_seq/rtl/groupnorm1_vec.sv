// -----------------------------------------------------------------------------
// groupnorm1_vec -- GroupNorm(num_groups=1) over a whole [C][T] tensor, the
// Stage 1 conv front-end's "groupnorm1" (gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md
// Stage 1 table: `conv1d(conv1) -> tanh_ -> groupnorm1 -> conv2 ...`,
// moonshine's own `nn.GroupNorm(num_groups=1, num_channels=288, eps=1e-5)`,
// modeling_moonshine.py: `hidden_states = self.groupnorm(hidden_states)`).
//
// num_groups=1 means ONE mean/var, computed by reducing over ALL C*T
// elements jointly (not per-row like LayerNorm's own per-D-vector reduction),
// then a per-CHANNEL affine WITH bias:
//   y[c,t] = (x[c,t] - mean) * rsqrt(var + eps) * gamma[c] + beta[c]
// LayerNorm (layernorm_vec_gendiv.sv, this dir) has no bias term at all --
// checkpoint C's/this project's own LayerNorm convention never needed one.
//
// Reduction/rsqrt core (floor-divide mean/var, seed+2-Newton rsqrt) is
// layernorm_vec_gendiv.sv's own machinery, COPIED VERBATIM and unchanged --
// same eps (GroupNorm's eps=1e-5 == LayerNorm's own EPS_A=671), just a much
// bigger divisor (N=C*T instead of D). Two real differences from
// layernorm_vec_gendiv.sv:
//   1. gamma/beta are PER-CHANNEL (C values), not per-element (D values) --
//      loaded ONCE via a separate preload phase (S_GLOAD, CROWS=C/P cycles)
//      before the main x stream (S_LOAD, ROWS=C*T/P cycles), then read back
//      with a WRAPPING row counter (crow, period CROWS) during S_OUT so each
//      channel's gamma/beta is reused across all T timesteps.
//   2. beta needs one MORE pipeline register hop than gamma (brow_0..brow_3
//      vs grow_0/grow_r) since it's added at the FINAL output stage (after
//      the gamma multiply-shift), not consumed mid-pipeline like gamma is.
//
// Storage layout (row r = t*CROWS + cr, matching encoder_block_seq.sv's own
// per-position xres_bank convention -- T is the OUTER loop, channel-rows the
// INNER): element (c,t) with c = cr*P + lane sits at xbank[t*CROWS+cr],
// lane `lane`. gamma_in/beta_in at S_GLOAD time cr (CROWS total cycles) carry
// channels [cr*P, cr*P+P).
//
// Formats: x signed Q6.25 (32b, SAME as layernorm_vec_gendiv.sv -- tanh's own
// Q4.12 output must be left-shifted by 13 before feeding this module, a
// top-level-FSM/Python-reference concern, not this module's). gamma signed
// Q4.20 (SAME as layernorm). beta signed Q10.22 (BETA_FRAC=22 == OUT_FRAC, so
// no extra shift is needed at the final add -- picked deliberately for that
// reason). y signed Q.22 (SAME as layernorm's own y_out).
//
// Protocol: pulse `start`; drive gamma_in/beta_in with gvalid_in for CROWS
// cycles (FIRST); then drive x_in with valid_in for ROWS=C*T/P cycles.
// Results stream on y_out (P*64-bit packed) with y_valid for ROWS cycles;
// `done` pulses at the end. iverilog-2012 safe (plain-vector copies before
// any indexed read of an unpacked element; packed-vector +: part-selects
// only) -- same discipline as layernorm_vec_gendiv.sv.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module groupnorm1_vec #(
    parameter integer P = 8,                // vector lanes (C must be divisible by P)
    parameter integer C = 288,               // channels
    parameter integer T = 45                 // timesteps (conv1 output length at this gate's L=3000 audio convention)
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  start,         // pulse before streaming a new [C][T] tensor
    input  wire                  gvalid_in,     // gamma_in/beta_in valid (CROWS cycles, FIRST)
    input  wire [P*32-1:0]       gamma_in,      // P x Q4.20  (lane k = [32k +: 32])
    input  wire [P*32-1:0]       beta_in,       // P x Q10.22 (lane k = [32k +: 32])
    input  wire                  valid_in,      // x_in valid (ROWS cycles, AFTER gamma/beta)
    input  wire [P*32-1:0]       x_in,          // P x Q6.25  (lane k = [32k +: 32])
    output reg                   y_valid,
    output reg  [P*64-1:0]       y_out,         // P x Q.22 (each sign-extended into 64b)
    output reg                   done
);
    localparam integer CROWS    = C / P;        // gamma/beta rows (loaded once)
    localparam integer N        = C * T;        // total reduction size
    localparam integer ROWS     = N / P;        // x rows (streamed)
    localparam integer QX       = 25;
    localparam integer G_FRAC   = 20;
    localparam integer BETA_FRAC = 22;
    localparam integer A_FRAC   = 26;
    localparam integer Y_FRAC   = 26;
    localparam integer OUT_FRAC = 22;
    localparam integer VAR_FRAC = 50;
    localparam integer SEED_IDX_BITS = 6;
    localparam integer SEED_OUT_FRAC = 16;
    localparam integer OUT_SH   = (QX + Y_FRAC + G_FRAC) - OUT_FRAC;   // 49
    localparam [63:0] EPS_A   = 64'd671;                  // round(1e-5 * 2^26) -- GroupNorm's own eps=1e-5, same as LayerNorm's
    localparam [31:0] SQRT2Q15 = 32'd46341;               // round(sqrt(2)*2^15)
    localparam [127:0] ONE_P5 = 128'd3 <<< (2*Y_FRAC - 1);

    // ---- exact FLOOR integer divide (round toward -infinity), verbatim from
    // layernorm_vec_gendiv.sv (see that file's header for the derivation) --
    // only the call-site divisor changes (N instead of D).
    function automatic signed [39:0] floordiv40;
        input signed [39:0] num;
        input integer d;
        begin
            if (num >= 0) floordiv40 = num / d;
            else          floordiv40 = -(((-num) + d - 1) / d);
        end
    endfunction
    function automatic signed [71:0] floordiv72;
        input signed [71:0] num;
        input integer d;
        begin
            if (num >= 0) floordiv72 = num / d;
            else          floordiv72 = -(((-num) + d - 1) / d);
        end
    endfunction

    // ---- x storage: one wide row per (t,cr) -- SAME wide-word-per-row idiom
    // as layernorm_vec_gendiv.sv (see that file's header: avoids the
    // runtime-indexed-2D-array mux-tree blowup). gamma/beta storage is now
    // MUCH smaller (CROWS rows, not ROWS) since they don't vary with t.
    (* ram_style = "distributed" *) reg [P*32-1:0] xbank [0:ROWS-1];
    (* ram_style = "distributed" *) reg [P*32-1:0] gbank [0:CROWS-1];
    (* ram_style = "distributed" *) reg [P*32-1:0] bbank [0:CROWS-1];
    reg [$clog2(ROWS+1)-1:0]  wptr;              // x write/row pointer
    reg [$clog2(CROWS+1)-1:0] gptr;              // gamma/beta write pointer
    reg [$clog2(CROWS+1)-1:0] crow;              // gamma/beta READ pointer during S_OUT (wraps every CROWS)

    // ---- seed ROM (64 x Q1.16) -- verbatim, shared table with LayerNorm ------
    (* rom_style = "block" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);

    // ---- accumulators (built DURING the P-wide x load) ------------------------
    reg signed [39:0]  sum;                     // sum of X (Q6.25)
    reg signed [71:0]  sumxx;                   // sum of x*x (Q.50)
    reg signed [39:0]  mean;                    // Q6.25
    reg signed [71:0]  ssq;                     // centered sum (Q.50)
    reg signed [71:0]  var_q;                   // Q12.50
    reg [$clog2(ROWS+1)-1:0] ridx, oidx;        // output row counters

    // ---- rsqrt registers (verbatim from layernorm_vec_gendiv.sv) --------------
    reg signed [63:0] A, Yr;
    reg [1:0]  newt;
    reg [7:0]  msb;
    reg [5:0]  seed_idx;
    reg signed [8:0]  Eexp, half, qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;
    reg [19:0]        seed_val;
    reg signed [127:0] yy, ayy, term;
    reg signed [191:0] ynew;
    reg signed [127:0] cterm, msq;

    // ---- load reduction: per-cycle P-input partial sums (PIPELINED), verbatim
    // from layernorm_vec_gendiv.sv -- same two-stage skew (Fmax rationale),
    // same 1-row/cycle iteration, just no gbank write here (gamma/beta load
    // separately in S_GLOAD, below).
    integer lp;
    reg signed [39:0]  xe_r   [0:P-1];
    reg signed [63:0]  sq_r   [0:P-1];
    reg signed [39:0]  psum;
    reg signed [71:0]  psumxx;
    reg signed [39:0]  psum_r;
    reg signed [71:0]  psumxx_r;
    reg                lv_a, lv1;
    reg [$clog2(ROWS+1)-1:0] acnt;
    reg signed [31:0]  xl;
    reg signed [39:0]  xej;
    reg signed [63:0]  sqj;
    always @(*) begin
        psum   = 40'sd0;
        psumxx = 72'sd0;
        for (lp = 0; lp < P; lp = lp + 1) begin
            xej    = xe_r[lp];
            sqj    = sq_r[lp];
            psum   = psum   + xej;
            psumxx = psumxx + $signed({{8{sqj[63]}}, sqj});
        end
    end

    // ---- combinational MSB index of A (verbatim) -------------------------------
    integer b;
    reg [7:0] msb_c;
    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    // ---- output pipeline: same 5-stage split as layernorm_vec_gendiv.sv, PLUS
    // a 4-hop beta pass-through (brow_0..brow_3) since beta is consumed one
    // stage LATER than gamma (at the final add, not the multiply).
    reg signed [39:0]  xc     [0:P-1];
    reg signed [95:0]  prod   [0:P-1];
    reg signed [31:0]  grow_0 [0:P-1];
    reg signed [31:0]  grow_r [0:P-1];
    reg signed [31:0]  brow_0 [0:P-1];
    reg signed [31:0]  brow_1 [0:P-1];
    reg signed [31:0]  brow_2 [0:P-1];
    reg signed [31:0]  brow_3 [0:P-1];
    reg signed [127:0] pp_hi  [0:P-1];
    reg signed [127:0] pp_lo  [0:P-1];
    reg signed [127:0] prod2_r [0:P-1];
    reg                s0v, s1v, s2v, s2bv;
    reg signed [31:0]  xo, go, bo;
    reg [P*32-1:0]     xword, gword, bword;
    reg signed [39:0]  xj;
    reg signed [95:0]  pj;
    reg signed [47:0]  pjhi;
    reg        [47:0]  pjlo;
    reg signed [127:0] prod2;
    reg signed [127:0] p2j;
    reg signed [127:0] hj, lj;

    // ---- FSM --------------------------------------------------------------------
    localparam [4:0]
        S_IDLE=0, S_GLOAD=1, S_LOAD=2, S_VAR=3, S_VAR2=4, S_MSB=5, S_SEED=6, S_SEED2=7,
        S_NEWT0=8, S_NEWT0B=9, S_NEWTA=10, S_NEWTA2=11, S_NEWTB=12, S_NEWTB2=13,
        S_NEWTC=14, S_NEWTC2=15, S_NEWTC3=16, S_OUT=17, S_DONE=18;
    reg [4:0] state;
    reg signed [127:0] p_ct, p_mq;
    reg signed [127:0] yy_p;
    reg signed [191:0] ay_p;
    reg signed [63:0]  yn_r;
    reg signed [63:0]  Y0_r;
    reg                rbit_r;
    integer wp;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; wptr <= 0; gptr <= 0; crow <= 0; ridx <= 0; oidx <= 0;
            y_valid <= 1'b0; done <= 1'b0; s0v <= 1'b0; s1v <= 1'b0; s2v <= 1'b0; s2bv <= 1'b0;
            sum <= 0; sumxx <= 0; newt <= 0;
        end else begin
            y_valid <= 1'b0;
            done    <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        gptr <= 0;
                        state <= S_GLOAD;
                    end
                end
                // ---- per-channel gamma/beta preload: CROWS cycles, once -------
                S_GLOAD: begin
                    if (gvalid_in) begin
                        gbank[gptr] <= gamma_in;
                        bbank[gptr] <= beta_in;
                        gptr <= gptr + 1'b1;
                        if (gptr == CROWS-1) begin
                            wptr <= 0; sum <= 0; sumxx <= 0; newt <= 0;
                            acnt <= 0; lv_a <= 1'b0; lv1 <= 1'b0;
                            state <= S_LOAD;
                        end
                    end
                end
                // ---- P-wide x load: identical structure to
                // layernorm_vec_gendiv.sv's own S_LOAD (no gbank write here). ---
                S_LOAD: begin
                    if (valid_in) begin
                        xbank[wptr] <= x_in;
                        for (wp = 0; wp < P; wp = wp + 1) begin
                            xl              =  x_in[wp*32 +: 32];
                            xe_r[wp]        <= $signed({{8{xl[31]}}, xl});
                            sq_r[wp]        <= $signed(xl) * $signed(xl);
                        end
                        wptr     <= wptr + 1'b1;
                    end
                    lv_a <= valid_in;
                    if (lv_a) begin
                        psum_r   <= psum;
                        psumxx_r <= psumxx;
                    end
                    lv1 <= lv_a;
                    if (lv1) begin
                        sum   <= sum   + psum_r;
                        sumxx <= sumxx + psumxx_r;
                        acnt  <= acnt + 1'b1;
                        if (acnt == ROWS-1) state <= S_VAR;
                    end
                end
                // ---- algebraic centered sum, divisor is N=C*T now (was D) -----
                S_VAR: begin
                    mean  <= floordiv40(sum, N);
                    p_ct  <= $signed(floordiv40(sum, N)) * $signed(sum);
                    p_mq  <= $signed(floordiv40(sum, N)) * $signed(floordiv40(sum, N));
                    state <= S_VAR2;
                end
                S_VAR2: begin
                    cterm = p_ct <<< 1;
                    msq   = N * p_mq;
                    ssq   <= $signed(sumxx) - cterm + msq;
                    state <= S_MSB;
                end
                S_MSB: begin
                    var_q <= floordiv72(ssq, N);
                    A <= ($signed(floordiv72(ssq, N)) >>> (VAR_FRAC - A_FRAC)) + $signed(EPS_A);
                    state <= S_SEED;
                end
                S_SEED: begin
                    msb  <= msb_c;
                    Eexp <= $signed({1'b0, msb_c}) - A_FRAC;
                    half <= A_FRAC - $signed({1'b0, msb_c});
                    if (msb_c >= SEED_IDX_BITS)
                        seed_idx <= (A >> (msb_c - SEED_IDX_BITS)) & 6'h3F;
                    else
                        seed_idx <= (A << (SEED_IDX_BITS - msb_c)) & 6'h3F;
                    state <= S_SEED2;
                end
                S_SEED2: begin
                    seed_val <= seed_rom[seed_idx];
                    if (half[8] == 1'b0) begin qsh <= half >>> 1; rbit <= half[0]; end
                    else begin qsh <= -(((-half) + 1) >>> 1); rbit <= half[0]; end
                    newt  <= 0;
                    state <= S_NEWT0;
                end
                S_NEWT0: begin
                    seed_shifted = $signed({44'd0, seed_val}) <<< (Y_FRAC - SEED_OUT_FRAC);
                    if (qsh[8] == 1'b0) Y0_r <= seed_shifted <<< qsh;
                    else                Y0_r <= seed_shifted >>> (-qsh);
                    rbit_r <= rbit;
                    state  <= S_NEWT0B;
                end
                S_NEWT0B: begin
                    if (rbit_r) Yr <= (Y0_r * $signed({33'd0, SQRT2Q15})) >>> 15;
                    else        Yr <= Y0_r;
                    state <= S_NEWTA;
                end
                S_NEWTA:  begin yy_p <= Yr * Yr;             state <= S_NEWTA2; end
                S_NEWTA2: begin yy   <= yy_p;                state <= S_NEWTB;  end
                S_NEWTB:  begin ay_p <= A * yy;              state <= S_NEWTB2; end
                S_NEWTB2: begin ayy  <= ay_p >>> A_FRAC;     state <= S_NEWTC;  end
                S_NEWTC: begin
                    term  <= ONE_P5 - (ayy >>> 1);
                    state <= S_NEWTC2;
                end
                S_NEWTC2: begin
                    ynew = (Yr * term) >>> (2*Y_FRAC);
                    yn_r <= ynew[63:0];
                    state <= S_NEWTC3;
                end
                S_NEWTC3: begin
                    Yr   <= yn_r;
                    newt <= newt + 1'b1;
                    if (newt == 2'd1) begin
                        ridx <= 0; oidx <= 0; crow <= 0;
                        s0v <= 1'b0; s1v <= 1'b0; s2v <= 1'b0; s2bv <= 1'b0;
                        state <= S_OUT;
                    end else state <= S_NEWTA;
                end
                // ---- P-wide pipelined output stream: gamma/beta read via crow,
                // which wraps every CROWS rows (period matches how S_LOAD's own
                // xbank rows are t-major/channel-row-minor). ---------------------
                S_OUT: begin
                    if (ridx < ROWS) begin
                        xword = xbank[ridx];
                        gword = gbank[crow];
                        bword = bbank[crow];
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            xo = xword[lp*32 +: 32];
                            go = gword[lp*32 +: 32];
                            bo = bword[lp*32 +: 32];
                            xc[lp]     <= $signed({{8{xo[31]}}, xo}) - mean;
                            grow_0[lp] <= go;
                            brow_0[lp] <= bo;
                        end
                        ridx <= ridx + 1'b1;
                        crow <= (crow == CROWS-1) ? {$clog2(CROWS+1){1'b0}} : crow + 1'b1;
                        s0v  <= 1'b1;
                    end else s0v <= 1'b0;
                    s1v <= s0v;
                    if (s0v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            xj = xc[lp];
                            prod[lp]   <= xj * Yr;
                            grow_r[lp] <= grow_0[lp];
                            brow_1[lp] <= brow_0[lp];
                        end
                    end
                    s2v <= s1v;
                    if (s1v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            pj    = prod[lp];
                            go    = grow_r[lp];
                            pjhi  = pj[95:48];
                            pjlo  = pj[47:0];
                            pp_hi[lp]  <= $signed(pjhi) * $signed({{96{go[31]}}, go});
                            pp_lo[lp]  <= $signed({1'b0, pjlo}) * $signed({{96{go[31]}}, go});
                            brow_2[lp] <= brow_1[lp];
                        end
                    end
                    s2bv <= s2v;
                    if (s2v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            hj = pp_hi[lp];
                            lj = pp_lo[lp];
                            prod2_r[lp] <= (hj <<< 48) + lj;
                            brow_3[lp]  <= brow_2[lp];
                        end
                    end
                    if (s2bv) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            p2j = prod2_r[lp];
                            y_out[lp*64 +: 64] <= (p2j >>> OUT_SH)
                                                 + $signed({{32{brow_3[lp][31]}}, brow_3[lp]});
                        end
                        y_valid <= 1'b1;
                        if (oidx == ROWS-1) state <= S_DONE;
                        oidx <= oidx + 1'b1;
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule // groupnorm1_vec
