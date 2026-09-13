// -----------------------------------------------------------------------------
// layernorm_vec — P-WIDE-I/O gamma-only LayerNorm over d=256 (the 10k-datapath block).
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * gamma[i]
//
// Bit-EXACT to the SAME integer reference as rtl/layernorm.sv and run_layernorm.ln_int.
// The difference from layernorm.sv / layernorm_par is the I/O width: P (x,gamma) accepted
// per cycle and P y emitted per cycle (D/P rows instead of D cycles each way). The variance
// is built ALGEBRAICALLY during the load (no second pass), the rsqrt Newton is pipelined
// (3 single-multiply stages), and the output is pipelined (2 stages) — so the deep
// combinational paths that capped Fmax in the scalar core are broken here too.
//
// Bit-exactness rests on three integer identities (all exact, no approximation):
//   * sum and sum(x*x) accumulated P/cycle == the serial 1/cycle sums (integer add assoc.)
//   * sum_i (x_i-mean)^2 == sum(x*x) - 2*mean*sum + D*mean^2  for mean = sum>>>8 (floored)
//   * the rsqrt seed+2-Newton and the (d*Yr*gamma)>>>49 output are copied verbatim.
//
// Formats (pinned, identical to layernorm.sv):
//   x  signed Q6.25 (32b)   gamma signed Q4.20   mean=sum>>>8 (Q6.25)
//   ssq = sum(xx) - 2*mean*sum + D*mean^2 (Q12.50)   var=ssq>>>8   A=(var>>>24)+eps (Q.26)
//   rsqrt -> Q.26     y = ((d*rsqrt)*gamma) >>> 49  (Q.22)
//
// Protocol: pulse `start`; then drive x_in/gamma_in (each P*32-bit packed, lane k = [32k+:32])
// with valid_in for D/P cycles. Results stream on y_out (P*64-bit packed) with y_valid for
// D/P cycles; `done` pulses at the end. iverilog-2012 safe (plain-vector copies before any
// indexed read of an unpacked element; packed-vector +: part-selects only).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module layernorm_vec #(
    parameter integer P = 8,                // vector lanes (D must be divisible by P)
    // Genesys2 port: the deployed KV260 build only ever ran D=256. The mean/var
    // divide below is an exact arithmetic right-shift by $clog2(D) (see D_SH),
    // which is only bit-exact for a power-of-2 D -- every embedding width this
    // project trains at (128, 256, ...) is one; a non-power-of-2 D would need a
    // real divider here, not a parameter change.
    parameter integer D = 256
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  start,         // pulse before streaming a new vector
    input  wire                  valid_in,      // x_in/gamma_in valid (D/P cycles)
    input  wire [P*32-1:0]       x_in,          // P x Q6.25  (lane k = [32k +: 32])
    input  wire [P*32-1:0]       gamma_in,      // P x Q4.20
    output reg                   y_valid,
    output reg  [P*64-1:0]       y_out,         // P x Q.22 (each sign-extended into 64b)
    output reg                   done
);
    localparam integer ROWS     = D / P;        // rows of P
    localparam integer D_SH     = $clog2(D);    // exact /D shift; D must be pow2
    localparam integer QX       = 25;
    localparam integer G_FRAC   = 20;
    localparam integer A_FRAC   = 26;
    localparam integer Y_FRAC   = 26;
    localparam integer OUT_FRAC = 22;
    localparam integer VAR_FRAC = 50;
    localparam integer SEED_IDX_BITS = 6;
    localparam integer SEED_OUT_FRAC = 16;
    localparam integer OUT_SH   = (QX + Y_FRAC + G_FRAC) - OUT_FRAC;   // 49
    localparam [63:0] EPS_A   = 64'd671;                  // round(1e-5 * 2^26)
    localparam [31:0] SQRT2Q15 = 32'd46341;               // round(sqrt(2)*2^15)
    localparam [127:0] ONE_P5 = 128'd3 <<< (2*Y_FRAC - 1);

    // ---- P-banked input storage: wide-word rows (lane l = [l*32 +: 32]) -------
    // Was reg [0:P-1][0:ROWS-1] (P separate row-addressed arrays) -> the project's
    // own "wide-word banking, not [P][rows]" gotcha: a variable-row access to that
    // shape synthesises to P per-lane row-muxes (MUXF7/LUT blow-up + huge CE-net
    // fanout, the confirmed kv256 congestion hotspot). One row-addressed wide word
    // per buffer makes the variable row a memory ADDRESS, not a mux. Bit-exact
    // (same values, same layout on the wire): x_in/gamma_in are ALREADY the packed
    // P-lane words, so the write is a single whole-row store. LUTRAM (distributed)
    // frees ~16k FFs + the two 400+-fanout CE trees. Gated bit-exact by run_layernorm.
    (* ram_style = "distributed" *) reg [P*32-1:0] xbank [0:ROWS-1];
    (* ram_style = "distributed" *) reg [P*32-1:0] gbank [0:ROWS-1];
    reg [$clog2(ROWS+1)-1:0] wptr;              // 0..ROWS write/row pointer

    // ---- seed ROM (64 x Q1.16) -----------------------------------------------
    (* rom_style = "block" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);

    // ---- accumulators (built DURING the P-wide load) -------------------------
    reg signed [39:0]  sum;                     // sum of X (Q6.25)
    reg signed [71:0]  sumxx;                   // sum of x*x (Q.50)
    reg signed [39:0]  mean;                    // Q6.25
    reg signed [71:0]  ssq;                     // centered sum (Q.50)
    reg signed [71:0]  var_q;                   // Q12.50
    reg [$clog2(ROWS+1)-1:0] ridx, oidx;        // output row counters

    // ---- rsqrt registers (verbatim from the scalar core) ---------------------
    reg signed [63:0] A, Yr;
    reg [1:0]  newt;
    reg [7:0]  msb;
    reg [5:0]  seed_idx;
    reg signed [8:0]  Eexp, half, qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;
    reg [19:0]        seed_val;
    // Newton pipeline regs (single multiply per stage)
    reg signed [127:0] yy, ayy, term;
    reg signed [191:0] ynew;
    // var algebra temporaries
    reg signed [127:0] cterm, msq;

    // ---- load reduction: per-cycle P-input partial sums (PIPELINED) -----------
    // Fmax (P=16): the 16-lane x*x products + the wide adder tree was one giant
    // combinational cloud from x_in to psumxx_r (worst impl path at P=16). Split
    // into two register stages: (stage A) register the per-lane squares + the
    // sign-extended lane values; (stage B) sum the REGISTERED per-lane terms into
    // psum_r/psumxx_r. This costs one extra cycle of load latency (folded into the
    // existing valid skew, see lv_a -> lv1) but does NOT change the 1-row/cycle
    // load iteration: xbank writes and wptr still advance on valid_in.
    integer lp;
    reg signed [39:0]  xe_r   [0:P-1];           // stage-A: per-lane sign-extended x
    reg signed [63:0]  sq_r   [0:P-1];           // stage-A: per-lane x*x (32x32 -> 64b)
    reg signed [39:0]  psum;                     // stage-B sum of the P lanes (comb)
    reg signed [71:0]  psumxx;                   // stage-B sum of x*x of the P lanes
    reg signed [39:0]  psum_r;                   // registered partials (Fmax: tree
    reg signed [71:0]  psumxx_r;                 //   retimes into DSPs / a clean add)
    reg                lv_a, lv1;
    reg [$clog2(ROWS+1)-1:0] acnt;
    reg signed [31:0]  xl;                       // plain-vector lane copy
    reg signed [39:0]  xej;                      // plain-vector copies (stage-B read)
    reg signed [63:0]  sqj;
    // stage-B: sum the REGISTERED per-lane terms (xe_r/sq_r) — a clean adder tree
    always @(*) begin
        psum   = 40'sd0;
        psumxx = 72'sd0;
        for (lp = 0; lp < P; lp = lp + 1) begin
            xej    = xe_r[lp];                    // plain-reg copies (iverilog-safe)
            sqj    = sq_r[lp];
            psum   = psum   + xej;
            psumxx = psumxx + $signed({{8{sqj[63]}}, sqj});
        end
    end

    // ---- combinational MSB index of A ----------------------------------------
    integer b;
    reg [7:0] msb_c;
    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    // ---- output pipeline (5 stages): s0 xc=(x-mean) ; s1 prod=xc*Yr ; s2 hi/lo partial
    //   products of prod*gamma ; s2b combine the partials -> prod2_r ; s3 y=prod2>>>OUT_SH.
    //   Stage 2/3 split (Fmax §21: register the raw product prod2_r then shift+pack next cyc).
    //   NEW stage-2 split (Fmax @TMAX=16): the worst u_ln path was the 96-bit DSP output
    //   prod -> the 96x32 prod*gamma multiply -> prod2_r (WNS -0.936, 15 logic levels,
    //   CARRY8=4): a 96-bit operand forces a 4-DSP cascade stitched by a long CARRY8 chain,
    //   and the source register is ALSO a DSP output (xc*Yr), so the whole wide multiply sits
    //   in one cycle DSP->DSP. Split prod into two positional halves across a register
    //   boundary: prod == $signed(prod[95:48])*2^48 + prod[47:0] (low 48 UNSIGNED, high 48
    //   signed — exact two's-complement positional decomposition). Each partial is a <=48x32
    //   multiply (short carry); register pp_hi/pp_lo, then prod2_r = (pp_hi<<<48)+pp_lo the
    //   next cycle. prod2_r is BIT-IDENTICAL, just one LN-latency cycle later (absorbed by the
    //   done/y_valid handshake — LN is arbitrated hold-until-done).
    reg signed [39:0]  xc     [0:P-1];
    reg signed [95:0]  prod   [0:P-1];
    reg signed [31:0]  grow_0 [0:P-1];
    reg signed [31:0]  grow_r [0:P-1];
    reg signed [127:0] pp_hi  [0:P-1];           // s2: $signed(prod[95:48]) * gamma
    reg signed [127:0] pp_lo  [0:P-1];           // s2: $unsigned(prod[47:0]) * gamma
    reg signed [127:0] prod2_r [0:P-1];          // s2b: registered combined product
    reg                s0v, s1v, s2v, s2bv;
    reg signed [31:0]  xo, go;                   // plain-vector copies
    reg [P*32-1:0]     xword, gword;             // plain-reg copies of a wide bank row
    reg signed [39:0]  xj;
    reg signed [95:0]  pj;
    reg signed [47:0]  pjhi;                     // s2 plain-vector copy: prod[95:48] (signed)
    reg        [47:0]  pjlo;                     // s2 plain-vector copy: prod[47:0] (unsigned)
    reg signed [127:0] prod2;
    reg signed [127:0] p2j;                      // s3 plain-vector copy of prod2_r[lp]
    reg signed [127:0] hj, lj;                   // s2b plain-vector copies of pp_hi/pp_lo

    // ---- FSM ------------------------------------------------------------------
    localparam [3:0]
        S_IDLE=0, S_LOAD=1, S_VAR=2, S_MSB=3, S_SEED=4, S_SEED2=5,
        S_NEWT0=6, S_NEWTA=7, S_NEWTB=8, S_NEWTC=9, S_OUT=10, S_DONE=11,
        S_VAR2=12, S_NEWTA2=13, S_NEWTB2=14, S_NEWTC2=15;
    localparam [4:0] S_NEWTC3=16;
    localparam [4:0] S_NEWT0B=17;         // Fmax: barrel-shift result settle stage
    reg [4:0] state;
    reg signed [127:0] p_ct, p_mq;        // S_VAR product registers
    reg signed [127:0] yy_p;              // Newton stage product registers
    reg signed [191:0] ay_p;
    reg signed [63:0]  yn_r;              // Newton result settle register
    reg signed [63:0]  Y0_r;              // Fmax: registered barrel-shift seed (qsh shifter
    reg                rbit_r;            //   isolated from the Newton Yr*Yr multiply)
    integer wp;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; wptr <= 0; ridx <= 0; oidx <= 0;
            y_valid <= 1'b0; done <= 1'b0; s0v <= 1'b0; s1v <= 1'b0; s2v <= 1'b0; s2bv <= 1'b0;
            sum <= 0; sumxx <= 0; newt <= 0;
        end else begin
            y_valid <= 1'b0;
            done    <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        wptr <= 0; sum <= 0; sumxx <= 0; newt <= 0;
                        acnt <= 0; lv_a <= 1'b0; lv1 <= 1'b0;
                        state <= S_LOAD;
                    end
                end
                // ---- P-wide load: store P lanes; partial sums REGISTERED then folded.
                // 3-stage skew (Fmax @P=16): (A) on valid_in register per-lane squares
                // (xe_r/sq_r); (B) on lv_a sum the registered terms into psum_r/psumxx_r;
                // (C) on lv1 fold into sum/sumxx. Iteration is still 1 row/cycle. --------
                S_LOAD: begin
                    if (valid_in) begin                       // stage A
                        xbank[wptr] <= x_in;                  // one whole P-lane wide row
                        gbank[wptr] <= gamma_in;
                        for (wp = 0; wp < P; wp = wp + 1) begin
                            xl              =  x_in[wp*32 +: 32];   // plain-vector copy
                            xe_r[wp]        <= $signed({{8{xl[31]}}, xl});
                            sq_r[wp]        <= $signed(xl) * $signed(xl);
                        end
                        wptr     <= wptr + 1'b1;
                    end
                    lv_a <= valid_in;
                    if (lv_a) begin                           // stage B
                        psum_r   <= psum;
                        psumxx_r <= psumxx;
                    end
                    lv1 <= lv_a;
                    if (lv1) begin                            // stage C
                        sum   <= sum   + psum_r;
                        sumxx <= sumxx + psumxx_r;
                        acnt  <= acnt + 1'b1;
                        if (acnt == ROWS-1) state <= S_VAR;
                    end
                end
                // ---- algebraic centered sum (exact identity), 2-stage ----------
                S_VAR: begin
                    mean  <= sum >>> D_SH;                               // /D, Q6.25
                    p_ct  <= $signed(sum >>> D_SH) * $signed(sum);       // mean*sum
                    p_mq  <= $signed(sum >>> D_SH) * $signed(sum >>> D_SH); // mean^2
                    state <= S_VAR2;
                end
                S_VAR2: begin
                    cterm = p_ct <<< 1;                                  // 2*mean*sum
                    msq   = D * p_mq;                                    // D*mean^2
                    ssq   <= $signed(sumxx) - cterm + msq;               // centered Q.50
                    state <= S_MSB;
                end
                S_MSB: begin
                    var_q <= ssq >>> D_SH;                               // Q12.50
                    A <= ($signed(ssq >>> D_SH) >>> (VAR_FRAC - A_FRAC)) + $signed(EPS_A);
                    state <= S_SEED;       // msb_c combinational off A; sampled next state
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
                // ---- build Y0 from the registered seed, then 2 pipelined Newtons --
                // Fmax (the worst impl path qsh_reg -> yy_p DSP): the variable shift by qsh
                // is a deep barrel shifter, and Vivado retimes Yr forward into the Newton
                // Yr*Yr multiply, fusing the barrel shifter with the squarer into one
                // combinational cloud. Split into two cycles: S_NEWT0 registers ONLY the
                // qsh barrel-shift into a plain reg Y0_r (a hard boundary retiming cannot
                // cross); S_NEWT0B applies the conditional sqrt2 correction and sets Yr.
                // One extra LN-latency cycle, absorbed by the done/valid handshake. The
                // arithmetic is byte-identical (same shift, same SQRT2Q15 path).
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
                // 2 Newton iterations, each multiply split product-reg | shift (Fmax)
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
                    yn_r <= ynew[63:0];                 // settle one cycle: DSP out -> reg
                    state <= S_NEWTC3;
                end
                S_NEWTC3: begin                          // -> next Yr is a plain reg
                    Yr   <= yn_r;
                    newt <= newt + 1'b1;
                    if (newt == 2'd1) begin ridx <= 0; oidx <= 0; s0v <= 1'b0; s1v <= 1'b0; s2v <= 1'b0; s2bv <= 1'b0; state <= S_OUT; end
                    else state <= S_NEWTA;
                end
                // ---- P-wide pipelined output stream ----------------------------
                S_OUT: begin
                    // stage 0: P centered lanes (x-mean) for row ridx — registered so
                    // the subtract is not chained into the DSP multiply (Fmax)
                    if (ridx < ROWS) begin
                        xword = xbank[ridx];                    // read the wide row once
                        gword = gbank[ridx];
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            xo = xword[lp*32 +: 32];            // lane extract on a plain reg (iverilog-safe)
                            go = gword[lp*32 +: 32];
                            xc[lp]     <= $signed({{8{xo[31]}}, xo}) - mean;
                            grow_0[lp] <= go;
                        end
                        ridx <= ridx + 1'b1;
                        s0v  <= 1'b1;
                    end else s0v <= 1'b0;
                    // stage 1: P prods (x-mean)*Yr
                    s1v <= s0v;
                    if (s0v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            xj = xc[lp];
                            prod[lp]   <= xj * Yr;
                            grow_r[lp] <= grow_0[lp];
                        end
                    end
                    // stage 2: hi/lo partial products of prod*gamma (each <=48x32, short
                    // carry — splits the 96x32 cascade so neither half is the DSP->DSP cloud)
                    s2v <= s1v;
                    if (s1v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            pj    = prod[lp];                   // plain-vector copies
                            go    = grow_r[lp];
                            pjhi  = pj[95:48];                  // high half (signed)
                            pjlo  = pj[47:0];                   // low half  (unsigned)
                            pp_hi[lp]  <= $signed(pjhi) * $signed({{96{go[31]}}, go});
                            pp_lo[lp]  <= $signed({1'b0, pjlo}) * $signed({{96{go[31]}}, go});
                        end
                    end
                    // stage 2b: combine the partials -> prod2_r (a clean wide add, no multiply)
                    //   prod*gamma == (pp_hi <<< 48) + pp_lo  (exact positional recombination)
                    s2bv <= s2v;
                    if (s2v) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            hj = pp_hi[lp];                     // plain-vector copies
                            lj = pp_lo[lp];
                            prod2_r[lp] <= (hj <<< 48) + lj;
                        end
                    end
                    // stage 3: P y = prod2>>>OUT_SH (the wide shift + 64b pack), one row/cycle
                    if (s2bv) begin
                        for (lp = 0; lp < P; lp = lp + 1) begin
                            p2j = prod2_r[lp];                  // plain-vector copy
                            y_out[lp*64 +: 64] <= p2j >>> OUT_SH;
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
endmodule
