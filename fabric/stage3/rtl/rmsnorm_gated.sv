// rmsnorm_gated.sv — Mamba-2 (optionally gated) RMSNorm (doc 9; reference =
// model/mamba2_fixed.rmsnorm_fixed at fabric LUT/rsqrt resolution).
//
//   gated=1:  g[i] = sat16( rnd( y[i] * silu(z[i]) >>> 12 ) )     (Q3.12)
//   gated=0:  g[i] = y[i]                (silu operand = 1.0 Q3.12; exact)
//   out[i]   = sat16( rnd( g[i] * rsqrt(mean(g^2)+eps) * gamma[i] >>> 40 ) )
//
// Formats (contract):
//   gated=1:  y INT16 Q4.11 (scan out ~10), z INT16 Q6.9 (gate ~28) — both
//             widened from Q3.12 which saturated in the full engine. RMSNorm
//             is scale-invariant, so gate_sh/A/out_sh carry the format below.
//   gamma       INT16 Q1.14
//   short_len=1: y is INT16 Q6.9 (the engine's Q6.19-residual norm view),
//                256 elements; eps and the output shift are rescaled so the
//                output is still Q3.12 with true eps=1e-5 (see S_A / out_sh_n)
//   g           INT16 Q3.12 (rounded, saturated gated product)
//   sum(g^2)    Q6.24 accumulate (48-bit)
//   A = mean+eps Q.26  =  (sum >>> (log2(D)-2)) + round(1e-5*2^26)
//   rsqrt       Q.26 — 64-entry seed ROM ("seed.mem") + 2 Newton steps,
//               the silicon-proven layernorm.sv / layernorm_vec.sv datapath
//               WITHOUT the mean subtraction (RMS, not variance)
//   out         INT16 Q3.12: (g*Yr*gamma + 2^39) >>> 40, saturated
//
// SiLU: 256-entry LUT over [-4,4) Q6.9 input — the conv_silu.sv indexing
// contract at Q6.9: idx = top 8 bits of (z+2048), floor; tails y=x for z>=4,
// 0 for z<-4. silw is Q6.9 (identity tail carries z up to ~28; LUT rows are
// Q3.12 -> >>3). LUT loaded via write port.
//
// Scalar sequential (one lane): ~D cycles gate pass + rsqrt + ~D cycles
// output pass. Simplicity over speed — norm is a tiny slice of the token.
`timescale 1ns / 1ps
`default_nettype none

module rmsnorm_gated #(
    parameter integer D   = 512,     // vector length (power of 2)
    parameter integer QF  = 12,      // Q3.12 activations
    parameter integer RDP = 4        // output read lanes/cycle
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,      // pulse; y/z/gamma/lut written
    input  wire                    gated,      // 1: g=y*silu(z)  0: g=y
    input  wire                    short_len,  // 1: operate on D/2 elements
                                               // (the engine's pre/final norms
                                               // are 256-wide; gated norm 512)
    output reg                     done,

    // per-token inputs, written before start
    input  wire                    wr_y,
    input  wire [$clog2(D)-1:0]    wr_y_addr,
    input  wire signed [15:0]      wr_y_data,
    input  wire                    wr_z,
    input  wire [$clog2(D)-1:0]    wr_z_addr,
    input  wire signed [15:0]      wr_z_data,

    // gamma (once), Q1.14
    input  wire                    wr_g,
    input  wire [$clog2(D)-1:0]    wr_g_addr,
    input  wire signed [15:0]      wr_g_data,

    // silu LUT init (once), Q3.12 values
    input  wire                    wr_lut,
    input  wire [7:0]              wr_lut_addr,
    input  wire signed [15:0]      wr_lut_data,

    // rsqrt seed ROM init (once), 20-bit Q1.16 seed values — loaded over AXI so
    // the seed table does NOT depend on $readmemh baking into the bitstream
    // (on silicon the baked init read back as 0 -> rsqrt diverged -> norm
    // exploded). AXI-written after reset, before the first token; wins over the
    // sim-only initial below.
    input  wire                    wr_seed,
    input  wire [5:0]              wr_seed_addr,
    input  wire [19:0]             wr_seed_data,
    // DEBUG: read the seed table back (proves on silicon that the AXI load
    // landed, independent of whether the $readmemh init baked).
    input  wire [5:0]              rd_seed_addr,
    output wire [19:0]             rd_seed_data,

    input  wire [$clog2(D)-1:0]    rd_o_addr,
    output wire signed [15:0]      rd_o_data,

    // WIDE read: RDP consecutive outputs from `rd_ow_base` in one cycle, so the
    // sequencer quant streams (S_QIN/S_QOUT) can consume RDP/cycle. Combinational.
    input  wire [$clog2(D)-1:0]    rd_ow_base,
    output wire [RDP*16-1:0]       rd_ow_data
);
    localparam integer LOG2D  = $clog2(D);
    localparam integer A_FRAC = 26;
    localparam integer Y_FRAC = 26;
    localparam integer G_FRAC = 14;                    // gamma Q1.14
    localparam integer A_SH   = LOG2D + 2*QF - A_FRAC; // sum Q.2QF -> mean Q.26
    localparam integer OUT_SH = Y_FRAC + G_FRAC;       // 40: Q.(QF+26+14)->Q.QF
    localparam integer SEED_IDX_BITS = 6;
    localparam integer SEED_OUT_FRAC = 16;
    localparam [63:0]  EPS_A    = 64'd671;             // round(1e-5 * 2^26)
    localparam [31:0]  SQRT2Q15 = 32'd46341;           // round(sqrt(2)*2^15)
    localparam [127:0] ONE_P5   = 128'd3 <<< (2*Y_FRAC - 1);

    // ---- storage -------------------------------------------------------
    reg signed [15:0] yin  [0:D-1];
    reg signed [15:0] zin  [0:D-1];
    reg signed [15:0] grom [0:D-1];
    reg signed [15:0] lut  [0:255];
    reg signed [15:0] gbuf [0:D-1];       // gated product g (Q3.12)
    reg signed [15:0] obuf [0:D-1];

    assign rd_o_data = obuf[rd_o_addr];

    // RDP-wide combinational read (whole-element indexed, iverilog-safe). base+g
    // stays in-bounds: S_QIN reads 256, S_QOUT 512 — both RDP-multiples.
    genvar gr;
    generate for (gr = 0; gr < RDP; gr = gr + 1) begin : g_ordw
        assign rd_ow_data[gr*16 +: 16] = obuf[rd_ow_base + gr[$clog2(D)-1:0]];
    end endgenerate

    // ---- seed ROM (64 x Q1.16) — same values as layernorm.sv -------------
    // Runtime-loadable RAM (was a rom_style="block" ROM): written via wr_seed
    // over AXI. The baked $readmemh init did not survive into the bitstream on
    // silicon (the .mem was off the synth run-dir path -> seed_rom read 0 ->
    // Newton rsqrt diverged -> norm railed). The AXI load is now the source of
    // truth; the initial is a sim convenience / harmless fallback only.
    (* ram_style = "distributed" *) reg [19:0] seed_rom [0:63];
    initial $readmemh("seed.mem", seed_rom);
    reg [19:0] seed_dbg_q;
    assign rd_seed_data = seed_dbg_q;

    always @(posedge clk) begin
        if (wr_y)    yin [wr_y_addr]    <= wr_y_data;
        if (wr_z)    zin [wr_z_addr]    <= wr_z_data;
        if (wr_g)    grom[wr_g_addr]    <= wr_g_data;
        if (wr_lut)  lut [wr_lut_addr]  <= wr_lut_data;
        if (wr_seed) seed_rom[wr_seed_addr] <= wr_seed_data;
        seed_dbg_q <= seed_rom[rd_seed_addr];
    end

    // ---- FSM -------------------------------------------------------------
    localparam [4:0]
        S_IDLE=0, S_GATE=1, S_GDRAIN=2, S_A=3, S_MSB=4, S_SEED=5, S_SEED2=6,
        S_NEWT=7, S_NEWTB=8, S_NEWTC=9, S_OUT=10, S_DONE=11;
    reg [4:0] state;
    reg       g_mode;                      // sampled `gated` for this pass

    // ---- gate pass: 2-stage pipeline, one element/cycle ------------------
    reg [LOG2D-1:0] ci;
    wire issue = (state == S_GATE);
    reg               v1, v2;
    reg [LOG2D-1:0]   c1, c2;
    reg signed [15:0] y1, z1;
    reg signed [31:0] prod2;               // y*silu Q6.24

    // silu comb off z1 — z is INT16 Q6.9 in the engine (gate z reaches ~28,
    // which the old Q3.12 z clipped to +-8: the full-engine gate bug). LUT
    // domain [-4,4): 4.0 = 2048 in Q6.9, index = top 8 bits of (z+2048)>>4.
    // silw is Q6.9: identity tail returns z1 (Q6.9) up to ~28; LUT rows are
    // Q3.12 silu values -> >>3 (rnd) to Q6.9.
    wire signed [16:0] zb   = z1 + $signed(17'sd2048);
    wire        [7:0]  sidx = (zb < 0) ? 8'd0 :
                              (zb > $signed(17'sd4095)) ? 8'd255 : zb[11:4];
    wire signed [15:0] silw = (z1 >= $signed(16'sd2048)) ? z1 :
                              (z1 < -$signed(16'sd2048)) ? 16'sd0 :
                              ($signed(lut[sidx]) + 16'sd4) >>> 3;
    // ungated bypass: operand 1.0 (Q3.12) -> (y<<QF + 2^(QF-1))>>>QF == y exact
    wire signed [15:0] silsel = g_mode ? silw : 16'sd4096;

    // GATED-PRODUCT HEADROOM (GSH): g = y*silu(z) reaches |g|~270 (real) on the
    // per-layer gated norm — the gate values are large (z>=4 -> silu(z)=z up to
    // ~28, y up to ~10). g stored as g_true*2^(12-GSH) so a few big channels
    // don't collapse mean(g^2)/inflate rsqrt. RMSNorm is SCALE-INVARIANT in g,
    // so storing g>>GSH (real range +-8*2^GSH = +-512) cancels out of
    // out = g*rsqrt(mean g^2)*gamma exactly, with the A and output shifts
    // compensated below. Only gated mode shifts (ungated short/final norms
    // have bounded inputs, no saturation).
    localparam integer GSH = 6;
    // gated: prod2 = yin(Q4.11) * silsel(Q6.9) has frac 20; store g at frac
    // (12-GSH)=6 -> shift 14. ungated (short pre/final norm): yin(Q6.9) * 1.0
    // (Q3.12=4096) frac 21 >> QF=12 -> g = yin (Q6.9) exact.
    wire [5:0] gate_sh = g_mode ? 6'd14 : 6'd12;     // 14 gated / 12 ungated

    // stage-2 comb: round >>> gate_sh, saturate, square
    wire signed [31:0] grnd = $signed(32'sd1) <<< (gate_sh - 1);
    wire signed [20:0] pre = (prod2 + grnd) >>> gate_sh;
    wire signed [15:0] gw  = (pre > $signed(21'sd32767))  ? 16'sd32767 :
                             (pre < -$signed(21'sd32768)) ? -16'sd32768 :
                             pre[15:0];
    wire signed [31:0] gsq = gw * gw;                         // Q6.24

    reg signed [47:0] sumsq;                                  // sum g^2 Q6.24

    // ---- rsqrt registers (layernorm.sv datapath, minus the mean) ---------
    reg signed [63:0] A, Yr, Y0;
    reg [1:0]  newt;
    reg [7:0]  msb;
    reg [5:0]  seed_idx;
    reg signed [8:0]  half, qsh;
    reg               rbit;
    reg signed [63:0] seed_shifted;
    reg [19:0]        seed_val;
    reg signed [127:0] yy, ayy, term;
    reg signed [191:0] ynew;

    integer b;
    reg [7:0] msb_c;
    always @(*) begin
        msb_c = 8'd0;
        for (b = 0; b < 63; b = b + 1)
            if (A[b]) msb_c = b[7:0];
    end

    // ---- output pass: 3-stage pipeline, one element/cycle ----------------
    reg [LOG2D:0]     oi;
    reg               ov1, ov2, ov3;
    reg [LOG2D-1:0]   oc1, oc2, oc3;
    reg signed [15:0] go1, gm1, gm2;
    reg signed [95:0]  prod_gy;            // g*Yr        Q.(QF+26)
    reg signed [111:0] prod_gyg;           // g*Yr*gamma  Q.(QF+40)
    // short mode: g is Q6.9 and Yr is true-scale (8x smaller than the Q3.12
    // path), so shift 3 less to land the output back in Q3.12 exactly:
    // y*2^9 * Yr*2^26 * gamma*2^14 >>> 37 = y*gamma*rsqrt * 2^12.
    // gated: g stored 2^GSH smaller -> Yr is 2^GSH larger (A shifted below), so
    // g*Yr keeps scale; shift OUT_SH-GSH to land Q3.12. short: -3 (Q6.9 g).
    wire [5:0] out_sh_n = short_len ? OUT_SH - 3 :
                          g_mode    ? OUT_SH - GSH : OUT_SH;
    wire signed [111:0] ornd = 112'sd1 <<< (out_sh_n - 1);
    wire signed [112:0] osh  = (prod_gyg + ornd) >>> out_sh_n;
    wire signed [15:0]  osat = (osh > $signed(113'sd32767))  ? 16'sd32767 :
                               (osh < -$signed(113'sd32768)) ? -16'sd32768 :
                               osh[15:0];

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            state <= S_IDLE; v1 <= 0; v2 <= 0;
            ov1 <= 0; ov2 <= 0; ov3 <= 0; newt <= 0;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    ci <= '0; sumsq <= 48'sd0; g_mode <= gated;
                    v1 <= 1'b0; v2 <= 1'b0;
                    state <= S_GATE;
                end
                S_GATE: begin
                    if (ci == (short_len ? D/2-1 : D-1)) state <= S_GDRAIN;
                    ci <= ci + 1'b1;
                end
                S_GDRAIN: if (!v1 && !v2) state <= S_A;
                // ---- A = mean(g^2) + eps  (Q.26): the LN core minus mean --
                // short mode inputs are Q6.9 (engine pre/final norms feed the
                // Q6.19 residual view): sum of 256 Q6.9 squares IS mean*2^26
                // (2^18*256), so eps lands at true scale with NO shift. The
                // old A_SH-1 shift left eps effectively 64x too large — a
                // -5..8% norm shrink on small-residual layers (gate6 FAIL).
                S_A: begin
                    // gated: gw is 2^GSH smaller -> sumsq 2^(2*GSH) smaller, so
                    // LEFT-shift (2*GSH - A_SH) to restore mean*2^26 in Q.26.
                    A <= (short_len ? $signed(sumsq)
                          : g_mode  ? ($signed(sumsq) <<< (2*GSH - A_SH))
                                    : ($signed(sumsq) >>> A_SH))
                         + $signed(EPS_A);
                    state <= S_MSB;
                end
                S_MSB: begin
                    msb  <= msb_c;
                    half <= A_FRAC - $signed({1'b0, msb_c});
                    if (msb_c >= SEED_IDX_BITS)
                        seed_idx <= (A >> (msb_c - SEED_IDX_BITS)) & 6'h3F;
                    else
                        seed_idx <= (A << (SEED_IDX_BITS - msb_c)) & 6'h3F;
                    state <= S_SEED;
                end
                S_SEED: begin
                    seed_val <= seed_rom[seed_idx];
                    if (half[8] == 1'b0) begin
                        qsh  <= half >>> 1;
                        rbit <= half[0];
                    end else begin
                        qsh  <= -(((-half) + 1) >>> 1);
                        rbit <= half[0];
                    end
                    state <= S_SEED2;
                end
                S_SEED2: begin
                    seed_shifted = $signed({44'd0, seed_val}) <<< (Y_FRAC - SEED_OUT_FRAC);
                    if (qsh[8] == 1'b0) Y0 = seed_shifted <<< qsh;
                    else                Y0 = seed_shifted >>> (-qsh);
                    if (rbit) Y0 = (Y0 * $signed({33'd0, SQRT2Q15})) >>> 15;
                    Yr   <= Y0;
                    newt <= 0;
                    state <= S_NEWT;
                end
                // Newton y <- y*(1.5 - 0.5*a*y*y), 3 stages (layernorm.sv verbatim)
                S_NEWT:  begin yy  <= Yr * Yr;              state <= S_NEWTB; end
                S_NEWTB: begin ayy <= (A * yy) >>> A_FRAC;  state <= S_NEWTC; end
                S_NEWTC: begin
                    term = ONE_P5 - (ayy >>> 1);
                    ynew = (Yr * term) >>> (2*Y_FRAC);
                    Yr   <= ynew[63:0];
                    newt <= newt + 1'b1;
                    if (newt == 2'd1) begin
                        oi <= '0; ov1 <= 0; ov2 <= 0; ov3 <= 0;
                        state <= S_OUT;
                    end else state <= S_NEWT;
                end
                S_OUT: begin
                    // o-stage 0: fetch g / gamma
                    if (oi < (short_len ? D/2 : D)) begin
                        go1 <= gbuf[oi[LOG2D-1:0]];
                        gm1 <= grom[oi[LOG2D-1:0]];
                        oc1 <= oi[LOG2D-1:0];
                        ov1 <= 1'b1;
                        oi  <= oi + 1'b1;
                    end else ov1 <= 1'b0;
                    // o-stage 1: g * Yr
                    ov2 <= ov1; oc2 <= oc1; gm2 <= gm1;
                    if (ov1) prod_gy <= go1 * Yr;
                    // o-stage 2: * gamma
                    ov3 <= ov2; oc3 <= oc2;
                    if (ov2) prod_gyg <= prod_gy * gm2;
                    // o-stage 3: round >>> 40, sat, write
                    if (ov3) begin
                        obuf[oc3] <= osat;
                        if (oc3 == (short_len ? D/2-1 : D-1)) state <= S_DONE;
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase

            // gate pipeline (runs during S_GATE/S_GDRAIN)
            v1 <= issue; c1 <= ci;
            y1 <= yin[ci]; z1 <= zin[ci];
            v2 <= v1; c2 <= c1;
            prod2 <= y1 * silsel;                              // Q6.24
            if (v2) begin
                gbuf[c2] <= gw;
                sumsq    <= sumsq + gsq;
            end
        end
    end

endmodule

`default_nettype wire
