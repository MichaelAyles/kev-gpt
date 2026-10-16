// -----------------------------------------------------------------------------
// gelu_wide_vec -- P-lane GELU with WIDE (32-bit/lane) I/O, fixing the
// Q4.12-domain-clipping precision loss conv_front_end_seq.sv's own header
// originally documented as a real, not-yet-fixed limitation: gelu_lut2.sv's
// own domain is a FIXED Q4.12 (+-8 real units); reused UNMODIFIED here for
// in-domain inputs, but for POSITIVE values outside that domain (x > +8 real
// units), GELU(x) -> x extremely fast (Phi(x) is already indistinguishable
// from 1.0 at x=8 to far more decimal places than Q4.12 could represent
// anyway) -- so this module passes the WIDE input straight through instead
// of clipping it to gelu_lut2's own saturated ~+8 output, a real ~1000x
// magnitude error found gating this project's own real-audio conv3 output
// (|.|~1004 real units).
//
// NEGATIVE overflow needs no such correction: GELU(x) -> 0 just as fast as
// x -> -infinity, and gelu_lut2's own boundary value at x=-8 already rounds
// to ~0 -- matching the true asymptote -- so the existing sat16-then-LUT
// path is ALREADY correct on that side; this module changes nothing there.
//
// I/O: P x Q4.12-SCALED values, but 32 bits/lane (not saturated to 16 bits
// at the port boundary -- restoring exactly the precision the old
// sat16-before-vec_gelu path was discarding). Same P-wide streaming
// protocol/latency as vec_gelu.sv (in_valid/x -> out_valid/y, 3 cycles
// later) -- a real wrapper around that proven, UNMODIFIED module, not a
// redesign of GELU itself.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gelu_wide_vec #(
    parameter integer P = 8
) (
    input  wire                    clk,
    input  wire                    in_valid,
    input  wire signed [32*P-1:0]  x,          // P x Q4.12-scaled, 32b/lane (wide, unsaturated)
    output wire                    out_valid,
    output wire signed [32*P-1:0]  y           // P x Q4.12-scaled, 32b/lane
);
    function automatic signed [15:0] sat16;
        input signed [31:0] v;
        begin
            if (v > 32'sd32767) sat16 = 16'sd32767;
            else if (v < -32'sd32768) sat16 = -16'sd32768;
            else sat16 = v[15:0];
        end
    endfunction

    integer lp;
    reg signed [16*P-1:0] lut_x;
    always @(*) begin
        lut_x = {(16*P){1'b0}};
        for (lp = 0; lp < P; lp = lp + 1)
            lut_x[lp*16 +: 16] = sat16($signed(x[lp*32 +: 32]));
    end

    wire lut_ov;
    wire signed [16*P-1:0] lut_y;
    vec_gelu #(.P(P)) u_gelu (
        .clk(clk), .in_valid(in_valid), .x(lut_x), .out_valid(lut_ov), .y(lut_y)
    );

    // 3-cycle delay of the WIDE input, matching vec_gelu.sv's own fixed
    // latency exactly (its own header: P-wide output 3 cycles later) -- a
    // plain pipeline register, no valid-gating needed (only ever read when
    // out_valid is high, which itself only follows a real in_valid pulse).
    reg signed [32*P-1:0] x_d1, x_d2, x_d3;
    always @(posedge clk) begin
        x_d1 <= x; x_d2 <= x_d1; x_d3 <= x_d2;
    end

    integer op;
    reg signed [32*P-1:0] y_word;
    always @(*) begin
        y_word = {(32*P){1'b0}};
        for (op = 0; op < P; op = op + 1) begin
            if ($signed(x_d3[op*32 +: 32]) > 32'sd32767)
                y_word[op*32 +: 32] = x_d3[op*32 +: 32];                            // GELU(x)->x, x>+8
            else
                y_word[op*32 +: 32] = {{16{lut_y[op*16+15]}}, lut_y[op*16 +: 16]};   // in-domain or very-negative
        end
    end
    assign out_valid = lut_ov;
    assign y = y_word;
endmodule // gelu_wide_vec
