// -----------------------------------------------------------------------------
// vec_silu — P-lane parallel SiLU. Applies SiLU to P elements per cycle by
// instantiating P independent copies of the single-lane silu_lut core. Each
// lane is bit-identical to silu_lut, so the whole vector is bit-true to
// fabric/asr_seq/run_silu.silu_q.
//
// Unlike checkpoint C's vec_gelu.sv (which pairs lanes onto gelu_lut2 to
// halve BRAM use), this is P independent unpaired lanes -- correct first,
// not yet BRAM-optimized; a silu_lut2-style paired core is a real, later
// resource-optimization option if needed, not a correctness requirement.
//
// I/O = signed Q4.12 (16-bit, range +-8), P-wide. Streaming: accept a P-wide
// input every cycle (in_valid), emit the corresponding P-wide output 3 cycles
// later (out_valid), matching silu_lut's own pipeline latency exactly.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module vec_silu #(
    parameter integer P = 8
) (
    input  wire                       clk,
    input  wire                       in_valid,
    input  wire signed [16*P-1:0]     x,         // P x Q4.12, lane k = x[16*k +: 16]
    output wire                       out_valid,
    output wire signed [16*P-1:0]     y          // P x Q4.12, lane k = y[16*k +: 16]
);
    genvar k;
    generate
        for (k = 0; k < P; k = k + 1) begin : lane
            silu_lut u_silu (.clk(clk), .x(x[16*k +: 16]), .y(y[16*k +: 16]));
        end
    endgenerate

    reg [2:0] vsr = 3'b000;
    always @(posedge clk)
        vsr <= {vsr[1:0], in_valid};
    assign out_valid = vsr[2];
endmodule
