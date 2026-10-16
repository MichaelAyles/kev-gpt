// -----------------------------------------------------------------------------
// silu_lut — fabric SiLU (x * sigmoid(x)) as an 8192-entry LUT with 3-bit
// linear interpolation. Same structure as checkpoint C's gelu_lut.sv (same
// Q4.12 I/O format, same 8192-entry/3-bit-interp shape, same 3-cycle
// latency) -- the decoder MLP's real gate activation (test_generate_kv.c's
// `silu_(gate, FFN)` inside the SwiGLU stage: h1 = fc1(xn3); value, gate =
// split(h1); gate = silu_(gate); combined = value * gate; h2 = fc2(combined))
// is genuinely new -- gelu_lut.sv/gelu_lut2.sv are GELU-shaped, not SiLU, and
// nothing in kevgpt_seq implements it. I/O = signed Q4.12 (16-bit, range
// +-8). Index = (x + 8) in Q4.12 = x + 0x8000 (unsigned), top 13 bits select
// the entry, low 3 bits are the interp fraction:
//     y = lut[i] + ((lut[i+1]-lut[i]) * f) >>> 3            (all integer, Q4.12)
// Bit-true to fabric/asr_seq/run_silu.silu_q. LUT inits from silu_lut.mem
// ($readmemh, single registered read -> infers a BRAM ROM, init preserved).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module silu_lut (
    input  wire                clk,
    input  wire signed [15:0]  x,      // Q4.12
    output reg  signed [15:0]  y       // Q4.12
);
    (* rom_style = "block" *) reg signed [15:0] lut [0:8191];
    initial $readmemh("silu_lut.mem", lut);

    // stage 0: offset to unsigned index space
    reg [15:0] u;
    always @(posedge clk) u <= x + 16'h8000;

    wire [12:0] idx  = u[15:3];
    wire [2:0]  frac = u[2:0];
    wire [12:0] idx1 = (idx == 13'd8191) ? idx : idx + 1'b1;

    // stage 1: registered LUT reads (BRAM) + carry the fraction
    reg signed [15:0] l0, l1;
    reg [2:0]         f1;
    always @(posedge clk) begin
        l0 <= lut[idx];
        l1 <= lut[idx1];
        f1 <= frac;
    end

    // stage 2: linear interpolation
    wire signed [16:0] diff = l1 - l0;                     // up to +-2^16
    wire signed [19:0] step = diff * $signed({1'b0, f1});  // *frac (0..7)
    always @(posedge clk) y <= l0 + (step >>> 3);
endmodule
