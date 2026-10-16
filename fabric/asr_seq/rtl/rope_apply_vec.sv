// -----------------------------------------------------------------------------
// rope_apply_vec — partial-rotary RoPE for Moonshine-tiny's decoder/encoder
// self-attention, the ONE genuinely new primitive standing between
// checkpoint C's already-proven causal-attention RTL (kv_bank.sv +
// vec_attn_w.sv, both already parameterized on HEAD_DIM and directly
// reusable for ASR's HEAD_DIM=36 self-attention) and actually reusing it
// for ASR (see gen2asr/ASR-ACCELERATOR-OP-SEQUENCE.md's "Two attention
// shapes, one primitive" + "correction to the record" sections — the LLM
// itself has ZERO RoPE hardware, confirmed by grep, so this is not a
// generalization of an existing unit, it's new).
//
// Math (ops.c's rope_table_at + rope_apply, bit-for-bit intent, not
// approximated): only the first ROT_DIM of HEAD_DIM dims rotate, in
// interleaved (2i, 2i+1) pairs; dims [ROT_DIM:HEAD_DIM) pass through
// unchanged.
//   inv_freq[i] = 1/theta^(2i/ROT_DIM)              i = 0..ROT_PAIRS-1
//   angle       = position * inv_freq[i]
//   x1' = x1*cos(angle) - x2*sin(angle)
//   x2' = x2*cos(angle) + x1*sin(angle)
//
// cos(angle)/sin(angle) depend only on (position, i) — both bounded and
// known at synthesis time (TMAX positions x ROT_PAIRS frequencies), so
// this is a PRECOMPUTED ROM (built by fabric/asr_seq/pack_rope.py, same
// $readmemh-a-ROM idiom as gelu_lut.sv), not a runtime sin/cos engine —
// avoids building general trigonometric hardware for a bounded, known
// index space.
//
// I/O format matches vec_attn_w.sv's/kv_bank.sv's own internal qreg/kreg
// convention exactly: the WHOLE head row as HEAD_DIM Q.16 lanes (32 bits
// each, matching P Q.16 lanes elsewhere in this project's own streams),
// not a P-wide streamed interface — those two modules already register a
// full head row internally before consuming it, so this module produces
// the same shape directly, no extra reshaping needed at the integration
// point.
//
// Fixed-point: head lanes Q.16 (signed 32-bit, this project's own VFRAC
// convention). cos/sin ROM entries Q1.15 (signed 16-bit, |value|<=1
// exactly represented, matches gelu_lut.sv-style compact LUT entries).
// Product Q.16 x Q1.15 = Q.31 (48-bit, headroom to spare at these
// magnitudes); rsh_round (round-half-away-from-zero, copied VERBATIM
// from vec_attn_w.sv/sequencer_fast.sv's own function of the same name,
// this project's established rounding convention) shifts back to Q.16.
//
// POST_SCALE_Q16 (default 65536 = 1.0, a true no-op -- every existing/
// future HEAD_DIM=64 caller is byte-for-byte unchanged): compensates a
// real mismatch found integrating this module with vec_attn_w.sv, NOT
// part of RoPE's own math. vec_attn_w.sv hardcodes SCORE_SH=27, derived
// from "2*VFRAC + ISQRT - SCORE_FRAC" with ISQRT=3 -- an assumed
// 1/sqrt(HEAD_DIM)=1/8 scaling via a plain power-of-2 right-shift, exact
// ONLY for checkpoint C's own HEAD_DIM=64. ASR's HEAD_DIM=36 needs
// 1/sqrt(36)=1/6, not a power of 2 -- vec_attn_w.sv's shift-only design
// can't represent that directly. Fix: since score = dot(q,k) and BOTH q
// and k pass through this module, scaling each by sqrt((1/6)/(1/8)) =
// sqrt(4/3) makes their dot product pick up the missing 4/3 factor,
// landing on the correct 1/sqrt(36) once combined with vec_attn_w.sv's
// own built-in 1/8 -- applied here (not folded silently into the cos/sin
// ROM) because it must hit ALL HEAD_DIM lanes uniformly, including the
// [ROT_DIM:HEAD_DIM) pass-through dims RoPE itself never touches.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module rope_apply_vec #(
    parameter integer HEAD_DIM  = 36,
    parameter integer ROT_DIM   = 32,
    parameter integer ROT_PAIRS = 16,           // ROT_DIM / 2
    parameter integer TMAX      = 128,          // max position the ROM covers
    parameter integer POST_SCALE_Q16 = 65536,   // see header -- 65536 = 1.0 = no-op
    parameter               ROM_FILE_COS = "rope_cos.mem",
    parameter               ROM_FILE_SIN = "rope_sin.mem"
) (
    input  wire                        clk,
    input  wire                        start,      // pulse; position sampled here
    input  wire [$clog2(TMAX)-1:0]     position,
    input  wire [HEAD_DIM*32-1:0]      head_in,    // HEAD_DIM Q.16 lanes
    output reg                         done,       // pulses when head_out is valid
    output reg  [HEAD_DIM*32-1:0]      head_out
);
    localparam integer ROM_DEPTH = TMAX * ROT_PAIRS;
    localparam integer ROM_AW    = $clog2(ROM_DEPTH);

    reg signed [15:0] cos_rom [0:ROM_DEPTH-1];
    reg signed [15:0] sin_rom [0:ROM_DEPTH-1];
    initial begin
        $readmemh(ROM_FILE_COS, cos_rom);
        $readmemh(ROM_FILE_SIN, sin_rom);
    end

    function signed [47:0] rsh_round(input signed [47:0] v, input integer s);
        reg signed [47:0] half;
        begin
            if (s <= 0) rsh_round = v <<< (-s);
            else begin
                half = (48'sd1 <<< (s-1));
                if (v >= 0) rsh_round = (v + half) >>> s;
                else        rsh_round = -(((-v) + half) >>> s);
            end
        end
    endfunction

    // ---- one-shot combinational rotate, registered output (1-cycle latency
    // after `start`, matching the "address this cycle, data next cycle"
    // discipline this project's synchronous-read RTL already follows) ------
    integer i;
    reg signed [31:0] x1, x2, xpt;
    reg signed [15:0] c, s;
    reg signed [47:0] p1, p2, p3, p4;
    reg signed [47:0] r1, r2, rpt;
    always @(posedge clk) begin
        done <= 1'b0;
        if (start) begin
            for (i = 0; i < ROT_PAIRS; i = i + 1) begin
                x1 = $signed(head_in[(2*i)*32 +: 32]);
                x2 = $signed(head_in[(2*i+1)*32 +: 32]);
                c  = cos_rom[position * ROT_PAIRS + i[ROM_AW-1:0]];
                s  = sin_rom[position * ROT_PAIRS + i[ROM_AW-1:0]];
                // x1*c - x2*s, x2*c + x1*s -- both in Q.31, rounded back to Q.16
                p1 = x1 * c; p2 = x2 * s; p3 = x2 * c; p4 = x1 * s;
                r1 = rsh_round(p1 - p2, 15);
                r2 = rsh_round(p3 + p4, 15);
                // POST_SCALE_Q16 (see header): Q.16 x Q.16 = Q.32, rsh_round back to Q.16
                r1 = rsh_round(r1 * POST_SCALE_Q16, 16);
                r2 = rsh_round(r2 * POST_SCALE_Q16, 16);
                head_out[(2*i)*32   +: 32] <= r1[31:0];
                head_out[(2*i+1)*32 +: 32] <= r2[31:0];
            end
            for (i = ROT_DIM; i < HEAD_DIM; i = i + 1) begin
                // pass-through dims still get POST_SCALE_Q16 -- must match the
                // rotated dims' scaling exactly, or the per-lane scale would be
                // inconsistent across the head vector (see header).
                xpt = $signed(head_in[i*32 +: 32]);
                rpt = rsh_round(xpt * POST_SCALE_Q16, 16);
                head_out[i*32 +: 32] <= rpt[31:0];
            end
            done <= 1'b1;
        end
    end
endmodule
