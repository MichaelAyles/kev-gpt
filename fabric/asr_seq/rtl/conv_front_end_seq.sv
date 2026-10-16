// -----------------------------------------------------------------------------
// conv_front_end_seq -- Stage 1 top-level FSM: chains conv1 -> tanh_ ->
// groupnorm1 -> conv2 -> gelu -> conv3 -> gelu (gen2asr/
// ASR-ACCELERATOR-OP-SEQUENCE.md Stage 1 table) into ONE `go` pulse,
// reusing every block already gated standalone: conv1d_seq.sv (x3, real
// shapes below), vec_tanh.sv, groupnorm1_vec.sv, vec_gelu.sv (x2, from
// fabric/stage3/rtl/ -- unmodified, already proven by encoder_block_seq.sv's
// own MLP). No new arithmetic anywhere in this file, only sequencing +
// format glue between blocks. "permute" (the op-sequence table's own last
// Stage 1 step) needs no dedicated logic at all: every block here already
// stores/streams t-major/channel-minor (row = t*channels/P + channel-row),
// the SAME convention the encoder's own per-position LayerNorm input
// expects, by construction.
//
// ---- Per-boundary wiring strategy (two kinds) ------------------------------
// Some sub-block inputs are FSM-state-gated (only respond to valid data
// during a specific internal state): groupnorm1_vec.sv's own valid_in only
// does anything during its S_LOAD state. Those need an explicit top-level
// BUFFER + a dedicated re-drive FSM state (this file has exactly ONE such
// boundary: tanh -> groupnorm1, via bank_tanh below).
//
// Every other sub-block input is state-INDEPENDENT: conv1d_seq.sv's own
// xt_we write (own always block, no FSM gating at all -- see that file's
// header) and vec_tanh.sv/vec_gelu.sv's own in_valid/x (pure combinational
// pipes, no internal state machine). Those boundaries are wired DIRECTLY,
// combinationally, with only a format-conversion function in between --
// conv1->tanh, conv2->gelu1->conv3, conv3->gelu2->(this module's own
// output), and groupnorm1->conv2 (groupnorm1_vec.sv's own y_valid stream
// IS the last thing before its `done` pulses, so no extra downstream
// pipeline lag to buffer against -- see below).
//
// A direct-wired boundary still needs the TOP-LEVEL FSM to know when the
// downstream block has consumed everything, before triggering the NEXT
// stage's own go/start: for a boundary that ends at a LUT (tanh/gelu),
// that LUT's own out_valid pulses lag the upstream producer's y_valid by
// its own fixed pipeline latency (3 cycles), so this file counts the LUT's
// OWN out_valid pulses up to the expected row count (cnt_tanh, cnt_c3xt,
// cnt_ge2) rather than trusting the upstream module's `done` (which fires
// too early relative to the LUT's own drain). groupnorm1_vec.sv's own
// `done` needs no such counter: its S_OUT->S_DONE transition happens the
// cycle AFTER its own last y_valid row (this file writes conv2's xt_we
// combinationally on that same y_valid cycle, so the write is already
// registered by the time `done` pulses).
//
// ---- Format glue --------------------------------------------------------
// Every conv1d_seq.sv instance's own per-row (mant,exp) dequant table
// (preloaded once via c1_dq_we/c2_dq_we/c3_dq_we -- see conv1d_seq.sv's own
// header for why per-row, not a single shared shift) is chosen by the
// Python packer so each raw INT32 GEMV accumulator, per OUTPUT CHANNEL,
// lands DIRECTLY in Q4.12 scale when the next stage is a LUT (conv1->tanh,
// conv2->gelu1, conv3->gelu2) -- no further shift needed at those 3
// boundaries. conv1->tanh still uses a plain sat16() clip (int32->int16,
// the SAME idiom encoder_block_seq.sv's own sat16() uses before its own
// vec_gelu call) -- safe because tanh saturates well inside Q4.12's +-8
// range. The two GELU boundaries (conv2->gelu1, conv3->gelu2) go through
// gelu_wide_vec.sv instead of a plain sat16()+vec_gelu (see the section
// below for why: GELU does NOT saturate the same way, so a REAL widening
// fix was needed, not just format bookkeeping). tanh_lut.sv's own Q4.12
// output needs an EXPLICIT widening shift (<<<13, exact -- Q4.12 has 12
// fractional bits, Q6.25 has 25, the difference is a power of 2 so this
// loses no precision) wherever the NEXT stage wants Q6.25: tanh->
// groupnorm1's x_in, and gelu2->this module's own final y_data (which
// feeds the encoder's own first LayerNorm, itself Q6.25) -- the latter via
// widen1312_sat (32-bit input, since gelu_wide_vec.sv's own output no
// longer fits in 16 bits; "_sat" because that boundary needs to SATURATE,
// not wrap -- see below). Wherever the next stage is another conv1d_seq
// call (groupnorm1->conv2, gelu1->conv3), an actquant()-style shift+clip
// to INT8 is unavoidable (GEMV activations are always INT8) -- gn_ashift/
// ge1_ashift are runtime ports, chosen by the Python packer the same way
// the per-row dequant tables are.
//
// ---- Q4.12 GELU precision: found gating this file, then fixed -----------
// pack_conv_front_end.py's own real-audio run (torch.manual_seed(0),
// L=3000) originally showed conv3's raw pre-GELU output reaching |.|~1004
// in real units -- vastly outside Q4.12's fixed +-8 range -- with the
// GELU boundaries (conv2->gelu1, conv3->gelu2) simply sat16()-clipping
// before the LUT, same as tanh's own conv1->tanh boundary. That's benign
// for tanh (saturates to +-1 well inside +-8) but NOT for GELU, which has
// no such saturation for large positive inputs -- clipping there was a
// real ~1000x magnitude error on this project's own real-audio test,
// dragging the informational cosine against the real float front-end
// output down to ~0.72 (bit-exactness against this file's own Python
// reference held throughout -- that's this project's own gate bar, not
// matching the real unquantized model -- but a reference that's THIS far
// from the real float computation isn't a reference worth trusting for
// its own sake). moonshine-tiny's real firmware sidesteps the whole
// problem by running the conv front-end UNQUANTIZED (gen2asr's own
// test_generate_kv_i8.c) -- no existing real precedent to match.
//
// Fixed via gelu_wide_vec.sv (this dir): a WIDE (32-bit/lane) wrapper
// around vec_gelu.sv/gelu_lut2.sv, reused completely UNMODIFIED for
// in-domain values, but passing the wide input straight through instead
// of clipping it whenever x > +8 real units -- GELU(x)->x that fast on
// the positive side (Phi(x) already indistinguishable from 1.0 at x=8 to
// far more precision than Q4.12 could represent anyway), the same way
// tanh's own saturation already made clipping harmless on ITS boundary.
// The negative side needed no such fix: GELU(x)->0 just as fast as
// x->-infinity, and gelu_lut2.sv's own boundary value at x=-8 already
// rounds to ~0, so sat16-then-LUT was already correct there. Gated
// standalone first (GELU_WIDE_VERDICT bitexact=1, mismatches=0/2048, in
// and out of domain, both signs) before wiring in here -- see
// run_gelu_wide.py. End-to-end RTL-vs-Python STAYS bit-exact after this
// fix (CONV_FRONT_END_VERDICT bitexact=1, mismatches=0/216) -- the fix
// changes what value gets computed, not whether RTL matches its own
// reference.
//
// ---- Q6.25 final-output range: the GELU fix moved the bottleneck here,
// then this was fixed too (saturate, not wrap) -- but a real, DEEPER
// contributor was found underneath and is NOT fixed -----------------------
// With GELU no longer clipping (the fix above), gelu2's own real output
// can correctly reach magnitude ~1000 on this project's own real-audio
// test -- but that overflows Q6.25's own ~+-64 representable range at the
// FINAL widen step. First tried wrapping (ordinary 32-bit truncation,
// matching decoder_block_seq.sv/encoder_block_seq.sv/output_head_seq.sv's
// own xres_bank precedent) -- informational cosine against the real
// float front-end output actually got WORSE (~0.72 -> ~0.33), since
// wrapping turns a bounded, constant-ish clipping error into effectively
// RANDOM noise (a value can wrap to any bit pattern, even flip sign).
// Switched to widen1312_sat() instead -- SATURATE at
// Q6.25's own representable ceiling (+-262143 pre-shift, i.e. +-~64 real
// units) rather than wrap. Chosen deliberately for this NEW boundary
// (unlike the older xres_bank precedent, which was never revisited, not
// concluded to be worse): a saturated value is at least bounded and
// sign-correct, the same "sat" idiom sat16()/act_quantize's own clip-
// with-WARNING convention already uses elsewhere in this project.
// Recovered PART of the loss: cosine -> ~0.58 (better than wrapping's
// ~0.33, still well below every other block's own 0.99+). End-to-end
// RTL-vs-Python stays bit-exact throughout both attempts (CONV_FRONT_END_
// VERDICT bitexact=1, mismatches=0/216).
//
// The REMAINING gap traced to a genuinely different, deeper cause, found
// while debugging this: real INT8-quantization noise compounding across
// THREE cascaded activation-quantization boundaries (audio->conv1,
// groupnorm1->conv2, gelu1->conv3) before conv3's own small kernel
// (KW3=3) amplifies it further for some output positions. Confirmed
// directly: one real element has TRUE conv3-then-gelu value 21.19 (real
// units), but this file's own quantized pipeline computed 92.97 at that
// SAME element -- a real ~4.4x error, not a clipping/wrapping artifact
// (GELU is near-identity there, x>>0, so the error is inherited straight
// from conv3's own raw dequantized output, not introduced by GELU or the
// final widen).
//
// ---- Partial fix: stop wasting INT8 headroom (conv1d_ref.py's own
// choose_ashift, pack_conv_front_end.py's own choose_rshift_from_max) ----
// Both of this pipeline's activation-quantization scale searches
// defaulted to `target_max=100`, not INT8's real ceiling of 127 -- ~21%
// of the format's own dynamic range was being left unused at EVERY
// activation-quantization boundary in this chain, for no reason (the
// search is a floor-style/round-nearest shift choice that provably can't
// overshoot 127 even at target_max=127 -- see choose_ashift's own updated
// docstring). Raising both to 127 recovered a real bit of precision at
// gelu1->conv3 specifically (`ge1_shift` 12->11) and at conv1's own audio
// quantization (`ashift` 0->1 in the standalone conv3 gate). The same
// concrete element above improved from 92.97 to 50.75 (still ~2.4x off
// the true 21.19, better than 4.4x but not fixed) -- and the whole-chain
// informational cosine rose from ~0.58 to ~0.83. All of conv1/conv2/
// conv3's own standalone gates AND this file's own end-to-end gate stayed
// bit-exact throughout (a runtime shift/scale choice can't affect
// bit-exactness by construction, only the resulting precision).
//
// ---- Second fix: per-channel (not per-matrix) weight scales -------------
// The above closed part of the gap, not all of it -- a real ~2.4x error
// remained on the same element, needing a materially different
// quantization strategy for the gelu1->conv3 boundary specifically:
// per-channel rather than per-matrix WEIGHT scales. conv1d_seq.sv's own
// dequant step is now a per-row (mant,exp) table via vec_dequant.sv
// (checkpoint C's real, unmodified per-row dequant -- the SAME module
// output_head_seq.sv's own lm_head GEMV already uses), replacing the
// single shared runtime dq_shift every conv1d_seq.sv call used before --
// see that file's own header for the full design (why per-row weight
// scale needs a matching per-row dequant, the new dq_we preload port,
// the gi/ro-separated drain FSM vec_dequant.sv's own 3-cycle pipeline
// needs). conv1d_ref.quantize_weight_per_row (one wshift per OUTPUT
// channel) replaces quantize_weight_per_matrix at every one of this
// pipeline's 3 conv weight quantizations.
//
// Result: a real, if MODEST, further improvement -- whole-chain cosine
// rose ~0.83 -> ~0.85 -- but the SAME tracked element barely moved (50.75
// -> 53.996, no closer to the true 21.19). Honest, but not the whole
// story: per-row WEIGHT scaling only helps output channels whose own
// weight dynamic range differs from the matrix's own max, it can't touch
// ACTIVATION-side quantization noise, and gelu1's own INT8 activation
// feeding conv3 was STILL a single shared scale (ge1_ashift) across all
// 576 input channels. Still bit-exact throughout every one of
// conv1/conv2/conv3's own standalone gates and this file's own
// end-to-end gate.
//
// ---- Third fix: calibrate the shared shift instead of just sizing it
// to never clip -- a real, worthwhile gain, but NOT the final answer ----
// Every activation shift in this pipeline (gn_shift, ge1_shift) was
// sized by `choose_rshift_from_max`: the SMALLEST shift keeping the
// ABSOLUTE MAX under INT8's ceiling -- "never clip a single element,"
// the textbook-safe choice. Tried, first, the standard alternative: pick
// the shift minimizing INPUT round-trip (quantize-then-dequantize) MSE
// instead -- picked the EXACT SAME shift as max-based here, because
// gelu1's own real distribution has a long tail that dominates a plain
// MSE sum. Fixed by grid-searching the shift directly against THIS
// STAGE's own real FP32 output instead (re-quantize, re-run the GEMV/
// dequant chain, compare to the real PyTorch reference, keep the lowest-
// MSE candidate) -- standard INT8 calibration practice. ge1_shift moved
// 11->10 (one bit finer, accepting rare hard clipping): whole-chain
// cosine rose ~0.85 -> ~0.96, still bit-exact throughout.
//
// ---- Fourth fix: TRUE per-channel activation quantization -- this is
// what actually closed it, and it needed NO new GEMV hardware ----------
// The earlier "per-channel ACTIVATION scale isn't available without a
// genuinely different architecture" claim (written at the per-row-WEIGHT
// stage above) was WRONG -- found while chasing the remaining ~0.04 gap.
// A GEMV's raw accumulator sum_k w_int[row,k]*x_int[k] only represents a
// uniformly-rescaled true dot product if EVERY term shares the same
// combined scale -- true whenever the activation scale is constant across
// k, which per-channel activation quantization breaks BY DEFINITION
// (ashift now varies with k's own channel). But the fix doesn't need the
// GEMV core to know anything about per-channel scale at all: pre-divide
// each WEIGHT COLUMN by exactly the same 2^ashift[channel(k)] the
// matching activation column was multiplied by
// (conv1d_ref.rescale_weight_cols_per_channel), BEFORE the existing
// per-row weight quantization search -- the per-channel correction
// cancels out of the algebra completely, leaving gemvy[row] == the true
// dot product * 2^wshift[row] alone, no ashift term anywhere in the
// final dequant scale (worked through in full in that function's own
// docstring). This makes per-channel activation scale a pure WEIGHT-
// QUANTIZATION-TIME trick, exactly like per-row weight scale was --
// gemv_banked_resident_vec.sv/vec_dequant.sv stay completely unmodified.
//
// What DID need a real (contained) RTL change: the ACTQUANT step itself
// (this file's own gn_xt_word/ge1_xt_word logic, which converts
// groupnorm1's/gelu1's own streamed output to INT8 before feeding
// conv2/conv3) used a single shared gn_ashift/ge1_ashift scalar port --
// per-channel precision needs a per-channel TABLE there too, or the
// activation itself never gets quantized any better regardless of how
// smart the weight-side correction is. Added: gn_ash_we/gn_ash_data and
// ge1_ash_we/ge1_ash_data preload tables (CROWS_GN / MROWS2 rows, same
// auto-incrementing-pointer convention as every other preload table
// here), plus gn_row/ge1_row -- plain wrapping counters tracking which
// channel-row is CURRENTLY streaming out of gn_yv/ge1_ov (groupnorm1's
// and, by pass-through, conv2's own output order is t-major/channel-row-
// minor, so row-index-modulo-CROWS IS the channel-row). No change to
// conv1d_seq.sv, groupnorm1_vec.sv, vec_dequant.sv, or gelu_wide_vec.sv
// at all -- gemv_banked_resident_vec.sv's own K-uniform-scale limitation
// was real, just irrelevant here, since the correction lives entirely on
// the weight-quantization side.
//
// Two real bugs found getting this bit-exact (both in the NEW code, not
// in anything reused):
// 1. The per-channel table's own port ('shift', fed straight to actq())
//    is a RIGHT-SHIFT applied to the raw Q.22/Q4.12 integer -- the first
//    attempt wrote the "ashift" MULTIPLY-EXPONENT quantize_act_per_
//    channel_from_int computes directly into that table instead of the
//    matching right-shift (target_frac - ashift). Compiled and ran fine;
//    not remotely bit-exact.
// 2. Even after fixing (1), still not bit-exact: the per-channel
//    quantization itself used round-to-nearest-on-real-values (the
//    natural thing to reach for), but actq() is a plain `x >>> shift` --
//    floor/truncate-toward-negative-infinity, NOT round-half-*. Ended up
//    off by exactly 1 LSB for roughly half of all activation elements
//    (any element whose true fractional part was >=0.5), and that 1-LSB
//    activation error, compounded through a 1700+-deep GEMV reduction,
//    was enough to make most OUTPUT elements differ -- bit-exactness is
//    all-or-nothing, so even a small per-element drift shows up as a
//    large mismatch count (166/216 mismatched, barely different from
//    before bug (1) was even fixed -- a strong reminder that a clean
//    compile and a plausible-looking informational cosine prove nothing
//    about bit-exactness on their own). Fixed by matching actq()'s own
//    floor semantics exactly (a real arithmetic right-shift on the
//    integer, not float round-trip) -- see quantize_act_per_channel_
//    from_int's own docstring in conv1d_ref.py for the full account.
//
// Both bugs were caught the same way every other bug in this project's
// own history has been: added temporary debug taps (hierarchical
// references into a scratch copy of the testbench, dumping the RTL's own
// intermediate per-channel INT8 activations) and compared element-by-
// element against the Python reference's own intermediate arrays, rather
// than trying to reason the mismatch pattern out from the final output
// alone.
//
// Result: whole-chain cosine rose ~0.96 -> **~0.999**, matching every
// other block's own 0.99+ precision -- CONV_FRONT_END_VERDICT
// bitexact=1, mismatches=0/216, and all 3 standalone conv gates stay
// bit-exact too. This closes the activation-side gap this file's own
// history had, until this point, called architecturally out of reach.
//
// ---- Checked for more, found the practical ceiling instead -----------
// With per-channel scaling in place, checked whether the earlier
// "calibrate against the real percentile, not the absolute max" trick
// (the third fix, above) still helps when applied PER CHANNEL/PER ROW
// too (percentile-based quantize_act_per_channel_from_int/quantize_
// weight_per_row variants, tried standalone, not committed). It does
// NOT -- cosine got slightly WORSE (~0.999 -> ~0.9987). That's the
// expected signature of having already fixed the real problem: percentile
// clipping only helps when a SHARED scale is being dragged down by
// outliers in channels/rows it doesn't otherwise serve well; once every
// channel/row has its OWN max-based scale, there's no cross-channel tail
// left to trade away, so clipping a channel's own genuine outliers just
// throws away real information for no offsetting gain. Checked the worst
// remaining conv3 element directly too: true value ~-959, this pipeline
// computes ~-935 -- a proportional ~2.6% error at large magnitude, not a
// gross mismatch like the bugs found above. This is genuine, well-
// distributed INT8 quantization noise, not a further-fixable systematic
// error -- the practical precision ceiling of INT8 weights/activations
// at this reduction depth (K up to 2016), matching every other INT8-
// quantized block in this project. Closing this further would need
// wider activations/weights than INT8 in gemv_banked_resident_vec.sv --
// an already-proven, heavily-reused shared module -- a materially
// bigger, separate undertaking, not justified at the current precision.

// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module conv_front_end_seq #(
    parameter integer P    = 8,
    parameter integer LANES = 128,
    parameter integer WBW  = 8,
    parameter integer TIN1 = 3000        // real audio length (this project's own gate convention)
) (
    input  wire clk,
    input  wire rst,

    // ---- conv1 preload (weight + per-row dequant once; input audio once per go) --
    input  wire        c1_gv_ld_rst,
    input  wire        c1_gv_ld_we,
    input  wire [31:0] c1_gv_ld_data,
    input  wire             c1_dq_we,
    input  wire [P*24-1:0]  c1_dq_wmant,
    input  wire [P*8-1:0]   c1_dq_wexp,
    input  wire            c1_xt_we,
    input  wire [P*8-1:0]  c1_xt_data,

    // ---- groupnorm1 preload (gamma/beta once) --------------------------------
    input  wire            gn_g_we,
    input  wire [P*32-1:0] gn_g_data,
    input  wire            gn_b_we,
    input  wire [P*32-1:0] gn_b_data,
    // groupnorm1 output -> conv2 INT8 actquant: PER-CHANNEL shift table
    // (CROWS_GN=C_GN/P rows, P lanes/row, one signed 8-bit shift per real
    // channel) -- NOT a single shared shift, see this file's own "closing
    // the activation-side gap" section for why a per-channel table (paired
    // with a matching per-channel weight-column rescale in the Python
    // packer, conv1d_ref.rescale_weight_cols_per_channel) was needed.
    input  wire             gn_ash_we,
    input  wire [P*8-1:0]   gn_ash_data,

    // ---- conv2 preload (weight + per-row dequant + bias once) ----------------
    input  wire        c2_gv_ld_rst,
    input  wire        c2_gv_ld_we,
    input  wire [31:0] c2_gv_ld_data,
    input  wire             c2_dq_we,
    input  wire [P*24-1:0]  c2_dq_wmant,
    input  wire [P*8-1:0]   c2_dq_wexp,
    input  wire            c2_b_we,
    input  wire [P*32-1:0] c2_b_data,

    // gelu1 output -> conv3 INT8 actquant: PER-CHANNEL shift table
    // (MROWS2=COUT2/P rows, P lanes/row) -- same reasoning as gn_ash above.
    input  wire             ge1_ash_we,
    input  wire [P*8-1:0]   ge1_ash_data,

    // ---- conv3 preload (weight + per-row dequant + bias once) ----------------
    input  wire        c3_gv_ld_rst,
    input  wire        c3_gv_ld_we,
    input  wire [31:0] c3_gv_ld_data,
    input  wire             c3_dq_we,
    input  wire [P*24-1:0]  c3_dq_wmant,
    input  wire [P*8-1:0]   c3_dq_wexp,
    input  wire            c3_b_we,
    input  wire [P*32-1:0] c3_b_data,

    // ---- run ------------------------------------------------------------------
    input  wire  go,
    output reg   done,
    output wire          y_valid,          // final Q6.25 output, t-major/channel-minor
    output wire [P*32-1:0] y_data
);
    // ---- real shapes (moonshine-tiny's own conv front end) -------------------
    localparam integer CIN1  = 8;                        // padded from 1 (see conv1d_seq.sv's own header)
    localparam integer COUT1 = 288;
    localparam integer KW1   = 127;
    localparam integer STRIDE1 = 64;
    localparam integer TOUT1 = (TIN1 - KW1) / STRIDE1 + 1;
    localparam integer MROWS1 = COUT1 / P;
    localparam integer ROWS_C1 = TOUT1 * MROWS1;
    localparam integer KMAX1 = KW1 * CIN1;
    localparam integer WWORDS_C1 = ((COUT1 + LANES - 1) / LANES) * KMAX1;

    localparam integer C_GN = COUT1;                     // groupnorm1's own C
    localparam integer T_GN = TOUT1;                      // groupnorm1's own T
    localparam integer CROWS_GN = C_GN / P;

    localparam integer CIN2  = COUT1;
    localparam integer COUT2 = 576;
    localparam integer KW2   = 7;
    localparam integer STRIDE2 = 3;
    localparam integer TIN2  = TOUT1;
    localparam integer TOUT2 = (TIN2 - KW2) / STRIDE2 + 1;
    localparam integer MROWS2 = COUT2 / P;
    localparam integer ROWS_C2 = TOUT2 * MROWS2;
    localparam integer KMAX2 = KW2 * CIN2;
    localparam integer WWORDS_C2 = ((COUT2 + LANES - 1) / LANES) * KMAX2;

    localparam integer CIN3  = COUT2;
    localparam integer COUT3 = 288;
    localparam integer KW3   = 3;
    localparam integer STRIDE3 = 2;
    localparam integer TIN3  = TOUT2;
    localparam integer TOUT3 = (TIN3 - KW3) / STRIDE3 + 1;
    localparam integer MROWS3 = COUT3 / P;
    localparam integer ROWS_C3 = TOUT3 * MROWS3;
    localparam integer KMAX3 = KW3 * CIN3;
    localparam integer WWORDS_C3 = ((COUT3 + LANES - 1) / LANES) * KMAX3;

    // XTROWS_C2 must equal ROWS_GN (groupnorm1's own output row count):
    //   ROWS_GN = C_GN*T_GN/P = COUT1*TOUT1/P = MROWS1*TOUT1 = ROWS_C1
    // XTROWS_C3 must equal ROWS_C2 -- both hold by construction (TIN2=TOUT1,
    // CIN2=COUT1 etc above), asserted in the Python packer, not re-checked here.

    // ---- format-glue functions -------------------------------------------------
    function automatic signed [15:0] sat16;
        input signed [31:0] x;
        begin
            if (x > 32'sd32767) sat16 = 16'sd32767;
            else if (x < -32'sd32768) sat16 = -16'sd32768;
            else sat16 = x[15:0];
        end
    endfunction
    // Two widths: tanh's own output is always 16-bit (Q4.12, sat16-clipped
    // -- that path is correct as-is, see gelu_wide_vec.sv's own header for
    // why GELU needs the wide variant but tanh does not; tanh's own +-1
    // output range is ALWAYS well inside Q6.25's own +-64, so a plain
    // exact <<<13 can never overflow there). gelu_wide_vec.sv's own output
    // is 32-bit (may hold values outside int16 range for its own
    // positive-passthrough case) -- widen1312_sat takes that width
    // directly, and SATURATES instead of wrapping: x > +262143 (real
    // > ~+64) clips to INT32_MAX, x < -262144 (real < ~-64) clips to
    // INT32_MIN, otherwise the shift is exact (no overflow possible in
    // that range). See this file's own "Q6.25 final-output range" section
    // below for why saturate, not wrap, was chosen for this NEW boundary
    // (decoder/encoder/output_head's own xres_bank wraps instead -- an
    // already-shipped, already-gated design decision this file does not
    // revisit; wrapping was simply never reconsidered there, not concluded
    // to be better).
    function automatic signed [31:0] widen1312;
        input signed [15:0] x;
        begin
            widen1312 = $signed(x) <<< 13;
        end
    endfunction
    function automatic signed [31:0] widen1312_sat;
        input signed [31:0] x;
        begin
            if (x > 32'sd262143)        widen1312_sat = 32'sh7FFFFFFF;
            else if (x < -32'sd262144)  widen1312_sat = 32'sh80000000;
            else                         widen1312_sat = $signed(x) <<< 13;
        end
    endfunction
    function automatic signed [7:0] actq;
        input signed [31:0] x;
        input signed [7:0]  shift;
        reg signed [31:0] q;
        begin
            q = (shift >= 0) ? (x >>> shift) : (x <<< (-shift));
            if (q > 32'sd127) actq = 8'sd127;
            else if (q < -32'sd128) actq = -8'sd128;
            else actq = q[7:0];
        end
    endfunction

    // ---- conv1 ------------------------------------------------------------------
    reg  c1_go; wire c1_done; wire c1_yv; wire [P*32-1:0] c1_ydata;
    conv1d_seq #(.P(P), .WBW(WBW), .CIN(CIN1), .COUT(COUT1), .KW(KW1), .STRIDE(STRIDE1),
                 .TIN(TIN1), .HAS_BIAS(0), .LANES(LANES), .WWORDS(WWORDS_C1)) u_conv1 (
        .clk(clk), .rst(rst),
        .gv_ld_rst(c1_gv_ld_rst), .gv_ld_we(c1_gv_ld_we), .gv_ld_data(c1_gv_ld_data),
        .dq_we(c1_dq_we), .dq_wmant(c1_dq_wmant), .dq_wexp(c1_dq_wexp),
        .b_we(1'b0), .b_data({(P*32){1'b0}}),
        .xt_we(c1_xt_we), .xt_data(c1_xt_data),
        .go(c1_go), .done(c1_done), .y_valid(c1_yv), .y_data(c1_ydata)
    );

    // conv1 -> tanh: direct wire, sat16 only (per-row dequant table already targets Q4.12)
    integer c1p;
    reg signed [16*P-1:0] tanh_x;
    always @(*) begin
        tanh_x = {(16*P){1'b0}};
        for (c1p = 0; c1p < P; c1p = c1p + 1)
            tanh_x[c1p*16 +: 16] = sat16($signed(c1_ydata[c1p*32 +: 32]));
    end
    wire tanh_ov; wire signed [16*P-1:0] tanh_y;
    vec_tanh #(.P(P)) u_tanh (
        .clk(clk), .in_valid(c1_yv), .x(tanh_x), .out_valid(tanh_ov), .y(tanh_y)
    );

    // tanh -> bank_tanh (widened to Q6.25) -- the ONE buffered boundary, see header
    reg [P*32-1:0] bank_tanh [0:ROWS_C1-1];
    reg [$clog2(ROWS_C1+1)-1:0] tanh_wptr;
    integer twp;
    reg [P*32-1:0] tanh_word;
    always @(*) begin
        tanh_word = {(P*32){1'b0}};
        for (twp = 0; twp < P; twp = twp + 1)
            tanh_word[twp*32 +: 32] = widen1312($signed(tanh_y[twp*16 +: 16]));
    end
    reg tanh_wptr_clr;
    always @(posedge clk) begin
        if (rst || tanh_wptr_clr) tanh_wptr <= 0;
        else if (tanh_ov) begin
            bank_tanh[tanh_wptr] <= tanh_word;
            tanh_wptr <= tanh_wptr + 1'b1;
        end
    end

    // ---- groupnorm1 ---------------------------------------------------------------
    // gamma/beta preload banks: written ONCE at init (gn_g_we/gn_b_we, own
    // pointers), then RE-DRIVEN into groupnorm1_vec's own gamma_in/beta_in
    // every go call (that module's own S_GLOAD has no "skip if already
    // loaded" path -- see groupnorm1_vec.sv's own header) via gn_gi below.
    (* ram_style = "distributed" *) reg [P*32-1:0] bank_gn_gamma [0:CROWS_GN-1];
    (* ram_style = "distributed" *) reg [P*32-1:0] bank_gn_beta  [0:CROWS_GN-1];
    reg [$clog2(CROWS_GN+1)-1:0] gn_g_wptr, gn_b_wptr;
    always @(posedge clk) begin
        if (rst) gn_g_wptr <= 0;
        else if (gn_g_we) begin bank_gn_gamma[gn_g_wptr] <= gn_g_data; gn_g_wptr <= gn_g_wptr + 1'b1; end
    end
    always @(posedge clk) begin
        if (rst) gn_b_wptr <= 0;
        else if (gn_b_we) begin bank_gn_beta[gn_b_wptr] <= gn_b_data; gn_b_wptr <= gn_b_wptr + 1'b1; end
    end

    reg gn_start; reg gn_gvalid; reg gn_valid;
    reg [P*32-1:0] gn_x_in, gn_gamma_in, gn_beta_in;
    wire gn_yv; wire [P*64-1:0] gn_yout; wire gn_done;
    groupnorm1_vec #(.P(P), .C(C_GN), .T(T_GN)) u_gn (
        .clk(clk), .rst(rst), .start(gn_start),
        .gvalid_in(gn_gvalid), .gamma_in(gn_gamma_in), .beta_in(gn_beta_in),
        .valid_in(gn_valid), .x_in(gn_x_in),
        .y_valid(gn_yv), .y_out(gn_yout), .done(gn_done)
    );

    // per-channel ashift table: CROWS_GN rows, loaded once (same
    // auto-incrementing-pointer convention as every other preload table
    // in this file).
    (* ram_style = "distributed" *) reg [P*8-1:0] bank_gn_ash [0:CROWS_GN-1];
    reg [$clog2(CROWS_GN+1)-1:0] gn_ash_wptr;
    always @(posedge clk) begin
        if (rst) gn_ash_wptr <= 0;
        else if (gn_ash_we) begin bank_gn_ash[gn_ash_wptr] <= gn_ash_data; gn_ash_wptr <= gn_ash_wptr + 1'b1; end
    end
    // gn_row: which of the CROWS_GN channel-rows is CURRENTLY streaming
    // out of gn_yv, wrapping every CROWS_GN rows -- groupnorm1_vec.sv's
    // own S_OUT emits t-major/channel-row-minor (see that module's header),
    // so row index modulo CROWS_GN IS the channel-row, tracked here with a
    // plain wrapping counter (cheaper than a real modulo, CROWS_GN isn't a
    // power of 2).
    reg [$clog2(CROWS_GN+1)-1:0] gn_row;
    reg gn_row_clr;
    always @(posedge clk) begin
        if (rst || gn_row_clr) gn_row <= 0;
        else if (gn_yv) gn_row <= (gn_row == CROWS_GN-1) ? {$clog2(CROWS_GN+1){1'b0}} : gn_row + 1'b1;
    end

    // groupnorm1 -> conv2 xt_we: direct wire, actquant, PER-CHANNEL shift
    // (real quantization to INT8, see conv1d_ref.quantize_act_per_channel's
    // own docstring for why this needed a matching weight-column rescale
    // in the Python packer -- not just this table).
    integer gnp;
    reg [P*8-1:0] gn_xt_word;
    always @(*) begin
        gn_xt_word = {(P*8){1'b0}};
        for (gnp = 0; gnp < P; gnp = gnp + 1)
            gn_xt_word[gnp*8 +: 8] = actq($signed(gn_yout[gnp*64 +: 32]),
                                           $signed(bank_gn_ash[gn_row][gnp*8 +: 8]));
    end

    // ---- conv2 ------------------------------------------------------------------
    reg  c2_go; wire c2_done; wire c2_yv; wire [P*32-1:0] c2_ydata;
    conv1d_seq #(.P(P), .WBW(WBW), .CIN(CIN2), .COUT(COUT2), .KW(KW2), .STRIDE(STRIDE2),
                 .TIN(TIN2), .HAS_BIAS(1), .LANES(LANES), .WWORDS(WWORDS_C2)) u_conv2 (
        .clk(clk), .rst(rst),
        .gv_ld_rst(c2_gv_ld_rst), .gv_ld_we(c2_gv_ld_we), .gv_ld_data(c2_gv_ld_data),
        .dq_we(c2_dq_we), .dq_wmant(c2_dq_wmant), .dq_wexp(c2_dq_wexp),
        .b_we(c2_b_we), .b_data(c2_b_data),
        .xt_we(gn_yv), .xt_data(gn_xt_word),
        .go(c2_go), .done(c2_done), .y_valid(c2_yv), .y_data(c2_ydata)
    );

    // conv2 -> gelu1: direct wire, WIDE (no sat16 pre-clip -- gelu_wide_vec.sv
    // does its own internal clip for the LUT path while preserving the wide
    // value for its own positive-passthrough case, see that module's header;
    // this closes conv_front_end_seq.sv's own earlier "Honest limitation" note).
    wire ge1_ov; wire signed [32*P-1:0] ge1_y;
    gelu_wide_vec #(.P(P)) u_gelu1 (
        .clk(clk), .in_valid(c2_yv), .x(c2_ydata), .out_valid(ge1_ov), .y(ge1_y)
    );

    // per-channel ashift table: MROWS2 rows, same convention as gn's own above.
    (* ram_style = "distributed" *) reg [P*8-1:0] bank_ge1_ash [0:MROWS2-1];
    reg [$clog2(MROWS2+1)-1:0] ge1_ash_wptr;
    always @(posedge clk) begin
        if (rst) ge1_ash_wptr <= 0;
        else if (ge1_ash_we) begin bank_ge1_ash[ge1_ash_wptr] <= ge1_ash_data; ge1_ash_wptr <= ge1_ash_wptr + 1'b1; end
    end
    // ge1_row: which of the MROWS2 channel-rows is CURRENTLY streaming out
    // of ge1_ov, wrapping every MROWS2 rows -- conv2's own y_data stream
    // (which gelu_wide_vec.sv passes straight through, just delayed) is
    // t-major/channel-row-minor, same reasoning as gn_row above.
    reg [$clog2(MROWS2+1)-1:0] ge1_row;
    reg ge1_row_clr;
    always @(posedge clk) begin
        if (rst || ge1_row_clr) ge1_row <= 0;
        else if (ge1_ov) ge1_row <= (ge1_row == MROWS2-1) ? {$clog2(MROWS2+1){1'b0}} : ge1_row + 1'b1;
    end

    // gelu1 -> conv3 xt_we: direct wire, actquant, PER-CHANNEL shift
    integer gep;
    reg [P*8-1:0] ge1_xt_word;
    always @(*) begin
        ge1_xt_word = {(P*8){1'b0}};
        for (gep = 0; gep < P; gep = gep + 1)
            ge1_xt_word[gep*8 +: 8] = actq($signed(ge1_y[gep*32 +: 32]),
                                            $signed(bank_ge1_ash[ge1_row][gep*8 +: 8]));
    end
    reg [$clog2(ROWS_C2+1)-1:0] cnt_c3xt;
    reg cnt_c3xt_clr;
    always @(posedge clk) begin
        if (rst || cnt_c3xt_clr) cnt_c3xt <= 0;
        else if (ge1_ov) cnt_c3xt <= cnt_c3xt + 1'b1;
    end

    // ---- conv3 ------------------------------------------------------------------
    reg  c3_go; wire c3_done; wire c3_yv; wire [P*32-1:0] c3_ydata;
    conv1d_seq #(.P(P), .WBW(WBW), .CIN(CIN3), .COUT(COUT3), .KW(KW3), .STRIDE(STRIDE3),
                 .TIN(TIN3), .HAS_BIAS(1), .LANES(LANES), .WWORDS(WWORDS_C3)) u_conv3 (
        .clk(clk), .rst(rst),
        .gv_ld_rst(c3_gv_ld_rst), .gv_ld_we(c3_gv_ld_we), .gv_ld_data(c3_gv_ld_data),
        .dq_we(c3_dq_we), .dq_wmant(c3_dq_wmant), .dq_wexp(c3_dq_wexp),
        .b_we(c3_b_we), .b_data(c3_b_data),
        .xt_we(ge1_ov), .xt_data(ge1_xt_word),
        .go(c3_go), .done(c3_done), .y_valid(c3_yv), .y_data(c3_ydata)
    );

    // conv3 -> gelu2: direct wire, WIDE (same fix as gelu1, see above)
    wire ge2_ov; wire signed [32*P-1:0] ge2_y;
    gelu_wide_vec #(.P(P)) u_gelu2 (
        .clk(clk), .in_valid(c3_yv), .x(c3_ydata), .out_valid(ge2_ov), .y(ge2_y)
    );

    // gelu2 -> this module's own output: direct wire, widen to Q6.25 (encoder's
    // own first LayerNorm input format) -- the front end's own final output.
    // widen1312_sat() SATURATES (not wraps) when the real value exceeds
    // Q6.25's own ~+-64 representable range -- see that function's own
    // comment and this file's "Q6.25 final-output range" section below.
    integer gep2;
    reg [P*32-1:0] ge2_word;
    always @(*) begin
        ge2_word = {(P*32){1'b0}};
        for (gep2 = 0; gep2 < P; gep2 = gep2 + 1)
            ge2_word[gep2*32 +: 32] = widen1312_sat($signed(ge2_y[gep2*32 +: 32]));
    end
    assign y_valid = ge2_ov;
    assign y_data  = ge2_word;

    reg [$clog2(ROWS_C3+1)-1:0] cnt_ge2;
    reg cnt_ge2_clr;
    always @(posedge clk) begin
        if (rst || cnt_ge2_clr) cnt_ge2 <= 0;
        else if (ge2_ov) cnt_ge2 <= cnt_ge2 + 1'b1;
    end

    // ---- top-level FSM ------------------------------------------------------------
    localparam [3:0]
        S_IDLE=0, S_C1_START=1, S_C1_WAIT=2,
        S_GN_START=3, S_GN_GFEED=4, S_GN_XFEED=5, S_GN_WAIT=6,
        S_C2_START=7, S_C2_WAIT=8,
        S_C3_START=9, S_C3_WAIT=10,
        S_DONE=11;
    reg [3:0] state;
    reg [$clog2(CROWS_GN+1)-1:0] gn_gi;
    reg [$clog2(ROWS_C1+1)-1:0]  gn_xi;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0;
            c1_go <= 1'b0; c2_go <= 1'b0; c3_go <= 1'b0;
            gn_start <= 1'b0; gn_gvalid <= 1'b0; gn_valid <= 1'b0;
            tanh_wptr_clr <= 1'b0; cnt_c3xt_clr <= 1'b0; cnt_ge2_clr <= 1'b0;
            gn_row_clr <= 1'b0; ge1_row_clr <= 1'b0;
            gn_gi <= 0; gn_xi <= 0;
        end else begin
            done <= 1'b0;
            c1_go <= 1'b0; c2_go <= 1'b0; c3_go <= 1'b0;
            gn_start <= 1'b0; gn_gvalid <= 1'b0; gn_valid <= 1'b0;
            tanh_wptr_clr <= 1'b0; cnt_c3xt_clr <= 1'b0; cnt_ge2_clr <= 1'b0;
            gn_row_clr <= 1'b0; ge1_row_clr <= 1'b0;
            case (state)
                S_IDLE: if (go) begin
                    tanh_wptr_clr <= 1'b1; cnt_c3xt_clr <= 1'b1; cnt_ge2_clr <= 1'b1;
                    gn_row_clr <= 1'b1; ge1_row_clr <= 1'b1;
                    c1_go <= 1'b1;
                    state <= S_C1_WAIT;
                end
                // conv1's own weight/audio already preloaded by the host; its
                // own internal FSM paces tanh directly (see the header) -- just
                // wait for tanh's OWN drain to reach ROWS_C1, not conv1's done.
                S_C1_WAIT: if (tanh_wptr == ROWS_C1[$clog2(ROWS_C1+1)-1:0]) begin
                    gn_start <= 1'b1;
                    state <= S_GN_START;
                end
                S_GN_START: begin
                    gn_gi <= 0;
                    state <= S_GN_GFEED;
                end
                S_GN_GFEED: begin
                    gn_gvalid   <= 1'b1;
                    gn_gamma_in <= bank_gn_gamma[gn_gi];
                    gn_beta_in  <= bank_gn_beta[gn_gi];
                    if (gn_gi != CROWS_GN-1) gn_gi <= gn_gi + 1'b1;
                    else begin gn_xi <= 0; state <= S_GN_XFEED; end
                end
                S_GN_XFEED: begin
                    gn_valid <= 1'b1;
                    gn_x_in  <= bank_tanh[gn_xi];
                    if (gn_xi != ROWS_C1-1) gn_xi <= gn_xi + 1'b1;
                    else state <= S_GN_WAIT;
                end
                // groupnorm1's own y_valid stream is wired directly into
                // conv2's own xt_we (state-independent write, see header) --
                // gn_done is the correct completion signal here (no extra
                // downstream LUT lag to wait out, unlike S_C1_WAIT/S_C2_WAIT).
                S_GN_WAIT: if (gn_done) begin
                    c2_go <= 1'b1;
                    state <= S_C2_START;
                end
                S_C2_START: state <= S_C2_WAIT;
                S_C2_WAIT: if (cnt_c3xt == ROWS_C2[$clog2(ROWS_C2+1)-1:0]) begin
                    c3_go <= 1'b1;
                    state <= S_C3_START;
                end
                S_C3_START: state <= S_C3_WAIT;
                S_C3_WAIT: if (cnt_ge2 == ROWS_C3[$clog2(ROWS_C3+1)-1:0]) begin
                    state <= S_DONE;
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule // conv_front_end_seq
