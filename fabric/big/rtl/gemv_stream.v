// -----------------------------------------------------------------------------
// gemv_stream — streaming 1-bit x BFP10 GEMV (doc 10, rung R2a).
//
// Consumes the weight stream written by fabric/big/pack_weights.py (its module
// docstring is the normative stream format) from NPORTS 128-bit valid/ready
// inputs — the AXI R channels of the port-interleaved read masters — and emits
// one IEEE fp32 per output row AND PER BATCH ELEMENT, in row order. Arithmetic is
// exactly fabric/big/ref/fabric_ref.py: ExactBfpGemv.mm:
//
//   per 128-group:  inner = sum_j sign_j * q_j           (17-bit signed, exact)
//                   term  = inner * m_d                  (fp16 scale = m_d * 2^e_d,
//                                                         m_d signed 12 bit, as fp16_decompose)
//                   acc  += term << sh,  sh = e_d + e_x + 40   (sh < 0: arithmetic
//                                                         right shift = floor)
//   per row:        y = RNE_fp32(acc) * 2^-40            (acc 64-bit two's complement)
//
// Everything is modulo 2^64, as int64 is: a left shift of 64 or more contributes
// 0 and a right shift past the term's width contributes 0 / -1 (floor).
//
// Stream recap (pack_weights.py): a row is ceil(G/8) blocks of
// [SCALE beat (8 fp16), up to 8 SIGN beats (one 128-group each)], zero-padded to
// a multiple of NPORTS beats; logical beat i is on port i % NPORTS. The engine
// pops one TUPLE (one beat from every port) per cycle, so at NPORTS = 3 it eats
// 3 beats = 2.667 groups = 341 weights per cycle when nothing stalls.
//
// Datapath (LANE m handles the groups g with g % NPORTS == m; a tuple holds
// consecutive groups, so every lane sees at most one group per cycle and each
// lane's activation bank is read by address, never muxed):
//
//   port skid buffers -> tuple pop + beat classify (S0) [-> S0R, NB > 1: one
//   registered copy per batch element] -> sign/scale steer + activation read
//   (S1) -> 7-level add/sub sign tree (one LUT + carry per bit, registered per
//   TREE_REGMASK) || fp16 decode + shift count -> 17x12 multiply
//   (3 regs: DSP48E2 A/B, M, P) -> 2-stage 64-bit shifter -> lane sum
//   (ceil(log2 NPORTS) stages) -> 64-bit row accumulator -> 7-stage
//   int64 -> fp32 RNE -> output FIFO.
//
// The sign tree never negates: each node carries a pending sign and does
// a +/- b, and the root's pending sign is folded into the sign of the scale
// mantissa, so "sum of +-q" costs exactly a plain adder tree.
//
// BATCH (parameter NB = 1, 2 or 4; run-time cfg_batch = 1..NB). The weight stream
// is the bottleneck (DDR bandwidth), so ONE pass over it can serve up to NB
// activation vectors -- NB token positions: speculative decoding (R4), prompt
// prefill. Everything that depends only on the weights is built once: skids,
// tuple pop, beat classify, scale beat hold / pick, fp16 decode, the shift-count
// base e_d + 69. Everything that depends on the activation exists NB times: the
// activation banks (mantissas AND e_x: the shift count is e_d + e_x + 40 and e_x
// is per element), the sign trees, the multipliers, the shifters, the lane sums,
// the row accumulators and the fp32 converters. Element b of a row is, bit for
// bit, the word the single-vector engine produces for activation b alone; it is
// in m_data[32b +: 32], all elements of a row in the same m_valid cycle.
// Elements >= cfg_batch are NOT part of the contract and may read anything (as
// built, their group valids are gated off, so they convert a zero accumulator
// and read +0.0 whatever their banks hold; do not rely on it). Element 0 is
// always on. NB = 1 is the single-vector engine: same pipeline, same latency
// (cfg_batch is ignored, act_bank must be 0).
//   Fan-out (NB > 1): every batch element has its own REGISTERED copy of the
// popped tuple, of the sign-beat steering select and of the bank read addresses
// (stage S0R, one extra pipeline stage; `keep`, so synthesis cannot merge the
// copies), its own registered copy of the activation write port, and its own
// accumulator enable / clear. No stage of the single-vector pipeline gained a
// logic level; copy 0 of S0R carries the load stage 0 carried before.
//
// Back-pressure: the datapath never stalls. A tuple is popped only when every
// port has a beat and an output credit is free (one credit per output-FIFO
// slot, taken when a row's last tuple is popped, returned when its result
// leaves), so the FIFO cannot overflow whatever m_ready does.
//
// Latency: tuple pop -> m_valid is 16 + TL + LS + REP cycles (TL = registered
// tree levels, LS = ceil(log2 NPORTS), REP = 1 if NB > 1 else 0: stage S0R) = 25
// at the defaults (26 with NB > 1), plus one cycle through the input skid.
// MEASURED in sim: start -> !busy = tuples + 26 with no stalls (NB = 1), + 27
// (NB = 2, 4).
//
// Resources at NPORTS = 3, GMAX = 96.
//   NB = 1, PROJECTED (OOC place + route at 3.0 ns, results/ooc_gemv_stream.txt):
//     9,318 LUT (7,015 logic + 2,303 LUT RAM) / 13,297 FF / 823 CARRY8 / 6 DSP48E2
//     (two per 17x12 multiply as inferred) / 0 BRAM, WNS +0.197 ns.
//   NB = 2 and 4, DERIVED by counting from that run (NOT synthesised):
//     per batch element: 3 sign trees (1,517 add/sub bits each) ~ 4.6k LUT,
//       3 x 128 sign-beat mux LUTs, 3 shifters ~ 0.75k, lane sum + accumulator +
//       fp32 convert ~ 0.7k, i.e. ~6.5k logic LUT; 3 banks x 1280 b x 32 words =
//       2.3k LUT RAM; ~10.7k FF (trees 4.9k, bank read registers 3.8k, sign beats,
//       multiply / shift / sum / convert) + (NB > 1) the S0R copy 0.4k FF and the
//       write-port copy 1.3k FF = ~12.4k FF; 6 DSP48E2.
//     shared: skids, front end, scale side, write port, output FIFO ~ 0.5k LUT,
//       ~2.6k FF.
//     NB = 2: ~18k LUT (13.5k logic + 4.6k LUT RAM, 15% of the K26) / ~27k FF
//             (12%) / 12 DSP48E2 / 0 BRAM.
//     NB = 4: ~36k LUT (26.5k logic + 9.2k LUT RAM = 16% of the LUT-RAM-capable
//             LUTs; 30% of the K26) / ~52k FF (22%) / 24 DSP48E2 / 0 BRAM.
//   The banks cannot move to block RAM (1280 b x 12 banks = 216 RAMB36 > 144).
//   Timing at 3.0 ns is NOT known for NB > 1: per copy the paths are the
//   single-vector ones (the two that set the single-vector slack, both > 90%
//   routing: steering select -> 128 sign mux LUTs, OOC +0.197 ns; bank read
//   address -> 1280 b of LUT RAM -> read register, the worst path of the board
//   build, results/impl_gemv_axi_v2.txt), but 4 copies are 30% of the device and
//   must all sit close to the three skid outputs; expect the S0 -> S0R nets
//   (fan-out NB) and the routing of each copy's steering / read address to its
//   own lanes to be the tight ones.
//
// Host contract:
//   * load the activation (act_we: one 1280-bit word of 128 10-bit two's
//     complement mantissas, weight j in bits [10j +: 10], plus the group's signed
//     BFP exponent) for groups 0..G-1 OF EVERY BATCH ELEMENT 0..cfg_batch-1
//     (act_bank = element) while !busy. A bank keeps its contents until it is
//     written again, across runs and across cfg_batch changes;
//   * pulse start with cfg_groups = G (1..GMAX), cfg_rows = R (>= 1), cfg_pad =
//     pad tuples to discard after the matrix (stripe alignment; usually 0),
//     cfg_batch = elements in use (1..NB);
//   * busy falls once the last row has left m_* and the pad is consumed. The
//     input streams are untouched outside a run, so matrices follow each other
//     on the same streams with no gap.
// Not in this block: the AXI read masters / address generation (axi_rd_probe is
// the template), the activation BFP quantiser, and the layer sequencer.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------- skid buffer
// Registered ready AND registered valid/data: what an AXI R channel wants.
module gs_skid #(parameter integer W = 128) (
    input  wire         clk,
    input  wire         rst,
    input  wire         s_valid,
    output wire         s_ready,
    input  wire [W-1:0] s_data,
    output wire         m_valid,
    input  wire         m_ready,
    output wire [W-1:0] m_data
);
    reg         v0, v1;
    reg [W-1:0] d0, d1;
    assign s_ready = ~v1;
    assign m_valid = v0;
    assign m_data  = d0;
    wire push = s_valid & ~v1;
    wire pop  = v0 & m_ready;
    always @(posedge clk) begin
        if (rst) begin
            v0 <= 1'b0; v1 <= 1'b0;
        end else begin
            if (push) begin
                if (!v0 || pop) begin d0 <= s_data; v0 <= 1'b1; end
                else begin d1 <= s_data; v1 <= 1'b1; end
            end else if (pop) begin
                if (v1) begin d0 <= d1; v1 <= 1'b0; end
                else v0 <= 1'b0;
            end
        end
    end
endmodule

// ---------------------------------------------------------------- delay line
module gs_delay #(parameter integer W = 1, parameter integer D = 1, parameter integer RST = 0) (
    input  wire         clk,
    input  wire         rst,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (D == 0) begin : g0
            assign q = d;
        end else begin : g1
            reg [W-1:0] sr [0:D-1];
            integer i;
            always @(posedge clk) begin
                if (RST != 0 && rst) begin
                    for (i = 0; i < D; i = i + 1) sr[i] <= {W{1'b0}};
                end else begin
                    sr[0] <= d;
                    for (i = 1; i < D; i = i + 1) sr[i] <= sr[i-1];
                end
            end
            assign q = sr[D-1];
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- sign tree
// d * (sg ? +1 : -1) = sum_j (s[j] ? +q_j : -q_j), q_j = q[10j +: 10] signed.
// Node: out = a +/- b with the pending sign of a; subtract when the two pending
// signs differ. Level l (0..6) is registered when REGMASK[l]; REGMASK[6] must be 1.
module gs_signtree #(parameter [6:0] REGMASK = 7'b1111111) (
    input  wire          clk,
    input  wire [1279:0] q,
    input  wire [127:0]  s,
    output wire [16:0]   d,
    output wire          sg
);
    genvar l, i;
    generate
        for (l = 0; l < 7; l = l + 1) begin : lvl
            localparam integer WI = 10 + l;
            localparam integer WO = 11 + l;
            localparam integer NO = 64 >> l;
            reg  [NO*WO-1:0]   dq;      // this level's outputs (registered or not)
            reg  [NO-1:0]      sq;
            wire [2*NO*WI-1:0] din;
            wire [2*NO-1:0]    sin;
            if (l == 0) begin : in0
                assign din = q;
                assign sin = s;
            end else begin : inl
                assign din = lvl[l-1].dq;
                assign sin = lvl[l-1].sq;
            end
            wire [NO*WO-1:0] dn;
            wire [NO-1:0]    sn;
            for (i = 0; i < NO; i = i + 1) begin : add
                wire signed [WI-1:0] a = din[(2*i)*WI +: WI];
                wire signed [WI-1:0] b = din[(2*i+1)*WI +: WI];
                wire                 sub = sin[2*i] ^ sin[2*i+1];
                wire signed [WO-1:0] r = sub ? (a - b) : (a + b);
                assign dn[i*WO +: WO] = r;
                assign sn[i] = sin[2*i];
            end
            if (REGMASK[l]) begin : r1
                always @(posedge clk) begin dq <= dn; sq <= sn; end
            end else begin : r0
                always @* begin dq = dn; sq = sn; end
            end
        end
    endgenerate
    assign d  = lvl[6].dq;
    assign sg = lvl[6].sq;
endmodule

// ---------------------------------------------------------------- 17x12 multiply
// Three register stages = DSP48E2 with AREG/BREG, MREG, PREG.
module gs_mul (
    input  wire               clk,
    input  wire signed [16:0] a,
    input  wire signed [11:0] b,
    output wire signed [28:0] p
);
    (* use_dsp = "yes" *) reg signed [16:0] a_r;
    (* use_dsp = "yes" *) reg signed [11:0] b_r;
    (* use_dsp = "yes" *) reg signed [28:0] m_r;
    (* use_dsp = "yes" *) reg signed [28:0] p_r;
    always @(posedge clk) begin
        a_r <= a;
        b_r <= b;
        m_r <= a_r * b_r;
        p_r <= m_r;
    end
    assign p = p_r;
endmodule

// ---------------------------------------------------------------- lane sum
// Registered pairwise tree over N W-bit words; latency = ceil(log2 N) (0 for N = 1).
module gs_lanesum #(parameter integer N = 3, parameter integer W = 64) (
    input  wire           clk,
    input  wire [N*W-1:0] d,
    output wire [W-1:0]   s
);
    localparam integer LV = (N > 1) ? $clog2(N) : 0;
    localparam integer P2 = 1 << LV;
    wire [P2*W-1:0] dpad = d;      // zero-extends when N is not a power of two
    genvar l, i;
    generate
        if (LV == 0) begin : g0
            assign s = d[W-1:0];
        end else begin : g1
            for (l = 0; l < LV; l = l + 1) begin : lvl
                localparam integer NO = P2 >> (l + 1);
                reg  [NO*W-1:0]   dq;
                wire [2*NO*W-1:0] din;
                if (l == 0) begin : in0
                    assign din = dpad;
                end else begin : inl
                    assign din = lvl[l-1].dq;
                end
                for (i = 0; i < NO; i = i + 1) begin : add
                    always @(posedge clk)
                        dq[i*W +: W] <= din[(2*i)*W +: W] + din[(2*i+1)*W +: W];
                end
            end
            assign s = lvl[LV-1].dq;
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- int64 -> fp32
// y = RNE_fp32(x) * 2^-40, x 64-bit two's complement. Seven register stages:
// |x|, byte-level leading zeros, byte shift, bit-level leading zeros, bit shift,
// guard/sticky, round + pack. Bit-identical to torch `x.to(float32) * 2**-40`
// (x = 0 -> +0.0; INT64_MIN -> -2^23).
module gs_i64_to_f32 (
    input  wire        clk,
    input  wire        rst,
    input  wire        in_valid,
    input  wire        in_last,
    input  wire [63:0] in_data,
    output wire        out_valid,
    output wire        out_last,
    output wire [31:0] out_data
);
    function [2:0] lzc8;            // leading zeros of a non-zero byte
        input [7:0] x;
        begin
            casez (x)
                8'b1???????: lzc8 = 3'd0;
                8'b01??????: lzc8 = 3'd1;
                8'b001?????: lzc8 = 3'd2;
                8'b0001????: lzc8 = 3'd3;
                8'b00001???: lzc8 = 3'd4;
                8'b000001??: lzc8 = 3'd5;
                8'b0000001?: lzc8 = 3'd6;
                default:     lzc8 = 3'd7;
            endcase
        end
    endfunction

    reg [6:0] v;                    // valid per stage
    reg [6:0] lst;
    always @(posedge clk) begin
        if (rst) v <= 7'd0;
        else     v <= {v[5:0], in_valid};
        lst <= {lst[5:0], in_last};
    end

    // C1: magnitude
    reg        c1_sgn;
    reg [63:0] c1_mag;
    always @(posedge clk) begin
        c1_sgn <= in_data[63];
        c1_mag <= in_data[63] ? (64'd0 - in_data) : in_data;
    end

    // C2: leading zero BYTES
    wire [7:0] nzb;
    genvar gi;
    generate for (gi = 0; gi < 8; gi = gi + 1) begin : nz
        assign nzb[gi] = |c1_mag[8*gi +: 8];
    end endgenerate
    reg        c2_sgn, c2_zero;
    reg [2:0]  c2_cz;
    reg [63:0] c2_mag;
    always @(posedge clk) begin
        c2_sgn  <= c1_sgn;
        c2_mag  <= c1_mag;
        c2_zero <= ~|nzb;
        c2_cz   <= lzc8(nzb);       // nzb[7] is the top byte
    end

    // C3: shift left by whole bytes
    reg        c3_sgn, c3_zero;
    reg [2:0]  c3_cz;
    reg [63:0] c3_mag;
    always @(posedge clk) begin
        c3_sgn  <= c2_sgn;
        c3_zero <= c2_zero;
        c3_cz   <= c2_cz;
        c3_mag  <= c2_mag << {c2_cz, 3'b000};
    end

    // C4: leading zero bits of the (now non-zero) top byte
    reg        c4_sgn, c4_zero;
    reg [2:0]  c4_cz, c4_lz;
    reg [63:0] c4_mag;
    always @(posedge clk) begin
        c4_sgn  <= c3_sgn;
        c4_zero <= c3_zero;
        c4_cz   <= c3_cz;
        c4_mag  <= c3_mag;
        c4_lz   <= lzc8(c3_mag[63:56]);
    end

    // C5: normalise (bit 63 set unless zero)
    reg        c5_sgn, c5_zero;
    reg [5:0]  c5_lzc;
    reg [63:0] c5_mag;
    always @(posedge clk) begin
        c5_sgn  <= c4_sgn;
        c5_zero <= c4_zero;
        c5_lzc  <= {c4_cz, c4_lz};
        c5_mag  <= c4_mag << c4_lz;
    end

    // C6: mantissa, guard, sticky, exponent (= 127 + (63 - lzc) - 40)
    reg        c6_sgn, c6_zero, c6_guard, c6_sticky;
    reg [23:0] c6_man;
    reg [7:0]  c6_exp;
    always @(posedge clk) begin
        c6_sgn    <= c5_sgn;
        c6_zero   <= c5_zero;
        c6_man    <= c5_mag[63:40];
        c6_guard  <= c5_mag[39];
        c6_sticky <= |c5_mag[38:0];
        c6_exp    <= 8'd150 - {2'b00, c5_lzc};
    end

    // C7: round to nearest even, pack
    wire        c6_inc = c6_guard & (c6_sticky | c6_man[0]);
    wire [24:0] c6_sum = {1'b0, c6_man} + {24'd0, c6_inc};
    reg [31:0]  c7_out;
    always @(posedge clk) begin
        if (c6_zero) c7_out <= 32'd0;
        else         c7_out <= {c6_sgn, c6_exp + {7'd0, c6_sum[24]}, c6_sum[22:0]};
    end

    assign out_valid = v[6];
    assign out_last  = lst[6];
    assign out_data  = c7_out;
endmodule

// ---------------------------------------------------------------- output FIFO
module gs_fifo #(parameter integer W = 33, parameter integer LG = 4) (
    input  wire         clk,
    input  wire         rst,
    input  wire         wr,
    input  wire [W-1:0] din,
    output wire         m_valid,
    input  wire         m_ready,
    output wire [W-1:0] m_data
);
    reg [W-1:0] mem [0:(1<<LG)-1];
    reg [LG:0]  wp, rp;
    assign m_valid = (wp != rp);
    assign m_data  = mem[rp[LG-1:0]];
    always @(posedge clk) begin
        if (rst) begin
            wp <= 0; rp <= 0;
        end else begin
            if (wr) begin mem[wp[LG-1:0]] <= din; wp <= wp + 1'b1; end
            if (m_valid && m_ready) rp <= rp + 1'b1;
        end
    end
endmodule

// ---------------------------------------------------------------- top
module gemv_stream #(
    parameter integer NPORTS        = 3,                // 1..8 weight stream ports
    parameter integer GMAX          = 96,               // max 128-groups per row (<= 248)
    parameter integer EXW           = 9,                // activation exponent width (signed)
    parameter [6:0]   TREE_REGMASK  = 7'b1111111,       // sign-tree levels that are registered
    parameter integer OFIFO_LG      = 4,                // log2 output FIFO depth (= credits)
    parameter         ACT_RAM_STYLE = "distributed",
    parameter integer NB            = 1                 // max batch (activation vectors per weight pass): 1, 2, 4
) (
    input  wire                  clk,
    input  wire                  rst,
    // run control
    input  wire                  start,                 // 1-cycle pulse, only while !busy
    input  wire [7:0]            cfg_groups,            // G, 1..GMAX
    input  wire [31:0]           cfg_rows,              // R >= 1
    input  wire [31:0]           cfg_pad,               // pad tuples to discard after the matrix
    input  wire [2:0]            cfg_batch,             // batch elements in use, 1..NB (latched at start;
                                                        // element 0 is always on, so NB = 1 ignores it)
    output wire                  busy,
    // activation load (while !busy)
    input  wire                  act_we,
    input  wire [1:0]            act_bank,              // batch element 0..NB-1 (>= NB: the write is ignored)
    input  wire [7:0]            act_group,
    input  wire [1279:0]         act_mant,
    input  wire signed [EXW-1:0] act_exp,
    // weight streams, one per port
    input  wire [NPORTS-1:0]     s_valid,
    output wire [NPORTS-1:0]     s_ready,
    input  wire [NPORTS*128-1:0] s_data,
    // result stream: one word per row, row order; batch element b in bits [32b +: 32]
    output wire                  m_valid,
    input  wire                  m_ready,
    output wire [NB*32-1:0]      m_data,
    output wire                  m_last,                // last row of the matrix
    // telemetry taps for the board wrapper (gemv_axi): registered copies of the tuple
    // handshake state, one cycle late. No functional effect; leave unconnected otherwise.
    output reg                   dbg_pop,               // a DATA tuple was popped (pad tuples not counted)
    output reg                   dbg_stall_in,          // in a run, no pop: some port had no beat
    output reg                   dbg_stall_out,         // in a run, every port had a beat, no output credit
    output reg  [NPORTS-1:0]     dbg_empty              // the ports that had no beat (while in a run)
);
    localparam integer BW    = (NPORTS > 1) ? $clog2(NPORTS) : 1;
    localparam integer DEPTH = (GMAX + NPORTS - 1) / NPORTS;        // words per lane bank
    localparam integer AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam integer PW    = 10;                                    // row beat counter
    localparam integer LS    = (NPORTS > 1) ? $clog2(NPORTS) : 0;   // lane-sum latency
    localparam integer TL    = TREE_REGMASK[0] + TREE_REGMASK[1] + TREE_REGMASK[2] + TREE_REGMASK[3]
                             + TREE_REGMASK[4] + TREE_REGMASK[5] + TREE_REGMASK[6];
    localparam integer KW    = ((EXW > 8) ? EXW : 8) + 1;            // shift-count adder width
    localparam [3:0]   N9    = NPORTS % 9;
    localparam integer REP   = (NB > 1) ? 1 : 0;                     // the per-copy fan-out stage S0R exists

    // ------------------------------------------------------------ input skid
    wire [NPORTS-1:0]     k_valid;
    wire [NPORTS*128-1:0] k_data;
    wire                  pop;
    genvar gp;
    generate for (gp = 0; gp < NPORTS; gp = gp + 1) begin : skid
        gs_skid #(.W(128)) u (
            .clk(clk), .rst(rst),
            .s_valid(s_valid[gp]), .s_ready(s_ready[gp]), .s_data(s_data[gp*128 +: 128]),
            .m_valid(k_valid[gp]), .m_ready(pop), .m_data(k_data[gp*128 +: 128]));
    end endgenerate
    wire all_valid = &k_valid;

    // ------------------------------------------------------------ front end
    // State describes the NEXT tuple to pop: pos = row beat index of its port-0
    // beat, p9 = pos mod 9 (a row beat is a SCALE iff its index mod 9 is 0),
    // g8 / gm = sign beats already taken in this row mod 8 / mod NPORTS.
    localparam [1:0] ST_IDLE = 2'd0, ST_RUN = 2'd1, ST_PAD = 2'd2;
    reg [1:0]           st;
    reg [PW-1:0]        pos, rb;
    reg [3:0]           p9;
    reg [2:0]           g8;
    reg [BW-1:0]        gm;
    reg [NPORTS*AW-1:0] badr;
    reg [31:0]          rows_left, pad_left;
    reg [OFIFO_LG:0]    cred;
    reg                 out_pending;
    reg [NB-1:0]        ben;        // batch element in use this run (latched at start)

    reg [NPORTS-1:0]    d_bv;       // lane has a group in this tuple
    reg [NPORTS*BW-1:0] d_bsrc;     // which port carries it
    reg [NPORTS*3-1:0]  d_bsidx;    // its slot in the scale beat (group mod 8)
    reg [NPORTS-1:0]    d_bnew;     // its scale beat is in this same tuple
    reg                 d_hs;       // tuple has a scale beat
    reg [BW-1:0]        d_sp;       // on this port
    reg [3:0]           d_cnt;      // sign beats in the tuple
    reg                 d_newf;
    reg [PW:0]          d_b;
    reg [4:0]           d_ph;
    reg [BW+1:0]        d_bk;
    integer             j;
    always @* begin
        d_bv = 0; d_bsrc = 0; d_bsidx = 0; d_bnew = 0;
        d_hs = 1'b0; d_sp = 0; d_cnt = 4'd0; d_newf = 1'b0;
        d_b = 0; d_ph = 0; d_bk = 0;
        for (j = 0; j < NPORTS; j = j + 1) begin
            d_b  = pos + j;
            d_ph = p9 + j;
            if (d_ph >= 5'd9) d_ph = d_ph - 5'd9;
            if (d_b < rb) begin
                if (d_ph == 5'd0) begin
                    d_hs = 1'b1; d_sp = j; d_newf = 1'b1;
                end else begin
                    d_bk = gm + d_cnt;
                    if (d_bk >= NPORTS) d_bk = d_bk - NPORTS;
                    d_bv[d_bk] = 1'b1;
                    d_bsrc[d_bk*BW +: BW] = j;
                    d_bsidx[d_bk*3 +: 3]  = g8 + d_cnt[2:0];
                    d_bnew[d_bk] = d_newf;
                    d_cnt = d_cnt + 4'd1;
                end
            end
        end
    end

    wire          row_end  = ({1'b0, pos} + NPORTS >= {1'b0, rb});
    wire          mat_end  = row_end && (rows_left == 32'd1);
    wire          cred_ok  = (cred != 0);
    wire          fire     = (st == ST_RUN) && all_valid && cred_ok;
    wire          fire_pad = (st == ST_PAD) && all_valid;
    assign pop = fire | fire_pad;

    wire [4:0]    p9_sum = p9 + N9;
    wire [BW+1:0] gm_sum = gm + d_cnt;
    wire          fifo_pop;

    // stage 0: the popped tuple and its steering
    reg                  s0_v, s0_rl, s0_ml, s0_hs;
    reg [NPORTS*128-1:0] s0_data;
    reg [NPORTS-1:0]     s0_bv, s0_bnew;
    reg [NPORTS*BW-1:0]  s0_bsrc;
    reg [NPORTS*3-1:0]   s0_bsidx;
    reg [BW-1:0]         s0_sp;
    reg [NPORTS*AW-1:0]  s0_badr;

    integer m_i, b_i;
    always @(posedge clk) begin
        if (rst) begin
            st <= ST_IDLE; pos <= 0; rb <= 0; p9 <= 0; g8 <= 0; gm <= 0; badr <= 0;
            rows_left <= 0; pad_left <= 0; cred <= (1 << OFIFO_LG); out_pending <= 1'b0;
            s0_v <= 1'b0;
        end else begin
            s0_v <= fire;
            cred <= cred - ((fire && row_end) ? 1'b1 : 1'b0) + (fifo_pop ? 1'b1 : 1'b0);
            if (fifo_pop && m_last) out_pending <= 1'b0;

            if (st == ST_IDLE) begin
                if (start) begin
                    st <= ST_RUN; out_pending <= 1'b1;
                    rows_left <= cfg_rows; pad_left <= cfg_pad;
                    rb <= cfg_groups + ((cfg_groups + 8'd7) >> 3);
                    pos <= 0; p9 <= 0; g8 <= 0; gm <= 0; badr <= 0;
                end
            end else if (fire) begin
                if (row_end) begin
                    pos <= 0; p9 <= 0; g8 <= 0; gm <= 0; badr <= 0;
                    rows_left <= rows_left - 32'd1;
                    if (rows_left == 32'd1) st <= (pad_left != 32'd0) ? ST_PAD : ST_IDLE;
                end else begin
                    pos <= pos + NPORTS;
                    p9  <= (p9_sum >= 5'd9) ? p9_sum - 5'd9 : p9_sum;
                    g8  <= g8 + d_cnt[2:0];
                    gm  <= (gm_sum >= NPORTS) ? gm_sum - NPORTS : gm_sum;
                    for (m_i = 0; m_i < NPORTS; m_i = m_i + 1)
                        if (d_bv[m_i]) badr[m_i*AW +: AW] <= badr[m_i*AW +: AW] + 1'b1;
                end
            end else if (fire_pad) begin
                pad_left <= pad_left - 32'd1;
                if (pad_left == 32'd1) st <= ST_IDLE;
            end
        end
        if (fire) begin
            s0_data  <= k_data;
            s0_bv    <= d_bv;    s0_bnew <= d_bnew;
            s0_bsrc  <= d_bsrc;  s0_bsidx <= d_bsidx;
            s0_hs    <= d_hs;    s0_sp <= d_sp;
            s0_badr  <= badr;
            s0_rl    <= row_end; s0_ml <= mat_end;
        end
        // batch elements of this run: element 0 always, element b iff b < cfg_batch
        if (st == ST_IDLE && start)
            for (b_i = 0; b_i < NB; b_i = b_i + 1) ben[b_i] <= (b_i == 0) || (cfg_batch > b_i);
    end
    assign busy = (st != ST_IDLE) | out_pending;

    always @(posedge clk) begin
        if (rst) begin
            dbg_pop <= 1'b0; dbg_stall_in <= 1'b0; dbg_stall_out <= 1'b0; dbg_empty <= {NPORTS{1'b0}};
        end else begin
            dbg_pop       <= fire;
            dbg_stall_in  <= (st != ST_IDLE) && !all_valid;
            dbg_stall_out <= (st == ST_RUN) && all_valid && !cred_ok;
            dbg_empty     <= (st != ST_IDLE) ? ~k_valid : {NPORTS{1'b0}};
        end
    end

    // ------------------------------------------------------------ stage S0R: one copy per batch element
    // NB > 1 only (REP = 1; with NB = 1 these are wires and the pipeline is the single-vector one).
    // The single-vector core's worst path is the fan-out of the sign-beat steering select
    // (s0_bsrc -> 128 mux LUTs per lane). NB copies of the datapath must not hang on the same
    // registers, so every batch element gets its OWN registered copy of the tuple, of the
    // steering select and of the bank read addresses (keep: synthesis must not merge them
    // back into one), and stage 1 of copy b reads only copy b. The scale side is shared and
    // taps copy 0, so copy 0 carries exactly the load the single-vector stage 0 carries.
    // The batch enable is folded into the copy's group valids here (no gate later on).
    wire [NB*NPORTS*128-1:0] sr_data;
    wire [NB*NPORTS*BW-1:0]  sr_bsrc;
    wire [NB*NPORTS*AW-1:0]  sr_badr;
    wire [NB*NPORTS-1:0]     sr_bv;         // lane has a group AND the batch element is in use
    wire                     sr_v, sr_rl, sr_ml, sr_hs;
    wire [NPORTS-1:0]        sr_bnew;
    wire [NPORTS*3-1:0]      sr_bsidx;
    wire [BW-1:0]            sr_sp;
    genvar gb;
    generate
        if (REP == 0) begin : rep0
            assign sr_data = s0_data;  assign sr_bsrc = s0_bsrc;  assign sr_badr = s0_badr;
            assign sr_bv   = {NPORTS{s0_v}} & s0_bv;
            assign sr_v    = s0_v;     assign sr_rl = s0_rl;      assign sr_ml = s0_ml;
            assign sr_hs   = s0_hs;    assign sr_bnew = s0_bnew;  assign sr_bsidx = s0_bsidx;
            assign sr_sp   = s0_sp;
        end else begin : rep1
            for (gb = 0; gb < NB; gb = gb + 1) begin : cp
                (* keep = "true" *) reg [NPORTS*128-1:0] data;
                (* keep = "true" *) reg [NPORTS*BW-1:0]  bsrc;
                (* keep = "true" *) reg [NPORTS*AW-1:0]  badr_c;
                (* keep = "true" *) reg [NPORTS-1:0]     bv;
                always @(posedge clk) begin
                    data   <= s0_data;
                    bsrc   <= s0_bsrc;
                    badr_c <= s0_badr;
                    if (rst) bv <= {NPORTS{1'b0}};
                    else     bv <= {NPORTS{s0_v & ben[gb]}} & s0_bv;
                end
                assign sr_data[gb*NPORTS*128 +: NPORTS*128] = data;
                assign sr_bsrc[gb*NPORTS*BW +: NPORTS*BW]   = bsrc;
                assign sr_badr[gb*NPORTS*AW +: NPORTS*AW]   = badr_c;
                assign sr_bv[gb*NPORTS +: NPORTS]           = bv;
            end
            reg                r_v, r_rl, r_ml, r_hs;
            reg [NPORTS-1:0]   r_bnew;
            reg [NPORTS*3-1:0] r_bsidx;
            reg [BW-1:0]       r_sp;
            always @(posedge clk) begin
                if (rst) r_v <= 1'b0; else r_v <= s0_v;
                r_rl <= s0_rl; r_ml <= s0_ml; r_hs <= s0_hs;
                r_bnew <= s0_bnew; r_bsidx <= s0_bsidx; r_sp <= s0_sp;
            end
            assign sr_v = r_v;   assign sr_rl = r_rl;      assign sr_ml = r_ml;
            assign sr_hs = r_hs; assign sr_bnew = r_bnew;  assign sr_bsidx = r_bsidx;
            assign sr_sp = r_sp;
        end
    endgenerate

    // ------------------------------------------------------------ stage 1: steer
    function [15:0] pick16;         // fp16 slot i of a scale beat
        input [127:0] w;
        input [2:0]   i;
        begin
            case (i)
                3'd0: pick16 = w[15:0];
                3'd1: pick16 = w[31:16];
                3'd2: pick16 = w[47:32];
                3'd3: pick16 = w[63:48];
                3'd4: pick16 = w[79:64];
                3'd5: pick16 = w[95:80];
                3'd6: pick16 = w[111:96];
                default: pick16 = w[127:112];
            endcase
        end
    endfunction

    wire [NPORTS*128-1:0] sd0 = sr_data[0 +: NPORTS*128];     // copy 0 feeds the (shared) scale side
    reg [127:0] sbeat;              // the scale beat of the stage-0 tuple
    integer     jj;
    always @* begin
        sbeat = 128'd0;
        for (jj = 0; jj < NPORTS; jj = jj + 1)
            if (sr_sp == jj) sbeat = sd0[jj*128 +: 128];
    end
    reg [127:0] scur;               // the scale beat held from earlier tuples
    always @(posedge clk) if (sr_v && sr_hs) scur <= sbeat;

    reg s1_v, s1_rl, s1_ml;
    always @(posedge clk) begin
        if (rst) s1_v <= 1'b0; else s1_v <= sr_v;
        s1_rl <= sr_rl; s1_ml <= sr_ml;
    end

    // activation write port (registered; group -> lane = g % NPORTS, word = g / NPORTS)
    reg                  w_we;
    reg [1:0]            w_bk;
    reg [BW-1:0]         w_lane;
    reg [AW-1:0]         w_addr;
    reg [1279:0]         w_mant;
    reg signed [EXW-1:0] w_exp;
    always @(posedge clk) begin
        if (rst) w_we <= 1'b0; else w_we <= act_we;
        w_bk   <= act_bank;
        w_lane <= act_group % NPORTS;
        w_addr <= act_group / NPORTS;
        w_mant <= act_mant;
        w_exp  <= act_exp;
    end
    // ... and, NB > 1, one more registered copy per batch element (the 1280-bit word would
    // otherwise fan out to NPORTS * NB banks), with the bank select decoded into its enable
    wire [NB-1:0]         wb_we;
    wire [NB*BW-1:0]      wb_lane;
    wire [NB*AW-1:0]      wb_addr;
    wire [NB*1280-1:0]    wb_mant;
    wire [NB*EXW-1:0]     wb_exp;
    generate
        if (REP == 0) begin : wrep0
            assign wb_we = w_we & (w_bk == 2'd0);
            assign wb_lane = w_lane; assign wb_addr = w_addr; assign wb_mant = w_mant; assign wb_exp = w_exp;
        end else begin : wrep1
            for (gb = 0; gb < NB; gb = gb + 1) begin : cp
                reg                              we;
                (* keep = "true" *) reg [BW-1:0]   lane_c;
                (* keep = "true" *) reg [AW-1:0]   addr;
                (* keep = "true" *) reg [1279:0]   mant;
                (* keep = "true" *) reg [EXW-1:0]  expo;
                always @(posedge clk) begin
                    if (rst) we <= 1'b0; else we <= w_we & (w_bk == gb);
                    lane_c <= w_lane; addr <= w_addr; mant <= w_mant; expo <= w_exp;
                end
                assign wb_we[gb] = we;
                assign wb_lane[gb*BW +: BW]     = lane_c;
                assign wb_addr[gb*AW +: AW]     = addr;
                assign wb_mant[gb*1280 +: 1280] = mant;
                assign wb_exp[gb*EXW +: EXW]    = expo;
            end
        end
    endgenerate

    // ------------------------------------------------------------ lanes
    // Shared per lane (weight side): the scale pick, the fp16 decode, the shift-count base.
    // Per batch element: sign beat, activation bank (mantissas + e_x), sign tree, shift
    // count (e_d + e_x[b] + 40), multiply, shifter, term.
    wire [NB*NPORTS*64-1:0] terms;
    genvar gm_i;
    generate for (gm_i = 0; gm_i < NPORTS; gm_i = gm_i + 1) begin : lane
        wire [2:0]  sidx  = sr_bsidx[gm_i*3 +: 3];
        wire [15:0] scsel = sr_bnew[gm_i] ? pick16(sbeat, sidx) : pick16(scur, sidx);
        reg [15:0]  s1_sc;
        always @(posedge clk) s1_sc <= scsel;

        // stage 2 (shared part): fp16 decode.  k = sh + 29 = e_d + e_x + 69, and
        // e_d + 69 = (exp5 == 0 ? 45 : exp5 + 44)   (fp16_decompose: e_d = -24 | exp5 - 25)
        wire [4:0]  ex5   = s1_sc[14:10];
        wire [6:0]  ebase = (ex5 == 5'd0) ? 7'd45 : ({2'b00, ex5} + 7'd44);
        reg         s2_sgn;
        reg [10:0]  s2_mag;
        always @(posedge clk) begin
            s2_sgn <= s1_sc[15];
            s2_mag <= {ex5 != 5'd0, s1_sc[9:0]};
        end
        wire        r_sgn;          // aligned with the tree roots
        wire [10:0] r_mag;
        gs_delay #(.W(12), .D(TL-1)) u_ds (.clk(clk), .rst(rst), .d({s2_sgn, s2_mag}), .q({r_sgn, r_mag}));

        for (gb = 0; gb < NB; gb = gb + 1) begin : bat
            wire [NPORTS*128-1:0] sd   = sr_data[gb*NPORTS*128 +: NPORTS*128];
            wire [BW-1:0]         src  = sr_bsrc[(gb*NPORTS + gm_i)*BW +: BW];
            wire [AW-1:0]         radr = sr_badr[(gb*NPORTS + gm_i)*AW +: AW];

            reg [127:0] smux;
            integer     js;
            always @* begin
                smux = 128'd0;
                for (js = 0; js < NPORTS; js = js + 1)
                    if (src == js) smux = sd[js*128 +: 128];
            end

            // activation bank of (lane, batch element): mantissa word + exponent, synchronous read
            (* ram_style = ACT_RAM_STYLE *) reg [1279:0]   mem  [0:DEPTH-1];
            (* ram_style = "distributed" *) reg [EXW-1:0]  emem [0:DEPTH-1];
            wire [AW-1:0]        wadr = wb_addr[gb*AW +: AW];
            reg [1279:0]         s1_q;
            reg signed [EXW-1:0] s1_ex;
            always @(posedge clk) begin
                if (wb_we[gb] && wb_lane[gb*BW +: BW] == gm_i) begin
                    mem[wadr]  <= wb_mant[gb*1280 +: 1280];
                    emem[wadr] <= wb_exp[gb*EXW +: EXW];
                end
                s1_q  <= mem[radr];
                s1_ex <= emem[radr];
            end

            reg         s1_bv;
            reg [127:0] s1_sign;
            always @(posedge clk) begin
                if (rst) s1_bv <= 1'b0; else s1_bv <= sr_bv[gb*NPORTS + gm_i];
                s1_sign <= smux;
            end

            // sign tree: s1 -> root, TL cycles
            wire [16:0] t_d;
            wire        t_sg;
            gs_signtree #(.REGMASK(TREE_REGMASK)) u_tree (
                .clk(clk), .q(s1_q), .s(s1_sign), .d(t_d), .sg(t_sg));

            // stage 2 (per element): the shift count with this element's exponent
            wire signed [KW-1:0] ksum = $signed({1'b0, ebase}) + s1_ex;
            reg         s2_bv, s2_zero;
            reg [6:0]   s2_k;
            always @(posedge clk) begin
                if (rst) s2_bv <= 1'b0; else s2_bv <= s1_bv;
                s2_zero <= (ksum > 92);                         // sh >= 64: nothing lands in 64 bits
                s2_k    <= (ksum < 0) ? 7'd0 : ksum[6:0];       // sh <= -29: pure sign fill (floor)
            end

            // align with the tree root
            wire        r_bv, r_zero;
            wire [6:0]  r_k;
            gs_delay #(.W(1), .D(TL-1), .RST(1)) u_dv (.clk(clk), .rst(rst), .d(s2_bv), .q(r_bv));
            gs_delay #(.W(8), .D(TL-1)) u_dd (.clk(clk), .rst(rst), .d({s2_zero, s2_k}), .q({r_zero, r_k}));

            // multiply: term = d * (pending sign * m_d); the root's pending sign is
            // folded into the sign of the scale mantissa (no 17-bit negate)
            wire               negm = r_sgn ^ ~t_sg;
            wire signed [11:0] mb   = negm ? -$signed({1'b0, r_mag}) : $signed({1'b0, r_mag});
            wire signed [28:0] p;
            gs_mul u_mul (.clk(clk), .a($signed(t_d)), .b(mb), .p(p));

            wire       p_bv, p_zero;
            wire [6:0] p_k;
            gs_delay #(.W(1), .D(3), .RST(1)) u_pv (.clk(clk), .rst(rst), .d(r_bv), .q(p_bv));
            gs_delay #(.W(8), .D(3)) u_pd (.clk(clk), .rst(rst), .d({r_zero, r_k}), .q({p_zero, p_k}));

            // shifter: out = bits [92:29] of (sext(p) << k), k = 16*ka + kb
            reg signed [43:0] h_u;
            reg [2:0]         h_ka;
            reg               h_gate;
            always @(posedge clk) begin
                h_u  <= $signed(p) <<< p_k[3:0];
                h_ka <= p_k[6:4];
                if (rst) h_gate <= 1'b0; else h_gate <= p_bv & ~p_zero;
            end
            wire signed [92:0] h_ext = h_u;
            wire [92:0]        h_sh  = h_ext << {h_ka, 4'b0000};
            reg [63:0]         term;
            always @(posedge clk) term <= h_gate ? h_sh[92:29] : 64'd0;
            assign terms[(gb*NPORTS + gm_i)*64 +: 64] = term;
        end
    end endgenerate

    // ------------------------------------------------------------ lane sum + row accumulator
    // slot flags aligned with the lane sums: S1 + TL + 5 + LS. The last stage is one register
    // per batch element (keep), so no accumulator shares its enable / clear with another.
    wire a_v0, a_rl0, a_ml0;
    gs_delay #(.W(1), .D(TL + 4 + LS), .RST(1)) u_av (.clk(clk), .rst(rst), .d(s1_v), .q(a_v0));
    gs_delay #(.W(2), .D(TL + 4 + LS)) u_ad (.clk(clk), .rst(rst), .d({s1_rl, s1_ml}), .q({a_rl0, a_ml0}));

    wire [NB*64-1:0] res_all;
    wire [NB-1:0]    a_v_all, a_rl_all, a_ml_all;
    generate for (gb = 0; gb < NB; gb = gb + 1) begin : accu
        wire [63:0] lsum;
        gs_lanesum #(.N(NPORTS), .W(64)) u_ls (.clk(clk), .d(terms[gb*NPORTS*64 +: NPORTS*64]), .s(lsum));

        (* keep = "true" *) reg a_v, a_rl;
        reg                     a_ml;
        always @(posedge clk) begin
            if (rst) a_v <= 1'b0; else a_v <= a_v0;
            a_rl <= a_rl0; a_ml <= a_ml0;
        end

        reg [63:0]  acc, res;
        wire [63:0] acc_nx = acc + lsum;
        always @(posedge clk) begin
            if (rst)      acc <= 64'd0;
            else if (a_v) acc <= a_rl ? 64'd0 : acc_nx;
            res <= acc_nx;
        end
        assign res_all[gb*64 +: 64] = res;
        assign a_v_all[gb] = a_v; assign a_rl_all[gb] = a_rl; assign a_ml_all[gb] = a_ml;
    end endgenerate

    reg res_v, res_ml;
    always @(posedge clk) begin
        if (rst) res_v <= 1'b0; else res_v <= a_v_all[0] & a_rl_all[0];
        res_ml <= a_ml_all[0];
    end

    // ------------------------------------------------------------ int64 -> fp32, output FIFO
    wire [NB-1:0]    f_v, f_l;
    wire [NB*32-1:0] f_d;
    generate for (gb = 0; gb < NB; gb = gb + 1) begin : conv
        wire [31:0] cd;
        gs_i64_to_f32 u_cv (.clk(clk), .rst(rst), .in_valid(res_v), .in_last(res_ml),
                            .in_data(res_all[gb*64 +: 64]),
                            .out_valid(f_v[gb]), .out_last(f_l[gb]), .out_data(cd));
        assign f_d[gb*32 +: 32] = cd;
    end endgenerate

    wire [NB*32:0] fo;
    gs_fifo #(.W(NB*32 + 1), .LG(OFIFO_LG)) u_of (.clk(clk), .rst(rst), .wr(f_v[0]), .din({f_l[0], f_d}),
                                                  .m_valid(m_valid), .m_ready(m_ready), .m_data(fo));
    assign m_data   = fo[NB*32-1:0];
    assign m_last   = fo[NB*32];
    assign fifo_pop = m_valid & m_ready;
endmodule

`default_nettype wire
