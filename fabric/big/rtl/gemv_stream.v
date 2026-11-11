// -----------------------------------------------------------------------------
// gemv_stream — streaming 1-bit x BFP10 GEMV (doc 10, rung R2a).
//
// Consumes the weight stream written by fabric/big/pack_weights.py (its module
// docstring is the normative stream format) from NPORTS 128-bit valid/ready
// inputs — the AXI R channels of the port-interleaved read masters — and emits
// one IEEE fp32 per output row, in row order. Arithmetic is exactly
// fabric/big/ref/fabric_ref.py: ExactBfpGemv.mm:
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
//   port skid buffers -> tuple pop + beat classify (S0) -> sign/scale steer +
//   activation read (S1) -> 7-level add/sub sign tree (one LUT + carry per bit,
//   registered per TREE_REGMASK) || fp16 decode + shift count -> 17x12 multiply
//   (3 regs: DSP48E2 A/B, M, P) -> 2-stage 64-bit shifter -> lane sum
//   (ceil(log2 NPORTS) stages) -> 64-bit row accumulator -> 7-stage
//   int64 -> fp32 RNE -> output FIFO.
//
// The sign tree never negates: each node carries a pending sign and does
// a +/- b, and the root's pending sign is folded into the sign of the scale
// mantissa, so "sum of +-q" costs exactly a plain adder tree.
//
// Back-pressure: the datapath never stalls. A tuple is popped only when every
// port has a beat and an output credit is free (one credit per output-FIFO
// slot, taken when a row's last tuple is popped, returned when its result
// leaves), so the FIFO cannot overflow whatever m_ready does.
//
// Latency: tuple pop -> m_valid is 16 + TL + LS cycles (TL = registered tree
// levels, LS = ceil(log2 NPORTS)) = 25 at the defaults, plus one cycle through
// the input skid. MEASURED in sim: start -> !busy = tuples + 26 with no stalls.
//
// Intended resources at NPORTS = 3, GMAX = 96 (DERIVED by counting; NOT yet
// synthesised -- tcl/ooc_gemv_stream.tcl is the run that turns this into a
// PROJECTED number): 3 DSP48E2 (one 17x12 per lane); sign trees 3 x ~1.5k LUT
// (1,517 add/sub bits each, one LUT + carry per bit); activation banks as
// distributed RAM, 3 x 1280 b x 32 words ~ 2.2k LUT; steering, shifters, lane
// sum, accumulator, fp32 convert ~ 2k LUT. About 9k LUT / 12k FF / 3 DSP /
// 0 BRAM, i.e. ~8% of the K26's LUTs.
//
// Host contract:
//   * load the activation (act_we: one 1280-bit word of 128 10-bit two's
//     complement mantissas, weight j in bits [10j +: 10], plus the group's signed
//     BFP exponent) for groups 0..G-1 while !busy;
//   * pulse start with cfg_groups = G (1..GMAX), cfg_rows = R (>= 1), cfg_pad =
//     pad tuples to discard after the matrix (stripe alignment; usually 0);
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
    parameter         ACT_RAM_STYLE = "distributed"
) (
    input  wire                  clk,
    input  wire                  rst,
    // run control
    input  wire                  start,                 // 1-cycle pulse, only while !busy
    input  wire [7:0]            cfg_groups,            // G, 1..GMAX
    input  wire [31:0]           cfg_rows,              // R >= 1
    input  wire [31:0]           cfg_pad,               // pad tuples to discard after the matrix
    output wire                  busy,
    // activation load (while !busy)
    input  wire                  act_we,
    input  wire [7:0]            act_group,
    input  wire [1279:0]         act_mant,
    input  wire signed [EXW-1:0] act_exp,
    // weight streams, one per port
    input  wire [NPORTS-1:0]     s_valid,
    output wire [NPORTS-1:0]     s_ready,
    input  wire [NPORTS*128-1:0] s_data,
    // result stream: one fp32 per row, row order
    output wire                  m_valid,
    input  wire                  m_ready,
    output wire [31:0]           m_data,
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

    integer m_i;
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

    reg [127:0] sbeat;              // the scale beat of the stage-0 tuple
    integer     jj;
    always @* begin
        sbeat = 128'd0;
        for (jj = 0; jj < NPORTS; jj = jj + 1)
            if (s0_sp == jj) sbeat = s0_data[jj*128 +: 128];
    end
    reg [127:0] scur;               // the scale beat held from earlier tuples
    always @(posedge clk) if (s0_v && s0_hs) scur <= sbeat;

    reg s1_v, s1_rl, s1_ml;
    always @(posedge clk) begin
        if (rst) s1_v <= 1'b0; else s1_v <= s0_v;
        s1_rl <= s0_rl; s1_ml <= s0_ml;
    end

    // activation write port (registered; group -> lane = g % NPORTS, word = g / NPORTS)
    reg                  w_we;
    reg [BW-1:0]         w_bank;
    reg [AW-1:0]         w_addr;
    reg [1279:0]         w_mant;
    reg signed [EXW-1:0] w_exp;
    always @(posedge clk) begin
        if (rst) w_we <= 1'b0; else w_we <= act_we;
        w_bank <= act_group % NPORTS;
        w_addr <= act_group / NPORTS;
        w_mant <= act_mant;
        w_exp  <= act_exp;
    end

    // ------------------------------------------------------------ lanes
    wire [NPORTS*64-1:0] terms;
    genvar gm_i;
    generate for (gm_i = 0; gm_i < NPORTS; gm_i = gm_i + 1) begin : lane
        wire [BW-1:0] src  = s0_bsrc[gm_i*BW +: BW];
        wire [2:0]    sidx = s0_bsidx[gm_i*3 +: 3];
        wire [AW-1:0] radr = s0_badr[gm_i*AW +: AW];

        reg [127:0] smux;
        integer     js;
        always @* begin
            smux = 128'd0;
            for (js = 0; js < NPORTS; js = js + 1)
                if (src == js) smux = s0_data[js*128 +: 128];
        end
        wire [15:0] scsel = s0_bnew[gm_i] ? pick16(sbeat, sidx) : pick16(scur, sidx);

        // activation bank: mantissa word + exponent, synchronous read
        (* ram_style = ACT_RAM_STYLE *) reg [1279:0]   mem  [0:DEPTH-1];
        (* ram_style = "distributed" *) reg [EXW-1:0]  emem [0:DEPTH-1];
        reg [1279:0]         s1_q;
        reg signed [EXW-1:0] s1_ex;
        always @(posedge clk) begin
            if (w_we && w_bank == gm_i) begin
                mem[w_addr]  <= w_mant;
                emem[w_addr] <= w_exp;
            end
            s1_q  <= mem[radr];
            s1_ex <= emem[radr];
        end

        reg         s1_bv;
        reg [127:0] s1_sign;
        reg [15:0]  s1_sc;
        always @(posedge clk) begin
            if (rst) s1_bv <= 1'b0; else s1_bv <= s0_v & s0_bv[gm_i];
            s1_sign <= smux;
            s1_sc   <= scsel;
        end

        // sign tree: s1 -> root, TL cycles
        wire [16:0] t_d;
        wire        t_sg;
        gs_signtree #(.REGMASK(TREE_REGMASK)) u_tree (
            .clk(clk), .q(s1_q), .s(s1_sign), .d(t_d), .sg(t_sg));

        // stage 2: fp16 decode + shift count.  k = sh + 29 = e_d + e_x + 69, and
        // e_d + 69 = (exp5 == 0 ? 45 : exp5 + 44)   (fp16_decompose: e_d = -24 | exp5 - 25)
        wire [4:0]           ex5   = s1_sc[14:10];
        wire [6:0]           ebase = (ex5 == 5'd0) ? 7'd45 : ({2'b00, ex5} + 7'd44);
        wire signed [KW-1:0] ksum  = $signed({1'b0, ebase}) + s1_ex;
        reg         s2_bv, s2_sgn, s2_zero;
        reg [10:0]  s2_mag;
        reg [6:0]   s2_k;
        always @(posedge clk) begin
            if (rst) s2_bv <= 1'b0; else s2_bv <= s1_bv;
            s2_sgn  <= s1_sc[15];
            s2_mag  <= {ex5 != 5'd0, s1_sc[9:0]};
            s2_zero <= (ksum > 92);                         // sh >= 64: nothing lands in 64 bits
            s2_k    <= (ksum < 0) ? 7'd0 : ksum[6:0];       // sh <= -29: pure sign fill (floor)
        end

        // align with the tree root
        wire        r_bv, r_sgn, r_zero;
        wire [10:0] r_mag;
        wire [6:0]  r_k;
        gs_delay #(.W(1), .D(TL-1), .RST(1)) u_dv (.clk(clk), .rst(rst), .d(s2_bv), .q(r_bv));
        gs_delay #(.W(20), .D(TL-1)) u_dd (.clk(clk), .rst(rst),
            .d({s2_sgn, s2_zero, s2_mag, s2_k}), .q({r_sgn, r_zero, r_mag, r_k}));

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
        assign terms[gm_i*64 +: 64] = term;
    end endgenerate

    // ------------------------------------------------------------ lane sum + row accumulator
    wire [63:0] lsum;
    gs_lanesum #(.N(NPORTS), .W(64)) u_ls (.clk(clk), .d(terms), .s(lsum));

    wire a_v, a_rl, a_ml;           // slot flags aligned with lsum: S1 + TL + 5 + LS
    gs_delay #(.W(1), .D(TL + 5 + LS), .RST(1)) u_av (.clk(clk), .rst(rst), .d(s1_v), .q(a_v));
    gs_delay #(.W(2), .D(TL + 5 + LS)) u_ad (.clk(clk), .rst(rst), .d({s1_rl, s1_ml}), .q({a_rl, a_ml}));

    reg [63:0] acc, res;
    reg        res_v, res_ml;
    wire [63:0] acc_nx = acc + lsum;
    always @(posedge clk) begin
        if (rst) begin
            acc <= 64'd0; res_v <= 1'b0;
        end else begin
            res_v <= a_v & a_rl;
            if (a_v) acc <= a_rl ? 64'd0 : acc_nx;
        end
        res    <= acc_nx;
        res_ml <= a_ml;
    end

    // ------------------------------------------------------------ int64 -> fp32, output FIFO
    wire        f_v, f_l;
    wire [31:0] f_d;
    gs_i64_to_f32 u_cv (.clk(clk), .rst(rst), .in_valid(res_v), .in_last(res_ml), .in_data(res),
                        .out_valid(f_v), .out_last(f_l), .out_data(f_d));

    wire [32:0] fo;
    gs_fifo #(.W(33), .LG(OFIFO_LG)) u_of (.clk(clk), .rst(rst), .wr(f_v), .din({f_l, f_d}),
                                           .m_valid(m_valid), .m_ready(m_ready), .m_data(fo));
    assign m_data   = fo[31:0];
    assign m_last   = fo[32];
    assign fifo_pop = m_valid & m_ready;
endmodule

`default_nettype wire
