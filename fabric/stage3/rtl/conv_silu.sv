// conv_silu.sv — Mamba-2 depthwise conv1d (k taps) + SiLU LUT (doc 9 §4).
//
// Per channel c: y[c] = silu( sum_{j<k} shift[c][j]*w[c][j] + bias[c] )
// where shift[c] is the (k-1)-deep history + the incoming sample (decode
// mode: one new sample per channel per token).
//
// Formats (contract = mamba2_fixed at LUT resolution):
//   x in       : INT16 Q6.9  (the dequantized in_proj xBC slice reaches ~33,
//                which Q3.12 (+-8) clipped — the full-engine saturation bug)
//   w, bias    : INT16 Q1.14 (conv weights are small; exported per channel)
//   MAC        : product Q6.9 x Q1.14 = frac 23 -> rnd >> 12 -> Q4.11
//   SiLU       : 256-entry LUT over [-4,4) Q4.11 input (top 8 bits + sat),
//                INT16 Q4.11 out (silu out reaches ~10, Q3.12 clipped it);
//                |x|>=4 tails: y=x (pos) / 0 (neg). LUT rows are Q3.12 -> >>1.
//
// Sequential: one channel per cycle (CONV_DIM cycles/token) — conv is ~2% of
// the token budget; wide comes later if ever needed.

`default_nettype none

module conv_silu #(
    parameter int CH = 640,          // d_inner + 2*d_state
    parameter int K  = 4,
    parameter int L  = 7,            // layers: one persistent history bank each
    parameter int QF = 12,           // Q3.12 activations
    parameter int RDP = 4            // wide-read lanes/cycle (CONV_RD widen)
) (
    input  wire                  clk,
    input  wire                  rst,       // clears the shift history
    input  wire                  start,
    output reg                   done,
    output reg                   ready,      // 1 once the initial clear finished

    // per-layer history bank select: hist for this token's conv lives in
    // bank `layer` (rows layer*CH .. layer*CH+CH-1), persistent across tokens.
    // Constant for the duration of one start..done run.
    input  wire [$clog2(L)-1:0]  layer,

    // weight/bias load (once): addr = c*K+j for weights, c for bias
    input  wire                  wr_w,
    input  wire [$clog2(CH*K)-1:0] wr_w_addr,
    input  wire signed [15:0]    wr_w_data,
    input  wire                  wr_b,
    input  wire [$clog2(CH)-1:0] wr_b_addr,
    input  wire signed [15:0]    wr_b_data,

    // per-token input samples, written before start
    input  wire                  wr_x,
    input  wire [$clog2(CH)-1:0] wr_x_addr,
    input  wire signed [15:0]    wr_x_data,

    // silu LUT init
    input  wire                  wr_lut,
    input  wire [7:0]            wr_lut_addr,
    input  wire signed [15:0]    wr_lut_data,

    input  wire [$clog2(CH)-1:0] rd_y_addr,
    output wire signed [15:0]    rd_y_data,

    // WIDE read: RDP consecutive outputs from rd_yw_base in one cycle, so the
    // engine's CONV_RD stream can consume RDP/cycle (yout is a FF array, so this
    // is a free combinational fan-out — same values as rd_y). Combinational.
    input  wire [$clog2(CH)-1:0] rd_yw_base,
    output wire [RDP*16-1:0]     rd_yw_data
,

    // doc-9c instrument: checksum of every y value at the moment it is
    // WRITTEN into yout. Compared against the engine's read-back sum, it
    // separates 'the conv computed the wrong value' from 'the conv computed
    // the right value and the read port returned something else'. Costs one
    // adder; adds no read port, so memory inference is untouched.
    output reg signed [31:0]     dbg_ysum
);
    localparam int CW  = $clog2(CH);
    localparam int LCH = L*CH;               // total history rows across banks
    localparam int AW  = $clog2(LCH);

    // history: one wide word per channel (K-1 samples of 16b), doc-6 layout.
    // PER-LAYER banks: row = layer*CH + ch, so the variable row is a memory
    // ADDRESS (not a per-lane mux). Cleared once at rst (all L banks), then each
    // layer reads/updates ONLY its own bank, carried across tokens. ram_style
    // "block" + the deferred writeback below (write the shifted history one
    // cycle later, off the S0-registered copies) keeps read and write on
    // distinct addresses so this infers a simple-dual-port BRAM, not FFs.
    (* ram_style = "block" *)
    reg [(K-1)*16-1:0] hist [0:LCH-1];
    reg signed [15:0]  xin  [0:CH-1];
    reg signed [15:0]  wrom [0:CH*K-1];
    reg signed [15:0]  brom [0:CH-1];
    reg signed [15:0]  lut  [0:255];
    reg signed [15:0]  yout [0:CH-1];

    assign rd_y_data = yout[rd_y_addr];

    // RDP-wide combinational read (whole-element indexed, iverilog-safe). base+g
    // stays in-bounds: CONV_RD reads CH, an RDP-multiple.
    genvar gr;
    generate for (gr = 0; gr < RDP; gr = gr + 1) begin : g_yrdw
        assign rd_yw_data[gr*16 +: 16] = yout[rd_yw_base + gr[$clog2(CH)-1:0]];
    end endgenerate

    always @(posedge clk) begin
        if (wr_w)   wrom[wr_w_addr] <= wr_w_data;
        if (wr_b)   brom[wr_b_addr] <= wr_b_data;
        if (wr_x)   xin [wr_x_addr] <= wr_x_data;
        if (wr_lut) lut [wr_lut_addr] <= wr_lut_data;
    end

    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, DRAIN = 2'd2;
    reg [1:0]  st;
    reg [CW-1:0] ci;
    reg [AW-1:0] clr;                        // clears all L*CH history rows
    reg        clearing;
    wire issue = (st == RUN);

    // history row address for the current channel in the current layer's bank
    wire [AW-1:0] haddr = layer*CH + ci;

    // hist is SDP BRAM: ONE read port (haddr) + ONE write port. The clear
    // sweep and the deferred per-channel shift share that single write port via
    // a mux (they never fire the same cycle: v1 requires st==RUN, which only
    // follows clearing) — otherwise two write addresses + the read = 3 ports,
    // which a BRAM can't provide and synthesis refuses to infer.

    // S0: read channel operands -> regs
    reg               v1;
    reg [CW-1:0]      c1;
    reg [(K-1)*16-1:0] h1;
    reg signed [15:0] x1, b1;
    reg signed [15:0] w1 [0:K-1];

    // S1: MAC across K taps (K small: one adder tree level)
    reg               v2;
    reg [CW-1:0]      c2;
    reg signed [35:0] acc2;

    integer j;
    // S2 comb: round >> 12 (frac 23 -> Q4.11), sat, LUT index
    wire signed [21:0] pre  = (acc2 + $signed(36'sd1 <<< 11)) >>> 12;
    wire signed [15:0] prs  = (pre > $signed(22'sd32767))  ? 16'sd32767 :
                              (pre < -$signed(22'sd32768)) ? -16'sd32768 :
                              pre[15:0];
    // LUT domain [-4,4) in Q4.11: 4.0 = 8192; index = top 8 bits of prs+8192.
    wire signed [16:0] biased = prs + $signed(17'sd8192);
    wire [7:0] idx = (biased < 0) ? 8'd0 :
                     (biased > $signed(17'sd16383)) ? 8'd255 : biased[13:6];
    // LUT rows are Q3.12 silu values -> >>1 (rnd) to Q4.11; tails carry Q4.11 prs
    wire signed [15:0] ylut = (prs >= $signed(16'sd8192)) ? prs :
                              (prs < -$signed(16'sd8192)) ? 16'sd0 :
                              ($signed(lut[idx]) + 16'sd1) >>> 1;

    // single write port for hist (clear sweep OR deferred shift, never both)
    wire                hist_we    = clearing | v1;
    wire [AW-1:0]       hist_waddr = clearing ? clr : (layer*CH + c1);
    wire [(K-1)*16-1:0] hist_wdata = clearing ? {(K-1)*16{1'b0}}
                                              : {x1, h1[(K-1)*16-1:16]};

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            st <= IDLE; v1 <= 0; v2 <= 0; clearing <= 1'b1; clr <= '0;
            ready <= 1'b0;
            dbg_ysum <= 32'sd0;
        end else begin
            if (clearing) begin
                if (clr == LCH-1) begin clearing <= 1'b0; ready <= 1'b1; end
                clr <= clr + 1'b1;
            end
            // hist SDP write port (muxed clear-sweep / deferred-shift)
            if (hist_we) hist[hist_waddr] <= hist_wdata;
            case (st)
              IDLE: if (start && !clearing) begin st <= RUN; ci <= '0; end
              RUN:  begin
                if (ci == CH-1) st <= DRAIN;
                ci <= ci + 1'b1;
              end
              DRAIN: if (!v1 && !v2) begin st <= IDLE; done <= 1'b1; end
            endcase

            // S0 -> S1
            v1 <= issue; c1 <= ci;
            h1 <= hist[haddr]; x1 <= xin[ci]; b1 <= brom[ci];
            for (j = 0; j < K; j = j + 1) w1[j] <= wrom[ci*K + j];
            // (the deferred history shift is applied through the single hist
            // write port above: bit-identical, the row is not re-read until the
            // next token so a 1-cycle-late writeback is invisible.)

            // S1 -> S2: K-tap MAC (K=4: 3 taps from history + current)
            v2 <= v1; c2 <= c1;
            acc2 <= $signed(h1[15:0])  * w1[0]
                  + $signed(h1[31:16]) * w1[1]
                  + $signed(h1[47:32]) * w1[2]
                  + x1 * w1[3]
                  + (b1 <<< 9);        // bias Q1.14 -> frac 23 (products Q6.9 x Q1.14)

            // S2: LUT + write
            if (v2) begin
                yout[c2] <= ylut;
                dbg_ysum <= dbg_ysum + $signed(ylut);
            end
        end
    end

endmodule

`default_nettype wire
