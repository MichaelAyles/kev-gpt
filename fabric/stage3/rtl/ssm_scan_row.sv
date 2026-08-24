// ssm_scan_row.sv — wide Mamba-2 scan core: one full state row (N lanes) per
// cycle (doc 9 §4/§5, the throughput version of the silicon-proven ssm_scan).
//
// Same arithmetic as ssm_scan v5 / mamba2_fixed_scan.scan_step_fixed2:
//   h[p][n] <- rnd(( a*h + (dtx[p]*B[n]) << sh_i ) >>> 16)   [INT16, sat]
//   y[p]     = sat(( sum_n h'[p][n]*C[n] ) >>[rnd] sh_y)     [INT16]
//
// Layout: the doc-6 wide-word rule — state is ONE N*16-bit word per channel
// (reg [N*16-1:0] h [0:P-1]); the p index is a memory address, never a mux.
// Per token per head: P + pipe-depth cycles (64+8 at P=N=64) vs the
// sequential core's P*N (4096).
//
// Pipeline: S0 read row p | S1 lane products (a*h, dtx*B) | S2 fused update
// (single rounding) + row writeback | S3.. y adder tree (6 levels, 3 stages)
// | S_y shift+sat+write. Row sequence strictly increasing => the 2-cycle
// read-to-write lag never collides.

`default_nettype none

module ssm_scan_row #(
    parameter int P  = 64,               // rows per CALL (one head-slice)
    parameter int N  = 64,
    parameter int QA = 16,
    parameter int QH = 16,               // STORED state bits/lane (the doc-9 state-
                                         // quant lever). 16 = bit-exact INT16; 12 =
                                         // INT12 (keep range, drop low 4 bits) to
                                         // shrink the scan-h BRAM ~25% for high NC.
                                         // N*QH must be a multiple of 64 (QH even).
    parameter int CTX = 1,               // state contexts (layers*heads)
    parameter int RDP = 4                // wide-read lanes/cycle (SCAN_RD widen)
) (
    // which context this call updates: state rows [pbase*P .. +P)
    input  wire [$clog2(CTX)-1:0]  pbase,
    input  wire                 clk,
    input  wire                 rst,
    output reg                  ready,
    input  wire                 start,
    output reg                  done,

    input  wire [15:0]          a_q,
    input  wire signed [5:0]    sh_i,
    input  wire signed [5:0]    sh_y,

    input  wire                 wr_dtx,
    input  wire [$clog2(P)-1:0] wr_dtx_addr,
    input  wire signed [15:0]   wr_dtx_data,
    input  wire                 wr_b,
    input  wire [$clog2(N)-1:0] wr_b_addr,
    input  wire signed [7:0]    wr_b_data,
    input  wire                 wr_c,
    input  wire [$clog2(N)-1:0] wr_c_addr,
    input  wire signed [7:0]    wr_c_data,

    input  wire [$clog2(P)-1:0] rd_y_addr,
    output wire signed [15:0]   rd_y_data,

    // WIDE read: RDP consecutive y outputs from rd_yw_base in one cycle, so the
    // engine's SCAN_RD (y-readback + D-skip) can consume RDP/cycle. yout is a FF
    // array -> free combinational fan-out (same values as rd_y). Combinational.
    input  wire [$clog2(P)-1:0] rd_yw_base,
    output wire [RDP*16-1:0]    rd_yw_data
,
    // Same question for the scan, whose output reads EXACTLY ZERO on silicon:
    // what does it actually receive?
    output reg signed [31:0]    dbg_dsum, dbg_bsum, dbg_csum
);
    localparam int PW = $clog2(P);
    localparam int AW = (CTX > 1) ? $clog2(CTX*P) : PW;

    // state: one wide word per channel (lane n in bits [n*16 +: 16]);
    // CTX contexts stacked. Vivado won't shard a single 1024-bit RTL object
    // across parallel URAMs (it demotes the whole thing to LUTRAM + a giant
    // read mux), so the flat word is genvar-banked into NB x 64-bit block-RAM
    // slices — same raddr/waddr/we/wdata drivers, hq reassembled from banks.
    // Read latency stays 1 cycle; rows are strictly increasing so the write-
    // lags-read window never collides. Constant genvar part-selects are
    // iverilog-safe. N*16 must be a multiple of 64 (N=64 -> 16 banks of 64b).
    // STORED state is QH bits/lane (QH<=16 = the state-quant lever). The a*h
    // multiply reconstructs a 16-bit value (stored << SHW), so the recurrence
    // arithmetic is unchanged INT16; only the CARRIED state loses its low SHW
    // bits (range preserved) — the quantize-the-carry contract of probe_state_
    // quant. QH=16 => SHW=0 => bit-identical to the INT16 core.
    localparam int SHW = 16 - QH;
    localparam int NB = (N*QH)/64;
    reg [AW-1:0]   raddr;
    wire [N*QH-1:0] hq;
    reg            we;
    reg [AW-1:0]   waddr;
    reg [N*QH-1:0] wdata;

    genvar hbi;
    generate for (hbi = 0; hbi < NB; hbi = hbi + 1) begin : hbank
        (* ram_style = "block" *)
        reg [63:0] mem [0:CTX*P-1];
        reg [63:0] q;
        always @(posedge clk) begin
            q <= mem[raddr];
            if (we) mem[waddr] <= wdata[hbi*64 +: 64];
        end
        assign hq[hbi*64 +: 64] = q;
    end endgenerate

    reg signed [15:0] dtx [0:P-1];
    reg signed [7:0]  bvec[0:N-1];
    reg signed [7:0]  cvec[0:N-1];
    reg signed [15:0] yout[0:P-1];
    assign rd_y_data = yout[rd_y_addr];

    // RDP-wide combinational read. SCAN_RD reads P (=64), an RDP-multiple.
    genvar gyr;
    generate for (gyr = 0; gyr < RDP; gyr = gyr + 1) begin : g_yrdw
        assign rd_yw_data[gyr*16 +: 16] = yout[rd_yw_base + gyr[PW-1:0]];
    end endgenerate

    always @(posedge clk) begin
        if (rst) begin
            dbg_dsum <= 32'sd0; dbg_bsum <= 32'sd0; dbg_csum <= 32'sd0;
        end else begin
            if (wr_dtx) begin dtx [wr_dtx_addr] <= wr_dtx_data;
                              dbg_dsum <= dbg_dsum + $signed(wr_dtx_data); end
            if (wr_b)   begin bvec[wr_b_addr]   <= wr_b_data;
                              dbg_bsum <= dbg_bsum + $signed(wr_b_data); end
            if (wr_c)   begin cvec[wr_c_addr]   <= wr_c_data;
                              dbg_csum <= dbg_csum + $signed(wr_c_data); end
        end
`ifdef ROW_TRACE
        if (wr_b && wr_b_addr < 2) $display("RTW B[%0d]=%0d", wr_b_addr, wr_b_data);
        if (wr_c && wr_c_addr < 2) $display("RTW C[%0d]=%0d", wr_c_addr, wr_c_data);
`endif
    end

    // ---- control ----
    localparam [1:0] CLEARING = 2'd0, IDLE = 2'd1, RUN = 2'd2, DRAIN = 2'd3;
    reg [1:0]    st;
    reg [PW-1:0] pi;
    reg [AW-1:0] clr_addr;
    wire [AW-1:0] ctx_off = pbase * P;
    wire issue = (st == RUN);

    // ---- pipeline (one op per stage; II=1, write lags read by 4) ----
    reg              v1, v2, v3, v4;
    reg [PW-1:0]     p1, p2, p3, p4;
    reg signed [15:0] dtx1;
    reg signed [23:0] injl  [0:N-1];      // S1: dtx*B per lane
    reg signed [32:0] mull  [0:N-1];      // S1: a*h per lane
    reg signed [41:0] injs  [0:N-1];      // S2: barrel-shifted inject
    reg signed [32:0] mul2l [0:N-1];      // S2: delayed a*h
    reg signed [15:0] hnl   [0:N-1];      // S3: saturated new state
    reg signed [23:0] yprodl[0:63];       // S4: hnew * C (fixed 64 lanes: the
                                          // y adder tree below is a 64->16->4->1
                                          // reduction, so lanes N..63 are held at
                                          // 0 for N<64 — bit-exact, adds nothing)

    wire signed [16:0] a_s = {1'b0, a_q};

    // S3 combinational per lane: fused single rounding + sat. hnew_w is the full
    // INT16 new state (feeds y, unquantised — y uses full precision). hstore_w is
    // the QH-bit CARRIED state (round hn to a 2^SHW grid, sat to QH bits) that gets
    // written back and reconstructed (<<SHW) on the next step's a*h read.
    localparam signed [16:0] QHMAX = (17'sd1 <<< (QH-1)) - 17'sd1;
    localparam signed [16:0] QHMIN = -(17'sd1 <<< (QH-1));
    wire [N*16-1:0]  hnew_w;
    wire [N*QH-1:0]  hstore_w;
    genvar g;
    generate for (g = 0; g < N; g = g + 1) begin : lane
        wire signed [42:0] acc  = mul2l[g] + injs[g];
        wire signed [42:0] accr = acc + $signed(43'sd1 <<< (QA - 1));
        wire signed [26:0] shft = accr >>> QA;
        wire signed [15:0] hn = (shft > $signed(27'sd32767))  ? 16'sd32767 :
                                (shft < -$signed(27'sd32768)) ? -16'sd32768 :
                                shft[15:0];
        assign hnew_w[g*16 +: 16] = hn;
        // round-to-nearest into the QH grid, then saturate to QH bits
        wire signed [16:0] hn_r = (SHW > 0)
            ? ($signed({hn[15], hn}) + ($signed(17'sd1) <<< (SHW-1))) >>> SHW
            : $signed({hn[15], hn});
        wire signed [QH-1:0] hn_q = (hn_r > QHMAX) ? QHMAX[QH-1:0] :
                                    (hn_r < QHMIN) ? QHMIN[QH-1:0] : hn_r[QH-1:0];
        assign hstore_w[g*QH +: QH] = hn_q;
    end endgenerate

    // read reconstruction: stored QH-bit value sign-extended to 16b then <<SHW,
    // so the a*h multiply sees the same INT16-grid value the reference carries.
    wire [N*16-1:0] hrec_w;
    generate for (g = 0; g < N; g = g + 1) begin : lrec
        assign hrec_w[g*16 +: 16] =
            {{(16-QH){hq[g*QH + QH - 1]}}, hq[g*QH +: QH]} << SHW;
    end endgenerate
    wire [N*16-1:0] hnl_w;
    generate for (g = 0; g < N; g = g + 1) begin : lanew
        assign hnl_w[g*16 +: 16] = hnl[g];
    end endgenerate

    // y adder tree: 64 -> 16 -> 4 -> 1, one level-pair per stage
    reg               v5, v6, v7, v8, v9;
    reg [PW-1:0]      p5, p6, p7, p8, p9;
    reg signed [27:0] t16 [0:15];
    reg signed [29:0] t4  [0:3];
    reg signed [31:0] ysum5;
    reg signed [15:0] yfin6;

    wire signed [31:0] ysx   = ysum5;
    wire signed [31:0] yrnd  = (sh_y > 0)
                             ? ysx + ($signed(32'sd1) <<< (sh_y - 1)) : ysx;
    wire signed [31:0] yshft = (sh_y >= 0) ? (yrnd >>> sh_y)
                                           : (ysx <<< -sh_y);
    wire signed [15:0] yfin  = (yshft > $signed(32'sd32767))  ? 16'sd32767 :
                               (yshft < -$signed(32'sd32768)) ? -16'sd32768 :
                               yshft[15:0];

    integer k;
    always @(posedge clk) begin
        done <= 1'b0;
        we   <= 1'b0;
        if (rst) begin
            st <= CLEARING; ready <= 1'b0; clr_addr <= '0;
            v1 <= 0; v2 <= 0; v3 <= 0; v4 <= 0; v5 <= 0; v6 <= 0;
            v7 <= 0; v8 <= 0; v9 <= 0;
        end else begin
            case (st)
              CLEARING: begin
                we <= 1'b1; waddr <= clr_addr; wdata <= '0;
                if (clr_addr == {AW{1'b1}}) begin st <= IDLE; ready <= 1'b1; end
                clr_addr <= clr_addr + 1'b1;
              end
              IDLE: if (start) begin st <= RUN; pi <= '0; raddr <= ctx_off; end
              RUN: begin
                raddr <= ctx_off + pi + 1'b1;
                if (pi == P-1) st <= DRAIN; else pi <= pi + 1'b1;
              end
              DRAIN: if (!v1 && !v2 && !v3 && !v4 && !v5 && !v6 && !v7
                          && !v8 && !v9) begin
                st <= IDLE; done <= 1'b1;
              end
            endcase

            // S0 -> S1: row read lands in hq; register per-lane products
            v1 <= issue; p1 <= pi; dtx1 <= dtx[pi];
            // S1 -> S2: products
            v2 <= v1; p2 <= p1;
            for (k = 0; k < N; k = k + 1) begin
                mull[k] <= a_s * $signed(hrec_w[k*16 +: 16]);
                injl[k] <= dtx1 * bvec[k];
            end
            // S2 -> S3: barrel shift only
            v3 <= v2; p3 <= p2;
            for (k = 0; k < N; k = k + 1) begin
                mul2l[k] <= mull[k];
                injs[k]  <= (sh_i >= 0)
                          ? ($signed({{18{injl[k][23]}}, injl[k]}) <<< sh_i)
                          : (($signed({{18{injl[k][23]}}, injl[k]})
                              + ($signed(42'sd1) <<< (-sh_i - 1))) >>> -sh_i);
            end
            // S3 -> S4: register saturated state; write back
            v4 <= v3; p4 <= p3;
            for (k = 0; k < N; k = k + 1)
                hnl[k] <= hnew_w[k*16 +: 16];
            if (v3) begin we <= 1'b1; waddr <= ctx_off + p3; wdata <= hstore_w; end
            // S4 -> S5: y products (lanes N..63 forced 0 so the 64-wide adder
            // tree reduces correctly for any N <= 64)
            v5 <= v4; p5 <= p4;
            for (k = 0; k < N; k = k + 1)
                yprodl[k] <= hnl[k] * cvec[k];
            for (k = N; k < 64; k = k + 1)
                yprodl[k] <= 24'sd0;
`ifdef ROW_TRACE
            if (v4 && p4 == 0)
                $display("RT p0: hq0=%0d mull0=%0d injs0=%0d hnl0=%0d cvec0=%0d yprod0=%0d",
                         $signed(hq[15:0]), mull[0], injs[0],
                         hnl[0], cvec[0], yprodl[0]);
`endif
            // S5: tree 64 -> 16
            v6 <= v5; p6 <= p5;
            if (v5)
                for (k = 0; k < 16; k = k + 1)
                    t16[k] <= yprodl[4*k] + yprodl[4*k+1]
                            + yprodl[4*k+2] + yprodl[4*k+3];
            // S6: 16 -> 4
            v7 <= v6; p7 <= p6;
            if (v6)
                for (k = 0; k < 4; k = k + 1)
                    t4[k] <= t16[4*k] + t16[4*k+1] + t16[4*k+2] + t16[4*k+3];
            // S7: 4 -> 1
            v8 <= v7; p8 <= p7;
            if (v7) ysum5 <= t4[0] + t4[1] + t4[2] + t4[3];
            // S8: shift + sat; S9 write
            v9 <= v8; p9 <= p8;
            if (v8) yfin6 <= yfin;
            if (v9) yout[p9] <= yfin6;
        end
    end

endmodule

`default_nettype wire
