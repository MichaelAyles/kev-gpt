// AXI4 read-slave model for the R0 probe gate. In-order returns, fixed latency
// LAT from AR accept to first beat, optional random arready / rvalid gaps
// (GAP_PCT percent of cycles stalled). Data is a pure function of the beat
// address so the Python gate can recompute the XOR checksum:
//   lane j (32b) of the 16-byte beat at byte address a:
//       ((a + 4j) * 32'h9E3779B1) ^ a[48:32]
// Protocol checks: no burst crosses 4 KB, arsize==16B, INCR, and the number of
// accepted-but-unfinished bursts never exceeds MAXOUT_CHECK.
`timescale 1ns / 1ps

module axi_rd_slave_model #(
    parameter integer AW = 49,
    parameter integer IDW = 6,
    parameter integer LAT = 40,
    parameter integer GAP_PCT = 0,
    parameter integer SEED = 1,
    parameter integer QD = 512
) (
    input  wire            clk,
    input  wire            rst,
    input  wire [31:0]     maxout_check,
    input  wire [IDW-1:0]  arid,
    input  wire [AW-1:0]   araddr,
    input  wire [7:0]      arlen,
    input  wire [2:0]      arsize,
    input  wire [1:0]      arburst,
    input  wire            arvalid,
    output reg             arready,
    output reg  [IDW-1:0]  rid,
    output reg  [127:0]    rdata,
    output reg  [1:0]      rresp,
    output reg             rlast,
    output reg             rvalid,
    input  wire            rready,
    output reg  [31:0]     errors
);
    // queue of accepted bursts
    reg [AW-1:0]  q_addr [0:QD-1];
    reg [8:0]     q_len  [0:QD-1];
    reg [IDW-1:0] q_id   [0:QD-1];
    reg [63:0]    q_t    [0:QD-1];   // cycle at which data may start
    integer qh, qt, inflight;
    reg [63:0] now;
    reg [8:0]  beat;
    integer seed;
    function integer rpct; input dummy; begin rpct = ($random(seed) & 32'h7fffffff) % 100; end endfunction

    function [127:0] pat(input [AW-1:0] a);
        integer j;
        reg [31:0] lane;
        begin
            for (j = 0; j < 4; j = j + 1) begin
                lane = ((a[31:0] + 4*j) * 32'h9E3779B1) ^ {15'b0, a[48:32]};
                pat[j*32 +: 32] = lane;
            end
        end
    endfunction

    reg [AW-1:0] cur_addr;
    always @(posedge clk) begin
        if (rst) begin
            qh = 0; qt = 0; inflight = 0; now = 0; beat = 0;
            arready <= 1'b0; rvalid <= 1'b0; rlast <= 1'b0; rresp <= 2'b00;
            rid <= 0; rdata <= 0; errors <= 0;
            seed = SEED;
        end else begin
            now = now + 1;
            // ---- AR accept ----
            if (arvalid && arready) begin
                if (arsize != 3'b100 || arburst != 2'b01) begin
                    errors <= errors + 1; $display("SLAVE_ERR bad size/burst");
                end
                if ((araddr[11:0] + ((arlen + 1) * 16)) > 4096) begin
                    errors <= errors + 1; $display("SLAVE_ERR 4KB cross addr=%h len=%0d", araddr, arlen + 1);
                end
                q_addr[qt % QD] = araddr; q_len[qt % QD] = arlen + 1;
                q_id[qt % QD] = arid; q_t[qt % QD] = now + LAT;
                qt = qt + 1; inflight = inflight + 1;
                if (inflight > maxout_check) begin
                    errors <= errors + 1; $display("SLAVE_ERR inflight %0d > maxout %0d", inflight, maxout_check);
                end
            end
            arready <= ((qt - qh) < QD - 2) && (GAP_PCT == 0 || rpct(0) >= GAP_PCT);

            // ---- R beats ----
            if (rvalid && rready) begin
                if (rlast) begin
                    qh = qh + 1; inflight = inflight - 1; beat = 0;
                end else beat = beat + 1;
            end
            if (qh != qt && q_t[qh % QD] <= now &&
                (GAP_PCT == 0 || rpct(0) >= GAP_PCT)) begin
                cur_addr = q_addr[qh % QD] + beat * 16;
                rvalid <= 1'b1;
                rid    <= q_id[qh % QD];
                rdata  <= pat(cur_addr);
                rresp  <= 2'b00;
                rlast  <= (beat + 1 == q_len[qh % QD]);
            end else begin
                rvalid <= 1'b0; rlast <= 1'b0;
            end
        end
    end
endmodule
