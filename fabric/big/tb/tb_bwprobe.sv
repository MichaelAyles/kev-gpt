// tb_bwprobe — drives bwprobe_axi over AXI-Lite exactly like the board driver,
// with one axi_rd_slave_model per master. Config via plusargs; prints one
// PROBE_M line per master and PROBE_G for the global counter.
`timescale 1ns / 1ps

module tb_bwprobe;
    localparam integer NM = 6, AW = 49, IDW = 6;
    parameter integer LAT = 40;
    parameter integer GAP_PCT = 0;

    reg clk = 0, aresetn = 0;
    always #2 clk = ~clk;

    reg [9:0]  s_awaddr = 0, s_araddr = 0;
    reg        s_awvalid = 0, s_wvalid = 0, s_bready = 1, s_arvalid = 0, s_rready = 1;
    reg [31:0] s_wdata = 0;
    wire       s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
    wire [1:0] s_bresp, s_rresp;
    wire [31:0] s_rdata;

    reg [31:0] maxout_cfg;

    // per-master wires
    wire [IDW-1:0] arid [0:NM-1];
    wire [AW-1:0]  araddr [0:NM-1];
    wire [7:0]     arlen [0:NM-1];
    wire [2:0]     arsize [0:NM-1];
    wire [1:0]     arburst [0:NM-1];
    wire           arvalid [0:NM-1], arready [0:NM-1];
    wire [IDW-1:0] rid [0:NM-1];
    wire [127:0]   rdata [0:NM-1];
    wire [1:0]     rresp [0:NM-1];
    wire           rlast [0:NM-1], rvalid [0:NM-1], rready [0:NM-1];
    wire [31:0]    serr [0:NM-1];

    `include "tb_bwprobe_inst.svh"

    genvar g;
    generate for (g = 0; g < NM; g = g + 1) begin : sl
        axi_rd_slave_model #(.AW(AW), .IDW(IDW), .LAT(LAT + 7*g), .GAP_PCT(GAP_PCT),
                             .SEED(11 + g)) u_s (
            .clk(clk), .rst(~aresetn), .maxout_check(maxout_cfg),
            .arid(arid[g]), .araddr(araddr[g]), .arlen(arlen[g]), .arsize(arsize[g]),
            .arburst(arburst[g]), .arvalid(arvalid[g]), .arready(arready[g]),
            .rid(rid[g]), .rdata(rdata[g]), .rresp(rresp[g]), .rlast(rlast[g]),
            .rvalid(rvalid[g]), .rready(rready[g]), .errors(serr[g]));
    end endgenerate

    task axil_wr(input [9:0] a, input [31:0] d);
        begin
            @(posedge clk);
            s_awaddr <= a; s_wdata <= d; s_awvalid <= 1; s_wvalid <= 1;
            @(posedge clk);
            while (!s_awready) @(posedge clk);
            s_awvalid <= 0; s_wvalid <= 0;
            while (!s_bvalid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    task axil_rd(input [9:0] a, output [31:0] d);
        begin
            @(posedge clk);
            s_araddr <= a; s_arvalid <= 1;
            @(posedge clk);
            while (!s_arready) @(posedge clk);
            s_arvalid <= 0;
            while (!s_rvalid) @(posedge clk);
            d = s_rdata;
            @(posedge clk);
        end
    endtask

    integer i, beats, burst, maxout, nids, enable, timeout;
    reg [31:0] v, lo, hi, x0, x1, x2, x3, st;
    reg [63:0] base;
    reg [31:0] stride, step;

    initial begin
        if (!$value$plusargs("beats=%d", beats))   beats = 4096;
        if (!$value$plusargs("burst=%d", burst))   burst = 16;
        if (!$value$plusargs("maxout=%d", maxout)) maxout = 8;
        if (!$value$plusargs("nids=%d", nids))     nids = 1;
        if (!$value$plusargs("enable=%d", enable)) enable = 63;
        if (!$value$plusargs("base=%h", base))     base = 64'h8_0000_0000;
        if (!$value$plusargs("stride=%h", stride)) stride = 32'h0100_0000;
        if (!$value$plusargs("step=%h", step))     step = 0;
        maxout_cfg = maxout;
        repeat (10) @(posedge clk);
        aresetn = 1;
        repeat (5) @(posedge clk);

        axil_rd(10'h030, v);
        $display("IDCODE %h", v);
        axil_wr(10'h008, enable);
        axil_wr(10'h00C, base[31:0]);
        axil_wr(10'h010, base[63:32]);
        axil_wr(10'h014, stride);
        axil_wr(10'h018, beats);
        axil_wr(10'h01C, burst);
        axil_wr(10'h020, maxout);
        axil_wr(10'h024, nids);
        axil_wr(10'h03C, step);
        axil_wr(10'h000, 1);

        timeout = 0;
        st = 0;
        while (!((st[NM-1:0] | ~enable[NM-1:0]) == {NM{1'b1}} && !st[16]) && timeout < 2000000) begin
            repeat (50) @(posedge clk);
            axil_rd(10'h004, st);
            timeout = timeout + 50;
        end
        if (timeout >= 2000000) $display("PROBE_TIMEOUT status=%h", st);

        axil_rd(10'h028, lo); axil_rd(10'h02C, hi);
        $display("PROBE_G gcyc=%0d", {hi, lo});
        for (i = 0; i < NM; i = i + 1) begin
            axil_rd(10'h100 + i*32 + 0, lo);
            axil_rd(10'h100 + i*32 + 4, hi);
            axil_rd(10'h100 + i*32 + 8, v);
            axil_rd(10'h100 + i*32 + 12, st);
            axil_rd(10'h100 + i*32 + 16, x0);
            axil_rd(10'h100 + i*32 + 20, x1);
            axil_rd(10'h100 + i*32 + 24, x2);
            axil_rd(10'h100 + i*32 + 28, x3);
            $display("PROBE_M m=%0d cyc=%0d beats=%0d rresp_err=%0d slave_err=%0d xsum=%h%h%h%h",
                     i, {hi, lo}, v, st, serr[i], x3, x2, x1, x0);
        end
        $finish;
    end
endmodule
