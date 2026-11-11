// tb_gemv_axi — drives rtl/gemv_axi.v exactly as the board driver does: an AXI-Lite BFM
// replays a register script written by run_gemv_axi.py, one axi_rd_mem_slave per read
// master serves the packed weight image, and one axi_rw_mem_slave on aux_axi holds the
// activation records and receives the result rows (register map version 2; version 3 adds
// the NB parameter -- everything batch-specific is in the register script and the images).
//
// Script (+script=<file>, +nops=<n>): $readmemh, 4 x 32-bit hex words per op
//   0 a d -      write register a <- d
//   1 a - -      read register a                         log: R <a> <data>
//   2 a m v      poll a until (data & m) == v, +pollgap idle cycles between reads, at
//                most +polltimeout reads                 log: P <a> <last data> <reads>
//   3 n - -      wait n cycles
//   5 a d -      write register a <- d with no testbench side effect (a GO that the
//                design must ignore, or a run that will be aborted)
//   4 tag - -    mark: dump the bus-side events of the run that the last GO started
//                log: M <tag> <A> <B> <T> <L> <n_ar> <n_beats>
//                         <Aa> <Ba> <Bw> <aux n_ar> <aux n_rbeats> <aux n_aw> <aux n_wbeats> <aux n_wzero> <D>
//                  A = first AR handshake on any port, B = last R beat on any port,
//                  T = first BVALID cycle of the GO write, L = cycle the core handed
//                  over its last result row (all in tb cycles; -1 = none);
//                  Aa / Ba = first AR handshake / last R beat on aux_axi, Bw = the last
//                  BVALID on aux_axi, then the aux burst / beat counts of the run;
//                  D = the first cycle STATUS.done is set
// A write to CTRL with bit 0 re-arms the per-run event logs; writes to CHUNK / MAXOUT
// update what the slaves check ARLEN and the in-flight count against.
//
// Slave behaviour plusargs: +membase=<hex> +membeats= +argap= +rgap= (percent)
//   +latmin= +latmax= +seed= +slowport= +slowgap= (one port with its own R gap)
//   +errport= +errbeat= (one SLVERR beat) +bfmrand= (randomise the AXI-Lite master)
// Aux slave: +auximage=<file> +auxbase=<hex> +auxbeats= +auxout=<file> (memory dump at the
//   end) +auxargap= +auxrgap= +auxlatmin= +auxlatmax= +awgap= +wgap= +blatmin= +blatmax=
//   +rerrbeat= (aux R beat with SLVERR) +berrburst= (aux write burst with SLVERR)
// Final line:  AXI_TB_DONE ops=<n> slave_errors=<n> cycles=<n>   (AXI_TB_TIMEOUT otherwise)
`timescale 1ns / 1ps

module tb_gemv_axi;
    parameter integer NPORTS  = 3;
    parameter integer GMAX    = 96;
    parameter integer FIFO_LG = 11;
    parameter integer RES_LG  = 14;
    parameter integer MEM_LG  = 18;
    parameter integer AUX_LG  = 15;
    parameter integer NB      = 1;          // max batch of the DUT (register map version 3)
    parameter integer MAXOPS  = 1 << 18;
    localparam integer MP = 4, AW = 49, IDW = 6;

    reg clk = 0, aresetn = 0;
    always #1.5 clk = ~clk;

    reg [63:0] now = 0;
    always @(posedge clk) now <= now + 1;

    // ---- AXI-Lite
    reg  [16:0] s_awaddr = 0, s_araddr = 0;
    reg         s_awvalid = 0, s_wvalid = 0, s_bready = 1, s_arvalid = 0, s_rready = 1;
    reg  [31:0] s_wdata = 0;
    wire        s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
    wire [1:0]  s_bresp, s_rresp;
    wire [31:0] s_rdata;

    // ---- masters
    wire [IDW-1:0] arid [0:MP-1];
    wire [AW-1:0]  araddr [0:MP-1];
    wire [7:0]     arlen [0:MP-1];
    wire [2:0]     arsize [0:MP-1];
    wire [1:0]     arburst [0:MP-1];
    wire           arvalid [0:MP-1], arready [0:MP-1];
    wire [IDW-1:0] rid [0:MP-1];
    wire [127:0]   rdata [0:MP-1];
    wire [1:0]     rresp [0:MP-1];
    wire           rlast [0:MP-1], rvalid [0:MP-1], rready [0:MP-1];
    wire [MP-1:0]  awvalid_m, wvalid_m;
    wire [31:0]    serr [0:MP-1];
    wire [63:0]    first_ar [0:MP-1], last_r [0:MP-1];
    wire           seen_ar [0:MP-1];
    wire [31:0]    n_ar [0:MP-1], n_beats [0:MP-1];

    // ---- aux master
    wire [IDW-1:0] x_awid, x_bid, x_arid, x_rid;
    wire [AW-1:0]  x_awaddr, x_araddr;
    wire [7:0]     x_awlen, x_arlen;
    wire [2:0]     x_awsize, x_arsize, x_awprot, x_arprot;
    wire [1:0]     x_awburst, x_arburst, x_bresp, x_rresp;
    wire [3:0]     x_awcache, x_arcache, x_awqos, x_arqos;
    wire           x_awlock, x_arlock, x_awvalid, x_awready, x_wlast, x_wvalid, x_wready;
    wire           x_bvalid, x_bready, x_arvalid, x_arready, x_rlast, x_rvalid, x_rready;
    wire [127:0]   x_wdata, x_rdata;
    wire [15:0]    x_wstrb;
    wire [31:0]    x_err, x_n_ar, x_n_rbeats, x_n_aw, x_n_wbeats, x_n_wzero;
    wire [63:0]    x_first_ar, x_last_r, x_last_b;
    wire           x_seen_ar, x_seen_b;

    gemv_axi #(.NPORTS(NPORTS), .GMAX(GMAX), .FIFO_LG(FIFO_LG), .RES_LG(RES_LG), .NB(NB)) dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(aresetn),
        .S_AXI_AWADDR(s_awaddr), .S_AXI_AWPROT(3'b0), .S_AXI_AWVALID(s_awvalid),
        .S_AXI_AWREADY(s_awready), .S_AXI_WDATA(s_wdata), .S_AXI_WSTRB(4'hF),
        .S_AXI_WVALID(s_wvalid), .S_AXI_WREADY(s_wready), .S_AXI_BRESP(s_bresp),
        .S_AXI_BVALID(s_bvalid), .S_AXI_BREADY(s_bready), .S_AXI_ARADDR(s_araddr),
        .S_AXI_ARPROT(3'b0), .S_AXI_ARVALID(s_arvalid), .S_AXI_ARREADY(s_arready),
        .S_AXI_RDATA(s_rdata), .S_AXI_RRESP(s_rresp), .S_AXI_RVALID(s_rvalid),
        .S_AXI_RREADY(s_rready),
        .m0_axi_arid(arid[0]),
        .m0_axi_araddr(araddr[0]),
        .m0_axi_arlen(arlen[0]),
        .m0_axi_arsize(arsize[0]),
        .m0_axi_arburst(arburst[0]),
        .m0_axi_arvalid(arvalid[0]),
        .m0_axi_arready(arready[0]),
        .m0_axi_rid(rid[0]),
        .m0_axi_rdata(rdata[0]),
        .m0_axi_rresp(rresp[0]),
        .m0_axi_rlast(rlast[0]),
        .m0_axi_rvalid(rvalid[0]),
        .m0_axi_rready(rready[0]),
        .m0_axi_awready(1'b0),
        .m0_axi_wready(1'b0),
        .m0_axi_bid(6'd0),
        .m0_axi_bresp(2'b00),
        .m0_axi_bvalid(1'b0),
        .m0_axi_awvalid(awvalid_m[0]),
        .m0_axi_wvalid(wvalid_m[0]),
        .m1_axi_arid(arid[1]),
        .m1_axi_araddr(araddr[1]),
        .m1_axi_arlen(arlen[1]),
        .m1_axi_arsize(arsize[1]),
        .m1_axi_arburst(arburst[1]),
        .m1_axi_arvalid(arvalid[1]),
        .m1_axi_arready(arready[1]),
        .m1_axi_rid(rid[1]),
        .m1_axi_rdata(rdata[1]),
        .m1_axi_rresp(rresp[1]),
        .m1_axi_rlast(rlast[1]),
        .m1_axi_rvalid(rvalid[1]),
        .m1_axi_rready(rready[1]),
        .m1_axi_awready(1'b0),
        .m1_axi_wready(1'b0),
        .m1_axi_bid(6'd0),
        .m1_axi_bresp(2'b00),
        .m1_axi_bvalid(1'b0),
        .m1_axi_awvalid(awvalid_m[1]),
        .m1_axi_wvalid(wvalid_m[1]),
        .m2_axi_arid(arid[2]),
        .m2_axi_araddr(araddr[2]),
        .m2_axi_arlen(arlen[2]),
        .m2_axi_arsize(arsize[2]),
        .m2_axi_arburst(arburst[2]),
        .m2_axi_arvalid(arvalid[2]),
        .m2_axi_arready(arready[2]),
        .m2_axi_rid(rid[2]),
        .m2_axi_rdata(rdata[2]),
        .m2_axi_rresp(rresp[2]),
        .m2_axi_rlast(rlast[2]),
        .m2_axi_rvalid(rvalid[2]),
        .m2_axi_rready(rready[2]),
        .m2_axi_awready(1'b0),
        .m2_axi_wready(1'b0),
        .m2_axi_bid(6'd0),
        .m2_axi_bresp(2'b00),
        .m2_axi_bvalid(1'b0),
        .m2_axi_awvalid(awvalid_m[2]),
        .m2_axi_wvalid(wvalid_m[2]),
        .m3_axi_arid(arid[3]),
        .m3_axi_araddr(araddr[3]),
        .m3_axi_arlen(arlen[3]),
        .m3_axi_arsize(arsize[3]),
        .m3_axi_arburst(arburst[3]),
        .m3_axi_arvalid(arvalid[3]),
        .m3_axi_arready(arready[3]),
        .m3_axi_rid(rid[3]),
        .m3_axi_rdata(rdata[3]),
        .m3_axi_rresp(rresp[3]),
        .m3_axi_rlast(rlast[3]),
        .m3_axi_rvalid(rvalid[3]),
        .m3_axi_rready(rready[3]),
        .m3_axi_awready(1'b0),
        .m3_axi_wready(1'b0),
        .m3_axi_bid(6'd0),
        .m3_axi_bresp(2'b00),
        .m3_axi_bvalid(1'b0),
        .m3_axi_awvalid(awvalid_m[3]),
        .m3_axi_wvalid(wvalid_m[3]),
        .aux_axi_awid(x_awid), .aux_axi_awaddr(x_awaddr), .aux_axi_awlen(x_awlen),
        .aux_axi_awsize(x_awsize), .aux_axi_awburst(x_awburst), .aux_axi_awlock(x_awlock),
        .aux_axi_awcache(x_awcache), .aux_axi_awprot(x_awprot), .aux_axi_awqos(x_awqos),
        .aux_axi_awvalid(x_awvalid), .aux_axi_awready(x_awready),
        .aux_axi_wdata(x_wdata), .aux_axi_wstrb(x_wstrb), .aux_axi_wlast(x_wlast),
        .aux_axi_wvalid(x_wvalid), .aux_axi_wready(x_wready),
        .aux_axi_bid(x_bid), .aux_axi_bresp(x_bresp), .aux_axi_bvalid(x_bvalid),
        .aux_axi_bready(x_bready),
        .aux_axi_arid(x_arid), .aux_axi_araddr(x_araddr), .aux_axi_arlen(x_arlen),
        .aux_axi_arsize(x_arsize), .aux_axi_arburst(x_arburst), .aux_axi_arlock(x_arlock),
        .aux_axi_arcache(x_arcache), .aux_axi_arprot(x_arprot), .aux_axi_arqos(x_arqos),
        .aux_axi_arvalid(x_arvalid), .aux_axi_arready(x_arready),
        .aux_axi_rid(x_rid), .aux_axi_rdata(x_rdata), .aux_axi_rresp(x_rresp),
        .aux_axi_rlast(x_rlast), .aux_axi_rvalid(x_rvalid), .aux_axi_rready(x_rready)
    );

    // ---- slave configuration
    reg [63:0] membase;
    integer    membeats, argap, rgap, latmin, latmax, seed, slowport, slowgap, errport, errbeat, bfmrand;
    integer    timeout, polltimeout, pollgap, nops;
    reg [63:0] auxbase;
    integer    auxbeats, auxargap, auxrgap, auxlatmin, auxlatmax, awgap, wgap, blatmin, blatmax;
    integer    rerrbeat, berrburst;
    reg [1023:0] auxoutf;
    reg        have_auxout;
    reg [31:0] exp_len = 0, maxout_check = 0;
    reg        arm = 0;
    integer    unused_err = 0;

    genvar g;
    generate for (g = 0; g < MP; g = g + 1) begin : sl
        if (g < NPORTS) begin : on
            wire [31:0] sd = seed + 7919 * (g + 1);
            axi_rd_mem_slave #(.AW(AW), .IDW(IDW), .MEM_LG(MEM_LG)) u (
                .clk(clk), .rst(~aresetn), .now(now), .arm(arm),
                .mem_base(membase[AW-1:0]), .mem_beats(membeats), .exp_len(exp_len),
                .maxout_check(maxout_check), .ar_gap(argap),
                .r_gap((g == slowport) ? slowgap : rgap), .lat_min(latmin), .lat_max(latmax),
                .seed0(sd), .err_beat((g == errport) ? errbeat : 32'hFFFFFFFF),
                .arid(arid[g]), .araddr(araddr[g]), .arlen(arlen[g]), .arsize(arsize[g]),
                .arburst(arburst[g]), .arvalid(arvalid[g]), .arready(arready[g]),
                .rid(rid[g]), .rdata(rdata[g]), .rresp(rresp[g]), .rlast(rlast[g]),
                .rvalid(rvalid[g]), .rready(rready[g]), .errors(serr[g]),
                .first_ar(first_ar[g]), .last_r(last_r[g]), .seen_ar(seen_ar[g]),
                .n_ar(n_ar[g]), .n_beats(n_beats[g]));
        end else begin : off                      // master not in use: must stay silent
            assign arready[g] = 1'b0; assign rid[g] = 0; assign rdata[g] = 128'd0;
            assign rresp[g] = 2'b00; assign rlast[g] = 1'b0; assign rvalid[g] = 1'b0;
            assign serr[g] = 0; assign first_ar[g] = 0; assign last_r[g] = 0; assign seen_ar[g] = 1'b0;
            assign n_ar[g] = 0; assign n_beats[g] = 0;
            always @(posedge clk) if (aresetn && arvalid[g] !== 1'b0) unused_err = unused_err + 1;
        end
        always @(posedge clk)
            if (aresetn && (awvalid_m[g] !== 1'b0 || wvalid_m[g] !== 1'b0)) unused_err = unused_err + 1;
    end endgenerate

    wire [31:0] xsd = seed + 15485863;
    axi_rw_mem_slave #(.AW(AW), .IDW(IDW), .MEM_LG(AUX_LG)) aux (
        .clk(clk), .rst(~aresetn), .now(now), .arm(arm),
        .mem_base(auxbase[AW-1:0]), .mem_beats(auxbeats),
        .ar_gap(auxargap), .r_gap(auxrgap), .lat_min(auxlatmin), .lat_max(auxlatmax),
        .aw_gap(awgap), .w_gap(wgap), .b_lat_min(blatmin), .b_lat_max(blatmax), .seed0(xsd),
        .rerr_beat(rerrbeat), .berr_burst(berrburst),
        .awid(x_awid), .awaddr(x_awaddr), .awlen(x_awlen), .awsize(x_awsize), .awburst(x_awburst),
        .awvalid(x_awvalid), .awready(x_awready),
        .wdata(x_wdata), .wstrb(x_wstrb), .wlast(x_wlast), .wvalid(x_wvalid), .wready(x_wready),
        .bid(x_bid), .bresp(x_bresp), .bvalid(x_bvalid), .bready(x_bready),
        .arid(x_arid), .araddr(x_araddr), .arlen(x_arlen), .arsize(x_arsize), .arburst(x_arburst),
        .arvalid(x_arvalid), .arready(x_arready),
        .rid(x_rid), .rdata(x_rdata), .rresp(x_rresp), .rlast(x_rlast), .rvalid(x_rvalid),
        .rready(x_rready),
        .errors(x_err), .first_ar(x_first_ar), .last_r(x_last_r), .last_b(x_last_b),
        .seen_ar(x_seen_ar), .seen_b(x_seen_b), .n_ar(x_n_ar), .n_rbeats(x_n_rbeats),
        .n_aw(x_n_aw), .n_wbeats(x_n_wbeats), .n_wzero(x_n_wzero));
    // constant side-band of the aux master
    always @(posedge clk)
        if (aresetn && (x_awlock !== 1'b0 || x_arlock !== 1'b0 || x_awcache !== 4'b0011 || x_arcache !== 4'b0011
                        || x_awprot !== 3'b000 || x_arprot !== 3'b000 || x_awqos !== 4'b0000 || x_arqos !== 4'b0000))
            unused_err = unused_err + 1;

    // ---- event probes
    reg        bv_prev = 0;
    reg [63:0] bv_cyc = 0, l_cyc = 0, t_go = 0;
    reg        l_seen = 0, go_seen = 0;
    reg        d_prev = 0, d_seen = 0;
    reg [63:0] d_cyc = 0;
    always @(posedge clk) begin
        d_prev <= (dut.done === 1'b1);
        if (arm) d_seen <= 1'b0;
        else if (dut.done === 1'b1 && !d_prev) begin d_cyc <= now; d_seen <= 1'b1; end
        bv_prev <= s_bvalid;
        if (s_bvalid && !bv_prev) bv_cyc <= now;
        if (arm) l_seen <= 1'b0;
        else if (dut.u_core.m_valid === 1'b1 && dut.u_core.m_last === 1'b1 && dut.u_core.m_ready === 1'b1) begin
            l_cyc <= now; l_seen <= 1'b1;
        end
    end

    // ---- AXI-Lite BFM
    integer bseed;
    function integer rnd(input integer n); begin rnd = ($random(bseed) & 32'h7fffffff) % n; end endfunction

    task axil_wr(input [16:0] a, input [31:0] d);
        integer da, dw, db;
        begin
            da = bfmrand ? rnd(3) : 0; dw = bfmrand ? rnd(3) : 0; db = bfmrand ? rnd(4) : 0;
            if (db != 0) s_bready <= 1'b0;
            fork
                begin
                    repeat (da) @(posedge clk);
                    s_awaddr <= a; s_awvalid <= 1'b1;
                    @(posedge clk);
                    while (!s_awready) @(posedge clk);
                    s_awvalid <= 1'b0; s_awaddr <= {17{1'bx}};
                end
                begin
                    repeat (dw) @(posedge clk);
                    s_wdata <= d; s_wvalid <= 1'b1;
                    @(posedge clk);
                    while (!s_wready) @(posedge clk);
                    s_wvalid <= 1'b0; s_wdata <= {32{1'bx}};
                end
            join
            repeat (db) @(posedge clk);
            s_bready <= 1'b1;
            @(posedge clk);
            while (!(s_bvalid && s_bready)) @(posedge clk);
        end
    endtask

    task axil_rd(input [16:0] a, output [31:0] d);
        integer dr;
        begin
            dr = bfmrand ? rnd(3) : 0;
            if (dr != 0) s_rready <= 1'b0;
            s_araddr <= a; s_arvalid <= 1'b1;
            @(posedge clk);
            while (!s_arready) @(posedge clk);
            s_arvalid <= 1'b0; s_araddr <= {17{1'bx}};
            repeat (dr) @(posedge clk);
            s_rready <= 1'b1;
            @(posedge clk);
            while (!(s_rvalid && s_rready)) @(posedge clk);
            d = s_rdata;
        end
    endtask

    // ---- watchdog
    always @(posedge clk) if (now > timeout) begin
        $display("AXI_TB_TIMEOUT cyc=%0d", now);
        $finish;
    end

    reg [31:0]   ops [0:4*MAXOPS-1];
    reg [1023:0] scriptf, outf;
    integer      fout, i, k, npoll, tot_err;
    reg [31:0]   op, a0, a1, a2, v;
    reg [63:0]   ev_a, ev_b;
    reg          any_a;
    integer      tot_ar, tot_beats;

    initial begin
        if (!$value$plusargs("script=%s", scriptf)) begin $display("need +script"); $finish; end
        if (!$value$plusargs("out=%s", outf))       begin $display("need +out"); $finish; end
        if (!$value$plusargs("nops=%d", nops))      nops = 0;
        if (!$value$plusargs("membase=%h", membase)) membase = 64'h8_0000_0000;
        if (!$value$plusargs("membeats=%d", membeats)) membeats = 0;
        if (!$value$plusargs("argap=%d", argap))    argap = 0;
        if (!$value$plusargs("rgap=%d", rgap))      rgap = 0;
        if (!$value$plusargs("latmin=%d", latmin))  latmin = 20;
        if (!$value$plusargs("latmax=%d", latmax))  latmax = 20;
        if (!$value$plusargs("seed=%d", seed))      seed = 1;
        if (!$value$plusargs("slowport=%d", slowport)) slowport = -1;
        if (!$value$plusargs("slowgap=%d", slowgap)) slowgap = 0;
        if (!$value$plusargs("errport=%d", errport)) errport = -1;
        if (!$value$plusargs("errbeat=%d", errbeat)) errbeat = 0;
        if (!$value$plusargs("bfmrand=%d", bfmrand)) bfmrand = 0;
        if (!$value$plusargs("timeout=%d", timeout)) timeout = 50000000;
        if (!$value$plusargs("polltimeout=%d", polltimeout)) polltimeout = 100000;
        if (!$value$plusargs("pollgap=%d", pollgap)) pollgap = 8;
        if (!$value$plusargs("auxbase=%h", auxbase)) auxbase = 64'h9_0000_0000;
        if (!$value$plusargs("auxbeats=%d", auxbeats)) auxbeats = 0;
        if (!$value$plusargs("auxargap=%d", auxargap)) auxargap = 0;
        if (!$value$plusargs("auxrgap=%d", auxrgap)) auxrgap = 0;
        if (!$value$plusargs("auxlatmin=%d", auxlatmin)) auxlatmin = 20;
        if (!$value$plusargs("auxlatmax=%d", auxlatmax)) auxlatmax = 20;
        if (!$value$plusargs("awgap=%d", awgap)) awgap = 0;
        if (!$value$plusargs("wgap=%d", wgap)) wgap = 0;
        if (!$value$plusargs("blatmin=%d", blatmin)) blatmin = 2;
        if (!$value$plusargs("blatmax=%d", blatmax)) blatmax = 2;
        if (!$value$plusargs("rerrbeat=%d", rerrbeat)) rerrbeat = -1;
        if (!$value$plusargs("berrburst=%d", berrburst)) berrburst = -1;
        have_auxout = $value$plusargs("auxout=%s", auxoutf);
        if (auxbeats > (1 << AUX_LG)) begin $display("AXI_TB_ERR aux image > memory"); $finish; end
        if (nops > MAXOPS) begin $display("AXI_TB_ERR nops > MAXOPS"); $finish; end
        if (membeats > (1 << MEM_LG)) begin $display("AXI_TB_ERR image > memory"); $finish; end
        bseed = seed + 424242;
        $readmemh(scriptf, ops, 0, 4*nops - 1);
        fout = $fopen(outf, "w");
        repeat (10) @(posedge clk);
        aresetn <= 1'b1;
        repeat (12) @(posedge clk);

        for (i = 0; i < nops; i = i + 1) begin
            op = ops[4*i]; a0 = ops[4*i + 1]; a1 = ops[4*i + 2]; a2 = ops[4*i + 3];
            case (op)
                32'd0: begin
                    if (a0[16:0] == 17'h00020) exp_len <= a1;
                    if (a0[16:0] == 17'h00028) maxout_check <= (a1 == 0) ? 32'd1 : ((a1 > 255) ? 32'd255 : a1);
                    if (a0[16:0] == 17'h00000 && a1[0] && !a1[1]) begin
                        arm <= 1'b1; @(posedge clk); arm <= 1'b0; @(posedge clk);
                    end
                    axil_wr(a0[16:0], a1);
                    @(posedge clk);
                    if (a0[16:0] == 17'h00000 && a1[0] && !a1[1]) t_go = bv_cyc;
                end
                32'd1: begin
                    axil_rd(a0[16:0], v);
                    $fdisplay(fout, "R %05x %08x", a0[16:0], v);
                end
                32'd2: begin
                    npoll = 0; v = ~a2;
                    while (((v & a1) !== a2) && npoll < polltimeout) begin
                        if (npoll != 0) repeat (pollgap) @(posedge clk);
                        axil_rd(a0[16:0], v);
                        npoll = npoll + 1;
                    end
                    $fdisplay(fout, "P %05x %08x %0d", a0[16:0], v, npoll);
                end
                32'd3: repeat (a0) @(posedge clk);
                32'd5: begin axil_wr(a0[16:0], a1); @(posedge clk); end
                32'd4: begin
                    any_a = 0; ev_a = 0; ev_b = 0; tot_ar = 0; tot_beats = 0;
                    for (k = 0; k < NPORTS; k = k + 1) begin
                        if (seen_ar[k]) begin
                            if (!any_a || first_ar[k] < ev_a) ev_a = first_ar[k];
                            if (!any_a || last_r[k] > ev_b) ev_b = last_r[k];
                            any_a = 1;
                        end
                        tot_ar = tot_ar + n_ar[k]; tot_beats = tot_beats + n_beats[k];
                    end
                    $fdisplay(fout, "M %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d", a0,
                              any_a ? $signed(ev_a) : -1, any_a ? $signed(ev_b) : -1, $signed(t_go),
                              l_seen ? $signed(l_cyc) : -1, tot_ar, tot_beats,
                              x_seen_ar ? $signed(x_first_ar) : -1, x_seen_ar ? $signed(x_last_r) : -1,
                              x_seen_b ? $signed(x_last_b) : -1, x_n_ar, x_n_rbeats, x_n_aw, x_n_wbeats, x_n_wzero,
                              d_seen ? $signed(d_cyc) : -1);
                end
                default: begin $display("AXI_TB_ERR bad op %0d at %0d", op, i); $finish; end
            endcase
        end
        repeat (100) @(posedge clk);
        tot_err = unused_err + x_err;
        if (have_auxout && auxbeats > 0) $writememh(auxoutf, aux.mem, 0, auxbeats - 1);
        for (k = 0; k < MP; k = k + 1) tot_err = tot_err + serr[k];
        $display("AXI_TB_DONE ops=%0d slave_errors=%0d cycles=%0d", nops, tot_err, now);
        $fclose(fout);
        $finish;
    end
endmodule
