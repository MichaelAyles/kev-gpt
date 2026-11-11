// axi_rw_mem_slave — memory-backed AXI4 read + write slave model for the aux master of
// gemv_axi (register map version 2): serves the activation records and takes the result
// rows, so the gate can compare what landed in "DDR".
//
// One beat memory of 2^MEM_LG x 128 bit at byte address mem_base, loaded with $readmemh
// from +auximage=<file> (same format as axi_rd_mem_slave: one beat per line, byte 0 = the
// low byte of the word). The testbench dumps it at the end ($writememh of `mem`).
//
// READ: in-order INCR bursts, random latency lat_min..lat_max from AR accept to the first
//   beat, random ARREADY / RVALID gaps (percent), RRESP = SLVERR on global beat rerr_beat.
// WRITE: AW accepted with random gaps (aw_gap %), W beats accepted only for an addressed
//   burst with random gaps (w_gap %; 99 = a slave that stalls ~100 cycles per beat), each
//   byte written under its WSTRB bit; B is returned b_lat_min..b_lat_max cycles after the
//   last W beat (0 = the very next cycle), BRESP = SLVERR on global burst berr_burst.
//
// Protocol checks (each bumps `errors` and prints AUX_ERR):
//   * RREADY or BREADY low in any cycle (the wrapper promises both are 1 for ever)
//   * a read or write burst crossing a 4 KB boundary; size != 16 bytes; type != INCR;
//     an unaligned address; an address outside the memory; AxID != 0
//   * ARVALID / AWVALID / WVALID dropped, or their payload changed, before xREADY
//   * WLAST not exactly on beat AWLEN
//   * more than one read burst, or more than one write burst, outstanding (the wrapper
//     promises one at a time; a second AW before the previous B counts)
//
// Event log, restarted by `arm`: first_ar (cycle of the first AR handshake), last_r (cycle
// of the latest R beat), last_b (cycle of the latest BVALID), n_ar, n_rbeats, n_aw,
// n_wbeats, n_wzero (W beats with WSTRB == 0: an aborted burst being completed).
`timescale 1ns / 1ps

module axi_rw_mem_slave #(
    parameter integer AW     = 49,
    parameter integer IDW    = 6,
    parameter integer MEM_LG = 15,
    parameter integer QD     = 64
) (
    input  wire            clk,
    input  wire            rst,
    input  wire [63:0]     now,
    input  wire            arm,
    input  wire [AW-1:0]   mem_base,
    input  wire [31:0]     mem_beats,
    input  wire [31:0]     ar_gap,
    input  wire [31:0]     r_gap,
    input  wire [31:0]     lat_min,
    input  wire [31:0]     lat_max,
    input  wire [31:0]     aw_gap,
    input  wire [31:0]     w_gap,
    input  wire [31:0]     b_lat_min,
    input  wire [31:0]     b_lat_max,
    input  wire [31:0]     seed0,
    input  wire [31:0]     rerr_beat,           // global R beat index that returns SLVERR (0xFFFFFFFF: none)
    input  wire [31:0]     berr_burst,          // global write burst index that returns SLVERR
    // write address
    input  wire [IDW-1:0]  awid,
    input  wire [AW-1:0]   awaddr,
    input  wire [7:0]      awlen,
    input  wire [2:0]      awsize,
    input  wire [1:0]      awburst,
    input  wire            awvalid,
    output reg             awready,
    // write data
    input  wire [127:0]    wdata,
    input  wire [15:0]     wstrb,
    input  wire            wlast,
    input  wire            wvalid,
    output reg             wready,
    // write response
    output reg  [IDW-1:0]  bid,
    output reg  [1:0]      bresp,
    output reg             bvalid,
    input  wire            bready,
    // read
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
    // status
    output reg  [31:0]     errors,
    output reg  [63:0]     first_ar,
    output reg  [63:0]     last_r,
    output reg  [63:0]     last_b,
    output reg             seen_ar,
    output reg             seen_b,
    output reg  [31:0]     n_ar,
    output reg  [31:0]     n_rbeats,
    output reg  [31:0]     n_aw,
    output reg  [31:0]     n_wbeats,
    output reg  [31:0]     n_wzero
);
    reg [127:0] mem [0:(1<<MEM_LG)-1];
    reg [1023:0] imgfile;
    initial if ($value$plusargs("auximage=%s", imgfile)) $readmemh(imgfile, mem);

    // ---- read queue
    reg [AW-1:0]  rq_addr [0:QD-1];
    reg [8:0]     rq_len  [0:QD-1];
    reg [63:0]    rq_t    [0:QD-1];
    integer rqh, rqt, seed, lat, total_rbeats;
    reg [8:0]     rbeat;
    reg           ar_pend;
    reg [AW-1:0]  ar_pa;
    reg [7:0]     ar_pl;
    // ---- write queues
    reg [AW-1:0]  wq_addr [0:QD-1];
    reg [8:0]     wq_len  [0:QD-1];
    reg [63:0]    bq_t    [0:QD-1];
    reg [1:0]     bq_resp [0:QD-1];
    integer wqh, wqt, bqh, bqt, total_wbursts, wout, i;
    reg [8:0]     wbeat;
    reg           aw_pend, w_pend;
    reg [AW-1:0]  aw_pa;
    reg [7:0]     aw_pl;
    reg [127:0]   w_pd, wword;
    reg [15:0]    w_ps;
    reg           w_plast;
    reg [AW-1:0]  cur, off;
    reg           seeded;

    function integer rpct; input dummy; begin rpct = ($random(seed) & 32'h7fffffff) % 100; end endfunction
    function integer rrange; input integer lo; input integer hi;
        begin rrange = lo + ((hi > lo) ? (($random(seed) & 32'h7fffffff) % (hi - lo + 1)) : 0); end
    endfunction

    task err(input [255:0] what);
        begin
            errors = errors + 1;
            $display("AUX_ERR %0s cyc=%0d", what, now);
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            rqh = 0; rqt = 0; rbeat = 0; total_rbeats = 0; ar_pend = 0; seeded = 0;
            wqh = 0; wqt = 0; bqh = 0; bqt = 0; total_wbursts = 0; wbeat = 0; aw_pend = 0; w_pend = 0; wout = 0;
            arready <= 1'b0; rvalid <= 1'b0; rlast <= 1'b0; rresp <= 2'b00; rid <= 0; rdata <= 128'd0;
            awready <= 1'b0; wready <= 1'b0; bvalid <= 1'b0; bresp <= 2'b00; bid <= 0;
            errors = 0; first_ar <= 0; last_r <= 0; last_b <= 0; seen_ar <= 1'b0; seen_b <= 1'b0;
            n_ar <= 0; n_rbeats <= 0; n_aw <= 0; n_wbeats <= 0; n_wzero <= 0;
        end else begin
            if (!seeded) begin seed = seed0; seeded = 1; end
            if (arm) begin
                seen_ar <= 1'b0; seen_b <= 1'b0; first_ar <= 0; last_r <= 0; last_b <= 0;
                n_ar <= 0; n_rbeats <= 0; n_aw <= 0; n_wbeats <= 0; n_wzero <= 0;
            end
            if (rready !== 1'b1) err("RREADY low");
            if (bready !== 1'b1) err("BREADY low");

            // ================================================= AR
            if (ar_pend && !(arvalid === 1'b1 && araddr === ar_pa && arlen === ar_pl))
                err("ARVALID/ARADDR changed before ARREADY");
            ar_pend = (arvalid === 1'b1) && !arready;
            ar_pa = araddr; ar_pl = arlen;
            if (arvalid === 1'b1 && arready) begin
                if (arsize !== 3'b100 || arburst !== 2'b01) err("AR bad size/burst");
                if (({1'b0, araddr[11:0]} + ((arlen + 1) * 16)) > 4096) err("AR 4KB cross");
                if (arid !== {IDW{1'b0}}) err("ARID != 0");
                if (araddr[3:0] != 0) err("AR unaligned address");
                off = araddr - mem_base;
                if (araddr < mem_base || ((off >> 4) + arlen + 1) > mem_beats) err("AR outside memory");
                if (rqt != rqh) err("second read burst outstanding");
                lat = rrange(lat_min, lat_max);
                rq_addr[rqt % QD] = araddr; rq_len[rqt % QD] = arlen + 1; rq_t[rqt % QD] = now + lat;
                rqt = rqt + 1;
                if (!seen_ar && !arm) begin seen_ar <= 1'b1; first_ar <= now; end
                n_ar <= n_ar + 1;
            end
            arready <= ((rqt - rqh) < QD - 2) && (ar_gap == 0 || rpct(0) >= ar_gap);

            // ================================================= R (RREADY is 1: every beat transfers)
            if (rvalid) begin
                last_r <= now; n_rbeats <= n_rbeats + 1; total_rbeats = total_rbeats + 1;
                if (rlast) begin rqh = rqh + 1; rbeat = 0; end
                else rbeat = rbeat + 1;
            end
            if (rqh != rqt && rq_t[rqh % QD] <= now && (r_gap == 0 || rpct(0) >= r_gap)) begin
                cur = rq_addr[rqh % QD] + rbeat * 16;
                off = cur - mem_base;
                rvalid <= 1'b1;
                rid    <= {IDW{1'b0}};
                rdata  <= (cur >= mem_base && (off >> 4) < mem_beats) ? mem[off >> 4] : {128{1'bx}};
                rresp  <= (total_rbeats == rerr_beat) ? 2'b10 : 2'b00;
                rlast  <= (rbeat + 1 == rq_len[rqh % QD]);
            end else begin
                rvalid <= 1'b0; rlast <= 1'b0; rdata <= {128{1'bx}};
            end

            // ================================================= AW
            if (aw_pend && !(awvalid === 1'b1 && awaddr === aw_pa && awlen === aw_pl))
                err("AWVALID/AWADDR changed before AWREADY");
            aw_pend = (awvalid === 1'b1) && !awready;
            aw_pa = awaddr; aw_pl = awlen;
            if (awvalid === 1'b1 && awready) begin
                if (awsize !== 3'b100 || awburst !== 2'b01) err("AW bad size/burst");
                if (({1'b0, awaddr[11:0]} + ((awlen + 1) * 16)) > 4096) err("AW 4KB cross");
                if (awid !== {IDW{1'b0}}) err("AWID != 0");
                if (awaddr[3:0] != 0) err("AW unaligned address");
                off = awaddr - mem_base;
                if (awaddr < mem_base || ((off >> 4) + awlen + 1) > mem_beats) err("AW outside memory");
                if (wout != 0) err("second write burst outstanding");
                wq_addr[wqt % QD] = awaddr; wq_len[wqt % QD] = awlen + 1;
                wqt = wqt + 1; wout = wout + 1;
                n_aw <= n_aw + 1;
            end
            awready <= ((wqt - wqh) < QD - 2) && (aw_gap == 0 || rpct(0) >= aw_gap);

            // ================================================= W
            if (w_pend && !(wvalid === 1'b1 && wdata === w_pd && wstrb === w_ps && wlast === w_plast))
                err("WVALID/WDATA changed before WREADY");
            w_pend = (wvalid === 1'b1) && !wready;
            w_pd = wdata; w_ps = wstrb; w_plast = wlast;
            if (wvalid === 1'b1 && wready) begin
                if (wqh == wqt) err("W beat without an address");
                else begin
                    cur = wq_addr[wqh % QD] + wbeat * 16;
                    off = cur - mem_base;
                    if (^wstrb === 1'bx) err("WSTRB is X");
                    if (cur >= mem_base && (off >> 4) < mem_beats) begin
                        wword = mem[off >> 4];
                        for (i = 0; i < 16; i = i + 1)
                            if (wstrb[i] === 1'b1) wword[8*i +: 8] = wdata[8*i +: 8];
                        mem[off >> 4] = wword;
                    end
                    if (wlast !== ((wbeat + 1) == wq_len[wqh % QD])) err("WLAST position");
                    n_wbeats <= n_wbeats + 1;
                    if (wstrb === 16'h0000) n_wzero <= n_wzero + 1;
                    if ((wbeat + 1) == wq_len[wqh % QD]) begin
                        bq_t[bqt % QD] = now + rrange(b_lat_min, b_lat_max);
                        bq_resp[bqt % QD] = (total_wbursts == berr_burst) ? 2'b10 : 2'b00;
                        bqt = bqt + 1; total_wbursts = total_wbursts + 1;
                        wqh = wqh + 1; wbeat = 0;
                    end else wbeat = wbeat + 1;
                end
            end
            wready <= (wqh != wqt) && (w_gap == 0 || rpct(0) >= w_gap);

            // ================================================= B (BREADY is 1)
            if (bvalid) begin
                last_b <= now; seen_b <= 1'b1; bqh = bqh + 1; wout = wout - 1;
            end
            if (bqh != bqt && bq_t[bqh % QD] <= now) begin
                bvalid <= 1'b1; bid <= {IDW{1'b0}}; bresp <= bq_resp[bqh % QD];
            end else begin
                bvalid <= 1'b0; bresp <= 2'b00;
            end
        end
    end
endmodule
