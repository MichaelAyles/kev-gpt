// -----------------------------------------------------------------------------
// gemv_axi — the R2a board wrapper around gemv_stream (doc 10, rung R2a):
//
//   AXI4-Lite slave (register map: fabric/big/gemv_axi_regs.py, the single source
//   of truth) + NPORTS AXI4 128-bit read masters (axi_rd_stream: credit-based AR
//   issue + block-RAM FIFO, RREADY = 1 for ever) each feeding one gemv_stream
//   port + a result capture RAM + counters. One clock.
//
// Port names follow rtl/bwprobe_axi.v so a Vivado block-design module reference
// infers S_AXI and m0_axi..m3_axi and the masters connect straight to the PS
// S_AXI_HPx_FPD ports. This is plain Verilog-2001, so the port list always has
// FOUR master groups; masters >= NPORTS are tied off (ARVALID = 0) and can be
// left unconnected. NPORTS = 1..4.
//
// One run (what CTRL.GO does), in cycles after the GO write is accepted:
//   +0      snapshot ROWS/GROUPS/PAD/BURSTS/CHUNK/STEP/MAXOUT/BASE+PORT_OFF[k],
//           load the RES_START skip counter, clear the counters
//   +1..+3  flush: FIFOs, skids and the core's control state are reset (the
//           activation RAM is not), so nothing left over from an earlier run or
//           an aborted one can leak into this one
//   +9      start the core and the masters
//   ...     DONE = the core is idle (last row out, pad consumed) AND every
//           master has received its last beat; STATUS.done rises 5 cycles later
//
// Definitions the gate checks exactly (cycle = one rising clock edge):
//   CYC        = L - T: T = the first cycle S_AXI_BVALID is high for the accepted
//                GO write, L = the cycle the core hands over the last result row.
//   CYC_STREAM = B - A: A = the first AR handshake of the run on any port,
//                B = the last R beat of the run on any port.
//   TUPLES     = DATA tuples popped by the engine (rows * tuples-per-row; the pad
//                tuples are consumed but not counted). P_BEATS counts every beat.
//   STALL_IN   = cycles the engine was in the run (data or pad phase) and did not
//                pop because some port had no beat; P_EMPTY[k] = those cycles in
//                which port k was one of the empty ones.
//   STALL_OUT  = cycles a data tuple was ready but no output credit was free.
//
// Details the register map leaves open:
//   * every RW register stores and reads back all 32 bits as written; only the
//     documented low bits are used (GROUPS[7:0], CHUNK[8:0], BASE_HI[16:0],
//     ACT_GROUP[7:0], ACT_EXP[EXW-1:0]; MAXOUT is clamped to 1..255).
//   * the run configuration is snapshotted at GO: later writes cannot disturb a
//     run. ACT_COMMIT is ignored while a run is active or ACT_GROUP >= GMAX.
//   * SOFT_RST is a pulse on the write of bit 1 (the "then 0" write is a no-op).
//     It aborts a run, clears STATUS and the counters, keeps the RW registers and
//     the loaded activation. AXI bursts in flight are drained and discarded;
//     STATUS.busy stays 1 until that is over. A GO written during the drain is
//     held and starts the run when the drain ends (it is not lost).
//   * STATUS.axi_err / fifo_overflow are sticky until SOFT_RST or the hard reset
//     (GO does not clear them; ERRCNT is cleared by GO). STATUS.done is cleared
//     by GO.
//   * WSTRB is ignored (whole 32-bit writes). Reads take 4 cycles (the result
//     window is a registered block RAM). Unmapped addresses read 0.
//   * result window: RES_BASE + 4*i for i < 2^RES_LG (PARAMS[23:16]); rows the
//     run did not produce keep whatever an earlier run left there.
//
// VERSION 2 (MODE register; MODE = 0 is version 1 cycle for cycle, except that an
// AXI-Lite read takes 5 cycles instead of 4):
//   * aux_axi: one more AXI4 master (rtl/axi_aux_dma.v, read AND write, always
//     present) for the small host data. MODE.ACT_DMA: after the flush, GROUPS
//     256-byte activation records are fetched from ACT_ADDR into the core's
//     activation RAM (instead of the ACT_* register load), then the run starts;
//     a later run with ACT_DMA = 0 re-uses that activation. MODE.RES_DMA: every
//     result row is also written to DDR at RES_ADDR (4 fp32 per beat, last beat
//     zero padded); the result stream is back-pressured only if the write slave
//     stalls for a long time, and DONE waits for the last write response.
//     MODE.RES_NOCAP: the AXI-Lite window is not written.
//   * CYC_ACT = the number of cycles by which the activation fetch delays the start
//     of the run (engine AND weight streams; it is inside CYC, so CYC - CYC_ACT is
//     what CYC would have been with ACT_DMA = 0 and the same weight-bus behaviour).
//     = Ba - T - 4 for a GO written while idle, Ba = the cycle of the last R beat of
//     the fetch on aux_axi, T as in CYC. 0 without ACT_DMA.
//   * CYC_RES = Bw - L: L as in CYC, Bw = the cycle of the run's last write
//     response (BVALID) on aux_axi. 0 without RES_DMA.
//   * ARGMAX_IDX / ARGMAX_VAL (gemv_argmax below): always on; final before DONE.
//   * STATUS.aux_err: sticky until SOFT_RST / hard reset, like axi_err.
//   * the result window RAM is 2^(RES_LG-10) separate 1024-word RAMs, each with
//     its own output register, and a registered bank mux (the v1 build's only
//     failing paths were the cascade through one inferred 16384-word RAM).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

module gemv_axi #(
    parameter integer NPORTS  = 3,               // read masters in use: m0 .. m{NPORTS-1} (1..4)
    parameter integer GMAX    = 96,              // max 128-groups per row
    parameter integer EXW     = 9,               // activation exponent width (signed)
    parameter integer FIFO_LG = 11,              // log2 beats of each port FIFO (>= 8)
    parameter integer RES_LG  = 14,              // log2 rows of the result window (<= 14)
    parameter integer C_S_AXI_ADDR_WIDTH = 17,   // must be 17 (gemv_axi_regs.ADDR_BITS)
    parameter integer AW      = 49,
    parameter integer IDW     = 6
) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:m0_axi:m1_axi:m2_axi:m3_axi:aux_axi, ASSOCIATED_RESET S_AXI_ARESETN" *)
    input  wire                          S_AXI_ACLK,
    input  wire                          S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  wire [2:0]                    S_AXI_AWPROT,
    input  wire                          S_AXI_AWVALID,
    output reg                           S_AXI_AWREADY,
    input  wire [31:0]                   S_AXI_WDATA,
    input  wire [3:0]                    S_AXI_WSTRB,
    input  wire                          S_AXI_WVALID,
    output reg                           S_AXI_WREADY,
    output wire [1:0]                    S_AXI_BRESP,
    output reg                           S_AXI_BVALID,
    input  wire                          S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  wire [2:0]                    S_AXI_ARPROT,
    input  wire                          S_AXI_ARVALID,
    output reg                           S_AXI_ARREADY,
    output reg  [31:0]                   S_AXI_RDATA,
    output wire [1:0]                    S_AXI_RRESP,
    output reg                           S_AXI_RVALID,
    input  wire                          S_AXI_RREADY,
    output wire [IDW-1:0] m0_axi_awid,
    output wire [AW-1:0] m0_axi_awaddr,
    output wire [7:0] m0_axi_awlen,
    output wire [2:0] m0_axi_awsize,
    output wire [1:0] m0_axi_awburst,
    output wire m0_axi_awlock,
    output wire [3:0] m0_axi_awcache,
    output wire [2:0] m0_axi_awprot,
    output wire [3:0] m0_axi_awqos,
    output wire m0_axi_awvalid,
    input  wire m0_axi_awready,
    output wire [127:0] m0_axi_wdata,
    output wire [15:0] m0_axi_wstrb,
    output wire m0_axi_wlast,
    output wire m0_axi_wvalid,
    input  wire m0_axi_wready,
    input  wire [IDW-1:0] m0_axi_bid,
    input  wire [1:0] m0_axi_bresp,
    input  wire m0_axi_bvalid,
    output wire m0_axi_bready,
    output wire [IDW-1:0] m0_axi_arid,
    output wire [AW-1:0] m0_axi_araddr,
    output wire [7:0] m0_axi_arlen,
    output wire [2:0] m0_axi_arsize,
    output wire [1:0] m0_axi_arburst,
    output wire m0_axi_arlock,
    output wire [3:0] m0_axi_arcache,
    output wire [2:0] m0_axi_arprot,
    output wire [3:0] m0_axi_arqos,
    output wire m0_axi_arvalid,
    input  wire m0_axi_arready,
    input  wire [IDW-1:0] m0_axi_rid,
    input  wire [127:0] m0_axi_rdata,
    input  wire [1:0] m0_axi_rresp,
    input  wire m0_axi_rlast,
    input  wire m0_axi_rvalid,
    output wire m0_axi_rready,
    output wire [IDW-1:0] m1_axi_awid,
    output wire [AW-1:0] m1_axi_awaddr,
    output wire [7:0] m1_axi_awlen,
    output wire [2:0] m1_axi_awsize,
    output wire [1:0] m1_axi_awburst,
    output wire m1_axi_awlock,
    output wire [3:0] m1_axi_awcache,
    output wire [2:0] m1_axi_awprot,
    output wire [3:0] m1_axi_awqos,
    output wire m1_axi_awvalid,
    input  wire m1_axi_awready,
    output wire [127:0] m1_axi_wdata,
    output wire [15:0] m1_axi_wstrb,
    output wire m1_axi_wlast,
    output wire m1_axi_wvalid,
    input  wire m1_axi_wready,
    input  wire [IDW-1:0] m1_axi_bid,
    input  wire [1:0] m1_axi_bresp,
    input  wire m1_axi_bvalid,
    output wire m1_axi_bready,
    output wire [IDW-1:0] m1_axi_arid,
    output wire [AW-1:0] m1_axi_araddr,
    output wire [7:0] m1_axi_arlen,
    output wire [2:0] m1_axi_arsize,
    output wire [1:0] m1_axi_arburst,
    output wire m1_axi_arlock,
    output wire [3:0] m1_axi_arcache,
    output wire [2:0] m1_axi_arprot,
    output wire [3:0] m1_axi_arqos,
    output wire m1_axi_arvalid,
    input  wire m1_axi_arready,
    input  wire [IDW-1:0] m1_axi_rid,
    input  wire [127:0] m1_axi_rdata,
    input  wire [1:0] m1_axi_rresp,
    input  wire m1_axi_rlast,
    input  wire m1_axi_rvalid,
    output wire m1_axi_rready,
    output wire [IDW-1:0] m2_axi_awid,
    output wire [AW-1:0] m2_axi_awaddr,
    output wire [7:0] m2_axi_awlen,
    output wire [2:0] m2_axi_awsize,
    output wire [1:0] m2_axi_awburst,
    output wire m2_axi_awlock,
    output wire [3:0] m2_axi_awcache,
    output wire [2:0] m2_axi_awprot,
    output wire [3:0] m2_axi_awqos,
    output wire m2_axi_awvalid,
    input  wire m2_axi_awready,
    output wire [127:0] m2_axi_wdata,
    output wire [15:0] m2_axi_wstrb,
    output wire m2_axi_wlast,
    output wire m2_axi_wvalid,
    input  wire m2_axi_wready,
    input  wire [IDW-1:0] m2_axi_bid,
    input  wire [1:0] m2_axi_bresp,
    input  wire m2_axi_bvalid,
    output wire m2_axi_bready,
    output wire [IDW-1:0] m2_axi_arid,
    output wire [AW-1:0] m2_axi_araddr,
    output wire [7:0] m2_axi_arlen,
    output wire [2:0] m2_axi_arsize,
    output wire [1:0] m2_axi_arburst,
    output wire m2_axi_arlock,
    output wire [3:0] m2_axi_arcache,
    output wire [2:0] m2_axi_arprot,
    output wire [3:0] m2_axi_arqos,
    output wire m2_axi_arvalid,
    input  wire m2_axi_arready,
    input  wire [IDW-1:0] m2_axi_rid,
    input  wire [127:0] m2_axi_rdata,
    input  wire [1:0] m2_axi_rresp,
    input  wire m2_axi_rlast,
    input  wire m2_axi_rvalid,
    output wire m2_axi_rready,
    output wire [IDW-1:0] m3_axi_awid,
    output wire [AW-1:0] m3_axi_awaddr,
    output wire [7:0] m3_axi_awlen,
    output wire [2:0] m3_axi_awsize,
    output wire [1:0] m3_axi_awburst,
    output wire m3_axi_awlock,
    output wire [3:0] m3_axi_awcache,
    output wire [2:0] m3_axi_awprot,
    output wire [3:0] m3_axi_awqos,
    output wire m3_axi_awvalid,
    input  wire m3_axi_awready,
    output wire [127:0] m3_axi_wdata,
    output wire [15:0] m3_axi_wstrb,
    output wire m3_axi_wlast,
    output wire m3_axi_wvalid,
    input  wire m3_axi_wready,
    input  wire [IDW-1:0] m3_axi_bid,
    input  wire [1:0] m3_axi_bresp,
    input  wire m3_axi_bvalid,
    output wire m3_axi_bready,
    output wire [IDW-1:0] m3_axi_arid,
    output wire [AW-1:0] m3_axi_araddr,
    output wire [7:0] m3_axi_arlen,
    output wire [2:0] m3_axi_arsize,
    output wire [1:0] m3_axi_arburst,
    output wire m3_axi_arlock,
    output wire [3:0] m3_axi_arcache,
    output wire [2:0] m3_axi_arprot,
    output wire [3:0] m3_axi_arqos,
    output wire m3_axi_arvalid,
    input  wire m3_axi_arready,
    input  wire [IDW-1:0] m3_axi_rid,
    input  wire [127:0] m3_axi_rdata,
    input  wire [1:0] m3_axi_rresp,
    input  wire m3_axi_rlast,
    input  wire m3_axi_rvalid,
    output wire m3_axi_rready,
    // aux_axi: activation fetch (AR/R) + result write (AW/W/B), register map version 2
    output wire [IDW-1:0] aux_axi_awid,
    output wire [AW-1:0] aux_axi_awaddr,
    output wire [7:0] aux_axi_awlen,
    output wire [2:0] aux_axi_awsize,
    output wire [1:0] aux_axi_awburst,
    output wire aux_axi_awlock,
    output wire [3:0] aux_axi_awcache,
    output wire [2:0] aux_axi_awprot,
    output wire [3:0] aux_axi_awqos,
    output wire aux_axi_awvalid,
    input  wire aux_axi_awready,
    output wire [127:0] aux_axi_wdata,
    output wire [15:0] aux_axi_wstrb,
    output wire aux_axi_wlast,
    output wire aux_axi_wvalid,
    input  wire aux_axi_wready,
    input  wire [IDW-1:0] aux_axi_bid,
    input  wire [1:0] aux_axi_bresp,
    input  wire aux_axi_bvalid,
    output wire aux_axi_bready,
    output wire [IDW-1:0] aux_axi_arid,
    output wire [AW-1:0] aux_axi_araddr,
    output wire [7:0] aux_axi_arlen,
    output wire [2:0] aux_axi_arsize,
    output wire [1:0] aux_axi_arburst,
    output wire aux_axi_arlock,
    output wire [3:0] aux_axi_arcache,
    output wire [2:0] aux_axi_arprot,
    output wire [3:0] aux_axi_arqos,
    output wire aux_axi_arvalid,
    input  wire aux_axi_arready,
    input  wire [IDW-1:0] aux_axi_rid,
    input  wire [127:0] aux_axi_rdata,
    input  wire [1:0] aux_axi_rresp,
    input  wire aux_axi_rlast,
    input  wire aux_axi_rvalid,
    output wire aux_axi_rready
);
    localparam integer MP        = 4;                    // master port groups in the port list
    localparam integer RES_WORDS = 1 << RES_LG;
    localparam [31:0]  IDCODE_V  = 32'h47454D56;         // "GEMV"
    localparam [7:0]   VERSION_V = 8'd2;

    wire clk = S_AXI_ACLK;
    (* max_fanout = 200 *) reg rst_h;                    // hard reset, registered
    always @(posedge clk) rst_h <= ~S_AXI_ARESETN;

    // ------------------------------------------------------------ master port glue
    wire [MP*IDW-1:0] v_arid;
    wire [MP*AW-1:0] v_araddr;
    wire [MP*8-1:0] v_arlen;
    wire [MP*3-1:0] v_arsize;
    wire [MP*2-1:0] v_arburst;
    wire [MP*1-1:0] v_arlock;
    wire [MP*4-1:0] v_arcache;
    wire [MP*3-1:0] v_arprot;
    wire [MP*4-1:0] v_arqos;
    wire [MP*1-1:0] v_arvalid;
    wire [MP*1-1:0] v_rready;
    assign m0_axi_arid = v_arid[0*IDW +: IDW]; assign m0_axi_araddr = v_araddr[0*AW +: AW]; assign m0_axi_arlen = v_arlen[0*8 +: 8]; assign m0_axi_arsize = v_arsize[0*3 +: 3]; assign m0_axi_arburst = v_arburst[0*2 +: 2]; assign m0_axi_arlock = v_arlock[0*1 +: 1]; assign m0_axi_arcache = v_arcache[0*4 +: 4]; assign m0_axi_arprot = v_arprot[0*3 +: 3]; assign m0_axi_arqos = v_arqos[0*4 +: 4]; assign m0_axi_arvalid = v_arvalid[0*1 +: 1]; assign m0_axi_rready = v_rready[0*1 +: 1];
    assign m0_axi_awid = {IDW{1'b0}}; assign m0_axi_awaddr = {AW{1'b0}}; assign m0_axi_awlen = 8'd0;
    assign m0_axi_awsize = 3'b100; assign m0_axi_awburst = 2'b01; assign m0_axi_awlock = 1'b0;
    assign m0_axi_awcache = 4'b0011; assign m0_axi_awprot = 3'b000; assign m0_axi_awqos = 4'b0000;
    assign m0_axi_awvalid = 1'b0; assign m0_axi_wdata = 128'd0; assign m0_axi_wstrb = 16'd0;
    assign m0_axi_wlast = 1'b0; assign m0_axi_wvalid = 1'b0; assign m0_axi_bready = 1'b1;
    assign m1_axi_arid = v_arid[1*IDW +: IDW]; assign m1_axi_araddr = v_araddr[1*AW +: AW]; assign m1_axi_arlen = v_arlen[1*8 +: 8]; assign m1_axi_arsize = v_arsize[1*3 +: 3]; assign m1_axi_arburst = v_arburst[1*2 +: 2]; assign m1_axi_arlock = v_arlock[1*1 +: 1]; assign m1_axi_arcache = v_arcache[1*4 +: 4]; assign m1_axi_arprot = v_arprot[1*3 +: 3]; assign m1_axi_arqos = v_arqos[1*4 +: 4]; assign m1_axi_arvalid = v_arvalid[1*1 +: 1]; assign m1_axi_rready = v_rready[1*1 +: 1];
    assign m1_axi_awid = {IDW{1'b0}}; assign m1_axi_awaddr = {AW{1'b0}}; assign m1_axi_awlen = 8'd0;
    assign m1_axi_awsize = 3'b100; assign m1_axi_awburst = 2'b01; assign m1_axi_awlock = 1'b0;
    assign m1_axi_awcache = 4'b0011; assign m1_axi_awprot = 3'b000; assign m1_axi_awqos = 4'b0000;
    assign m1_axi_awvalid = 1'b0; assign m1_axi_wdata = 128'd0; assign m1_axi_wstrb = 16'd0;
    assign m1_axi_wlast = 1'b0; assign m1_axi_wvalid = 1'b0; assign m1_axi_bready = 1'b1;
    assign m2_axi_arid = v_arid[2*IDW +: IDW]; assign m2_axi_araddr = v_araddr[2*AW +: AW]; assign m2_axi_arlen = v_arlen[2*8 +: 8]; assign m2_axi_arsize = v_arsize[2*3 +: 3]; assign m2_axi_arburst = v_arburst[2*2 +: 2]; assign m2_axi_arlock = v_arlock[2*1 +: 1]; assign m2_axi_arcache = v_arcache[2*4 +: 4]; assign m2_axi_arprot = v_arprot[2*3 +: 3]; assign m2_axi_arqos = v_arqos[2*4 +: 4]; assign m2_axi_arvalid = v_arvalid[2*1 +: 1]; assign m2_axi_rready = v_rready[2*1 +: 1];
    assign m2_axi_awid = {IDW{1'b0}}; assign m2_axi_awaddr = {AW{1'b0}}; assign m2_axi_awlen = 8'd0;
    assign m2_axi_awsize = 3'b100; assign m2_axi_awburst = 2'b01; assign m2_axi_awlock = 1'b0;
    assign m2_axi_awcache = 4'b0011; assign m2_axi_awprot = 3'b000; assign m2_axi_awqos = 4'b0000;
    assign m2_axi_awvalid = 1'b0; assign m2_axi_wdata = 128'd0; assign m2_axi_wstrb = 16'd0;
    assign m2_axi_wlast = 1'b0; assign m2_axi_wvalid = 1'b0; assign m2_axi_bready = 1'b1;
    assign m3_axi_arid = v_arid[3*IDW +: IDW]; assign m3_axi_araddr = v_araddr[3*AW +: AW]; assign m3_axi_arlen = v_arlen[3*8 +: 8]; assign m3_axi_arsize = v_arsize[3*3 +: 3]; assign m3_axi_arburst = v_arburst[3*2 +: 2]; assign m3_axi_arlock = v_arlock[3*1 +: 1]; assign m3_axi_arcache = v_arcache[3*4 +: 4]; assign m3_axi_arprot = v_arprot[3*3 +: 3]; assign m3_axi_arqos = v_arqos[3*4 +: 4]; assign m3_axi_arvalid = v_arvalid[3*1 +: 1]; assign m3_axi_rready = v_rready[3*1 +: 1];
    assign m3_axi_awid = {IDW{1'b0}}; assign m3_axi_awaddr = {AW{1'b0}}; assign m3_axi_awlen = 8'd0;
    assign m3_axi_awsize = 3'b100; assign m3_axi_awburst = 2'b01; assign m3_axi_awlock = 1'b0;
    assign m3_axi_awcache = 4'b0011; assign m3_axi_awprot = 3'b000; assign m3_axi_awqos = 4'b0000;
    assign m3_axi_awvalid = 1'b0; assign m3_axi_wdata = 128'd0; assign m3_axi_wstrb = 16'd0;
    assign m3_axi_wlast = 1'b0; assign m3_axi_wvalid = 1'b0; assign m3_axi_bready = 1'b1;
    wire [MP-1:0]     v_arready = {m3_axi_arready, m2_axi_arready, m1_axi_arready, m0_axi_arready};
    wire [MP*IDW-1:0] v_rid     = {m3_axi_rid, m2_axi_rid, m1_axi_rid, m0_axi_rid};
    wire [MP*128-1:0] v_rdata   = {m3_axi_rdata, m2_axi_rdata, m1_axi_rdata, m0_axi_rdata};
    wire [MP*2-1:0]   v_rresp   = {m3_axi_rresp, m2_axi_rresp, m1_axi_rresp, m0_axi_rresp};
    wire [MP-1:0]     v_rlast   = {m3_axi_rlast, m2_axi_rlast, m1_axi_rlast, m0_axi_rlast};
    wire [MP-1:0]     v_rvalid  = {m3_axi_rvalid, m2_axi_rvalid, m1_axi_rvalid, m0_axi_rvalid};
    wire unused_wr = m0_axi_awready ^ m0_axi_wready ^ (^m0_axi_bid) ^ (^m0_axi_bresp) ^ m0_axi_bvalid ^ m1_axi_awready ^ m1_axi_wready ^ (^m1_axi_bid) ^ (^m1_axi_bresp) ^ m1_axi_bvalid ^ m2_axi_awready ^ m2_axi_wready ^ (^m2_axi_bid) ^ (^m2_axi_bresp) ^ m2_axi_bvalid ^ m3_axi_awready ^ m3_axi_wready ^ (^m3_axi_bid) ^ (^m3_axi_bresp) ^ m3_axi_bvalid;

    // ------------------------------------------------------------ AXI-Lite write channel
    // AW and W are taken independently (either order); the write happens (wr_stb) in
    // the first cycle BVALID is high.
    reg        aw_have, w_have, wr_stb;
    reg [14:0] wr_idx;                                   // address [16:2]
    reg [31:0] wr_data;
    assign S_AXI_BRESP = 2'b00;
    always @(posedge clk) begin
        if (rst_h) begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0; S_AXI_BVALID <= 1'b0;
            aw_have <= 1'b0; w_have <= 1'b0; wr_stb <= 1'b0;
        end else begin
            S_AXI_AWREADY <= 1'b0; S_AXI_WREADY <= 1'b0; wr_stb <= 1'b0;
            if (S_AXI_AWVALID && !aw_have && !S_AXI_AWREADY) begin
                S_AXI_AWREADY <= 1'b1; aw_have <= 1'b1; wr_idx <= S_AXI_AWADDR[16:2];
            end
            if (S_AXI_WVALID && !w_have && !S_AXI_WREADY) begin
                S_AXI_WREADY <= 1'b1; w_have <= 1'b1; wr_data <= S_AXI_WDATA;
            end
            if (aw_have && w_have && !S_AXI_BVALID) begin
                wr_stb <= 1'b1; aw_have <= 1'b0; w_have <= 1'b0; S_AXI_BVALID <= 1'b1;
            end else if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------ RW registers
    reg [31:0]          r_rows, r_groups, r_pad, r_bursts, r_chunk, r_step, r_maxout, r_res_start;
    reg [31:0]          r_base_lo, r_base_hi, r_act_group, r_act_exp;
    reg [31:0]          r_mode, r_act_addr_lo, r_act_addr_hi, r_res_addr_lo, r_res_addr_hi;
    reg [NPORTS*32-1:0] r_port_off;
    reg [1279:0]        r_act_mant;

    wire       wr_lo   = wr_stb && (wr_idx[14:6] == 9'd0);      // 0x000 .. 0x0FC
    wire       wr_mant = wr_stb && (wr_idx[14:6] == 9'd1);      // 0x100 .. 0x1FC
    wire [5:0] wsub    = wr_idx[5:0];
    wire       wr_ctrl   = wr_lo && (wsub == 6'd0);
    wire       wr_commit = wr_lo && (wsub == 6'd26);
    integer    pk;
    always @(posedge clk) begin
        if (rst_h) begin
            r_rows <= 32'd0; r_groups <= 32'd0; r_pad <= 32'd0; r_bursts <= 32'd0;
            r_chunk <= 32'd16; r_step <= 32'd0; r_maxout <= 32'd8; r_res_start <= 32'd0;
            r_base_lo <= 32'd0; r_base_hi <= 32'd0; r_act_group <= 32'd0; r_act_exp <= 32'd0;
            r_port_off <= {(NPORTS*32){1'b0}};
            r_mode <= 32'd0; r_act_addr_lo <= 32'd0; r_act_addr_hi <= 32'd0;
            r_res_addr_lo <= 32'd0; r_res_addr_hi <= 32'd0;
        end else begin
            if (wr_lo) begin
                case (wsub)
                    6'd4:  r_rows      <= wr_data;
                    6'd5:  r_groups    <= wr_data;
                    6'd6:  r_pad       <= wr_data;
                    6'd7:  r_bursts    <= wr_data;
                    6'd8:  r_chunk     <= wr_data;
                    6'd9:  r_step      <= wr_data;
                    6'd10: r_maxout    <= wr_data;
                    6'd11: r_res_start <= wr_data;
                    6'd12: r_base_lo   <= wr_data;
                    6'd13: r_base_hi   <= wr_data;
                    6'd14: r_mode      <= wr_data;
                    6'd24: r_act_group <= wr_data;
                    6'd25: r_act_exp   <= wr_data;
                    6'd28: r_act_addr_lo <= wr_data;
                    6'd29: r_act_addr_hi <= wr_data;
                    6'd30: r_res_addr_lo <= wr_data;
                    6'd31: r_res_addr_hi <= wr_data;
                    default: ;
                endcase
                if (wsub[5:3] == 3'b010)
                    for (pk = 0; pk < NPORTS; pk = pk + 1)
                        if (wsub[2:0] == pk) r_port_off[pk*32 +: 32] <= wr_data;
            end
        end
        // no reset on the wide activation word (keeps the reset net small); the driver
        // writes all 40 words of a group before ACT_COMMIT
        if (wr_mant && wsub < 6'd40) r_act_mant[wsub*32 +: 32] <= wr_data;
    end

    // ------------------------------------------------------------ run control
    reg [3:0]  srst_sr;
    reg        run_busy, done, armed, last_seen, go_pend;
    reg [11:0] seq;
    reg [3:0]  fin_sr;
    reg        cb_r, all_idle_r;
    reg        axi_err_st, ovf_st, aux_err_st;
    reg [2:0]  st_sr;
    reg        start_r;                                  // the run's start pulse (registered fan-out below)
    reg [2:0]  q_mode;                                   // snapshot of MODE: {RES_NOCAP, RES_DMA, ACT_DMA}
    wire       core_busy;
    wire       aux_idle, aux_err_evt, aux_act_done, aux_res_done, aux_res_room;
    wire [NPORTS-1:0] p_idle, p_ar_evt, p_fin, p_err, p_ovf;

    wire srst_req = wr_ctrl & wr_data[1];
    wire srst_any = srst_req | (|srst_sr);
    wire go_req   = wr_ctrl & wr_data[0] & ~wr_data[1] & ~run_busy & ~go_pend;
    wire go_acc   = go_pend & all_idle_r & ~(|srst_sr);  // registers only (a SOFT_RST in this very
                                                         // cycle still wins: the reset branches below)
    wire busy_now = run_busy | go_pend | (|srst_sr) | ~all_idle_r;

    reg  c_v, c_l;                                       // registered result stream
    reg [31:0] c_d;

    always @(posedge clk) begin
        cb_r       <= core_busy;
        all_idle_r <= (&p_idle) & aux_idle;
        if (rst_h) srst_sr <= 4'd0;
        else       srst_sr <= {srst_sr[2:0], srst_req};
        if (rst_h || srst_any) begin
            run_busy <= 1'b0; done <= 1'b0; armed <= 1'b0; last_seen <= 1'b0; go_pend <= 1'b0;
            seq <= 12'd0; fin_sr <= 4'd0; st_sr <= 3'd0;
        end else begin
            seq    <= {seq[10:0], go_acc};
            st_sr  <= {st_sr[1:0], start_r};
            fin_sr <= {fin_sr[2:0], run_busy & armed & last_seen & ~cb_r & all_idle_r
                                    & (~q_mode[1] | aux_res_done)};
            if (go_req) begin go_pend <= 1'b1; done <= 1'b0; end
            if (go_acc) begin
                go_pend <= 1'b0; run_busy <= 1'b1; armed <= 1'b0; last_seen <= 1'b0; fin_sr <= 4'd0;
            end
            if (st_sr[2])    armed <= 1'b1;               // 3 cycles after the start pulse
            if (c_v && c_l)  last_seen <= 1'b1;
            if (fin_sr[3] && run_busy && armed) begin
                run_busy <= 1'b0; armed <= 1'b0; done <= 1'b1;
            end
        end
        if (rst_h || srst_any) begin
            axi_err_st <= 1'b0; ovf_st <= 1'b0; aux_err_st <= 1'b0;
        end else begin
            if (|p_err) axi_err_st <= 1'b1;
            if (|p_ovf) ovf_st <= 1'b1;
            if (aux_err_evt) aux_err_st <= 1'b1;
        end
    end

    // registered control fan-out
    // start: 8 cycles after go_acc (flush over), or -- MODE.ACT_DMA -- when the activation
    // fetch that is started at that point has committed its last record
    reg flush_p, core_rst, clr_q, act_we_r, act_go_r, res_en_r;
    (* max_fanout = 128 *) reg act_sel;                  // the core's act_* port listens to the DMA
    always @(posedge clk) begin
        flush_p  <= rst_h | srst_any | go_acc | seq[0];
        core_rst <= rst_h | srst_any | go_acc | seq[0];
        start_r  <= (q_mode[0] ? (aux_act_done & run_busy) : seq[7]) & ~rst_h & ~srst_any;
        act_go_r <= seq[7] & q_mode[0] & ~rst_h & ~srst_any;
        res_en_r <= run_busy & q_mode[1] & ~rst_h & ~srst_any;
        if (rst_h || srst_any || start_r) act_sel <= 1'b0;
        else if (seq[6] && q_mode[0])     act_sel <= 1'b1;
        clr_q    <= rst_h | srst_any | go_acc;
        act_we_r <= wr_commit & ~run_busy & ~go_pend & ~srst_any & ~rst_h & (r_act_group < GMAX);
    end

    // configuration snapshot (what the core and the masters see)
    reg [31:0]          q_rows, q_pad, q_bursts, q_step;
    reg [7:0]           q_groups, q_maxout;
    reg [8:0]           q_chunk;
    reg [NPORTS*AW-1:0] q_base;
    reg [AW-1:0]        q_act_addr, q_res_addr;
    wire [AW-1:0]       base_w = {r_base_hi[AW-33:0], r_base_lo};
    integer             qk;
    always @(posedge clk) begin
        if (go_acc) begin
            q_rows   <= r_rows;
            q_groups <= r_groups[7:0];
            q_pad    <= r_pad;
            q_bursts <= r_bursts;
            q_chunk  <= r_chunk[8:0];
            q_step   <= r_step;
            q_maxout <= (|r_maxout[31:8]) ? 8'hFF : ((r_maxout[7:0] == 8'd0) ? 8'd1 : r_maxout[7:0]);
            q_mode     <= r_mode[2:0];
            q_act_addr <= {r_act_addr_hi[AW-33:0], r_act_addr_lo};
            q_res_addr <= {r_res_addr_hi[AW-33:0], r_res_addr_lo};
            for (qk = 0; qk < NPORTS; qk = qk + 1)
                q_base[qk*AW +: AW] <= base_w + {{(AW-32){1'b0}}, r_port_off[qk*32 +: 32]};
        end
    end

    // ------------------------------------------------------------ read masters + FIFOs
    wire [NPORTS-1:0]     st_valid, st_ready;
    wire [NPORTS*128-1:0] st_data;
    wire [NPORTS*32-1:0]  p_beats, p_fmax;
    wire [NPORTS*128-1:0] p_xsum;
    genvar gk;
    generate
        for (gk = 0; gk < MP; gk = gk + 1) begin : port
            if (gk < NPORTS) begin : on
                axi_rd_stream #(.AW(AW), .IDW(IDW), .FIFO_LG(FIFO_LG)) u (
                    .clk(clk), .rst(rst_h), .flush(flush_p), .go(start_r),
                    .cfg_base(q_base[gk*AW +: AW]), .cfg_bursts(q_bursts), .cfg_chunk(q_chunk),
                    .cfg_step(q_step), .cfg_maxout(q_maxout),
                    .arid(v_arid[gk*IDW +: IDW]), .araddr(v_araddr[gk*AW +: AW]),
                    .arlen(v_arlen[gk*8 +: 8]), .arsize(v_arsize[gk*3 +: 3]),
                    .arburst(v_arburst[gk*2 +: 2]), .arlock(v_arlock[gk]),
                    .arcache(v_arcache[gk*4 +: 4]), .arprot(v_arprot[gk*3 +: 3]),
                    .arqos(v_arqos[gk*4 +: 4]), .arvalid(v_arvalid[gk]), .arready(v_arready[gk]),
                    .rid(v_rid[gk*IDW +: IDW]), .rdata(v_rdata[gk*128 +: 128]),
                    .rresp(v_rresp[gk*2 +: 2]), .rlast(v_rlast[gk]), .rvalid(v_rvalid[gk]),
                    .rready(v_rready[gk]),
                    .m_valid(st_valid[gk]), .m_ready(st_ready[gk]), .m_data(st_data[gk*128 +: 128]),
                    .idle(p_idle[gk]), .ar_evt(p_ar_evt[gk]), .fin_seen(p_fin[gk]),
                    .err_evt(p_err[gk]), .ovf_evt(p_ovf[gk]),
                    .beats(p_beats[gk*32 +: 32]), .xsum(p_xsum[gk*128 +: 128]),
                    .fifo_max(p_fmax[gk*32 +: 32]));
            end else begin : off
                assign v_arid[gk*IDW +: IDW]  = {IDW{1'b0}};
                assign v_araddr[gk*AW +: AW]  = {AW{1'b0}};
                assign v_arlen[gk*8 +: 8]     = 8'd0;
                assign v_arsize[gk*3 +: 3]    = 3'b100;
                assign v_arburst[gk*2 +: 2]   = 2'b01;
                assign v_arlock[gk]           = 1'b0;
                assign v_arcache[gk*4 +: 4]   = 4'b0011;
                assign v_arprot[gk*3 +: 3]    = 3'b000;
                assign v_arqos[gk*4 +: 4]     = 4'b0000;
                assign v_arvalid[gk]          = 1'b0;
                assign v_rready[gk]           = 1'b1;
            end
        end
    endgenerate

    // ------------------------------------------------------------ the engine
    wire              c_mv, c_ml;
    wire [31:0]       c_md;
    wire              dbg_pop, dbg_stall_in, dbg_stall_out;
    wire [NPORTS-1:0] dbg_empty;
    // activation write port: the ACT_* registers, or the aux DMA while it fetches
    wire                  aux_act_we;
    wire [7:0]            aux_act_group;
    wire [1279:0]         aux_act_mant;
    wire signed [EXW-1:0] aux_act_exp;
    wire                  core_act_we    = act_we_r | aux_act_we;       // never both: act_we_r needs !run_busy
    wire [7:0]            core_act_group = act_sel ? aux_act_group : r_act_group[7:0];
    wire [1279:0]         core_act_mant  = act_sel ? aux_act_mant  : r_act_mant;
    wire [EXW-1:0]        core_act_exp   = act_sel ? aux_act_exp   : r_act_exp[EXW-1:0];
    // result stream: always taken, except when the DDR write FIFO is nearly full (RES_DMA only)
    wire                  core_m_rdy = aux_res_room;

    gemv_stream #(.NPORTS(NPORTS), .GMAX(GMAX), .EXW(EXW)) u_core (
        .clk(clk), .rst(core_rst),
        .start(start_r), .cfg_groups(q_groups), .cfg_rows(q_rows), .cfg_pad(q_pad), .busy(core_busy),
        .act_we(core_act_we), .act_group(core_act_group), .act_mant(core_act_mant),
        .act_exp(core_act_exp),
        .s_valid(st_valid), .s_ready(st_ready), .s_data(st_data),
        .m_valid(c_mv), .m_ready(core_m_rdy), .m_data(c_md), .m_last(c_ml),
        .dbg_pop(dbg_pop), .dbg_stall_in(dbg_stall_in), .dbg_stall_out(dbg_stall_out),
        .dbg_empty(dbg_empty));

    // ------------------------------------------------------------ result capture
    always @(posedge clk) begin
        c_v <= c_mv & core_m_rdy & ~core_rst;
        c_l <= c_ml;
        c_d <= c_md;
    end

    // ------------------------------------------------------------ aux DMA (version 2)
    axi_aux_dma #(.AW(AW), .IDW(IDW), .EXW(EXW), .GMAX(GMAX)) u_aux (
        .clk(clk), .rst(rst_h), .flush(flush_p),
        .act_go(act_go_r), .act_groups(q_groups), .act_addr(q_act_addr), .act_done(aux_act_done),
        .act_we(aux_act_we), .act_group(aux_act_group), .act_mant(aux_act_mant), .act_exp(aux_act_exp),
        .res_en(res_en_r), .res_start(start_r), .res_addr(q_res_addr),
        .res_v(c_v), .res_d(c_d), .res_last(c_l), .res_room(aux_res_room), .res_done(aux_res_done),
        .idle(aux_idle), .err_evt(aux_err_evt),
        .awid(aux_axi_awid), .awaddr(aux_axi_awaddr), .awlen(aux_axi_awlen), .awsize(aux_axi_awsize),
        .awburst(aux_axi_awburst), .awlock(aux_axi_awlock), .awcache(aux_axi_awcache),
        .awprot(aux_axi_awprot), .awqos(aux_axi_awqos), .awvalid(aux_axi_awvalid),
        .awready(aux_axi_awready),
        .wdata(aux_axi_wdata), .wstrb(aux_axi_wstrb), .wlast(aux_axi_wlast), .wvalid(aux_axi_wvalid),
        .wready(aux_axi_wready),
        .bid(aux_axi_bid), .bresp(aux_axi_bresp), .bvalid(aux_axi_bvalid), .bready(aux_axi_bready),
        .arid(aux_axi_arid), .araddr(aux_axi_araddr), .arlen(aux_axi_arlen), .arsize(aux_axi_arsize),
        .arburst(aux_axi_arburst), .arlock(aux_axi_arlock), .arcache(aux_axi_arcache),
        .arprot(aux_axi_arprot), .arqos(aux_axi_arqos), .arvalid(aux_axi_arvalid),
        .arready(aux_axi_arready),
        .rid(aux_axi_rid), .rdata(aux_axi_rdata), .rresp(aux_axi_rresp), .rlast(aux_axi_rlast),
        .rvalid(aux_axi_rvalid), .rready(aux_axi_rready));

    // ------------------------------------------------------------ result window
    // 2^(RES_LG - 10) separate RAMs of 1024 x 32 (one RAMB36 each), every one with its own
    // output register; the bank mux is registered again (cq3). Nothing is cascaded.
    localparam integer BANK_LG = (RES_LG > 10) ? 10 : RES_LG;
    localparam integer NB      = 1 << (RES_LG - BANK_LG);
    reg [31:0]       skip_left;
    reg              skip_z;
    reg [RES_LG:0]   wptr;
    reg              cw_we;
    reg [RES_LG-1:0] cw_addr;
    reg [31:0]       cw_data;
    reg [31:0]       rows_done, x_or, sum_lo, sum_hi;
    reg              sum_c;
    wire [32:0]      sum33 = {1'b0, sum_lo} + {1'b0, c_d};
    always @(posedge clk) begin
        if (go_acc) begin
            skip_left <= r_res_start; skip_z <= (r_res_start == 32'd0); wptr <= {(RES_LG+1){1'b0}};
        end else if (c_v) begin
            if (!skip_z) begin
                skip_left <= skip_left - 32'd1; skip_z <= (skip_left == 32'd1);
            end else if (!wptr[RES_LG]) begin
                wptr <= wptr + 1'b1;
            end
        end
        cw_we   <= c_v & skip_z & ~wptr[RES_LG] & ~clr_q & ~q_mode[2];
        cw_addr <= wptr[RES_LG-1:0];
        cw_data <= c_d;

        if (clr_q) begin
            rows_done <= 32'd0; x_or <= 32'd0; sum_lo <= 32'd0; sum_hi <= 32'd0; sum_c <= 1'b0;
        end else begin
            if (c_v) begin
                rows_done <= rows_done + 32'd1;
                x_or      <= x_or ^ c_d;
                sum_lo    <= sum33[31:0];
            end
            sum_c <= c_v & sum33[32];
            if (sum_c) sum_hi <= sum_hi + 32'd1;
        end
    end

    reg  [14:0]       rd_idx;                            // AXI-Lite read address [16:2]
    wire [RES_LG:0]   cw_bank_w = {1'b0, cw_addr} >> BANK_LG;
    wire [RES_LG:0]   rd_bank_w = {1'b0, rd_idx[RES_LG-1:0]} >> BANK_LG;
    reg  [NB*32-1:0]  cq2_all;
    genvar gb;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : bank
            (* ram_style = "block" *) reg [31:0] mem [0:(1<<BANK_LG)-1];
            reg [31:0] q1;
            always @(posedge clk) begin
                if (cw_we && cw_bank_w == gb) mem[cw_addr[BANK_LG-1:0]] <= cw_data;
                q1 <= mem[rd_idx[BANK_LG-1:0]];
                cq2_all[gb*32 +: 32] <= q1;
            end
        end
    endgenerate
    reg [31:0] cq3;
    always @(posedge clk) cq3 <= cq2_all[rd_bank_w*32 +: 32];

    // ------------------------------------------------------------ argmax (version 2)
    wire [31:0] amax_idx, amax_val;
    gemv_argmax u_argmax (.clk(clk), .clr(clr_q), .v(c_v), .d(c_d), .idx(amax_idx), .val(amax_val));

    // ------------------------------------------------------------ counters
    reg        act_on, res_on;
    reg [1:0]  res_sr;
    reg [31:0] cyc_act, cyc_res;
    reg        cyc_on, cyc_c, str_on, str_started, str_c, any_ar, str_set, all_fin_r;
    reg [31:0] cyc_lo, cyc_hi, str_lo, str_hi;
    reg [31:0] tuples, stall_in, stall_out, errcnt;
    reg [2:0]  err_n;
    integer    ek;
    always @* begin
        err_n = 3'd0;
        for (ek = 0; ek < NPORTS; ek = ek + 1) err_n = err_n + {2'b00, p_err[ek]};
    end
    always @(posedge clk) begin
        // CYC: GO -> last result row
        if (rst_h || srst_any || go_acc) begin
            cyc_on <= ~rst_h & ~srst_any; cyc_lo <= 32'd0; cyc_hi <= 32'd0; cyc_c <= 1'b0;
        end else begin
            if (c_v && c_l) cyc_on <= 1'b0;
            if (cyc_on) cyc_lo <= cyc_lo + 32'd1;
            cyc_c <= cyc_on & (cyc_lo == 32'hFFFF_FFFF);
            if (cyc_c) cyc_hi <= cyc_hi + 32'd1;
        end
        // CYC_STREAM: first AR handshake -> last R beat
        any_ar    <= (|p_ar_evt) & run_busy;
        str_set   <= any_ar;
        all_fin_r <= &p_fin;
        if (clr_q) begin
            str_on <= 1'b0; str_started <= 1'b0; str_lo <= 32'd0; str_hi <= 32'd0; str_c <= 1'b0;
            any_ar <= 1'b0; str_set <= 1'b0;
        end else begin
            if (str_set && !str_started) begin str_on <= 1'b1; str_started <= 1'b1; end
            else if (all_fin_r)          str_on <= 1'b0;
            if (str_on) str_lo <= str_lo + 32'd1;
            str_c <= str_on & (str_lo == 32'hFFFF_FFFF);
            if (str_c) str_hi <= str_hi + 32'd1;
        end
        // CYC_ACT: seq[7] (where a MODE.ACT_DMA = 0 run would start) -> the fetch is done
        // CYC_RES: the last result row -> the last write response
        if (clr_q) begin
            act_on <= 1'b0; res_on <= 1'b0; cyc_act <= 32'd0; cyc_res <= 32'd0; res_sr <= 2'd0;
        end else begin
            res_sr <= {res_sr[0], c_v & c_l & res_en_r};
            if (aux_act_done || !run_busy) act_on <= 1'b0;
            else if (seq[7] && q_mode[0])  act_on <= 1'b1;
            if (act_on) cyc_act <= cyc_act + 32'd1;
            if (aux_res_done || !run_busy)       res_on <= 1'b0;
            else if (res_sr[1])                  res_on <= 1'b1;   // so that the count is exactly Bw - L
            if (res_on) cyc_res <= cyc_res + 32'd1;
        end
        // engine telemetry
        if (clr_q) begin
            tuples <= 32'd0; stall_in <= 32'd0; stall_out <= 32'd0; errcnt <= 32'd0;
        end else begin
            if (dbg_pop)       tuples    <= tuples + 32'd1;
            if (dbg_stall_in)  stall_in  <= stall_in + 32'd1;
            if (dbg_stall_out) stall_out <= stall_out + 32'd1;
            errcnt <= errcnt + {29'd0, err_n};
        end
    end

    reg [NPORTS*32-1:0] p_empty;
    generate
        for (gk = 0; gk < NPORTS; gk = gk + 1) begin : pe
            always @(posedge clk) begin
                if (clr_q)              p_empty[gk*32 +: 32] <= 32'd0;
                else if (dbg_empty[gk]) p_empty[gk*32 +: 32] <= p_empty[gk*32 +: 32] + 32'd1;
            end
        end
    endgenerate

    // ------------------------------------------------------------ AXI-Lite read channel
    // AR accept (cycle 0) -> group muxes / RAM read (1) -> OR / RAM output registers (2) ->
    // window bank mux (3) -> RDATA + RVALID (4).
    wire [NPORTS*256-1:0] pregs;
    generate
        for (gk = 0; gk < NPORTS; gk = gk + 1) begin : pr
            assign pregs[gk*256 +: 256] = {p_xsum[gk*128 +: 128], 32'd0, p_fmax[gk*32 +: 32],
                                           p_empty[gk*32 +: 32], p_beats[gk*32 +: 32]};
        end
    endgenerate
    wire [31:0] status_w = {27'd0, aux_err_st, ovf_st, axi_err_st, done, busy_now};
    wire [31:0] params_w = {VERSION_V, RES_LG[7:0], GMAX[7:0], 4'd0, NPORTS[3:0]};

    reg [3:0]  rd_sr;
    reg        rd_busy;
    reg [31:0] g_cfg, g_act, g_cnt, g_port, g_all;
    assign S_AXI_RRESP = 2'b00;
    always @(posedge clk) begin
        if (rst_h) begin
            S_AXI_ARREADY <= 1'b0; S_AXI_RVALID <= 1'b0; rd_sr <= 4'd0; rd_busy <= 1'b0;
        end else begin
            S_AXI_ARREADY <= 1'b0;
            rd_sr <= {rd_sr[2:0], 1'b0};
            if (S_AXI_ARVALID && !S_AXI_ARREADY && !rd_busy) begin
                S_AXI_ARREADY <= 1'b1; rd_busy <= 1'b1; rd_sr[0] <= 1'b1;
                rd_idx <= S_AXI_ARADDR[16:2];
            end
            if (rd_sr[3]) begin
                S_AXI_RVALID <= 1'b1;
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0; rd_busy <= 1'b0;
            end
        end

        // stage 1: one registered mux per register group (zero when not addressed)
        g_cfg <= 32'd0; g_act <= 32'd0; g_cnt <= 32'd0; g_port <= 32'd0;
        if (rd_idx[14:6] == 9'd0) begin
            case (rd_idx[5:0])
                6'd1:  g_cfg <= status_w;
                6'd2:  g_cfg <= IDCODE_V;
                6'd3:  g_cfg <= params_w;
                6'd4:  g_cfg <= r_rows;
                6'd5:  g_cfg <= r_groups;
                6'd6:  g_cfg <= r_pad;
                6'd7:  g_cfg <= r_bursts;
                6'd8:  g_cfg <= r_chunk;
                6'd9:  g_cfg <= r_step;
                6'd10: g_cfg <= r_maxout;
                6'd11: g_cfg <= r_res_start;
                6'd12: g_cfg <= r_base_lo;
                6'd13: g_cfg <= r_base_hi;
                6'd14: g_cfg <= r_mode;
                6'd24: g_cfg <= r_act_group;
                6'd25: g_cfg <= r_act_exp;
                6'd28: g_cfg <= r_act_addr_lo;
                6'd29: g_cfg <= r_act_addr_hi;
                6'd30: g_cfg <= r_res_addr_lo;
                6'd31: g_cfg <= r_res_addr_hi;
                default: ;
            endcase
            if (rd_idx[5:3] == 3'b010 && rd_idx[2:0] < NPORTS)
                g_cfg <= r_port_off[rd_idx[2:0]*32 +: 32];
        end
        if (rd_idx[14:6] == 9'd1 && rd_idx[5:0] < 6'd40)
            g_act <= r_act_mant[rd_idx[5:0]*32 +: 32];
        if (rd_idx[14:6] == 9'd2 && rd_idx[5:4] == 2'b00) begin
            case (rd_idx[3:0])
                4'd0:  g_cnt <= cyc_lo;
                4'd1:  g_cnt <= cyc_hi;
                4'd2:  g_cnt <= str_lo;
                4'd3:  g_cnt <= str_hi;
                4'd4:  g_cnt <= rows_done;
                4'd5:  g_cnt <= sum_lo;
                4'd6:  g_cnt <= sum_hi;
                4'd7:  g_cnt <= x_or;
                4'd8:  g_cnt <= tuples;
                4'd9:  g_cnt <= stall_in;
                4'd10: g_cnt <= stall_out;
                4'd11: g_cnt <= errcnt;
                4'd12: g_cnt <= amax_idx;
                4'd13: g_cnt <= amax_val;
                4'd14: g_cnt <= cyc_act;
                4'd15: g_cnt <= cyc_res;
                default: ;
            endcase
        end
        if (rd_idx[14:6] == 9'd3 && rd_idx[5:0] < NPORTS*8)
            g_port <= pregs[rd_idx[5:0]*32 +: 32];

        // stage 2 (the window RAMs' output registers sit in the bank generate above)
        g_all <= g_cfg | g_act | g_cnt | g_port;

        // stage 4 (stage 3 = cq3, the registered window bank mux; rd_idx is stable throughout)
        if (rd_sr[3]) S_AXI_RDATA <= rd_idx[14] ? cq3 : g_all;
    end

    wire unused_ok = &{1'b0, S_AXI_AWPROT, S_AXI_ARPROT, S_AXI_WSTRB, unused_wr, 1'b0};
endmodule

// -----------------------------------------------------------------------------
// gemv_argmax — index and value of the largest result row of a run.
// Rows arrive in order on v/d (raw fp32 bits, at most one per cycle). Ordering is
// the IEEE comparison of finite values: the word is mapped to a signed key
// (sign ? -magnitude : +magnitude, so -0 == +0) and a row replaces the best one
// only if its key is strictly greater -- the LOWEST index wins a tie. Two
// register stages; idx/val are final two cycles after the last row.
// -----------------------------------------------------------------------------
module gemv_argmax (
    input  wire        clk,
    input  wire        clr,
    input  wire        v,
    input  wire [31:0] d,
    output reg  [31:0] idx,
    output reg  [31:0] val
);
    reg               k_v, have;
    reg signed [31:0] k_key, best;
    reg [31:0]        k_d, k_idx, n;
    wire [31:0]       mag = {1'b0, d[30:0]};
    always @(posedge clk) begin
        if (clr) begin
            k_v <= 1'b0; have <= 1'b0; n <= 32'd0; idx <= 32'd0; val <= 32'd0;
        end else begin
            k_v <= v;
            if (v) begin
                k_key <= d[31] ? (32'd0 - mag) : mag;
                k_d   <= d;
                k_idx <= n;
                n     <= n + 32'd1;
            end
            if (k_v && (!have || k_key > best)) begin
                have <= 1'b1; best <= k_key; idx <= k_idx; val <= k_d;
            end
        end
    end
endmodule

`default_nettype wire
