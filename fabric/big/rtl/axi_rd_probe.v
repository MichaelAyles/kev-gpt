// -----------------------------------------------------------------------------
// axi_rd_probe — one AXI4 read master that streams a DDR region as fast as the
// port will let it (doc 10, rung R0: the bandwidth probe).
//
// Reads BEATS 16-byte beats starting at BASE in INCR bursts of BURST beats,
// keeping up to MAXOUT bursts in flight, rotating ARID over NIDS ids. The data
// is folded into a 128-bit XOR checksum (order-insensitive, so out-of-order
// returns across ids are fine) so the board run can prove the bytes were real,
// and every non-OKAY RRESP is counted.
//
// Host contract (enforced by the driver, asserted in the testbench):
//   * BURST is a power of two in 1..256 and BASE is BURST*16-byte aligned, so no
//     burst ever crosses a 4 KB boundary (AXI rule).
//   * BEATS is a multiple of BURST.
//   * STEP (if nonzero) is a multiple of BURST*16. STEP = n*BURST*16 with master i
//     based at i*BURST*16 gives a port-interleaved layout: the ports read
//     disjoint data from neighbouring addresses (the weight-packer layout).
//
// AR issue: a burst is *allocated* (issued count and outstanding bumped) when
// arvalid is raised, so the outstanding cap is exact and back-to-back ARs issue
// one per cycle when arready stays high. Read-only master: the write channels
// are tied off in the wrapper.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module axi_rd_probe #(
    parameter integer AW  = 49,
    parameter integer IDW = 6
) (
    input  wire            clk,
    input  wire            rst,
    // control (static while running)
    input  wire            go,          // 1-cycle pulse
    input  wire            enable,
    input  wire [AW-1:0]   base,
    input  wire [31:0]     beats,
    input  wire [8:0]      burst,       // beats per burst, 1..256
    input  wire [31:0]     step,        // bytes between successive bursts; 0 = burst*16 (contiguous)
    input  wire [7:0]      maxout,      // bursts in flight, >= 1
    input  wire [6:0]      nids,        // ids to rotate over, 1..64
    input  wire [3:0]      qos,
    input  wire [3:0]      cache,
    // status
    output reg             done,
    output reg  [63:0]     cycles,
    output reg  [31:0]     rcv_beats,
    output reg  [31:0]     err_count,
    output reg  [127:0]    xsum,
    // AXI4 read address
    output reg  [IDW-1:0]  arid,
    output reg  [AW-1:0]   araddr,
    output wire [7:0]      arlen,
    output wire [2:0]      arsize,
    output wire [1:0]      arburst,
    output wire            arlock,
    output wire [3:0]      arcache,
    output wire [2:0]      arprot,
    output wire [3:0]      arqos,
    output reg             arvalid,
    input  wire            arready,
    // AXI4 read data
    input  wire [IDW-1:0]  rid,
    input  wire [127:0]    rdata,
    input  wire [1:0]      rresp,
    input  wire            rlast,
    input  wire            rvalid,
    output wire            rready
);
    assign arlen   = burst[7:0] - 8'd1;   // burst=256 -> 8'hFF (burst[7:0]=0, wraps)
    assign arsize  = 3'b100;              // 16 bytes
    assign arburst = 2'b01;               // INCR
    assign arlock  = 1'b0;
    assign arcache = cache;
    assign arprot  = 3'b000;
    assign arqos   = qos;
    assign rready  = 1'b1;                // a probe never back-pressures

    reg          running;
    reg [31:0]   issued_beats;
    reg [8:0]    outstanding;
    reg [AW-1:0] next_addr;
    reg [6:0]    id_ctr;

    wire ar_fire     = arvalid & arready;
    wire r_fire      = rvalid & rready;
    wire r_last_fire = r_fire & rlast;
    wire can_raise   = running && (issued_beats < beats) && (outstanding < {1'b0, maxout});
    wire raise       = (!arvalid || ar_fire) && can_raise;

    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0; done <= 1'b0; arvalid <= 1'b0;
            issued_beats <= 0; outstanding <= 0; next_addr <= 0; id_ctr <= 0;
            arid <= 0; araddr <= 0;
            cycles <= 0; rcv_beats <= 0; err_count <= 0; xsum <= 0;
        end else if (go) begin
            running <= enable; done <= ~enable; arvalid <= 1'b0;
            issued_beats <= 0; outstanding <= 0; next_addr <= base; id_ctr <= 0;
            cycles <= 0; rcv_beats <= 0; err_count <= 0; xsum <= 0;
        end else if (running) begin
            cycles <= cycles + 64'd1;

            if (raise) begin
                arvalid      <= 1'b1;
                araddr       <= next_addr;
                arid         <= id_ctr[IDW-1:0];
                next_addr    <= next_addr + ((step == 32'd0) ? {burst, 4'b0000} : step);
                issued_beats <= issued_beats + burst;
                id_ctr       <= (id_ctr + 7'd1 >= nids) ? 7'd0 : id_ctr + 7'd1;
            end else if (ar_fire) begin
                arvalid <= 1'b0;
            end
            outstanding <= outstanding + (raise ? 9'd1 : 9'd0) - (r_last_fire ? 9'd1 : 9'd0);

            if (r_fire) begin
                xsum      <= xsum ^ rdata;
                rcv_beats <= rcv_beats + 32'd1;
                if (rresp != 2'b00) err_count <= err_count + 32'd1;
                if (rcv_beats + 32'd1 == beats) begin
                    running <= 1'b0; done <= 1'b1;
                end
            end
        end
    end
endmodule
