// -----------------------------------------------------------------------------
// axi_rd_stream — one AXI4 read master + FIFO: turns a striped DDR region into a
// valid/ready stream of 128-bit beats (doc 10, rung R2a; the bus side is R0's
// axi_rd_probe, the consumer is one port of gemv_stream).
//
// On `go` it latches the configuration and issues cfg_bursts INCR bursts of
// cfg_chunk beats at cfg_base + n * cfg_step, all with ARID 0 (data returns in
// order), keeping at most cfg_maxout bursts in flight.
//
// CREDITS. A burst is issued only when the FIFO has room for it on top of
// everything already promised: `free` = DEPTH - (beats reserved by issued bursts
// that have not yet left the FIFO); a burst needs free >= chunk and takes chunk
// off it, every FIFO pop gives one back. So RREADY is tied to 1 for ever (as in
// the R0 measurement) and the FIFO cannot overflow; `ovf_evt` exists to prove it.
//
// TIMING. Nothing but flip-flops touches the bus pins:
//   * ARREADY feeds one flop (arvalid) and one event flop. The price is one idle
//     cycle after every AR handshake: at most one AR per 2 cycles, which is one
//     beat per cycle for every chunk >= 2 (chunk = 1 is limited to 0.5).
//   * R is registered twice (r1 next to the pins, r2 next to the RAM) before the
//     FIFO write, the checksum and the counters.
//   * the issue decision `can_r` is a registered flag, one cycle stale. That is
//     exact with respect to issues (an issue raises arvalid, which blocks the next
//     issue for at least a cycle) and conservative with respect to everything
//     else (pops and RLASTs only make it more permissive).
//   * the FIFO read side is a 2-stage block-RAM read (address -> RAM register ->
//     output register) with clock enables, then a skid buffer, so m_valid/m_data
//     come from flops and m_ready feeds only the skid.
//
// flush (soft reset, or the wrapper's pre-run flush): stop issuing, empty the
// FIFO and the read pipeline, clear the counters. AXI state is NOT dropped: a
// pending ARVALID is held until ARREADY and bursts already issued are received
// and discarded (`drop`); `idle` stays low until that drain has finished. Only
// `rst` (the hard reset) clears the AXI state.
//
// FIFO latency: R beat on the bus -> m_valid is 6 cycles when the stream is
// flowing (r1, r2, RAM write, RAM read, output register, skid).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

// registered ready AND registered valid/data (2 entries)
module ars_skid #(parameter integer W = 128) (
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

module axi_rd_stream #(
    parameter integer AW      = 49,
    parameter integer IDW     = 6,
    parameter integer FIFO_LG = 11               // log2 FIFO depth in beats
) (
    input  wire               clk,
    input  wire               rst,               // hard reset (also resets the AXI state)
    input  wire               flush,             // abort + empty (AXI state drains, see header)
    input  wire               go,                // 1-cycle pulse, only while idle
    // configuration, sampled at go
    input  wire [AW-1:0]      cfg_base,          // address of this port's first burst
    input  wire [31:0]        cfg_bursts,
    input  wire [8:0]         cfg_chunk,         // beats per burst, power of two in 1..256
    input  wire [31:0]        cfg_step,          // bytes between successive bursts
    input  wire [7:0]         cfg_maxout,        // bursts in flight, >= 1
    // AXI4 read address
    output wire [IDW-1:0]     arid,
    output reg  [AW-1:0]      araddr,
    output wire [7:0]         arlen,
    output wire [2:0]         arsize,
    output wire [1:0]         arburst,
    output wire               arlock,
    output wire [3:0]         arcache,
    output wire [2:0]         arprot,
    output wire [3:0]         arqos,
    output reg                arvalid,
    input  wire               arready,
    // AXI4 read data
    input  wire [IDW-1:0]     rid,
    input  wire [127:0]       rdata,
    input  wire [1:0]         rresp,
    input  wire               rlast,
    input  wire               rvalid,
    output wire               rready,
    // beat stream
    output wire               m_valid,
    input  wire               m_ready,
    output wire [127:0]       m_data,
    // status
    output reg                idle,              // nothing to issue, nothing in flight, not draining
    output reg                ar_evt,            // an AR handshake happened (1 cycle late)
    output reg                fin_seen,          // the run's last R beat has arrived (sticky; 2 cycles late)
    output reg                err_evt,           // a counted beat had RRESP != OKAY (pulse)
    output reg                ovf_evt,           // a beat arrived with the FIFO full (pulse; must never happen)
    output reg  [31:0]        beats,             // beats received (not counting drained ones)
    output reg  [127:0]       xsum,              // xor of those beats
    output reg  [31:0]        fifo_max           // maximum FIFO fill seen
);
    localparam integer DEPTH = 1 << FIFO_LG;

    // ------------------------------------------------------------ local resets
    reg rst_q, fl_q, go_q;
    always @(posedge clk) begin
        rst_q <= rst;
        fl_q  <= rst | flush;
        go_q  <= go & ~rst & ~flush;
    end

    // ------------------------------------------------------------ R input registers
    reg         r1_v, r1_last, r2_v, r2_bad;
    reg [1:0]   r1_resp;
    reg [127:0] r1_d, r2_d;
    assign rready = 1'b1;
    always @(posedge clk) begin
        r1_v    <= rvalid;                       // r1: flops only, straight off the pins
        r1_last <= rlast;
        r1_resp <= rresp;
        r1_d    <= rdata;
        r2_v    <= r1_v & ~rst_q;
        r2_bad  <= (r1_resp != 2'b00);
        r2_d    <= r1_d;
    end

    // ------------------------------------------------------------ AR issue
    reg [31:0]        bursts_left;
    reg               none_left;
    reg [8:0]         chunk_q;
    reg [31:0]        step_q;
    reg [8:0]         maxout_q;
    reg [AW-1:0]      next_addr;
    reg [8:0]         outstanding;       // bursts allocated (arvalid raised) whose RLAST has not arrived
    reg [FIFO_LG:0]   free;              // FIFO credits, in beats
    reg               can_r, drop;
    wire              rd;                // FIFO pop (below)

    assign arid    = {IDW{1'b0}};
    assign arlen   = chunk_q[7:0] - 8'd1;        // chunk = 256 -> 8'hFF
    assign arsize  = 3'b100;                     // 16 bytes
    assign arburst = 2'b01;                      // INCR
    assign arlock  = 1'b0;
    assign arcache = 4'b0011;                    // as the R0 measurement
    assign arprot  = 3'b000;
    assign arqos   = 4'b0000;

    wire [FIFO_LG:0] chunk_w = chunk_q;          // zero-extended (FIFO_LG >= 8)
    wire raise       = can_r & ~arvalid & ~fl_q;
    wire r1_lastfire = r1_v & r1_last;

    always @(posedge clk) begin
        if (rst_q) begin
            arvalid <= 1'b0; outstanding <= 9'd0; drop <= 1'b0; ar_evt <= 1'b0;
        end else begin
            arvalid     <= raise | (arvalid & ~arready);
            ar_evt      <= arvalid & arready & ~drop;
            outstanding <= outstanding + (raise ? 9'd1 : 9'd0) - (r1_lastfire ? 9'd1 : 9'd0);
            if (fl_q)                     drop <= 1'b1;
            else if (outstanding == 9'd0) drop <= 1'b0;
        end
        if (raise) araddr <= next_addr;

        if (fl_q) begin
            bursts_left <= 32'd0; none_left <= 1'b1; can_r <= 1'b0; free <= DEPTH;
            fin_seen <= 1'b0;
            chunk_q <= 9'd1; step_q <= 32'd0; maxout_q <= 9'd1; next_addr <= {AW{1'b0}};
        end else begin
            if (go_q) begin
                bursts_left <= cfg_bursts;
                none_left   <= (cfg_bursts == 32'd0);
                chunk_q     <= cfg_chunk;
                step_q      <= cfg_step;
                maxout_q    <= {1'b0, cfg_maxout};
                next_addr   <= cfg_base;
            end else if (raise) begin
                bursts_left <= bursts_left - 32'd1;
                none_left   <= (bursts_left == 32'd1);
                next_addr   <= next_addr + {{(AW-32){1'b0}}, step_q};
            end
            can_r <= ~none_left & ~go_q & (outstanding < maxout_q) & (free >= chunk_w);
            free  <= free - (raise ? chunk_w : {(FIFO_LG+1){1'b0}}) + (rd ? 1'b1 : 1'b0);
            if (r1_lastfire && none_left && !arvalid && outstanding == 9'd1 && !drop) fin_seen <= 1'b1;
        end
    end

    // ------------------------------------------------------------ FIFO (block RAM)
    (* ram_style = "block" *) reg [127:0] mem [0:DEPTH-1];
    reg [FIFO_LG-1:0] wp, rp;
    reg [FIFO_LG:0]   fill;
    reg               ne;                        // fill != 0
    reg               v1, v2;
    reg [127:0]       q1, q2;
    wire              sk_ready;
    wire              wr    = r2_v & ~drop;
    wire              full  = fill[FIFO_LG];
    wire              wr_ok = wr & ~full;
    wire              adv   = ~v2 | sk_ready;
    assign            rd    = ne & adv;

    always @(posedge clk) begin
        if (wr_ok) mem[wp] <= r2_d;
        if (rd)    q1 <= mem[rp];                // never the address being written: rd implies fill >= 1
        if (adv)   q2 <= q1;
    end

    always @(posedge clk) begin
        if (fl_q) begin
            wp <= {FIFO_LG{1'b0}}; rp <= {FIFO_LG{1'b0}}; fill <= {(FIFO_LG+1){1'b0}}; ne <= 1'b0;
            v1 <= 1'b0; v2 <= 1'b0;
            beats <= 32'd0; xsum <= 128'd0; fifo_max <= 32'd0; err_evt <= 1'b0; ovf_evt <= 1'b0;
        end else begin
            if (wr_ok) wp <= wp + 1'b1;
            if (rd)    rp <= rp + 1'b1;
            fill <= fill + (wr_ok ? 1'b1 : 1'b0) - (rd ? 1'b1 : 1'b0);
            ne   <= wr_ok | (rd ? (fill != 1) : ne);
            if (adv) begin v1 <= rd; v2 <= v1; end
            if (wr) begin
                beats <= beats + 32'd1;
                xsum  <= xsum ^ r2_d;
            end
            err_evt <= wr & r2_bad;
            ovf_evt <= wr & full;
            if (fill > fifo_max[FIFO_LG:0]) fifo_max <= fill;
        end
    end

    ars_skid #(.W(128)) u_skid (
        .clk(clk), .rst(fl_q),
        .s_valid(v2), .s_ready(sk_ready), .s_data(q2),
        .m_valid(m_valid), .m_ready(m_ready), .m_data(m_data));

    // ------------------------------------------------------------ idle
    always @(posedge clk) begin
        if (rst_q) idle <= 1'b0;
        else       idle <= ~arvalid & (outstanding == 9'd0) & none_left & ~drop & ~r2_v & ~fl_q & ~go_q;
    end

    // rid is not checked: one ARID, so the slave must return data in order
    wire unused_rid = ^rid;
endmodule

`default_nettype wire
