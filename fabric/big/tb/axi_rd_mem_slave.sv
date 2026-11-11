// axi_rd_mem_slave — memory-backed AXI4 read slave model for the gemv_axi gate.
//
// Serves 128-bit INCR bursts from a beat memory loaded with $readmemh from the
// plusarg +image=<file> (one 32-hex-digit beat per line, beat i = bytes
// [mem_base + 16 i, +16), most significant byte first, i.e. byte 0 is the low
// byte of the word = RDATA[7:0]). In-order returns (one ID), per-burst random
// latency lat_min..lat_max from AR accept to the first beat, random ARREADY and
// RVALID gaps (percent of cycles), optional RRESP = SLVERR on one beat.
//
// Protocol checks (each bumps `errors` and prints SLAVE_ERR):
//   * RREADY low in any cycle (the wrapper promises RREADY = 1 for ever)
//   * a burst crossing a 4 KB boundary; size != 16 bytes; type != INCR
//   * ARLEN + 1 != exp_len (when exp_len != 0)
//   * ARID changing between bursts (single-ID ordering is what the wrapper relies on)
//   * ARVALID dropped, or ARADDR/ARLEN changed, before ARREADY
//   * an address outside the loaded image or not 16-byte aligned (data is then X)
//   * more than maxout_check bursts accepted and unfinished (when != 0)
//
// Event log for the CYC_STREAM check, restarted by `arm`: first_ar = cycle (`now`)
// of the first AR handshake, last_r = cycle of the latest R beat, n_ar, n_beats.
`timescale 1ns / 1ps

module axi_rd_mem_slave #(
    parameter integer AW     = 49,
    parameter integer IDW    = 6,
    parameter integer MEM_LG = 18,              // log2 beats of backing memory
    parameter integer QD     = 4096
) (
    input  wire            clk,
    input  wire            rst,
    input  wire [63:0]     now,
    input  wire            arm,
    input  wire [AW-1:0]   mem_base,
    input  wire [31:0]     mem_beats,
    input  wire [31:0]     exp_len,
    input  wire [31:0]     maxout_check,
    input  wire [31:0]     ar_gap,
    input  wire [31:0]     r_gap,
    input  wire [31:0]     lat_min,
    input  wire [31:0]     lat_max,
    input  wire [31:0]     seed0,
    input  wire [31:0]     err_beat,            // global beat index that returns SLVERR (0xFFFFFFFF: none)
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
    output reg  [31:0]     errors,
    output reg  [63:0]     first_ar,
    output reg  [63:0]     last_r,
    output reg             seen_ar,
    output reg  [31:0]     n_ar,
    output reg  [31:0]     n_beats
);
    reg [127:0] mem [0:(1<<MEM_LG)-1];
    reg [1023:0] imgfile;
    initial if ($value$plusargs("image=%s", imgfile)) $readmemh(imgfile, mem);

    reg [AW-1:0]  q_addr [0:QD-1];
    reg [8:0]     q_len  [0:QD-1];
    reg [IDW-1:0] q_id   [0:QD-1];
    reg [63:0]    q_t    [0:QD-1];
    integer qh, qt, inflight, seed, lat, total_beats;
    reg [8:0]     beat;
    reg           have_id, pend;
    reg [IDW-1:0] id0;
    reg [AW-1:0]  pend_addr;
    reg [7:0]     pend_len;
    reg [AW-1:0]  cur;
    reg [AW-1:0]  off;
    reg           seeded;

    function integer rpct; input dummy; begin rpct = ($random(seed) & 32'h7fffffff) % 100; end endfunction

    task err(input [255:0] what);
        begin
            errors = errors + 1;
            $display("SLAVE_ERR %0s cyc=%0d addr=%h len=%0d", what, now, araddr, arlen + 1);
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            qh = 0; qt = 0; inflight = 0; beat = 0; total_beats = 0; have_id = 0; pend = 0; seeded = 0;
            arready <= 1'b0; rvalid <= 1'b0; rlast <= 1'b0; rresp <= 2'b00; rid <= 0; rdata <= 128'd0;
            errors = 0; first_ar <= 0; last_r <= 0; seen_ar <= 1'b0; n_ar <= 0; n_beats <= 0;
        end else begin
            if (!seeded) begin seed = seed0; seeded = 1; end
            if (arm) begin seen_ar <= 1'b0; n_ar <= 0; n_beats <= 0; first_ar <= 0; last_r <= 0; end
            if (rready !== 1'b1) err("RREADY low");

            // ---- AR channel
            if (pend && !(arvalid === 1'b1 && araddr === pend_addr && arlen === pend_len))
                err("ARVALID/ARADDR changed before ARREADY");
            pend = (arvalid === 1'b1) && !arready;
            pend_addr = araddr; pend_len = arlen;
            if (arvalid === 1'b1 && arready) begin
                if (arsize != 3'b100 || arburst != 2'b01) err("bad size/burst");
                if (({1'b0, araddr[11:0]} + ((arlen + 1) * 16)) > 4096) err("4KB cross");
                if (exp_len != 0 && (arlen + 1) != exp_len) err("burst length");
                if (have_id && arid !== id0) err("ARID changed");
                if (araddr[3:0] != 0) err("unaligned address");
                off = araddr - mem_base;
                if (araddr < mem_base || ((off >> 4) + arlen + 1) > mem_beats) err("address outside image");
                have_id = 1; id0 = arid;
                lat = lat_min + ((lat_max > lat_min) ? (($random(seed) & 32'h7fffffff) % (lat_max - lat_min + 1)) : 0);
                q_addr[qt % QD] = araddr; q_len[qt % QD] = arlen + 1; q_id[qt % QD] = arid;
                q_t[qt % QD] = now + lat;
                qt = qt + 1; inflight = inflight + 1;
                if (maxout_check != 0 && inflight > maxout_check) err("bursts in flight > MAXOUT");
                if (!seen_ar && !arm) begin seen_ar <= 1'b1; first_ar <= now; end
                n_ar <= n_ar + 1;
            end
            arready <= ((qt - qh) < QD - 2) && (ar_gap == 0 || rpct(0) >= ar_gap);

            // ---- R channel (RREADY is 1: every presented beat transfers)
            if (rvalid) begin
                last_r <= now; n_beats <= n_beats + 1; total_beats = total_beats + 1;
                if (rlast) begin qh = qh + 1; inflight = inflight - 1; beat = 0; end
                else beat = beat + 1;
            end
            if (qh != qt && q_t[qh % QD] <= now && (r_gap == 0 || rpct(0) >= r_gap)) begin
                cur = q_addr[qh % QD] + beat * 16;
                off = cur - mem_base;
                rvalid <= 1'b1;
                rid    <= q_id[qh % QD];
                rdata  <= (cur >= mem_base && (off >> 4) < mem_beats) ? mem[off >> 4] : {128{1'bx}};
                rresp  <= (total_beats == err_beat) ? 2'b10 : 2'b00;
                rlast  <= (beat + 1 == q_len[qh % QD]);
            end else begin
                rvalid <= 1'b0; rlast <= 1'b0; rdata <= {128{1'bx}};
            end
        end
    end
endmodule
