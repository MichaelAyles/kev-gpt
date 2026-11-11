// -----------------------------------------------------------------------------
// axi_aux_dma — the aux AXI4 master of gemv_axi (register map version 2): the
// small host <-> fabric data that used to cross AXI-Lite word by word.
//
//   ACTIVATION FETCH (AR/R).  act_go: read act_batch * act_groups 256-byte
//   records starting at act_addr (4 KB aligned) and write each into the core's
//   activation RAM through its act_* port. The records are one contiguous run:
//   group g of batch element b at act_addr + 256*(b*act_groups + g) (register map
//   version 3; act_batch = 1 is version 2: record g at act_addr + 256*g), so the
//   element (= bank) advances every act_groups records, wherever the 4 KB bursts
//   fall. One record = 16 beats:
//       beats 0..9   the 1280-bit mantissa word, least significant beat first
//       beat  10     byte 0 = the group's exponent (int8, sign-extended to EXW)
//       beats 11..15 ignored
//   Bursts are whole 4 KB pages (256 beats = 16 records; the last one shorter),
//   ONE outstanding at a time, RREADY = 1 for ever. act_done pulses when the last
//   record has been committed.
//
//   RESULT WRITE (AW/W/B).  While res_en, every result row (res_v/res_d, in row
//   order, res_last on the final one) is packed four to a beat (row r = bytes
//   [4r, 4r+4) from res_addr, little endian; the last beat is zero padded) into a
//   small FIFO and written in INCR bursts of 2^WB_LG beats (the last one shorter)
//   from res_addr (4 KB aligned), so no burst crosses 4 KB. STORE AND FORWARD: AW
//   is only issued for beats that are already in the FIFO, so a burst never waits
//   for the engine. One burst outstanding at a time, BREADY = 1 for ever.
//   (Version 3: the wrapper hands the BATCH elements of a row over as BATCH
//   consecutive res_v words, element 0 first, res_last on the final element of
//   the final row -- so "row" here means "word" and word n lands at res_addr +
//   4*n, i.e. row r / element b at res_addr + 4*(r*BATCH + b).)
//   res_room is the back-pressure for the row source (registered; it drops when
//   fewer than 4 beats are free, which covers the rows already in flight).
//   res_done = every row of the run has been written AND acknowledged.
//
// TIMING. Only flops on the bus pins: R and B are registered before any use,
// ARREADY / AWREADY / WREADY each feed one valid flop and one event flop. The
// price is one idle cycle per W beat (0.5 beat/cycle = 2 rows/cycle, more than
// the engine can produce) -- this path carries a few MB per token. The 1280-bit
// mantissa word is assembled one stage after the FSM (r2_d / sh_en, a registered
// and replicated enable), and the commit to the core is delayed to match.
//
// flush (soft reset / the wrapper's pre-run flush): the fetch stops committing
// and the FIFO is emptied, but AXI state is never dropped: a pending xVALID is
// held, a read burst in flight is received and discarded, a write burst that
// has been addressed is completed with WSTRB = 0 (nothing lands in DDR) and its
// B awaited. `idle` is low until all of that is over.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
`default_nettype none

module axi_aux_dma #(
    parameter integer AW    = 49,
    parameter integer IDW   = 6,
    parameter integer EXW   = 9,
    parameter integer GMAX  = 96,
    parameter integer WF_LG = 6,                 // log2 result FIFO depth, beats (>= WB_LG + 2)
    parameter integer WB_LG = 4                  // log2 beats per write burst (<= 8)
) (
    input  wire                  clk,
    input  wire                  rst,            // hard reset (also resets the AXI state)
    input  wire                  flush,
    // activation fetch
    input  wire                  act_go,         // 1-cycle pulse, only while idle
    input  wire [7:0]            act_groups,     // records per batch element (0: act_done at once)
    input  wire [2:0]            act_batch,      // batch elements to fetch, 1..4 (static from GO on)
    input  wire [AW-1:0]         act_addr,
    output reg                   act_done,       // 1-cycle pulse
    output reg                   act_we,         // to the core's activation write port
    output reg  [1:0]            act_bank,       // batch element the record belongs to
    output reg  [7:0]            act_group,
    output reg  [1279:0]         act_mant,
    output reg  signed [EXW-1:0] act_exp,
    // result write
    input  wire                  res_en,         // level: this run writes its rows to DDR
    input  wire                  res_start,      // 1-cycle pulse at the start of the run (latches res_addr)
    input  wire [AW-1:0]         res_addr,
    input  wire                  res_v,
    input  wire [31:0]           res_d,
    input  wire                  res_last,
    output reg                   res_room,
    output reg                   res_done,
    // status
    output reg                   idle,           // no transaction pending or draining
    output reg                   err_evt,        // RRESP / BRESP != OKAY (pulse)
    // AXI4 master
    output wire [IDW-1:0]        awid,
    output reg  [AW-1:0]         awaddr,
    output reg  [7:0]            awlen,
    output wire [2:0]            awsize,
    output wire [1:0]            awburst,
    output wire                  awlock,
    output wire [3:0]            awcache,
    output wire [2:0]            awprot,
    output wire [3:0]            awqos,
    output reg                   awvalid,
    input  wire                  awready,
    output reg  [127:0]          wdata,
    output reg  [15:0]           wstrb,
    output reg                   wlast,
    output reg                   wvalid,
    input  wire                  wready,
    input  wire [IDW-1:0]        bid,
    input  wire [1:0]            bresp,
    input  wire                  bvalid,
    output wire                  bready,
    output wire [IDW-1:0]        arid,
    output reg  [AW-1:0]         araddr,
    output reg  [7:0]            arlen,
    output wire [2:0]            arsize,
    output wire [1:0]            arburst,
    output wire                  arlock,
    output wire [3:0]            arcache,
    output wire [2:0]            arprot,
    output wire [3:0]            arqos,
    output reg                   arvalid,
    input  wire                  arready,
    input  wire [IDW-1:0]        rid,
    input  wire [127:0]          rdata,
    input  wire [1:0]            rresp,
    input  wire                  rlast,
    input  wire                  rvalid,
    output wire                  rready
);
    localparam integer WFD = 1 << WF_LG;
    localparam integer WBL = 1 << WB_LG;

    assign awid = {IDW{1'b0}};  assign arid = {IDW{1'b0}};
    assign awsize = 3'b100;     assign arsize = 3'b100;          // 16 bytes
    assign awburst = 2'b01;     assign arburst = 2'b01;          // INCR
    assign awlock = 1'b0;       assign arlock = 1'b0;
    assign awcache = 4'b0011;   assign arcache = 4'b0011;
    assign awprot = 3'b000;     assign arprot = 3'b000;
    assign awqos = 4'b0000;     assign arqos = 4'b0000;
    assign rready = 1'b1;
    assign bready = 1'b1;

    reg rst_q, fl_q;
    always @(posedge clk) begin
        rst_q <= rst;
        fl_q  <= rst | flush;
    end

    // ------------------------------------------------------------ bus input registers
    reg         r1_v, r1_last, b1_v;
    reg [1:0]   r1_resp, b1_resp;
    reg [127:0] r1_d;
    always @(posedge clk) begin
        r1_v <= rvalid; r1_last <= rlast; r1_resp <= rresp; r1_d <= rdata;
        b1_v <= bvalid; b1_resp <= bresp;
    end

    // ============================================================ activation fetch
    localparam [1:0] F_IDLE = 2'd0, F_SETUP = 2'd1, F_BURST = 2'd2, F_FIN = 2'd3;
    reg [1:0]    fs;
    reg [9:0]    recs_left;                      // records not yet received
    reg [9:0]    recs_tot;                       // act_batch * act_groups, registered ahead of act_go
    reg [7:0]    g_last;                         // act_groups - 1: the last record of a batch element
    reg [1:0]    bk, bank_p;                     // batch element of the record being received
    reg [4:0]    brecs;                          // records in the burst in flight (1..16)
    reg [AW-1:0] f_addr;
    reg          f_drop;                         // flushed: finish the bus transaction, commit nothing
    reg [3:0]    bcnt;                           // beat within the 16-beat record
    reg [7:0]    gcnt;                           // record = group index
    reg [1:0]    fin_cnt;
    reg          f_err;
    reg          we_p;                           // commit, one stage ahead of act_we
    reg [7:0]    grp_p;
    reg signed [EXW-1:0] exp_p;
    reg [127:0]  r2_d;
    (* max_fanout = 160 *) reg sh_en;            // shift r2_d into the mantissa word
    wire signed [7:0] e8 = r1_d[7:0];
    wire         r1_fire = r1_v & (fs == F_BURST);

    always @(posedge clk) begin
        if (rst_q) begin
            fs <= F_IDLE; arvalid <= 1'b0; f_drop <= 1'b0; act_done <= 1'b0; act_we <= 1'b0; f_err <= 1'b0;
            we_p <= 1'b0; sh_en <= 1'b0;
        end else begin
            arvalid  <= arvalid & ~arready;
            act_done <= 1'b0;
            we_p     <= 1'b0;
            act_we   <= we_p;                    // by now beats 0..9 are all in act_mant
            sh_en    <= r1_fire & ~f_drop & (bcnt < 4'd10);
            f_err    <= r1_fire & ~f_drop & (r1_resp != 2'b00);
            if (fl_q && fs != F_IDLE) f_drop <= 1'b1;
            case (fs)
                F_IDLE: begin
                    f_drop <= 1'b0;
                    if (act_go && !fl_q) begin
                        recs_left <= recs_tot; f_addr <= act_addr; gcnt <= 8'd0; bcnt <= 4'd0; bk <= 2'd0;
                        fs <= (recs_tot == 10'd0) ? F_FIN : F_SETUP;
                        fin_cnt <= 2'd0;
                    end
                end
                F_SETUP: begin
                    if (f_drop || fl_q) fs <= F_IDLE;
                    else begin
                        arvalid <= 1'b1;
                        araddr  <= f_addr;
                        arlen   <= (recs_left >= 10'd16) ? 8'hFF : {recs_left[3:0] - 4'd1, 4'hF};
                        brecs   <= (recs_left >= 10'd16) ? 5'd16 : {1'b0, recs_left[3:0]};
                        fs      <= F_BURST;
                    end
                end
                F_BURST: begin
                    if (r1_v) begin
                        bcnt <= bcnt + 4'd1;
                        if (!f_drop && bcnt == 4'd10) begin
                            we_p   <= (gcnt < GMAX);
                            grp_p  <= gcnt;
                            bank_p <= bk;
                            exp_p  <= e8;
                            if (gcnt == g_last) begin gcnt <= 8'd0; bk <= bk + 2'd1; end     // next batch element
                            else                gcnt <= gcnt + 8'd1;
                        end
                        if (r1_last) begin
                            recs_left <= recs_left - {5'd0, brecs};
                            f_addr    <= f_addr + 4096;
                            fin_cnt   <= 2'd0;
                            if (f_drop || fl_q)                    fs <= F_IDLE;
                            else if (recs_left == {5'd0, brecs})   fs <= F_FIN;
                            else                                   fs <= F_SETUP;
                        end
                    end
                end
                default: begin                   // F_FIN: let the last commit land in the core
                    fin_cnt <= fin_cnt + 2'd1;
                    if (f_drop || fl_q) fs <= F_IDLE;
                    else if (fin_cnt == 2'd2) begin act_done <= 1'b1; fs <= F_IDLE; end
                end
            endcase
        end
    end

    always @(posedge clk) begin
        r2_d <= r1_d;
        if (sh_en) act_mant <= {r2_d, act_mant[1279:128]};       // least significant beat first
        act_group <= grp_p;
        act_bank  <= bank_p;
        act_exp   <= exp_p;
        // act_groups / act_batch are the wrapper's snapshot of GO: stable long before act_go
        case (act_batch)
            3'd2:    recs_tot <= {1'b0, act_groups, 1'b0};
            3'd3:    recs_tot <= {2'b00, act_groups} + {1'b0, act_groups, 1'b0};
            3'd4:    recs_tot <= {act_groups, 2'b00};
            default: recs_tot <= {2'b00, act_groups};
        endcase
        g_last <= act_groups - 8'd1;
    end

    // ============================================================ result write
    // ---- pack 4 rows per beat
    reg [1:0]   widx;
    reg [127:0] pk;
    reg         push_r, push_last;
    // ---- FIFO (distributed RAM, read straight into the WDATA register)
    (* ram_style = "distributed" *) reg [127:0] wmem [0:WFD-1];
    reg [WF_LG-1:0] wp, rp;
    reg [WF_LG:0]   fill;
    reg             all_in;                      // the run's last beat has been pushed
    // ---- write FSM
    localparam [1:0] W_IDLE = 2'd0, W_AW = 2'd1, W_DATA = 2'd2, W_B = 2'd3;
    reg [1:0]      ws;
    reg [AW-1:0]   w_addr;
    reg [WB_LG:0]  blen, left;
    reg            wabort, need_load, aw_fire_r, w_fire_r, w_err;
    wire           burst_ok = (fill >= WBL) | (all_in & (fill != 0));
    wire           do_load  = (ws == W_DATA) & (need_load | (w_fire_r & (left != 0)));
    wire           pop      = do_load & ~wabort;
    wire           push_ok  = push_r & ~fill[WF_LG];

    always @(posedge clk) begin
        if (push_ok) wmem[wp] <= pk;
        if (do_load) begin
            wdata <= wabort ? 128'd0 : wmem[rp];
            wstrb <= wabort ? 16'h0000 : 16'hFFFF;
            wlast <= (left == 1);
        end
    end

    always @(posedge clk) begin
        // packer + FIFO bookkeeping
        if (fl_q) begin
            widx <= 2'd0; push_r <= 1'b0; push_last <= 1'b0;
            wp <= {WF_LG{1'b0}}; rp <= {WF_LG{1'b0}}; fill <= {(WF_LG+1){1'b0}}; all_in <= 1'b0;
            res_room <= 1'b1; res_done <= 1'b0;
        end else begin
            push_r <= 1'b0;
            if (res_v && res_en) begin
                case (widx)
                    2'd0: pk <= {96'd0, res_d};
                    2'd1: pk[63:32]  <= res_d;
                    2'd2: pk[95:64]  <= res_d;
                    default: pk[127:96] <= res_d;
                endcase
                widx <= res_last ? 2'd0 : widx + 2'd1;
                if (widx == 2'd3 || res_last) begin push_r <= 1'b1; push_last <= res_last; end
            end
            if (push_ok) wp <= wp + 1'b1;
            if (pop)     rp <= rp + 1'b1;
            fill <= fill + (push_ok ? 1'b1 : 1'b0) - (pop ? 1'b1 : 1'b0);
            if (push_r && push_last) all_in <= 1'b1;
            if (res_start) all_in <= 1'b0;
            res_room <= ~res_en | (fill <= WFD - 4);
            res_done <= all_in & (fill == 0) & (ws == W_IDLE) & ~push_r;
        end
        if (res_start && ws == W_IDLE) w_addr <= res_addr;

        // AW / W / B
        if (rst_q) begin
            ws <= W_IDLE; awvalid <= 1'b0; wvalid <= 1'b0; wabort <= 1'b0; need_load <= 1'b0;
            aw_fire_r <= 1'b0; w_fire_r <= 1'b0; w_err <= 1'b0;
        end else begin
            awvalid   <= awvalid & ~awready;
            aw_fire_r <= awvalid & awready;
            wvalid    <= do_load | (wvalid & ~wready);
            w_fire_r  <= wvalid & wready;
            w_err     <= 1'b0;
            if (fl_q && ws != W_IDLE) wabort <= 1'b1;
            case (ws)
                W_IDLE: begin
                    wabort <= 1'b0;
                    if (burst_ok && !fl_q) begin
                        awvalid <= 1'b1;
                        awaddr  <= w_addr;
                        blen    <= (fill >= WBL) ? WBL : fill[WB_LG:0];
                        left    <= (fill >= WBL) ? WBL : fill[WB_LG:0];
                        awlen   <= ((fill >= WBL) ? WBL : fill[WB_LG:0]) - 1'b1;
                        ws      <= W_AW;
                    end
                end
                W_AW: if (aw_fire_r) begin ws <= W_DATA; need_load <= 1'b1; end
                W_DATA: begin
                    if (do_load) begin need_load <= 1'b0; left <= left - 1'b1; end
                    if (w_fire_r && left == 0) ws <= W_B;
                end
                default: if (b1_v) begin         // W_B
                    w_err  <= ~wabort & (b1_resp != 2'b00);
                    w_addr <= w_addr + {blen, 4'b0000};
                    ws     <= W_IDLE;
                end
            endcase
        end

        err_evt <= ~rst_q & (f_err | w_err);
        if (rst_q) idle <= 1'b0;
        else       idle <= (fs == F_IDLE) & (ws == W_IDLE) & ~arvalid & ~awvalid & ~wvalid & ~f_drop & ~wabort
                           & ~fl_q & ~act_go;
    end

    wire unused_id = ^{rid, bid};
endmodule

`default_nettype wire
