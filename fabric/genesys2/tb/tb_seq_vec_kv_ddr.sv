// Full-sequencer gate for the DDR-backed KV cache (KV_DDR_BACKED=1): a
// byte-for-byte copy of fabric/stage3/tb/tb_seq_vec_kv.sv's own stimulus
// (weight stream, prompt/gen GO-pulse loop, cycle counting, KVDBG hooks) --
// ONLY the sequencer_vec instantiation differs (KV_DDR_BACKED(1) plus the
// new kv_wr_*/kv_rd_* DMA ports, wired to a real mig_write_engine +
// mig_read_engine + mig_behav_model stack, not a shortcut). This proves the
// generate-selected kv_bank_ddr path inside sequencer_vec.sv produces the
// SAME bit-exact token stream as the resident kv_bank.sv path already does
// (fabric.stage3.run_vec_kv's own gate), under REALISTIC multi-token,
// multi-layer, multi-head traffic -- not just the two isolated write+read
// pairs tb_kv_bank_ddr.sv already covers.
`timescale 1ns / 1ps
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef WROMN
 `define WROMN 199936
`endif
`ifndef LVAL
 `define LVAL 16
`endif
`ifndef TMAXVAL
 `define TMAXVAL 256
`endif
`ifndef PLEN
 `define PLEN 4
`endif
`ifndef NGEN
 `define NGEN 6
`endif
`ifndef SEEDVAL
 `define SEEDVAL 0
`endif
`ifndef DVAL
 `define DVAL 256
`endif
`ifndef NLAYERVAL
 `define NLAYERVAL 4
`endif
`ifndef NHEADVAL
 `define NHEADVAL 4
`endif
`ifndef VOCABVAL
 `define VOCABVAL 193
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = `LVAL;
    localparam integer TMAXP = `TMAXVAL;
    localparam integer DP      = `DVAL;
    localparam integer NLAYERP = `NLAYERVAL;
    localparam integer NHEADP  = `NHEADVAL;
    localparam integer VOCABP  = `VOCABVAL;
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;
    localparam integer PLEN  = `PLEN;
    localparam integer NGEN  = `NGEN;
    localparam integer NPASS = PLEN + NGEN - 1;

    localparam integer ADDR_W = 29;
    localparam integer DATA_W = 256;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [8:0] tok, pos;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire [8:0] tok_out;
    wire signed [63:0] rdata;
    reg wl_rst, wl_we; reg [31:0] wl_data;
    reg [31:0] seed_r; reg seed_we_r;

`ifndef KVSTOP
 `define KVSTOP 0
`endif
    reg [1:0] dbgstop_r = 2'b0;

    // ---- DDR-backed KV cache: kv_bank_ddr's DMA ports, wired to a real
    // mig_write_engine + mig_read_engine + mig_behav_model stack (same
    // pattern as tb_kv_bank_ddr.sv, not a shortcut). -------------------------
    wire                 kv_wr_pkt_valid, kv_wr_pkt_ready;
    wire [ADDR_W-1:0]    kv_wr_pkt_addr;
    wire [DATA_W-1:0]    kv_wr_pkt_data;
    wire [DATA_W/8-1:0]  kv_wr_pkt_mask;
    wire                 kv_wr_ack_valid, kv_wr_ack_ready;
    wire                 kv_rd_req_valid, kv_rd_req_ready;
    wire [ADDR_W-1:0]    kv_rd_req_addr;
    wire                 kv_rd_ret_valid, kv_rd_ret_ready;
    wire [DATA_W-1:0]    kv_rd_ret_data;

    sequencer_vec #(.P(P), .LANES(LANES), .TMAX(TMAXP), .D(DP), .D3(3*DP),
                     .D_MLP(4*DP), .NLAYER(NLAYERP), .NHEAD(NHEADP), .VOCAB(VOCABP),
                     .KV_DDR_BACKED(1)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .tok_out(tok_out), .rd_sel(rsel), .rd_addr(raddr), .rd_data(rdata),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data), .dbg_stop(dbgstop_r),
        .seed(seed_r), .seed_we(seed_we_r),
        .kv_wr_pkt_valid(kv_wr_pkt_valid), .kv_wr_pkt_ready(kv_wr_pkt_ready),
        .kv_wr_pkt_addr(kv_wr_pkt_addr), .kv_wr_pkt_data(kv_wr_pkt_data), .kv_wr_pkt_mask(kv_wr_pkt_mask),
        .kv_wr_ack_valid(kv_wr_ack_valid), .kv_wr_ack_ready(kv_wr_ack_ready),
        .kv_rd_req_valid(kv_rd_req_valid), .kv_rd_req_ready(kv_rd_req_ready), .kv_rd_req_addr(kv_rd_req_addr),
        .kv_rd_ret_valid(kv_rd_ret_valid), .kv_rd_ret_ready(kv_rd_ret_ready), .kv_rd_ret_data(kv_rd_ret_data));

    wire              wr_cmd_valid, wr_cmd_grant;
    wire [ADDR_W-1:0] wr_cmd_addr;
    wire              wr_wdf_valid;
    wire [DATA_W-1:0] wr_wdf_data;
    wire [DATA_W/8-1:0] wr_wdf_mask;

    mig_write_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_wr_engine (
        .clk_i(clk), .rst_ni(!rst),
        .pkt_valid_i(kv_wr_pkt_valid), .pkt_ready_o(kv_wr_pkt_ready),
        .pkt_addr_i(kv_wr_pkt_addr), .pkt_data_i(kv_wr_pkt_data), .pkt_mask_i(kv_wr_pkt_mask),
        .cmd_valid_o(wr_cmd_valid), .cmd_grant_i(wr_cmd_grant), .cmd_addr_o(wr_cmd_addr),
        .wdf_valid_o(wr_wdf_valid), .wdf_ready_i(app_wdf_rdy),
        .wdf_data_o(wr_wdf_data), .wdf_mask_o(wr_wdf_mask),
        .ack_valid_o(kv_wr_ack_valid), .ack_ready_i(kv_wr_ack_ready),
        .stall_cycles_o()
    );

    wire              rd_cmd_valid, rd_cmd_grant;
    wire [ADDR_W-1:0] rd_cmd_addr;
    wire [DATA_W-1:0] app_rd_data;
    wire              app_rd_data_valid;

    mig_read_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .RETURN_DEPTH(32),
                       .MAX_OUTSTANDING(16), .SAFETY_MARGIN(4)) u_rd_engine (
        .clk_i(clk), .rst_ni(!rst),
        .req_valid_i(kv_rd_req_valid), .req_ready_o(kv_rd_req_ready), .req_addr_i(kv_rd_req_addr),
        .cmd_valid_o(rd_cmd_valid), .cmd_grant_i(rd_cmd_grant), .cmd_addr_o(rd_cmd_addr),
        .app_rd_data_i(app_rd_data), .app_rd_data_valid_i(app_rd_data_valid),
        .ret_valid_o(kv_rd_ret_valid), .ret_ready_i(kv_rd_ret_ready), .ret_data_o(kv_rd_ret_data),
        .outstanding_o(), .credit_stall_cycles_o(), .overflow_error_o()
    );

    wire [ADDR_W-1:0]   app_addr;
    wire [2:0]          app_cmd;
    wire                 app_en, app_rdy, app_wdf_rdy;

    mig_rw_arbiter #(.ADDR_W(ADDR_W), .BATCH_LIMIT(16)) u_cmd_arb (
        .clk_i(clk), .rst_ni(!rst),
        .rd_valid_i(rd_cmd_valid), .rd_addr_i(rd_cmd_addr), .rd_grant_o(rd_cmd_grant),
        .wr_valid_i(wr_cmd_valid), .wr_addr_i(wr_cmd_addr), .wr_grant_o(wr_cmd_grant),
        .app_addr_o(app_addr), .app_cmd_o(app_cmd), .app_en_o(app_en), .app_rdy_i(app_rdy),
        .rd_grants_o(), .wr_grants_o(), .direction_switches_o()
    );

    // KV cache footprint at NLAYER=4/NHEAD=4/TMAX=256/HEAD_DIM=64/KBITS=8:
    // HROWS=4*2*4*256=8192, ROW_BEATS=3 (2 code + 1 hdr beat at HEAD_DIM=64),
    // so MEM_WORDS must cover 8192*3=24576 words minimum -- sized generously.
    mig_behav_model #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(32768), .READ_LATENCY(6)) u_mem (
        .clk_i(clk), .rst_ni(!rst),
        .app_addr_i(app_addr), .app_cmd_i(app_cmd), .app_en_i(app_en), .app_rdy_o(app_rdy),
        .app_wdf_data_i(wr_wdf_data), .app_wdf_mask_i(wr_wdf_mask),
        .app_wdf_wren_i(wr_wdf_valid), .app_wdf_end_i(wr_wdf_valid), .app_wdf_rdy_o(app_wdf_rdy),
        .app_rd_data_o(app_rd_data), .app_rd_data_valid_o(app_rd_data_valid), .app_rd_data_end_o()
    );

    task dump(input [3:0] sel, input integer n, input [127:0] fname);
        integer k, ff;
        begin
            ff = $fopen(fname, "w");
            for (k = 0; k < n; k = k + 1) begin
                rsel = sel; raddr = k[10:0];
                @(posedge clk); @(posedge clk); #1;
                $fwrite(ff, "%016x\n", rdata);
            end
            $fclose(ff);
        end
    endtask

    reg [WBITS-1:0] wimg [0:`WROMN-1];
    reg [WBITS-1:0] wword;
    reg [8:0] prompt [0:PLEN-1];
    reg [8:0] stream [0:PLEN+NGEN-1];
    integer i, s, f, fs, fc, cyc0, pi;
    integer dbgcyc = 0;

    initial begin
        rst = 1'b1; go = 1'b0; tok = 9'd0; pos = 9'd0; rsel = 0; raddr = 0;
        wl_rst = 1'b0; wl_we = 1'b0; wl_data = 32'b0;
        seed_r = 32'b0; seed_we_r = 1'b0;
        $readmemh("wrom.mem", wimg);
        $readmemh("prompt.mem", prompt);
        for (i = 0; i < PLEN; i = i + 1) stream[i] = prompt[i];
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        wl_rst = 1'b1; @(posedge clk); #1; wl_rst = 1'b0;
        for (i = 0; i < `WROMN; i = i + 1) begin
            wword = wimg[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                wl_we = 1'b1; wl_data = wword[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        wl_we = 1'b0; @(posedge clk); #1;

        fc = $fopen("cycs.out", "w");
        for (pi = 0; pi < NPASS; pi = pi + 1) begin
            if (pi == PLEN-1 && `SEEDVAL != 0) begin
                seed_r = `SEEDVAL; seed_we_r = 1'b1; @(posedge clk); #1; seed_we_r = 1'b0;
            end
            tok = stream[pi]; pos = pi[8:0];
            dbgstop_r = (pi == NPASS-1) ? `KVSTOP : 2'b0;
            cyc0 = dbgcyc;
            go = 1'b1; @(posedge clk); #1; go = 1'b0;
            wait (done == 1'b1); @(posedge clk); #1;
            $fwrite(fc, "%0d\n", dbgcyc - cyc0);
            if (pi + 1 >= PLEN) begin
                stream[pi+1] = tok_out;
                $display("GEN pos=%0d tok=%0d", pi, tok_out);
            end
        end
        $fclose(fc);
        if (`KVSTOP != 0) dump(4'd7, 256, "x.out");
        fs = $fopen("stream.out", "w");
        for (i = PLEN; i < PLEN + NGEN; i = i + 1) $fwrite(fs, "%0d\n", stream[i]);
        $fclose(fs);
        $display("TB_DONE");
        $finish;
    end

    integer stcnt [0:31];
    integer running = 0, k2, fprof;
    initial for (k2 = 0; k2 < 32; k2 = k2 + 1) stcnt[k2] = 0;
    always @(posedge clk) begin
        dbgcyc = dbgcyc + 1;
        if (go) running = 1;
        if (done) begin
            if (running) begin
                fprof = $fopen("profile.out", "w");
                for (k2 = 0; k2 < 32; k2 = k2 + 1)
                    if (stcnt[k2] > 0) $fwrite(fprof, "%0d %0d\n", k2, stcnt[k2]);
                $fclose(fprof);
            end
        end
        if (running) stcnt[dut.st] = stcnt[dut.st] + 1;
        if (dbgcyc % 500000 == 0)
            $display("[cyc %0d] st=%0d blk=%0d pos=%0d done=%b", dbgcyc, dut.st, dut.blk, pos, done);
    end
    initial begin #400000000; $display("TB_TIMEOUT cyc=%0d st=%0d blk=%0d", dbgcyc, dut.st, dut.blk); $finish; end
endmodule
