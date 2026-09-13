// Full-sequencer gate for the DDR-backed weight-window loader
// (WEIGHT_DDR_BACKED=1): a byte-for-byte copy of fabric/stage3/tb/
// tb_seq_vec_kv.sv's own stimulus (prompt/gen GO-pulse loop, cycle
// counting), EXCEPT the entire weight image is loaded through
// weight_loader_ddr.sv's DMA path instead of the firmware wl_we register
// stream: the image is staged into a behavioral DDR model (mig_behav_model,
// poked directly the same way tb_weight_loader_ddr.sv's own isolated gate
// does), then one wld_ld_start pulse streams it through a real
// mig_read_engine into the SAME unmodified weight_bank_tdp.sv the resident
// design uses. This proves sequencer_vec's new WEIGHT_DDR_BACKED=1 wiring
// (the ld_rst/w_we/w_data OR-mux onto gemv_banked_resident_vec's existing
// boot-load port) produces the SAME bit-exact token stream as the firmware-
// streamed path, under REALISTIC full-sequencer multi-token, multi-layer
// traffic -- not just weight_bank_tdp's own isolated readback that
// tb_weight_loader_ddr.sv already covers.
//
// LANES=64 (not run_vec_kv.py's usual LVAL=16 default) is chosen
// deliberately: WBITS=LANES*4=256=DATA_W, so one weight_bank_tdp row is
// exactly one 256-bit DMA beat -- ld_words = WROMN*8 is trivially a
// multiple of weight_loader_ddr's own SUBW=8, and wrom.mem's own words can
// be staged into mig_behav_model's mem[] one-for-one, no repacking
// arithmetic of its own to get wrong here (matches tb_weight_loader_ddr.sv's
// own reasoning). This is also the ACTUAL deployed Option A LANES value
// (xilinx_core_v_mini_mcu_wrapper_kevgpt.sv), not an arbitrary test-only
// choice.
`timescale 1ns / 1ps
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef WROMN
 `define WROMN 199936
`endif
`ifndef LVAL
 `define LVAL 64
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

    // ---- DDR-backed weight-window loader: weight_loader_ddr's DMA read
    // port, wired to a real mig_read_engine + mig_behav_model stack (same
    // pattern tb_weight_loader_ddr.sv's own isolated gate uses -- no write
    // engine/arbiter needed, weight_loader_ddr never writes). ---------------
    reg          wld_ld_start_r;
    reg  [28:0]  wld_ld_ddr_addr_r;
    reg  [31:0]  wld_ld_words_r;
    wire         wld_ld_done;
    wire         wl_rd_req_valid, wl_rd_req_ready;
    wire [28:0]  wl_rd_req_addr;
    wire         wl_rd_ret_valid, wl_rd_ret_ready;
    wire [255:0] wl_rd_ret_data;

    sequencer_vec #(.P(P), .LANES(LANES), .TMAX(TMAXP), .D(DP), .D3(3*DP),
                     .D_MLP(4*DP), .NLAYER(NLAYERP), .NHEAD(NHEADP), .VOCAB(VOCABP),
                     .WEIGHT_DDR_BACKED(1)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .tok_out(tok_out), .rd_sel(rsel), .rd_addr(raddr), .rd_data(rdata),
        .wl_rst(wl_rst), .wl_we(wl_we), .wl_data(wl_data), .dbg_stop(dbgstop_r),
        .seed(seed_r), .seed_we(seed_we_r),
        .wld_ld_start(wld_ld_start_r), .wld_ld_ddr_addr(wld_ld_ddr_addr_r),
        .wld_ld_words(wld_ld_words_r), .wld_ld_done(wld_ld_done),
        .wl_rd_req_valid(wl_rd_req_valid), .wl_rd_req_ready(wl_rd_req_ready), .wl_rd_req_addr(wl_rd_req_addr),
        .wl_rd_ret_valid(wl_rd_ret_valid), .wl_rd_ret_ready(wl_rd_ret_ready), .wl_rd_ret_data(wl_rd_ret_data));

    wire              rd_cmd_valid;
    wire [ADDR_W-1:0] rd_cmd_addr;
    wire              app_rdy;
    wire [DATA_W-1:0] app_rd_data;
    wire              app_rd_data_valid;

    mig_read_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .RETURN_DEPTH(32),
                       .MAX_OUTSTANDING(16), .SAFETY_MARGIN(4)) u_rd_engine (
        .clk_i(clk), .rst_ni(!rst),
        .req_valid_i(wl_rd_req_valid), .req_ready_o(wl_rd_req_ready), .req_addr_i(wl_rd_req_addr),
        .cmd_valid_o(rd_cmd_valid), .cmd_grant_i(app_rdy), .cmd_addr_o(rd_cmd_addr),
        .app_rd_data_i(app_rd_data), .app_rd_data_valid_i(app_rd_data_valid),
        .ret_valid_o(wl_rd_ret_valid), .ret_ready_i(wl_rd_ret_ready), .ret_data_o(wl_rd_ret_data),
        .outstanding_o(), .credit_stall_cycles_o(), .overflow_error_o()
    );

    localparam [2:0] MIG_CMD_READ = 3'b001;

    // Backing store must cover WROMN beats (1 beat/row at LANES=64) -- sized
    // generously above the largest wrom_n this project has ever produced.
    mig_behav_model #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(262144), .READ_LATENCY(9)) u_mem (
        .clk_i(clk), .rst_ni(!rst),
        .app_addr_i(rd_cmd_addr), .app_cmd_i(MIG_CMD_READ), .app_en_i(rd_cmd_valid), .app_rdy_o(app_rdy),
        .app_wdf_data_i({DATA_W{1'b0}}), .app_wdf_mask_i({(DATA_W/8){1'b1}}),
        .app_wdf_wren_i(1'b0), .app_wdf_end_i(1'b0), .app_wdf_rdy_o(),
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
    reg [8:0] prompt [0:PLEN-1];
    reg [8:0] stream [0:PLEN+NGEN-1];
    integer i, s, f, fs, fc, cyc0, pi;
    integer dbgcyc = 0;

    initial begin
        rst = 1'b1; go = 1'b0; tok = 9'd0; pos = 9'd0; rsel = 0; raddr = 0;
        wl_rst = 1'b0; wl_we = 1'b0; wl_data = 32'b0;
        seed_r = 32'b0; seed_we_r = 1'b0;
        wld_ld_start_r = 1'b0; wld_ld_ddr_addr_r = 29'd0; wld_ld_words_r = 32'd0;
        $readmemh("wrom.mem", wimg);
        $readmemh("prompt.mem", prompt);
        for (i = 0; i < PLEN; i = i + 1) stream[i] = prompt[i];
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;

        // ---- stage the weight image into DDR (one WBITS-wide row == one
        // 256-bit beat at LANES=64), then load it ALL through weight_loader_ddr
        // instead of firmware's wl_we stream. ------------------------------------
        for (i = 0; i < `WROMN; i = i + 1) u_mem.mem[i] = wimg[i];
        wld_ld_ddr_addr_r = 29'd0;
        wld_ld_words_r    = `WROMN * SUBW;
        @(posedge clk); #1;
        wld_ld_start_r = 1'b1; @(posedge clk); #1; wld_ld_start_r = 1'b0;
        begin : wait_wld_done
            reg [7:0] done_cnt;
            done_cnt = 0;
            while (done_cnt == 0) begin
                @(posedge clk);
                if (wld_ld_done) done_cnt = done_cnt + 1;
            end
        end
        @(posedge clk); #1;

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
        if (running) stcnt[dut.st] = stcnt[dut.st] + 1;
        if (dbgcyc % 500000 == 0)
            $display("[cyc %0d] st=%0d blk=%0d pos=%0d done=%b", dbgcyc, dut.st, dut.blk, pos, done);
    end
    initial begin #400000000; $display("TB_TIMEOUT cyc=%0d st=%0d blk=%0d", dbgcyc, dut.st, dut.blk); $finish; end
endmodule
