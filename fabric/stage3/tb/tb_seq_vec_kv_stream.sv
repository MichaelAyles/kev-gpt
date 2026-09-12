// Testbench for sequencer_vec's PER-LAYER WEIGHT STREAMING path
// (WEIGHT_STREAM_PER_LAYER=1, PORT-NOTES.md "per-layer weight streaming"):
// a variant of tb_seq_vec_kv.sv that stages the SAME wrom.mem content into
// a SIMULATED DDR3 image (via mig_behav_model, the same one tb_weight_
// loader_ddr.sv uses) instead of bulk-loading it into weight_bank_tdp via
// wl_we, then drives the SAME PLEN prompt + NGEN greedy feedback passes --
// this exercises weight_loader_ddr triggered INTERNALLY, mid-inference, by
// sequencer_vec's own S_STRW state, for the first time (every prior real
// use of weight_loader_ddr, including tb_weight_loader_ddr.sv's own gate,
// only ever triggered it externally, once). Gate: the generated token
// stream must be BIT-IDENTICAL to tb_seq_vec_kv.sv's own (same checkpoint,
// same prompt, same seed) -- proving streaming reproduces the fully-
// resident design's exact behavior, not a separate golden reference.
//
// Compile alongside (no run_*.py harness yet, matching tb_weight_loader_
// ddr.sv's own "no run_*.py harness yet" precedent -- -DSYNTHESIS bracketed
// around JUST mig_read_engine/sync_fifo via a `define SYNTHESIS shim, since
// weight_bank_tdp.sv itself must NOT see SYNTHESIS defined):
//   fabric/stage3/rtl/*.sv (the SAME RTL_FILES list run_vec_kv.py uses)
//   fabric/genesys2/rtl/weight_loader_ddr.sv
//   fabric/genesys2/tb/mig_behav_model.sv
//   <define_synth.sv shim>
//   <ai_accel>/rtl/accelerator/streamer/mig_read_engine.sv
//   <ai_accel>/rtl/accelerator/common/sync_fifo.sv
//   fabric/stage3/tb/tb_seq_vec_kv_stream.sv
// plus wrom.mem (and the other ROMs run_vec_kv.py's own run() already
// generates) in the run directory -- reuse, don't regenerate.
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
// WWORDSVAL: the shrunk, per-block resident-window size streaming actually
// needs (>= max(GW_BLK, GW_HEAD, GW_EMB) for the checkpoint under test --
// the whole point of this gate; NOT WROMN, which is the full image's own
// size and only sizes the SIMULATED DDR3 image below).
`ifndef WWORDSVAL
 `define WWORDSVAL 3072
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer LANES = `LVAL;
    localparam integer TMAXP = `TMAXVAL;
    localparam integer DP      = `DVAL;
    localparam integer NLAYERP = `NLAYERVAL;
    localparam integer NHEADP  = `NHEADVAL;
    localparam integer VOCABP  = `VOCABVAL;
    localparam integer WWORDSP = `WWORDSVAL;
    // tok/tok_out/prompt/stream width: was hardcoded 9 bits -- must track
    // sequencer_vec's own VIDXW (fabric/genesys2/PORT-NOTES.md "word-level
    // vocabulary").
    localparam integer VIDXWP = $clog2(VOCABP);
    localparam integer WBITS = LANES * 4;
    localparam integer SUBW  = WBITS / 32;
    localparam integer PLEN  = `PLEN;
    localparam integer NGEN  = `NGEN;
    localparam integer NPASS = PLEN + NGEN - 1;
    localparam integer ADDR_W = 29;
    localparam integer DATA_W = 256;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, go;
    reg [VIDXWP-1:0] tok;
    reg [8:0] pos;
    reg [3:0]  rsel;
    reg [10:0] raddr;
    wire done;
    wire [VIDXWP-1:0] tok_out;
    wire signed [63:0] rdata;
    reg [31:0] seed_r; reg seed_we_r;

    // ---- DMA fabric for the per-layer weight-stream trigger: a real
    // mig_read_engine + behavioral MIG, exactly like tb_weight_loader_
    // ddr.sv's own plumbing -- sequencer_vec's WEIGHT_DDR_BACKED=1 generate
    // block instantiates weight_loader_ddr INTERNALLY, so this testbench
    // only needs to supply the OUTER DDR3-side fabric it plugs into. -----
    wire        wl_rd_req_valid, wl_rd_req_ready;
    wire [ADDR_W-1:0] wl_rd_req_addr;
    wire        wl_rd_ret_valid, wl_rd_ret_ready;
    wire [DATA_W-1:0] wl_rd_ret_data;
    wire        wld_ld_done;
    // FIXATION-WORD-CDC-INVESTIGATION.md Sec8 item 1: same free-running
    // CRC32 real hardware exposes via KEVGPT_REG_WEIGHT_STREAM_CRC --
    // simulation's own value here, for a real captured seed, is the
    // "expected" value to compare real hardware's own report against.
    wire [31:0] weight_stream_crc;

    wire              rd_cmd_valid;
    wire [ADDR_W-1:0] rd_cmd_addr;
    wire              app_rdy;
    wire [DATA_W-1:0] app_rd_data;
    wire              app_rd_data_valid;
    localparam [2:0] MIG_CMD_READ = 3'b001;

    mig_read_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .RETURN_DEPTH(32),
                       .MAX_OUTSTANDING(16), .SAFETY_MARGIN(4)) u_rd_engine (
        .clk_i(clk), .rst_ni(!rst),
        .req_valid_i(wl_rd_req_valid), .req_ready_o(wl_rd_req_ready), .req_addr_i(wl_rd_req_addr),
        .cmd_valid_o(rd_cmd_valid), .cmd_grant_i(app_rdy), .cmd_addr_o(rd_cmd_addr),
        .app_rd_data_i(app_rd_data), .app_rd_data_valid_i(app_rd_data_valid),
        .ret_valid_o(wl_rd_ret_valid), .ret_ready_i(wl_rd_ret_ready), .ret_data_o(wl_rd_ret_data),
        .outstanding_o(), .credit_stall_cycles_o(), .overflow_error_o()
    );

    // MEM_WORDS covers the FULL wrom.mem image (all blocks + head + embed
    // tables) -- reuse WROMN, the same total this checkpoint's own
    // write_mems_wideword output already sizes to.
    mig_behav_model #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(`WROMN), .READ_LATENCY(9)) u_mem (
        .clk_i(clk), .rst_ni(!rst),
        .app_addr_i(rd_cmd_addr), .app_cmd_i(MIG_CMD_READ), .app_en_i(rd_cmd_valid), .app_rdy_o(app_rdy),
        .app_wdf_data_i({DATA_W{1'b0}}), .app_wdf_mask_i({(DATA_W/8){1'b1}}),
        .app_wdf_wren_i(1'b0), .app_wdf_end_i(1'b0), .app_wdf_rdy_o(),
        .app_rd_data_o(app_rd_data), .app_rd_data_valid_o(app_rd_data_valid), .app_rd_data_end_o()
    );

    // D3/D_MLP follow this codebase's fixed 3x/4x-of-D convention, matching
    // tb_seq_vec_kv.sv's own DUT instantiation exactly except for the new
    // WEIGHT_DDR_BACKED/WEIGHT_STREAM_PER_LAYER/WWORDS/WEIGHTS_DDR_BASE
    // parameters and the wl_rd_*/wld_ld_done port wiring those require.
    sequencer_vec #(.P(P), .LANES(LANES), .TMAX(TMAXP), .D(DP), .D3(3*DP),
                     .D_MLP(4*DP), .NLAYER(NLAYERP), .NHEAD(NHEADP), .VOCAB(VOCABP),
                     .WWORDS(WWORDSP), .WEIGHT_DDR_BACKED(1),
                     .WEIGHT_STREAM_PER_LAYER(1), .WEIGHTS_DDR_BASE(0)) dut (
        .clk(clk), .rst(rst), .go(go), .tok_id(tok), .pos(pos), .done(done),
        .tok_out(tok_out), .rd_sel(rsel), .rd_addr(raddr), .rd_data(rdata),
        .wl_rst(1'b0), .wl_we(1'b0), .wl_data(32'd0), .dbg_stop(2'b0),
        .seed(seed_r), .seed_we(seed_we_r),
        // KV cache stays resident (kv_bank.sv) -- unrelated to this gate,
        // KV_DDR_BACKED defaults to 0, these ports go unused same as
        // tb_seq_vec_kv.sv's own (unconnected) instantiation.
        .kv_wr_pkt_valid(), .kv_wr_pkt_ready(1'b0), .kv_wr_pkt_addr(),
        .kv_wr_pkt_data(), .kv_wr_pkt_mask(), .kv_wr_ack_valid(1'b0),
        .kv_wr_ack_ready(), .kv_rd_req_valid(), .kv_rd_req_ready(1'b0),
        .kv_rd_req_addr(), .kv_rd_ret_valid(1'b0), .kv_rd_ret_ready(),
        .kv_rd_ret_data(256'd0),
        // firmware-facing WLD_* ports unused under streaming (S_STRW drives
        // weight_loader_ddr internally instead) -- tied off.
        .wld_ld_start(1'b0), .wld_ld_ddr_addr(29'd0), .wld_ld_words(32'd0),
        .wld_ld_done(wld_ld_done),
        .wl_rd_req_valid(wl_rd_req_valid), .wl_rd_req_ready(wl_rd_req_ready),
        .wl_rd_req_addr(wl_rd_req_addr),
        .wl_rd_ret_valid(wl_rd_ret_valid), .wl_rd_ret_ready(wl_rd_ret_ready),
        .wl_rd_ret_data(wl_rd_ret_data),
        .weight_stream_crc(weight_stream_crc));

    reg [VIDXWP-1:0] prompt [0:PLEN-1];
    reg [VIDXWP-1:0] stream [0:PLEN+NGEN-1];
    integer i, fs, fc, cyc0, pi;
    integer dbgcyc = 0;

    initial begin
        rst = 1'b1; go = 1'b0; tok = 0; pos = 9'd0; rsel = 0; raddr = 0;
        seed_r = 32'b0; seed_we_r = 1'b0;
        // stage the FULL weight image into the simulated DDR3 directly --
        // WBITS(=LANES*4)=256=DATA_W at LANES=64, so one wrom.mem line is
        // exactly one DMA beat; no unpacking arithmetic of its own here,
        // same as tb_weight_loader_ddr.sv's own beat-for-word equivalence.
        $readmemh("wrom.mem", u_mem.mem);
        $readmemh("prompt.mem", prompt);
        for (i = 0; i < PLEN; i = i + 1) stream[i] = prompt[i];
        repeat (4) @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;

        fc = $fopen("cycs_stream.out", "w");
        for (pi = 0; pi < NPASS; pi = pi + 1) begin
            if (pi == PLEN-1 && `SEEDVAL != 0) begin
                seed_r = `SEEDVAL; seed_we_r = 1'b1; @(posedge clk); #1; seed_we_r = 1'b0;
            end
            tok = stream[pi]; pos = pi[8:0];
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
        fs = $fopen("stream_stream.out", "w");
        for (i = PLEN; i < PLEN + NGEN; i = i + 1) $fwrite(fs, "%0d\n", stream[i]);
        $fclose(fs);
        $display("WEIGHT_STREAM_CRC,0x%08x", weight_stream_crc);
        $display("TB_DONE");
        $finish;
    end

    always @(posedge clk) dbgcyc = dbgcyc + 1;
    initial begin #400000000; $display("TB_TIMEOUT cyc=%0d st=%0d blk=%0d", dbgcyc, dut.st, dut.blk); $finish; end
endmodule
