`timescale 1ns / 1ps
// Gate for weight_loader_ddr.sv: streams a known image out of a behavioral
// DDR model, through a real mig_read_engine, into an UNMODIFIED
// weight_bank_tdp.sv instance (the same module the on-chip-resident design
// already uses), then reads the resident bank back via its existing
// raddr_b/rword_b port and checks it matches the source DDR image exactly.
// At LANES=64, WBITS(=LANES*4)=256 == DATA_W, so one DMA beat maps to
// exactly one weight_bank_tdp word -- a direct beat-for-word comparison,
// no unpacking arithmetic of its own to get wrong in the testbench.
//
// Compile alongside (invoke directly with iverilog; no run_*.py harness yet):
//   fabric/stage3/rtl/weight_bank_tdp.sv
//   fabric/genesys2/rtl/weight_loader_ddr.sv
//   fabric/genesys2/tb/mig_behav_model.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_read_engine.sv
//   <ai_accel>/rtl/accelerator/common/sync_fifo.sv
// (mig_read_engine/sync_fifo's SVA needs -DSYNTHESIS to skip under Icarus --
// weight_bank_tdp.sv has its own `ifdef SYNTHESIS` XPM branch, so bracket the
// define with undef_synth.sv/define_synth.sv shims exactly like
// tb_kv_bank_ddr.sv does, not a blanket -DSYNTHESIS on the command line.)
module tb_weight_loader_ddr;
  localparam integer LANES  = 64;
  // Genesys2 mini-story-teller shape's real per-block window (D=128/D3=384/
  // D_MLP=512/LANES=64): GW_BLK = GW_QKV+GW_PROJ+GW_FC+GW_MP =
  // 768+256+1024+1024 = 3072 wide words -- the realistic per-layer-reload
  // size the per-layer weight-streaming feasibility check (PORT-NOTES.md)
  // needs a real cycle count for, not the 16/24-word correctness-only
  // cases below.
  localparam integer WWORDS = 3072;
  localparam integer ADDR_W = 29;
  localparam integer DATA_W = 256;
  localparam integer WBITS  = LANES * 4;   // = 256 = DATA_W at LANES=64
  localparam integer SUBW   = WBITS / 32;  // 8

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  // ---- weight_bank_tdp under test (block/BRAM primitive, Genesys2 shape) ----
  wire [$clog2(WWORDS)-1:0] raddr_b;
  wire [WBITS-1:0]          rword_b;
  wire                      wb_ld_rst;
  wire                      wb_w_we;
  wire [31:0]                wb_w_data;

  reg [$clog2(WWORDS)-1:0] rd_addr_r;
  assign raddr_b = rd_addr_r;

  weight_bank_tdp #(.LANES(LANES), .WWORDS(WWORDS), .DP(0), .MEM_PRIMITIVE("block")) u_wb (
      .clk(clk), .clk2x(clk),
      .ld_rst(wb_ld_rst), .w_we(wb_w_we), .w_data(wb_w_data),
      .raddr_b(raddr_b), .rword_b(rword_b), .rword1_b(),
      .raddr_a({$clog2(WWORDS){1'b0}}), .rword_a(), .rword1_a()
  );

  // ---- loader under test ----
  reg               ld_start;
  reg  [ADDR_W-1:0] ld_ddr_addr;
  reg  [31:0]       ld_words;
  wire              ld_done;

  wire                  rd_req_valid;
  wire                  rd_req_ready;
  wire [ADDR_W-1:0]     rd_req_addr;
  wire                  rd_ret_valid;
  wire                  rd_ret_ready;
  wire [DATA_W-1:0]     rd_ret_data;

  weight_loader_ddr #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_dut (
      .clk(clk), .rst(rst),
      .ld_start(ld_start), .ld_ddr_addr(ld_ddr_addr), .ld_words(ld_words), .ld_done(ld_done),
      .wb_ld_rst(wb_ld_rst), .wb_w_we(wb_w_we), .wb_w_data(wb_w_data),
      .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready), .rd_req_addr(rd_req_addr),
      .rd_ret_valid(rd_ret_valid), .rd_ret_ready(rd_ret_ready), .rd_ret_data(rd_ret_data)
  );

  // ---- read engine + behavioral MIG, single reader (no arbiter needed) ----
  wire              rd_cmd_valid;
  wire [ADDR_W-1:0] rd_cmd_addr;
  wire              app_rdy;
  wire [DATA_W-1:0] app_rd_data;
  wire              app_rd_data_valid;

  mig_read_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .RETURN_DEPTH(32),
                     .MAX_OUTSTANDING(16), .SAFETY_MARGIN(4)) u_rd_engine (
      .clk_i(clk), .rst_ni(!rst),
      .req_valid_i(rd_req_valid), .req_ready_o(rd_req_ready), .req_addr_i(rd_req_addr),
      .cmd_valid_o(rd_cmd_valid), .cmd_grant_i(app_rdy), .cmd_addr_o(rd_cmd_addr),
      .app_rd_data_i(app_rd_data), .app_rd_data_valid_i(app_rd_data_valid),
      .ret_valid_o(rd_ret_valid), .ret_ready_i(rd_ret_ready), .ret_data_o(rd_ret_data),
      .outstanding_o(), .credit_stall_cycles_o(), .overflow_error_o()
  );

  localparam [2:0] MIG_CMD_READ = 3'b001;

  // MEM_WORDS covers the third (realistic-block-size) test case's own
  // offset + word count -- see base_beat[2]/nwords[2] below.
  mig_behav_model #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(8192), .READ_LATENCY(9)) u_mem (
      .clk_i(clk), .rst_ni(!rst),
      .app_addr_i(rd_cmd_addr), .app_cmd_i(MIG_CMD_READ), .app_en_i(rd_cmd_valid), .app_rdy_o(app_rdy),
      .app_wdf_data_i({DATA_W{1'b0}}), .app_wdf_mask_i({(DATA_W/8){1'b1}}),
      .app_wdf_wren_i(1'b0), .app_wdf_end_i(1'b0), .app_wdf_rdy_o(),
      .app_rd_data_o(app_rd_data), .app_rd_data_valid_o(app_rd_data_valid), .app_rd_data_end_o()
  );

  // ---- source image: fill DDR with a known pseudo-random pattern before
  // each load (mirrors how firmware's boot-time staging, Phase 3, will have
  // already populated this region) -- three cases: two small ones with
  // DIFFERENT base beat offset AND word count each (the KV-cache gate found
  // real bugs -- address truncation, off-by-ones -- that a single small/
  // zero-offset case did not exercise), plus a THIRD sized to GW_BLK=3072
  // (the mini-story-teller shape's real per-block window) purely to get a
  // real ld_start-to-ld_done cycle count for the per-layer weight-streaming
  // feasibility question (PORT-NOTES.md) -- correctness-checked the same
  // way as the other two, not just timed.
  integer wi, ci, tc;
  integer base_beat [0:2];
  integer nwords    [0:2];
  reg [31:0] lfsr;
  function automatic [31:0] next_lfsr(input [31:0] x);
    next_lfsr = {x[30:0], x[31] ^ x[21] ^ x[1] ^ x[0]};
  endfunction

  integer errors;
  reg [WBITS-1:0] want_word;
  integer cyc_count;

  initial begin
    ld_start = 0; ld_ddr_addr = 0; ld_words = 0; rd_addr_r = 0;
    errors = 0;
    base_beat[0] = 0;    nwords[0] = 16;
    base_beat[1] = 100;  nwords[1] = 24;
    base_beat[2] = 200;  nwords[2] = 3072;

    rst = 1;
    repeat (4) @(posedge clk);
    rst = 0;
    @(posedge clk);

    for (tc = 0; tc < 3; tc = tc + 1) begin
      // seed DDR starting at base_beat[tc] with nwords[tc]*SUBW sequential
      // 32-bit LFSR values, packed low-chunk-first into 256-bit beats -- the
      // SAME chunk order weight_bank_tdp's own assembler expects (w_data
      // chunk k lands at bits [k*32 +: 32] of wbuf).
      lfsr = 32'hDEADBEEF ^ (tc * 32'h13579BDF);
      for (wi = 0; wi < nwords[tc]; wi = wi + 1) begin
        for (ci = 0; ci < SUBW; ci = ci + 1) begin
          lfsr = next_lfsr(lfsr);
          u_mem.mem[base_beat[tc] + wi][ci*32 +: 32] = lfsr;
        end
      end

      // ---- trigger the load ----
      ld_ddr_addr <= base_beat[tc] * (DATA_W/8);
      ld_words    <= nwords[tc] * SUBW;
      @(posedge clk);
      ld_start <= 1'b1;
      @(posedge clk);
      ld_start <= 1'b0;

      // wait for ld_done via a monotonic counter (see PORT-NOTES.md /
      // tb_kv_bank_ddr.sv -- plain wait()/@(posedge ld_done) on this kind of
      // single-cycle pulse is unreliable under Icarus). cyc_count measures
      // ld_start-to-ld_done latency -- a simulation lower bound (this
      // behavioral MIG's fixed READ_LATENCY(9) is optimistic vs. real
      // hardware's full CDC/arbitration path) but a real, non-guessed
      // number for the per-layer weight-streaming feasibility question.
      begin : wait_done
        reg [3:0] done_cnt;
        done_cnt = 0;
        cyc_count = 0;
        while (done_cnt == 0) begin
          @(posedge clk);
          cyc_count = cyc_count + 1;
          if (ld_done) done_cnt = done_cnt + 1;
        end
      end
      $display("WEIGHT_LOADER_DDR_CYCLES,tc=%0d,words=%0d,cycles=%0d", tc, nwords[tc], cyc_count);
      @(posedge clk);

      // ---- readback: compare weight_bank_tdp's resident words against the
      // source DDR image directly (1 DMA beat == 1 weight word at LANES=64) --
      for (wi = 0; wi < nwords[tc]; wi = wi + 1) begin
        rd_addr_r <= wi[$clog2(WWORDS)-1:0];
        @(posedge clk);  // weight_bank_tdp's rd_b<=mem[raddr_b] samples here
        @(posedge clk);  // one extra cycle of settle margin (not strictly needed)
        want_word = u_mem.mem[base_beat[tc] + wi];
        if (rword_b !== want_word) begin
          $display("WEIGHT_LOADER_DDR_MISMATCH,tc=%0d,word=%0d,got=%h,want=%h", tc, wi, rword_b, want_word);
          errors = errors + 1;
        end
      end
      $display("WEIGHT_LOADER_DDR_CASE,tc=%0d,checked_words=%0d,errors_so_far=%0d", tc, nwords[tc], errors);
    end

    if (errors == 0) $display("WEIGHT_LOADER_DDR_VERDICT,PASS");
    else $display("WEIGHT_LOADER_DDR_VERDICT,FAIL,errors=%0d", errors);
    $finish;
  end

  initial begin
    // Bumped from 200000 for the new 3072-word case's own readback loop
    // alone (~6144 cycles = 61440ns at 10ns/cycle) on top of the reload
    // itself and the two small cases -- generous margin, not tuned tight.
    #2000000;
    $display("WEIGHT_LOADER_DDR_VERDICT,TIMEOUT");
    $finish;
  end
endmodule
