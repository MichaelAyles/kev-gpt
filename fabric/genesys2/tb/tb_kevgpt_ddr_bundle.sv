`timescale 1ns / 1ps
// Integration gate for the top-level DDR arbiter wiring (PORT-NOTES.md
// "Phase 2 architecture"): kv_bank_ddr's read+write and weight_loader_ddr's
// read, merged by kevgpt_ddr_bundle.sv into one app-level bundle, plugged
// into mig_dual_master_arbiter's side A -- side B is a synthetic driver
// standing in for cpu_ddr_bridge (that module itself isn't modified by this
// port, so it isn't re-gated here; this test only proves the WIRING and
// genuine two-master sharing, not cpu_ddr_bridge's own already-established
// correctness).
//
// Four phases:
//   1. kv_bank_ddr write+read through the FULL bundle+dual-arbiter stack
//      (side B idle) -- crosschecked against kv_bank.sv, same as
//      tb_kv_bank_ddr.sv but now through the real arbiter chain, not a
//      private engine pair.
//   2. weight_loader_ddr load through the FULL stack (side B idle) --
//      crosschecked against the source DDR image, same as
//      tb_weight_loader_ddr.sv but through the real arbiter chain.
//   3. A synthetic side-B write+readback through the dual-arbiter (side A
//      idle) -- proves side B's own data path is wired correctly (not
//      swapped with side A).
//   4. kv_bank_ddr write + a synthetic side-B write+read, launched
//      concurrently (fork/join) -- real two-master sharing, not just two
//      sequential single-master phases.
//
// Compile alongside (invoke directly with iverilog; bracket -DSYNTHESIS
// around ONLY the mig_read_engine/mig_dual_master_arbiter/sync_fifo files
// via undef_synth.sv/define_synth.sv shims, since kv_bank.sv and
// weight_bank_tdp.sv must NOT see SYNTHESIS defined):
//   fabric/stage3/rtl/kv_bank.sv
//   fabric/stage3/rtl/weight_bank_tdp.sv
//   fabric/genesys2/rtl/kv_bank_ddr.sv
//   fabric/genesys2/rtl/weight_loader_ddr.sv
//   fabric/genesys2/rtl/mig_read_mux2.sv
//   fabric/genesys2/rtl/kevgpt_ddr_bundle.sv
//   fabric/genesys2/tb/mig_behav_model.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_write_engine.sv
//   <undef_synth.sv shim>
//   <ai_accel>/rtl/accelerator/streamer/mig_read_engine.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_rw_arbiter.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_dual_master_arbiter.sv
//   <ai_accel>/rtl/accelerator/common/sync_fifo.sv
//   <define_synth.sv shim -- NOT needed again after, but harmless if present>
module tb_kevgpt_ddr_bundle;
  localparam integer P        = 8;
  localparam integer HEAD_DIM = 64;
  localparam integer NHEAD    = 2;
  localparam integer NLAYER   = 2;
  localparam integer TMAX     = 128;
  localparam integer KBITS    = 8;
  localparam integer INV_SH   = 24;
  localparam integer ADDR_W   = 29;
  localparam integer DATA_W   = 256;
  localparam integer HR       = HEAD_DIM / P;

  localparam integer LANES  = 64;
  localparam integer WWORDS = 64;
  localparam integer WBITS  = LANES * 4;   // = 256 = DATA_W at LANES=64
  localparam integer SUBW   = WBITS / 32;  // 8

  // TWO independent clock domains, not one -- kevgpt_ddr_bundle now does its
  // own CDC (async_fifo_gray) between kv_bank_ddr's gen_clk domain and MIG's
  // ui_clk domain (see that module's header for why: MIG's app port is a
  // hard ui_clk-synchronous requirement, kv_bank_ddr lives in kevgpt's own
  // compute-clock hierarchy, and these are genuinely different frequencies
  // on real hardware -- clk_gen=50MHz, MIG ui_clk~200MHz). Deliberately a
  // non-integer-multiple period ratio here (10ns vs 7ns) to actually stress
  // the crossing rather than accidentally look synchronous.
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;
  reg ui_clk = 0, ui_rst = 1;
  always #3.5 ui_clk = ~ui_clk;

  // =====================================================================
  // kv_bank (reference) + kv_bank_ddr (DUT)
  // =====================================================================
  reg        wq_start;
  reg [3:0]  wq_layer;
  reg        wq_kv;
  reg [1:0]  wq_head;
  reg [8:0]  wq_pos;
  reg        wq_valid;
  reg [P*32-1:0] wq_data;
  wire wq_done_ref, wq_done_ddr;

  reg         rd_start;
  reg  [3:0]  rd_layer;
  reg         rd_kv;
  reg  [1:0]  rd_head;
  reg  [8:0]  rd_tcount;
  wire        rd_valid, rd_done;
  wire [HEAD_DIM*32-1:0] rd_data;

  kv_bank #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER),
            .TMAX(TMAX), .KBITS(KBITS), .INV_SH(INV_SH)) u_ref (
      .clk(clk), .rst(rst),
      .wq_start(wq_start), .wq_layer(wq_layer), .wq_kv(wq_kv), .wq_head(wq_head),
      .wq_pos(wq_pos), .wq_valid(wq_valid), .wq_data(wq_data), .wq_done(wq_done_ref),
      .rd_start(rd_start), .rd_layer(rd_layer), .rd_kv(rd_kv), .rd_head(rd_head),
      .rd_tcount(rd_tcount), .rd_valid(rd_valid), .rd_data(rd_data), .rd_done(rd_done),
      .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head(2'd0),
      .rd2_tcount(9'd0), .rd2_valid(), .rd2_data(), .rd2_done()
  );

  wire                 kv_wr_pkt_valid, kv_wr_pkt_ready;
  wire [ADDR_W-1:0]    kv_wr_pkt_addr;
  wire [DATA_W-1:0]    kv_wr_pkt_data;
  wire [DATA_W/8-1:0]  kv_wr_pkt_mask;
  wire                 kv_wr_ack_valid, kv_wr_ack_ready;

  wire         rd_start_ddr;
  reg   [3:0]  rd_layer_ddr;
  reg          rd_kv_ddr;
  reg   [1:0]  rd_head_ddr;
  reg   [8:0]  rd_tcount_ddr;
  wire         rd_valid_ddr, rd_done_ddr;
  wire [HEAD_DIM*32-1:0] rd_data_ddr;

  wire                 kv_rd_req_valid, kv_rd_req_ready;
  wire [ADDR_W-1:0]    kv_rd_req_addr;
  wire                 kv_rd_ret_valid, kv_rd_ret_ready;
  wire [DATA_W-1:0]    kv_rd_ret_data;

  reg rd_start_ddr_r;
  assign rd_start_ddr = rd_start_ddr_r;

  kv_bank_ddr #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER),
                .TMAX(TMAX), .KBITS(KBITS), .INV_SH(INV_SH),
                .ADDR_W(ADDR_W), .DATA_W(DATA_W), .KV_DDR_BASE(0)) u_kv_dut (
      .clk(clk), .rst(rst),
      .wq_start(wq_start), .wq_layer(wq_layer), .wq_kv(wq_kv), .wq_head(wq_head),
      .wq_pos(wq_pos), .wq_valid(wq_valid), .wq_data(wq_data), .wq_done(wq_done_ddr),
      .wr_pkt_valid(kv_wr_pkt_valid), .wr_pkt_ready(kv_wr_pkt_ready),
      .wr_pkt_addr(kv_wr_pkt_addr), .wr_pkt_data(kv_wr_pkt_data), .wr_pkt_mask(kv_wr_pkt_mask),
      .wr_ack_valid(kv_wr_ack_valid), .wr_ack_ready(kv_wr_ack_ready),
      .rd_start(rd_start_ddr), .rd_layer(rd_layer_ddr), .rd_kv(rd_kv_ddr),
      .rd_head(rd_head_ddr), .rd_tcount(rd_tcount_ddr),
      .rd_valid(rd_valid_ddr), .rd_data(rd_data_ddr), .rd_done(rd_done_ddr),
      .rd_req_valid(kv_rd_req_valid), .rd_req_ready(kv_rd_req_ready), .rd_req_addr(kv_rd_req_addr),
      .rd_ret_valid(kv_rd_ret_valid), .rd_ret_ready(kv_rd_ret_ready), .rd_ret_data(kv_rd_ret_data)
  );

  // =====================================================================
  // weight_bank_tdp (resident, unmodified) + weight_loader_ddr (DUT)
  // =====================================================================
  reg [$clog2(WWORDS)-1:0] wb_rd_addr_r;
  wire [$clog2(WWORDS)-1:0] wb_raddr_b;
  wire [WBITS-1:0]          wb_rword_b;
  assign wb_raddr_b = wb_rd_addr_r;

  wire wb_ld_rst, wb_w_we;
  wire [31:0] wb_w_data;

  // clk (== gen_clk) here, not ui_clk: weight_bank_tdp/weight_loader_ddr live
  // inside sequencer_vec.sv's gen_clk-domain hierarchy on real hardware, and
  // kevgpt_ddr_bundle.sv now does a REAL CDC for wl_rd_req/wl_rd_ret
  // (u_wl_rd_req_cdc/u_wl_rd_ret_cdc, async_fifo_gray) -- so this side must
  // actually run on gen_clk for that CDC's rd_clk_i/rd_ready_i domain
  // assumptions to hold. (This corrects an earlier version of this
  // testbench, from before that CDC existed, that ran these two DUTs on
  // ui_clk as a simplification -- seen in the ldn_cnt comment below. Left
  // on ui_clk after the CDC was added, weight_loader_ddr's rd_ret_ready
  // became an ASYNCHRONOUS signal to u_wl_rd_ret_cdc's own rd_clk_i, with no
  // synchronizer of its own -- caused rd_ready_i to be sampled at ui_clk's
  // faster, unrelated rate instead of the FIFO's own gen_clk rate, which
  // intermittently double-popped a single logical beat (confirmed via a
  // direct wr/rd accept count on u_wl_rd_ret_cdc's own ports: 10 real
  // writes vs 14 observed reads, a 4-pop excess exactly equal to
  // CDC_FIFO_DEPTH -- the classic signature of this class of bug, not a
  // pointer-math defect in async_fifo_gray.sv itself, which this incident
  // re-confirms is correct once both of its ports are driven from their
  // OWN true clock domains). This was a testbench-only wiring bug -- real
  // hardware never had this mismatch, since weight_loader_ddr's clk port is
  // always gen_clk there.
  weight_bank_tdp #(.LANES(LANES), .WWORDS(WWORDS), .DP(0), .MEM_PRIMITIVE("block")) u_wb (
      .clk(clk), .clk2x(clk),
      .ld_rst(wb_ld_rst), .w_we(wb_w_we), .w_data(wb_w_data),
      .raddr_b(wb_raddr_b), .rword_b(wb_rword_b), .rword1_b(),
      .raddr_a({$clog2(WWORDS){1'b0}}), .rword_a(), .rword1_a()
  );

  reg               ld_start;
  reg  [ADDR_W-1:0] ld_ddr_addr;
  reg  [31:0]       ld_words;
  wire              ld_done;

  wire                  wl_rd_req_valid, wl_rd_req_ready;
  wire [ADDR_W-1:0]     wl_rd_req_addr;
  wire                  wl_rd_ret_valid, wl_rd_ret_ready;
  wire [DATA_W-1:0]     wl_rd_ret_data;

  // clk (== gen_clk) here, not ui_clk -- see u_wb's instantiation above.
  weight_loader_ddr #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_wl_dut (
      .clk(clk), .rst(rst),
      .ld_start(ld_start), .ld_ddr_addr(ld_ddr_addr), .ld_words(ld_words), .ld_done(ld_done),
      .wb_ld_rst(wb_ld_rst), .wb_w_we(wb_w_we), .wb_w_data(wb_w_data),
      .rd_req_valid(wl_rd_req_valid), .rd_req_ready(wl_rd_req_ready), .rd_req_addr(wl_rd_req_addr),
      .rd_ret_valid(wl_rd_ret_valid), .rd_ret_ready(wl_rd_ret_ready), .rd_ret_data(wl_rd_ret_data)
  );

  // =====================================================================
  // kevgpt_ddr_bundle (side A) + synthetic side B + mig_dual_master_arbiter
  // + mig_behav_model (the "real MIG")
  // =====================================================================
  wire [ADDR_W-1:0]   a_app_addr;
  wire [2:0]          a_app_cmd;
  wire                 a_app_en, a_app_rdy;
  wire [DATA_W-1:0]    a_app_wdf_data;
  wire [DATA_W/8-1:0]  a_app_wdf_mask;
  wire                 a_app_wdf_wren, a_app_wdf_rdy;
  wire [DATA_W-1:0]    a_app_rd_data;
  wire                 a_app_rd_data_valid;

  kevgpt_ddr_bundle #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_bundle (
      .gen_clk(clk), .gen_rst(rst),
      .ui_clk(ui_clk), .ui_rst(ui_rst),
      .kv_wr_pkt_valid(kv_wr_pkt_valid), .kv_wr_pkt_ready(kv_wr_pkt_ready),
      .kv_wr_pkt_addr(kv_wr_pkt_addr), .kv_wr_pkt_data(kv_wr_pkt_data), .kv_wr_pkt_mask(kv_wr_pkt_mask),
      .kv_wr_ack_valid(kv_wr_ack_valid), .kv_wr_ack_ready(kv_wr_ack_ready),
      .kv_rd_req_valid(kv_rd_req_valid), .kv_rd_req_ready(kv_rd_req_ready), .kv_rd_req_addr(kv_rd_req_addr),
      .kv_rd_ret_valid(kv_rd_ret_valid), .kv_rd_ret_ready(kv_rd_ret_ready), .kv_rd_ret_data(kv_rd_ret_data),
      .wl_rd_req_valid(wl_rd_req_valid), .wl_rd_req_ready(wl_rd_req_ready), .wl_rd_req_addr(wl_rd_req_addr),
      .wl_rd_ret_valid(wl_rd_ret_valid), .wl_rd_ret_ready(wl_rd_ret_ready), .wl_rd_ret_data(wl_rd_ret_data),
      .app_addr(a_app_addr), .app_cmd(a_app_cmd), .app_en(a_app_en), .app_rdy(a_app_rdy),
      .app_wdf_data(a_app_wdf_data), .app_wdf_mask(a_app_wdf_mask),
      .app_wdf_wren(a_app_wdf_wren), .app_wdf_end(), .app_wdf_rdy(a_app_wdf_rdy),
      .app_rd_data(a_app_rd_data), .app_rd_data_valid(a_app_rd_data_valid)
  );

  // ---- synthetic side B: a plain register-driven app-level requester,
  // standing in for cpu_ddr_bridge (not re-gated here -- see header) --------
  localparam [2:0] MIG_CMD_WRITE = 3'b000;
  localparam [2:0] MIG_CMD_READ  = 3'b001;

  reg [ADDR_W-1:0]   b_app_addr_r;
  reg [2:0]          b_app_cmd_r;
  reg                 b_app_en_r;
  wire                b_app_rdy;
  reg [DATA_W-1:0]    b_app_wdf_data_r;
  reg [DATA_W/8-1:0]  b_app_wdf_mask_r;
  reg                 b_app_wdf_wren_r;
  wire                b_app_wdf_rdy;
  wire [DATA_W-1:0]   b_app_rd_data;
  wire                b_app_rd_data_valid;

  wire [ADDR_W-1:0]   phys_app_addr;
  wire [2:0]          phys_app_cmd;
  wire                 phys_app_en, phys_app_rdy;
  wire [DATA_W-1:0]    phys_app_wdf_data;
  wire [DATA_W/8-1:0]  phys_app_wdf_mask;
  wire                 phys_app_wdf_wren, phys_app_wdf_end, phys_app_wdf_rdy;
  wire [DATA_W-1:0]    phys_app_rd_data;
  wire                 phys_app_rd_data_valid;

  mig_dual_master_arbiter #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .BATCH_LIMIT(16)) u_dual (
      .clk_i(ui_clk), .rst_ni(!ui_rst),
      .a_app_addr_i(a_app_addr), .a_app_cmd_i(a_app_cmd), .a_app_en_i(a_app_en), .a_app_rdy_o(a_app_rdy),
      .a_app_wdf_data_i(a_app_wdf_data), .a_app_wdf_mask_i(a_app_wdf_mask),
      .a_app_wdf_wren_i(a_app_wdf_wren), .a_app_wdf_rdy_o(a_app_wdf_rdy),
      .a_app_rd_data_o(a_app_rd_data), .a_app_rd_data_valid_o(a_app_rd_data_valid),
      .b_app_addr_i(b_app_addr_r), .b_app_cmd_i(b_app_cmd_r), .b_app_en_i(b_app_en_r), .b_app_rdy_o(b_app_rdy),
      .b_app_wdf_data_i(b_app_wdf_data_r), .b_app_wdf_mask_i(b_app_wdf_mask_r),
      .b_app_wdf_wren_i(b_app_wdf_wren_r), .b_app_wdf_rdy_o(b_app_wdf_rdy),
      .b_app_rd_data_o(b_app_rd_data), .b_app_rd_data_valid_o(b_app_rd_data_valid),
      .app_addr_o(phys_app_addr), .app_cmd_o(phys_app_cmd), .app_en_o(phys_app_en), .app_rdy_i(phys_app_rdy),
      .app_wdf_data_o(phys_app_wdf_data), .app_wdf_mask_o(phys_app_wdf_mask),
      .app_wdf_wren_o(phys_app_wdf_wren), .app_wdf_end_o(phys_app_wdf_end), .app_wdf_rdy_i(phys_app_wdf_rdy),
      .app_rd_data_i(phys_app_rd_data), .app_rd_data_valid_i(phys_app_rd_data_valid)
  );

  mig_behav_model #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(8192), .READ_LATENCY(7)) u_mem (
      .clk_i(ui_clk), .rst_ni(!ui_rst),
      .app_addr_i(phys_app_addr), .app_cmd_i(phys_app_cmd), .app_en_i(phys_app_en), .app_rdy_o(phys_app_rdy),
      .app_wdf_data_i(phys_app_wdf_data), .app_wdf_mask_i(phys_app_wdf_mask),
      .app_wdf_wren_i(phys_app_wdf_wren), .app_wdf_end_i(phys_app_wdf_end), .app_wdf_rdy_o(phys_app_wdf_rdy),
      .app_rd_data_o(phys_app_rd_data), .app_rd_data_valid_o(phys_app_rd_data_valid), .app_rd_data_end_o()
  );

  // =====================================================================
  // stimulus + checking
  // =====================================================================
  integer errors, tc, li, ci, wi;
  reg signed [31:0] vec [0:HEAD_DIM-1];
  reg [3:0] t_layer [0:1];
  reg       t_kv    [0:1];
  reg [1:0] t_head  [0:1];
  reg [8:0] t_pos   [0:1];

  reg [8:0] rd_valid_count, rd_valid_count_ddr;
  reg [HEAD_DIM*32-1:0] rd_data_last, rd_data_last_ddr;
  always @(posedge clk) begin
    if (rst || rd_start) rd_valid_count <= 9'd0;
    else if (rd_valid) rd_valid_count <= rd_valid_count + 9'd1;
  end
  always @(posedge clk) if (rd_valid) rd_data_last <= rd_data;
  always @(posedge clk) begin
    if (rst || rd_start_ddr) rd_valid_count_ddr <= 9'd0;
    else if (rd_valid_ddr) rd_valid_count_ddr <= rd_valid_count_ddr + 9'd1;
  end
  always @(posedge clk) if (rd_valid_ddr) rd_data_last_ddr <= rd_data_ddr;

  reg [3:0] wqr_cnt, wqd_cnt;
  always @(posedge clk) begin
    if (rst) wqr_cnt <= 4'd0; else if (wq_done_ref) wqr_cnt <= wqr_cnt + 4'd1;
  end
  always @(posedge clk) begin
    if (rst) wqd_cnt <= 4'd0; else if (wq_done_ddr) wqd_cnt <= wqd_cnt + 4'd1;
  end
  // weight_loader_ddr/weight_bank_tdp run on clk (== gen_clk) -- see their
  // instantiations above -- matching real deployment (both live inside
  // sequencer_vec.sv's gen_clk-domain hierarchy) now that
  // kevgpt_ddr_bundle.sv does a real CDC for wl_rd_req/wl_rd_ret.
  reg [3:0] ldn_cnt;
  always @(posedge clk) begin
    if (rst) ldn_cnt <= 4'd0; else if (ld_done) ldn_cnt <= ldn_cnt + 4'd1;
  end

  task automatic do_kv_write(input [3:0] layer, input kv, input [1:0] head, input [8:0] pos);
    integer b;
    begin
      @(posedge clk);
      wq_layer <= layer; wq_kv <= kv; wq_head <= head; wq_pos <= pos;
      wq_start <= 1'b1;
      @(posedge clk);
      wq_start <= 1'b0;
      for (b = 0; b < HR; b = b + 1) begin
        wq_data <= {vec[b*P+7], vec[b*P+6], vec[b*P+5], vec[b*P+4],
                    vec[b*P+3], vec[b*P+2], vec[b*P+1], vec[b*P+0]};
        wq_valid <= 1'b1;
        @(posedge clk);
      end
      wq_valid <= 1'b0;
    end
  endtask

  reg [31:0] lfsr;
  function automatic [31:0] next_lfsr(input [31:0] x);
    next_lfsr = {x[30:0], x[31] ^ x[21] ^ x[1] ^ x[0]};
  endfunction

  // Synthetic side-B: write one beat at a chosen DDR word index, then read
  // it back and compare, entirely at the raw app-level protocol.
  reg [DATA_W-1:0] b_want, b_got;
  reg [3:0] b_done_track;
  task automatic side_b_write_read(input integer word_idx, input [DATA_W-1:0] pattern);
    begin
      // write
      @(posedge ui_clk);
      b_app_addr_r     <= word_idx * (DATA_W/8);
      b_app_cmd_r      <= MIG_CMD_WRITE;
      b_app_en_r       <= 1'b1;
      b_app_wdf_data_r <= pattern;
      b_app_wdf_mask_r <= {(DATA_W/8){1'b0}};
      b_app_wdf_wren_r <= 1'b1;
      @(posedge ui_clk);
      while (!(b_app_rdy && b_app_en_r)) @(posedge ui_clk);
      b_app_en_r <= 1'b0;
      while (!(b_app_wdf_rdy && b_app_wdf_wren_r)) @(posedge ui_clk);
      b_app_wdf_wren_r <= 1'b0;
      // read back
      @(posedge ui_clk);
      b_app_addr_r <= word_idx * (DATA_W/8);
      b_app_cmd_r  <= MIG_CMD_READ;
      b_app_en_r   <= 1'b1;
      @(posedge ui_clk);
      while (!(b_app_rdy && b_app_en_r)) @(posedge ui_clk);
      b_app_en_r <= 1'b0;
      while (!b_app_rd_data_valid) @(posedge ui_clk);
      b_got = b_app_rd_data;
      if (b_got !== pattern) begin
        $display("SIDE_B_MISMATCH,word=%0d,got=%h,want=%h", word_idx, b_got, pattern);
        errors = errors + 1;
      end
    end
  endtask

  reg [WBITS-1:0] wl_want_word;

  initial begin
    errors = 0;
    wq_start = 0; wq_layer = 0; wq_kv = 0; wq_head = 0; wq_pos = 0; wq_valid = 0; wq_data = 0;
    rd_start = 0; rd_layer = 0; rd_kv = 0; rd_head = 0; rd_tcount = 0;
    rd_start_ddr_r = 0; rd_layer_ddr = 0; rd_kv_ddr = 0; rd_head_ddr = 0; rd_tcount_ddr = 0;
    wb_rd_addr_r = 0;
    ld_start = 0; ld_ddr_addr = 0; ld_words = 0;
    b_app_addr_r = 0; b_app_cmd_r = 0; b_app_en_r = 0;
    b_app_wdf_data_r = 0; b_app_wdf_mask_r = {(DATA_W/8){1'b1}}; b_app_wdf_wren_r = 0;

    t_layer[0] = 4'd1; t_kv[0] = 1'b0; t_head[0] = 2'd1; t_pos[0] = 9'd3;
    t_layer[1] = 4'd0; t_kv[1] = 1'b1; t_head[1] = 2'd0; t_pos[1] = 9'd5;

    rst = 1;
    ui_rst = 1;
    repeat (4) @(posedge clk);
    rst = 0;
    repeat (4) @(posedge ui_clk);
    ui_rst = 0;
    @(posedge clk);

    // ==================== Phase 1: kv_bank_ddr through the full stack ======
    tc = 0;
    for (li = 0; li < HEAD_DIM; li = li + 1)
      vec[li] = 32'sd1000 + li * 7 - 32'sd200;
    do_kv_write(t_layer[tc], t_kv[tc], t_head[tc], t_pos[tc]);
    wait (wqr_cnt == tc + 1);
    wait (wqd_cnt == tc + 1);
    @(posedge clk);

    rd_layer <= t_layer[tc]; rd_kv <= t_kv[tc]; rd_head <= t_head[tc];
    rd_tcount <= t_pos[tc] + 9'd1;
    @(posedge clk);
    rd_start <= 1'b1; @(posedge clk); rd_start <= 1'b0;
    wait (rd_valid_count == t_pos[tc] + 9'd1);
    @(posedge clk);

    rd_layer_ddr <= t_layer[tc]; rd_kv_ddr <= t_kv[tc]; rd_head_ddr <= t_head[tc];
    rd_tcount_ddr <= t_pos[tc] + 9'd1;
    @(posedge clk);
    rd_start_ddr_r <= 1'b1; @(posedge clk); rd_start_ddr_r <= 1'b0;
    wait (rd_valid_count_ddr == t_pos[tc] + 9'd1);
    @(posedge clk);

    for (li = 0; li < HEAD_DIM; li = li + 1) begin
      if (rd_data_last_ddr[li*32 +: 32] !== rd_data_last[li*32 +: 32]) begin
        $display("PHASE1_KV_MISMATCH,lane=%0d,got=%0d,want=%0d", li,
                  $signed(rd_data_last_ddr[li*32 +: 32]), $signed(rd_data_last[li*32 +: 32]));
        errors = errors + 1;
      end
    end
    $display("PHASE1_KV_DONE,errors_so_far=%0d", errors);

    // ==================== Phase 2: weight_loader_ddr through the full stack
    lfsr = 32'hCAFEF00D;
    for (wi = 0; wi < 16; wi = wi + 1) begin
      for (ci = 0; ci < SUBW; ci = ci + 1) begin
        lfsr = next_lfsr(lfsr);
        u_mem.mem[200 + wi][ci*32 +: 32] = lfsr;
      end
    end
    ld_ddr_addr <= 200 * (DATA_W/8);
    ld_words    <= 16 * SUBW;
    @(posedge clk);
    ld_start <= 1'b1; @(posedge clk); ld_start <= 1'b0;
    wait (ldn_cnt == 1);
    @(posedge clk);

    for (wi = 0; wi < 16; wi = wi + 1) begin
      wb_rd_addr_r <= wi[$clog2(WWORDS)-1:0];
      @(posedge clk);
      @(posedge clk);
      wl_want_word = u_mem.mem[200 + wi];
      if (wb_rword_b !== wl_want_word) begin
        $display("PHASE2_WL_MISMATCH,word=%0d,got=%h,want=%h", wi, wb_rword_b, wl_want_word);
        errors = errors + 1;
      end
    end
    $display("PHASE2_WL_DONE,errors_so_far=%0d", errors);

    // ==================== Phase 3: synthetic side B alone ==================
    side_b_write_read(500, {8{32'hA5A5A5A5}});
    $display("PHASE3_SIDEB_DONE,errors_so_far=%0d", errors);

    // ==================== Phase 4: kv_bank_ddr write + side-B write/read,
    // launched concurrently -- real two-master sharing. =====================
    tc = 1;
    for (li = 0; li < HEAD_DIM; li = li + 1)
      vec[li] = (li % 2 == 0) ? -(li * 311) : (li * 271);

    fork
      do_kv_write(t_layer[tc], t_kv[tc], t_head[tc], t_pos[tc]);
      side_b_write_read(600, {8{32'h5A5A5A5A}});
    join

    wait (wqr_cnt == tc + 1);
    wait (wqd_cnt == tc + 1);
    @(posedge clk);

    rd_layer <= t_layer[tc]; rd_kv <= t_kv[tc]; rd_head <= t_head[tc];
    rd_tcount <= t_pos[tc] + 9'd1;
    @(posedge clk);
    rd_start <= 1'b1; @(posedge clk); rd_start <= 1'b0;
    wait (rd_valid_count == t_pos[tc] + 9'd1);
    @(posedge clk);

    rd_layer_ddr <= t_layer[tc]; rd_kv_ddr <= t_kv[tc]; rd_head_ddr <= t_head[tc];
    rd_tcount_ddr <= t_pos[tc] + 9'd1;
    @(posedge clk);
    rd_start_ddr_r <= 1'b1; @(posedge clk); rd_start_ddr_r <= 1'b0;
    wait (rd_valid_count_ddr == t_pos[tc] + 9'd1);
    @(posedge clk);

    for (li = 0; li < HEAD_DIM; li = li + 1) begin
      if (rd_data_last_ddr[li*32 +: 32] !== rd_data_last[li*32 +: 32]) begin
        $display("PHASE4_KV_MISMATCH,lane=%0d,got=%0d,want=%0d", li,
                  $signed(rd_data_last_ddr[li*32 +: 32]), $signed(rd_data_last[li*32 +: 32]));
        errors = errors + 1;
      end
    end
    $display("PHASE4_DONE,errors_so_far=%0d", errors);

    if (errors == 0) $display("KEVGPT_DDR_BUNDLE_VERDICT,PASS");
    else $display("KEVGPT_DDR_BUNDLE_VERDICT,FAIL,errors=%0d", errors);
    $finish;
  end

  initial begin
    #400000;
    $display("KEVGPT_DDR_BUNDLE_VERDICT,TIMEOUT");
    $finish;
  end
endmodule
