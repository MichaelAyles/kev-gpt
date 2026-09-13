`timescale 1ns / 1ps
// Compile with -DSEEDVAL=<n> to sweep seeds across runs (default below is
// just a fallback for a bare compile) -- Icarus's $urandom stream is
// otherwise identical run-to-run, so without this every invocation of this
// gate silently re-tests the exact same interleaving.
`ifndef SEEDVAL
`define SEEDVAL 32'hFEED0001
`endif
// Full multi-master contention gate (fabric/genesys2/FIXATION-WORD-CDC-
// INVESTIGATION.md Sec8 item 5): tb_kevgpt_ddr_bundle.sv proves the WIRING is
// correct under light, mostly-sequential traffic (its own header says so
// explicitly). This gate is the thing that document flags as the project's
// biggest missing regression -- KV + weight + a synthetic-CPU side B, all
// three CONTINUOUSLY and CONCURRENTLY active for many iterations (not phased
// one-at-a-time), through a RANDOMIZED-latency, randomized-backpressure MIG
// model (mig_behav_model_rand.sv, not the always-ready/fixed-latency
// mig_behav_model.sv every single-master gate uses), with a randomized
// gen_clk/ui_clk startup phase offset varied per simulation seed. If the
// owner-FIFO backpressure fix (Sec4/Sec8 item 2) or any owner-FIFO idiom in
// this DMA path is wrong under real contention -- misrouted/misordered
// returns, corrupted write-data pairing -- THIS is the gate built to expose
// it; the CDC timing-constraint investigation (Sec6) already ruled out static
// synchronizer margins on real hardware, so contention/ordering bugs are the
// next real candidate.
//
// Every transaction is checked against a golden reference at the moment it
// completes (kv_bank_ddr vs. kv_bank, weight_bank_tdp vs. the source DDR
// image, side B's own write-then-readback) -- not a final end-of-test dump,
// so a corruption is attributed to the exact iteration/generator that hit it.
//
// Toolchain note (see FIXATION-WORD-CDC-INVESTIGATION.md Sec4a): this local
// Icarus install cannot parse `assert property (... disable iff ...)`, so
// the owner-FIFO/backpressure assertions added in mig_dual_master_arbiter.sv
// and mig_read_mux2.sv are compiled out here via the same `SYNTHESIS`
// bracketing every other gate in this session used, NOT because they're
// unwanted -- this gate's own pass/fail verdict rests entirely on the
// functional data-correctness checks below, same as every gate before it.
//
// Compile alongside (manual iverilog invocation, no run_*.py wrapper yet):
//   fabric/stage3/rtl/kv_bank.sv
//   fabric/stage3/rtl/weight_bank_tdp.sv
//   fabric/genesys2/rtl/kv_bank_ddr.sv
//   fabric/genesys2/rtl/weight_loader_ddr.sv
//   <define_synth.sv shim>
//   fabric/genesys2/rtl/mig_read_mux2.sv
//   <undef_synth.sv shim>
//   fabric/genesys2/rtl/kevgpt_ddr_bundle.sv
//   fabric/genesys2/tb/mig_behav_model_rand.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_write_engine.sv
//   <define_synth.sv shim>
//   <ai_accel>/rtl/accelerator/streamer/mig_read_engine.sv
//   <undef_synth.sv shim>
//   <ai_accel>/rtl/accelerator/streamer/mig_rw_arbiter.sv
//   <define_synth.sv shim>
//   <ai_accel>/rtl/accelerator/streamer/mig_dual_master_arbiter.sv
//   <ai_accel>/rtl/accelerator/common/sync_fifo.sv
//   <undef_synth.sv shim>
//   fabric/genesys2/tb/tb_kevgpt_ddr_bundle_full.sv
// plus inv_lut_lo.mem/inv_lut_hi.mem in the run directory.
module tb_kevgpt_ddr_bundle_full;
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
  localparam integer WBITS  = LANES * 4;
  localparam integer SUBW   = WBITS / 32;

  // Non-overlapping DDR word regions for the three concurrent traffic
  // sources. KV needs ROW_BEATS(3)*NLAYER*2*NHEAD*TMAX = 3*2*2*2*128 = 3072
  // words at kv_bank_ddr's own KV_DDR_BASE=0; weight and side-B get their
  // own generously-separated bases so a bug in one path's addressing can't
  // masquerade as a false pass by accidentally aliasing another's region.
  localparam integer WEIGHT_BASE = 4096;
  localparam integer WEIGHT_SPAN = 2048;   // beats, i.e. words/SUBW... conservatively generous
  localparam integer SIDEB_BASE  = 8192;
  localparam integer SIDEB_SPAN  = 2048;
  localparam integer MEM_WORDS   = 16384;

  localparam integer N_KV_ITERS    = 40;
  localparam integer N_WEIGHT_ITERS = 30;
  localparam integer N_SIDEB_ITERS  = 60;

  // Deliberately non-integer-multiple periods (matches tb_kevgpt_ddr_bundle.sv's
  // own reasoning) PLUS a randomized startup phase offset on ui_clk -- real
  // gen_clk/ui_clk have no guaranteed fixed phase relationship at power-on
  // (Sec6), so re-running this gate with a different $random seed should
  // sweep through different relative phases, not always the same one.
  integer seed, seed_reseed_tmp;
  reg [31:0] seed_u;   // unsigned view -- SEEDVAL is a bit pattern, not a signed
                       // count, and `seed % 5` on the signed `integer` view can
                       // go negative (e.g. 32'hFEED0001 % 5 == -4); a negative
                       // `#delay` silently wraps to a huge unsigned time value
                       // in Verilog, which reads as a dead sim hang (found via
                       // this gate's own seed sweep: 32'hFEED0001 never even
                       // reached the SEED= print before this fix).
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;
  reg ui_clk = 0, ui_rst = 1;
  initial begin
    seed = `SEEDVAL;
    seed_u = seed;
    seed_reseed_tmp = seed;
    seed_reseed_tmp = $urandom(seed_reseed_tmp);  // reseeds the global $urandom stream (Icarus requires a variable arg)
    #(3.5 + (seed_u % 5));  // 0-4ns extra startup skew before ui_clk's first edge
    forever #3.5 ui_clk = ~ui_clk;
  end

  // =====================================================================
  // kv_bank (reference) + kv_bank_ddr (DUT) -- gen_clk domain throughout,
  // matching real hardware (both live inside sequencer_vec's own
  // compute-clock hierarchy).
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
  // weight_bank_tdp (resident) + weight_loader_ddr (DUT) -- ALSO gen_clk,
  // matching real hardware (Sec4a: an earlier gate got this wrong for
  // weight_loader_ddr specifically and it cost real debugging time --
  // getting it right from the start here).
  // =====================================================================
  reg [$clog2(WWORDS)-1:0] wb_rd_addr_r;
  wire [$clog2(WWORDS)-1:0] wb_raddr_b;
  wire [WBITS-1:0]          wb_rword_b;
  assign wb_raddr_b = wb_rd_addr_r;

  wire wb_ld_rst, wb_w_we;
  wire [31:0] wb_w_data;

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

  weight_loader_ddr #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_wl_dut (
      .clk(clk), .rst(rst),
      .ld_start(ld_start), .ld_ddr_addr(ld_ddr_addr), .ld_words(ld_words), .ld_done(ld_done),
      .wb_ld_rst(wb_ld_rst), .wb_w_we(wb_w_we), .wb_w_data(wb_w_data),
      .rd_req_valid(wl_rd_req_valid), .rd_req_ready(wl_rd_req_ready), .rd_req_addr(wl_rd_req_addr),
      .rd_ret_valid(wl_rd_ret_valid), .rd_ret_ready(wl_rd_ret_ready), .rd_ret_data(wl_rd_ret_data)
  );

  // =====================================================================
  // kevgpt_ddr_bundle (side A) + synthetic side B (CPU stand-in) +
  // mig_dual_master_arbiter + mig_behav_model_rand
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

  mig_behav_model_rand #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(MEM_WORDS),
      .LATENCY_MIN(4), .LATENCY_MAX(20), .QUEUE_DEPTH(64),
      .RDY_STALL_PCT(20), .WDF_STALL_PCT(20), .SEED(`SEEDVAL)) u_mem (
      .clk_i(ui_clk), .rst_ni(!ui_rst),
      .app_addr_i(phys_app_addr), .app_cmd_i(phys_app_cmd), .app_en_i(phys_app_en), .app_rdy_o(phys_app_rdy),
      .app_wdf_data_i(phys_app_wdf_data), .app_wdf_mask_i(phys_app_wdf_mask),
      .app_wdf_wren_i(phys_app_wdf_wren), .app_wdf_end_i(phys_app_wdf_end), .app_wdf_rdy_o(phys_app_wdf_rdy),
      .app_rd_data_o(phys_app_rd_data), .app_rd_data_valid_o(phys_app_rd_data_valid), .app_rd_data_end_o()
  );

  // =====================================================================
  // shared error counter + per-generator done flags
  // =====================================================================
  integer errors = 0;
  reg kv_done_flag = 0, weight_done_flag = 0, sideb_done_flag = 0;

  // =====================================================================
  // KV generator: N_KV_ITERS random (layer,kv,head,pos) write+read cycles,
  // checked against kv_bank (reference) every time.
  //
  // Completion is detected on rd_done/rd_done_ddr's own one-cycle pulse
  // (polled every clk in the tasks below), NOT by comparing a derived
  // rd_valid-counter against a target count -- a real bug found by this
  // gate's own seed sweep (SEEDVAL=32'h7FFFFFFF/32'h13579BDF hung forever):
  // the counter only resets on rd_start's OWN edge, so if kv_worker draws
  // the same `pos` twice in a row (same target count), the *stale* count
  // left over from the PREVIOUS read already equals the new target the
  // instant the new rd_start pulse fires, and the testbench's `wait()` --
  // racing kv_bank_ddr's own reset-on-rd_start against the counter's NBA
  // update -- can observe that stale match before a single new rd_valid
  // has actually arrived, letting kv_worker sail past a read that has
  // barely started. kv_bank_ddr then silently drops the FOLLOWING
  // iteration's rd_start (its rrst FSM is still mid-stream, busy finishing
  // the read kv_worker just mis-timed), and kv_worker hangs forever
  // waiting on a completion that can now never come. rd_done/rd_done_ddr
  // are genuine single-cycle pulses (unconditionally cleared every cycle
  // in kv_bank.sv/kv_bank_ddr.sv's own FSMs), immune to this staleness.
  reg [HEAD_DIM*32-1:0] rd_data_last, rd_data_last_ddr;
  always @(posedge clk) if (rd_valid) rd_data_last <= rd_data;
  always @(posedge clk) if (rd_valid_ddr) rd_data_last_ddr <= rd_data_ddr;

  reg [7:0] wqr_cnt, wqd_cnt;
  always @(posedge clk) begin
    if (rst) wqr_cnt <= 8'd0; else if (wq_done_ref) wqr_cnt <= wqr_cnt + 8'd1;
  end
  always @(posedge clk) begin
    if (rst) wqd_cnt <= 8'd0; else if (wq_done_ddr) wqd_cnt <= wqd_cnt + 8'd1;
  end

  reg [7:0] ldn_cnt;
  always @(posedge clk) begin
    if (rst) ldn_cnt <= 8'd0; else if (ld_done) ldn_cnt <= ldn_cnt + 8'd1;
  end

  task automatic do_kv_write(input [3:0] layer, input kv, input [1:0] head, input [8:0] pos,
                              input [HEAD_DIM*32-1:0] vflat);
    integer b, l;
    begin
      @(posedge clk);
      wq_layer <= layer; wq_kv <= kv; wq_head <= head; wq_pos <= pos;
      wq_start <= 1'b1;
      @(posedge clk);
      wq_start <= 1'b0;
      for (b = 0; b < HR; b = b + 1) begin
        for (l = 0; l < P; l = l + 1)
          wq_data[l*32 +: 32] <= vflat[(b*P+l)*32 +: 32];
        wq_valid <= 1'b1;
        @(posedge clk);
      end
      wq_valid <= 1'b0;
    end
  endtask

  task automatic kv_worker;
    integer it, li;
    integer wq_issued;
    reg [3:0] layer;
    reg kv_sel;
    reg [1:0] head;
    reg [8:0] pos;
    reg [HEAD_DIM*32-1:0] vec;
    begin
      wq_issued = 0;
      for (it = 0; it < N_KV_ITERS; it = it + 1) begin
        layer  = $urandom_range(0, NLAYER-1);
        kv_sel = $urandom_range(0, 1);
        head   = $urandom_range(0, NHEAD-1);
        pos    = $urandom_range(0, TMAX-1);
        for (li = 0; li < HEAD_DIM; li = li + 1)
          vec[li*32 +: 32] = $urandom;

        do_kv_write(layer, kv_sel, head, pos, vec);
        wq_issued = wq_issued + 1;
        wait (wqr_cnt == wq_issued[7:0]);
        wait (wqd_cnt == wq_issued[7:0]);
        @(posedge clk);

        rd_layer <= layer; rd_kv <= kv_sel; rd_head <= head; rd_tcount <= pos + 9'd1;
        @(posedge clk);
        rd_start <= 1'b1; @(posedge clk); rd_start <= 1'b0;
        while (!rd_done) @(posedge clk);
        @(posedge clk);

        rd_layer_ddr <= layer; rd_kv_ddr <= kv_sel; rd_head_ddr <= head; rd_tcount_ddr <= pos + 9'd1;
        @(posedge clk);
        rd_start_ddr_r <= 1'b1; @(posedge clk); rd_start_ddr_r <= 1'b0;
        while (!rd_done_ddr) @(posedge clk);
        @(posedge clk);

        for (li = 0; li < HEAD_DIM; li = li + 1) begin
          if (rd_data_last_ddr[li*32 +: 32] !== rd_data_last[li*32 +: 32]) begin
            $display("KV_MISMATCH,it=%0d,layer=%0d,kv=%0d,head=%0d,pos=%0d,lane=%0d,got=%0d,want=%0d",
                      it, layer, kv_sel, head, pos, li,
                      $signed(rd_data_last_ddr[li*32 +: 32]), $signed(rd_data_last[li*32 +: 32]));
            errors = errors + 1;
          end
        end
        #($urandom_range(0, 20));
      end
      $display("KV_WORKER_DONE,iters=%0d,errors_so_far=%0d", N_KV_ITERS, errors);
      kv_done_flag = 1;
    end
  endtask

  // =====================================================================
  // Weight generator: N_WEIGHT_ITERS random-offset/random-size loads,
  // checked against the source DDR image every time.
  // =====================================================================
  reg [31:0] lfsr_w;
  function automatic [31:0] next_lfsr(input [31:0] x);
    next_lfsr = {x[30:0], x[31] ^ x[21] ^ x[1] ^ x[0]};
  endfunction

  task automatic weight_worker;
    integer it, wi, ci;
    integer beats, words, base_beat;
    reg [WBITS-1:0] want_word;
    begin
      lfsr_w = 32'hCAFEF00D;
      for (it = 0; it < N_WEIGHT_ITERS; it = it + 1) begin
        beats     = $urandom_range(1, WWORDS / SUBW);   // 1..8 beats
        words     = beats * SUBW;
        base_beat = WEIGHT_BASE + $urandom_range(0, WEIGHT_SPAN - beats);

        for (wi = 0; wi < beats; wi = wi + 1) begin
          for (ci = 0; ci < SUBW; ci = ci + 1) begin
            lfsr_w = next_lfsr(lfsr_w);
            u_mem.mem[base_beat + wi][ci*32 +: 32] = lfsr_w;
          end
        end

        ld_ddr_addr <= base_beat * (DATA_W / 8);
        ld_words    <= words;
        @(posedge clk);
        ld_start <= 1'b1; @(posedge clk); ld_start <= 1'b0;
        wait (ldn_cnt == (it + 1));
        @(posedge clk);

        for (wi = 0; wi < beats; wi = wi + 1) begin
          wb_rd_addr_r <= wi[$clog2(WWORDS)-1:0];
          @(posedge clk);
          @(posedge clk);
          want_word = u_mem.mem[base_beat + wi];
          if (wb_rword_b !== want_word) begin
            $display("WEIGHT_MISMATCH,it=%0d,base_beat=%0d,beat=%0d,got=%h,want=%h",
                      it, base_beat, wi, wb_rword_b, want_word);
            errors = errors + 1;
          end
        end
        #($urandom_range(0, 30));
      end
      $display("WEIGHT_WORKER_DONE,iters=%0d,errors_so_far=%0d", N_WEIGHT_ITERS, errors);
      weight_done_flag = 1;
    end
  endtask

  // =====================================================================
  // Side-B generator (synthetic CPU stand-in): N_SIDEB_ITERS random
  // write-then-readback cycles at random addresses in its own region.
  // =====================================================================
  reg [DATA_W-1:0] b_got;
  reg [31:0] lfsr_b;

  task automatic side_b_write_read(input integer word_idx, input [DATA_W-1:0] pattern);
    begin
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
        $display("SIDEB_MISMATCH,word=%0d,got=%h,want=%h", word_idx, b_got, pattern);
        errors = errors + 1;
      end
    end
  endtask

  task automatic sideb_worker;
    integer it, li, word_idx;
    reg [DATA_W-1:0] pattern;
    begin
      lfsr_b = 32'h5A5AF00D;
      for (it = 0; it < N_SIDEB_ITERS; it = it + 1) begin
        word_idx = SIDEB_BASE + $urandom_range(0, SIDEB_SPAN - 1);
        for (li = 0; li < DATA_W/32; li = li + 1) begin
          lfsr_b = next_lfsr(lfsr_b);
          pattern[li*32 +: 32] = lfsr_b;
        end
        side_b_write_read(word_idx, pattern);
        #($urandom_range(0, 15));
      end
      $display("SIDEB_WORKER_DONE,iters=%0d,errors_so_far=%0d", N_SIDEB_ITERS, errors);
      sideb_done_flag = 1;
    end
  endtask

  // =====================================================================
  // top-level sequencing: reset, then all three generators concurrently
  // =====================================================================
  initial begin
    wq_start = 0; wq_layer = 0; wq_kv = 0; wq_head = 0; wq_pos = 0; wq_valid = 0; wq_data = 0;
    rd_start = 0; rd_layer = 0; rd_kv = 0; rd_head = 0; rd_tcount = 0;
    rd_start_ddr_r = 0; rd_layer_ddr = 0; rd_kv_ddr = 0; rd_head_ddr = 0; rd_tcount_ddr = 0;
    wb_rd_addr_r = 0;
    ld_start = 0; ld_ddr_addr = 0; ld_words = 0;
    b_app_addr_r = 0; b_app_cmd_r = 0; b_app_en_r = 0;
    b_app_wdf_data_r = 0; b_app_wdf_mask_r = {(DATA_W/8){1'b1}}; b_app_wdf_wren_r = 0;

    rst = 1;
    ui_rst = 1;
    repeat (4) @(posedge clk);
    rst = 0;
    repeat (4) @(posedge ui_clk);
    ui_rst = 0;
    @(posedge clk);

    $display("SEED=%0d", seed);

    fork
      kv_worker;
      weight_worker;
      sideb_worker;
    join

    if (errors == 0) $display("KEVGPT_DDR_BUNDLE_FULL_VERDICT,PASS");
    else $display("KEVGPT_DDR_BUNDLE_FULL_VERDICT,FAIL,errors=%0d", errors);
    $finish;
  end

  initial begin
    #4000000;
    $display("KEVGPT_DDR_BUNDLE_FULL_VERDICT,TIMEOUT,kv_done=%0d,weight_done=%0d,sideb_done=%0d,errors_so_far=%0d",
              kv_done_flag, weight_done_flag, sideb_done_flag, errors);
    $finish;
  end
endmodule
