`timescale 1ns / 1ps
// Gate for kv_bank_ddr's write- AND read-side FSMs: drives the SAME wq_*
// stimulus into both kv_bank (on-chip, already-verified reference) and
// kv_bank_ddr (new DDR-backed variant), then checks kv_bank_ddr's DDR-
// resident data two ways for the same (layer,kv,head,pos): (a) decoded by
// hand in the testbench straight out of the DDR memory model (checks the
// WRITE path's bytes are correct), and (b) read back through kv_bank_ddr's
// own real rd_start/rd_tcount read port -- via an actual mig_read_engine +
// mig_rw_arbiter, not a shortcut (checks the READ FSM itself) -- both
// compared against kv_bank's own rd_data output for the same row. This is
// an RTL-vs-RTL crosscheck, not yet a Python-golden gate (see PORT-NOTES.md)
// -- kv_bank.sv itself is already gated against model.goformer_kvq/seq_ref
// via fabric.stage3.run_vec_kv, so agreement here transitively confirms
// kv_bank_ddr's quantiser + DMA packing + dequant reproduce that same,
// already-verified behavior.
//
// Compile alongside (no run_*.py harness yet -- invoke directly with iverilog,
// -DSYNTHESIS bracketed around JUST the mig_read_engine/mig_rw_arbiter/
// sync_fifo files via undef_synth.sv/define_synth.sv shims, since kv_bank.sv
// itself must NOT see SYNTHESIS defined -- its `ifdef SYNTHESIS` branch picks
// xpm_memory_tdpram, which Icarus can't elaborate):
//   fabric/stage3/rtl/kv_bank.sv
//   fabric/genesys2/rtl/kv_bank_ddr.sv
//   fabric/genesys2/tb/mig_behav_model.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_write_engine.sv
//   <undef_synth.sv shim>
//   <ai_accel>/rtl/accelerator/streamer/mig_read_engine.sv
//   <ai_accel>/rtl/accelerator/streamer/mig_rw_arbiter.sv
//   <ai_accel>/rtl/accelerator/common/sync_fifo.sv
// plus inv_lut_lo.mem/inv_lut_hi.mem in the run directory (same files
// fabric.stage3.run_vec_kv generates -- reuse, don't regenerate: see that
// gate's INV_SH=24/KBITS=8 q_round_div(2^24, scale) derivation).
module tb_kv_bank_ddr;
  localparam integer P        = 8;
  localparam integer HEAD_DIM = 64;
  localparam integer NHEAD    = 2;
  localparam integer NLAYER   = 2;
  localparam integer TMAX     = 128;  // matches Option A; keeps $clog2(HROWS)>=9 (kv_bank.sv assumes wq_pos/rd_tcount are 9-bit fields)
  localparam integer KBITS    = 8;
  localparam integer INV_SH   = 24;
  localparam integer ADDR_W   = 29;
  localparam integer DATA_W   = 256;
  localparam integer HR       = HEAD_DIM / P;

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  // ---- shared write stimulus ----
  reg        wq_start;
  reg [3:0]  wq_layer;
  reg        wq_kv;
  reg [1:0]  wq_head;
  reg [8:0]  wq_pos;
  reg        wq_valid;
  reg [P*32-1:0] wq_data;

  wire wq_done_ref, wq_done_ddr;

  // ---- reference on-chip kv_bank ----
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

  // ---- kv_bank_ddr under test ----
  wire                 wr_pkt_valid;
  wire                 wr_pkt_ready;
  wire [ADDR_W-1:0]    wr_pkt_addr;
  wire [DATA_W-1:0]    wr_pkt_data;
  wire [DATA_W/8-1:0]  wr_pkt_mask;
  wire                 wr_ack_valid;
  wire                 wr_ack_ready;

  wire         rd_start_ddr;
  reg   [3:0]  rd_layer_ddr;
  reg          rd_kv_ddr;
  reg   [1:0]  rd_head_ddr;
  reg   [8:0]  rd_tcount_ddr;
  wire         rd_valid_ddr, rd_done_ddr;
  wire [HEAD_DIM*32-1:0] rd_data_ddr;

  wire                 rd_req_valid;
  wire                 rd_req_ready;
  wire [ADDR_W-1:0]    rd_req_addr;
  wire                 rd_ret_valid;
  wire                 rd_ret_ready;
  wire [DATA_W-1:0]    rd_ret_data;

  reg rd_start_ddr_r;
  assign rd_start_ddr = rd_start_ddr_r;

  kv_bank_ddr #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER),
                .TMAX(TMAX), .KBITS(KBITS), .INV_SH(INV_SH),
                .ADDR_W(ADDR_W), .DATA_W(DATA_W), .KV_DDR_BASE(0)) u_dut (
      .clk(clk), .rst(rst),
      .wq_start(wq_start), .wq_layer(wq_layer), .wq_kv(wq_kv), .wq_head(wq_head),
      .wq_pos(wq_pos), .wq_valid(wq_valid), .wq_data(wq_data), .wq_done(wq_done_ddr),
      .wr_pkt_valid(wr_pkt_valid), .wr_pkt_ready(wr_pkt_ready),
      .wr_pkt_addr(wr_pkt_addr), .wr_pkt_data(wr_pkt_data), .wr_pkt_mask(wr_pkt_mask),
      .wr_ack_valid(wr_ack_valid), .wr_ack_ready(wr_ack_ready),
      .rd_start(rd_start_ddr), .rd_layer(rd_layer_ddr), .rd_kv(rd_kv_ddr),
      .rd_head(rd_head_ddr), .rd_tcount(rd_tcount_ddr),
      .rd_valid(rd_valid_ddr), .rd_data(rd_data_ddr), .rd_done(rd_done_ddr),
      .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready), .rd_req_addr(rd_req_addr),
      .rd_ret_valid(rd_ret_valid), .rd_ret_ready(rd_ret_ready), .rd_ret_data(rd_ret_data)
  );

  // ---- write engine + read engine, arbitrated onto one behavioral MIG ----
  wire              wr_cmd_valid, wr_cmd_grant;
  wire [ADDR_W-1:0] wr_cmd_addr;
  wire [DATA_W-1:0] wr_wdf_data;
  wire [DATA_W/8-1:0] wr_wdf_mask;
  wire              wr_wdf_valid;
  wire              app_wdf_rdy;
  wire              app_rdy;

  mig_write_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_wr_engine (
      .clk_i(clk), .rst_ni(!rst),
      .pkt_valid_i(wr_pkt_valid), .pkt_ready_o(wr_pkt_ready),
      .pkt_addr_i(wr_pkt_addr), .pkt_data_i(wr_pkt_data), .pkt_mask_i(wr_pkt_mask),
      .cmd_valid_o(wr_cmd_valid), .cmd_grant_i(wr_cmd_grant), .cmd_addr_o(wr_cmd_addr),
      .wdf_valid_o(wr_wdf_valid), .wdf_ready_i(app_wdf_rdy),
      .wdf_data_o(wr_wdf_data), .wdf_mask_o(wr_wdf_mask),
      .ack_valid_o(wr_ack_valid), .ack_ready_i(wr_ack_ready),
      .stall_cycles_o()
  );

  wire              rd_cmd_valid, rd_cmd_grant;
  wire [ADDR_W-1:0] rd_cmd_addr;
  wire [DATA_W-1:0] app_rd_data;
  wire              app_rd_data_valid;
  wire [$clog2(8+1)-1:0] rd_outstanding;  // width must track u_rd_engine's MAX_OUTSTANDING below

  mig_read_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .RETURN_DEPTH(16),
                     .MAX_OUTSTANDING(8), .SAFETY_MARGIN(2)) u_rd_engine (
      .clk_i(clk), .rst_ni(!rst),
      .req_valid_i(rd_req_valid), .req_ready_o(rd_req_ready), .req_addr_i(rd_req_addr),
      .cmd_valid_o(rd_cmd_valid), .cmd_grant_i(rd_cmd_grant), .cmd_addr_o(rd_cmd_addr),
      .app_rd_data_i(app_rd_data), .app_rd_data_valid_i(app_rd_data_valid),
      .ret_valid_o(rd_ret_valid), .ret_ready_i(rd_ret_ready), .ret_data_o(rd_ret_data),
      .outstanding_o(rd_outstanding), .credit_stall_cycles_o(), .overflow_error_o()
  );

  wire [ADDR_W-1:0] app_addr;
  wire [2:0]        app_cmd;
  wire               app_en;

  mig_rw_arbiter #(.ADDR_W(ADDR_W), .BATCH_LIMIT(16)) u_cmd_arb (
      .clk_i(clk), .rst_ni(!rst),
      .rd_valid_i(rd_cmd_valid), .rd_addr_i(rd_cmd_addr), .rd_grant_o(rd_cmd_grant),
      .wr_valid_i(wr_cmd_valid), .wr_addr_i(wr_cmd_addr), .wr_grant_o(wr_cmd_grant),
      .app_addr_o(app_addr), .app_cmd_o(app_cmd), .app_en_o(app_en), .app_rdy_i(app_rdy),
      .rd_grants_o(), .wr_grants_o(), .direction_switches_o()
  );

  // MEM_WORDS must cover ROW_BEATS*HROWS (here 3*1024=3072) -- mig_behav_model
  // silently wraps/truncates addresses beyond its backing store via word_idx()'s
  // fixed-width slice, so an undersized MEM_WORDS does NOT error, it just
  // aliases distant rows onto the same word (caught the hard way once already:
  // an early version of this test used MEM_WORDS=1024, which happened to still
  // "pass" the small-row-index test case while silently corrupting the
  // large-row-index one -- see PORT-NOTES.md).
  mig_behav_model #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .MEM_WORDS(8192), .READ_LATENCY(6)) u_mem (
      .clk_i(clk), .rst_ni(!rst),
      .app_addr_i(app_addr), .app_cmd_i(app_cmd), .app_en_i(app_en), .app_rdy_o(app_rdy),
      .app_wdf_data_i(wr_wdf_data), .app_wdf_mask_i(wr_wdf_mask),
      .app_wdf_wren_i(wr_wdf_valid), .app_wdf_end_i(wr_wdf_valid), .app_wdf_rdy_o(app_wdf_rdy),
      .app_rd_data_o(app_rd_data), .app_rd_data_valid_o(app_rd_data_valid), .app_rd_data_end_o()
  );

  // ---- test vectors: 64 signed Q16-style lanes each, split into HR=8 beats
  // of P=8 lanes -----------------------------------------------------------
  integer errors = 0;
  integer tc, li, bi;
  reg signed [31:0] vec [0:HEAD_DIM-1];
  reg [3:0] t_layer [0:1];
  reg       t_kv    [0:1];
  reg [1:0] t_head  [0:1];
  reg [8:0] t_pos   [0:1];

  // decode helpers (mirror kv_bank_ddr's DDR layout: ROW_BEATS=3, CODE_BEATS=2)
  localparam integer ROW_BEATS = 3;
  reg [$clog2(NLAYER*2*NHEAD*TMAX)-1:0] w_pbase_chk;
  reg [255:0] beat0, beat1, beat_hdr;
  reg [15:0] dec_scale;
  reg signed [31:0] dec_lo;
  reg [7:0] dec_code;
  reg signed [63:0] dec_prod;
  reg signed [33:0] dec_full;
  reg [31:0] want, got;

  // Capture the LAST beat of kv_bank's read stream via a monotonic counter,
  // not a pulse wait -- avoids Verilog's "wait() on an already-satisfied
  // level returns immediately" hazard, which single-cycle-pulse signals like
  // rd_done are exactly the case that trips up.
  reg [8:0] rd_valid_count;
  reg [HEAD_DIM*32-1:0] rd_data_last;
  always @(posedge clk) begin
    if (rst || rd_start) rd_valid_count <= 9'd0;
    else if (rd_valid) rd_valid_count <= rd_valid_count + 9'd1;
  end
  always @(posedge clk) if (rd_valid) rd_data_last <= rd_data;

  // Same capture, for kv_bank_ddr's own real read port (via the read engine).
  reg [8:0] rd_valid_count_ddr;
  reg [HEAD_DIM*32-1:0] rd_data_last_ddr;
  always @(posedge clk) begin
    if (rst || rd_start_ddr) rd_valid_count_ddr <= 9'd0;
    else if (rd_valid_ddr) rd_valid_count_ddr <= rd_valid_count_ddr + 9'd1;
  end
  always @(posedge clk) if (rd_valid_ddr) rd_data_last_ddr <= rd_data_ddr;

  // Same monotonic-counter workaround for wq_done_ref/wq_done_ddr: Icarus's
  // edge detection on a reg driven by the common "default-clear-then-
  // conditionally-set" NBA pulse idiom (wq_done<=0 unconditionally, then
  // wq_done<=1 in one specific case branch -- what both kv_bank.sv and
  // kv_bank_ddr.sv use) triggered @(posedge wq_done_ref)/wait(wq_done_ref)
  // one full cycle early in this testbench, observed directly via debug
  // prints. Counting pulses sidesteps relying on edge timing entirely.
  reg [3:0] wqr_cnt, wqd_cnt;
  always @(posedge clk) begin
    if (rst) wqr_cnt <= 4'd0; else if (wq_done_ref) wqr_cnt <= wqr_cnt + 4'd1;
  end
  always @(posedge clk) begin
    if (rst) wqd_cnt <= 4'd0; else if (wq_done_ddr) wqd_cnt <= wqd_cnt + 4'd1;
  end

  task automatic do_write(input [3:0] layer, input kv, input [1:0] head, input [8:0] pos);
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

  initial begin
    wq_start = 0; wq_layer = 0; wq_kv = 0; wq_head = 0; wq_pos = 0; wq_valid = 0; wq_data = 0;
    rd_start = 0; rd_layer = 0; rd_kv = 0; rd_head = 0; rd_tcount = 0;
    rd_start_ddr_r = 0; rd_layer_ddr = 0; rd_kv_ddr = 0; rd_head_ddr = 0; rd_tcount_ddr = 0;

    t_layer[0] = 4'd1; t_kv[0] = 1'b0; t_head[0] = 2'd1; t_pos[0] = 9'd3;
    t_layer[1] = 4'd0; t_kv[1] = 1'b1; t_head[1] = 2'd0; t_pos[1] = 9'd5;

    rst = 1;
    repeat (4) @(posedge clk);
    rst = 0;
    @(posedge clk);

    for (tc = 0; tc < 2; tc = tc + 1) begin
      // build a test vector: tc=0 small span near a positive offset,
      // tc=1 wide span straddling zero (exercises negative lo)
      for (li = 0; li < HEAD_DIM; li = li + 1) begin
        if (tc == 0) vec[li] = 32'sd1000 + li * 7 - 32'sd200;
        else vec[li] = (li % 2 == 0) ? -(li * 311) : (li * 271);
      end

      fork
        do_write(t_layer[tc], t_kv[tc], t_head[tc], t_pos[tc]);
      join

      // wait for both writers to finish committing
      wait (wqr_cnt == tc + 1);
      wait (wqd_cnt == tc + 1);
      @(posedge clk);

      // ---- reference: kv_bank's read port streams positions 0..tcount-1 of
      // the (layer,kv,head) selector (no direct "read position p" input) --
      // request tcount = pos+1 and keep the LAST valid beat (captured below).
      rd_layer <= t_layer[tc]; rd_kv <= t_kv[tc]; rd_head <= t_head[tc];
      rd_tcount <= t_pos[tc] + 9'd1;
      @(posedge clk);
      rd_start <= 1'b1;
      @(posedge clk);
      rd_start <= 1'b0;
      wait (rd_valid_count == t_pos[tc] + 9'd1);
      @(posedge clk);  // let the final rd_data_last <= rd_data capture settle

      // ---- DUT: decode straight out of the DDR memory model ----
      w_pbase_chk = ((t_layer[tc]*2 + {3'b0,t_kv[tc]})*NHEAD + {2'b0,t_head[tc]})*TMAX
                    + {3'b0,t_pos[tc]};
      beat0    = u_mem.mem[w_pbase_chk*ROW_BEATS + 0];
      beat1    = u_mem.mem[w_pbase_chk*ROW_BEATS + 1];
      beat_hdr = u_mem.mem[w_pbase_chk*ROW_BEATS + 2];
      dec_scale = beat_hdr[47:32];
      dec_lo    = beat_hdr[31:0];

      for (li = 0; li < HEAD_DIM; li = li + 1) begin
        dec_code = (li < 32) ? beat0[li*8 +: 8] : beat1[(li-32)*8 +: 8];
        dec_prod = dec_code * dec_scale;
        dec_full = $signed({1'b0, dec_prod[47:0]}) + $signed({{2{dec_lo[31]}}, dec_lo});
        got  = dec_full[31:0];
        want = rd_data_last[li*32 +: 32];
        if (got !== want) begin
          $display("KV_BANK_DDR_MISMATCH,tc=%0d,lane=%0d,got=%0d,want=%0d", tc, li, $signed(got), $signed(want));
          errors = errors + 1;
        end
      end

      // ---- kv_bank_ddr's OWN real read port (through the DMA read engine) ----
      rd_layer_ddr <= t_layer[tc]; rd_kv_ddr <= t_kv[tc]; rd_head_ddr <= t_head[tc];
      rd_tcount_ddr <= t_pos[tc] + 9'd1;
      @(posedge clk);
      rd_start_ddr_r <= 1'b1;
      @(posedge clk);
      rd_start_ddr_r <= 1'b0;
      wait (rd_valid_count_ddr == t_pos[tc] + 9'd1);
      @(posedge clk);  // let the final rd_data_last_ddr <= rd_data_ddr capture settle

      for (li = 0; li < HEAD_DIM; li = li + 1) begin
        got  = rd_data_last_ddr[li*32 +: 32];
        want = rd_data_last[li*32 +: 32];
        if (got !== want) begin
          $display("KV_BANK_DDR_READ_MISMATCH,tc=%0d,lane=%0d,got=%0d,want=%0d", tc, li, $signed(got), $signed(want));
          errors = errors + 1;
        end
      end
      $display("KV_BANK_DDR_CASE,tc=%0d,checked_lanes=%0d,errors_so_far=%0d", tc, HEAD_DIM, errors);
    end

    if (errors == 0) $display("KV_BANK_DDR_VERDICT,PASS");
    else $display("KV_BANK_DDR_VERDICT,FAIL,errors=%0d", errors);
    $finish;
  end

  initial begin
    #200000;
    $display("KV_BANK_DDR_VERDICT,TIMEOUT");
    $finish;
  end
endmodule
