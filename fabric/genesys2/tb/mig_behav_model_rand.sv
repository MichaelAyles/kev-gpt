// -----------------------------------------------------------------------------
// mig_behav_model_rand — a stress variant of mig_behav_model.sv for the full
// multi-master contention gate (fabric/genesys2/FIXATION-WORD-CDC-INVESTIGATION.md
// Sec8 item 5): real MIG native-UI backpressure is NOT constant-ready the way
// mig_behav_model.sv deliberately assumes (see that file's own header -- a
// DELIBERATE simplification for the single-master DMA-correctness gates it was
// built for). This variant adds two things those gates never needed to exercise:
//
// 1. Randomized (but still-in-order) read latency: real MIG can and does vary
//    how long a given read command takes to return data (row/bank state,
//    refresh, etc.) -- what it does NOT do is reorder returns relative to
//    request order (the same "native-UI is a strict per-port FIFO" guarantee
//    every owner-FIFO idiom in this project's DMA path already assumes and
//    depends on). Modeled as a small pending-request queue where each entry
//    gets its OWN random extra latency at push time, but entries can only ever
//    be emitted in push order -- a later entry's shorter random draw can never
//    let it jump ahead of an earlier, still-pending entry. This is the
//    ordering invariant the owner-FIFO fix (Sec4/Sec8 item 2) depends on; if
//    that fix or any owner-FIFO idiom elsewhere is wrong, THIS is what should
//    expose it, unlike mig_behav_model.sv's fixed-latency pipe, which can't.
//
// 2. Randomized command/write-data backpressure: app_rdy_o/app_wdf_rdy_o
//    deassert with a configurable, bounded probability instead of being tied
//    high -- exercises every `app_rdy_i`/`app_wdf_rdy_i`-gated stall path in
//    mig_dual_master_arbiter.sv, mig_read_engine.sv, and mig_write_engine.sv
//    that mig_behav_model.sv's always-ready assumption never touches.
//
// NOT a timing model of real MIG any more than mig_behav_model.sv is (no
// calibration, no bank/row conflicts, no refresh) -- this only randomizes the
// two specific behaviors above, deliberately, to stress the code paths that
// assume real MIG *can* do them. Same data-correctness contract as
// mig_behav_model.sv otherwise: writes commit to `mem[]` byte-masked, reads
// return mem[] contents, app_wdf_mask_i is active-low (0 = write this byte).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module mig_behav_model_rand #(
    parameter int unsigned ADDR_W        = 29,
    parameter int unsigned DATA_W        = 256,
    parameter int unsigned MEM_WORDS     = 16384,
    parameter int unsigned LATENCY_MIN   = 4,
    parameter int unsigned LATENCY_MAX   = 20,
    parameter int unsigned WR_ADDR_DEPTH = 16,
    parameter int unsigned QUEUE_DEPTH   = 64,
    parameter int unsigned RDY_STALL_PCT  = 25,  // 0-100, chance app_rdy_o is LOW on a given cycle
    parameter int unsigned WDF_STALL_PCT  = 25,  // 0-100, chance app_wdf_rdy_o is LOW on a given cycle
    parameter int unsigned SEED           = 32'hC0FFEE
) (
    input  logic                    clk_i,
    input  logic                    rst_ni,

    input  logic [    ADDR_W-1:0]   app_addr_i,
    input  logic [           2:0]   app_cmd_i,
    input  logic                    app_en_i,
    output logic                    app_rdy_o,

    input  logic [    DATA_W-1:0]   app_wdf_data_i,
    input  logic [  DATA_W/8-1:0]   app_wdf_mask_i,
    input  logic                    app_wdf_wren_i,
    input  logic                    app_wdf_end_i,
    output logic                    app_wdf_rdy_o,

    output logic [    DATA_W-1:0]   app_rd_data_o,
    output logic                    app_rd_data_valid_o,
    output logic                    app_rd_data_end_o
);
  localparam logic [2:0] MIG_CMD_WRITE = 3'b000;
  localparam logic [2:0] MIG_CMD_READ  = 3'b001;
  localparam int unsigned WORD_A = $clog2(MEM_WORDS);
  localparam int unsigned QPTR_W = $clog2(QUEUE_DEPTH);

  reg [DATA_W-1:0] mem [0:MEM_WORDS-1];

  localparam integer BYTE_SHIFT = $clog2(DATA_W/8);
  function automatic [WORD_A-1:0] word_idx(input [ADDR_W-1:0] byte_addr);
    word_idx = byte_addr[WORD_A+BYTE_SHIFT-1:BYTE_SHIFT];
  endfunction

  // ---- randomized command-channel backpressure -----------------------------
  int unsigned rdy_rng_q;
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) rdy_rng_q <= SEED;
    else rdy_rng_q <= rdy_rng_q * 1103515245 + 12345;
  end
  assign app_rdy_o = ((rdy_rng_q >> 8) % 100) >= RDY_STALL_PCT;
  wire cmd_fire  = app_en_i && app_rdy_o;
  wire wr_accept = cmd_fire && (app_cmd_i == MIG_CMD_WRITE);
  wire rd_accept = cmd_fire && (app_cmd_i == MIG_CMD_READ);

  // ---- randomized write-data-channel backpressure --------------------------
  int unsigned wdf_rng_q;
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) wdf_rng_q <= SEED ^ 32'h5A5A5A5A;
    else wdf_rng_q <= wdf_rng_q * 1103515245 + 12345;
  end
  assign app_wdf_rdy_o = ((wdf_rng_q >> 8) % 100) >= WDF_STALL_PCT;
  wire wdf_fire = app_wdf_wren_i && app_wdf_rdy_o;

  reg [ADDR_W-1:0] wr_addr_q [0:WR_ADDR_DEPTH-1];
  reg [$clog2(WR_ADDR_DEPTH+1)-1:0] wr_cnt;
  reg [$clog2(WR_ADDR_DEPTH)-1:0]   wr_head, wr_tail;
  integer wi;

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_cnt <= 0; wr_head <= 0; wr_tail <= 0;
    end else begin
      if (wr_accept) begin
        wr_addr_q[wr_tail] <= app_addr_i;
        wr_tail <= wr_tail + 1'b1;
      end
      if (wdf_fire) begin
        for (wi = 0; wi < DATA_W/8; wi = wi + 1)
          if (!app_wdf_mask_i[wi])
            mem[word_idx(wr_addr_q[wr_head])][wi*8 +: 8] <= app_wdf_data_i[wi*8 +: 8];
        wr_head <= wr_head + 1'b1;
      end
      case ({wr_accept, wdf_fire})
        2'b10: wr_cnt <= wr_cnt + 1'b1;
        2'b01: wr_cnt <= wr_cnt - 1'b1;
        default: wr_cnt <= wr_cnt;
      endcase
    end
  end

  // ---- randomized-but-in-order read pipe -----------------------------------
  // A real queue, not a fixed shift register: each pushed entry gets its OWN
  // random extra latency, but only the HEAD entry's countdown is ever
  // observed/decremented -- entries behind it cannot be emitted early no
  // matter how short their own random draw turned out to be, matching real
  // MIG's strict per-port in-order return guarantee exactly.
  reg [ADDR_W-1:0]       q_addr  [0:QUEUE_DEPTH-1];
  reg [31:0]             q_delay [0:QUEUE_DEPTH-1];
  reg [QPTR_W-1:0]       q_head, q_tail;
  reg [QPTR_W:0]         q_cnt;
  int unsigned lat_rng_q;

  wire q_push = rd_accept;
  wire q_head_ready = (q_cnt != 0) && (q_delay[q_head] == 0);
  wire q_pop = q_head_ready;

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      lat_rng_q <= SEED ^ 32'h9E3779B9;
    end else begin
      lat_rng_q <= lat_rng_q * 1103515245 + 12345;
    end
  end
  wire [31:0] rand_latency = LATENCY_MIN + ((lat_rng_q >> 8) % (LATENCY_MAX - LATENCY_MIN + 1));

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      q_head <= '0;
      q_tail <= '0;
      q_cnt  <= '0;
    end else begin
      if (q_push) begin
        if (q_cnt == QUEUE_DEPTH) begin
          $display("FATAL: mig_behav_model_rand read-return queue overflow (QUEUE_DEPTH=%0d) -- size it to cover the real sum of MAX_OUTSTANDING across every master sharing this model", QUEUE_DEPTH);
          $finish;
        end
        q_addr[q_tail]  <= app_addr_i;
        q_delay[q_tail] <= rand_latency;
        q_tail <= q_tail + 1'b1;
      end
      if (q_cnt != 0 && !q_pop) begin
        q_delay[q_head] <= q_delay[q_head] - 1'b1;
      end
      if (q_pop) q_head <= q_head + 1'b1;
      case ({q_push, q_pop})
        2'b10: q_cnt <= q_cnt + 1'b1;
        2'b01: q_cnt <= q_cnt - 1'b1;
        default: q_cnt <= q_cnt;
      endcase
    end
  end

  assign app_rd_data_valid_o = q_pop;
  assign app_rd_data_end_o   = q_pop;
  assign app_rd_data_o       = mem[word_idx(q_addr[q_head])];

`ifndef SYNTHESIS
  initial begin
    if (WR_ADDR_DEPTH < 2 || (WR_ADDR_DEPTH & (WR_ADDR_DEPTH-1)) != 0)
      $display("mig_behav_model_rand: WARNING WR_ADDR_DEPTH=%0d is not a power of 2", WR_ADDR_DEPTH);
    if (QUEUE_DEPTH < 2 || (QUEUE_DEPTH & (QUEUE_DEPTH-1)) != 0)
      $display("mig_behav_model_rand: WARNING QUEUE_DEPTH=%0d is not a power of 2", QUEUE_DEPTH);
    if (RDY_STALL_PCT > 60)
      $display("mig_behav_model_rand: WARNING RDY_STALL_PCT=%0d risks starving forward progress", RDY_STALL_PCT);
  end
`endif
endmodule
