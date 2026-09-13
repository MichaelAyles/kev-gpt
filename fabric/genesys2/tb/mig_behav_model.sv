// -----------------------------------------------------------------------------
// mig_behav_model — a simulation-only stand-in for genesys2_mig_native_shell's
// app-level UI port, for gating the new DDR-backed kv_bank/weight-window RTL
// (fabric/genesys2's DDR3-streaming redesign, see PORT-NOTES.md "Phase 2
// architecture") the same bit-exact way every other fabric/stage3 block is
// gated: against a Python reference, before any real hardware time is spent.
//
// NOT a timing model of real MIG (no calibration sequence, no bank/row
// conflicts, no realistic app_rdy_o backpressure) -- always-ready command and
// wdf channels, a fixed READ_LATENCY pipe for reads. This is deliberate: the
// thing this model needs to prove is DATA/ADDRESS correctness of the new
// request-issuing FSMs (mig_read_engine/mig_write_engine wiring, address
// arithmetic, beat packing) against the golden reference, not real DDR3
// throughput -- Phase 0 already established bandwidth/latency headroom is
// ample (3.2GB/s peak vs. the 50 tok/s cycle budget), so timing fidelity here
// would cost real design effort for no bit-exactness payoff. Real latency
// behavior is a Phase 6 (real-hardware) concern, not a Phase 2 gate concern.
//
// app_wdf_mask_i polarity follows Xilinx UG586's MIG7 native-UI convention:
// active LOW (0 = byte written, 1 = byte masked/held). Flag this explicitly:
// it is an assumption read from the spec, not yet cross-checked against
// genesys2_mig_native_shell's actual generated wrapper -- re-verify at Phase 6
// before trusting a real-hardware write.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module mig_behav_model #(
    parameter int unsigned ADDR_W       = 29,
    parameter int unsigned DATA_W       = 256,
    parameter int unsigned MEM_WORDS    = 4096,   // sim-scale backing store (DATA_W-wide words)
    parameter int unsigned READ_LATENCY = 12,     // cycles from accepted READ cmd to data valid
    parameter int unsigned WR_ADDR_DEPTH = 8
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

  reg [DATA_W-1:0] mem [0:MEM_WORDS-1];

  // app_addr_i is a BYTE address, beat-aligned (low BYTE_SHIFT bits always 0
  // by convention -- see this codebase's cpu_ddr_bridge.sv header), NOT a
  // DATA_W-wide word index -- shift it down before indexing mem[].
  localparam integer BYTE_SHIFT = $clog2(DATA_W/8);
  function automatic [WORD_A-1:0] word_idx(input [ADDR_W-1:0] byte_addr);
    word_idx = byte_addr[WORD_A+BYTE_SHIFT-1:BYTE_SHIFT];
  endfunction

  // ---- command channel: always ready (see header) ---------------------------
  assign app_rdy_o = 1'b1;
  wire cmd_fire  = app_en_i && app_rdy_o;
  wire wr_accept = cmd_fire && (app_cmd_i == MIG_CMD_WRITE);
  wire rd_accept = cmd_fire && (app_cmd_i == MIG_CMD_READ);

  // ---- write-data channel: always ready; pairs the Nth accepted WRITE cmd
  // with the Nth wdf beat via a small FIFO of pending addresses (mirrors the
  // real ordering contract mig_dual_master_arbiter's header documents) -------
  assign app_wdf_rdy_o = 1'b1;
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
          if (!app_wdf_mask_i[wi])  // active-low mask: 0 = write this byte
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

  // ---- read pipe: fixed READ_LATENCY shift register of {valid, addr} -------
  reg                   rp_valid [0:READ_LATENCY-1];
  reg [ADDR_W-1:0]      rp_addr  [0:READ_LATENCY-1];
  integer rp;

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (rp = 0; rp < READ_LATENCY; rp = rp + 1) rp_valid[rp] <= 1'b0;
    end else begin
      rp_valid[0] <= rd_accept;
      rp_addr[0]  <= app_addr_i;
      for (rp = 1; rp < READ_LATENCY; rp = rp + 1) begin
        rp_valid[rp] <= rp_valid[rp-1];
        rp_addr[rp]  <= rp_addr[rp-1];
      end
    end
  end

  assign app_rd_data_valid_o = rp_valid[READ_LATENCY-1];
  assign app_rd_data_end_o   = rp_valid[READ_LATENCY-1];
  assign app_rd_data_o       = mem[word_idx(rp_addr[READ_LATENCY-1])];

`ifndef SYNTHESIS
  initial begin
    if (WR_ADDR_DEPTH < 2 || (WR_ADDR_DEPTH & (WR_ADDR_DEPTH-1)) != 0)
      $display("mig_behav_model: WARNING WR_ADDR_DEPTH=%0d is not a power of 2 -- wr_head/wr_tail wraparound is not modular-safe", WR_ADDR_DEPTH);
  end
`endif
endmodule
