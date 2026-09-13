`timescale 1ns / 1ps
// Unit gate for ddr_health_monitor.sv (fabric/genesys2/FIXATION-WORD-CDC-
// INVESTIGATION.md Sec8 item 7): drives each of the 8 raw ui_clk-domain
// violation inputs directly (synthetic stimulus -- this gate does not
// instantiate the real mig_read_mux2/mig_dual_master_arbiter, it only
// proves the monitor's own sticky-latch/CDC/clear/saturating-counter
// logic is correct in isolation, on genuinely different, non-integer-
// multiple gen_clk/ui_clk periods to actually stress the CDC rather than
// accidentally look synchronous), and reads back the gen_clk-domain
// health_sticky_o/health_count_o outputs kevgpt's firmware would read via
// xheep_kevgpt_peripheral's DDR_HEALTH/DDR_ERR_COUNT registers.
//
// Compile alongside:
//   <esl_epfl_x_heep>/hw/vendor/pulp_platform/common_cells/src/sync.sv
//   fabric/genesys2/rtl/ddr_health_monitor.sv
//   fabric/genesys2/tb/tb_ddr_health_monitor.sv
module tb_ddr_health_monitor;
  reg gen_clk = 0, ui_clk = 0;
  reg gen_rst = 1, ui_rst = 1;
  always #7 gen_clk = ~gen_clk;   // deliberately non-integer-multiple vs. ui_clk
  always #5 ui_clk  = ~ui_clk;

  reg rdmux_owner_mismatch, rdmux_push_not_ready, rdmux_pop_when_empty;
  reg dualarb_rd_mismatch, dualarb_wr_mismatch, dualarb_rd_push_not_ready,
      dualarb_wr_push_not_ready, dualarb_rd_pop_when_empty;
  reg health_clear;
  wire [8:0] health_sticky;
  wire [7:0] health_count;

  ddr_health_monitor u_dut (
      .gen_clk(gen_clk), .gen_rst(gen_rst), .ui_clk(ui_clk), .ui_rst(ui_rst),
      .rdmux_owner_mismatch(rdmux_owner_mismatch),
      .rdmux_push_not_ready(rdmux_push_not_ready),
      .rdmux_pop_when_empty(rdmux_pop_when_empty),
      .dualarb_rd_mismatch(dualarb_rd_mismatch),
      .dualarb_wr_mismatch(dualarb_wr_mismatch),
      .dualarb_rd_push_not_ready(dualarb_rd_push_not_ready),
      .dualarb_wr_push_not_ready(dualarb_wr_push_not_ready),
      .dualarb_rd_pop_when_empty(dualarb_rd_pop_when_empty),
      .health_clear_i(health_clear),
      .health_sticky_o(health_sticky),
      .health_count_o(health_count)
  );

  integer errors = 0;

  task automatic check(input string label, input logic [8:0] got, input logic [8:0] want);
    begin
      if (got !== want) begin
        $display("MISMATCH,%0s,got=%0d,want=%0d,t=%0t", label, got, want, $time);
        errors = errors + 1;
      end
    end
  endtask

  // No `ref` task args (unsupported by this Icarus install) -- each of the
  // 8 raw-flag pulses below is written out inline instead of through a
  // shared helper task.

  // conservative settle margin for a level through 2 sync stages on BOTH
  // clocks plus toggle-edge detection -- generous on purpose, this gate is
  // about correctness, not tight CDC-latency timing.
  localparam integer SETTLE_NS = 200;

  initial begin
    rdmux_owner_mismatch = 0; rdmux_push_not_ready = 0; rdmux_pop_when_empty = 0;
    dualarb_rd_mismatch = 0; dualarb_wr_mismatch = 0; dualarb_rd_push_not_ready = 0;
    dualarb_wr_push_not_ready = 0; dualarb_rd_pop_when_empty = 0;
    health_clear = 0;

    repeat (4) @(posedge gen_clk);
    repeat (4) @(posedge ui_clk);
    gen_rst = 0; ui_rst = 0;
    #SETTLE_NS;

    // ---- 1. at reset, everything reads zero --------------------------------
    check("reset_sticky", health_sticky, 9'b0);
    check("reset_count", health_count, 8'b0);

    // ---- 2. each of the 8 raw flags sets its own sticky bit + the combined
    //         "any" bit, and nothing else -------------------------------------
    @(posedge ui_clk); rdmux_owner_mismatch <= 1'b1; @(posedge ui_clk); rdmux_owner_mismatch <= 1'b0; #SETTLE_NS;
    check("bit0_owner_mismatch", health_sticky[0], 1'b1);
    check("bit0_others_clear", health_sticky[7:1], 7'b0);
    check("bit0_any", health_sticky[8], 1'b1);
    check("bit0_count", health_count, 8'd1);

    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;
    check("clear1_sticky", health_sticky, 9'b0);
    check("clear1_count", health_count, 8'b0);

    @(posedge ui_clk); rdmux_push_not_ready <= 1'b1; @(posedge ui_clk); rdmux_push_not_ready <= 1'b0; #SETTLE_NS;
    check("bit1", health_sticky, 9'b1_0000_0010);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    @(posedge ui_clk); rdmux_pop_when_empty <= 1'b1; @(posedge ui_clk); rdmux_pop_when_empty <= 1'b0; #SETTLE_NS;
    check("bit2", health_sticky, 9'b1_0000_0100);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    @(posedge ui_clk); dualarb_rd_mismatch <= 1'b1; @(posedge ui_clk); dualarb_rd_mismatch <= 1'b0; #SETTLE_NS;
    check("bit3", health_sticky, 9'b1_0000_1000);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    @(posedge ui_clk); dualarb_wr_mismatch <= 1'b1; @(posedge ui_clk); dualarb_wr_mismatch <= 1'b0; #SETTLE_NS;
    check("bit4", health_sticky, 9'b1_0001_0000);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    @(posedge ui_clk); dualarb_rd_push_not_ready <= 1'b1; @(posedge ui_clk); dualarb_rd_push_not_ready <= 1'b0; #SETTLE_NS;
    check("bit5", health_sticky, 9'b1_0010_0000);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    @(posedge ui_clk); dualarb_wr_push_not_ready <= 1'b1; @(posedge ui_clk); dualarb_wr_push_not_ready <= 1'b0; #SETTLE_NS;
    check("bit6", health_sticky, 9'b1_0100_0000);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    @(posedge ui_clk); dualarb_rd_pop_when_empty <= 1'b1; @(posedge ui_clk); dualarb_rd_pop_when_empty <= 1'b0; #SETTLE_NS;
    check("bit7", health_sticky, 9'b1_1000_0000);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    // ---- 3. sticky really is sticky: fire once, then confirm it stays set
    //         across many idle ui_clk cycles with no further violation -------
    @(posedge ui_clk); rdmux_owner_mismatch <= 1'b1; @(posedge ui_clk); rdmux_owner_mismatch <= 1'b0; #SETTLE_NS;
    repeat (50) @(posedge ui_clk);
    check("sticky_holds", health_sticky[0], 1'b1);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;

    // ---- 4. counter increments per event and saturates at 0xFF -------------
    repeat (10) begin
      @(posedge ui_clk); dualarb_wr_mismatch <= 1'b1; @(posedge ui_clk); dualarb_wr_mismatch <= 1'b0;
      @(posedge ui_clk);  // one idle cycle between events
    end
    #SETTLE_NS;
    check("count_after_10", health_count, 8'd10);

    // drive 300 more events (well past the 8-bit ceiling) to prove it holds
    // at 255 rather than wrapping
    repeat (300) begin
      @(posedge ui_clk); dualarb_wr_mismatch <= 1'b1; @(posedge ui_clk); dualarb_wr_mismatch <= 1'b0;
      @(posedge ui_clk);
    end
    #SETTLE_NS;
    check("count_saturates", health_count, 8'd255);
    check("sticky_still_set_at_saturation", health_sticky[4], 1'b1);

    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;
    check("clear_after_saturation", health_count, 8'b0);
    check("clear_after_saturation_sticky", health_sticky, 9'b0);

    // ---- 5. a violation concurrent with the START of a clear window is a
    //         documented edge case (clear wins) -- not asserted here, just
    //         confirming the monitor returns to a clean, quiescent state
    //         afterward and continues to function normally -------------------
    @(posedge ui_clk); rdmux_pop_when_empty <= 1'b1; @(posedge ui_clk); rdmux_pop_when_empty <= 1'b0; #SETTLE_NS;
    check("post_clear_still_works", health_sticky[2], 1'b1);
    health_clear = 1; #SETTLE_NS; health_clear = 0; #SETTLE_NS;
    check("post_clear_final", health_sticky, 9'b0);

    if (errors == 0) $display("DDR_HEALTH_MONITOR_VERDICT,PASS");
    else $display("DDR_HEALTH_MONITOR_VERDICT,FAIL,errors=%0d", errors);
    $finish;
  end

  initial begin
    #2000000;
    $display("DDR_HEALTH_MONITOR_VERDICT,TIMEOUT");
    $finish;
  end
endmodule
