// -----------------------------------------------------------------------------
// ddr_health_monitor — permanent, always-on hardware health monitor for the
// owner-FIFO architectural invariants that fabric/genesys2/FIXATION-WORD-
// CDC-INVESTIGATION.md Sec8 item 6's ILA already watches (mig_read_mux2.sv's
// single owner FIFO, mig_dual_master_arbiter.sv's separate rd/wr owner
// FIFOs). Sec8 item 7: "keep a small permanent hardware health monitor ...
// so any future regression of this class surfaces as an explicit error
// rather than a silently wrong generated word." An ILA capture needs a
// debug bitstream and a live JTAG/Vivado session re-armed by hand every
// time (see item 6's own real-hardware bring-up log); this module is the
// always-present, plain-register-readable version of the same 8
// invariants, no ILA session required — armed 111 real generations under
// item 6 with zero violations, so this is a "stay quiet, catch anything
// that regresses" monitor, not (yet) evidence of anything actively wrong.
//
// All 8 raw violation flags are ui_clk-domain signals (MIG's native clock
// — both mig_read_mux2 and mig_dual_master_arbiter live there). Sticky-
// latches each one (set on first violation, held until firmware clears),
// keeps an 8-bit saturating total-violation-event counter, and crosses
// both into gen_clk (kevgpt's own compute clock, where xheep_kevgpt_
// peripheral's register file lives) using this project's existing
// common_cells `sync` primitive — plain single-bit 2-flop synchronizers,
// never used here on the raw multi-bit datapath itself, only on sticky
// bits and a toggle pulse (see below).
//
// Clear is a LEVEL, not a pulse: firmware writes health_clear_i=1, waits
// a few gen_clk cycles for the synchronized level to reach and settle on
// ui_clk, then writes health_clear_i=0. Deliberately avoids pulse-CDC — a
// single-cycle pulse can be missed entirely when crossing into a faster
// or unrelated-phase clock domain; a held level cannot be.
//
// The saturating counter crosses domains via a toggle-bit pulse
// synchronizer (a different, simpler idiom than the Gray-code counter CDC
// this project's async_fifo_gray.sv already uses for its own pointers):
// every ui_clk-side increment flips a 1-bit toggle flop; the toggle bit
// itself is CDC-safe (single bit, `sync`'d); gen_clk reconstructs the
// event count by edge-detecting the synchronized toggle bit and
// incrementing its OWN local saturating counter once per detected edge.
// Chosen over Gray-coding the 8-bit count directly because violations are
// expected to be rare, isolated events (not a continuously-running
// pointer stream), so a toggle-per-event is simpler to reason about and
// no less safe.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module ddr_health_monitor (
    input  wire        gen_clk,
    input  wire        gen_rst,     // active-high
    input  wire        ui_clk,
    input  wire        ui_rst,      // active-high

    // ---- raw violation flags, ui_clk domain ---------------------------------
    // mig_read_mux2 (kevgpt_ddr_bundle's u_rd_mux)
    input  wire         rdmux_owner_mismatch,
    input  wire         rdmux_push_not_ready,
    input  wire         rdmux_pop_when_empty,
    // mig_dual_master_arbiter (top-level u_mig_dual_arb)
    input  wire         dualarb_rd_mismatch,
    input  wire         dualarb_wr_mismatch,
    input  wire         dualarb_rd_push_not_ready,
    input  wire         dualarb_wr_push_not_ready,
    input  wire         dualarb_rd_pop_when_empty,

    // ---- gen_clk-domain register interface (xheep_kevgpt_peripheral) -------
    input  wire        health_clear_i,   // level, held while clearing
    output wire [8:0]  health_sticky_o,  // [7:0] one bit per invariant above
                                          // (same bit order as raw, below);
                                          // [8] = combined "any violation ever"
    output wire [7:0]  health_count_o    // saturating total-violation-event count
);
    // ---- ui_clk domain: raw OR, sticky latches, saturating counter ---------
    wire [7:0] raw = {dualarb_rd_pop_when_empty, dualarb_wr_push_not_ready,
                       dualarb_rd_push_not_ready, dualarb_wr_mismatch,
                       dualarb_rd_mismatch, rdmux_pop_when_empty,
                       rdmux_push_not_ready, rdmux_owner_mismatch};
    wire any_raw = |raw;

    wire clear_ui_sync;
    sync #(.STAGES(2), .ResetValue(1'b0)) u_clear_sync (
        .clk_i(ui_clk), .rst_ni(!ui_rst),
        .serial_i(health_clear_i), .serial_o(clear_ui_sync)
    );

    reg [7:0] sticky_q;
    reg       sticky_any_q;
    always @(posedge ui_clk or posedge ui_rst) begin
        if (ui_rst) begin
            sticky_q <= 8'b0; sticky_any_q <= 1'b0;
        end else if (clear_ui_sync) begin
            sticky_q <= 8'b0; sticky_any_q <= 1'b0;
        end else begin
            sticky_q     <= sticky_q | raw;
            sticky_any_q <= sticky_any_q | any_raw;
        end
    end

    // Only the toggle bit crosses to gen_clk (see this file's own header) --
    // no local ui_clk-side count is kept; an earlier draft had one, but
    // nothing ever read it, and Vivado correctly flagged it as dead logic
    // ("Unused sequential element ... was removed") during synthesis.
    reg count_toggle_q;
    always @(posedge ui_clk or posedge ui_rst) begin
        if (ui_rst) count_toggle_q <= 1'b0;
        else if (any_raw) count_toggle_q <= ~count_toggle_q;
    end

    // ---- gen_clk domain: sticky/any bits + reconstructed counter -----------
    wire [7:0] sticky_gen;
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : g_sticky_sync
            sync #(.STAGES(2), .ResetValue(1'b0)) u_sticky_sync (
                .clk_i(gen_clk), .rst_ni(!gen_rst),
                .serial_i(sticky_q[gi]), .serial_o(sticky_gen[gi])
            );
        end
    endgenerate

    wire sticky_any_gen;
    sync #(.STAGES(2), .ResetValue(1'b0)) u_sticky_any_sync (
        .clk_i(gen_clk), .rst_ni(!gen_rst),
        .serial_i(sticky_any_q), .serial_o(sticky_any_gen)
    );

    wire toggle_gen;
    sync #(.STAGES(2), .ResetValue(1'b0)) u_toggle_sync (
        .clk_i(gen_clk), .rst_ni(!gen_rst),
        .serial_i(count_toggle_q), .serial_o(toggle_gen)
    );

    reg       toggle_gen_d;
    reg [7:0] count_gen_q;
    always @(posedge gen_clk or posedge gen_rst) begin
        if (gen_rst) begin
            toggle_gen_d <= 1'b0; count_gen_q <= 8'b0;
        end else begin
            toggle_gen_d <= toggle_gen;
            if (health_clear_i) begin
                count_gen_q <= 8'b0;
            end else if (toggle_gen != toggle_gen_d) begin
                if (count_gen_q != 8'hFF) count_gen_q <= count_gen_q + 8'd1;
            end
        end
    end

    assign health_sticky_o = {sticky_any_gen, sticky_gen};
    assign health_count_o  = count_gen_q;
endmodule
