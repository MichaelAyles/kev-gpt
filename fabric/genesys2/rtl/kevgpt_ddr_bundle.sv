// -----------------------------------------------------------------------------
// kevgpt_ddr_bundle — combines ALL of kevgpt's own DDR traffic (KV-cache
// read+write from kv_bank_ddr.sv, weight-window reads from
// weight_loader_ddr.sv) into ONE self-consistent app-level bundle, the shape
// mig_dual_master_arbiter.sv's side A (or side B) expects. PORT-NOTES.md
// "Phase 2 architecture": "one mig_read_engine + one mig_write_engine + one
// mig_rw_arbiter for ALL of kevgpt's own traffic".
//
// This module does NOT instantiate kv_bank_ddr/weight_loader_ddr themselves
// (those belong wherever sequencer_vec.sv's own hierarchy puts them, a later
// integration step) -- it only takes their DMA-facing ports (pkt_*/ack_* for
// the write side, req_*/ret_* for both read sides) as pass-through ports and
// produces the merged bundle:
//
//   kv_bank_ddr(write)     --pkt/ack--CDC--> mig_write_engine --cmd--\
//                                                                      mig_rw_arbiter -> app_*
//   kv_bank_ddr(read)      --req/ret--CDC--\                         /
//                                            mig_read_mux2 -> mig_read_engine --cmd--/
//   weight_loader_ddr(read)--req/ret--CDC--/
//
// TWO CLOCK DOMAINS, not one: kv_bank_ddr/weight_loader_ddr live inside
// sequencer_vec.sv's own hierarchy, clocked on kevgpt's compute clock
// (`gen_clk`, "clk_gen" at the top-level wrapper -- itself PLL-derived FROM
// MIG's ui_clk, a genuinely different frequency, not just a renamed wire:
// PORT-NOTES.md's earlier session work confirmed clk_gen=50MHz on this
// board). mig_read_engine/mig_write_engine/mig_rw_arbiter and the produced
// app-level bundle MUST run on `ui_clk` -- MIG's native-UI app port is a
// hard ui_clk-synchronous requirement, not a design choice. This module
// therefore does its own clock-domain crossing for kv_bank_ddr's ports,
// using async_fifo_gray -- the SAME CDC primitive and the SAME 4-FIFO shape
// (write-packet, write-ack, read-request, read-return) cpu_ddr_bridge.sv
// already uses for its own clk_i/ui_clk_i boundary. Caught late (after the
// wiring was first drafted single-clock, matching how every Icarus
// simulation gate in this session's work used one shared `clk` for
// everything, silently hiding this real requirement) -- worth remembering:
// simulating a DMA path entirely on one clock does NOT prove it's CDC-safe
// on real two-clock-domain hardware.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kevgpt_ddr_bundle #(
    parameter integer ADDR_W = 29,
    parameter integer DATA_W = 256,
    parameter integer CDC_FIFO_DEPTH = 4
) (
    // gen_clk domain: kevgpt's own compute clock (matches kv_bank_ddr's own
    // clock -- it lives inside sequencer_vec's clk_gen-domain hierarchy).
    input  wire        gen_clk,
    input  wire        gen_rst,

    // ui_clk domain: MIG's own UI clock.
    input  wire        ui_clk,
    input  wire        ui_rst,

    // ---- kv_bank_ddr's write port (pkt_*/ack_*), gen_clk domain -------------
    input  wire                 kv_wr_pkt_valid,
    output wire                 kv_wr_pkt_ready,
    input  wire [ADDR_W-1:0]    kv_wr_pkt_addr,
    input  wire [DATA_W-1:0]    kv_wr_pkt_data,
    input  wire [DATA_W/8-1:0]  kv_wr_pkt_mask,
    output wire                 kv_wr_ack_valid,
    input  wire                 kv_wr_ack_ready,

    // ---- kv_bank_ddr's read port (req_*/ret_*), gen_clk domain --------------
    input  wire                 kv_rd_req_valid,
    output wire                 kv_rd_req_ready,
    input  wire [ADDR_W-1:0]    kv_rd_req_addr,
    output wire                 kv_rd_ret_valid,
    input  wire                 kv_rd_ret_ready,
    output wire [DATA_W-1:0]    kv_rd_ret_data,

    // ---- weight_loader_ddr's read port (req_*/ret_*), gen_clk domain --
    // now the SAME req/ret CDC treatment kv_bank_ddr's read port gets below
    // (async_fifo_gray, same 2-FIFO shape) is applied to this port too. ----
    input  wire                 wl_rd_req_valid,
    output wire                 wl_rd_req_ready,
    input  wire [ADDR_W-1:0]    wl_rd_req_addr,
    output wire                 wl_rd_ret_valid,
    input  wire                 wl_rd_ret_ready,
    output wire [DATA_W-1:0]    wl_rd_ret_data,

    // ---- produced app-level bundle, ui_clk domain: one side of a
    // mig_dual_master_arbiter -------------------------------------------------
    output wire [ADDR_W-1:0]    app_addr,
    output wire [2:0]           app_cmd,
    output wire                 app_en,
    input  wire                 app_rdy,
    output wire [DATA_W-1:0]    app_wdf_data,
    output wire [DATA_W/8-1:0]  app_wdf_mask,
    output wire                 app_wdf_wren,
    output wire                 app_wdf_end,
    input  wire                 app_wdf_rdy,
    input  wire [DATA_W-1:0]    app_rd_data,
    input  wire                 app_rd_data_valid,

    // ---- owner-FIFO health taps (fabric/genesys2/FIXATION-WORD-CDC-
    // INVESTIGATION.md Sec8 item 7): raw, ui_clk-domain pass-through of
    // u_rd_mux's own item-6 debug flags, for ddr_health_monitor.sv at the
    // top level to latch/count/CDC into a permanent, software-readable
    // status -- no logic added here, this module stays a pure pass-
    // through so its own existing simulation gates are unaffected.
    output wire                 rdmux_owner_mismatch,
    output wire                 rdmux_push_not_ready,
    output wire                 rdmux_pop_when_empty
);
    // ---- CDC: kv_bank_ddr's write packet (gen_clk -> ui_clk) ----------------
    wire kv_wr_pkt_valid_ui, kv_wr_pkt_ready_ui;
    wire [ADDR_W+DATA_W+DATA_W/8-1:0] kv_wr_pkt_bus_ui;
    async_fifo_gray #(.DATA_W(ADDR_W+DATA_W+DATA_W/8), .DEPTH(CDC_FIFO_DEPTH)) u_kv_wr_pkt_cdc (
        .wr_clk_i(gen_clk), .wr_rst_ni(!gen_rst),
        .wr_valid_i(kv_wr_pkt_valid), .wr_ready_o(kv_wr_pkt_ready),
        .wr_data_i({kv_wr_pkt_addr, kv_wr_pkt_data, kv_wr_pkt_mask}),
        .wr_full_o(), .wr_almost_full_o(),
        .rd_clk_i(ui_clk), .rd_rst_ni(!ui_rst),
        .rd_valid_o(kv_wr_pkt_valid_ui), .rd_ready_i(kv_wr_pkt_ready_ui),
        .rd_data_o(kv_wr_pkt_bus_ui),
        .rd_empty_o(), .rd_almost_empty_o()
    );
    wire [ADDR_W-1:0]   kv_wr_pkt_addr_ui = kv_wr_pkt_bus_ui[ADDR_W+DATA_W+DATA_W/8-1 -: ADDR_W];
    wire [DATA_W-1:0]   kv_wr_pkt_data_ui = kv_wr_pkt_bus_ui[DATA_W+DATA_W/8-1 -: DATA_W];
    wire [DATA_W/8-1:0] kv_wr_pkt_mask_ui = kv_wr_pkt_bus_ui[DATA_W/8-1:0];

    // ---- CDC: kv_bank_ddr's write ack (ui_clk -> gen_clk) -------------------
    wire kv_wr_ack_valid_ui, kv_wr_ack_ready_ui;
    async_fifo_gray #(.DATA_W(1), .DEPTH(CDC_FIFO_DEPTH)) u_kv_wr_ack_cdc (
        .wr_clk_i(ui_clk), .wr_rst_ni(!ui_rst),
        .wr_valid_i(kv_wr_ack_valid_ui), .wr_ready_o(kv_wr_ack_ready_ui),
        .wr_data_i(1'b1),
        .wr_full_o(), .wr_almost_full_o(),
        .rd_clk_i(gen_clk), .rd_rst_ni(!gen_rst),
        .rd_valid_o(kv_wr_ack_valid), .rd_ready_i(kv_wr_ack_ready),
        .rd_data_o(),
        .rd_empty_o(), .rd_almost_empty_o()
    );

    // ---- CDC: kv_bank_ddr's read request (gen_clk -> ui_clk) ----------------
    wire kv_rd_req_valid_ui, kv_rd_req_ready_ui;
    wire [ADDR_W-1:0] kv_rd_req_addr_ui;
    async_fifo_gray #(.DATA_W(ADDR_W), .DEPTH(CDC_FIFO_DEPTH)) u_kv_rd_req_cdc (
        .wr_clk_i(gen_clk), .wr_rst_ni(!gen_rst),
        .wr_valid_i(kv_rd_req_valid), .wr_ready_o(kv_rd_req_ready),
        .wr_data_i(kv_rd_req_addr),
        .wr_full_o(), .wr_almost_full_o(),
        .rd_clk_i(ui_clk), .rd_rst_ni(!ui_rst),
        .rd_valid_o(kv_rd_req_valid_ui), .rd_ready_i(kv_rd_req_ready_ui),
        .rd_data_o(kv_rd_req_addr_ui),
        .rd_empty_o(), .rd_almost_empty_o()
    );

    // ---- CDC: kv_bank_ddr's read return (ui_clk -> gen_clk) -----------------
    wire kv_rd_ret_valid_ui, kv_rd_ret_ready_ui;
    wire [DATA_W-1:0] kv_rd_ret_data_ui;
    async_fifo_gray #(.DATA_W(DATA_W), .DEPTH(CDC_FIFO_DEPTH)) u_kv_rd_ret_cdc (
        .wr_clk_i(ui_clk), .wr_rst_ni(!ui_rst),
        .wr_valid_i(kv_rd_ret_valid_ui), .wr_ready_o(kv_rd_ret_ready_ui),
        .wr_data_i(kv_rd_ret_data_ui),
        .wr_full_o(), .wr_almost_full_o(),
        .rd_clk_i(gen_clk), .rd_rst_ni(!gen_rst),
        .rd_valid_o(kv_rd_ret_valid), .rd_ready_i(kv_rd_ret_ready),
        .rd_data_o(kv_rd_ret_data),
        .rd_empty_o(), .rd_almost_empty_o()
    );

    // ---- CDC: weight_loader_ddr's read request (gen_clk -> ui_clk) ---------
    wire wl_rd_req_valid_ui, wl_rd_req_ready_ui;
    wire [ADDR_W-1:0] wl_rd_req_addr_ui;
    async_fifo_gray #(.DATA_W(ADDR_W), .DEPTH(CDC_FIFO_DEPTH)) u_wl_rd_req_cdc (
        .wr_clk_i(gen_clk), .wr_rst_ni(!gen_rst),
        .wr_valid_i(wl_rd_req_valid), .wr_ready_o(wl_rd_req_ready),
        .wr_data_i(wl_rd_req_addr),
        .wr_full_o(), .wr_almost_full_o(),
        .rd_clk_i(ui_clk), .rd_rst_ni(!ui_rst),
        .rd_valid_o(wl_rd_req_valid_ui), .rd_ready_i(wl_rd_req_ready_ui),
        .rd_data_o(wl_rd_req_addr_ui),
        .rd_empty_o(), .rd_almost_empty_o()
    );

    // ---- CDC: weight_loader_ddr's read return (ui_clk -> gen_clk) ----------
    wire wl_rd_ret_valid_ui, wl_rd_ret_ready_ui;
    wire [DATA_W-1:0] wl_rd_ret_data_ui;
    async_fifo_gray #(.DATA_W(DATA_W), .DEPTH(CDC_FIFO_DEPTH)) u_wl_rd_ret_cdc (
        .wr_clk_i(ui_clk), .wr_rst_ni(!ui_rst),
        .wr_valid_i(wl_rd_ret_valid_ui), .wr_ready_o(wl_rd_ret_ready_ui),
        .wr_data_i(wl_rd_ret_data_ui),
        .wr_full_o(), .wr_almost_full_o(),
        .rd_clk_i(gen_clk), .rd_rst_ni(!gen_rst),
        .rd_valid_o(wl_rd_ret_valid), .rd_ready_i(wl_rd_ret_ready),
        .rd_data_o(wl_rd_ret_data),
        .rd_empty_o(), .rd_almost_empty_o()
    );

    // ---- read side (ui_clk domain throughout): merge kv_bank_ddr and
    // weight_loader_ddr (both post-CDC) onto one shared mig_read_engine ------
    wire                 rdm_req_valid, rdm_req_ready;
    wire [ADDR_W-1:0]    rdm_req_addr;
    wire                 rdm_ret_valid, rdm_ret_ready;
    wire [DATA_W-1:0]    rdm_ret_data;

    mig_read_mux2 #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_rd_mux (
        .clk(ui_clk), .rst(ui_rst),
        .a_req_valid(kv_rd_req_valid_ui), .a_req_ready(kv_rd_req_ready_ui), .a_req_addr(kv_rd_req_addr_ui),
        .a_ret_valid(kv_rd_ret_valid_ui), .a_ret_ready(kv_rd_ret_ready_ui), .a_ret_data(kv_rd_ret_data_ui),
        .b_req_valid(wl_rd_req_valid_ui), .b_req_ready(wl_rd_req_ready_ui), .b_req_addr(wl_rd_req_addr_ui),
        .b_ret_valid(wl_rd_ret_valid_ui), .b_ret_ready(wl_rd_ret_ready_ui), .b_ret_data(wl_rd_ret_data_ui),
        .req_valid(rdm_req_valid), .req_ready(rdm_req_ready), .req_addr(rdm_req_addr),
        .ret_valid(rdm_ret_valid), .ret_ready(rdm_ret_ready), .ret_data(rdm_ret_data),
        .dbg_owner_mismatch_o(rdmux_owner_mismatch),
        .dbg_push_not_ready_o(rdmux_push_not_ready),
        .dbg_pop_when_empty_o(rdmux_pop_when_empty)
    );

    wire              rd_cmd_valid, rd_cmd_grant;
    wire [ADDR_W-1:0] rd_cmd_addr;

    mig_read_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .RETURN_DEPTH(32),
                       .MAX_OUTSTANDING(16), .SAFETY_MARGIN(4)) u_rd_engine (
        .clk_i(ui_clk), .rst_ni(!ui_rst),
        .req_valid_i(rdm_req_valid), .req_ready_o(rdm_req_ready), .req_addr_i(rdm_req_addr),
        .cmd_valid_o(rd_cmd_valid), .cmd_grant_i(rd_cmd_grant), .cmd_addr_o(rd_cmd_addr),
        .app_rd_data_i(app_rd_data), .app_rd_data_valid_i(app_rd_data_valid),
        .ret_valid_o(rdm_ret_valid), .ret_ready_i(rdm_ret_ready), .ret_data_o(rdm_ret_data),
        .outstanding_o(), .credit_stall_cycles_o(), .overflow_error_o()
    );

    // ---- write side (ui_clk domain): kv_bank_ddr's pkt/ack, post-CDC -------
    wire              wr_cmd_valid, wr_cmd_grant;
    wire [ADDR_W-1:0] wr_cmd_addr;
    wire              wr_wdf_valid;
    wire [DATA_W-1:0] wr_wdf_data;
    wire [DATA_W/8-1:0] wr_wdf_mask;

    mig_write_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_wr_engine (
        .clk_i(ui_clk), .rst_ni(!ui_rst),
        .pkt_valid_i(kv_wr_pkt_valid_ui), .pkt_ready_o(kv_wr_pkt_ready_ui),
        .pkt_addr_i(kv_wr_pkt_addr_ui), .pkt_data_i(kv_wr_pkt_data_ui), .pkt_mask_i(kv_wr_pkt_mask_ui),
        .cmd_valid_o(wr_cmd_valid), .cmd_grant_i(wr_cmd_grant), .cmd_addr_o(wr_cmd_addr),
        .wdf_valid_o(wr_wdf_valid), .wdf_ready_i(app_wdf_rdy),
        .wdf_data_o(wr_wdf_data), .wdf_mask_o(wr_wdf_mask),
        .ack_valid_o(kv_wr_ack_valid_ui), .ack_ready_i(kv_wr_ack_ready_ui),
        .stall_cycles_o()
    );

    assign app_wdf_data = wr_wdf_data;
    assign app_wdf_mask = wr_wdf_mask;
    assign app_wdf_wren = wr_wdf_valid;
    assign app_wdf_end  = wr_wdf_valid;

    // ---- command channel (ui_clk domain): kevgpt's own read engine vs.
    // write engine --------------------------------------------------------
    mig_rw_arbiter #(.ADDR_W(ADDR_W), .BATCH_LIMIT(16)) u_cmd_arb (
        .clk_i(ui_clk), .rst_ni(!ui_rst),
        .rd_valid_i(rd_cmd_valid), .rd_addr_i(rd_cmd_addr), .rd_grant_o(rd_cmd_grant),
        .wr_valid_i(wr_cmd_valid), .wr_addr_i(wr_cmd_addr), .wr_grant_o(wr_cmd_grant),
        .app_addr_o(app_addr), .app_cmd_o(app_cmd), .app_en_o(app_en), .app_rdy_i(app_rdy),
        .rd_grants_o(), .wr_grants_o(), .direction_switches_o()
    );
endmodule
