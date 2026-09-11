// -----------------------------------------------------------------------------
// mig_read_mux2 — merges TWO independent read requesters (kv_bank_ddr's read
// FSM and weight_loader_ddr's read FSM, PORT-NOTES.md "Phase 2 architecture")
// onto ONE shared mig_read_engine, so kevgpt's own DDR traffic needs only a
// single read engine + single write engine (the plan's own stated design:
// "ample bandwidth headroom from Phase 0 makes time-multiplexing one engine
// across two DDR regions fine").
//
// This is the SAME owner-FIFO IDEA mig_dual_master_arbiter.sv already uses
// for its own read-return demux (push an owner tag on every accepted
// command, pop it in the same order against real returns) -- applied one
// layer earlier in the stack: THERE it demuxes mig_read_engine's ret_data_o
// between two already-formed app-level bundles sharing the physical MIG;
// HERE it merges two REQUESTERS onto one mig_read_engine's req_*/ret_* pair,
// upstream of the engine. mig_read_engine returns data in the order
// requests were accepted (it's a single FIFO internally), so an owner FIFO
// pushed in that same accepted-order correctly tracks who owns each
// upcoming return -- but the POP condition is NOT identical to
// mig_dual_master_arbiter's: that module's owner FIFO pops on bare
// app_rd_data_valid_i, because the raw MIG native-UI return path has no
// backpressure (a return must be consumed the cycle it arrives, so valid
// alone means consumed). mig_read_engine's OWN ret_valid_o/ret_ready_i pair,
// by contrast, IS a real handshake (backed by its own return FIFO, data
// waits if the consumer isn't ready) -- so this owner FIFO must pop on
// `ret_valid && ret_ready` (the real "beat was actually consumed" event),
// not on ret_valid alone. Using bare ret_valid here was tried first and
// found wrong empirically (tb_kevgpt_ddr_bundle.sv hung: the owner FIFO
// drained faster than real consumption, going empty while the engine still
// had buffered returns waiting, permanently deadlocking ret_ready at 0) --
// worth remembering before reusing this owner-FIFO idiom against any other
// port that has its own real valid/ready handshake rather than MIG's raw,
// backpressure-free return path.
//
// Command-channel arbitration is the same priority-with-hysteresis scheme
// mig_rw_arbiter.sv/mig_dual_master_arbiter.sv already use (BATCH_LIMIT-style
// anti-thrash), reused here rather than re-derived.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module mig_read_mux2 #(
    parameter integer ADDR_W       = 29,
    parameter integer DATA_W       = 256,
    parameter integer BATCH_LIMIT  = 16,
    parameter integer OWNER_DEPTH  = 32
) (
    input  wire        clk,
    input  wire        rst,

    // ---- side A (e.g. kv_bank_ddr's rd_req_*/rd_ret_*) ----------------------
    input  wire                 a_req_valid,
    output wire                 a_req_ready,
    input  wire [ADDR_W-1:0]    a_req_addr,
    output wire                 a_ret_valid,
    input  wire                 a_ret_ready,
    output wire [DATA_W-1:0]    a_ret_data,

    // ---- side B (e.g. weight_loader_ddr's rd_req_*/rd_ret_*) ---------------
    input  wire                 b_req_valid,
    output wire                 b_req_ready,
    input  wire [ADDR_W-1:0]    b_req_addr,
    output wire                 b_ret_valid,
    input  wire                 b_ret_ready,
    output wire [DATA_W-1:0]    b_ret_data,

    // ---- shared mig_read_engine's req_*/ret_* ports -------------------------
    output wire                 req_valid,
    input  wire                 req_ready,
    output wire [ADDR_W-1:0]    req_addr,
    input  wire                 ret_valid,
    output wire                 ret_ready,
    input  wire [DATA_W-1:0]    ret_data
);
    // ---- command-channel arbitration (priority-with-hysteresis) -------------
    reg prefer_b_q;
    reg [$clog2(BATCH_LIMIT+1)-1:0] batch_count_q;
    wire select_b, both_valid, accepted;

    assign both_valid = a_req_valid && b_req_valid;
    assign select_b   = both_valid ? prefer_b_q : b_req_valid;

    // Owner-FIFO backpressure: don't present a request downstream to
    // mig_read_engine until the owner FIFO (below) has room to track its
    // return. owner_ready comes from u_owner_fifo's in_ready_o -- previously
    // left unconnected, so a request could be accepted here while the owner
    // FIFO was full, the push silently dropped, and the pop-side/return-demux
    // scrambled for every request after it (see mig_dual_master_arbiter.sv's
    // header for the same class of bug, fixed there the same way).
    assign req_valid = (a_req_valid || b_req_valid) && owner_ready;
    assign req_addr  = select_b ? b_req_addr : a_req_addr;
    assign accepted  = req_valid && req_ready;

    assign a_req_ready = accepted && !select_b;
    assign b_req_ready = accepted && select_b;

    always @(posedge clk) begin
        if (rst) begin
            prefer_b_q <= 1'b0;
            batch_count_q <= {($clog2(BATCH_LIMIT+1)){1'b0}};
        end else if (accepted) begin
            if (both_valid) begin
                if (batch_count_q == BATCH_LIMIT - 1) begin
                    prefer_b_q <= !select_b;
                    batch_count_q <= {($clog2(BATCH_LIMIT+1)){1'b0}};
                end else begin
                    prefer_b_q <= select_b;
                    batch_count_q <= batch_count_q + 1'b1;
                end
            end else begin
                prefer_b_q <= select_b;
                batch_count_q <= {{($clog2(BATCH_LIMIT+1)-1){1'b0}}, 1'b1};
            end
        end
    end

    // ---- read-return demux: owner FIFO, pushed on every accepted request,
    // popped in the same order against real ret_valid pulses -----------------
    wire owner_push_valid = accepted;
    wire owner_push_side  = select_b;
    wire owner_pop_side;
    wire owner_empty;
    wire owner_ready;
    wire [$clog2(OWNER_DEPTH+1)-1:0] owner_count;
    wire owner_overflow, owner_underflow;

    sync_fifo #(
        .DATA_W(1),
        .DEPTH (OWNER_DEPTH)
    ) u_owner_fifo (
        .clk_i(clk),
        .rst_ni(!rst),
        .clear_i(1'b0),
        .in_valid_i(owner_push_valid),
        .in_ready_o(owner_ready),
        .in_data_i(owner_push_side),
        .out_valid_o(),
        .out_ready_i(ret_valid && ret_ready),
        .out_data_o(owner_pop_side),
        .full_o(),
        .empty_o(owner_empty),
        .almost_full_o(),
        .almost_empty_o(),
        .count_o(owner_count),
        .overflow_o(owner_overflow),
        .underflow_o(owner_underflow)
    );

    assign a_ret_data  = ret_data;
    assign b_ret_data  = ret_data;
    assign a_ret_valid = ret_valid && !owner_empty && !owner_pop_side;
    assign b_ret_valid = ret_valid && !owner_empty && owner_pop_side;
    // ret_ready must reflect whichever side's ret_valid is actually asserted
    // (only one of a_ret_valid/b_ret_valid is ever high at once), else the
    // engine's return FIFO would drain a beat neither side acknowledged.
    assign ret_ready = (owner_empty) ? 1'b0
                      : owner_pop_side ? b_ret_ready : a_ret_ready;

    // Outstanding-request accounting: independent push/pop counter,
    // cross-checked against the owner FIFO's own count_o every cycle --
    // deliberately redundant with the FIFO's internal bookkeeping, to catch
    // a divergence between what this mux's own accept/pop logic believes is
    // outstanding and what the FIFO primitive itself believes (exactly the
    // class of bug the previously-unconnected in_ready_o allowed).
    reg [$clog2(OWNER_DEPTH+1)-1:0] outstanding_q;
    wire owner_pop = ret_valid && ret_ready;
    always @(posedge clk) begin
        if (rst) outstanding_q <= {($clog2(OWNER_DEPTH+1)){1'b0}};
        else outstanding_q <= outstanding_q
                             + (owner_push_valid && owner_ready)
                             - owner_pop;
    end

`ifndef SYNTHESIS
    a_no_owner_underflow :
    assert property (@(posedge clk) disable iff (rst) ret_valid |-> !owner_empty)
    else $error("mig_read_mux2: read return with no owner recorded");

    a_owner_never_pushed_when_not_ready :
    assert property (@(posedge clk) disable iff (rst) owner_push_valid |-> owner_ready)
    else $error("mig_read_mux2: owner FIFO pushed while not ready -- backpressure gate failed");

    a_owner_no_overflow :
    assert property (@(posedge clk) disable iff (rst) !owner_overflow)
    else $error("mig_read_mux2: owner FIFO overflow");

    a_owner_no_underflow :
    assert property (@(posedge clk) disable iff (rst) !owner_underflow)
    else $error("mig_read_mux2: owner FIFO underflow");

    a_outstanding_matches_fifo :
    assert property (@(posedge clk) disable iff (rst) outstanding_q == owner_count)
    else $error("mig_read_mux2: outstanding-request counter diverged from owner FIFO level");
`endif
endmodule
