// -----------------------------------------------------------------------------
// weight_loader_ddr — the DDR3-backed weight-window loader (PORT-NOTES.md
// "Phase 2 architecture", the weight-streaming half). Unlike kv_bank_ddr.sv
// (which replaces kv_bank.sv's own read/write ports with DMA), this module
// touches NEITHER weight_bank_tdp.sv NOR gemv_banked_resident_vec.sv's MAC
// pipeline at all -- both stay exactly as already gated/verified. Instead it
// is a hardware-driven ALTERNATIVE SOURCE for the SAME boot-load port
// (ld_rst/w_we/w_data) firmware already streams the whole weight image
// through today via the wl_we register (xheep_kevgpt_peripheral.sv ->
// sequencer_vec.sv's wl_rst/wl_we/wl_data -> gemv_banked_resident_vec.sv's
// ld_rst/w_we/w_data) -- this loader plays the exact same role, just sourced
// from DDR3 at RUNTIME (once per weight-window reload) instead of from
// firmware ONCE at boot.
//
// This sidesteps the harder problem entirely: gemv_banked_resident_vec.sv's
// MAC pipeline assumes weight reads are ALWAYS ready (no backpressure --
// weight_bank_tdp is a fixed 1-cycle-latency on-chip memory today, never
// stalls). Making the MAC's read port itself DMA-backed would need real
// surgery on that timing-critical pipeline (RLAT-deep, addend/accumulate
// staged for a closed 5ns cone -- this session's own earlier notes flag DSP
// margin as already tight). Loading a window of weights into the SAME
// resident weight_bank_tdp BEFORE compute starts, then letting the existing
// MAC pipeline run against it exactly as today, needs no such surgery: by
// the time a GEMV `start` fires, the window is already fully resident, same
// "always ready" assumption weight_bank_tdp already satisfies. The tradeoff
// is single-buffered (compute must wait for the window to finish loading,
// no double-buffered overlap yet) -- correctness first, matching every
// other DMA piece built so far in this port; double-buffering is future
// work once this baseline is proven.
//
// weight_bank_tdp's own write-word assembler (SUBW=WBITS/32 chunks/word,
// unmodified) is reused verbatim: this loader just needs to present a
// sequential stream of ld_words 32-bit chunks via w_we/w_data, exactly the
// shape firmware's own kevgpt_weight_load_word() loop already produces one
// register write at a time -- this module produces the same pulse sequence
// in hardware from DMA beats instead.
//
// DMA beats are DATA_W=256 bits = 8x 32-bit chunks; ld_words is assumed a
// multiple of 8 (asserted in sim, not handled generally) -- true for every
// weight image this project has ever produced (write_mems_wideword's own
// wide-word packing is already 256-bit-beat-aligned by construction).
//
// One outstanding DMA request at a time is NOT how this is built --
// requests are issued as fast as rd_req_ready allows (mig_read_engine's own
// credit system throttles correctly), decoupled from how fast beats are
// drained/unpacked, so the round-trip latency is paid ONCE (pipeline fill),
// not once per beat -- unlike kv_bank_ddr.sv's read side, which is
// low-frequency enough that sequential-only was an acceptable simplification;
// a weight-window reload happens once per layer per token, so throughput
// here matters for the cycle budget in a way KV reads did not.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module weight_loader_ddr #(
    parameter integer ADDR_W = 29,
    parameter integer DATA_W = 256
) (
    input  wire        clk,
    input  wire        rst,

    // ---- trigger: load ld_words 32-bit chunks starting at ld_ddr_addr ------
    input  wire                  ld_start,      // pulse; selectors sampled here
    input  wire [ADDR_W-1:0]     ld_ddr_addr,   // byte address, beat-aligned
    input  wire [31:0]           ld_words,      // 32-bit chunks to stream (multiple of DATA_W/32)
    output reg                   ld_done,       // pulses when the last chunk commits

    // ---- drives weight_bank_tdp's existing boot-load port verbatim --------
    output reg                   wb_ld_rst,
    output reg                   wb_w_we,
    output reg  [31:0]           wb_w_data,

    // ---- DMA read-request port: wires straight to a mig_read_engine's
    // req_*/ret_* ports ---------------------------------------------------------
    output wire                  rd_req_valid,
    input  wire                  rd_req_ready,
    output wire [ADDR_W-1:0]     rd_req_addr,
    input  wire                  rd_ret_valid,
    output wire                  rd_ret_ready,
    input  wire [DATA_W-1:0]     rd_ret_data
);
    localparam integer SUBW      = DATA_W / 32;         // 32-bit chunks per DMA beat
    localparam integer BEAT_BYTES = DATA_W / 8;
    localparam integer CNT_W     = 32;                  // beat/chunk counters

    localparam [0:0] LD_IDLE = 1'b0, LD_RUN = 1'b1;
    reg ldst;

    reg [ADDR_W-1:0]  ld_base;
    reg [CNT_W-1:0]   total_beats;
    reg [CNT_W-1:0]   issue_cnt;    // beats requested so far
    reg [CNT_W-1:0]   drain_cnt;    // 32-bit chunks committed so far

    reg [DATA_W-1:0]  beat_buf;
    reg               beat_have;
    reg [$clog2(SUBW)-1:0] sub_i;

    // ---- issue side: request beats as fast as the read engine allows,
    // decoupled from the drain side below (this is what lets round-trip
    // latency be paid once, not once per beat). ------------------------------
    wire issuing = (ldst == LD_RUN) && (issue_cnt < total_beats);
    assign rd_req_valid = issuing;
    assign rd_req_addr  = ld_base + issue_cnt * BEAT_BYTES;

    always @(posedge clk) begin
        if (rst) begin
            issue_cnt <= {CNT_W{1'b0}};
        end else if (ldst == LD_IDLE) begin
            if (ld_start) issue_cnt <= {CNT_W{1'b0}};
        end else if (rd_req_valid && rd_req_ready) begin
            issue_cnt <= issue_cnt + 1'b1;
        end
    end

    // ---- drain side: pop returned beats (mig_read_engine's own FIFO holds
    // them until we're ready), unpack SUBW 32-bit chunks per beat into
    // sequential w_we/w_data pulses feeding weight_bank_tdp's existing
    // assembler. ---------------------------------------------------------------
    assign rd_ret_ready = (ldst == LD_RUN) && !beat_have;

    always @(posedge clk) begin
        wb_w_we   <= 1'b0;
        wb_ld_rst <= 1'b0;
        ld_done   <= 1'b0;
        if (rst) begin
            ldst <= LD_IDLE;
            beat_have <= 1'b0;
        end else begin
            case (ldst)
                LD_IDLE: if (ld_start) begin
                    ld_base     <= ld_ddr_addr;
                    total_beats <= ld_words[CNT_W-1:$clog2(SUBW)];  // ld_words / SUBW
                    drain_cnt   <= {CNT_W{1'b0}};
                    beat_have   <= 1'b0;
                    sub_i       <= {($clog2(SUBW)){1'b0}};
                    wb_ld_rst   <= 1'b1;  // reset weight_bank_tdp's write pointer to row 0
                    ldst        <= LD_RUN;
                end
                LD_RUN: begin
                    if (!beat_have) begin
                        if (rd_ret_valid) begin
                            beat_buf  <= rd_ret_data;
                            beat_have <= 1'b1;
                            sub_i     <= {($clog2(SUBW)){1'b0}};
                        end
                    end else begin
                        wb_w_data <= beat_buf[sub_i*32 +: 32];
                        wb_w_we   <= 1'b1;
                        drain_cnt <= drain_cnt + 1'b1;
                        if (sub_i == SUBW[$clog2(SUBW)-1:0] - 1'b1) begin
                            beat_have <= 1'b0;
                        end else begin
                            sub_i <= sub_i + 1'b1;
                        end
                        if (drain_cnt == ld_words - 1'b1) begin
                            ld_done <= 1'b1;
                            ldst    <= LD_IDLE;
                        end
                    end
                end
                default: ldst <= LD_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        if ((SUBW & (SUBW - 1)) != 0)
            $display("weight_loader_ddr: WARNING SUBW=%0d is not a power of 2 -- the ld_words[CNT_W-1:$clog2(SUBW)] beat-count slice above assumes it is", SUBW);
    end
    always @(posedge clk) begin
        if (!rst && ldst == LD_IDLE && ld_start && (ld_words % SUBW != 0))
            $display("weight_loader_ddr: WARNING ld_words=%0d is not a multiple of SUBW=%0d -- the last beat's extra chunks would be silently committed as garbage words", ld_words, SUBW);
    end
`endif
endmodule
