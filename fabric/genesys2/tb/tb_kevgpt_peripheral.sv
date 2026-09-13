// -----------------------------------------------------------------------------
// tb_kevgpt_peripheral — protocol-conformance testbench for
// xheep_kevgpt_peripheral.sv. Checks that a handful of reg_req_t/reg_rsp_t
// transactions (IDCODE read, STATUS before/after go, a full go..done round
// trip, CYCLES readback) reach the same register semantics gemv_axi_seq_vec.v
// exposed over AXI4-Lite. This is a bus-protocol check, NOT a model-
// correctness check -- no weights are loaded, so tok_out is not compared
// against anything; the fabric/stage3/run_vec_kv.py gate (unmodified by this
// port) already covers model correctness against model.goformer_kvq.
//
// reg_pkg is redefined locally here (matching X-HEEP's own
// hw/core-v-mini-mcu/include/reg_pkg.sv byte-for-byte) since kev-gpt does not
// vendor X-HEEP; once synced into the SoC repo (Phase 4), the real reg_pkg.sv
// supersedes this local copy -- xheep_kevgpt_peripheral.sv's reg_req_t/
// reg_rsp_t are type parameters for exactly this reason.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

package reg_pkg;
    typedef struct packed {
        logic        valid;
        logic        write;
        logic [3:0]  wstrb;
        logic [31:0] addr;
        logic [31:0] wdata;
    } reg_req_t;

    typedef struct packed {
        logic        error;
        logic        ready;
        logic [31:0] rdata;
    } reg_rsp_t;
endpackage

module tb_kevgpt_peripheral;
    import reg_pkg::*;

    reg clk = 0;
    reg rst_ni = 0;
    always #5 clk = ~clk;

    reg_req_t req;
    reg_rsp_t rsp;

    integer errors = 0;
    integer cycles;

    task automatic reg_write(input [7:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            req.valid = 1; req.write = 1; req.wstrb = 4'hF;
            req.addr = {24'b0, addr}; req.wdata = data;
            @(posedge clk); #1;
            if (!rsp.ready) begin
                $display("FAIL write addr=%0h: ready did not assert same cycle", addr);
                errors = errors + 1;
            end
            @(negedge clk);
            req.valid = 0; req.write = 0;
        end
    endtask

    task automatic reg_read(input [7:0] addr, output [31:0] data);
        begin
            @(negedge clk);
            req.valid = 1; req.write = 0; req.addr = {24'b0, addr};
            @(posedge clk); #1;
            if (!rsp.ready) begin
                $display("FAIL read addr=%0h: ready did not assert same cycle", addr);
                errors = errors + 1;
            end
            data = rsp.rdata;
            @(negedge clk);
            req.valid = 0;
        end
    endtask

    reg [31:0] rd;

    xheep_kevgpt_peripheral #(.P(16), .LANES(128), .NLAYER(4), .WWORDS(262144), .TMAX(64),
                               .MEM_PRIMITIVE("block")) dut (
        .clk_i(clk), .rst_ni(rst_ni), .reg_req_i(req), .reg_rsp_o(rsp)
    );

    initial begin
        req = '0;
        repeat (4) @(negedge clk);
        rst_ni = 1;
        repeat (2) @(negedge clk);

        // 1. IDCODE readback ("SQRV" 0x53515256), same word map as the AXI shell.
        reg_read(8'h2C, rd);
        if (rd !== 32'h5351_5256) begin
            $display("FAIL IDCODE: got %08h want 53515256", rd);
            errors = errors + 1;
        end else $display("PASS IDCODE = %08h", rd);

        // 2. STATUS before go: busy=0, done=0.
        reg_read(8'h04, rd);
        if (rd[1:0] !== 2'b00) begin
            $display("FAIL STATUS pre-go: got %02b want 00 (busy,done)", rd[1:0]);
            errors = errors + 1;
        end else $display("PASS STATUS pre-go = %02b", rd[1:0]);

        // 3. CTRL.wl_rst pulse (bit1) -- exercises the write-decode path once
        //    before go, matching the documented boot sequence (wl_rst before
        //    streaming weights). No weights are streamed in this protocol test.
        reg_write(8'h00, 32'h0000_0002);

        // 4. CTRL.go (bit0) -> a full go..done round trip through the register
        //    interface into sequencer_vec and back, with NO weights loaded
        //    (values are garbage; only the handshake is checked).
        reg_write(8'h00, 32'h0000_0001);
        reg_read(8'h04, rd);
        if (rd[1] !== 1'b1) begin
            $display("FAIL STATUS post-go: busy did not assert (got %02b)", rd[1:0]);
            errors = errors + 1;
        end else $display("PASS STATUS post-go busy=1");

        cycles = 0;
        while (rd[1] === 1'b1 && cycles < 200000) begin
            reg_read(8'h04, rd);
            cycles = cycles + 1;
        end
        if (cycles >= 200000) begin
            $display("FAIL: STATUS.busy never cleared within 200000 poll iterations");
            errors = errors + 1;
        end else if (rd[0] !== 1'b1) begin
            $display("FAIL: busy cleared but STATUS.done=0 (got %02b)", rd[1:0]);
            errors = errors + 1;
        end else $display("PASS go..done round trip completed after %0d polls", cycles);

        // 5. CYCLES readback should be nonzero once a run has completed.
        reg_read(8'h28, rd);
        if (rd == 32'b0) begin
            $display("FAIL CYCLES: read back 0 after a completed run");
            errors = errors + 1;
        end else $display("PASS CYCLES = %0d", rd);

        if (errors == 0) $display("KEVGPT_PERIPH_VERDICT pass=True errors=0");
        else $display("KEVGPT_PERIPH_VERDICT pass=False errors=%0d", errors);
        $finish;
    end
endmodule
