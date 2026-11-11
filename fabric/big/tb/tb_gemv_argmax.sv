// tb_gemv_argmax — unit test of gemv_argmax (rtl/gemv_axi.v): the engine never produces
// -0, so the "-0 == +0, lowest index wins" rule is tested here on directed vectors.
//
// +in=<file> ($readmemh, 32-bit words): a list of tests, each  <n> <gaps> <n result words>.
// gaps = 0: the rows arrive in consecutive cycles; else 0..3 idle cycles between rows.
// For every test the module is cleared, the rows are fed, and idx / val are sampled TWO
// cycles after the last row (the latency the wrapper relies on).  +out: "<idx> <val hex>".
`timescale 1ns / 1ps

module tb_gemv_argmax;
    reg clk = 0;
    always #1.5 clk = ~clk;
    reg         clr = 1, v = 0;
    reg  [31:0] d = 0;
    wire [31:0] idx, val;
    gemv_argmax dut (.clk(clk), .clr(clr), .v(v), .d(d), .idx(idx), .val(val));

    reg [31:0]   mem [0:(1<<20)-1];
    reg [1023:0] inf, outf;
    integer      nw, p, n, gaps, i, fout, seed, ntests;

    initial begin
        if (!$value$plusargs("in=%s", inf))   begin $display("need +in"); $finish; end
        if (!$value$plusargs("out=%s", outf)) begin $display("need +out"); $finish; end
        if (!$value$plusargs("nw=%d", nw))    nw = 0;
        $readmemh(inf, mem, 0, nw - 1);
        fout = $fopen(outf, "w");
        p = 0; ntests = 0; seed = 12345;
        repeat (4) @(posedge clk);
        while (p < nw) begin
            n = mem[p]; gaps = mem[p + 1]; p = p + 2;
            clr <= 1'b1; v <= 1'b0;
            @(posedge clk);
            clr <= 1'b0;
            @(posedge clk);
            for (i = 0; i < n; i = i + 1) begin
                v <= 1'b1; d <= mem[p + i];
                @(posedge clk);
                if (gaps != 0) begin
                    v <= 1'b0; d <= 32'hxxxxxxxx;
                    repeat (($random(seed) & 32'h7fffffff) % 4) @(posedge clk);
                end
            end
            v <= 1'b0; d <= 32'hxxxxxxxx;      // the last row was sampled at the edge just passed (end of
                                               // cycle c); the key stage is valid in c + 1
            @(posedge clk);                    // end of c + 1: idx / val take the answer ...
            #0.1 $fdisplay(fout, "%0d %08x", idx, val);     // ... and hold it from the start of c + 2
            p = p + n; ntests = ntests + 1;
        end
        $display("ARGMAX_TB_DONE tests=%0d", ntests);
        $fclose(fout);
        $finish;
    end
endmodule
