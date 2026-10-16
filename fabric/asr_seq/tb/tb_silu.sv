// Testbench for silu_lut: stream Q4.12 inputs from x.mem, dump y.out for the
// Python bit-true compare (run_silu.py). Latency is resolved on the Python
// side. Structure identical to fabric/stage3/tb/tb_gelu.sv.
`timescale 1ns / 1ps
`ifndef N
 `define N 2048
`endif

module tb;
    localparam integer N = `N;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg signed [15:0] x;
    wire signed [15:0] y;
    reg signed [15:0] xin [0:N-1];
    integer i, f;

    silu_lut dut (.clk(clk), .x(x), .y(y));

    initial begin
        $readmemh("x.mem", xin);
        x = 0;
        @(posedge clk); #1;
        f = $fopen("y.out", "w");
        for (i = 0; i < N + 6; i = i + 1) begin
            x = (i < N) ? xin[i] : 16'sd0;
            @(posedge clk); #1;
            $fwrite(f, "%04x\n", y[15:0]);          // one y per cycle; align in Python
        end
        $fclose(f);
        $display("TB_DONE N=%0d", N);
        $finish;
    end
endmodule
