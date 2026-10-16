// Testbench for gelu_wide_vec: stream P-wide 32-bit Q4.12-scaled vectors
// from xin.mem, dump y.out (one packed 32*P-bit hex word per out_valid
// cycle) for the Python gate (run_gelu_wide.py) to compare lane-by-lane
// against its own gelu_wide_q412 reference. Structure mirrors
// fabric/asr_seq/tb/tb_vec_tanh.sv.
`timescale 1ns / 1ps
`ifndef N
 `define N 1024
`endif
`ifndef P
 `define P 8
`endif

module tb;
    localparam integer N = `N;
    localparam integer P = `P;
    localparam integer W = 32 * P;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg               in_valid;
    reg  [W-1:0]      x;
    wire              out_valid;
    wire [W-1:0]      y;

    reg  [W-1:0]      xin [0:N-1];
    integer i, f;

    gelu_wide_vec #(.P(P)) dut (
        .clk(clk), .in_valid(in_valid), .x(x),
        .out_valid(out_valid), .y(y)
    );

    initial begin
        $readmemh("xin.mem", xin);
        x = {W{1'b0}};
        in_valid = 1'b0;
        @(posedge clk); #1;
        f = $fopen("y.out", "w");
        for (i = 0; i < N + 6; i = i + 1) begin
            x        = (i < N) ? xin[i] : {W{1'b0}};
            in_valid = (i < N) ? 1'b1   : 1'b0;
            @(posedge clk); #1;
            if (out_valid)
                $fwrite(f, "%0x\n", y);
        end
        $fclose(f);
        $display("TB_DONE N=%0d P=%0d", N, P);
        $finish;
    end
endmodule
