// Testbench for vec_tanh: stream P-wide Q4.12 vectors from xin.mem (one
// packed 16*P-bit word per line, lane k in bits [16*k +: 16]), assert
// in_valid, and dump y.out (one packed 16*P-bit hex word per cycle when
// out_valid is high). The Python gate (run_vec_tanh.py) compares against
// tanh_q lane-by-lane. Structure identical to
// fabric/asr_seq/tb/tb_vec_silu.sv.
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
    localparam integer W = 16 * P;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg               in_valid;
    reg  [W-1:0]      x;
    wire              out_valid;
    wire [W-1:0]      y;

    reg  [W-1:0]      xin [0:N-1];
    integer i, f;

    vec_tanh #(.P(P)) dut (
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
