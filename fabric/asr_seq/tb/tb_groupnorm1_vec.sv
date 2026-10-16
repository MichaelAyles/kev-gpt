// Testbench for groupnorm1_vec.sv: two-phase load (gamma/beta preload, CROWS
// cycles; then x stream, ROWS cycles), single real [C][T] tensor, dump y.out
// for the Python bit-true compare (run_groupnorm1.py / pack_groupnorm1.py's
// own gn_int reference). Structure mirrors tb_layernorm_gendiv.sv, split
// into the extra gamma/beta preload phase groupnorm1_vec.sv's own protocol
// needs (see that file's header).
`timescale 1ns / 1ps
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef CVAL
 `define CVAL 288
`endif
`ifndef TVAL
 `define TVAL 45
`endif
module tb;
    localparam integer P     = `PVAL;
    localparam integer C     = `CVAL;
    localparam integer T     = `TVAL;
    localparam integer CROWS = C / P;
    localparam integer ROWS  = (C * T) / P;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst, start, gvin, vin;
    reg [P*32-1:0] xin, gin, bin_;
    wire yv, done;
    wire [P*64-1:0] yout;

    groupnorm1_vec #(.P(P), .C(C), .T(T)) dut (
        .clk(clk), .rst(rst), .start(start),
        .gvalid_in(gvin), .gamma_in(gin), .beta_in(bin_),
        .valid_in(vin), .x_in(xin),
        .y_valid(yv), .y_out(yout), .done(done));

    reg [P*32-1:0] grows [0:CROWS-1];
    reg [P*32-1:0] brows [0:CROWS-1];
    reg [P*32-1:0] xrows [0:ROWS-1];
    integer r, f, k;
    reg [63:0] ylane;

    initial begin
        rst = 1'b1; start = 1'b0; gvin = 1'b0; vin = 1'b0;
        xin = 0; gin = 0; bin_ = 0;
        $readmemh("g.mem", grows);
        $readmemh("b.mem", brows);
        $readmemh("x.mem", xrows);
        @(posedge clk); #1; @(posedge clk); #1; rst = 1'b0; @(posedge clk); #1;
        f = $fopen("y.out", "w");

        start = 1'b1; @(posedge clk); #1; start = 1'b0;
        for (r = 0; r < CROWS; r = r + 1) begin
            gvin = 1'b1; gin = grows[r]; bin_ = brows[r];
            @(posedge clk); #1;
        end
        gvin = 1'b0;
        for (r = 0; r < ROWS; r = r + 1) begin
            vin = 1'b1; xin = xrows[r];
            @(posedge clk); #1;
        end
        vin = 1'b0;

        r = 0;
        while (r < ROWS) begin
            if (yv) begin
                for (k = 0; k < P; k = k + 1) begin
                    ylane = yout[k*64 +: 64];
                    $fwrite(f, "%016x\n", ylane);
                end
                r = r + 1;
            end
            @(posedge clk); #1;
        end

        $fclose(f);
        $display("TB_DONE");
        $finish;
    end

    initial begin #200000000; $display("TB_TIMEOUT"); $finish; end
endmodule
