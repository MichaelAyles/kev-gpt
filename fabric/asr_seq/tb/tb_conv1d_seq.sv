// Generic testbench for conv1d_seq.sv -- loads the resident weight image
// (w.mem), the whole padded input tensor (xt.mem, INT8, t-major/channel-
// minor), optionally the bias table (b.mem, only if HASBIAS), pulses `go`
// once (computes ALL TOUT output positions internally), and dumps y.out
// (one P*32-bit hex word per y_valid cycle) for the Python bit-true compare
// (run_conv1.py/run_conv2.py/run_conv3.py against their own pack_conv*.py
// golden). Structure mirrors tb_encoder_block_seq.sv's own weight-load
// procedure.
`timescale 1ns / 1ps
`ifndef PVAL
 `define PVAL 8
`endif
`ifndef CINVAL
 `define CINVAL 8
`endif
`ifndef COUTVAL
 `define COUTVAL 288
`endif
`ifndef KWVAL
 `define KWVAL 127
`endif
`ifndef STRIDEVAL
 `define STRIDEVAL 64
`endif
`ifndef TINVAL
 `define TINVAL 3000
`endif
`ifndef HASBIAS
 `define HASBIAS 0
`endif
`ifndef NWORDS
 `define NWORDS 1
`endif
`ifndef WWORDSVAL
 `define WWORDSVAL 1024
`endif

module tb;
    localparam integer P      = `PVAL;
    localparam integer CIN    = `CINVAL;
    localparam integer COUT   = `COUTVAL;
    localparam integer KW     = `KWVAL;
    localparam integer STRIDE = `STRIDEVAL;
    localparam integer TIN    = `TINVAL;
    localparam integer HASBIAS = `HASBIAS;
    localparam integer LANES  = 128;
    localparam integer WBW    = 8;
    localparam integer WBITS  = LANES*WBW;
    localparam integer SUBW   = WBITS/32;
    localparam integer NWORDS = `NWORDS;
    localparam integer WWORDS = `WWORDSVAL;
    localparam integer CGRP   = CIN/P;
    localparam integer MROWS  = COUT/P;
    localparam integer XTROWS = TIN*CGRP;
    localparam integer TOUT   = (TIN - KW) / STRIDE + 1;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst = 1'b1;

    reg gv_ld_rst, gv_ld_we; reg [31:0] gv_ld_data;
    reg dq_we; reg [P*24-1:0] dq_wmant; reg [P*8-1:0] dq_wexp;
    reg b_we; reg [P*32-1:0] b_data;
    reg xt_we; reg [P*8-1:0] xt_data;
    reg go; wire done;
    wire yv; wire [P*32-1:0] ydata;

    conv1d_seq #(.P(P), .WBW(WBW), .CIN(CIN), .COUT(COUT), .KW(KW), .STRIDE(STRIDE),
                 .TIN(TIN), .HAS_BIAS(HASBIAS), .LANES(LANES), .WWORDS(WWORDS)) dut (
        .clk(clk), .rst(rst),
        .gv_ld_rst(gv_ld_rst), .gv_ld_we(gv_ld_we), .gv_ld_data(gv_ld_data),
        .dq_we(dq_we), .dq_wmant(dq_wmant), .dq_wexp(dq_wexp),
        .b_we(b_we), .b_data(b_data),
        .xt_we(xt_we), .xt_data(xt_data),
        .go(go), .done(done), .y_valid(yv), .y_data(ydata)
    );

    reg [WBITS-1:0]  wload [0:NWORDS-1];
    reg [P*8-1:0]    xtload [0:XTROWS-1];
    reg [P*32-1:0]   bload [0:(MROWS>0?MROWS-1:0)];
    reg [P*24-1:0]   dqmload [0:(MROWS>0?MROWS-1:0)];
    reg [P*8-1:0]    dqeload [0:(MROWS>0?MROWS-1:0)];
    integer i, s, f;
    reg [WBITS-1:0] word_tmp;

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("xt.mem", xtload);
        $readmemh("dq_mant.mem", dqmload);
        $readmemh("dq_exp.mem", dqeload);
        if (HASBIAS) $readmemh("b.mem", bload);

        gv_ld_rst=0; gv_ld_we=0; gv_ld_data=0;
        dq_we=0; dq_wmant=0; dq_wexp=0;
        b_we=0; b_data=0; xt_we=0; xt_data=0;
        go = 0;

        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        gv_ld_rst = 1; @(posedge clk); #1; gv_ld_rst = 0;
        for (i = 0; i < NWORDS; i = i + 1) begin
            word_tmp = wload[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                gv_ld_we = 1; gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        gv_ld_we = 0;

        for (i = 0; i < MROWS; i = i + 1) begin
            dq_we = 1; dq_wmant = dqmload[i]; dq_wexp = dqeload[i]; @(posedge clk); #1;
        end
        dq_we = 0;

        if (HASBIAS) begin
            for (i = 0; i < MROWS; i = i + 1) begin
                b_we = 1; b_data = bload[i]; @(posedge clk); #1;
            end
            b_we = 0;
        end

        for (i = 0; i < XTROWS; i = i + 1) begin
            xt_we = 1; xt_data = xtload[i]; @(posedge clk); #1;
        end
        xt_we = 0;

        f = $fopen("y.out", "w");
        go = 1; @(posedge clk); #1; go = 0;

        i = 0;
        while (i < TOUT*MROWS) begin
            if (yv) begin
                $fwrite(f, "%0x\n", ydata);
                i = i + 1;
            end
            @(posedge clk); #1;
        end
        $fclose(f);

        $display("TB_DONE");
        $finish;
    end

    initial begin #400000000; $display("TB_TIMEOUT"); $finish; end
endmodule
