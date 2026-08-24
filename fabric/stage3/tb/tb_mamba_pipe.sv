// tb_mamba_pipe — drives the pipelined NC-stream mamba engine. Loads every table
// (ms_t<sel>.mem, sizes from ms_cfg.mem) + the per-(stream,token) input tokens
// (ms_tok.mem, NC*T entries, stream-major), runs ONE start, and after done dumps
// each stream's per-token logits + final residual (mp_logit.out / mp_x.out, in
// reshape(NC,T,*) order) and prints the MEASURED cycle count (TOTCYC).
`timescale 1ns/1ps

`ifndef NST_SIM
`define NST_SIM 64
`endif
`ifndef QH_SIM
`define QH_SIM 16
`endif
module tb_mamba_pipe;
    localparam int D = 256, DIN = 512, V = 1024;
    localparam int NC  = `NC_SIM;
    localparam int T   = `T_SIM;
    localparam int NST = `NST_SIM;    // d_state: 64 (config-A) / 32 (config-B)
    localparam int QH  = `QH_SIM;     // scan-h stored bits (16=INT16, 12=INT12)

    reg clk = 0, rst = 1, start = 0;
    wire ready, done;
    wire [31:0] cyc_count;
    reg        wr_en = 0;
    reg  [3:0] wr_sel;
    reg  [18:0] wr_addr;
    reg  [31:0] wr_data;
    reg        tw_en = 0;
    reg  [$clog2(NC*T)-1:0] tw_addr;
    reg  [9:0] tw_data;
    reg  [3:0] dbg_sel = 0;
    reg  [18:0] dbg_addr = 0;
    wire signed [31:0] dbg_data;

    mamba_pipe #(.NC(NC), .NST(NST), .QH(QH), .TMAX(T), .T_TOKENS(T)) dut (
        .clk(clk), .rst(rst), .ready(ready), .start(start), .done(done),
        .cyc_count(cyc_count),
        .wr_en(wr_en), .wr_sel(wr_sel), .wr_addr(wr_addr), .wr_data(wr_data),
        .tw_en(tw_en), .tw_addr(tw_addr), .tw_data(tw_data),
        .dbg_sel(dbg_sel), .dbg_addr(dbg_addr), .dbg_data(dbg_data));
    always #2 clk = ~clk;

    reg [31:0] tbl [0:524287];
    reg [31:0] cfg [0:31];
    reg [15:0] toks [0:1023];
    integer i, s, t, fd, fdx;
    integer j;

    task load_sel(input integer sel, input integer count, input [1023:0] fname);
        begin
            $readmemh(fname, tbl, 0, count-1);
            for (i = 0; i < count; i = i + 1) begin
                @(posedge clk);
                wr_en <= 1; wr_sel <= sel[3:0]; wr_addr <= i[18:0];
                wr_data <= tbl[i];
            end
            @(posedge clk); wr_en <= 0;
        end
    endtask

    // progress beacon
    integer wd; integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;
    initial begin
        wd = 0;
        forever begin
            repeat (100000) @(posedge clk);
            wd = wd + 1;
            $display("BEAT %0d cyc=%0d est=%0d nst=%0d gst=%0d cst=%0d sst=%0d pc0=%0d pc1=%0d act0=%0d act1=%0d",
                     wd, cyc_count, dut.est, dut.nst, dut.gst, dut.cst, dut.sst,
                     dut.op_pc[0], (NC>1)?dut.op_pc[1]:0,
                     dut.active[0], (NC>1)?dut.active[1]:0);
            if (wd > 60) begin $display("TB_MS_HUNG"); $finish; end
        end
    end

    initial begin
        $readmemh("ms_cfg.mem", cfg);
        $readmemh("ms_tok.mem", toks, 0, NC*T-1);
        fd  = $fopen("mp_logit.out", "w");
        fdx = $fopen("mp_x.out", "w");

        repeat (4) @(posedge clk);
        rst <= 0;
        wait (ready);
        $display("READY %0t", $time);

        load_sel(1,  cfg[1],  "ms_t1.mem");
        load_sel(0,  cfg[15], "ms_t0.mem");
        load_sel(1,  cfg[1],  "ms_t1.mem");
        load_sel(2,  cfg[2],  "ms_t2.mem");
        load_sel(3,  cfg[3],  "ms_t3.mem");
        load_sel(4,  cfg[4],  "ms_t4.mem");
        load_sel(5,  cfg[5],  "ms_t5.mem");
        load_sel(6,  cfg[6],  "ms_t6.mem");
        load_sel(7,  cfg[7],  "ms_t7.mem");
        load_sel(8,  cfg[8],  "ms_t8.mem");
        load_sel(9,  cfg[9],  "ms_t9.mem");
        load_sel(10, cfg[10], "ms_t10.mem");
        load_sel(11, cfg[11], "ms_t11.mem");
        load_sel(12, cfg[12], "ms_t12.mem");
        load_sel(13, cfg[13], "ms_t13.mem");
        load_sel(14, cfg[14], "ms_t14.mem");
        load_sel(15, cfg[17], "ms_t15.mem");   // rsqrt seed (WSEL_SEED), cnt cfg[17]
        $display("TABLES_LOADED %0t", $time);

        // load per-(stream,token) input tokens (stream-major)
        for (i = 0; i < NC*T; i = i + 1) begin
            @(posedge clk);
            tw_en <= 1; tw_addr <= i[$clog2(NC*T)-1:0]; tw_data <= toks[i][9:0];
        end
        @(posedge clk); tw_en <= 0;

        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;
        wait (done);
        @(posedge clk);
        $display("TOTCYC %0d", cyc_count);

        // dump per (stream,token): V logits then D residual (reshape(NC,T,*))
        dbg_sel <= 1;
        for (s = 0; s < NC; s = s + 1)
          for (t = 0; t < T; t = t + 1)
            for (i = 0; i < V; i = i + 1) begin
                dbg_addr <= (s*T + t)*V + i;
                @(posedge clk); @(posedge clk);
                $fdisplay(fd, "%04x", dbg_data[15:0]);
            end
        dbg_sel <= 0;
        for (s = 0; s < NC; s = s + 1)
          for (t = 0; t < T; t = t + 1)
            for (i = 0; i < D; i = i + 1) begin
                dbg_addr <= (s*T + t)*D + i;
                @(posedge clk); @(posedge clk);
                $fdisplay(fdx, "%08x", dbg_data);
            end
        $fclose(fd); $fclose(fdx);
        dbg_sel <= 12; dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_xn=%0d", $signed(dbg_data));
        dbg_sel <= 13; dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_yb=%0d", $signed(dbg_data));
        dbg_sel <= 14; dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_zx=%0d", $signed(dbg_data));
        dbg_sel <= 15; dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_q8=%0d", $signed(dbg_data));
        dbg_sel <= 10; dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_ny=%0d", $signed(dbg_data));
        dbg_sel <= 11; dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_nout=%0d", $signed(dbg_data));
        // conv compute-vs-readback pair: sum_xna must equal c_ysum
        dbg_sel <= 6; dbg_addr <= (1<<18)|0; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_readback_all=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|1; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_written_all=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|2; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_xna=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|3; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM first_ysum=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|4; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_recv_w=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|5; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_recv_b=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|6; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_recv_x=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|7; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_recv_lut=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|8; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM conv_hist=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|9; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM scan_recv_dtx=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|10; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM scan_recv_B=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|11; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM scan_recv_C=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|12; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM zxw_cnt=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|13; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM zxw_asum=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|14; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM xnw_cnt=%0d", $signed(dbg_data));
        dbg_sel <= 6; dbg_addr <= (1<<18)|15; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM xnw_asum=%0d", $signed(dbg_data));
        // raw buffer rows covering the offsets whose reads came back ZERO
        for (i = 0; i < 8; i = i + 1) begin
            dbg_sel <= 6; dbg_addr <= (1<<18)|(1<<16)|((512/4)+i);
            @(posedge clk); @(posedge clk);
            $display("STAGE_ROW zx[%0d]lo=%0d", (512/4)+i, $signed(dbg_data));
            dbg_sel <= 6; dbg_addr <= (1<<18)|(1<<16)|(1<<14)|((512/4)+i);
            @(posedge clk); @(posedge clk);
            $display("STAGE_ROW zx[%0d]hi=%0d", (512/4)+i, $signed(dbg_data));
        end
        for (i = 0; i < 8; i = i + 1) begin
            dbg_sel <= 6; dbg_addr <= (1<<18)|(1<<16)|(1<<15)|((512/4)+i);
            @(posedge clk); @(posedge clk);
            $display("STAGE_ROW xn[%0d]lo=%0d", (512/4)+i, $signed(dbg_data));
        end
        // op-boundary snapshots: the whole layer-0 walk in one sweep.
        // sig 0=nout 1=q8 2=zx 3=xn 4=yb 5=xw 6=kind{n,s,c,g} 7=ev_cnt
        for (i = 0; i < 8; i = i + 1)
          for (j = 0; j < 8; j = j + 1) begin
            dbg_sel <= 3; dbg_addr <= (1<<18) | (j<<3) | i;
            @(posedge clk); @(posedge clk);
            $display("STAGE_SNAP ev=%0d sig=%0d val=%0d", i, j, $signed(dbg_data));
          end
        // per-table AXI load checksums
        for (i = 0; i < 16; i = i + 1) begin
            dbg_sel <= 8; dbg_addr <= (1<<18) | i;
            @(posedge clk); @(posedge clk);
            $display("STAGE_WSUM sel=%0d val=%0d", i, $signed(dbg_data));
        end
        dbg_addr <= 0;
        dbg_sel <= 7;  dbg_addr <= 0; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM xw=%0d", $signed(dbg_data));
        dbg_sel <= 6;  dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM xbuf0=%0d", $signed(dbg_data));
        dbg_addr <= 0;
        dbg_sel <= 0;  dbg_addr <= 0; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM ny=%0d", $signed(dbg_data));
        dbg_sel <= 2;  dbg_addr <= (1<<17); @(posedge clk); @(posedge clk);
        $display("STAGE_SUM ng=%0d", $signed(dbg_data));
        dbg_addr <= 0;
        dbg_sel <= 4;  @(posedge clk); @(posedge clk);
        $display("STAGE_SUM emb=%0d", $signed(dbg_data));
        dbg_sel <= 5;  @(posedge clk); @(posedge clk);
        $display("STAGE_SUM nout=%0d", $signed(dbg_data));
        dbg_sel <= 14; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM zx=%0d", $signed(dbg_data));
        dbg_sel <= 15; @(posedge clk); @(posedge clk);
        $display("STAGE_SUM xn=%0d", $signed(dbg_data));
        dbg_sel <= 3;  @(posedge clk); @(posedge clk);
        $display("STAGE_SUM q8=%0d", $signed(dbg_data));
        $display("TB_MS_DONE NC=%0d T=%0d", NC, T);
        $finish;
    end
endmodule
