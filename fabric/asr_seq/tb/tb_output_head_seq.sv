// tb_output_head_seq -- functional gate for output_head_seq.sv: loads the
// real tied lm_head weight image (INT8, PER-ROW quantized per
// pack_output_head.py's own documented scheme), the real final-decoder-
// LayerNorm gamma, and the real per-row (mant,exp) dequant table ONCE,
// then for each of NSTEPS real decode steps: writes that step's own real
// (pre-final-norm) decoder hidden state, pulses `go`, and checks the
// resulting argmax index against pack_output_head.py's own golden --
// bit-exact.
`timescale 1ns / 1ps
`ifndef NWORDS
 `define NWORDS 73728
`endif
`ifndef NSTEPS
 `define NSTEPS 3
`endif

module tb;
    localparam integer P     = 8;
    localparam integer D     = 288;
    localparam integer VOCAB = 32768;
    localparam integer ROWS_D     = D/P;
    localparam integer ROWS_VOCAB = VOCAB/P;
    localparam integer LANES = 128;
    localparam integer WBW   = 8;
    localparam integer WBITS = LANES*WBW;
    localparam integer SUBW  = WBITS/32;
    localparam integer NWORDS = `NWORDS;
    localparam integer NSTEPS = `NSTEPS;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    reg go; wire done;
    wire [$clog2(VOCAB)-1:0] argmax_idx;
    wire signed [31:0] argmax_val;
    reg [19:0] wb_lm;
    reg xres_wr; reg [$clog2(D/P)-1:0] xres_waddr; reg [P*32-1:0] xres_wdata;
    reg gv_ld_rst, gv_ld_we; reg [31:0] gv_ld_data;
    reg gam_we; reg [$clog2(D/P)-1:0] gam_waddr; reg [P*32-1:0] gam_wdata;
    reg dq_we; reg [$clog2(VOCAB/P)-1:0] dq_waddr; reg [P*24-1:0] dq_wmant; reg [P*8-1:0] dq_wexp;

    output_head_seq #(.P(P), .D(D), .VOCAB(VOCAB), .ACT_LM(19), .DQ_FRAC(0)) dut (
        .clk(clk), .rst(rst), .go(go), .done(done),
        .argmax_idx(argmax_idx), .argmax_val(argmax_val),
        .wb_lm(wb_lm),
        .xres_wr(xres_wr), .xres_waddr(xres_waddr), .xres_wdata(xres_wdata),
        .gv_ld_rst(gv_ld_rst), .gv_ld_we(gv_ld_we), .gv_ld_data(gv_ld_data),
        .gam_we(gam_we), .gam_waddr(gam_waddr), .gam_wdata(gam_wdata),
        .dq_we(dq_we), .dq_waddr(dq_waddr), .dq_wmant(dq_wmant), .dq_wexp(dq_wexp)
    );

    reg [WBITS-1:0] wload [0:NWORDS-1];
    reg [31:0] gammalnf [0:ROWS_D*P-1];
    reg [P*24-1:0] dqmant [0:ROWS_VOCAB-1];
    reg [P*8-1:0]  dqexp  [0:ROWS_VOCAB-1];
    reg [31:0] xresin [0:NSTEPS*ROWS_D*P-1];
    reg [15:0] argmaxref [0:NSTEPS-1];

    integer i, s, l, mism, checked, st_i;
    reg [WBITS-1:0] word_tmp;
    reg [P*32-1:0] rowbuf;

    // watchdog: force a diagnostic dump + finish if `done` hasn't fired
    // within a generous margin of one output-head pass (a 32768-row GEMV
    // takes real time to simulate -- 256 groups x ~288 MAC cycles each --
    // so this watchdog is sized generously wider than the decoder/encoder
    // gates' own).
    reg dbg_armed;
    reg [31:0] watchdog_cnt;
    always @(posedge clk) begin
        if (dbg_armed && !done) watchdog_cnt <= watchdog_cnt + 1;
        else watchdog_cnt <= 0;
        if (dbg_armed && watchdog_cnt == 32'd200000) begin
            $display("TB_WATCHDOG_TIMEOUT,t=%0t,st=%0d", $time, dut.st);
            $finish;
        end
    end

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("gamma_lnf.mem", gammalnf);
        $readmemh("dq_mant.mem", dqmant);
        $readmemh("dq_exp.mem", dqexp);
        $readmemh("xres_in.mem", xresin);
        $readmemh("argmax_ref.mem", argmaxref);

        go=0; wb_lm=0; xres_wr=0; xres_waddr=0; xres_wdata=0;
        gv_ld_rst=0; gv_ld_we=0; gv_ld_data=0;
        gam_we=0; gam_waddr=0; gam_wdata=0;
        dq_we=0; dq_waddr=0; dq_wmant=0; dq_wexp=0;

        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ---- 1. load GEMV resident weight image (once) ----
        gv_ld_rst = 1; @(posedge clk); #1; gv_ld_rst = 0;
        for (i = 0; i < NWORDS; i = i + 1) begin
            word_tmp = wload[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                gv_ld_we = 1; gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        gv_ld_we = 0;
        $display("TB_WEIGHTS_LOADED,nwords=%0d", NWORDS);

        // ---- 2. load final-LN gamma (once) ----
        for (i = 0; i < ROWS_D; i = i + 1) begin
            for (l = 0; l < P; l = l + 1) rowbuf[l*32 +: 32] = gammalnf[i*P+l];
            gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
            @(posedge clk); #1;
        end
        gam_we = 0;
        $display("TB_GAMMA_LOADED");

        // ---- 3. load per-row (mant,exp) dequant table (once) ----
        for (i = 0; i < ROWS_VOCAB; i = i + 1) begin
            dq_waddr = i[$clog2(VOCAB/P)-1:0];
            dq_wmant = dqmant[i]; dq_wexp = dqexp[i]; dq_we = 1;
            @(posedge clk); #1;
        end
        dq_we = 0;
        $display("TB_DQTABLE_LOADED");

        // ---- 4. run NSTEPS output-head passes, checking argmax each time ----
        mism = 0; checked = 0;
        wb_lm = 20'd0;
        for (st_i = 0; st_i < NSTEPS; st_i = st_i + 1) begin
            for (i = 0; i < ROWS_D; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = xresin[st_i*ROWS_D*P + i*P + l];
                xres_waddr = i[$clog2(D/P)-1:0]; xres_wdata = rowbuf; xres_wr = 1;
                @(posedge clk); #1;
            end
            xres_wr = 0;
            $display("TB_XRESIN_LOADED,step=%0d", st_i);

            dbg_armed = 1'b1;
            go = 1; @(posedge clk); #1; go = 0;
            while (!done) @(posedge clk); #1;
            dbg_armed = 1'b0;
            $display("TB_STEP_DONE,step=%0d,argmax_idx=%0d,argmax_val=%0d",
                      st_i, argmax_idx, $signed(argmax_val));

            checked = checked + 1;
            if (argmax_idx !== argmaxref[st_i]) begin
                mism = mism + 1;
                $display("MISMATCH,step=%0d,got=%0d,ref=%0d", st_i, argmax_idx, argmaxref[st_i]);
            end
        end

        $display("TB_DONE,checked=%0d,mismatches=%0d", checked, mism);
        $display("OUTPUT_HEAD_SEQ_VERDICT,bitexact=%0d,mismatches=%0d,checked=%0d,nsteps=%0d",
                  (mism == 0), mism, checked, NSTEPS);
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
