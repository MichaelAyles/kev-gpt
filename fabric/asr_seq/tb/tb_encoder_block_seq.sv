// tb_encoder_block_seq -- the MULTI-LAYER functional gate for
// encoder_block_seq.sv: loads the real moonshine-tiny resident weight
// image (ALL NLAYER encoder layers' 6 GEMVs each, INT8-quantized per
// pack_encoder_block.py's own documented scheme) ONCE, writes the real
// per-position conv-front-end input (layer 0's own xres_in, ALL T2
// positions), then for each of NLAYER layers in order: reloads that
// layer's own gamma/bias tables and wb_*/gf_* runtime constants, pulses
// `go` (which internally processes ALL T2 positions -- write-phase then
// attend-phase, see encoder_block_seq.sv's own header), and checks every
// position's own residual output against pack_encoder_block.py's own
// per-(layer,position) golden xres_out -- bit-exact. Layer L's own RES2
// output sitting in xres_bank becomes layer L+1's own LN1 input for free
// (same physical memory across `go` pulses), no explicit hand-off needed.
`timescale 1ns / 1ps
`ifndef NWORDS
 `define NWORDS 57024
`endif
`ifndef NLAYER
 `define NLAYER 6
`endif

module tb;
    localparam integer P        = 8;
    localparam integer D        = 288;
    localparam integer FFN      = 1152;
    localparam integer NHEAD    = 8;
    localparam integer HEAD_DIM = 36;
    localparam integer ATTN_P   = 4;
    localparam integer T2       = 6;
    localparam integer ROWS_D   = D/P;
    localparam integer ROWS_FFN = FFN/P;
    localparam integer LANES    = 128;
    localparam integer WBW      = 8;
    localparam integer WBITS    = LANES*WBW;
    localparam integer SUBW     = WBITS/32;
    localparam integer NWORDS   = `NWORDS;
    localparam integer NLAYER   = `NLAYER;
    localparam integer NCALL    = 6;   // q,k,v,o,fc1,fc2 -- CALL_NAMES order

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    reg go; reg [3:0] blk; wire done;
    reg xres_wr; reg [$clog2(T2)-1:0] xres_wpos, xres_rpos_dbg;
    reg [$clog2(D/P)-1:0] xres_waddr; reg [P*32-1:0] xres_wdata;
    wire [P*32-1:0] xres_rdata_dbg;

    reg [19:0] wb_q, wb_k, wb_v, wb_o, wb_fc1, wb_fc2;
    reg signed [7:0] gf_q, gf_k, gf_v, gf_o, gf_fc1, gf_fc2;

    reg gv_ld_rst, gv_ld_we; reg [31:0] gv_ld_data;
    reg gam_we, gam_sel; reg [$clog2(D/P)-1:0] gam_waddr; reg [P*32-1:0] gam_wdata;
    reg bias_we, bias_sel; reg [$clog2(FFN/P)-1:0] bias_waddr; reg [P*32-1:0] bias_wdata;

    encoder_block_seq #(
        .P(P), .D(D), .FFN(FFN), .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .ATTN_P(ATTN_P), .T2(T2),
        .POST_SCALE_Q16(75674),
        .ACT_Q(19), .ACT_K(19), .ACT_V(19), .ACT_O(23), .ACT_FC1(19), .ACT_FC2(9)
    ) dut (
        .clk(clk), .rst(rst), .go(go), .blk(blk), .done(done),
        .wb_q(wb_q), .wb_k(wb_k), .wb_v(wb_v), .wb_o(wb_o), .wb_fc1(wb_fc1), .wb_fc2(wb_fc2),
        .gf_q(gf_q), .gf_k(gf_k), .gf_v(gf_v), .gf_o(gf_o), .gf_fc1(gf_fc1), .gf_fc2(gf_fc2),
        .xres_wr(xres_wr), .xres_wpos(xres_wpos), .xres_waddr(xres_waddr), .xres_wdata(xres_wdata),
        .xres_rpos_dbg(xres_rpos_dbg), .xres_rdata_dbg(xres_rdata_dbg),
        .gv_ld_rst(gv_ld_rst), .gv_ld_we(gv_ld_we), .gv_ld_data(gv_ld_data),
        .gam_we(gam_we), .gam_sel(gam_sel), .gam_waddr(gam_waddr), .gam_wdata(gam_wdata),
        .bias_we(bias_we), .bias_sel(bias_sel), .bias_waddr(bias_waddr), .bias_wdata(bias_wdata)
    );

    // ---- test vectors ----
    reg [WBITS-1:0] wload [0:NWORDS-1];
    reg [31:0] gamma1 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] gamma2 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] biasfc1 [0:NLAYER*ROWS_FFN*P-1];
    reg [31:0] biasfc2 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] xresin [0:T2*ROWS_D*P-1];
    reg [31:0] xresoutref [0:NLAYER*T2*D-1];
    reg [19:0] wboff [0:NLAYER*NCALL-1];
    reg [7:0]  gfsh  [0:NLAYER*NCALL-1];

    integer i, s, l, mism, checked, ly_i, p_i, mism_ly;
    reg [WBITS-1:0] word_tmp;
    reg [P*32-1:0] rowbuf;

    // watchdog: force a diagnostic dump + finish if `done` hasn't fired
    // within a generous margin of one encoder-layer forward pass (all T2
    // positions), so a hang doesn't run out the full $finish-at-2e9
    // backstop below.
    reg dbg_armed;
    reg [31:0] watchdog_cnt;
    always @(posedge clk) begin
        if (dbg_armed && !done) watchdog_cnt <= watchdog_cnt + 1;
        else watchdog_cnt <= 0;
        if (dbg_armed && watchdog_cnt == 32'd200000) begin
            $display("TB_WATCHDOG_TIMEOUT,t=%0t,st=%0d,pos=%0d,hh=%0d,wi=%0d",
                $time, dut.st, dut.pos, dut.hh, dut.wi);
            $finish;
        end
    end

    task set_wb_gf(input integer li);
        reg [19:0] base;
        begin
            base = li * NCALL;
            wb_q = wboff[base+0]; wb_k = wboff[base+1]; wb_v = wboff[base+2];
            wb_o = wboff[base+3]; wb_fc1 = wboff[base+4]; wb_fc2 = wboff[base+5];
            gf_q = $signed(gfsh[base+0]); gf_k = $signed(gfsh[base+1]); gf_v = $signed(gfsh[base+2]);
            gf_o = $signed(gfsh[base+3]); gf_fc1 = $signed(gfsh[base+4]); gf_fc2 = $signed(gfsh[base+5]);
        end
    endtask

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("gamma_ln1_all.mem", gamma1);
        $readmemh("gamma_ln2_all.mem", gamma2);
        $readmemh("bias_fc1_all.mem", biasfc1);
        $readmemh("bias_fc2_all.mem", biasfc2);
        $readmemh("xres_in.mem", xresin);
        $readmemh("xres_out_ref.mem", xresoutref);
        $readmemh("wb_offsets.mem", wboff);
        $readmemh("gf_shifts.mem", gfsh);

        go=0; blk=0; xres_wr=0; xres_wpos=0; xres_waddr=0; xres_wdata=0; xres_rpos_dbg=0;
        gv_ld_rst=0; gv_ld_we=0; gv_ld_data=0;
        gam_we=0; gam_sel=0; gam_waddr=0; gam_wdata=0;
        bias_we=0; bias_sel=0; bias_waddr=0; bias_wdata=0;
        wb_q=0; wb_k=0; wb_v=0; wb_o=0; wb_fc1=0; wb_fc2=0;
        gf_q=0; gf_k=0; gf_v=0; gf_o=0; gf_fc1=0; gf_fc2=0;

        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ---- 1. load GEMV resident weight image (once, ALL layers) ----
        gv_ld_rst = 1; @(posedge clk); #1; gv_ld_rst = 0;
        for (i = 0; i < NWORDS; i = i + 1) begin
            word_tmp = wload[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                gv_ld_we = 1; gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        gv_ld_we = 0;
        $display("TB_WEIGHTS_LOADED,nwords=%0d", NWORDS);

        // ---- 2. write layer 0's own real per-position input (ALL T2
        // positions) -- no cross-attn/KV preload needed at all. ----
        for (p_i = 0; p_i < T2; p_i = p_i + 1) begin
            for (i = 0; i < ROWS_D; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = xresin[p_i*ROWS_D*P + i*P + l];
                xres_wpos = p_i[$clog2(T2)-1:0]; xres_waddr = i[$clog2(D/P)-1:0];
                xres_wdata = rowbuf; xres_wr = 1;
                @(posedge clk); #1;
            end
        end
        xres_wr = 0;
        $display("TB_XRESIN_LOADED");

        // ---- 3. run NLAYER layers in order, chaining through xres_bank
        // (same physical memory, no re-write needed between layers). ----
        mism = 0; checked = 0;
        for (ly_i = 0; ly_i < NLAYER; ly_i = ly_i + 1) begin
            for (i = 0; i < ROWS_D; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = gamma1[ly_i*ROWS_D*P + i*P + l];
                gam_sel = 1'b0; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
                @(posedge clk); #1;
            end
            for (i = 0; i < ROWS_D; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = gamma2[ly_i*ROWS_D*P + i*P + l];
                gam_sel = 1'b1; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
                @(posedge clk); #1;
            end
            gam_we = 0;

            for (i = 0; i < ROWS_FFN; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = biasfc1[ly_i*ROWS_FFN*P + i*P + l];
                bias_sel = 1'b0; bias_waddr = i[$clog2(FFN/P)-1:0]; bias_wdata = rowbuf; bias_we = 1;
                @(posedge clk); #1;
            end
            for (i = 0; i < ROWS_D; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = biasfc2[ly_i*ROWS_D*P + i*P + l];
                bias_sel = 1'b1; bias_waddr = i[$clog2(FFN/P)-1:0]; bias_wdata = rowbuf; bias_we = 1;
                @(posedge clk); #1;
            end
            bias_we = 0;

            set_wb_gf(ly_i);
            blk = ly_i[3:0];
            dbg_armed = 1'b1;
            go = 1; @(posedge clk); #1; go = 0;
            while (!done) @(posedge clk); #1;
            dbg_armed = 1'b0;
            $display("TB_LAYER_DONE,layer=%0d", ly_i);

            mism_ly = 0;
            for (p_i = 0; p_i < T2; p_i = p_i + 1) begin
                xres_rpos_dbg = p_i[$clog2(T2)-1:0];
                for (i = 0; i < ROWS_D; i = i + 1) begin
                    xres_waddr = i[$clog2(D/P)-1:0];
                    #1;
                    for (l = 0; l < P; l = l + 1) begin
                        checked = checked + 1;
                        if (xres_rdata_dbg[l*32 +: 32] !==
                            xresoutref[(ly_i*T2 + p_i)*D + i*P + l]) begin
                            mism = mism + 1;
                            mism_ly = mism_ly + 1;
                            if (mism <= 10)
                                $display("MISMATCH,layer=%0d,pos=%0d,row=%0d,lane=%0d,got=%0d,ref=%0d",
                                          ly_i, p_i, i, l, $signed(xres_rdata_dbg[l*32 +: 32]),
                                          $signed(xresoutref[(ly_i*T2 + p_i)*D + i*P + l]));
                        end
                    end
                end
            end
            $display("TB_LAYER_CHECK,layer=%0d,mismatches=%0d", ly_i, mism_ly);
        end

        $display("TB_DONE,checked=%0d,mismatches=%0d", checked, mism);
        $display("ENCODER_BLOCK_SEQ_VERDICT,bitexact=%0d,mismatches=%0d,checked=%0d,nlayer=%0d",
                  (mism == 0), mism, checked, NLAYER);
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
