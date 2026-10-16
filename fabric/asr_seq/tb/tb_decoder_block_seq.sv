// tb_decoder_block_seq -- the MULTI-LAYER, MULTI-STEP functional gate for
// decoder_block_seq.sv: loads the real moonshine-tiny resident weight
// image (ALL NLAYER decoder layers' 8 GEMVs each, INT8-quantized per
// pack_decoder_block.py's own documented scheme) and per-layer cross-attn
// K/V (T2=6) ONCE, then for each of NSTEPS real decode steps, chains
// through all NLAYER layers in order (layer L's own RES3 output sitting
// in xres_bank becomes layer L+1's own LN1 input -- the SAME physical
// memory across `go` pulses, no explicit hand-off needed): reloads that
// layer's own gamma/bias tables and wb_*/gf_* runtime constants, pulses
// `go`, and checks that layer's own residual output against
// pack_decoder_block.py's own per-(step,layer) golden xres3 -- bit-exact.
//
// Weight-load protocol: identical to fabric/stage3/run_resident_banked_vec.py's
// own established gemv_banked_resident_vec.sv gate -- one-time resident load,
// each LANES*WBW=1024-bit wide word streamed as SUBW=32 32-bit chunks
// (low-chunk-first) via w_we/w_data, after one ld_rst pulse.
`timescale 1ns / 1ps
`ifndef NWORDS
 `define NWORDS 13824
`endif
`ifndef NSTEPS
 `define NSTEPS 3
`endif
`ifndef NLAYER
 `define NLAYER 6
`endif

module tb;
    localparam integer P        = 8;
    localparam integer D        = 288;
    localparam integer FFN      = 1152;
    localparam integer DFFN2    = 2*FFN;
    localparam integer NHEAD    = 8;
    localparam integer HEAD_DIM = 36;
    localparam integer ATTN_P   = 4;
    localparam integer T2       = 6;
    localparam integer ROWS_D   = D/P;
    localparam integer ROWS_FFN2 = DFFN2/P;
    localparam integer LANES    = 128;
    localparam integer WBW      = 8;
    localparam integer WBITS    = LANES*WBW;
    localparam integer SUBW     = WBITS/32;
    localparam integer NWORDS   = `NWORDS;
    localparam integer NSTEPS   = `NSTEPS;
    localparam integer NLAYER   = `NLAYER;
    localparam integer NCALL    = 8;   // q,k,v,o,cq,oc,fc1,fc2 -- CALL_NAMES order

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    reg go; reg [3:0] blk; reg [8:0] step; wire done;
    reg xres_wr; reg [$clog2(D/P)-1:0] xres_waddr; reg [P*32-1:0] xres_wdata;
    wire [P*32-1:0] xres_rdata_dbg;

    reg [19:0] wb_q, wb_k, wb_v, wb_o, wb_cq, wb_co, wb_fc1, wb_fc2;
    reg signed [7:0] gf_q, gf_k, gf_v, gf_o, gf_cq, gf_co, gf_fc1, gf_fc2;

    reg gv_ld_rst, gv_ld_we; reg [31:0] gv_ld_data;
    reg gam_we; reg [1:0] gam_sel; reg [$clog2(D/P)-1:0] gam_waddr; reg [P*32-1:0] gam_wdata;
    reg bias_we, bias_sel; reg [$clog2(DFFN2/P)-1:0] bias_waddr; reg [P*32-1:0] bias_wdata;
    reg xkv_wstart, xkv_wkv; reg [$clog2(NHEAD)-1:0] xkv_whead; reg [8:0] xkv_wpos;
    reg xkv_wvalid; reg [ATTN_P*32-1:0] xkv_wdata; wire xkv_wdone;

    // ACT_* stay compile-time parameters -- ONE fixed value per call site,
    // profiled by pack_decoder_block.py across ALL NLAYER layers x NSTEPS
    // steps (verified to clip nowhere). wb_*/gf_* are runtime ports (see
    // decoder_block_seq.sv's own header for why GF_* specifically needed
    // converting: WSHIFT, baked into g_frac, is genuinely per-layer).
    decoder_block_seq #(
        .P(P), .D(D), .FFN(FFN), .NHEAD(NHEAD), .HEAD_DIM(HEAD_DIM), .ATTN_P(ATTN_P),
        .ATTN_TMAX(8), .T2(T2),
        .POST_SCALE_Q16(75674),
        .ACT_Q(18), .ACT_K(18), .ACT_V(18), .ACT_O(21),
        .ACT_CQ(19), .ACT_OC(23), .ACT_FC1(17), .ACT_FC2(12)
    ) dut (
        .clk(clk), .rst(rst), .go(go), .blk(blk), .step(step), .done(done),
        .wb_q(wb_q), .wb_k(wb_k), .wb_v(wb_v), .wb_o(wb_o),
        .wb_cq(wb_cq), .wb_co(wb_co), .wb_fc1(wb_fc1), .wb_fc2(wb_fc2),
        .gf_q(gf_q), .gf_k(gf_k), .gf_v(gf_v), .gf_o(gf_o),
        .gf_cq(gf_cq), .gf_co(gf_co), .gf_fc1(gf_fc1), .gf_fc2(gf_fc2),
        .xres_wr(xres_wr), .xres_waddr(xres_waddr), .xres_wdata(xres_wdata),
        .xres_rdata_dbg(xres_rdata_dbg),
        .gv_ld_rst(gv_ld_rst), .gv_ld_we(gv_ld_we), .gv_ld_data(gv_ld_data),
        .gam_we(gam_we), .gam_sel(gam_sel), .gam_waddr(gam_waddr), .gam_wdata(gam_wdata),
        .bias_we(bias_we), .bias_sel(bias_sel), .bias_waddr(bias_waddr), .bias_wdata(bias_wdata),
        .xkv_wstart(xkv_wstart), .xkv_wkv(xkv_wkv), .xkv_whead(xkv_whead), .xkv_wpos(xkv_wpos),
        .xkv_wvalid(xkv_wvalid), .xkv_wdata(xkv_wdata), .xkv_wdone(xkv_wdone)
    );

    // ---- test vectors ----
    reg [WBITS-1:0] wload [0:NWORDS-1];   // NWORDS is the TOTAL across all NLAYER layers, from the manifest
    reg [31:0] gamma1 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] gamma2 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] gamma3 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] biasfc1 [0:NLAYER*ROWS_FFN2*P-1];
    reg [31:0] biasfc2 [0:NLAYER*ROWS_D*P-1];
    reg [31:0] xkvin [0:NLAYER*T2*NHEAD*HEAD_DIM*2-1];
    reg [31:0] xres0steps [0:NSTEPS*ROWS_D*P-1];
    reg [31:0] xres3refsteps [0:NSTEPS*NLAYER*D-1];
    reg [19:0] wboff [0:NLAYER*NCALL-1];
    reg [7:0]  gfsh  [0:NLAYER*NCALL-1];

    integer i, s, hcnt, pos, head, b, l, mism, checked, st_i, ly_i, mism_ly;
    reg [HEAD_DIM*32-1:0] kvec, vvec;
    reg [WBITS-1:0] word_tmp;
    reg [P*32-1:0] rowbuf;

    // watchdog: force a diagnostic dump + finish if `done` hasn't fired
    // within a generous margin of one decoder-layer forward pass, so a
    // hang doesn't run out the full $finish-at-2e9 backstop below.
    reg dbg_armed;
    reg [31:0] watchdog_cnt;
    always @(posedge clk) begin
        if (dbg_armed && !done) watchdog_cnt <= watchdog_cnt + 1;
        else watchdog_cnt <= 0;
        if (dbg_armed && watchdog_cnt == 32'd60000) begin
            $display("TB_WATCHDOG_TIMEOUT,t=%0t,st=%0d,hh=%0d,wi=%0d", $time, dut.st, dut.hh, dut.wi);
            $finish;
        end
    end

    task set_wb_gf(input integer li);
        reg [19:0] base;
        begin
            base = li * NCALL;
            wb_q = wboff[base+0]; wb_k = wboff[base+1]; wb_v = wboff[base+2]; wb_o = wboff[base+3];
            wb_cq = wboff[base+4]; wb_co = wboff[base+5]; wb_fc1 = wboff[base+6]; wb_fc2 = wboff[base+7];
            gf_q = $signed(gfsh[base+0]); gf_k = $signed(gfsh[base+1]);
            gf_v = $signed(gfsh[base+2]); gf_o = $signed(gfsh[base+3]);
            gf_cq = $signed(gfsh[base+4]); gf_co = $signed(gfsh[base+5]);
            gf_fc1 = $signed(gfsh[base+6]); gf_fc2 = $signed(gfsh[base+7]);
        end
    endtask

    initial begin
        $readmemh("w.mem", wload);
        $readmemh("gamma_ln1_all.mem", gamma1);
        $readmemh("gamma_ln2_all.mem", gamma2);
        $readmemh("gamma_ln3_all.mem", gamma3);
        $readmemh("bias_fc1_all.mem", biasfc1);
        $readmemh("bias_fc2_all.mem", biasfc2);
        $readmemh("xkv_in_all.mem", xkvin);
        $readmemh("xres0_steps.mem", xres0steps);
        $readmemh("xres3_ref_steps.mem", xres3refsteps);
        $readmemh("wb_offsets.mem", wboff);
        $readmemh("gf_shifts.mem", gfsh);

        go=0; blk=0; step=0; xres_wr=0; xres_waddr=0; xres_wdata=0;
        gv_ld_rst=0; gv_ld_we=0; gv_ld_data=0;
        gam_we=0; gam_sel=0; gam_waddr=0; gam_wdata=0;
        bias_we=0; bias_sel=0; bias_waddr=0; bias_wdata=0;
        xkv_wstart=0; xkv_wkv=0; xkv_whead=0; xkv_wpos=0; xkv_wvalid=0; xkv_wdata=0;
        wb_q=0; wb_k=0; wb_v=0; wb_o=0; wb_cq=0; wb_co=0; wb_fc1=0; wb_fc2=0;
        gf_q=0; gf_k=0; gf_v=0; gf_o=0; gf_cq=0; gf_co=0; gf_fc1=0; gf_fc2=0;

        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ---- 1. load GEMV resident weight image: ALL NLAYER layers' own
        // 8 GEMVs each, one combined resident image, loaded ONCE (never
        // reloaded between layers). ----
        gv_ld_rst = 1; @(posedge clk); #1; gv_ld_rst = 0;
        for (i = 0; i < NWORDS; i = i + 1) begin
            word_tmp = wload[i];
            for (s = 0; s < SUBW; s = s + 1) begin
                gv_ld_we = 1; gv_ld_data = word_tmp[s*32 +: 32]; @(posedge clk); #1;
            end
        end
        gv_ld_we = 0;
        $display("TB_WEIGHTS_LOADED,nwords=%0d", NWORDS);

        // ---- 2. preload cross-attn K/V for EVERY layer (once each --
        // kv_bank.sv's own storage is layer-indexed internally via `blk`,
        // so all NLAYER layers' cross K/V persist simultaneously). ----
        for (ly_i = 0; ly_i < NLAYER; ly_i = ly_i + 1) begin
            blk = ly_i[3:0];
            for (pos = 0; pos < T2; pos = pos + 1) begin
                for (head = 0; head < NHEAD; head = head + 1) begin
                    hcnt = ((ly_i*T2 + pos) * NHEAD + head) * HEAD_DIM * 2;
                    for (i = 0; i < HEAD_DIM; i = i + 1) begin
                        kvec[i*32 +: 32] = xkvin[hcnt + i];
                        vvec[i*32 +: 32] = xkvin[hcnt + HEAD_DIM + i];
                    end
                    // write K
                    xkv_wkv = 1'b0; xkv_whead = head[$clog2(NHEAD)-1:0]; xkv_wpos = pos[8:0];
                    xkv_wstart = 1'b1; @(posedge clk); #1; xkv_wstart = 1'b0;
                    for (b = 0; b < HEAD_DIM/ATTN_P; b = b + 1) begin
                        for (l = 0; l < ATTN_P; l = l + 1)
                            xkv_wdata[l*32 +: 32] = kvec[(b*ATTN_P+l)*32 +: 32];
                        xkv_wvalid = 1'b1; @(posedge clk); #1;
                    end
                    xkv_wvalid = 1'b0;
                    while (!xkv_wdone) @(posedge clk); #1;
                    // write V
                    xkv_wkv = 1'b1; xkv_whead = head[$clog2(NHEAD)-1:0]; xkv_wpos = pos[8:0];
                    xkv_wstart = 1'b1; @(posedge clk); #1; xkv_wstart = 1'b0;
                    for (b = 0; b < HEAD_DIM/ATTN_P; b = b + 1) begin
                        for (l = 0; l < ATTN_P; l = l + 1)
                            xkv_wdata[l*32 +: 32] = vvec[(b*ATTN_P+l)*32 +: 32];
                        xkv_wvalid = 1'b1; @(posedge clk); #1;
                    end
                    xkv_wvalid = 1'b0;
                    while (!xkv_wdone) @(posedge clk); #1;
                end
            end
        end
        $display("TB_CROSS_KV_LOADED,t2=%0d,nhead=%0d,nlayer=%0d", T2, NHEAD, NLAYER);

        // ---- 3. run NSTEPS decode steps, chaining through all NLAYER
        // layers each step. Step is the OUTER loop, layer the INNER loop
        // (real decode semantics: layer L's own step-t input needs layer
        // L-1's own step-t output, and layer L's own step-(t+1) self-attn
        // needs layer L's own step-t KV cache -- both directions require
        // this exact nesting). Only step 0's layer-0 gets a fresh xres0
        // write (the real per-token embedding); every other (step,layer)
        // reads the residual xres_bank ALREADY sitting there from the
        // previous `go` pulse -- layer chaining and step chaining are both
        // "free" this way, no explicit re-write needed except the very
        // first token embedding per step. ----
        mism = 0; checked = 0;
        for (st_i = 0; st_i < NSTEPS; st_i = st_i + 1) begin
            // fresh per-step token embedding -> layer 0's own xres0
            for (i = 0; i < ROWS_D; i = i + 1) begin
                for (l = 0; l < P; l = l + 1)
                    rowbuf[l*32 +: 32] = xres0steps[st_i*ROWS_D*P + i*P + l];
                xres_waddr = i[$clog2(D/P)-1:0]; xres_wdata = rowbuf; xres_wr = 1;
                @(posedge clk); #1;
            end
            xres_wr = 0;
            $display("TB_XRES0_LOADED,step=%0d", st_i);

            for (ly_i = 0; ly_i < NLAYER; ly_i = ly_i + 1) begin
                // reload this layer's own gamma/bias (RTL banks are sized
                // for one layer, reused sequentially -- real access
                // pattern never revisits a layer within a step).
                for (i = 0; i < ROWS_D; i = i + 1) begin
                    for (l = 0; l < P; l = l + 1)
                        rowbuf[l*32 +: 32] = gamma1[ly_i*ROWS_D*P + i*P + l];
                    gam_sel = 0; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
                    @(posedge clk); #1;
                end
                for (i = 0; i < ROWS_D; i = i + 1) begin
                    for (l = 0; l < P; l = l + 1)
                        rowbuf[l*32 +: 32] = gamma2[ly_i*ROWS_D*P + i*P + l];
                    gam_sel = 1; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
                    @(posedge clk); #1;
                end
                for (i = 0; i < ROWS_D; i = i + 1) begin
                    for (l = 0; l < P; l = l + 1)
                        rowbuf[l*32 +: 32] = gamma3[ly_i*ROWS_D*P + i*P + l];
                    gam_sel = 2; gam_waddr = i[$clog2(D/P)-1:0]; gam_wdata = rowbuf; gam_we = 1;
                    @(posedge clk); #1;
                end
                gam_we = 0;

                for (i = 0; i < ROWS_FFN2; i = i + 1) begin
                    for (l = 0; l < P; l = l + 1)
                        rowbuf[l*32 +: 32] = biasfc1[ly_i*ROWS_FFN2*P + i*P + l];
                    bias_sel = 0; bias_waddr = i[$clog2(DFFN2/P)-1:0]; bias_wdata = rowbuf; bias_we = 1;
                    @(posedge clk); #1;
                end
                for (i = 0; i < ROWS_D; i = i + 1) begin
                    for (l = 0; l < P; l = l + 1)
                        rowbuf[l*32 +: 32] = biasfc2[ly_i*ROWS_D*P + i*P + l];
                    bias_sel = 1; bias_waddr = i[$clog2(DFFN2/P)-1:0]; bias_wdata = rowbuf; bias_we = 1;
                    @(posedge clk); #1;
                end
                bias_we = 0;

                set_wb_gf(ly_i);
                blk = ly_i[3:0];
                step = st_i[8:0];
                dbg_armed = 1'b1;
                go = 1; @(posedge clk); #1; go = 0;
                while (!done) @(posedge clk); #1;
                dbg_armed = 1'b0;
                $display("TB_LAYER_DONE,step=%0d,layer=%0d", st_i, ly_i);

                mism_ly = 0;
                for (i = 0; i < ROWS_D; i = i + 1) begin
                    xres_waddr = i[$clog2(D/P)-1:0];
                    #1;
                    for (l = 0; l < P; l = l + 1) begin
                        checked = checked + 1;
                        if (xres_rdata_dbg[l*32 +: 32] !==
                            xres3refsteps[(st_i*NLAYER + ly_i)*D + i*P + l]) begin
                            mism = mism + 1;
                            mism_ly = mism_ly + 1;
                            if (mism <= 10)
                                $display("MISMATCH,step=%0d,layer=%0d,row=%0d,lane=%0d,got=%0d,ref=%0d",
                                          st_i, ly_i, i, l, $signed(xres_rdata_dbg[l*32 +: 32]),
                                          $signed(xres3refsteps[(st_i*NLAYER + ly_i)*D + i*P + l]));
                        end
                    end
                end
                $display("TB_LAYER_CHECK,step=%0d,layer=%0d,mismatches=%0d", st_i, ly_i, mism_ly);
            end
        end

        $display("TB_DONE,checked=%0d,mismatches=%0d", checked, mism);
        $display("DECODER_BLOCK_SEQ_VERDICT,bitexact=%0d,mismatches=%0d,checked=%0d,nsteps=%0d,nlayer=%0d",
                  (mism == 0), mism, checked, NSTEPS, NLAYER);
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
