// -----------------------------------------------------------------------------
// tb_encoder_self_attn — the "static full-attend" access pattern gate: does
// checkpoint C's real, UNMODIFIED kv_bank.sv + vec_attn_w.sv, chained with
// rope_apply_vec.sv, correctly compute ASR's ENCODER bidirectional
// self-attention -- write a FIXED T2-row K/V set ONCE, then read the SAME
// set repeatedly (once per query row, every query attending every key, no
// causal masking) -- with ZERO new storage RTL?
//
// Why this gate exists: ASR-ACCELERATOR-OP-SEQUENCE.md's "Two attention
// shapes, one primitive" section originally classified this access pattern
// as needing NEW storage RTL ("architecturally closer to weight_bank_tdp.sv"),
// reasoning that kv_bank.sv's design is built for a cache that grows. A
// cheap isolated probe (write once, read the same fixed tcount 3 separate
// times) showed the read port has no actual incremental dependency baked
// into a single rd_start call -- "growing" is purely a caller convention
// (the decoder varying tcount step by step), not RTL-enforced. This gate is
// the real, model-weight-driven proof: writes ALL T2*NHEAD K/V rows first
// (no interleaving with reads, unlike the decoder gate), THEN loops every
// query position over every head, each call reading the full, unchanging
// T2-row set via kv_bank's ordinary read port with a FIXED tcount=T2.
//
// Scope, same as tb_decoder_self_attn.sv: LayerNorm + the Q/K/V GEMVs are
// out of scope (already separately gated) -- this TB drives the three
// ATTENTION-specific modules directly with real Q/K/V (from the real HF
// model's own encoder layer-0 weights, on a real conv front-end output),
// checked against pack_encoder_self_attn.py's own golden ctx_q25 output.
//
// P=4 (matches the decoder gate, same HEAD_DIM=36/P=8-doesn't-divide
// reasoning). T2=6 (this TB's own real-weights-derived sequence length --
// see pack_encoder_self_attn.py's AUDIO_LEN choice, not a floor).
//
// STATUS: see ASR-ACCELERATOR-OP-SEQUENCE.md's "Encoder self-attention
// block gate" section for the full writeup.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_encoder_self_attn;
    localparam integer P        = 4;
    localparam integer HEAD_DIM = 36;
    localparam integer NHEAD    = 8;
    localparam integer NLAYER   = 1;
    localparam integer TMAX     = 32;
    localparam integer T2       = 6;    // real encoder positions (pack_encoder_self_attn.py)
    localparam integer HR       = HEAD_DIM / P;
    localparam integer ROPE_TMAX = 128;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    // ---- rope_apply_vec ------------------------------------------------------
    reg                       rope_start;
    reg  [$clog2(ROPE_TMAX)-1:0] rope_pos;
    reg  [HEAD_DIM*32-1:0]    rope_head_in;
    wire                      rope_done;
    wire [HEAD_DIM*32-1:0]    rope_head_out;

    rope_apply_vec #(.HEAD_DIM(HEAD_DIM), .ROT_DIM(32), .ROT_PAIRS(16), .TMAX(ROPE_TMAX),
                      .POST_SCALE_Q16(75674),
                      .ROM_FILE_COS("rope_cos.mem"), .ROM_FILE_SIN("rope_sin.mem")) u_rope (
        .clk(clk), .start(rope_start), .position(rope_pos),
        .head_in(rope_head_in), .done(rope_done), .head_out(rope_head_out)
    );

    // ---- kv_bank ---------------------------------------------------------------
    reg         kb_wstart, kb_wvalid, kb_rstart;
    reg  [3:0]  kb_wlayer;
    reg         kb_wkv;
    reg  [$clog2(NHEAD)-1:0]  kb_whead;
    reg  [8:0]  kb_wpos;
    reg  [P*32-1:0] kb_wdata;
    wire        kb_wdone;
    reg  [3:0]  kb_rlayer;
    reg         kb_rkv;
    reg  [$clog2(NHEAD)-1:0]  kb_rhead;
    reg  [8:0]  kb_rtcount;
    wire        kb_rvalid, kb_rdone;
    wire [HEAD_DIM*32-1:0] kb_rdata;

    kv_bank #(.P(P), .HEAD_DIM(HEAD_DIM), .NHEAD(NHEAD), .NLAYER(NLAYER), .TMAX(TMAX),
              .KBITS(8), .MEM_PRIMITIVE("block")) u_kvb (
        .clk(clk), .rst(rst),
        .wq_start(kb_wstart), .wq_layer(kb_wlayer), .wq_kv(kb_wkv), .wq_head(kb_whead),
        .wq_pos(kb_wpos), .wq_valid(kb_wvalid), .wq_data(kb_wdata), .wq_done(kb_wdone),
        .rd_start(kb_rstart), .rd_layer(kb_rlayer), .rd_kv(kb_rkv), .rd_head(kb_rhead),
        .rd_tcount(kb_rtcount), .rd_valid(kb_rvalid), .rd_data(kb_rdata), .rd_done(kb_rdone),
        .rd2_start(1'b0), .rd2_layer(4'd0), .rd2_kv(1'b0), .rd2_head(2'd0), .rd2_tcount(9'd0),
        .rd2_valid(), .rd2_data(), .rd2_done()
    );

    // ---- vec_attn_w --------------------------------------------------------------
    reg                    at_start;
    reg  [8:0]             at_tcount;
    reg                    at_qvalid;
    reg  [P*32-1:0]        at_qdata;
    wire                   at_kdone, at_ctxvalid, at_done;
    wire [6:0]             at_ctxidx;
    wire [P*32-1:0]        at_ctxdata;

    vec_attn_w #(.P(P), .HEAD_DIM(HEAD_DIM), .TMAX(TMAX)) u_attn (
        .clk(clk), .rst(rst), .start(at_start), .tcount(at_tcount),
        .q_valid(at_qvalid), .q_data(at_qdata),
        .kv_valid(kb_rvalid), .kv_data(kb_rdata),
        .k_done(at_kdone), .ctx_valid(at_ctxvalid), .ctx_idx(at_ctxidx), .ctx_data(at_ctxdata),
        .done(at_done)
    );

    // ---- test vectors ------------------------------------------------------------
    reg [31:0] kv_in [0:T2*NHEAD*HEAD_DIM*2-1];   // pos-major, head-major: [k(36) v(36)]
    reg [31:0] q_in  [0:T2*NHEAD*HEAD_DIM-1];      // pos-major, head-major: [q(36)]
    reg [31:0] ctx_ref [0:T2*NHEAD*HEAD_DIM-1];    // qpos-major, head-major

    reg [HEAD_DIM*32-1:0] k_h, v_h, q_h, k_rope_h, q_rope_h, ctx_got;

    integer pos, qpos, head, i, base, mismatches, checked;
    integer f;

    task do_rope(input integer p, input reg [HEAD_DIM*32-1:0] vin, output reg [HEAD_DIM*32-1:0] vout);
        begin
            rope_head_in = vin;
            rope_pos = p[$clog2(ROPE_TMAX)-1:0];
            rope_start = 1'b1;
            @(posedge clk); #1;
            rope_start = 1'b0;
            if (!rope_done) begin
                $display("TB_FAIL,rope_done_not_asserted,pos=%0d", p);
                $finish;
            end
            vout = rope_head_out;
        end
    endtask

    task do_kv_write(input integer kv, input integer head_i, input integer p,
                      input reg [HEAD_DIM*32-1:0] vec);
        integer b, l;
        begin
            kb_wlayer = 4'd0; kb_wkv = kv[0]; kb_whead = head_i[$clog2(NHEAD)-1:0]; kb_wpos = p[8:0];
            kb_wstart = 1'b1;
            @(posedge clk); #1;
            kb_wstart = 1'b0;
            for (b = 0; b < HR; b = b + 1) begin
                for (l = 0; l < P; l = l + 1) kb_wdata[l*32 +: 32] = vec[(b*P+l)*32 +: 32];
                kb_wvalid = 1'b1;
                @(posedge clk); #1;
            end
            kb_wvalid = 1'b0;
            while (!kb_wdone) @(posedge clk);
            #1;
        end
    endtask

    // full attention call for one head: feed Q, stream K then V from kv_bank
    // (ALWAYS tcount=T2 -- the fixed, already-fully-written set, never grown)
    task do_attn(input integer head_i, input reg [HEAD_DIM*32-1:0] qvec,
                 output reg [HEAD_DIM*32-1:0] ctxvec);
        integer b, l;
        begin
            at_tcount = T2[8:0];
            at_start = 1'b1;
            @(posedge clk); #1;
            at_start = 1'b0;
            for (b = 0; b < HR; b = b + 1) begin
                for (l = 0; l < P; l = l + 1) at_qdata[l*32 +: 32] = qvec[(b*P+l)*32 +: 32];
                at_qvalid = 1'b1;
                @(posedge clk); #1;
            end
            at_qvalid = 1'b0;

            kb_rlayer = 4'd0; kb_rkv = 1'b0; kb_rhead = head_i[$clog2(NHEAD)-1:0]; kb_rtcount = T2[8:0];
            kb_rstart = 1'b1;
            @(posedge clk); #1;
            kb_rstart = 1'b0;

            while (!at_kdone) @(posedge clk);

            kb_rlayer = 4'd0; kb_rkv = 1'b1; kb_rhead = head_i[$clog2(NHEAD)-1:0]; kb_rtcount = T2[8:0];
            kb_rstart = 1'b1;
            @(posedge clk); #1;
            kb_rstart = 1'b0;

            begin : ctx_drain
                reg done_seen;
                done_seen = 1'b0;
                while (!done_seen) begin
                    @(posedge clk); #1;
                    if (at_ctxvalid) begin
                        for (l = 0; l < P; l = l + 1)
                            ctxvec[(at_ctxidx*P + l)*32 +: 32] = at_ctxdata[l*32 +: 32];
                    end
                    if (at_done) done_seen = 1'b1;
                end
            end
        end
    endtask

    initial begin
        $readmemh("kv_in.mem", kv_in);
        $readmemh("q_in.mem", q_in);
        $readmemh("ctx_ref.mem", ctx_ref);

        rope_start = 1'b0; rope_pos = 0; rope_head_in = 0;
        kb_wstart = 1'b0; kb_wvalid = 1'b0; kb_rstart = 1'b0;
        kb_wlayer = 0; kb_wkv = 0; kb_whead = 0; kb_wpos = 0; kb_wdata = 0;
        kb_rlayer = 0; kb_rkv = 0; kb_rhead = 0; kb_rtcount = 0;
        at_start = 1'b0; at_tcount = 0; at_qvalid = 1'b0; at_qdata = 0;

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk); #1;

        mismatches = 0; checked = 0;
        f = $fopen("run_log.txt", "w");

        // ---- pass 1: write the FULL K/V set once, all T2 positions, all
        // NHEAD heads -- no query read happens until every position is
        // written, unlike the decoder gate's per-step interleaving. ----------
        for (pos = 0; pos < T2; pos = pos + 1) begin
            for (head = 0; head < NHEAD; head = head + 1) begin
                base = (pos * NHEAD + head) * HEAD_DIM * 2;
                for (i = 0; i < HEAD_DIM; i = i + 1) begin
                    k_h[i*32 +: 32] = kv_in[base + i];
                    v_h[i*32 +: 32] = kv_in[base + HEAD_DIM + i];
                end
                do_rope(pos, k_h, k_rope_h);
                do_kv_write(0, head, pos, k_rope_h);   // K, RoPE'd
                do_kv_write(1, head, pos, v_h);        // V, never RoPE'd
            end
        end
        $display("TB_KV_WRITTEN,t2=%0d,nhead=%0d", T2, NHEAD);

        // ---- pass 2: EVERY query position attends the FULL, unchanging
        // T2-row set -- bidirectional, no causal masking, tcount==T2 always.
        for (qpos = 0; qpos < T2; qpos = qpos + 1) begin
            for (head = 0; head < NHEAD; head = head + 1) begin
                base = (qpos * NHEAD + head) * HEAD_DIM;
                for (i = 0; i < HEAD_DIM; i = i + 1)
                    q_h[i*32 +: 32] = q_in[base + i];

                do_rope(qpos, q_h, q_rope_h);
                do_attn(head, q_rope_h, ctx_got);

                for (i = 0; i < HEAD_DIM; i = i + 1) begin
                    checked = checked + 1;
                    if (ctx_got[i*32 +: 32] !== ctx_ref[base + i]) begin
                        mismatches = mismatches + 1;
                        if (mismatches <= 10)
                            $fwrite(f, "MISMATCH qpos=%0d head=%0d dim=%0d got=%0d ref=%0d\n",
                                    qpos, head, i, $signed(ctx_got[i*32 +: 32]), $signed(ctx_ref[base + i]));
                    end
                end
                $display("TB_QUERY,qpos=%0d,head=%0d,done", qpos, head);
            end
        end
        $fclose(f);

        $display("TB_DONE,checked=%0d,mismatches=%0d", checked, mismatches);
        $display("ENCODER_SELF_ATTN_VERDICT,bitexact=%0d,mismatches=%0d,checked=%0d",
                  (mismatches == 0), mismatches, checked);
        $finish;
    end

    initial begin
        #10_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
