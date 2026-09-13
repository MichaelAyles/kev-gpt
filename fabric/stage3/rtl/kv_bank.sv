// -----------------------------------------------------------------------------
// kv_bank — the on-chip K/V cache that makes the single-stream sequencer decode
// FAITHFULLY (doc 7 R1). Quantise-at-write, dequantise-at-read-stream, K8/V8,
// per-(head, position) asymmetric — the PINNED goformer_kvq divfree contract:
//
//   write (one head's K or V vector, HEAD_DIM Q.16 ints, streamed HR rows wide):
//     lo    = min(x),  span = max(x) - lo
//     scale = max( rdiv(span, QMAX), 1 )            rdiv = round-half-up (a+b/2)/b
//     inv   = rdiv(1<<INV_SH, scale)                one serial divide per head*pos
//     code  = clip( (u*inv + HALF) >> INV_SH, QMAX) u = x - lo  (all non-negative)
//   read  (stream tcount positions of one head, HR rows/cycle after prefetch):
//     x_hat = code * scale + lo                      exact integer, kv_dma verbatim
//
// Storage (KBITS=8, TMAX=256, 4 layers, 4 heads): codes 512 KB + hdr 48 KB.
// Banks are row-addressed wide words (CLAUDE.md banking rule): the position is a
// memory ADDRESS, never a per-lane mux. Banks are NOT cleared between tokens or
// conversations — a conversation restarts at pos 0 and overwrites; attention at
// pos p reads only rows 0..p, all freshly written (the same argument that lets
// the bench skip KV resets).
//
// iverilog-2012 safe: variable part-selects only on plain regs (vecbuf, code_r),
// unpacked arrays read by element into a reg first.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_bank #(
    parameter integer P        = 8,
    parameter integer HEAD_DIM = 64,
    parameter integer NHEAD    = 4,
    parameter integer NLAYER   = 4,
    parameter integer TMAX     = 256,
    parameter integer KBITS    = 8,
    parameter integer INV_SH   = 24,
    // Genesys2 (Kintex-7) port: no URAM primitive exists on that part at all,
    // so the code bank must fall back to ordinary BRAM TDP. "block" is a
    // strictly synthesizable XPM_MEMORY_TDPRAM choice, not a capability
    // downgrade (7-series BRAM36 is natively true-dual-port); only the
    // KV260 default is left at "ultra" so its own gates/bitstream are
    // untouched.
    parameter               MEM_PRIMITIVE = "ultra"
) (
    input  wire        clk,
    input  wire        rst,

    // ---- quantise-write port: one head's K or V vector per wq_start ----------
    input  wire        wq_start,            // pulse; selectors sampled here
    input  wire [3:0]  wq_layer,
    input  wire        wq_kv,               // 0 = K, 1 = V
    input  wire [1:0]  wq_head,
    input  wire [8:0]  wq_pos,
    input  wire        wq_valid,            // HR beats of P Q.16 lanes follow
    input  wire [P*32-1:0] wq_data,
    output reg         wq_done,             // pulses when codes + hdr are stored

    // ---- dequant read-stream port A: tcount positions of one head ------------
    input  wire        rd_start,            // pulse; selectors sampled here
    input  wire [3:0]  rd_layer,
    input  wire        rd_kv,
    input  wire [1:0]  rd_head,
    input  wire [8:0]  rd_tcount,           // positions to stream (1..TMAX)
    output reg         rd_valid,            // a dequantised wide row is on rd_data
    output reg  [HEAD_DIM*32-1:0] rd_data,  // ONE POSITION's head row, dequantised
    output reg         rd_done,             // pulses after the last position

    // ---- dequant read-stream port B (doc-7 R4e): the URAM/BRAM second port ----
    input  wire        rd2_start,
    input  wire [3:0]  rd2_layer,
    input  wire        rd2_kv,
    input  wire [1:0]  rd2_head,
    input  wire [8:0]  rd2_tcount,
    output reg         rd2_valid,
    output reg  [HEAD_DIM*32-1:0] rd2_data,
    output reg         rd2_done
);
    localparam integer HR    = HEAD_DIM / P;
    localparam integer QMAX  = (1 << KBITS) - 1;
    localparam integer NHSEL = NLAYER * 2 * NHEAD;          // (layer,kv,head) combos
    localparam integer HROWS = NHSEL * TMAX;                // one row per (sel, pos)
    localparam integer DIVW  = INV_SH + 2;                  // divider width (26)


    // ---- write FSM ---------------------------------------------------------------
    localparam [3:0] W_IDLE=4'd0, W_COLL=4'd1, W_SCALE=4'd2, W_INVL=4'd3,
                     W_INVR=4'd4, W_SCALE2=4'd5, W_QNT=4'd6, W_CWR=4'd7,
                     W_MMD=4'd8;             // min/max merge drain (last beat)
    reg [3:0] wst;

    // ---- read FSMs (A and B) -------------------------------------------------
    localparam [1:0] R_IDLE=2'd0, R_RUN=2'd2;
    reg [1:0] rst_st, rst2_st;
    // ---- banks (doc-7 R2: one POSITION per code row — KBITS=8 makes a head's
    // position exactly HEAD_DIM*8 = 512 bits, read 1 position/cycle).
    // DUAL-DIALECT (the weight_bank_tdp pattern): HDL inference of TDP UltraRAM
    // is dead in 2025.2, so the SYNTHESIS branch is xpm_memory_tdpram and the
    // sim branch a behavioral 2-port array. Gates verify sim; the board run is
    // the final bit-exactness check on the XPM side.
    wire [HEAD_DIM*KBITS-1:0] code_r,  code_r2;
    wire [47:0]               hdr_rd,  hdr_rd2;

    // ---- write-side state ------------------------------------------------------
    reg [3:0]  w_layer; reg w_kv; reg [1:0] w_head; reg [8:0] w_pos;
    reg [HEAD_DIM*32-1:0] vecbuf;            // plain reg: the head vector, collected
    reg signed [31:0] minv, maxv;
    reg [$clog2(HR+1)-1:0] w_vi;             // collect beat counter
    reg [$clog2(HR+1)-1:0] w_qi;             // quantise row counter
    reg signed [31:0] w_lo;
    reg [15:0] w_scale;
    reg [INV_SH:0] w_inv;                    // inv <= 2^INV_SH

    // doc-7 R4a: NO divider. scale = rdiv(span,255) by the EXACT magic multiply
    // ((span+127)*0x80808081)>>39, proven == floor((span+127)/255) over the full
    // span range [0, 2^22) by exhaustive python check; inv = rdiv(2^24, scale)
    // from a constant ROM (one q_round_div per entry, data-independent, written
    // by the harness as inv_lut.mem). Same numbers as the serial divides, ~52
    // cycles fewer per (head, pos).
    // Two-ROM split (BRAM diet): scale < 4096 needs the full 25-bit inv;
    // scale >= 4096 has inv = rdiv(2^24, scale) <= 4096 -> 13 bits. Same values,
    // ~1/3 the BRAM bits of one padded 32k x 25 ROM.
    localparam integer INVD = 16512;         // scale <= rdiv(2^22, 255) = 16,449
    (* rom_style = "block" *) reg [INV_SH:0] inv_lut_lo [0:4095];
    (* rom_style = "block" *) reg [12:0]     inv_lut_hi [0:INVD-4097];
    initial begin
        $readmemh("inv_lut_lo.mem", inv_lut_lo);
        $readmemh("inv_lut_hi.mem", inv_lut_hi);
    end
    reg [22:0] w_span;                       // span+127, explicitly unsigned
    reg [53:0] w_magic;                      // w_span * 0x80808081 (23b x 32b, unsigned)
    reg [INV_SH:0] inv_lo_rd;
    reg [12:0]     inv_hi_rd;
    reg            inv_hi_sel;
    // sync ROM read: address is the combinational scale during W_INVL (so the
    // reads land in W_INVR), then the registered w_scale.
    wire [14:0] sc_now = (w_magic[53:39] == 0) ? 15'd1 : w_magic[53:39];
    wire [14:0] lut_sc = (wst == W_INVL) ? sc_now : w_scale[14:0];
    always @(posedge clk) begin
        inv_lo_rd  <= inv_lut_lo[lut_sc[11:0]];
        inv_hi_rd  <= inv_lut_hi[lut_sc - 15'd4096];
        inv_hi_sel <= (lut_sc >= 15'd4096);
    end
    wire [INV_SH:0] inv_rd = inv_hi_sel ? {12'b0, inv_hi_rd} : inv_lo_rd;

    // TIMING (5ns cone wq_data -> beat min/max -> running merge, formerly one
    // 25-level cycle): stage A registers two QUAD min/max reductions (2 compare
    // levels) of the incoming beat; stage B merges them into the running
    // minv/maxv ONE BEAT BEHIND (2 compare levels), gated by qm_v. The LAST
    // beat's pending quads drain in W_MMD (min/max are associative + idempotent
    // — same numbers, +1 cycle per write). W_SCALE reads plain minv/maxv.
    integer mp;
    reg signed [31:0] lane_v, quad_min [0:1], quad_max [0:1];
    always @* begin
        quad_min[0] = 32'sh7FFFFFFF; quad_max[0] = 32'sh80000000;
        quad_min[1] = 32'sh7FFFFFFF; quad_max[1] = 32'sh80000000;
        for (mp = 0; mp < P; mp = mp + 1) begin
            lane_v = $signed(wq_data[mp*32 +: 32]);
            if (lane_v < quad_min[mp/(P/2)]) quad_min[mp/(P/2)] = lane_v;
            if (lane_v > quad_max[mp/(P/2)]) quad_max[mp/(P/2)] = lane_v;
        end
    end
    reg signed [31:0] qm_min0, qm_min1, qm_max0, qm_max1;
    reg               qm_v;
    wire signed [31:0] qmn = (qm_min0 < qm_min1) ? qm_min0 : qm_min1;
    wire signed [31:0] qmx = (qm_max0 > qm_max1) ? qm_max0 : qm_max1;

    // TIMING (8ns worst cone w_qi_reg -> wstage_reg): W_QNT is a 2-stage pipeline.
    // Stage A registers the SELECTED+SUBTRACTED row (the vecbuf row-mux + w_lo
    // subtract, the heavy front half); stage B does mult+round+clip and writes
    // wstage ONE ROW BEHIND (at w_qi_d). One extra drain cycle before W_CWR so
    // the commit always sees the COMPLETE wstage. Same numbers, +1 cycle per
    // (head,kv,pos) write — feeder-side, hidden under attention.
    reg [P*32-1:0] qd_r;                     // stage-A reg: u = x - lo per lane (>= 0)
    reg [$clog2(HR+1)-1:0] w_qi_d;           // stage-A reg: the row qd_r belongs to
    reg            qd_v;                     // stage-A valid
    reg [P*32-1:0] qd_now;                   // combinational diffs of row w_qi
    reg [$clog2(HR+1)-1:0] q_sel;            // w_qi clamped on the drain cycle
    reg signed [32:0] q_diff;
    integer qp2;
    always @* begin
        q_sel  = (w_qi >= HR) ? {($clog2(HR+1)){1'b0}} : w_qi;
        qd_now = {(P*32){1'b0}};
        for (qp2 = 0; qp2 < P; qp2 = qp2 + 1) begin
            q_diff = $signed(vecbuf[(q_sel*P + qp2)*32 +: 32]) - w_lo; // >= 0 by construction
            qd_now[qp2*32 +: 32] = q_diff[31:0];
        end
    end

    // stage B: quantise P lanes of the REGISTERED diffs (mult+round+clip only)
    reg [P*KBITS-1:0] q_codes;
    reg [31:0]  q_u;
    reg [31+INV_SH+1:0] q_prod;              // u(<=2^21) * inv(<=2^24) + rounding
    reg [31:0]  q_code;
    integer qp;
    always @* begin
        q_codes = {(P*KBITS){1'b0}};
        for (qp = 0; qp < P; qp = qp + 1) begin
            q_u    = qd_r[qp*32 +: 32];
            q_prod = q_u * w_inv + (1 << (INV_SH-1));
            q_code = q_prod >> INV_SH;
            if (q_code > QMAX) q_code = QMAX;
            q_codes[qp*KBITS +: KBITS] = q_code[KBITS-1:0];
        end
    end

    // ---- read-side state -------------------------------------------------------
    // ZERO-BUBBLE stream, one POSITION per cycle: the hdr read is pipelined
    // ALONGSIDE the code-row read (separate memories, same 1-cycle latency).
    // rowi walks positions 0..T-1. Total = T + 2 cycles per stream.
    reg [8:0]  r_rowi,  r2_rowi;               // position counters (0..TMAX)
    reg [8:0]  r_nrows, r2_nrows;              // T
    reg [8:0]  r_ecnt,  r2_ecnt;               // emitted-position counters
    reg        r_v0,    r2_v0;                 // addr-stage valids
    reg        r_v1,    r2_v1;                 // mem-out-register stage valids
    // TIMING (take-5 worst path): one register stage between the hdr/code memory
    // outputs and the 64-lane dequant; the stream grows by one cycle.
    reg [HEAD_DIM*KBITS-1:0] code_q, code_q2;
    reg [47:0]               hdr_q,  hdr_q2;

    // base addresses (registered at start; code and hdr share the per-position base)
    reg [$clog2(HROWS)-1:0] w_pbase, r_pbase, r2_pbase;
    reg [HEAD_DIM*KBITS-1:0] wstage;           // staged code row (P codes per W_QNT cycle)

    // dequant HEAD_DIM lanes of code_r with the CO-READ header (aligned):
    //   x_hat = code * scale + lo   (exact integer, kv_dma verbatim)
    reg [HEAD_DIM*32-1:0] deq_word, deq_word2;
    reg [KBITS-1:0] d_code;
    reg [KBITS+16-1:0] d_prod;
    reg signed [33:0] d_full;
    integer dp;
    always @* begin
        deq_word = {(HEAD_DIM*32){1'b0}};
        for (dp = 0; dp < HEAD_DIM; dp = dp + 1) begin
            d_code = code_q[dp*KBITS +: KBITS];
            d_prod = d_code * hdr_q[47:32];                  // unsigned mul (kv_dma shape)
            d_full = $signed({1'b0, d_prod}) + $signed(hdr_q[31:0]); // + signed lo
            deq_word[dp*32 +: 32] = d_full[31:0];
        end
    end
    reg [KBITS-1:0] d2_code;
    reg [KBITS+16-1:0] d2_prod;
    reg signed [33:0] d2_full;
    integer dq2;
    always @* begin
        deq_word2 = {(HEAD_DIM*32){1'b0}};
        for (dq2 = 0; dq2 < HEAD_DIM; dq2 = dq2 + 1) begin
            d2_code = code_q2[dq2*KBITS +: KBITS];
            d2_prod = d2_code * hdr_q2[47:32];
            d2_full = $signed({1'b0, d2_prod}) + $signed(hdr_q2[31:0]);
            deq_word2[dq2*32 +: 32] = d2_full[31:0];
        end
    end


    // sync reads: code row + hdr row for position rowi — co-read every cycle.
    // Stream B reads through the memories' SECOND port (URAM/BRAM TDP).
    // PORT DISCIPLINE: the quant-write (W_CWR) is FOLDED INTO port A — it commits
    // only when stream A is idle (W_CWR stalls on rst_st, see the write FSM), so
    // each memory is exactly write-else-read on A plus read on B = two ports.
    wire [$clog2(HROWS)-1:0] pos_ra  = r_pbase  + {{($clog2(HROWS)-9){1'b0}}, r_rowi};
    wire [$clog2(HROWS)-1:0] pos_ra2 = r2_pbase + {{($clog2(HROWS)-9){1'b0}}, r2_rowi};
    wire cwr_fire = (wst == W_CWR) && (rst_st == R_IDLE);
    // ONE ADDRESS NET PER PORT (the URAM contract): write and read share port A's
    // muxed address; port B is the second read stream.
    wire [$clog2(HROWS)-1:0] kva_a = cwr_fire ? w_pbase : pos_ra;
`ifdef SYNTHESIS
    xpm_memory_tdpram #(
        .ADDR_WIDTH_A($clog2(HROWS)), .ADDR_WIDTH_B($clog2(HROWS)),
        .BYTE_WRITE_WIDTH_A(HEAD_DIM*KBITS), .BYTE_WRITE_WIDTH_B(HEAD_DIM*KBITS),
        .CLOCKING_MODE("common_clock"), .MEMORY_PRIMITIVE(MEM_PRIMITIVE),
        .MEMORY_SIZE(HEAD_DIM*KBITS*HROWS),
        .READ_DATA_WIDTH_A(HEAD_DIM*KBITS), .READ_DATA_WIDTH_B(HEAD_DIM*KBITS),
        .READ_LATENCY_A(1), .READ_LATENCY_B(1),
        .WRITE_DATA_WIDTH_A(HEAD_DIM*KBITS), .WRITE_DATA_WIDTH_B(HEAD_DIM*KBITS),
        .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
    ) u_code (
        .clka(clk), .clkb(clk), .rsta(1'b0), .rstb(1'b0), .ena(1'b1), .enb(1'b1),
        .regcea(1'b1), .regceb(1'b1), .wea(cwr_fire), .web(1'b0),
        .addra(kva_a), .addrb(pos_ra2),
        .dina(wstage), .dinb({(HEAD_DIM*KBITS){1'b0}}),
        .douta(code_r), .doutb(code_r2),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb(), .sleep(1'b0));
    xpm_memory_tdpram #(
        .ADDR_WIDTH_A($clog2(HROWS)), .ADDR_WIDTH_B($clog2(HROWS)),
        .BYTE_WRITE_WIDTH_A(48), .BYTE_WRITE_WIDTH_B(48),
        .CLOCKING_MODE("common_clock"), .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(48*HROWS),
        .READ_DATA_WIDTH_A(48), .READ_DATA_WIDTH_B(48),
        .READ_LATENCY_A(1), .READ_LATENCY_B(1),
        .WRITE_DATA_WIDTH_A(48), .WRITE_DATA_WIDTH_B(48),
        .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
    ) u_hdr (
        .clka(clk), .clkb(clk), .rsta(1'b0), .rstb(1'b0), .ena(1'b1), .enb(1'b1),
        .regcea(1'b1), .regceb(1'b1), .wea(cwr_fire), .web(1'b0),
        .addra(kva_a), .addrb(pos_ra2),
        .dina({w_scale, w_lo}), .dinb(48'b0),
        .douta(hdr_rd), .doutb(hdr_rd2),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb(), .sleep(1'b0));
`else
    reg [HEAD_DIM*KBITS-1:0] code_bank [0:HROWS-1];
    reg [47:0]               hdr_bank  [0:HROWS-1];      // {scale16, lo32}
    reg [HEAD_DIM*KBITS-1:0] code_r_b,  code_r2_b;
    reg [47:0]               hdr_rd_b,  hdr_rd2_b;
    always @(posedge clk) begin
        if (cwr_fire) code_bank[kva_a] <= wstage;
        code_r_b  <= code_bank[kva_a];
        code_r2_b <= code_bank[pos_ra2];                  // port B: read
    end
    always @(posedge clk) begin
        if (cwr_fire) hdr_bank[kva_a] <= {w_scale, w_lo};
        hdr_rd_b  <= hdr_bank[kva_a];
        hdr_rd2_b <= hdr_bank[pos_ra2];
    end
    assign code_r  = code_r_b;
    assign code_r2 = code_r2_b;
    assign hdr_rd  = hdr_rd_b;
    assign hdr_rd2 = hdr_rd2_b;
`endif
    always @(posedge clk) begin
        code_q  <= code_r;   hdr_q  <= hdr_rd;
        code_q2 <= code_r2;  hdr_q2 <= hdr_rd2;
    end

    always @(posedge clk) begin
        wq_done <= 1'b0; rd_done <= 1'b0; rd2_done <= 1'b0;
        if (rst) begin
            wst <= W_IDLE; rst_st <= R_IDLE; rst2_st <= R_IDLE; qd_v <= 1'b0;
            rd_valid <= 1'b0; r_v0 <= 1'b0; r_v1 <= 1'b0;
            rd2_valid <= 1'b0; r2_v0 <= 1'b0; r2_v1 <= 1'b0;
        end else begin
            // =================== write side ===================
            case (wst)
                W_IDLE: if (wq_start) begin
                    w_layer <= wq_layer; w_kv <= wq_kv; w_head <= wq_head; w_pos <= wq_pos;
                    w_pbase <= ((wq_layer*2 + {3'b0,wq_kv})*NHEAD + {2'b0,wq_head})*TMAX
                                + {3'b0,wq_pos};
                    minv <= 32'sh7FFFFFFF; maxv <= 32'sh80000000;
                    qm_v <= 1'b0;
                    w_vi <= 0; wst <= W_COLL;
                end
                W_COLL: begin
                    // merge stage: the PREVIOUS beat's registered quad min/max
                    if (qm_v) begin
                        if (qmn < minv) minv <= qmn;
                        if (qmx > maxv) maxv <= qmx;
                    end
                    qm_v <= wq_valid;
                    if (wq_valid) begin
                        vecbuf[w_vi*P*32 +: P*32] <= wq_data;
                        qm_min0 <= quad_min[0]; qm_min1 <= quad_min[1];
                        qm_max0 <= quad_max[0]; qm_max1 <= quad_max[1];
                        if (w_vi == HR-1) begin
                            wst <= W_MMD;
                        end else w_vi <= w_vi + 1'b1;
                    end
                end
                W_MMD: begin
                    // drain: merge the LAST beat's pending quads (qm_v is always
                    // set here — W_COLL exits on a wq_valid beat)
                    if (qmn < minv) minv <= qmn;
                    if (qmx > maxv) maxv <= qmx;
                    qm_v <= 1'b0;
                    wst <= W_SCALE;
                end
                W_SCALE: begin
                    // scale = rdiv(span,255) = ((span+127)*0x80808081)>>39 (EXACT;
                    // two registered steps, all-unsigned, no mixed-sign multiply)
                    w_lo   <= minv;
                    w_span <= (maxv - minv + (QMAX >> 1));
                    wst <= W_SCALE2;
                end
                W_SCALE2: begin
                    w_magic <= {31'd0, w_span} * 54'd2155905153;
                    wst <= W_INVL;
                end
                W_INVL: begin
                    w_scale <= {1'b0, sc_now};
                    wst <= W_INVR;                     // ROM read for sc_now lands next cycle
                end
                W_INVR: begin
                    // inv_rd (sync ROM read at w_scale) lands NOW
                    w_inv <= inv_rd;
                    w_qi  <= 0;
                    qd_v  <= 1'b0;
                    wst   <= W_QNT;
                end
                W_QNT: begin
                    // stage A: register row w_qi's diffs (vecbuf row-mux + subtract)
                    qd_r   <= qd_now;
                    w_qi_d <= w_qi;
                    qd_v   <= (w_qi != HR);
                    // stage B: codes for row w_qi_d land ONE cycle behind; the
                    // w_qi == HR drain cycle writes the last row, then commit
                    if (qd_v) wstage[w_qi_d*P*KBITS +: P*KBITS] <= q_codes;
                    if (w_qi == HR) wst <= W_CWR;
                    else w_qi <= w_qi + 1'b1;
                end
                W_CWR: if (cwr_fire) begin
                    // commit happens in the port-A always block this same cycle
                    // (cwr_fire); stalls while stream A is mid-read (the feeder's
                    // writes land in the softmax gaps between K and V streams).
                    wq_done <= 1'b1;
                    wst <= W_IDLE;
                end
                default: wst <= W_IDLE;
            endcase

            // =================== read side (stream A) ===================
            rd_valid <= 1'b0;
            case (rst_st)
                R_IDLE: if (rd_start) begin
                    r_nrows <= rd_tcount;
                    r_rowi  <= 0;
                    r_ecnt  <= 0;
                    r_v0    <= 1'b0;
                    r_pbase <= ((rd_layer*2 + {3'b0,rd_kv})*NHEAD
                                 + {2'b0,rd_head})*TMAX;
                    rst_st <= R_RUN;
                end
                R_RUN: begin
                    // addr stage: present row rowi (code + its position's hdr together)
                    r_v0 <= (r_rowi != r_nrows);
                    r_v1 <= r_v0;
                    if (r_rowi != r_nrows) r_rowi <= r_rowi + 1'b1;
                    // emit stage: code_q/hdr_q (mem-out + one reg) land now
                    if (r_v1) begin
                        rd_valid <= 1'b1;
                        rd_data  <= deq_word;
                        if (r_ecnt == r_nrows - 1) begin
                            rd_done <= 1'b1;
                            rst_st  <= R_IDLE;
                        end else r_ecnt <= r_ecnt + 1'b1;
                    end
                end
                default: rst_st <= R_IDLE;
            endcase

            // =================== read side (stream B, port B) ===================
            rd2_valid <= 1'b0;
            case (rst2_st)
                R_IDLE: if (rd2_start) begin
                    r2_nrows <= rd2_tcount;
                    r2_rowi  <= 0;
                    r2_ecnt  <= 0;
                    r2_v0    <= 1'b0;
                    r2_pbase <= ((rd2_layer*2 + {3'b0,rd2_kv})*NHEAD
                                  + {2'b0,rd2_head})*TMAX;
                    rst2_st <= R_RUN;
                end
                R_RUN: begin
                    r2_v0 <= (r2_rowi != r2_nrows);
                    r2_v1 <= r2_v0;
                    if (r2_rowi != r2_nrows) r2_rowi <= r2_rowi + 1'b1;
                    if (r2_v1) begin
                        rd2_valid <= 1'b1;
                        rd2_data  <= deq_word2;
                        if (r2_ecnt == r2_nrows - 1) begin
                            rd2_done <= 1'b1;
                            rst2_st  <= R_IDLE;
                        end else r2_ecnt <= r2_ecnt + 1'b1;
                    end
                end
                default: rst2_st <= R_IDLE;
            endcase
        end
    end
endmodule
