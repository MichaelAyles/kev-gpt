// mamba_pipe.sv — the PIPELINED (wave) multi-stream Mamba-2 engine (doc 9 Rung C).
//
// Realizes the mamba_pipe_ref wave schedule in hardware: NC INDEPENDENT token
// streams share ONE instance each of the four sub-cores (gemv_i4i8, conv_silu,
// ssm_scan_row, rmsnorm_gated) — but instead of the monolithic mamba_seq FSM
// that runs them strictly serially (75% core idle), FIVE concurrent WORKERS
// (EMB / NORM / GEMV / CONV / SCAN) each drive their own core and serve one
// stream's whole op at a time. A round-robin scheduler advances each stream
// through its op-chain and hands a DIFFERENT stream to each free worker, so the
// gemv MAC of stream A overlaps the scan of stream B overlaps the conv of C —
// the plateau-breaking overlap the phase-major cohort could not get.
//
// BIT-EXACTNESS: every datapath here is lifted VERBATIM from mamba_seq.sv (same
// shifts / saturations / LUT indices / Q-formats). The schedule changes only
// WHICH stream occupies a core each cycle, never the arithmetic — so each
// stream's per-token logits/residual are bit-identical to single-stream
// mamba_seq_ref.step() (gated by run_mamba_pipe.py).
//
// STATE BANKING: per-stream residual (xbuf), working buffers (zxbuf/xnbuf/
// ybuf/q8buf/logit) are banked x NC (flat stream-major arrays). The scan h
// state is banked via ssm_scan_row's CTX=NC*L*H (pbase=stream*L*H+li*H+hi); the
// conv history via conv_silu's L=NC*L (layer=stream*L+li). Weights/LUTs/consts
// are shared read-only (the weight-stationary amortization).
//
// Each sub-core is driven by EXACTLY ONE worker, so there is NO port arbitration
// — the only shared resource a scheduler guards is "one stream per worker, one
// worker per stream." Ops within a stream stay strictly ordered (data deps);
// tokens are serial within a stream (state carry); overlap is only across the
// NC streams.

`default_nettype none

module mamba_pipe #(
    parameter int D     = 256,
    parameter int DIN   = 512,
    parameter int NST   = 64,
    parameter int H     = 8,
    parameter int LR    = 7,                  // real layers
    parameter int V     = 1024,
    parameter int NC    = 2,                  // streams (wave width)
    parameter int QH    = 16,                 // scan-h STORED bits/lane (state-quant
                                              // lever: 16=INT16 bit-exact, 12=INT12
                                              // to shrink the NC-scaling scan-h BRAM)
    parameter int TMAX  = 4,                  // max tokens/stream (dump sizing)
    parameter int T_TOKENS = 2,               // tokens each stream decodes
    parameter int DBG   = 1,                  // 0 = tie off the sim-only debug
                                              // readback (dump_x/dump_logit go
                                              // write-only -> pruned); bitstream
                                              // builds set DBG=0 for the LUT back.
    parameter int CONVD = DIN + 2*NST,        // 640
    parameter int INROWS = 2*DIN + 2*NST + H  // 1160
) (
    input  wire        clk,
    input  wire        rst,
    output wire        ready,                 // both cores' clear sweeps done
    input  wire        start,                 // begin decoding the loaded streams
    output reg         done,                  // all streams finished all tokens
    output reg [31:0]  cyc_count,             // MEASURED cyc from first dispatch

    // table writes: sel picks the target memory (same map as mamba_seq)
    input  wire        wr_en,
    input  wire [3:0]  wr_sel,
    input  wire [18:0] wr_addr,
    input  wire [31:0] wr_data,

    // per-(stream,token) input token load. Width expression is inline, NOT the
    // body localparam TW: a forward reference from the port list is a hard
    // error in xsim and only a WARNING in Vivado synth, which then elaborates
    // a mis-sized port — the first-silicon garbage-output bug (all-railed
    // logits from corrupted token/stream addressing).
    input  wire        tw_en,
    input  wire [((NC*TMAX <= 1) ? 1 : $clog2(NC*TMAX))-1:0] tw_addr,
    input  wire [9:0]  tw_data,

    // dump reads (combinational): sel 0 = dump_x (32b), 1 = dump_logit (16b)
    input  wire [3:0]  dbg_sel,
    input  wire [18:0] dbg_addr,   // 19b: the gemv weight image needs >18b
    output reg  signed [31:0] dbg_data
);
    localparam int SL   = NC*LR;              // conv history banks
    localparam int SCTX = NC*LR*H;            // scan state contexts
    localparam int SW   = (NC <= 1) ? 1 : $clog2(NC);   // stream-index width
    localparam int TW   = (NC*TMAX <= 1) ? 1 : $clog2(NC*TMAX);
    // write selectors (identical to mamba_seq)
    localparam [3:0] WSEL_GW = 0, WSEL_EMB = 1, WSEL_ESC = 2, WSEL_CW = 3,
                     WSEL_CB = 4, WSEL_SLUT = 5, WSEL_ALUT = 6, WSEL_DLUT = 7,
                     WSEL_LNG = 8, WSEL_NRMG = 9, WSEL_NFG = 10,
                     WSEL_RSIN = 11, WSEL_RSOUT = 12, WSEL_RSHD = 13,
                     WSEL_CONST = 14, WSEL_SEED = 15;

    // ------------------------------------------------------------ tables ----
    (* ram_style = "ultra", cascade_height = 4 *)
    reg [63:0] emb  [0:V*D/8/2-1];
    reg [31:0] emb_wlo;
    reg [15:0] esc   [0:V-1];
    reg [15:0] convw [0:LR*CONVD*4-1];
    reg [15:0] convb [0:LR*CONVD-1];
    reg [15:0] slut  [0:255];
    reg [15:0] alut  [0:LR*H*256-1];
    reg [15:0] dlut  [0:LR*H*256-1];
    reg [15:0] lng   [0:LR*D-1];
    reg [15:0] nrmg  [0:LR*DIN-1];
    reg [15:0] nfg   [0:D-1];
    reg [15:0] rsin  [0:LR*INROWS-1];
    reg [15:0] rsout [0:LR*D-1];
    reg [15:0] rshd  [0:V-1];
    reg [31:0] consts[0:127];

    always @(posedge clk) if (wr_en) begin
        case (wr_sel)
            WSEL_GW:    ;
            WSEL_EMB:   if (~wr_addr[0]) emb_wlo <= wr_data;
                        else emb[wr_addr[14:1]] <= {wr_data, emb_wlo};
            WSEL_ESC:   esc  [wr_addr[9:0]]  <= wr_data[15:0];
            WSEL_CW:    convw[wr_addr[14:0]] <= wr_data[15:0];
            WSEL_CB:    convb[wr_addr[12:0]] <= wr_data[15:0];
            WSEL_SLUT:  slut [wr_addr[7:0]]  <= wr_data[15:0];
            WSEL_ALUT:  alut [wr_addr[14:0]] <= wr_data[15:0];
            WSEL_DLUT:  dlut [wr_addr[14:0]] <= wr_data[15:0];
            WSEL_LNG:   lng  [wr_addr[10:0]] <= wr_data[15:0];
            WSEL_NRMG:  nrmg [wr_addr[11:0]] <= wr_data[15:0];
            WSEL_NFG:   nfg  [wr_addr[7:0]]  <= wr_data[15:0];
            WSEL_RSIN:  rsin [wr_addr[12:0]] <= wr_data[15:0];
            WSEL_RSOUT: rsout[wr_addr[10:0]] <= wr_data[15:0];
            WSEL_RSHD:  rshd [wr_addr[9:0]]  <= wr_data[15:0];
            WSEL_CONST: consts[wr_addr[6:0]] <= wr_data;
            default: ;
        endcase
    end

    // token sequences (per stream, teacher-forced)
    reg [9:0] tok_seq [0:NC*TMAX-1];
    always @(posedge clk) if (tw_en) tok_seq[tw_addr] <= tw_data;

    // --------------------------------------------- working buffers (LUTRAM) --
    // FIT DISCIPLINE (doc 9a §6): each wave buffer here has ONE writer so it
    // infers single-write-port distributed RAM (LUT-as-Memory) instead of the
    // FF+address-mux tree (LUT-as-Logic) the flat multi-port form demoted to.
    //  - The 4-lane-written buffers (zxbuf, q8buf) are packed into WIDE-WORD
    //    rows (4 elements per row) so the P=4 parallel write is ONE row write,
    //    not four write ports. Element reads slice the row (a cheap 4:1 mux);
    //    async reads are unchanged, so the FSM schedule/cycle-count is identical.
    //  - xnbuf/ybuf are 1-lane single-writer already: only a ram_style hint.
    //  - q8buf: NORM's two write sites (pre/gate P=4 + final P=1) are funnelled
    //    into ONE wide-word port (see q8w_* below) so it too infers DRAM.
    //  - xbuf (residual) has TWO writers (EMB init + GEMV out_proj RMW) that can
    //    fire on DIFFERENT streams the same cycle, so it is split into NC
    //    per-stream single-write-port wide-word banks (the `xbk` generate below).
    localparam int ZXW = INROWS/4;            // zxbuf wide-word rows per stream
    localparam int Q8W = DIN/4;               // q8buf wide-word rows per stream
    localparam int XNW = CONVD/4;             // xnbuf wide-word rows per stream
    localparam int YBW = DIN/4;               // ybuf  wide-word rows per stream
    // xnbuf/ybuf are wide-word (4x16b/row) so the P=4 CONV_RD / SCAN_RD writes are
    // ONE row write (single write port -> DRAM), not four ports. Element reads
    // slice the row (xn_rd/yb_rd), matching the zxbuf pattern. Each has ONE writer
    // (C_RD / SC_RD), so single-write-port distributed RAM infers cleanly.
    (* ram_style = "distributed" *)
    reg        [63:0] xnbuf_w [0:NC*XNW-1];   // conv out Q4.11, 4x16b/row
    (* ram_style = "distributed" *)
    reg        [63:0] zxbuf_w [0:NC*ZXW-1];   // in_proj dequant Q6.9, 4x16b/row
    (* ram_style = "distributed" *)
    reg        [63:0] ybuf_w  [0:NC*YBW-1];   // scan out Q4.11, 4x16b/row
    (* ram_style = "distributed" *)
    reg        [31:0] q8buf_w [0:NC*Q8W-1];   // quantized activations, 4x8b/row
    reg signed [15:0] logit [0:NC*V-1];

    // dumps: per (stream,token) snapshot for the gate (DBG=1 only — big, and
    // pruned in the bitstream build). dump_tok is the COMPACT per-(stream,token)
    // argmax token — always present (the board's record protocol reads it, and
    // it anchors the whole compute chain so DBG=0 can't optimise the datapath
    // away). dump_logit/dump_x are the wide bit-exact-gate readbacks.
    reg        [9:0]  dump_tok   [0:NC*TMAX-1];
    reg signed [15:0] dump_logit [0:NC*TMAX*V-1];
    reg signed [31:0] dump_x     [0:NC*TMAX*D-1];

    // wide-word zxbuf element read: flat index idx = stream*INROWS + offset.
    // INROWS is a multiple of 4, so row = idx>>2, element = idx[1:0] (a plain-reg
    // part-select — the only variable +: form iverilog/Vivado handle safely).
    function automatic signed [15:0] zx_rd(input [15:0] idx);
        reg [63:0] row;
        begin
            row   = zxbuf_w[idx[15:2]];
            zx_rd = row[idx[1:0]*16 +: 16];
        end
    endfunction

    // wide-word q8buf element read: flat index idx = stream*DIN + offset (DIN is
    // a multiple of 4). row = idx>>2, byte = idx[1:0].
    function automatic signed [7:0] q8_rd(input [15:0] idx);
        reg [31:0] row;
        begin
            row   = q8buf_w[idx[15:2]];
            q8_rd = row[idx[1:0]*8 +: 8];
        end
    endfunction

    // wide-word xnbuf element read: idx = stream*CONVD + offset (CONVD mult of 4).
    function automatic signed [15:0] xn_rd(input [15:0] idx);
        reg [63:0] row;
        begin
            row   = xnbuf_w[idx[15:2]];
            xn_rd = row[idx[1:0]*16 +: 16];
        end
    endfunction

    // wide-word ybuf element read: idx = stream*DIN + offset (DIN mult of 4).
    function automatic signed [15:0] yb_rd(input [15:0] idx);
        reg [63:0] row;
        begin
            row   = ybuf_w[idx[15:2]];
            yb_rd = row[idx[1:0]*16 +: 16];
        end
    endfunction

    // ------------------------------------------------------- scale helpers --
    function automatic signed [47:0] mul_m11(input signed [31:0] a,
                                             input [15:0] fp16);
        reg [10:0] m;
        begin m = {1'b1, fp16[9:0]}; mul_m11 = a * $signed({1'b0, m}); end
    endfunction
    function automatic signed [15:0] sat16f(input signed [47:0] v);
        sat16f = (v > 48'sd32767) ? 16'sd32767 :
                 (v < -48'sd32768) ? -16'sd32768 : v[15:0];
    endfunction
    function automatic signed [31:0] sat32f(input signed [47:0] v);
        sat32f = (v > 48'sd2147483647) ? 32'sd2147483647 :
                 (v < -48'sd2147483648) ? -32'sd2147483648 : v[31:0];
    endfunction
    function automatic signed [47:0] rshr(input signed [47:0] v,
                                          input signed [7:0] n);
        if (n <= 0)      rshr = v <<< (-n);
        else             rshr = (v + (48'sd1 <<< (n - 1))) >>> n;
    endfunction
    function automatic [4:0] bitlen15(input [14:0] v);
        integer b;
        begin
            bitlen15 = 0;
            for (b = 14; b >= 0; b = b - 1)
                if (v[b] && bitlen15 == 0) bitlen15 = b + 1;
        end
    endfunction

    // ---------------------------------------------------------- sub-cores ---
    // gemv (driven only by the GEMV worker)
    reg         g_start;  wire g_done;
    reg  [18:0] g_base;   reg [10:0] g_rows;  reg [6:0] g_wpr;
    reg         g_wrx;    reg [8:0] g_wrx_a;  reg signed [7:0] g_wrx_d;
    wire [31:0] g_wdbg;   wire [19:0] n_seed_dbg;   // silicon table readback
    reg  [10:0] g_rda;    wire signed [31:0] g_acc;
    reg  [10:0] g_rdaw;   wire [4*32-1:0] g_accw;
    gemv_i4i8 #(.PE(16), .ROWS(INROWS), .D_IN(DIN), .WMEM(409600), .RDP(4)) u_gemv (
        .clk(clk), .rst(rst), .start(g_start), .done(g_done),
        .base(g_base), .rows(g_rows), .wpr(g_wpr),
        .wr_w(wr_en && wr_sel == WSEL_GW), .wr_w_addr(wr_addr), .wr_w_data(wr_data),
        .wr_x(g_wrx), .wr_x_addr(g_wrx_a), .wr_x_data(g_wrx_d),
        .rd_acc_addr(g_rda), .rd_acc_data(g_acc),
        .rd_accw_base(g_rdaw), .rd_accw_data(g_accw),
        .rd_w_addr(dbg_addr[18:0]), .rd_w_data(g_wdbg));

    // conv+silu (driven only by the CONV worker); NC*LR history banks
    reg         c_start;  wire c_done;  wire c_ready;
    reg  [$clog2(SL)-1:0] c_layer;
    reg         c_wrw, c_wrb, c_wrx, c_wrl;
    reg  [11:0] c_wrw_a;  reg [9:0] c_wrb_a, c_wrx_a, c_rda;
    reg  [7:0]  c_wrl_a;
    reg signed [15:0] c_wrd;                     // weight/LUT load data (shared)
    // CONV_LDX collapse: bias + x are independent ports -> written the same cycle
    // (own data reg each). CONV_LDX 1280->640.
    reg signed [15:0] c_wrb_d, c_wrx_d;
    wire signed [15:0] c_y;
    reg  [9:0]  c_rdaw;   wire [4*16-1:0] c_yw;   // CONV_RD wide read (P=4)
    conv_silu #(.CH(CONVD), .K(4), .L(SL), .RDP(4)) u_conv (
        .clk(clk), .rst(rst), .start(c_start), .done(c_done), .ready(c_ready),
        .layer(c_layer),
        .wr_w(c_wrw), .wr_w_addr(c_wrw_a), .wr_w_data(c_wrd),
        .wr_b(c_wrb), .wr_b_addr(c_wrb_a), .wr_b_data(c_wrb_d),
        .wr_x(c_wrx), .wr_x_addr(c_wrx_a), .wr_x_data(c_wrx_d),
        .wr_lut(c_wrl), .wr_lut_addr(c_wrl_a), .wr_lut_data(c_wrd),
        .rd_y_addr(c_rda), .rd_y_data(c_y),
        .rd_yw_base(c_rdaw), .rd_yw_data(c_yw));

    // scan row (driven only by the SCAN worker); NC*LR*H state contexts
    reg         s_start;  wire s_done, s_ready;
    reg  [$clog2(SCTX)-1:0] s_pbase;
    reg  [15:0] s_aq;     reg signed [5:0] s_shi, s_shy;
    reg         s_wrdtx, s_wrb, s_wrc;
    reg  [5:0]  s_wrdtx_a, s_wrb_a, s_wrc_a, s_rda;
    reg signed [15:0] s_dtx_d;
    // SC_PREP collapse: B and C are independent ports -> written the same cycle
    // (own data reg each). SC_PREP 4*NST -> NST cycles.
    reg signed [7:0]  s_b_d, s_c_d;
    wire signed [15:0] s_y;
    reg  [5:0]  s_rdaw;   wire [4*16-1:0] s_yw;   // SCAN_RD wide read (P=4)
    ssm_scan_row #(.P(64), .N(NST), .QH(QH), .CTX(SCTX), .RDP(4)) u_scan (
        .clk(clk), .rst(rst), .ready(s_ready), .start(s_start), .done(s_done),
        .pbase(s_pbase), .a_q(s_aq), .sh_i(s_shi), .sh_y(s_shy),
        .wr_dtx(s_wrdtx), .wr_dtx_addr(s_wrdtx_a), .wr_dtx_data(s_dtx_d),
        .wr_b(s_wrb), .wr_b_addr(s_wrb_a), .wr_b_data(s_b_d),
        .wr_c(s_wrc), .wr_c_addr(s_wrc_a), .wr_c_data(s_c_d),
        .rd_y_addr(s_rda), .rd_y_data(s_y),
        .rd_yw_base(s_rdaw), .rd_yw_data(s_yw));
    assign ready = s_ready & c_ready;

    // gated rmsnorm (driven only by the NORM worker)
    reg         n_start;  wire n_done;
    reg         n_gated, n_short;
    reg         n_wry, n_wrz, n_wrg, n_wrl;
    reg  [8:0]  n_wry_a, n_wrz_a, n_wrg_a, n_rda;
    reg  [7:0]  n_wrl_a;
    reg signed [15:0] n_wrd;                     // LUT-load data (shared)
    // NORM feed collapse: y/z/g are three INDEPENDENT write ports on rmsnorm, so
    // the stream-in drives all three in ONE cycle (own data reg each) instead of
    // 3 (gated) / 2 (pre/final) serial sub-steps. NG_LD 1536->512, NIN/NF 512->256.
    reg signed [15:0] n_wry_d, n_wrz_d, n_wrg_d;
    wire signed [15:0] n_out;
    reg  [8:0]  n_rdaw;   wire [4*16-1:0] n_ow;
    rmsnorm_gated #(.D(DIN), .RDP(4)) u_norm (
        .clk(clk), .rst(rst), .start(n_start), .done(n_done), .gated(n_gated),
        .short_len(n_short),
        .wr_y(n_wry), .wr_y_addr(n_wry_a), .wr_y_data(n_wry_d),
        .wr_z(n_wrz), .wr_z_addr(n_wrz_a), .wr_z_data(n_wrz_d),
        .wr_g(n_wrg), .wr_g_addr(n_wrg_a), .wr_g_data(n_wrg_d),
        .wr_lut(n_wrl), .wr_lut_addr(n_wrl_a), .wr_lut_data(n_wrd),
        .wr_seed(wr_en && wr_sel == WSEL_SEED),
        .wr_seed_addr(wr_addr[5:0]), .wr_seed_data(wr_data[19:0]),
        .rd_seed_addr(dbg_addr[5:0]), .rd_seed_data(n_seed_dbg),
        .rd_o_addr(n_rda), .rd_o_data(n_out),
        .rd_ow_base(n_rdaw), .rd_ow_data(n_ow));

    // ============================================================ scheduler ==
    // op-chain: pc 0 = EMB; pc 1..42 = 7 x [NPRE,GIN,CONV,SCAN,NGATE,GOUT];
    // pc 43 = NFIN; pc 44 = GHD; pc 45 = token complete.
    localparam int NPC = 45;
    localparam [3:0] OP_EMB=0, OP_NPRE=1, OP_GIN=2, OP_CONV=3, OP_SCAN=4,
                     OP_NGATE=5, OP_GOUT=6, OP_NFIN=7, OP_GHD=8, OP_DONE=9;

    // op_of / layer_of are on the wave dispatch critical path (re-evaluated per
    // stream inside each worker's dispatch loop). The original `% 6` / `/ 6`
    // synthesise to CARRY8 divider chains and — fanned across the chip to the
    // worker layer regs (g_li/c_li/s_li) — were the MEASURED routed critical path
    // (op_pc -> /6 -> worker layer reg, route-dominated). Grouped-case lookups
    // (identical truth table) map to a shallow LUT mux instead, taking the divider
    // off that path. Values identical => bit-exact, same schedule / cycle count.
    function automatic [3:0] op_of(input [7:0] pc);
        case (pc)
            8'd0:                                             op_of = OP_EMB;
            8'd1, 8'd7, 8'd13, 8'd19, 8'd25, 8'd31, 8'd37:    op_of = OP_NPRE;
            8'd2, 8'd8, 8'd14, 8'd20, 8'd26, 8'd32, 8'd38:    op_of = OP_GIN;
            8'd3, 8'd9, 8'd15, 8'd21, 8'd27, 8'd33, 8'd39:    op_of = OP_CONV;
            8'd4, 8'd10, 8'd16, 8'd22, 8'd28, 8'd34, 8'd40:   op_of = OP_SCAN;
            8'd5, 8'd11, 8'd17, 8'd23, 8'd29, 8'd35, 8'd41:   op_of = OP_NGATE;
            8'd6, 8'd12, 8'd18, 8'd24, 8'd30, 8'd36, 8'd42:   op_of = OP_GOUT;
            8'd43:                                            op_of = OP_NFIN;
            8'd44:                                            op_of = OP_GHD;
            default:                                          op_of = OP_DONE;
        endcase
    endfunction
    function automatic [3:0] layer_of(input [7:0] pc);
        case (pc)
            8'd7,  8'd8,  8'd9,  8'd10, 8'd11, 8'd12: layer_of = 4'd1;
            8'd13, 8'd14, 8'd15, 8'd16, 8'd17, 8'd18: layer_of = 4'd2;
            8'd19, 8'd20, 8'd21, 8'd22, 8'd23, 8'd24: layer_of = 4'd3;
            8'd25, 8'd26, 8'd27, 8'd28, 8'd29, 8'd30: layer_of = 4'd4;
            8'd31, 8'd32, 8'd33, 8'd34, 8'd35, 8'd36: layer_of = 4'd5;
            8'd37, 8'd38, 8'd39, 8'd40, 8'd41, 8'd42: layer_of = 4'd6;
            default:                                  layer_of = 4'd0;
        endcase
    endfunction
    // 0 none,1 emb,2 norm,3 gemv,4 conv,5 scan
    function automatic [2:0] wrk_of(input [3:0] op);
        case (op)
            OP_EMB:  wrk_of = 1;
            OP_NPRE, OP_NGATE, OP_NFIN: wrk_of = 2;
            OP_GIN, OP_GOUT, OP_GHD:    wrk_of = 3;
            OP_CONV: wrk_of = 4;
            OP_SCAN: wrk_of = 5;
            default: wrk_of = 0;
        endcase
    endfunction

    reg [7:0]  op_pc  [0:NC-1];
    reg [$clog2(TMAX+1)-1:0] tokcnt [0:NC-1];
    reg        active [0:NC-1];               // still has tokens to decode
    reg        busy   [0:NC-1];               // currently held by a worker
    reg [SW-1:0] rr [0:5];            // per-worker round-robin ptr

    reg started; reg all_done;

    // ------------------------------------------------------------- init ------
    // one-time: stream the SiLU LUT into conv + norm cores (mamba_seq S_LUTLD)
    reg [1:0]  ist;                            // 0 idle,1 lutld,2 ready
    reg [7:0]  i_i;

    // ---------------------------------------------------------- workers ------
    integer kk, ss;
    reg found; reg [SW-1:0] pick;
    reg [3:0] clv;                            // CONV dispatch: chosen layer (skip test)

    // EMB worker
    localparam [1:0] E_IDLE=0, E_RUN=1;
    reg [1:0]  est; reg [SW-1:0] e_st_s;
    reg [11:0] e_i; reg [2:0] e_sub;
    reg signed [31:0] e_acc; reg [15:0] e_fp; reg [63:0] e_embq; reg e_embsel;
    reg [9:0]  e_tok;
    wire [14:0] e_emb_wa = e_tok*(D/8) + e_i[11:3];
    wire [31:0] e_word = e_embsel ? e_embq[63:32] : e_embq[31:0];

    // NORM worker
    localparam [2:0] N_IDLE=0, N_LD=1, N_RUN=2, N_QUANT=3;
    reg [2:0]  nst; reg [SW-1:0] n_st_s; reg [3:0] n_li; reg [1:0] n_op;
    reg [11:0] n_i; reg [2:0] n_sub;
    reg signed [47:0] n_t14 [0:3];
    reg [$clog2(TMAX+1)-1:0] n_tok;
    wire [31:0] n_c_inr  = consts[n_li*16 + 0];
    wire [31:0] n_c_outr = consts[n_li*16 + 2];
    wire [31:0] n_c_hdr  = consts[112];
    // pre(0)->in-act recip, gate(1)->out-act recip (the P=4 wide-quant path);
    // pre-selected into a wire so the requant multiply avoids a ternary inside a
    // concatenation (iverilog-legal but Vivado's parser rejects that form).
    wire [31:0] n_qrecip = (n_op == 2'd0) ? n_c_inr : n_c_outr;

    // GEMV worker
    localparam [2:0] G_IDLE=0, G_LD=1, G_RUN=2, G_RD=3;
    reg [2:0]  gst; reg [SW-1:0] g_st_s; reg [3:0] g_li; reg [1:0] g_op;
    reg [11:0] g_i; reg [2:0] g_sub;
    reg signed [31:0] g_acc4 [0:3];
    reg        [15:0] g_fp4  [0:3];
    reg signed [47:0] g_t14  [0:3];
    reg signed [15:0] g_lv0, g_lv1, g_lv2, g_lv3;
    reg signed [15:0] g_best; reg [9:0] g_besti;
    reg signed [15:0] g_nb;   reg [9:0] g_ni;
    reg [$clog2(TMAX+1)-1:0] g_tok;
    wire [31:0] g_c_ins  = consts[g_li*16 + 1];
    wire [31:0] g_c_outs = consts[g_li*16 + 3];
    wire [31:0] g_c_hds  = consts[113];

    // CONV worker
    localparam [2:0] C_IDLE=0, C_LDW=1, C_LDX=2, C_RUN=3, C_RD=4;
    reg [2:0]  cst; reg [SW-1:0] c_st_s; reg [3:0] c_li;
    reg [11:0] c_i; reg [2:0] c_sub; reg signed [47:0] c_t1;
    // CONV_LDW skip: conv_silu.wrom holds ONE layer's weights (shared across all
    // NC streams — weights are stream-independent) and PERSISTS across runs, so
    // re-stream it only when the loaded layer changes. cw_valid/cw_layer track
    // what is currently resident; a same-layer dispatch skips the 2560-cyc reload
    // (amortising the conv-weight load ~/NC as the wave clusters same-layer convs).
    reg [3:0]  cw_layer; reg cw_valid;

    // SCAN worker
    localparam [3:0] SC_IDLE=0, SC_PREP=1, SC_H=2, SC_HWAIT=3, SC_RD=4;
    reg [3:0]  sst; reg [SW-1:0] s_st_s; reg [3:0] s_li; reg [3:0] s_hi;
    reg [11:0] sc_i; reg [3:0] sc_sub;
    reg [15:0] sc_dtq; reg [4:0] sc_eh;
    reg signed [31:0] sc_acc; reg signed [47:0] sc_t1; reg [15:0] sc_fp;
    reg signed [15:0] sc_dtq3;
    reg signed [47:0] sc_b1, sc_c1;              // SC_PREP: B/C shift temps
    reg signed [31:0] sc_yacc [0:3];             // SC_RD P4: 4 scan-y lanes
    reg signed [15:0] sc_dsk;                    // SC_RD P4: per-head D-skip const
    reg        [63:0] sc_xrow;                   // SC_RD P4: xnbuf row (4 lanes)
    wire [3:0] s_bB = consts[s_li*16 + 4][3:0];
    wire [3:0] s_bC = consts[s_li*16 + 4][7:4];
    wire [3:0] s_bX = consts[s_li*16 + 4][11:8];
    // dt_raw for the scan — read from the wide-word zxbuf inside the clocked
    // block (a continuous assign calling zx_rd would not re-trigger on zxbuf_w
    // writes, since the function's only sensitivity is its argument).
    reg  signed [15:0] s_dtraw;

    // ============================================ xbuf residual banks =========
    // Residual Q6.19, split into NC per-stream single-write-port wide-word banks
    // (4x32b rows). Two writers — EMB init (staged 4 elems -> 1 row) and GEMV
    // out_proj RMW (read old row, add 4 lanes, write) — can target DIFFERENT
    // streams the same cycle, so each bank has its OWN write port muxed between
    // the two (mutually exclusive: a stream is held by exactly one worker). The
    // per-bank write commits one cycle after the intent — safe, since out_proj
    // touches strictly increasing rows (no back-to-back same-row) and EMB writes
    // every row exactly once at token start. Reads stay async (no schedule change).
    localparam int XW = D/4;                   // 64 rows/stream
    localparam int XRW = (XW <= 1) ? 1 : $clog2(XW);
    reg              xw_emb_we; reg [SW-1:0] xw_emb_s; reg [XRW-1:0] xw_emb_r; reg [127:0] xw_emb_d;
    reg              xw_gem_we; reg [SW-1:0] xw_gem_s; reg [XRW-1:0] xw_gem_r; reg [127:0] xw_gem_d;
    reg [127:0]      emb_stage;                // EMB row accumulator (lanes 0..2)
    reg signed [31:0] e_val;                   // EMB per-element value (blocking)
    reg signed [31:0] n_xv;                    // NORM xbuf read (blocking)
    reg signed [31:0] g_xo0, g_xo1, g_xo2, g_xo3; // GEMV RMW old residual lanes
    wire [XRW-1:0]   xr_norm_r = n_i[XRW+1:2]; // NORM element row  (n_i>>2)
    wire [XRW-1:0]   xr_gem_r  = g_i[XRW+1:2]; // GEMV element row  (g_i>>2)
    wire [127:0]     xrow_norm [0:NC-1];
    wire [127:0]     xrow_gem  [0:NC-1];
    genvar xbi;
    generate for (xbi = 0; xbi < NC; xbi = xbi + 1) begin : xbk
        (* ram_style = "distributed" *)
        reg [127:0] mem [0:XW-1];
        always @(posedge clk) begin
            if      (xw_emb_we && xw_emb_s == xbi[SW-1:0]) mem[xw_emb_r] <= xw_emb_d;
            else if (xw_gem_we && xw_gem_s == xbi[SW-1:0]) mem[xw_gem_r] <= xw_gem_d;
        end
        assign xrow_norm[xbi] = mem[xr_norm_r];
        assign xrow_gem [xbi] = mem[xr_gem_r];
    end endgenerate
    // element reads: pick the holding stream's bank row, slice the element (a
    // plain-reg part-select — the safe variable +: form).
    function automatic signed [31:0] xrd_norm(input [7:0] e);
        reg [127:0] row; begin row = xrow_norm[n_st_s]; xrd_norm = row[e[1:0]*32 +: 32]; end
    endfunction
    function automatic signed [31:0] xrd_gem(input [7:0] e);
        reg [127:0] row; begin row = xrow_gem[g_st_s]; xrd_gem = row[e[1:0]*32 +: 32]; end
    endfunction

    // q8buf funnelled write (one wide-word port, muxed between NORM's two sites)
    reg          q8w_we; reg [15:0] q8w_row; reg [31:0] q8w_word;
    reg [31:0]   q8_stage;                     // final-path byte accumulator
    reg signed [7:0] q8_byte;                  // final-path per-byte (blocking)
    always @(posedge clk) if (q8w_we) q8buf_w[q8w_row] <= q8w_word;

    // ---- per-stream token boundary + dispatch helpers -----------------------
    // (computed in the clocked block below)

    integer w;
    always @(posedge clk) begin
        done <= 1'b0;
        g_start <= 0; c_start <= 0; s_start <= 0; n_start <= 0;
        g_wrx <= 0; c_wrw <= 0; c_wrb <= 0; c_wrx <= 0; c_wrl <= 0;
        s_wrdtx <= 0; s_wrb <= 0; s_wrc <= 0;
        n_wry <= 0; n_wrz <= 0; n_wrg <= 0; n_wrl <= 0;
        xw_emb_we <= 0; xw_gem_we <= 0; q8w_we <= 0;

        if (rst) begin
            ist <= 0; i_i <= 0; started <= 0; all_done <= 0; cyc_count <= 0;
            est <= E_IDLE; nst <= N_IDLE; gst <= G_IDLE; cst <= C_IDLE;
            sst <= SC_IDLE; cw_valid <= 1'b0;
            for (w = 0; w < NC; w = w + 1) begin
                op_pc[w] <= 0; tokcnt[w] <= 0; active[w] <= 0; busy[w] <= 0;
            end
            for (w = 0; w < 6; w = w + 1) rr[w] <= 0;
        end else begin
            // measured cycle counter: from first dispatch to all-done
            if (started && !all_done) cyc_count <= cyc_count + 1;

            // ---------- init: SiLU LUT into conv + norm, then arm streams -----
            if (ist == 0) begin
                if (start && ready) begin ist <= 1; i_i <= 0; end
            end else if (ist == 1) begin
                c_wrl <= 1; c_wrl_a <= i_i; c_wrd <= slut[i_i];
                n_wrl <= 1; n_wrl_a <= i_i; n_wrd <= slut[i_i];
                if (i_i == 255) begin
                    ist <= 2;
                    for (w = 0; w < NC; w = w + 1) begin
                        active[w] <= 1; op_pc[w] <= 0; tokcnt[w] <= 0; busy[w] <= 0;
                    end
                end else i_i <= i_i + 1;
            end

            // =============================================== DISPATCH =========
            // For each idle worker, round-robin pick a ready stream whose next op
            // targets that worker. A stream held by a worker is `busy`; a stream
            // whose op_pc reached NPC is retired/advanced by the token-boundary
            // block below. Dispatch reads registered state (1-cycle handoff bubble).
            if (ist == 2) begin
                // ---- EMB worker ----
                if (est == E_IDLE) begin
                    found = 0; pick = 0;
                    for (kk = 0; kk < NC; kk = kk + 1) begin
                        ss = rr[1] + 1 + kk; if (ss >= NC) ss = ss - NC;  // %NC, divider-free (rr+1+kk < 2NC)
                        if (!found && active[ss] && !busy[ss]
                            && wrk_of(op_of(op_pc[ss])) == 3'd1) begin
                            found = 1; pick = ss[SW-1:0];
                        end
                    end
                    if (found) begin
                        busy[pick] <= 1; rr[1] <= pick; started <= 1;
                        e_st_s <= pick; e_tok <= tok_seq[pick*TMAX + tokcnt[pick]];
                        e_i <= 0; e_sub <= 0; est <= E_RUN;
                    end
                end
                // ---- NORM worker ----
                if (nst == N_IDLE) begin
                    found = 0; pick = 0;
                    for (kk = 0; kk < NC; kk = kk + 1) begin
                        ss = rr[2] + 1 + kk; if (ss >= NC) ss = ss - NC;  // %NC, divider-free
                        if (!found && active[ss] && !busy[ss]
                            && wrk_of(op_of(op_pc[ss])) == 3'd2) begin
                            found = 1; pick = ss[SW-1:0];
                        end
                    end
                    if (found) begin
                        busy[pick] <= 1; rr[2] <= pick; started <= 1;
                        n_st_s <= pick; n_li <= layer_of(op_pc[pick]);
                        n_tok <= tokcnt[pick];
                        n_op <= (op_of(op_pc[pick])==OP_NPRE) ? 2'd0 :
                                (op_of(op_pc[pick])==OP_NGATE)? 2'd1 : 2'd2;
                        n_i <= 0; n_sub <= 0; nst <= N_LD;
                        n_gated <= (op_of(op_pc[pick])==OP_NGATE);
                        n_short <= (op_of(op_pc[pick])!=OP_NGATE);
                    end
                end
                // ---- GEMV worker ----
                if (gst == G_IDLE) begin
                    found = 0; pick = 0;
                    for (kk = 0; kk < NC; kk = kk + 1) begin
                        ss = rr[3] + 1 + kk; if (ss >= NC) ss = ss - NC;  // %NC, divider-free
                        if (!found && active[ss] && !busy[ss]
                            && wrk_of(op_of(op_pc[ss])) == 3'd3) begin
                            found = 1; pick = ss[SW-1:0];
                        end
                    end
                    if (found) begin
                        busy[pick] <= 1; rr[3] <= pick; started <= 1;
                        g_st_s <= pick; g_li <= layer_of(op_pc[pick]);
                        g_tok <= tokcnt[pick];
                        g_op <= (op_of(op_pc[pick])==OP_GIN) ? 2'd0 :
                                (op_of(op_pc[pick])==OP_GOUT)? 2'd1 : 2'd2;
                        g_i <= 0; g_sub <= 0; gst <= G_LD;
                    end
                end
                // ---- CONV worker ----
                if (cst == C_IDLE) begin
                    found = 0; pick = 0;
                    for (kk = 0; kk < NC; kk = kk + 1) begin
                        ss = rr[4] + 1 + kk; if (ss >= NC) ss = ss - NC;  // %NC, divider-free
                        if (!found && active[ss] && !busy[ss]
                            && wrk_of(op_of(op_pc[ss])) == 3'd4) begin
                            found = 1; pick = ss[SW-1:0];
                        end
                    end
                    if (found) begin
                        busy[pick] <= 1; rr[4] <= pick; started <= 1;
                        clv = layer_of(op_pc[pick]);
                        c_st_s <= pick; c_li <= clv;
                        c_layer <= pick*LR + clv;    // history bank (set here so a
                                                     // C_LDW skip still selects it)
                        c_i <= 0; c_sub <= 0;
                        // skip the weight re-stream iff this layer's weights are
                        // already resident in conv_silu.wrom.
                        cst <= (cw_valid && cw_layer == clv) ? C_LDX : C_LDW;
                    end
                end
                // ---- SCAN worker ----
                if (sst == SC_IDLE) begin
                    found = 0; pick = 0;
                    for (kk = 0; kk < NC; kk = kk + 1) begin
                        ss = rr[5] + 1 + kk; if (ss >= NC) ss = ss - NC;  // %NC, divider-free
                        if (!found && active[ss] && !busy[ss]
                            && wrk_of(op_of(op_pc[ss])) == 3'd5) begin
                            found = 1; pick = ss[SW-1:0];
                        end
                    end
                    if (found) begin
                        busy[pick] <= 1; rr[5] <= pick; started <= 1;
                        s_st_s <= pick; s_li <= layer_of(op_pc[pick]);
                        sc_i <= 0; sc_sub <= 0; s_hi <= 0; sst <= SC_PREP;
                    end
                end

                // ---- token-boundary / all-done check --------------------------
                all_done <= 1'b1;
                for (w = 0; w < NC; w = w + 1) begin
                    if (active[w]) all_done <= 1'b0;
                end
                if (started && all_done) done <= 1'b1;
            end

            // ===================================================== EMB ========
            case (est)
              E_RUN: begin
                case (e_sub)
                  0: begin e_acc <= 32'sd0; e_sub <= 1;
                       e_fp <= esc[e_tok];
                       e_embq   <= emb[e_emb_wa[14:1]];
                       e_embsel <= e_emb_wa[0];
                  end
                  1: begin
                       e_acc <= $signed({{28{e_word[(e_i[2:0])*4+3]}},
                                 e_word[(e_i[2:0])*4 +: 4]});
                       e_sub <= 2;
                  end
                  2: begin
                       // stage 4 elements into one wide row; write it every 4th
                       // (D is a multiple of 4, so e_i==D-1 lands on lane 3).
                       e_val = sat32f(rshr(mul_m11(e_acc, e_fp),
                                      8'sd6 - $signed({3'b0, e_fp[14:10]})));
                       if (e_i[1:0] == 2'd3) begin
                           xw_emb_we <= 1; xw_emb_s <= e_st_s;
                           xw_emb_r  <= e_i[XRW+1:2];
                           xw_emb_d  <= {e_val, emb_stage[95:0]};
                       end else
                           emb_stage[e_i[1:0]*32 +: 32] <= e_val;
                       e_sub <= 0;
                       if (e_i == D-1) begin
                           est <= E_IDLE;
                           op_pc[e_st_s] <= op_pc[e_st_s] + 1;
                           busy[e_st_s] <= 0;
                       end else e_i <= e_i + 1;
                  end
                endcase
              end
              default: ;
            endcase

            // ===================================================== NORM =======
            case (nst)
              N_LD: begin
                // COLLAPSED feed: all writes for element n_i in ONE cycle (the
                // y/z/g ports are independent), n_i advances every cycle.
                if (n_op == 2'd1) begin       // gated (512): y + z + g
                  n_wry <= 1; n_wry_a <= n_i[9:0];
                  n_wry_d <= yb_rd(n_st_s*DIN + n_i[8:0]);
                  n_wrz <= 1; n_wrz_a <= n_i[9:0];
                  n_wrz_d <= zx_rd(n_st_s*INROWS + n_i[8:0]);
                  n_wrg <= 1; n_wrg_a <= n_i[9:0];
                  n_wrg_d <= nrmg[n_li*DIN + n_i[8:0]];
                  if (n_i == DIN-1) begin n_i <= 0; nst <= N_RUN; n_start <= 1; end
                  else n_i <= n_i + 1;
                end else begin                // pre / final (256), ungated: y + g
                  n_xv = xrd_norm(n_i[7:0]);
                  n_wry <= 1; n_wry_a <= n_i[9:0];
                  n_wry_d <= sat16f(rshr($signed({{16{n_xv[31]}}, n_xv}), 8'sd10));
                  // final-norm: snapshot residual (l6.x_out) into the dump
                  if (n_op == 2'd2)
                      dump_x[(n_st_s*TMAX + n_tok)*D + n_i[7:0]] <= n_xv;
                  n_wrg <= 1; n_wrg_a <= n_i[9:0];
                  n_wrg_d <= (n_op==2'd2) ? nfg[n_i[7:0]] : lng[n_li*D + n_i[7:0]];
                  if (n_i == D-1) begin n_i <= 0; nst <= N_RUN; n_start <= 1; end
                  else n_i <= n_i + 1;
                end
              end
              N_RUN: if (n_done) begin nst <= N_QUANT; n_i <= 0; n_sub <= 0; end
              N_QUANT: begin
                if (n_op == 2'd2) begin       // final: P=1, c_hdr -> q8buf
                  case (n_sub)
                    0: begin n_rda <= n_i[9:0]; n_sub <= 1; end
                    1: begin
                         n_t14[0] <= rshr($signed(n_out) * $signed({1'b0, n_c_hdr[15:0]}),
                                     $signed({1'b0, n_c_hdr[23:16]}) + 8'sd11);
                         n_sub <= 2;
                    end
                    2: begin
                         // final: P=1 -> stage into a wide-word row, one row write
                         // per 4 bytes (D%4==0, so n_i==D-1 lands on byte 3).
                         q8_byte = (n_t14[0] > 48'sd127) ? 8'sd127 :
                                   (n_t14[0] < -48'sd128) ? -8'sd128 : n_t14[0][7:0];
                         if (n_i[1:0] == 2'd3) begin
                             q8w_we <= 1; q8w_row <= (n_st_s*DIN + n_i[8:0]) >> 2;
                             q8w_word <= {q8_byte, q8_stage[23:0]};
                         end else
                             q8_stage[n_i[1:0]*8 +: 8] <= q8_byte;
                         n_sub <= 0;
                         if (n_i == D-1) begin
                             nst <= N_IDLE; op_pc[n_st_s] <= op_pc[n_st_s] + 1;
                             busy[n_st_s] <= 0;
                         end else n_i <= n_i + 1;
                    end
                  endcase
                end else begin                // pre(256)/gate(512): P=4, wide read
                  case (n_sub)
                    0: begin n_rdaw <= n_i[8:0]; n_sub <= 1; end
                    1: begin
                         n_t14[0] <= rshr($signed(n_ow[15:0])  * $signed({1'b0, n_qrecip[15:0]}), $signed({1'b0, n_qrecip[23:16]}) + 8'sd12);
                         n_t14[1] <= rshr($signed(n_ow[31:16]) * $signed({1'b0, n_qrecip[15:0]}), $signed({1'b0, n_qrecip[23:16]}) + 8'sd12);
                         n_t14[2] <= rshr($signed(n_ow[47:32]) * $signed({1'b0, n_qrecip[15:0]}), $signed({1'b0, n_qrecip[23:16]}) + 8'sd12);
                         n_t14[3] <= rshr($signed(n_ow[63:48]) * $signed({1'b0, n_qrecip[15:0]}), $signed({1'b0, n_qrecip[23:16]}) + 8'sd12);
                         n_sub <= 2;
                    end
                    2: begin
                         // pre/gate: P=4 -> one wide-word row write (n_i mult of 4)
                         q8w_we <= 1; q8w_row <= (n_st_s*DIN + n_i[8:0]) >> 2;
                         q8w_word <= {
                             ((n_t14[3] > 48'sd127) ? 8'sd127 : (n_t14[3] < -48'sd128) ? -8'sd128 : n_t14[3][7:0]),
                             ((n_t14[2] > 48'sd127) ? 8'sd127 : (n_t14[2] < -48'sd128) ? -8'sd128 : n_t14[2][7:0]),
                             ((n_t14[1] > 48'sd127) ? 8'sd127 : (n_t14[1] < -48'sd128) ? -8'sd128 : n_t14[1][7:0]),
                             ((n_t14[0] > 48'sd127) ? 8'sd127 : (n_t14[0] < -48'sd128) ? -8'sd128 : n_t14[0][7:0])};
                         n_sub <= 0;
                         if (n_i >= (n_op==2'd1 ? DIN-4 : D-4)) begin
                             nst <= N_IDLE; op_pc[n_st_s] <= op_pc[n_st_s] + 1;
                             busy[n_st_s] <= 0;
                         end else n_i <= n_i + 4;
                    end
                  endcase
                end
              end
              default: ;
            endcase

            // ===================================================== GEMV =======
            case (gst)
              G_LD: begin
                g_wrx <= 1; g_wrx_a <= g_i[8:0];
                g_wrx_d <= q8_rd(g_st_s*DIN + g_i[8:0]);
                if (g_i == DIN-1) begin
                    g_i <= 0; gst <= G_RUN; g_start <= 1;
                    if (g_op == 2'd0) begin
                        g_base <= g_li * (INROWS * (D/8));
                        g_rows <= INROWS; g_wpr <= D/8;
                    end else if (g_op == 2'd1) begin
                        g_base <= consts[120][18:0] + g_li * (D * (DIN/8));
                        g_rows <= D; g_wpr <= DIN/8;
                    end else begin
                        g_base <= consts[121][18:0];
                        g_rows <= V; g_wpr <= D/8;
                    end
                end else g_i <= g_i + 1;
              end
              G_RUN: if (g_done) begin
                  gst <= G_RD; g_i <= 0; g_sub <= 0;
                  g_best <= -16'sd32768; g_besti <= 0;
              end
              G_RD: begin
                if (g_op == 2'd0) begin        // in_proj dequant -> zxbuf
                  case (g_sub)
                    0: begin g_rdaw <= g_i[10:0]; g_sub <= 1; end
                    1: begin
                         g_acc4[0] <= $signed(g_accw[ 31:  0]);
                         g_acc4[1] <= $signed(g_accw[ 63: 32]);
                         g_acc4[2] <= $signed(g_accw[ 95: 64]);
                         g_acc4[3] <= $signed(g_accw[127: 96]);
                         g_fp4[0]  <= rsin[g_li*INROWS + g_i[10:0]];
                         g_fp4[1]  <= rsin[g_li*INROWS + g_i[10:0] + 1];
                         g_fp4[2]  <= rsin[g_li*INROWS + g_i[10:0] + 2];
                         g_fp4[3]  <= rsin[g_li*INROWS + g_i[10:0] + 3];
                         g_sub <= 2;
                    end
                    2: begin
                         g_t14[0] <= rshr(mul_m11(g_acc4[0], g_fp4[0]), 8'sd10 - $signed({3'b0, g_fp4[0][14:10]}));
                         g_t14[1] <= rshr(mul_m11(g_acc4[1], g_fp4[1]), 8'sd10 - $signed({3'b0, g_fp4[1][14:10]}));
                         g_t14[2] <= rshr(mul_m11(g_acc4[2], g_fp4[2]), 8'sd10 - $signed({3'b0, g_fp4[2][14:10]}));
                         g_t14[3] <= rshr(mul_m11(g_acc4[3], g_fp4[3]), 8'sd10 - $signed({3'b0, g_fp4[3][14:10]}));
                         g_sub <= 3;
                    end
                    3: begin
                         // wide-word: pack the 4 dequant lanes into one row write
                         // (g_i is a multiple of 4, so this covers g_i..g_i+3).
                         zxbuf_w[(g_st_s*INROWS + g_i[10:0]) >> 2] <= {
                             sat16f(rshr(g_t14[3] * $signed({1'b0, g_c_ins[15:0]}), $signed({1'b0, g_c_ins[23:16]}) + 8'sd15 - 8'sd9)),
                             sat16f(rshr(g_t14[2] * $signed({1'b0, g_c_ins[15:0]}), $signed({1'b0, g_c_ins[23:16]}) + 8'sd15 - 8'sd9)),
                             sat16f(rshr(g_t14[1] * $signed({1'b0, g_c_ins[15:0]}), $signed({1'b0, g_c_ins[23:16]}) + 8'sd15 - 8'sd9)),
                             sat16f(rshr(g_t14[0] * $signed({1'b0, g_c_ins[15:0]}), $signed({1'b0, g_c_ins[23:16]}) + 8'sd15 - 8'sd9))};
                         g_sub <= 0;
                         if (g_i >= INROWS-4) begin
                             gst <= G_IDLE; op_pc[g_st_s] <= op_pc[g_st_s] + 1;
                             busy[g_st_s] <= 0;
                         end else g_i <= g_i + 4;
                    end
                  endcase
                end else if (g_op == 2'd1) begin   // out_proj dequant + residual
                  case (g_sub)
                    0: begin g_rdaw <= g_i[10:0]; g_sub <= 1; end
                    1: begin
                         g_acc4[0] <= $signed(g_accw[ 31:  0]);
                         g_acc4[1] <= $signed(g_accw[ 63: 32]);
                         g_acc4[2] <= $signed(g_accw[ 95: 64]);
                         g_acc4[3] <= $signed(g_accw[127: 96]);
                         g_fp4[0]  <= rsout[g_li*D + g_i[7:0]];
                         g_fp4[1]  <= rsout[g_li*D + g_i[7:0] + 1];
                         g_fp4[2]  <= rsout[g_li*D + g_i[7:0] + 2];
                         g_fp4[3]  <= rsout[g_li*D + g_i[7:0] + 3];
                         g_sub <= 2;
                    end
                    2: begin
                         g_t14[0] <= rshr(mul_m11(g_acc4[0], g_fp4[0]), 8'sd10 - $signed({3'b0, g_fp4[0][14:10]}));
                         g_t14[1] <= rshr(mul_m11(g_acc4[1], g_fp4[1]), 8'sd10 - $signed({3'b0, g_fp4[1][14:10]}));
                         g_t14[2] <= rshr(mul_m11(g_acc4[2], g_fp4[2]), 8'sd10 - $signed({3'b0, g_fp4[2][14:10]}));
                         g_t14[3] <= rshr(mul_m11(g_acc4[3], g_fp4[3]), 8'sd10 - $signed({3'b0, g_fp4[3][14:10]}));
                         g_sub <= 3;
                    end
                    3: begin
                         // wide-word RMW: read the old residual row (async), add
                         // the 4 out_proj lanes, write the row back to the bank.
                         g_xo0 = xrd_gem(g_i[7:0]);
                         g_xo1 = xrd_gem(g_i[7:0] + 1);
                         g_xo2 = xrd_gem(g_i[7:0] + 2);
                         g_xo3 = xrd_gem(g_i[7:0] + 3);
                         xw_gem_we <= 1; xw_gem_s <= g_st_s; xw_gem_r <= g_i[XRW+1:2];
                         xw_gem_d  <= {
                             sat32f($signed({{16{g_xo3[31]}}, g_xo3}) + rshr(g_t14[3] * $signed({1'b0, g_c_outs[15:0]}), $signed({1'b0, g_c_outs[23:16]}) + 8'sd15 - 8'sd19)),
                             sat32f($signed({{16{g_xo2[31]}}, g_xo2}) + rshr(g_t14[2] * $signed({1'b0, g_c_outs[15:0]}), $signed({1'b0, g_c_outs[23:16]}) + 8'sd15 - 8'sd19)),
                             sat32f($signed({{16{g_xo1[31]}}, g_xo1}) + rshr(g_t14[1] * $signed({1'b0, g_c_outs[15:0]}), $signed({1'b0, g_c_outs[23:16]}) + 8'sd15 - 8'sd19)),
                             sat32f($signed({{16{g_xo0[31]}}, g_xo0}) + rshr(g_t14[0] * $signed({1'b0, g_c_outs[15:0]}), $signed({1'b0, g_c_outs[23:16]}) + 8'sd15 - 8'sd19))};
                         g_sub <= 0;
                         if (g_i >= D-4) begin
                             gst <= G_IDLE; op_pc[g_st_s] <= op_pc[g_st_s] + 1;
                             busy[g_st_s] <= 0;
                         end else g_i <= g_i + 4;
                    end
                  endcase
                end else begin                 // head dequant + argmax -> logit/dump
                  case (g_sub)
                    0: begin g_rdaw <= g_i[10:0]; g_sub <= 1; end
                    1: begin
                         g_acc4[0] <= $signed(g_accw[ 31:  0]);
                         g_acc4[1] <= $signed(g_accw[ 63: 32]);
                         g_acc4[2] <= $signed(g_accw[ 95: 64]);
                         g_acc4[3] <= $signed(g_accw[127: 96]);
                         g_fp4[0]  <= rshd[g_i[9:0]];
                         g_fp4[1]  <= rshd[g_i[9:0] + 1];
                         g_fp4[2]  <= rshd[g_i[9:0] + 2];
                         g_fp4[3]  <= rshd[g_i[9:0] + 3];
                         g_sub <= 2;
                    end
                    2: begin
                         g_t14[0] <= rshr(mul_m11(g_acc4[0], g_fp4[0]), 8'sd10 - $signed({3'b0, g_fp4[0][14:10]}));
                         g_t14[1] <= rshr(mul_m11(g_acc4[1], g_fp4[1]), 8'sd10 - $signed({3'b0, g_fp4[1][14:10]}));
                         g_t14[2] <= rshr(mul_m11(g_acc4[2], g_fp4[2]), 8'sd10 - $signed({3'b0, g_fp4[2][14:10]}));
                         g_t14[3] <= rshr(mul_m11(g_acc4[3], g_fp4[3]), 8'sd10 - $signed({3'b0, g_fp4[3][14:10]}));
                         g_sub <= 3;
                    end
                    3: begin
                         g_lv0 = sat16f(rshr(g_t14[0] * $signed({1'b0, g_c_hds[15:0]}), $signed({1'b0, g_c_hds[23:16]}) + 8'sd15 - 8'sd12));
                         g_lv1 = sat16f(rshr(g_t14[1] * $signed({1'b0, g_c_hds[15:0]}), $signed({1'b0, g_c_hds[23:16]}) + 8'sd15 - 8'sd12));
                         g_lv2 = sat16f(rshr(g_t14[2] * $signed({1'b0, g_c_hds[15:0]}), $signed({1'b0, g_c_hds[23:16]}) + 8'sd15 - 8'sd12));
                         g_lv3 = sat16f(rshr(g_t14[3] * $signed({1'b0, g_c_hds[15:0]}), $signed({1'b0, g_c_hds[23:16]}) + 8'sd15 - 8'sd12));
                         logit[g_st_s*V + g_i[9:0]]     <= g_lv0;
                         logit[g_st_s*V + g_i[9:0] + 1] <= g_lv1;
                         logit[g_st_s*V + g_i[9:0] + 2] <= g_lv2;
                         logit[g_st_s*V + g_i[9:0] + 3] <= g_lv3;
                         dump_logit[(g_st_s*TMAX+g_tok)*V + g_i[9:0]]     <= g_lv0;
                         dump_logit[(g_st_s*TMAX+g_tok)*V + g_i[9:0] + 1] <= g_lv1;
                         dump_logit[(g_st_s*TMAX+g_tok)*V + g_i[9:0] + 2] <= g_lv2;
                         dump_logit[(g_st_s*TMAX+g_tok)*V + g_i[9:0] + 3] <= g_lv3;
                         g_nb = g_best; g_ni = g_besti;
                         if (g_lv0 > g_nb) begin g_nb = g_lv0; g_ni = g_i[9:0];     end
                         if (g_lv1 > g_nb) begin g_nb = g_lv1; g_ni = g_i[9:0] + 1; end
                         if (g_lv2 > g_nb) begin g_nb = g_lv2; g_ni = g_i[9:0] + 2; end
                         if (g_lv3 > g_nb) begin g_nb = g_lv3; g_ni = g_i[9:0] + 3; end
                         g_best <= g_nb; g_besti <= g_ni;
                         g_sub <= 0;
                         if (g_i >= V-4) begin
                             gst <= G_IDLE; op_pc[g_st_s] <= NPC;  // token complete
                             busy[g_st_s] <= 0;
                             // latch the final argmax token (compact, always kept)
                             dump_tok[g_st_s*TMAX + g_tok] <= g_ni;
                         end else g_i <= g_i + 4;
                    end
                  endcase
                end
              end
              default: ;
            endcase

            // ===================================================== CONV =======
            case (cst)
              C_LDW: begin                       // re-stream this layer's weights
                c_wrw <= 1; c_wrw_a <= c_i[11:0];
                c_wrd <= convw[c_li*CONVD*4 + c_i[11:0]];
                if (c_i == CONVD*4-1) begin
                    c_i <= 0; cst <= C_LDX; c_sub <= 0;
                    cw_valid <= 1'b1; cw_layer <= c_li;   // now resident
                end else c_i <= c_i + 1;
              end
              C_LDX: begin                       // COLLAPSED: bias + x in one cycle
                c_wrb <= 1; c_wrb_a <= c_i[9:0];
                c_wrb_d <= convb[c_li*CONVD + c_i[9:0]];
                c_wrx <= 1; c_wrx_a <= c_i[9:0];
                c_wrx_d <= zx_rd(c_st_s*INROWS + DIN + c_i[9:0]);
                if (c_i == CONVD-1) begin c_i <= 0; cst <= C_RUN; c_start <= 1; end
                else c_i <= c_i + 1;
              end
              C_RUN: if (c_done) begin cst <= C_RD; c_i <= 0; c_sub <= 0; end
              C_RD: begin                        // P=4 wide read -> one xnbuf row/cyc
                case (c_sub)
                  0: begin c_rdaw <= c_i[9:0]; c_sub <= 1; end
                  1: begin
                       xnbuf_w[(c_st_s*CONVD + c_i[9:0]) >> 2] <= c_yw;
                       c_sub <= 0;
                       if (c_i >= CONVD-4) begin
                           cst <= C_IDLE; op_pc[c_st_s] <= op_pc[c_st_s] + 1;
                           busy[c_st_s] <= 0;
                       end else c_i <= c_i + 4;
                  end
                endcase
              end
              default: ;
            endcase

            // ===================================================== SCAN =======
            case (sst)
              SC_PREP: begin
                // COLLAPSED: B and C prep in ONE cycle (independent ports, both
                // reads + shifts combinational). NST cycles instead of 4*NST.
                sc_b1 = rshr($signed(xn_rd(s_st_s*CONVD + DIN + sc_i[6:0])),
                             $signed(8'sd11) - $signed({4'b0, s_bB}));
                sc_c1 = rshr($signed(xn_rd(s_st_s*CONVD + DIN + NST + sc_i[6:0])),
                             $signed(8'sd11) - $signed({4'b0, s_bC}));
                s_wrb <= 1; s_wrb_a <= sc_i[5:0];
                s_b_d <= (sc_b1 > 48'sd127) ? 8'sd127 :
                         (sc_b1 < -48'sd128) ? -8'sd128 : sc_b1[7:0];
                s_wrc <= 1; s_wrc_a <= sc_i[5:0];
                s_c_d <= (sc_c1 > 48'sd127) ? 8'sd127 :
                         (sc_c1 < -48'sd128) ? -8'sd128 : sc_c1[7:0];
                if (sc_i == NST-1) begin sc_i <= 0; s_hi <= 0; sst <= SC_H; sc_sub <= 0; end
                else sc_i <= sc_i + 1;
              end
              SC_H: begin
                case (sc_sub)
                  0: begin
                       s_dtraw = zx_rd(s_st_s*INROWS + 2*DIN + 2*NST + s_hi[2:0]);
                       sc_dtq3 = sat16f($signed({{32{s_dtraw[15]}}, s_dtraw}) <<< 3);
                       sc_fp <= {8'b0, ~sc_dtq3[15], sc_dtq3[14:8]};
                       sc_sub <= 1;
                  end
                  1: begin
                       sc_dtq <= dlut[s_li*H*256 + s_hi[2:0]*256 + sc_fp[7:0]];
                       s_aq   <= alut[s_li*H*256 + s_hi[2:0]*256 + sc_fp[7:0]];
                       sc_sub <= 2;
                  end
                  2: begin
                       sc_eh <= (21 - bitlen15(sc_dtq[14:0]) > 20) ? 5'd20 :
                                (bitlen15(sc_dtq[14:0]) > 21) ? 5'd0 :
                                5'd21 - bitlen15(sc_dtq[14:0]);
                       sc_sub <= 3;
                  end
                  3: begin
                       sc_t1 <= rshr($signed(xn_rd(s_st_s*CONVD + s_hi[2:0]*64 + sc_i[5:0])),
                                  $signed(8'sd11) - $signed({4'b0, s_bX}));
                       sc_sub <= 4;
                  end
                  4: begin
                       sc_acc <= (sc_t1 > 48'sd127) ? 32'sd127 :
                                 (sc_t1 < -48'sd128) ? -32'sd128 : sc_t1[31:0];
                       sc_sub <= 5;
                  end
                  5: begin
                       s_wrdtx <= 1; s_wrdtx_a <= sc_i[5:0];
                       s_dtx_d <= sat16f(rshr(sc_acc * $signed({1'b0, sc_dtq}),
                                          8'sd14 - $signed({3'b0, sc_eh})));
                       if (sc_i == 63) begin sc_i <= 0; sc_sub <= 6; end
                       else begin sc_i <= sc_i + 1; sc_sub <= 3; end
                  end
                  6: begin
                       s_pbase <= s_st_s*LR*H + s_li*H + s_hi[2:0];
                       s_shi <= 6'sd16 + 6'sd13 - $signed({2'b0, s_bX})
                                - $signed({1'b0, sc_eh}) - $signed({2'b0, s_bB});
                       s_shy <= $signed({2'b0, s_bC});
                       s_start <= 1; sst <= SC_HWAIT; sc_sub <= 0;
                  end
                endcase
              end
              SC_HWAIT: if (s_done) begin sst <= SC_RD; sc_i <= 0; sc_sub <= 0; end
              SC_RD: begin                       // P=4 wide: y-readback + D-skip
                case (sc_sub)
                  0: begin s_rdaw <= sc_i[5:0];
                       // per-head D-skip const (Q value; same for all 64 channels):
                       // even head -> low 16b, odd -> high 16b of the packed word.
                       sc_dsk <= s_hi[0]
                           ? consts[s_li*16 + 5 + {2'b0, s_hi[2:1]}][31:16]
                           : consts[s_li*16 + 5 + {2'b0, s_hi[2:1]}][15:0];
                       sc_sub <= 1;
                  end
                  1: begin                         // latch the 4 scan-y lanes
                       sc_yacc[0] <= $signed(s_yw[15:0]);
                       sc_yacc[1] <= $signed(s_yw[31:16]);
                       sc_yacc[2] <= $signed(s_yw[47:32]);
                       sc_yacc[3] <= $signed(s_yw[63:48]);
                       sc_sub <= 2;
                  end
                  2: begin                         // 4 ybuf lanes -> one row write
                       // the 4 D-skip xnbuf channels are one aligned row: read once.
                       sc_xrow = xnbuf_w[(s_st_s*CONVD + s_hi[2:0]*64 + sc_i[5:0]) >> 2];
                       ybuf_w[(s_st_s*DIN + s_hi[2:0]*64 + sc_i[5:0]) >> 2] <= {
                         sat16f(rshr({{32{sc_yacc[3][15]}}, sc_yacc[3][15:0]}, 8'sd2)
                          + rshr($signed(sc_dsk) * $signed(sc_xrow[48 +: 16]), 8'sd13)),
                         sat16f(rshr({{32{sc_yacc[2][15]}}, sc_yacc[2][15:0]}, 8'sd2)
                          + rshr($signed(sc_dsk) * $signed(sc_xrow[32 +: 16]), 8'sd13)),
                         sat16f(rshr({{32{sc_yacc[1][15]}}, sc_yacc[1][15:0]}, 8'sd2)
                          + rshr($signed(sc_dsk) * $signed(sc_xrow[16 +: 16]), 8'sd13)),
                         sat16f(rshr({{32{sc_yacc[0][15]}}, sc_yacc[0][15:0]}, 8'sd2)
                          + rshr($signed(sc_dsk) * $signed(sc_xrow[0 +: 16]), 8'sd13))};
                       sc_sub <= 0;
                       if (sc_i >= 60) begin       // 64 channels, P=4 -> last at 60
                           sc_i <= 0;
                           if (s_hi == H-1) begin
                               sst <= SC_IDLE; op_pc[s_st_s] <= op_pc[s_st_s] + 1;
                               busy[s_st_s] <= 0;
                           end else begin s_hi <= s_hi + 1; sst <= SC_H; sc_sub <= 0; end
                       end else sc_i <= sc_i + 4;
                  end
                endcase
              end
              default: ;
            endcase

            // ---- token-boundary: a stream whose op_pc reached NPC finishes a
            //      token; carry state and advance to the next token or retire ---
            for (w = 0; w < NC; w = w + 1) begin
                if (active[w] && !busy[w] && op_pc[w] == NPC) begin
                    if (tokcnt[w] + 1 < TMAX && tokcnt[w] + 1 < T_TOKENS) begin
                        tokcnt[w] <= tokcnt[w] + 1; op_pc[w] <= 0;
                    end else begin
                        active[w] <= 0;
                    end
                end
            end
        end
    end

    // ---------------------------------------------------- dump read mux -------
    // DBG=1 (all sim gates): the per-(stream,token) readback the run_mamba_pipe
    // harness compares against MambaSeqRef — the bit-honest path.
    // DBG=0 (bitstream builds): the readback is tied off, so dump_x/dump_logit
    // become write-only and the whole readback-mux tree + dump arrays are pruned
    // (the record protocol reads only tok/cyc, never dbg_data); compute is
    // identical either way.
    // sel 2 = compact argmax token — ALWAYS present (board record protocol +
    // datapath anchor). sel 0/1 = wide x/logit dumps — DBG=1 only.
    generate if (DBG) begin : g_dbg
        always @(posedge clk) begin
            case (dbg_sel)
                4'd0: dbg_data <= dump_x[dbg_addr];
                4'd1: dbg_data <= {{16{dump_logit[dbg_addr][15]}}, dump_logit[dbg_addr]};
                4'd2: dbg_data <= {22'b0, dump_tok[dbg_addr]};
                4'd8: dbg_data <= consts[dbg_addr[6:0]];
                4'd9: dbg_data <= g_wdbg;
                4'd10: dbg_data <= {12'b0, n_seed_dbg};
                4'd11: dbg_data <= {16'b0, esc[dbg_addr[9:0]]};
                default: dbg_data <= 32'sd0;
            endcase
        end
    end else begin : g_nodbg
        always @(posedge clk) begin
            case (dbg_sel)
                4'd2: dbg_data <= {22'b0, dump_tok[dbg_addr]};
                4'd8: dbg_data <= consts[dbg_addr[6:0]];
                4'd9: dbg_data <= g_wdbg;
                4'd10: dbg_data <= {12'b0, n_seed_dbg};
                4'd11: dbg_data <= {16'b0, esc[dbg_addr[9:0]]};
                default: dbg_data <= 32'sd0;
            endcase
        end
    end endgenerate

endmodule

`default_nettype wire
