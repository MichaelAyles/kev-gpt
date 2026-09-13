// -----------------------------------------------------------------------------
// gemv_banked_resident_vec — gemv_banked_resident with a P-WIDE boundary.
//
// The resident MAC core is untouched (one wide URAM word = LANES nibbles/cycle).
// What changes is the BOUNDARY: act feed accepts P INT8 lanes per write, and
// readback returns P INT32 outputs per address. With the boundary the sequencer's
// G_AQ (act-quant) feeds in ceil(K/P) cycles and G_RB drains in ceil(M/P) — the
// boundary phases were ~16k of the 50,324 cyc/token at P=8.
//
// Activations are banked wide-word, like every sequencer scratch:
//     xmem row r holds P consecutive INT8 acts (lane l = bits [l*8 +: 8])
//
// Readback: group word YBITS = LANES*32 holds y[g*LANES .. (g+1)*LANES-1];
// rd_addr is a P-group index — y_out = lanes [rd_addr*P .. +P-1]. P must divide
// LANES, P-blocks never straddle a group word.
//
// iverilog: variable +: part-selects only on plain-reg copies, same as base.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv_banked_resident_vec #(
    parameter integer LANES  = 128,       // PE lanes = nibbles per wide word (pow2)
    parameter integer P      = 8,         // boundary width (P divides LANES, KMAX, MMAX)
    parameter integer MMAX   = 1024,      // max output rows of any single layer
    parameter integer KMAX   = 1024,      // max reduction length of any single layer
    // width of gdone (below): must hold 0..GROUPS inclusive (GROUPS=ceil(MMAX/
    // LANES)) -- see gdone's own comment. Derived, not hardcoded, so it scales
    // automatically with whatever MMAX/LANES a caller instantiates.
    parameter integer GDONE_W = $clog2((MMAX + LANES - 1) / LANES + 1),
    parameter integer WWORDS = 25600,     // resident capacity in wide words
    parameter integer RLAT   = 2,         // read->mac pipeline depth (cycles)
    parameter integer K2     = 0,         // doc-7 R3: 2 K-steps/cycle via the URAM's
                                          // SECOND read port (free at N=1; the TDP claim
                                          // is silicon-proven by split-brain). Integer
                                          // associativity keeps the accumulate BIT-EXACT
                                          // (lane sums peak ~2^20, no mid-sum saturation).
    // Passed straight through to weight_bank_tdp -- see that module for why
    // "block" (Genesys2, no URAM on that part) is not a capability downgrade.
    parameter               MEM_PRIMITIVE = "ultra"
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire [$clog2(MMAX+1)-1:0]     m_count,
    input  wire [$clog2(KMAX+1)-1:0]     k_count,
    input  wire [$clog2(WWORDS)-1:0]     w_base,
    // one-time load: 32-bit chunks assembled into LANES*4-bit wide words
    input  wire                          ld_rst,
    input  wire                          w_we,
    input  wire [31:0]                   w_data,
    // per-call activation: P INT8 lanes per write
    input  wire                          x_we,
    input  wire [P*8-1:0]                x_data,
    // run
    input  wire                          start,
    output reg                           done,
    // committed-group count (groupwise RB overlap): increments the cycle group
    // g's ymem word commits, so the host may drain rows of groups < gdone while
    // the MAC computes the next group. Reset on start. Width is GDONE_W (below,
    // derived from GROUPS=ceil(MMAX/LANES)) -- a hardcoded 4 bits (max 15) here
    // silently wrapped and deadlocked the consumer's row-issue gate (sequencer_
    // vec.sv's G_RB, "ci>>GRPSH < gv_gdone") for any single GEMV call needing
    // >=16 groups (e.g. D_MLP=1024 at LANES=64 needs exactly 16) -- found via a
    // real hang while gating weight_loader_ddr against a larger candidate
    // checkpoint; never triggered by the KV260 deployment (LANES=128, <=8
    // groups) or Genesys2 Option A's real shape (D_MLP=512, <=8 groups at
    // LANES=64) -- see fabric/genesys2/PORT-NOTES.md.
    output reg  [GDONE_W-1:0]            gdone,
    // readback: P INT32 outputs per address (2-cycle latency)
    input  wire [$clog2(MMAX/P)-1:0]     rd_addr,
    output reg  [P*32-1:0]               y_out,
    // ---- embed read port (log §36 fit-plan 2) --------------------------------
    // The tok/pos embeds live in the SPARE DEPTH of the resident weight URAM,
    // above the GEMV image. When emb_sel=1 (the sequencer only raises it while
    // the GEMV FSM is IDLE — embed reads are phase-disjoint from GEMV reads)
    // port-B reads emb_addr instead of grp_base+kc. The DP=1 column-parity bank
    // returns the (even,odd) word PAIR of emb_addr's column one cycle later on
    // emb_pair = {word@(addr|1), word@(addr&~1)} — 2*WBITS bits = 8 embed rows
    // at LANES=256. The sequencer addresses embeds from even pair bases.
    input  wire                          emb_sel,
    input  wire [$clog2(WWORDS)-1:0]     emb_addr,
    output wire [LANES*8-1:0]            emb_pair,
    // ---- weight-bank diagnostic readback (fixation-word investigation, item 6:
    // "snapshot the suspect rows directly" -- see FIXATION-WORD-POSTMORTEM.md).
    // weight_bank_tdp's port A is otherwise COMPLETELY UNUSED for reading in
    // this instantiation (tied to raddr_a=0, rword_a left unconnected below) --
    // nothing in the compute path (MAC accumulation reads port B only, via
    // waddr/wword_rd) ever touches port A's read side. That makes it a free,
    // zero-risk tap: driving it from a CPU-controlled address and observing
    // the result can never perturb inference, at any time, running or idle.
    //
    // K2=1 (this design's own deployed config) puts weight_bank_tdp in DP=1
    // column-parity-split mode: raddr_a's LSB is IGNORED by the memory --
    // rword_a always returns the EVEN-indexed bank, rword1_a always the ODD
    // one, at the pair index raddr_a[WAW-1:1], regardless of raddr_a[0].
    // Exposing only rword_a (as an earlier version of this port did) silently
    // returns the wrong row's data for every odd address -- caught by item
    // 6's own simulation gate before real hardware (see
    // FIXATION-WORD-POSTMORTEM.md item 6's verification note). Fixed the
    // same way emb_pair (above) already solves this exact problem: expose
    // BOTH halves, let the consumer pick by address parity.
    input  wire [$clog2(WWORDS)-1:0]     wbdiag_addr,
    output wire [LANES*8-1:0]            wbdiag_pair   // {odd(rword1_a), even(rword_a)}
);
    localparam integer WBITS  = LANES*4;
    localparam integer YBITS  = LANES*32;
    localparam integer LSH    = $clog2(LANES);
    localparam integer LSHP   = $clog2(P);
    localparam integer GROUPS = (MMAX + LANES - 1) / LANES;
    localparam integer WAW    = $clog2(WWORDS);
    localparam integer XROWS  = KMAX / P;
    localparam integer XAW    = $clog2(XROWS);
    localparam integer SUBW   = WBITS / 32;
    localparam integer SSW    = (SUBW > 1) ? $clog2(SUBW) : 1;

    // Weight memory in URAM banks (generate: one real array per bank).
    // GEOMETRY IS THE CONSTRAINT: a URAM is 4096 x 72b, and Vivado pads each
    // memory up to a multiple of 72b x 4096. One 1024b x 12800 memory -> 16 URAM
    // wide x 4 cascade = 64 = the whole device -> "infeasible" -> ~400k LUTRAM.
    // Even two 512b banks pad to 2 x (8 wide x 4) = 64. The dense packing is
    // 72b-wide banks: 15 banks x 4 cascade = 60 URAM for LANES=256.
    // For LANES<=128 the proven single 512b x 25600 memory (56 URAM) is kept.
    localparam integer BANKW = (WBITS > 512) ? 72 : WBITS;        // URAM-native width
    localparam integer NB    = (WBITS + BANKW - 1) / BANKW;       // banks
    localparam integer WPAD  = NB * BANKW;                        // padded word width

    reg [P*8-1:0]    xmem [0:XROWS-1];   // P acts per row
    reg [YBITS-1:0]  ymem [0:GROUPS-1];

    // ---- per-call activation pointer (the weight assembler now lives in the bank)
    reg [XAW-1:0]    xptr;
    always @(posedge clk) begin
        if (ld_rst) xptr <= 0;
        else if (x_we) begin xmem[xptr] <= x_data; xptr <= xptr + 1'b1; end
    end

    // ---- resident weights: the silicon-proven TDP URAM bank (weight_bank_tdp).
    // HDL inference of TDP UltraRAM is DEAD in 2025.2 (three OOC takes confirmed
    // it again: two-address ports, then "invalid write mode" for every template
    // variant). The bank's SYNTHESIS branch is xpm_memory_tdpram("ultra"), the
    // sim branch behavioral — gates verify sim, the board verifies XPM (the
    // codebase's dual-dialect pattern). K2 rides the bank's DP=1 COLUMN-PARITY
    // mode: ONE port-B read at the even pair address returns BOTH words kc
    // (rword_b) and kc+1 (rword1_b); grp_base and k_count are always even here.
    wire [$clog2(WWORDS)-1:0] waddr;
    wire [WBITS-1:0] wword_rd, wword2_rd;
    weight_bank_tdp #(.LANES(LANES), .WWORDS(WWORDS), .DP((K2 != 0) ? 1 : 0),
                       .MEM_PRIMITIVE(MEM_PRIMITIVE)) u_wb (
        .clk(clk), .clk2x(clk),
        .ld_rst(ld_rst), .w_we(w_we), .w_data(w_data),
        .raddr_b(waddr), .rword_b(wword_rd), .rword1_b(wword2_rd),
        .raddr_a(wbdiag_addr), .rword_a(wbdiag_pair[WBITS-1:0]), .rword1_a(wbdiag_pair[2*WBITS-1:WBITS]));

    // ---- run FSM + RLAT-deep read/mac pipeline -------------------------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0]            state;
    reg [$clog2(GROUPS):0] g;
    reg [$clog2(KMAX):0] kc, kmac;
    reg [WAW-1:0]        grp_base;
    reg [YBITS-1:0]      accb;

    wire [$clog2(GROUPS):0] gcount = (m_count + LANES - 1) >> LSH;
    wire                 issue   = (kc < k_count);
    wire                 issue2  = (K2 != 0) && (kc + 1 < k_count);
    // embed port steals the (idle) port-B address; otherwise the GEMV K-walk
    assign               waddr   = emb_sel ? emb_addr : (grp_base + kc);
    assign               emb_pair = {wword2_rd, wword_rd};
    wire [$clog2(KMAX):0] kc1    = kc + 1;

    // pipeline: weight word (stage 0 = the per-bank URAM read reg) + act + valid
    reg [WBITS-1:0]      word_p [0:RLAT-2];
    reg [P*8-1:0]        xrow_p [0:RLAT-1];
    reg [LSHP-1:0]       xl_p   [0:RLAT-1];
    reg                  v_p    [0:RLAT-1];
    // K2 second lane (kc+1's word/act/valid ride the same depths)
    reg [WBITS-1:0]      word2_p [0:RLAT-2];
    reg [P*8-1:0]        xrow2_p [0:RLAT-1];
    reg [LSHP-1:0]       xl2_p   [0:RLAT-1];
    reg                  v2_p    [0:RLAT-1];
    integer i, L;
    reg signed [31:0]    prodL, sumL;
    reg signed [31:0]    prod2L;
    reg [WBITS-1:0]      wsel, wsel2;
    reg [P*8-1:0]        xrow, xrow2;
    reg signed [7:0]     xsel, xsel2;

    wire                 mac_v  = v_p[RLAT-1];
    wire                 mac_v2 = v2_p[RLAT-1];

    // ---- addend stage (timing): the MAC front-end (act lane-mux, INT4xINT8
    // nibble products, K2 mux) was 13 logic levels feeding the accb carry chain
    // — the @5ns worst path. Register the per-lane addend (prodL + prod2L) one
    // cycle ahead so the accumulate cycle is ONLY accb <= accb + sext(addend_r).
    // Bit-exact: same addends, same order, one cycle later in absolute time —
    // kmac now counts in the ADD stage, so the end-of-group sample of accb into
    // ymem (kmac == k_count) shifts with it automatically (+1 cyc per group).
    // |w*x| <= 1024 each, so prodL + prod2L is in [-2032, 2048] — ADW=14 holds
    // it exactly (two's-complement truncate then sign-extend is lossless).
    localparam integer ADW = 14;
    reg [LANES*ADW-1:0]  addend_r;
    reg                  add_v, add_v2;

    always @(posedge clk) begin
        word_p[0] <= wword_rd;
        xrow_p[0] <= xmem[kc[$clog2(KMAX)-1:0] >> LSHP];
        xl_p[0]   <= kc[LSHP-1:0];
        v_p[0]    <= (state == RUN) && issue;
        word2_p[0] <= wword2_rd;
        xrow2_p[0] <= xmem[kc1[$clog2(KMAX)-1:0] >> LSHP];
        xl2_p[0]   <= kc1[LSHP-1:0];
        v2_p[0]    <= (state == RUN) && issue2;
        for (i = 1; i < RLAT-1; i = i + 1) begin
            word_p[i]  <= word_p[i-1];
            word2_p[i] <= word2_p[i-1];
        end
        for (i = 1; i < RLAT; i = i + 1) begin
            xrow_p[i] <= xrow_p[i-1];
            xl_p[i]   <= xl_p[i-1];
            v_p[i]    <= v_p[i-1];
            xrow2_p[i] <= xrow2_p[i-1];
            xl2_p[i]   <= xl2_p[i-1];
            v2_p[i]    <= v2_p[i-1];
        end

        // addend stage: the heavy combinational front-end, registered. Runs
        // unconditionally on mac_v (mac_v only asserts in RUN); add_v/add_v2
        // are the delayed valid/K2 tags consumed by the accumulate below.
        add_v  <= mac_v;
        add_v2 <= mac_v && mac_v2;
        if (mac_v) begin
            wsel  = word_p[RLAT-2];
            xrow  = xrow_p[RLAT-1];
            xsel  = xrow[xl_p[RLAT-1]*8 +: 8];
            wsel2 = word2_p[RLAT-2];
            xrow2 = xrow2_p[RLAT-1];
            xsel2 = xrow2[xl2_p[RLAT-1]*8 +: 8];
            for (L = 0; L < LANES; L = L + 1) begin
                prodL  = $signed(wsel[L*4 +: 4]) * xsel;
                prod2L = mac_v2 ? $signed(wsel2[L*4 +: 4]) * xsel2 : 32'sd0;
                addend_r[L*ADW +: ADW] <= prodL + prod2L;  // fits ADW, lossless
            end
        end

        if (rst) begin
            state <= IDLE; done <= 1'b0; gdone <= {GDONE_W{1'b0}};
            g <= 0; kc <= 0; kmac <= 0; accb <= {YBITS{1'b0}}; grp_base <= 0;
            for (i = 0; i < RLAT; i = i + 1) begin v_p[i] <= 1'b0; v2_p[i] <= 1'b0; end
            add_v <= 1'b0; add_v2 <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        g <= 0; kc <= 0; kmac <= 0; accb <= {YBITS{1'b0}};
                        gdone <= {GDONE_W{1'b0}};
                        grp_base <= w_base;
                        for (i = 0; i < RLAT; i = i + 1) begin
                            v_p[i] <= 1'b0; v2_p[i] <= 1'b0;
                        end
                        add_v <= 1'b0; add_v2 <= 1'b0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    if (issue) kc <= kc + ((K2 != 0 && issue2) ? 2'd2 : 2'd1);
                    // accumulate stage: ONLY the add — the sole logic between
                    // accb and accb's D input is the sign-extended carry chain.
                    if (add_v) begin
                        for (L = 0; L < LANES; L = L + 1) begin
                            sumL = $signed(accb[L*32 +: 32])
                                 + $signed(addend_r[L*ADW +: ADW]);
                            accb[L*32 +: 32] <= sumL;
                        end
                        kmac <= kmac + (add_v2 ? 2'd2 : 2'd1);
                    end
                    if (kmac == k_count) begin
                        ymem[g[$clog2(GROUPS)-1:0]] <= accb;
                        gdone <= gdone + 1'b1;
                        if (g == gcount - 1) state <= FIN;
                        else begin
                            g <= g + 1'b1; kc <= 0; kmac <= 0;
                            accb <= {YBITS{1'b0}};
                            grp_base <= grp_base + k_count;
                            for (i = 0; i < RLAT; i = i + 1) begin
                                v_p[i] <= 1'b0; v2_p[i] <= 1'b0;
                            end
                            add_v <= 1'b0; add_v2 <= 1'b0;
                        end
                    end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    // ---- readback: P consecutive outputs per address (2-cycle latency) --------
    localparam integer PPG = LANES / P;            // P-groups per ymem word
    reg [YBITS-1:0]        rd_word;
    reg [$clog2(PPG)-1:0]  rd_off;
    always @(posedge clk) begin
        rd_word <= ymem[rd_addr / PPG];
        rd_off  <= rd_addr % PPG;
        y_out   <= rd_word[rd_off*(P*32) +: P*32];
    end
endmodule
