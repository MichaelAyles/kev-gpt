// gemv_i4i8.sv — WIDE (P-lane) INT4xINT8 GEMV for the mamba engine (doc 9 §7).
//
// Reduce-parallel wide GEMV: reads PE 32-bit weight words per cycle (PE*8
// nibble INT4xINT8 MACs) and folds them into ONE row accumulator, so a row of
// `wpr` words completes in wpr/PE cycles instead of wpr. out[r] = sum_c
// w[r][c]*x[c], INT32 accumulator, raw output (dequant happens downstream in
// the sequencer's scale unit). Bit-exact against the single-lane predecessor
// and mamba2_fixed.gemv_i4i8's acc: integer addition is associative, so
// summing PE*8 products/cycle yields the identical row sum as 8/cycle.
//
// Weight image (UNCHANGED external contract): row-major, D_IN/8 words per row
// (LSB-first nibbles, two's-complement INT4), word-addressed (base/wr_w_addr).
// Internally the image is banked PE words wide (WBITS = PE*32) in URAM so PE
// words read in one cycle. REQUIREMENTS on the runtime geometry:
//   * PE is a power of two,
//   * wpr is a multiple of PE (mamba: 32,64 both divisible by 8 and 16),
//   * base is a multiple of PE (mamba bases 0, li*37120, 259840+.., 374528 are),
//   * WMEM is a multiple of PE.
// ROWS x D_IN up to 1160 x 512; PE*8 parallel nibble-MACs per cycle.

`default_nettype none

module gemv_i4i8 #(
    parameter int PE    = 16,               // weight WORDS read per cycle (pow2)
    parameter int ROWS  = 1160,             // MAX geometry (buffer sizing)
    parameter int D_IN  = 512,
    parameter int WPR   = D_IN / 8,
    parameter int WMEM  = 262144,           // weight words (URAM budget)
    parameter int RDP   = 4                 // accumulator read lanes/cycle
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    output reg                      done,

    // runtime geometry: this call reads `rows` rows of `wpr` words starting
    // at weight word `base` (one banked image serves every projection)
    input  wire [$clog2(WMEM)-1:0]  base,
    input  wire [$clog2(ROWS+1)-1:0] rows,
    input  wire [$clog2(WPR+1)-1:0]  wpr,

    input  wire                     wr_w,
    input  wire [$clog2(WMEM)-1:0]  wr_w_addr,
    input  wire [31:0]              wr_w_data,

    input  wire                     wr_x,
    input  wire [$clog2(D_IN)-1:0]  wr_x_addr,
    input  wire signed [7:0]        wr_x_data,

    input  wire [$clog2(ROWS)-1:0]  rd_acc_addr,
    output wire signed [31:0]       rd_acc_data,

    // WIDE read: RDP consecutive accumulators from `rd_accw_base` in one cycle
    // (the sequencer dequant streams read accs 0..rows-1 in order, so a P-lane
    // dequant consumes RDP/cycle). Combinational, like the single-lane port.
    input  wire [$clog2(ROWS)-1:0]  rd_accw_base,
    output wire [RDP*32-1:0]        rd_accw_data,

    // DEBUG weight readback — verifies on SILICON that the AXI-loaded weight
    // image actually landed, which no simulation can check. It SHARES the
    // existing read port (steering wptr while the core is idle) instead of
    // adding a second one: a second read port makes wrom 3-ported, which
    // cannot map to URAM, and the design falls back to 714 BRAMs and stops
    // fitting. Read wq's sub-word — wq is a plain reg, so the variable
    // part-select is safe.
    input  wire                     dbg_rd_en,
    input  wire [$clog2(WMEM)-1:0]  dbg_rd_addr,
    output wire [31:0]              dbg_rd_data
);
    localparam int LP    = $clog2(PE);          // log2 words/cycle
    localparam int WBITS = PE * 32;             // wide weight word width
    localparam int WWW   = WMEM / PE;           // wide words in the image
    localparam int AWM   = $clog2(WMEM);        // word-address width (external)
    localparam int AWW   = $clog2(WWW);         // wide-word-address width
    localparam int RW    = $clog2(ROWS);
    localparam int WSW   = $clog2(WPR/PE + 1);  // wide-step counter width

    // -------- weight image: PE words per wide URAM word, SDP (write init +
    // 1-cycle read in one always block). Assembly places incoming word
    // wr_w_addr[LP-1:0] into wbuf[slot*32 +: 32] (variable part-select WRITE on
    // a PLAIN reg is iverilog-safe); the PE-th word commits the wide row with a
    // CONSTANT concat (the PE-th word always lands in the top slot under the
    // contiguous ascending load the sequencer/tb performs).
    (* ram_style = "ultra" *)
    reg [WBITS-1:0] wrom [0:WWW-1];
    reg [31:0]      wbufw [0:PE-2];              // wide-word assembly slots
    reg [WBITS-1:0] wq;                          // registered wide read
    reg [LP-1:0]    dbg_sub;                     // debug sub-word select

    wire [LP-1:0]     wsub  = wr_w_addr[LP-1:0];
    wire [AWW-1:0]    wwide = wr_w_addr[AWM-1:LP];
    reg  [AWW-1:0]    wptr;                       // running wide read pointer

    // Assemble the wide word from a per-slot register ARRAY, never a variable
    // part-select into one 512-bit reg. The part-select version simulated
    // correctly but synthesised to a wide word whose low 48 bits were dropped
    // (proven by RTL-vs-netlist equivalence: word 0 lost entirely, word 1 half
    // lost, every later word fine) — weights silently wrong on silicon while
    // every simulation passed. Whole-element array access is the style the
    // rest of this design already uses for exactly this reason.
    wire [WBITS-1:0] wcommit;
    genvar wq_i;
    generate
        for (wq_i = 0; wq_i < PE-1; wq_i = wq_i + 1) begin : g_wcommit
            assign wcommit[wq_i*32 +: 32] = wbufw[wq_i];
        end
    endgenerate
    assign wcommit[(PE-1)*32 +: 32] = wr_w_data;  // the PE-th word arrives now

    always @(posedge clk) begin
        if (wr_w) begin
            if (wsub != PE-1) wbufw[wsub] <= wr_w_data;
            else              wrom[wwide] <= wcommit;
        end
        wq <= wrom[dbg_rd_en ? dbg_rd_addr[AWM-1:LP] : wptr];
        dbg_sub <= dbg_rd_addr[LP-1:0];
    end
    assign dbg_rd_data = wq[dbg_sub*32 +: 32];

    // -------- x: INT8 lanes, 8 per 64-bit word (write path unchanged) ---------
    reg [63:0] xin [0:WPR-1];
    always @(posedge clk) begin
        if (wr_x) xin[wr_x_addr[$clog2(D_IN)-1:3]][(wr_x_addr[2:0])*8 +: 8]
                      <= wr_x_data;
    end

    reg signed [31:0] accs [0:ROWS-1];
    assign rd_acc_data = accs[rd_acc_addr];

    // RDP-wide read: whole-element indexed reads (iverilog-safe, like the
    // single-lane assign above). base+g stays in-bounds for every sequencer
    // call (in_proj 1160, out_proj 256, head 1024 are all RDP-multiples).
    genvar gr;
    generate for (gr = 0; gr < RDP; gr = gr + 1) begin : g_rdw
        assign rd_accw_data[gr*32 +: 32] = accs[rd_accw_base + gr[$clog2(ROWS)-1:0]];
    end endgenerate

    // -------- FSM: iterate rows x wide-steps, PE words per cycle --------------
    localparam [1:0] IDLE = 2'd0, RUN = 2'd1, DRAIN = 2'd2;
    reg [1:0]  st;
    reg [RW-1:0] ri;
    reg [WSW-1:0] wi;                            // wide-step within a row
    wire [WSW-1:0] wprw = wpr >> LP;             // wide steps per row (exact)
    wire issue = (st == RUN);

    // S0 -> S1: registered x wide word (PE 64-bit lanes) + tags
    reg               v1, last1;
    reg [RW-1:0]      r1;
    reg [PE*64-1:0]   x1;

    // S1 -> S2: PE partial sums (each = 8 nibble products of one word)
    reg               v2, last2;
    reg [RW-1:0]      r2;
    reg signed [15:0] ps [0:PE-1];

    // S2 accumulator
    reg signed [31:0] racc;

    // nibble decode: two's complement INT4 -> signed 5
    function automatic signed [4:0] nib(input [3:0] n);
        nib = {n[3], n};
    endfunction

    // sum racc + all PE partials (associative => bit-exact vs 8/cycle)
    function automatic signed [31:0] sum_ps(input signed [31:0] a);
        integer j;
        begin
            sum_ps = a;
            for (j = 0; j < PE; j = j + 1) sum_ps = sum_ps + ps[j];
        end
    endfunction

    // build the PE-wide x word for wide-step `wi` from xin[wi*PE .. +PE-1]
    // (whole-element indexed reads are iverilog-safe; only part-selects of an
    // unpacked-array element read X).
    integer xk;
    reg [PE*64-1:0] xrow;
    always @(*) begin
        xrow = {(PE*64){1'b0}};
        for (xk = 0; xk < PE; xk = xk + 1)
            xrow[xk*64 +: 64] = xin[wi*PE + xk];
    end

    integer p, k;
    reg signed [15:0] acc8;
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            // reset EVERY pipeline register, not just the valid bits: an
            // FPGA powers these up at 0 while RTL sim starts them X, so any
            // dependence on their initial value is a sim/silicon divergence
            // waiting to happen.
            st <= IDLE; v1 <= 0; v2 <= 0;
            ri <= '0; wi <= '0; wptr <= '0; r1 <= '0; r2 <= '0;
            last1 <= 1'b0; last2 <= 1'b0; racc <= 32'sd0;
        end else begin
            case (st)
              IDLE: if (start) begin
                st <= RUN; ri <= '0; wi <= '0; wptr <= base >> LP;
              end
              RUN: begin
                wptr <= wptr + 1'b1;
                if (wi == wprw-1) begin
                    wi <= '0;
                    if (ri == rows-1) st <= DRAIN; else ri <= ri + 1'b1;
                end else wi <= wi + 1'b1;
              end
              DRAIN: if (!v1 && !v2) begin st <= IDLE; done <= 1'b1; end
            endcase

            // S0 -> S1: latch x wide word + tags (weight read = wq, registered
            // in the wrom always block above -> aligned to this stage)
            v1 <= issue; r1 <= ri; last1 <= (wi == wprw-1);
            x1 <= xrow;

            // S1 -> S2: PE partial sums, each 8 nibble INT4xINT8 products of
            // word p summed (wq is a plain reg -> constant part-selects safe)
            v2 <= v1; r2 <= r1; last2 <= last1;
            for (p = 0; p < PE; p = p + 1) begin
                acc8 = 16'sd0;
                for (k = 0; k < 8; k = k + 1)
                    acc8 = acc8 + $signed(nib(wq[p*32 + k*4 +: 4]))
                                * $signed(x1[p*64 + k*8 +: 8]);
                ps[p] <= acc8;
            end

            // S2: fold PE partials into the row accumulator; on row end store
            if (v2) begin
                if (last2) begin
                    accs[r2] <= sum_ps(racc);
                    racc <= 32'sd0;
                end else
                    racc <= sum_ps(racc);
            end else
                racc <= 32'sd0;
        end
    end

endmodule

`default_nettype wire
