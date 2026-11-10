// tb_gemv_stream — drives gemv_stream from files written by run_gemv_stream.py.
//
// One simulation = a list of RUNS (matrices) that follow each other on the same
// NPORTS weight streams, exactly as they follow each other in the packed image.
// Per run: load the activation (G words), pulse start, wait for !busy.
//
// Files in +dir=<d>:
//   port<k>.mem   port k's beats in the order its read master delivers them (128-bit hex)
//   runs.mem      3 x 32-bit hex per run: G, R, pad tuples
//   act_m.mem     activation mantissa words, 1280-bit hex, all runs concatenated
//   act_e.mem     activation exponents (EXW-bit two's complement hex), same order
//   out.hex       (written) one fp32 per row, "<8 hex> <last>"
// Plusargs: +nbeats= (per port) +nruns= +gap= (% of cycles a port withholds valid)
//           +bp= (% of cycles m_ready is low) +seed=
// Each port's valid is randomised independently; data is X while valid is low,
// and valid/data are held until ready (AXI rule). Prints
//   GEMV_TB_RUN run=<i> cycles=<start..!busy> rows=<R> groups=<G>
//   GEMV_TB_DONE outputs=<n> xout=<n words with X> pops=<tuples the engine popped>
`timescale 1ns / 1ps

module tb_gemv_stream;
    parameter integer NPORTS = 3;
    parameter integer GMAX   = 96;
    parameter integer EXW    = 9;
    parameter [6:0]   TREE_REGMASK = 7'b1111111;
    parameter integer MAXB   = 1 << 16;     // beats per port
    parameter integer MAXRUN = 512;
    parameter integer MAXACT = 1 << 15;

    reg clk = 0, rst = 1;
    always #1.5 clk = ~clk;

    reg [127:0]   pmem [0:NPORTS*MAXB-1];
    reg [31:0]    runs [0:3*MAXRUN-1];
    reg [1279:0]  actm [0:MAXACT-1];
    reg [EXW-1:0] acte [0:MAXACT-1];

    integer nbeats, nruns, gap, bp, seed, fout, k, r, g, abase, timeout, nout, xout;
    reg [1023:0] dir;

    reg                   start = 0;
    reg [7:0]             cfg_groups = 0;
    reg [31:0]            cfg_rows = 0, cfg_pad = 0;
    wire                  busy;
    reg                   act_we = 0;
    reg [7:0]             act_group = 0;
    reg [1279:0]          act_mant = 0;
    reg [EXW-1:0]         act_exp = 0;
    wire [NPORTS-1:0]     s_valid, s_ready;
    wire [NPORTS*128-1:0] s_data;
    wire                  m_valid, m_last;
    reg                   m_ready = 0;
    wire [31:0]           m_data;

    gemv_stream #(.NPORTS(NPORTS), .GMAX(GMAX), .EXW(EXW), .TREE_REGMASK(TREE_REGMASK)) dut (
        .clk(clk), .rst(rst), .start(start), .cfg_groups(cfg_groups), .cfg_rows(cfg_rows),
        .cfg_pad(cfg_pad), .busy(busy), .act_we(act_we), .act_group(act_group),
        .act_mant(act_mant), .act_exp(act_exp), .s_valid(s_valid), .s_ready(s_ready),
        .s_data(s_data), .m_valid(m_valid), .m_ready(m_ready), .m_data(m_data), .m_last(m_last));

    // ---- stream sources: one per port, independent random valid gaps
    genvar gk;
    generate for (gk = 0; gk < NPORTS; gk = gk + 1) begin : src
        reg         v = 0;
        integer     ptr = 0;
        integer     sd;
        reg         go = 0;
        integer     nxt;
        assign s_valid[gk] = v;
        assign s_data[gk*128 +: 128] = v ? pmem[gk*MAXB + ptr] : {128{1'bx}};
        always @(posedge clk) begin
            if (rst) begin
                v <= 1'b0; ptr <= 0;
            end else if (go) begin
                if (!v || s_ready[gk]) begin
                    nxt = (v && s_ready[gk]) ? ptr + 1 : ptr;
                    ptr <= nxt;
                    v   <= (nxt < nbeats) && (({$random(sd)} % 100) >= gap);
                end
            end
        end
    end endgenerate

    // ---- result sink with random back-pressure
    integer sdo;
    always @(posedge clk) begin
        if (rst) m_ready <= 1'b0;
        else     m_ready <= (({$random(sdo)} % 100) >= bp);
        if (!rst && m_valid && m_ready) begin
            $fdisplay(fout, "%08x %0d", m_data, m_last);
            nout = nout + 1;
            if (^m_data === 1'bx) xout = xout + 1;
        end
    end

    // ---- tuples the engine actually popped (the skid buffers may hold beats beyond that)
    integer pops = 0;
    always @(posedge clk) if (!rst && dut.pop) pops <= pops + 1;

    // ---- watchdog
    integer cyc = 0;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc > timeout) begin
            $display("GEMV_TB_TIMEOUT cyc=%0d", cyc);
            $finish;
        end
    end

    integer t0;
    initial begin
        if (!$value$plusargs("dir=%s", dir))       begin $display("need +dir"); $finish; end
        if (!$value$plusargs("nbeats=%d", nbeats)) nbeats = 0;
        if (!$value$plusargs("nruns=%d", nruns))   nruns = 1;
        if (!$value$plusargs("gap=%d", gap))       gap = 0;
        if (!$value$plusargs("bp=%d", bp))         bp = 0;
        if (!$value$plusargs("seed=%d", seed))     seed = 1;
        if (!$value$plusargs("timeout=%d", timeout)) timeout = 20000000;
        nout = 0; xout = 0; sdo = seed + 99991;
        if (nbeats > MAXB)   begin $display("GEMV_TB_ERR nbeats > MAXB"); $finish; end
        if (nruns > MAXRUN)  begin $display("GEMV_TB_ERR nruns > MAXRUN"); $finish; end
        for (k = 0; k < NPORTS; k = k + 1)
            $readmemh($sformatf("%0s/port%0d.mem", dir, k), pmem, k*MAXB, k*MAXB + nbeats - 1);
        $readmemh($sformatf("%0s/runs.mem", dir), runs, 0, 3*nruns - 1);
        $readmemh($sformatf("%0s/act_m.mem", dir), actm);
        $readmemh($sformatf("%0s/act_e.mem", dir), acte);
        fout = $fopen($sformatf("%0s/out.hex", dir), "w");
        repeat (8) @(posedge clk);
        rst <= 1'b0;
        repeat (4) @(posedge clk);
        abase = 0;
        for (r = 0; r < nruns; r = r + 1) begin
            // activation load
            for (g = 0; g < runs[3*r]; g = g + 1) begin
                act_we <= 1'b1; act_group <= g[7:0];
                act_mant <= actm[abase + g]; act_exp <= acte[abase + g];
                @(posedge clk);
            end
            act_we <= 1'b0; act_mant <= {1280{1'bx}}; act_exp <= {EXW{1'bx}};
            abase = abase + runs[3*r];
            repeat (3) @(posedge clk);
            cfg_groups <= runs[3*r][7:0]; cfg_rows <= runs[3*r + 1]; cfg_pad <= runs[3*r + 2];
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0; t0 = cyc;
            @(posedge clk);
            while (busy) @(posedge clk);
            $display("GEMV_TB_RUN run=%0d cycles=%0d rows=%0d groups=%0d", r, cyc - t0,
                     runs[3*r + 1], runs[3*r]);
            repeat (({$random(seed)} % 4)) @(posedge clk);
        end
        repeat (60) @(posedge clk);         // anything still dribbling out is an error
        $display("GEMV_TB_DONE outputs=%0d xout=%0d pops=%0d", nout, xout, pops);
        $fclose(fout);
        $finish;
    end

    // sources run from reset release, each with its own seed
    generate for (gk = 0; gk < NPORTS; gk = gk + 1) begin : srcctl
        initial begin
            src[gk].sd = 0;
            @(negedge rst);
            src[gk].sd = seed + 7919 * (gk + 1);
            src[gk].go = 1'b1;
        end
    end endgenerate

endmodule
