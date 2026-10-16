// -----------------------------------------------------------------------------
// tb_rope — bit-exact gate for rope_apply_vec.sv against pack_rope.py's own
// integer reference (same Q.16 x Q1.15 -> rsh_round(.., 15) pipeline, not a
// float-tolerance check -- see that script's own header for why).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_rope;
    localparam integer HEAD_DIM  = 36;
    localparam integer ROT_DIM   = 32;
    localparam integer ROT_PAIRS = 16;
    localparam integer TMAX      = 128;
    localparam integer N_CASES   = 8;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg                          start;
    reg  [$clog2(TMAX)-1:0]      position;
    reg  [HEAD_DIM*32-1:0]       head_in;
    wire                         done;
    wire [HEAD_DIM*32-1:0]       head_out;

    rope_apply_vec #(.HEAD_DIM(HEAD_DIM), .ROT_DIM(ROT_DIM), .ROT_PAIRS(ROT_PAIRS), .TMAX(TMAX),
                      .ROM_FILE_COS("rope_cos.mem"), .ROM_FILE_SIN("rope_sin.mem")) dut (
        .clk(clk), .start(start), .position(position), .head_in(head_in),
        .done(done), .head_out(head_out)
    );

    reg [31:0] head_in_flat [0:N_CASES*HEAD_DIM-1];
    reg [31:0] positions_flat [0:N_CASES-1];
    integer f, ci, i;

    initial begin
        $readmemh("head_in.mem", head_in_flat);
        $readmemh("positions.mem", positions_flat);

        start = 1'b0; position = 0; head_in = {(HEAD_DIM*32){1'b0}};
        @(posedge clk); #1;

        f = $fopen("got_out.mem", "w");
        for (ci = 0; ci < N_CASES; ci = ci + 1) begin
            for (i = 0; i < HEAD_DIM; i = i + 1)
                head_in[i*32 +: 32] = head_in_flat[ci*HEAD_DIM + i];
            position = positions_flat[ci][$clog2(TMAX)-1:0];
            start = 1'b1;
            @(posedge clk); #1;
            start = 1'b0;
            // rope_apply_vec is 1-cycle latency (registered output on the
            // same edge `start` was sampled) -- done is already high here.
            if (!done) begin
                $display("TB_ROPE_FAIL,case=%0d,done_not_asserted", ci);
                $finish;
            end
            for (i = 0; i < HEAD_DIM; i = i + 1)
                $fwrite(f, "%08x\n", head_out[i*32 +: 32]);
            @(posedge clk); #1;
        end
        $fclose(f);

        $display("TB_DONE,n_cases=%0d", N_CASES);
        $finish;
    end

    initial begin
        #1_000_000;
        $display("TB_TIMEOUT");
        $finish;
    end
endmodule
