`timescale 1ns / 1ps
module tb_crc32_word;
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;
  reg word_valid;
  reg [31:0] word_data;
  wire [31:0] crc_out;
  crc32_word u_dut (.clk(clk), .rst(rst), .word_valid(word_valid), .word_data(word_data), .crc_out(crc_out));
  integer errors = 0;
  initial begin
    word_valid = 0; word_data = 0;
    repeat (2) @(posedge clk);
    rst = 0;
    @(posedge clk);
    // trial 0, back-to-back every cycle, 4 words
    rst <= 1'b1; @(posedge clk); rst <= 1'b0; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h06671ad1; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hbdd640fb; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h46685257; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h3eb13b90; @(posedge clk);
    word_valid <= 1'b0; @(posedge clk); @(posedge clk);
    if (crc_out !== 32'hc5e02710) begin $display("MISMATCH,trial=0,got=%08x,want=c5e02710", crc_out); errors = errors + 1; end
    else $display("trial 0 OK, crc=%08x", crc_out);
    // trial 1, back-to-back every cycle, 8 words
    rst <= 1'b1; @(posedge clk); rst <= 1'b0; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h23b8c1e9; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hbc8960a9; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h1a3d1fa7; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'had3c2d6d; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hbd9c66b3; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'he465e150; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h8b9d2434; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h16419f82; @(posedge clk);
    word_valid <= 1'b0; @(posedge clk); @(posedge clk);
    if (crc_out !== 32'hab5d672b) begin $display("MISMATCH,trial=1,got=%08x,want=ab5d672b", crc_out); errors = errors + 1; end
    else $display("trial 1 OK, crc=%08x", crc_out);
    // trial 2, back-to-back every cycle, 19 words
    rst <= 1'b1; @(posedge clk); rst <= 1'b0; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h6c031199; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h0822e8f3; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h07a0ca6e; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h17fc695a; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h37f8a88b; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h3b8faa18; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h815ef6d1; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h9a1de644; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h06cb0fb3; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h8fadc1a6; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h32e70629; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hb74d0fb1; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'ha65ed389; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hb38a088c; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h8b8148f6; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h6b65a6a4; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h386ecbe0; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h72ff5d2a; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h96da1dac; @(posedge clk);
    word_valid <= 1'b0; @(posedge clk); @(posedge clk);
    if (crc_out !== 32'hb130088c) begin $display("MISMATCH,trial=2,got=%08x,want=b130088c", crc_out); errors = errors + 1; end
    else $display("trial 2 OK, crc=%08x", crc_out);
    // trial 3, back-to-back every cycle, 9 words
    rst <= 1'b1; @(posedge clk); rst <= 1'b0; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hcf36d58b; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hde8a774b; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h01a9e71f; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hc241330b; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hce4a2bbd; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h28df6ec4; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hb2b9437a; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h6c307511; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h571aa876; @(posedge clk);
    word_valid <= 1'b0; @(posedge clk); @(posedge clk);
    if (crc_out !== 32'hf73e2ba9) begin $display("MISMATCH,trial=3,got=%08x,want=f73e2ba9", crc_out); errors = errors + 1; end
    else $display("trial 3 OK, crc=%08x", crc_out);
    // trial 4, back-to-back every cycle, 9 words
    rst <= 1'b1; @(posedge clk); rst <= 1'b0; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h27cd8130; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h371ecd7b; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hf50bea63; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'hc37459ee; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h562b0f79; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h1a2a73ed; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h17be3111; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h6142ea7d; @(posedge clk);
    word_valid <= 1'b1; word_data <= 32'h18c26797; @(posedge clk);
    word_valid <= 1'b0; @(posedge clk); @(posedge clk);
    if (crc_out !== 32'h6178ddd9) begin $display("MISMATCH,trial=4,got=%08x,want=6178ddd9", crc_out); errors = errors + 1; end
    else $display("trial 4 OK, crc=%08x", crc_out);
    if (errors == 0) $display("CRC32_WORD_VERDICT,PASS");
    else $display("CRC32_WORD_VERDICT,FAIL,errors=%0d", errors);
    $finish;
  end
endmodule
