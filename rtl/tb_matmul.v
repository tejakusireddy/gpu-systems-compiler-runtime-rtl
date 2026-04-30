// filename: tb_matmul.v
// purpose: Testbench for 4x4 matrix multiply accelerator verification.
// phase: Phase 5 - Hardware
// last modified: 2026-04-29

module tb_matmul;

  localparam integer N = 4;
  localparam integer TOTAL_ELEMS = N * N;
  localparam integer CLK_HALF_PERIOD = 5;

  reg clk;
  reg rst;
  reg start;
  reg [(32*TOTAL_ELEMS)-1:0] a_flat;
  reg [(32*TOTAL_ELEMS)-1:0] b_flat;
  wire [(32*TOTAL_ELEMS)-1:0] c_flat;
  wire done;
  integer idx;

  matmul_accelerator #(.N(N)) dut (
      .clk(clk),
      .rst(rst),
      .start(start),
      .a_flat(a_flat),
      .b_flat(b_flat),
      .c_flat(c_flat),
      .done(done)
  );

  always #(CLK_HALF_PERIOD) clk = ~clk;

  task set_elem;
    output [(32*TOTAL_ELEMS)-1:0] bus;
    input integer elem_idx;
    input [31:0] value;
    begin
      bus[(elem_idx * 32) +: 32] = value;
    end
  endtask

  function [31:0] get_elem;
    input [(32*TOTAL_ELEMS)-1:0] bus;
    input integer elem_idx;
    begin
      get_elem = bus[(elem_idx * 32) +: 32];
    end
  endfunction

  initial begin
    clk = 1'b0;
    rst = 1'b1;
    start = 1'b0;
    a_flat = {(32*TOTAL_ELEMS){1'b0}};
    b_flat = {(32*TOTAL_ELEMS){1'b0}};

    set_elem(a_flat, 0, 32'd1);   set_elem(a_flat, 1, 32'd2);   set_elem(a_flat, 2, 32'd3);   set_elem(a_flat, 3, 32'd4);
    set_elem(a_flat, 4, 32'd5);   set_elem(a_flat, 5, 32'd6);   set_elem(a_flat, 6, 32'd7);   set_elem(a_flat, 7, 32'd8);
    set_elem(a_flat, 8, 32'd9);   set_elem(a_flat, 9, 32'd10);  set_elem(a_flat, 10, 32'd11); set_elem(a_flat, 11, 32'd12);
    set_elem(a_flat, 12, 32'd13); set_elem(a_flat, 13, 32'd14); set_elem(a_flat, 14, 32'd15); set_elem(a_flat, 15, 32'd16);

    set_elem(b_flat, 0, 32'd16);  set_elem(b_flat, 1, 32'd15);  set_elem(b_flat, 2, 32'd14);  set_elem(b_flat, 3, 32'd13);
    set_elem(b_flat, 4, 32'd12);  set_elem(b_flat, 5, 32'd11);  set_elem(b_flat, 6, 32'd10);  set_elem(b_flat, 7, 32'd9);
    set_elem(b_flat, 8, 32'd8);   set_elem(b_flat, 9, 32'd7);   set_elem(b_flat, 10, 32'd6);  set_elem(b_flat, 11, 32'd5);
    set_elem(b_flat, 12, 32'd4);  set_elem(b_flat, 13, 32'd3);  set_elem(b_flat, 14, 32'd2);  set_elem(b_flat, 15, 32'd1);

    #(2 * CLK_HALF_PERIOD);
    rst = 1'b0;

    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    wait (done == 1'b1);
    @(posedge clk);

    for (idx = 0; idx < TOTAL_ELEMS; idx = idx + 1) begin
      $display("c_flat[%0d] = %0d", idx, get_elem(c_flat, idx));
    end

    if (get_elem(c_flat, 0) !== 32'd80) begin
      $fatal(1, "Assertion failed: c_flat[0] expected 80, got %0d", get_elem(c_flat, 0));
    end
    if (get_elem(c_flat, 15) !== 32'd386) begin
      $fatal(1, "Assertion failed: c_flat[15] expected 386, got %0d", get_elem(c_flat, 15));
    end

    $finish;
  end

endmodule
