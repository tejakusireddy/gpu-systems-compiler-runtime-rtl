// filename: matmul_accelerator.v
// purpose: Implements synthesizable FSM-driven matrix multiply accelerator.
// phase: Phase 5 - Hardware
// last modified: 2026-04-29

module matmul_accelerator #(parameter N = 4) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [(32*N*N)-1:0] a_flat,
    input  wire [(32*N*N)-1:0] b_flat,
    output reg  [(32*N*N)-1:0] c_flat,
    output reg         done
);

  localparam [1:0] STATE_IDLE    = 2'd0;
  localparam [1:0] STATE_COMPUTE = 2'd1;
  localparam [1:0] STATE_DONE    = 2'd2;

  reg [1:0] state;
  integer row_idx;
  integer col_idx;
  integer k_idx;
  integer out_index;
  reg [31:0] accum;
  integer init_idx;

  function [31:0] get_elem;
    input [(32*N*N)-1:0] bus;
    input integer idx;
    begin
      get_elem = bus[(idx * 32) +: 32];
    end
  endfunction

  always @(posedge clk) begin
    if (rst) begin
      state <= STATE_IDLE;
      row_idx <= 0;
      col_idx <= 0;
      done <= 1'b0;
      for (init_idx = 0; init_idx < (N * N); init_idx = init_idx + 1) begin
        c_flat[(init_idx * 32) +: 32] <= 32'd0;
      end
    end else begin
      done <= 1'b0;
      case (state)
        STATE_IDLE: begin
          if (start) begin
            row_idx <= 0;
            col_idx <= 0;
            state <= STATE_COMPUTE;
          end
        end

        STATE_COMPUTE: begin
          accum = 32'd0;
          for (k_idx = 0; k_idx < N; k_idx = k_idx + 1) begin
            accum = accum + (get_elem(a_flat, (row_idx * N) + k_idx) *
                             get_elem(b_flat, (k_idx * N) + col_idx));
          end
          out_index = (row_idx * N) + col_idx;
          c_flat[(out_index * 32) +: 32] <= accum;

          if (col_idx == (N - 1)) begin
            col_idx <= 0;
            if (row_idx == (N - 1)) begin
              row_idx <= 0;
              state <= STATE_DONE;
            end else begin
              row_idx <= row_idx + 1;
            end
          end else begin
            col_idx <= col_idx + 1;
          end
        end

        STATE_DONE: begin
          done <= 1'b1;
          state <= STATE_IDLE;
        end

        default: begin
          state <= STATE_IDLE;
        end
      endcase
    end
  end

endmodule
