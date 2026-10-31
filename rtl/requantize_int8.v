`timescale 1ns/1ps

// Quantized-inference post-processing unit.
// The INT32 accumulator is rescaled by an arithmetic right shift, optionally
// passed through ReLU, then saturated into the signed INT8 range.
module requantize_int8 (
    input  wire signed [31:0] data_in,
    input  wire        [4:0]  shift_amount,
    input  wire               relu_enable,
    output reg  signed [7:0]  data_out
);
    wire signed [31:0] shifted;
    assign shifted = data_in >>> shift_amount;

    always @(*) begin
        if (relu_enable && shifted < 0)
            data_out = 8'sd0;
        else if (shifted > 32'sd127)
            data_out = 8'sd127;
        else if (shifted < -32'sd128)
            data_out = -8'sd128;
        else
            data_out = shifted[7:0];
    end
endmodule
