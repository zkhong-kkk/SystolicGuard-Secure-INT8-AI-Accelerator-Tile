`timescale 1ns/1ps

// ReLU activation. Bypass preserves the raw signed accumulator value.
module relu #(parameter WIDTH = 32) (
    input wire bypass,
    input wire signed [WIDTH-1:0] data_in,
    output wire signed [WIDTH-1:0] data_out
);
    assign data_out = bypass ? data_in :
                      (data_in[WIDTH-1] ? {WIDTH{1'b0}} : data_in);
endmodule
