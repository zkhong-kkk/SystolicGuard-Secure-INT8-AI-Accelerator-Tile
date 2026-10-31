`timescale 1ns/1ps

// Signed INT8 multiply-accumulate Processing Element (PE).
// Activations move right, weights move down, and the partial sum stays here.
module mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear,
    input  wire                         enable,
    input  wire signed [DATA_WIDTH-1:0] a_in,
    input  wire signed [DATA_WIDTH-1:0] b_in,
    output reg  signed [DATA_WIDTH-1:0] a_out,
    output reg  signed [DATA_WIDTH-1:0] b_out,
    output reg  signed [ACC_WIDTH-1:0]  accumulator
);
    wire signed [(2*DATA_WIDTH)-1:0] product;
    wire signed [ACC_WIDTH-1:0] extended_product;

    assign product = a_in * b_in;
    assign extended_product = {
        {(ACC_WIDTH-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}}, product
    };

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= {DATA_WIDTH{1'b0}};
            b_out <= {DATA_WIDTH{1'b0}};
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (clear) begin
            a_out <= {DATA_WIDTH{1'b0}};
            b_out <= {DATA_WIDTH{1'b0}};
            accumulator <= {ACC_WIDTH{1'b0}};
        end else if (enable) begin
            a_out <= a_in;
            b_out <= b_in;
            accumulator <= accumulator + extended_product;
        end
    end
endmodule
