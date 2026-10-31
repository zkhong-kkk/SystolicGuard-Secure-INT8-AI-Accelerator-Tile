`timescale 1ns/1ps

// 2x2 output-stationary systolic array. Each PE computes one C element.
module systolic_array_2x2 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input wire clk, input wire rst_n, input wire clear, input wire enable,
    input wire signed [DATA_WIDTH-1:0] a_left_0, a_left_1,
    input wire signed [DATA_WIDTH-1:0] b_top_0, b_top_1,
    output wire signed [ACC_WIDTH-1:0] c00, c01, c10, c11
);
    wire signed [DATA_WIDTH-1:0] a_00_to_01, a_10_to_11;
    wire signed [DATA_WIDTH-1:0] b_00_to_10, b_01_to_11;
    wire signed [DATA_WIDTH-1:0] a_east_0, a_east_1;
    wire signed [DATA_WIDTH-1:0] b_south_0, b_south_1;

    mac_pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_00 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable),
        .a_in(a_left_0), .b_in(b_top_0),
        .a_out(a_00_to_01), .b_out(b_00_to_10), .accumulator(c00));
    mac_pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_01 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable),
        .a_in(a_00_to_01), .b_in(b_top_1),
        .a_out(a_east_0), .b_out(b_01_to_11), .accumulator(c01));
    mac_pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_10 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable),
        .a_in(a_left_1), .b_in(b_00_to_10),
        .a_out(a_10_to_11), .b_out(b_south_0), .accumulator(c10));
    mac_pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_11 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable),
        .a_in(a_10_to_11), .b_in(b_01_to_11),
        .a_out(a_east_1), .b_out(b_south_1), .accumulator(c11));
endmodule
