`timescale 1ns/1ps

// 2x2 INT8 AI matrix accelerator with quantized-inference post-processing.
// Full precision: C32 = ReLU(A x B + channel_bias)
// Quantized:      C8  = saturate_INT8(ReLU(C32 >>> requant_shift))
// cfg_sel=0 writes activation buffer A; cfg_sel=1 writes weight buffer B.
// cfg_addr 0,1,2,3 maps to matrix entries 00,01,10,11.
module ai_accelerator_top #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input wire clk, input wire rst_n,
    input wire cfg_we, input wire cfg_sel,
    input wire [1:0] cfg_addr,
    input wire signed [DATA_WIDTH-1:0] cfg_wdata,
    input wire bias_we, input wire bias_addr,
    input wire signed [ACC_WIDTH-1:0] bias_wdata,
    input wire start, input wire relu_enable,
    input wire [4:0] requant_shift,
    input wire secure_clear,
    output wire busy, output wire done,
    output reg security_fault,
    output reg [1:0] fault_code,
    output reg [7:0] cycle_count,
    input wire [1:0] result_addr,
    output reg signed [ACC_WIDTH-1:0] result_data,
    output reg signed [DATA_WIDTH-1:0] quant_result_data
);
    localparam [2:0] S_IDLE=3'd0, S_PREP=3'd1, S_COMPUTE=3'd2,
                     S_ACTIVATE=3'd3, S_DONE=3'd4;
    localparam [1:0] FAULT_NONE=2'd0, FAULT_ILLEGAL_WRITE=2'd1,
                     FAULT_INTEGRITY=2'd2;
    reg [2:0] state;
    reg [1:0] compute_cycle;
    reg relu_latched;
    reg [4:0] requant_shift_latched;
    reg signed [DATA_WIDTH-1:0] input_buffer [0:3];
    reg signed [DATA_WIDTH-1:0] weight_buffer [0:3];
    // One bias per output column, matching neural-network output channels.
    reg signed [ACC_WIDTH-1:0] bias_buffer [0:1];
    reg signed [ACC_WIDTH-1:0] output_buffer [0:3];
    reg signed [DATA_WIDTH-1:0] quant_output_buffer [0:3];
    reg input_parity [0:3];
    reg weight_parity [0:3];
    reg bias_parity [0:1];
    reg signed [DATA_WIDTH-1:0] a_left_0, a_left_1, b_top_0, b_top_1;
    wire array_clear = (state == S_PREP) || secure_clear;
    wire array_enable = (state == S_COMPUTE);
    wire integrity_error;
    wire signed [ACC_WIDTH-1:0] raw_c00, raw_c01, raw_c10, raw_c11;
    wire signed [ACC_WIDTH-1:0] biased_c00, biased_c01, biased_c10, biased_c11;
    wire signed [ACC_WIDTH-1:0] activated_c00, activated_c01,
                                activated_c10, activated_c11;
    wire signed [DATA_WIDTH-1:0] quant_c00, quant_c01, quant_c10, quant_c11;
    integer i;

    assign busy = (state==S_PREP) || (state==S_COMPUTE) ||
                  (state==S_ACTIVATE);
    assign done = (state==S_DONE);
    assign integrity_error =
        (input_parity[0]  != ^input_buffer[0])  ||
        (input_parity[1]  != ^input_buffer[1])  ||
        (input_parity[2]  != ^input_buffer[2])  ||
        (input_parity[3]  != ^input_buffer[3])  ||
        (weight_parity[0] != ^weight_buffer[0]) ||
        (weight_parity[1] != ^weight_buffer[1]) ||
        (weight_parity[2] != ^weight_buffer[2]) ||
        (weight_parity[3] != ^weight_buffer[3]) ||
        (bias_parity[0]   != ^bias_buffer[0])   ||
        (bias_parity[1]   != ^bias_buffer[1]);

    // Skewed input schedule aligns matching A/B values inside the array.
    // Explicit array-word sensitivity keeps older Verilog simulators quiet.
    always @(state or compute_cycle or
             input_buffer[0] or input_buffer[1] or
             input_buffer[2] or input_buffer[3] or
             weight_buffer[0] or weight_buffer[1] or
             weight_buffer[2] or weight_buffer[3]) begin
        a_left_0 = 0; a_left_1 = 0; b_top_0 = 0; b_top_1 = 0;
        if (state == S_COMPUTE) begin
            case (compute_cycle)
                2'd0: begin
                    a_left_0=input_buffer[0]; b_top_0=weight_buffer[0];
                end
                2'd1: begin
                    a_left_0=input_buffer[1]; a_left_1=input_buffer[2];
                    b_top_0=weight_buffer[2]; b_top_1=weight_buffer[1];
                end
                2'd2: begin
                    a_left_1=input_buffer[3]; b_top_1=weight_buffer[3];
                end
                default: begin end // pipeline drain cycle
            endcase
        end
    end

    systolic_array_2x2 #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) array_i (
        .clk(clk), .rst_n(rst_n), .clear(array_clear), .enable(array_enable),
        .a_left_0(a_left_0), .a_left_1(a_left_1),
        .b_top_0(b_top_0), .b_top_1(b_top_1),
        .c00(raw_c00), .c01(raw_c01), .c10(raw_c10), .c11(raw_c11));

    // A fully-connected output channel shares one bias across all input rows.
    assign biased_c00 = raw_c00 + bias_buffer[0];
    assign biased_c01 = raw_c01 + bias_buffer[1];
    assign biased_c10 = raw_c10 + bias_buffer[0];
    assign biased_c11 = raw_c11 + bias_buffer[1];

    relu #(.WIDTH(ACC_WIDTH)) relu_00
        (.bypass(!relu_latched), .data_in(biased_c00), .data_out(activated_c00));
    relu #(.WIDTH(ACC_WIDTH)) relu_01
        (.bypass(!relu_latched), .data_in(biased_c01), .data_out(activated_c01));
    relu #(.WIDTH(ACC_WIDTH)) relu_10
        (.bypass(!relu_latched), .data_in(biased_c10), .data_out(activated_c10));
    relu #(.WIDTH(ACC_WIDTH)) relu_11
        (.bypass(!relu_latched), .data_in(biased_c11), .data_out(activated_c11));

    requantize_int8 requant_00
        (.data_in(biased_c00), .shift_amount(requant_shift_latched),
         .relu_enable(relu_latched), .data_out(quant_c00));
    requantize_int8 requant_01
        (.data_in(biased_c01), .shift_amount(requant_shift_latched),
         .relu_enable(relu_latched), .data_out(quant_c01));
    requantize_int8 requant_10
        (.data_in(biased_c10), .shift_amount(requant_shift_latched),
         .relu_enable(relu_latched), .data_out(quant_c10));
    requantize_int8 requant_11
        (.data_in(biased_c11), .shift_amount(requant_shift_latched),
         .relu_enable(relu_latched), .data_out(quant_c11));

    always @(result_addr or
             output_buffer[0] or output_buffer[1] or
             output_buffer[2] or output_buffer[3] or
             quant_output_buffer[0] or quant_output_buffer[1] or
             quant_output_buffer[2] or quant_output_buffer[3]) begin
        case (result_addr)
            2'd0: begin result_data=output_buffer[0]; quant_result_data=quant_output_buffer[0]; end
            2'd1: begin result_data=output_buffer[1]; quant_result_data=quant_output_buffer[1]; end
            2'd2: begin result_data=output_buffer[2]; quant_result_data=quant_output_buffer[2]; end
            default: begin result_data=output_buffer[3]; quant_result_data=quant_output_buffer[3]; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; compute_cycle<=0; relu_latched<=0;
            requant_shift_latched<=0; cycle_count<=0;
            security_fault<=0; fault_code<=FAULT_NONE;
            bias_buffer[0]<=0; bias_buffer[1]<=0;
            bias_parity[0]<=0; bias_parity[1]<=0;
            for (i=0; i<4; i=i+1) begin
                input_buffer[i]<=0; weight_buffer[i]<=0; output_buffer[i]<=0;
                quant_output_buffer[i]<=0;
                input_parity[i]<=0; weight_parity[i]<=0;
            end
        end else if (secure_clear) begin
            // Emergency zeroization clears model/data state and unlocks faults.
            state<=S_IDLE; compute_cycle<=0; relu_latched<=0;
            requant_shift_latched<=0; cycle_count<=0;
            security_fault<=0; fault_code<=FAULT_NONE;
            bias_buffer[0]<=0; bias_buffer[1]<=0;
            bias_parity[0]<=0; bias_parity[1]<=0;
            for (i=0; i<4; i=i+1) begin
                input_buffer[i]<=0; weight_buffer[i]<=0; output_buffer[i]<=0;
                quant_output_buffer[i]<=0;
                input_parity[i]<=0; weight_parity[i]<=0;
            end
        end else begin
            // Writes during computation are blocked and recorded as a fault.
            if ((cfg_we || bias_we) && busy) begin
                security_fault<=1; fault_code<=FAULT_ILLEGAL_WRITE;
            end
            if (cfg_we && ((state==S_IDLE) || (state==S_DONE)) && !security_fault) begin
                if (cfg_sel) begin
                    weight_buffer[cfg_addr] <= cfg_wdata;
                    weight_parity[cfg_addr] <= ^cfg_wdata;
                end else begin
                    input_buffer[cfg_addr] <= cfg_wdata;
                    input_parity[cfg_addr] <= ^cfg_wdata;
                end
            end
            if (bias_we && ((state==S_IDLE) || (state==S_DONE)) && !security_fault) begin
                bias_buffer[bias_addr] <= bias_wdata;
                bias_parity[bias_addr] <= ^bias_wdata;
            end
            case (state)
                S_IDLE: if (start) begin
                    if (security_fault || integrity_error) begin
                        security_fault<=1;
                        if (integrity_error) fault_code<=FAULT_INTEGRITY;
                        state<=S_IDLE;
                    end else begin
                        relu_latched<=relu_enable; cycle_count<=0;
                        requant_shift_latched<=requant_shift;
                        compute_cycle<=0; state<=S_PREP;
                    end
                end
                S_PREP: begin
                    cycle_count<=cycle_count+1'b1; compute_cycle<=0;
                    state<=S_COMPUTE;
                end
                S_COMPUTE: begin
                    cycle_count<=cycle_count+1'b1;
                    if (compute_cycle==2'd3) state<=S_ACTIVATE;
                    else compute_cycle<=compute_cycle+1'b1;
                end
                S_ACTIVATE: begin
                    output_buffer[0]<=activated_c00;
                    output_buffer[1]<=activated_c01;
                    output_buffer[2]<=activated_c10;
                    output_buffer[3]<=activated_c11;
                    quant_output_buffer[0]<=quant_c00;
                    quant_output_buffer[1]<=quant_c01;
                    quant_output_buffer[2]<=quant_c10;
                    quant_output_buffer[3]<=quant_c11;
                    cycle_count<=cycle_count+1'b1; state<=S_DONE;
                end
                S_DONE: state<=S_IDLE;
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
