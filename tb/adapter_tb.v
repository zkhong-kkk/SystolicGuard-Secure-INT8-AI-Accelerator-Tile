`timescale 1ns/1ps

// File-driven testbench used by software/architecture_adapter.py.
module adapter_tb;
    reg clk=0, rst_n=0, cfg_we=0, cfg_sel=0, bias_we=0, bias_addr=0;
    reg start=0, relu_enable=0, secure_clear=0;
    reg [1:0] cfg_addr=0, result_addr=0;
    reg signed [7:0] cfg_wdata=0;
    reg signed [31:0] bias_wdata=0;
    reg [4:0] requant_shift=0;
    wire busy, done, security_fault;
    wire [1:0] fault_code;
    wire [7:0] cycle_count;
    wire signed [31:0] result_data;
    wire signed [7:0] quant_result_data;
    reg [31:0] vectors [0:18];
    integer i, errors=0;

    ai_accelerator_top dut (
        .clk(clk), .rst_n(rst_n), .cfg_we(cfg_we), .cfg_sel(cfg_sel),
        .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .bias_we(bias_we), .bias_addr(bias_addr), .bias_wdata(bias_wdata),
        .start(start), .relu_enable(relu_enable), .requant_shift(requant_shift),
        .secure_clear(secure_clear), .busy(busy), .done(done),
        .security_fault(security_fault), .fault_code(fault_code),
        .cycle_count(cycle_count), .result_addr(result_addr),
        .result_data(result_data), .quant_result_data(quant_result_data));

    always #5 clk=~clk;

    task write_buffer;
        input select_weight; input [1:0] address; input [7:0] value;
        begin
            @(negedge clk); cfg_we=1; cfg_sel=select_weight;
            cfg_addr=address; cfg_wdata=value;
            @(negedge clk); cfg_we=0;
        end
    endtask

    task write_bias;
        input address; input [31:0] value;
        begin
            @(negedge clk); bias_we=1; bias_addr=address; bias_wdata=value;
            @(negedge clk); bias_we=0;
        end
    endtask

    initial begin
        $dumpfile("build/adapter.vcd");
        $dumpvars(0,adapter_tb);
        $readmemh("build/adapter_vectors.hex",vectors);
        repeat(3) @(negedge clk); rst_n=1;
        for(i=0;i<4;i=i+1) write_buffer(0,i[1:0],vectors[i][7:0]);
        for(i=0;i<4;i=i+1) write_buffer(1,i[1:0],vectors[i+4][7:0]);
        write_bias(0,vectors[8]); write_bias(1,vectors[9]);
        relu_enable=vectors[10][0]; requant_shift=vectors[10][5:1];
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        wait(done); #1;
        for(i=0;i<4;i=i+1) begin
            result_addr=i[1:0]; #1;
            if(result_data !== vectors[11+i]) begin
                $display("ADAPTER_FAIL C32[%0d] got=%0d expected=%0d",
                         i,$signed(result_data),$signed(vectors[11+i]));
                errors=errors+1;
            end
            if(quant_result_data !== vectors[15+i][7:0]) begin
                $display("ADAPTER_FAIL C8[%0d] got=%0d expected=%0d",
                         i,$signed(quant_result_data),$signed(vectors[15+i][7:0]));
                errors=errors+1;
            end
            // Hold each address for a full 10 ns so all four outputs are
            // clearly visible in GTKWave instead of being compressed.
            #9;
        end
        if(errors==0 && !security_fault)
            $display("RTL_ADAPTER_PASS cycles=%0d",cycle_count);
        else
            $display("RTL_ADAPTER_FAILED errors=%0d fault=%0d",errors,fault_code);
        #10; $finish;
    end
endmodule
