`timescale 1ns/1ps

module ai_accelerator_tb;
    reg clk, rst_n, cfg_we, cfg_sel, bias_we, bias_addr, secure_clear;
    reg start, relu_enable;
    reg [1:0] cfg_addr, result_addr;
    reg signed [7:0] cfg_wdata;
    reg signed [31:0] bias_wdata;
    reg [4:0] requant_shift;
    wire busy, done;
    wire security_fault;
    wire [1:0] fault_code;
    wire [7:0] cycle_count;
    wire signed [31:0] result_data;
    wire signed [7:0] quant_result_data;
    integer errors, random_errors_before, test_index;
    integer a0,a1,a2,a3,b0,b1,b2,b3,rbias0,rbias1;
    integer e0,e1,e2,e3,random_relu,random_shift;

    ai_accelerator_top dut (
        .clk(clk), .rst_n(rst_n), .cfg_we(cfg_we), .cfg_sel(cfg_sel),
        .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .bias_we(bias_we), .bias_addr(bias_addr), .bias_wdata(bias_wdata),
        .start(start), .relu_enable(relu_enable), .requant_shift(requant_shift),
        .secure_clear(secure_clear), .busy(busy), .done(done),
        .security_fault(security_fault), .fault_code(fault_code),
        .cycle_count(cycle_count),
        .result_addr(result_addr), .result_data(result_data),
        .quant_result_data(quant_result_data));

    always #5 clk = ~clk;

    function integer full_expected;
        input integer value; input integer enable_relu;
        begin full_expected = (enable_relu && value < 0) ? 0 : value; end
    endfunction

    function integer quant_expected;
        input integer value; input integer shift; input integer enable_relu;
        integer shifted;
        begin
            shifted = value >>> shift;
            if (enable_relu && shifted < 0) quant_expected = 0;
            else if (shifted > 127) quant_expected = 127;
            else if (shifted < -128) quant_expected = -128;
            else quant_expected = shifted;
        end
    endfunction

    task write_buffer;
        input select_weight; input [1:0] address; input signed [7:0] value;
        begin
            @(negedge clk); cfg_we=1; cfg_sel=select_weight;
            cfg_addr=address; cfg_wdata=value;
            @(negedge clk); cfg_we=0;
        end
    endtask

    task write_bias;
        input address; input signed [31:0] value;
        begin
            @(negedge clk); bias_we=1; bias_addr=address; bias_wdata=value;
            @(negedge clk); bias_we=0;
        end
    endtask

    task launch_and_wait;
        input enable_relu; input [4:0] shift;
        begin
            @(negedge clk); relu_enable=enable_relu; requant_shift=shift; start=1;
            @(negedge clk); start=0;
            wait(done==1'b1); #1;
        end
    endtask

    task secure_wipe;
        begin
            @(negedge clk); secure_clear=1;
            @(negedge clk); secure_clear=0; #1;
        end
    endtask

    task expect_both;
        input [1:0] address;
        input signed [31:0] expected_full;
        input integer expected_quant;
        begin
            result_addr=address; #1;
            if ($signed(result_data) !== expected_full) begin
                $display("  FAIL C32[%0d]: expected %0d, got %0d",
                         address,expected_full,$signed(result_data)); errors=errors+1;
            end else $display("  PASS C32[%0d] = %0d",address,$signed(result_data));
            if (quant_result_data !== expected_quant[7:0]) begin
                $display("  FAIL C8[%0d]: expected %0d, got %0d",
                         address,expected_quant,$signed(quant_result_data)); errors=errors+1;
            end else $display("  PASS C8[%0d] = %0d",address,$signed(quant_result_data));
        end
    endtask

    task check_both_quiet;
        input [1:0] address; input integer expected_full, expected_quant;
        begin
            result_addr=address; #1;
            if (($signed(result_data) != expected_full) ||
                ($signed(quant_result_data) != expected_quant)) begin
                $display("  RANDOM FAIL case=%0d C[%0d]: C32 %0d/%0d, C8 %0d/%0d",
                         test_index,address,$signed(result_data),expected_full,
                         $signed(quant_result_data),expected_quant);
                errors=errors+1;
            end
        end
    endtask

    task load_positive_example;
        begin
            // A=[[1,2],[3,4]], B=[[5,6],[7,8]]
            write_buffer(0,0,8'sd1); write_buffer(0,1,8'sd2);
            write_buffer(0,2,8'sd3); write_buffer(0,3,8'sd4);
            write_buffer(1,0,8'sd5); write_buffer(1,1,8'sd6);
            write_buffer(1,2,8'sd7); write_buffer(1,3,8'sd8);
        end
    endtask

    task load_signed_example;
        begin
            // A=[[1,-2],[-3,4]], B=[[5,6],[-7,8]]
            write_buffer(0,0, 8'sd1); write_buffer(0,1,-8'sd2);
            write_buffer(0,2,-8'sd3); write_buffer(0,3, 8'sd4);
            write_buffer(1,0, 8'sd5); write_buffer(1,1, 8'sd6);
            write_buffer(1,2,-8'sd7); write_buffer(1,3, 8'sd8);
        end
    endtask

    initial begin
        $dumpfile("build/ai_accelerator.vcd");
        $dumpvars(0,ai_accelerator_tb);
        clk=0; rst_n=0; cfg_we=0; cfg_sel=0; cfg_addr=0; cfg_wdata=0;
        bias_we=0; bias_addr=0; bias_wdata=0; start=0; relu_enable=0;
        requant_shift=0; secure_clear=0; result_addr=0; errors=0;
        repeat(3) @(negedge clk); rst_n=1;

        $display("TEST 1: matrix multiply, zero bias, ReLU");
        load_positive_example(); write_bias(0,0); write_bias(1,0);
        launch_and_wait(1,0);
        expect_both(0,19,19); expect_both(1,22,22);
        expect_both(2,43,43); expect_both(3,50,50);

        $display("TEST 2: signed INT8 with ReLU");
        load_signed_example(); launch_and_wait(1,0);
        expect_both(0,19,19); expect_both(1,0,0);
        expect_both(2,0,0); expect_both(3,14,14);

        $display("TEST 3: signed INT8 with ReLU bypassed");
        load_signed_example(); launch_and_wait(0,0);
        expect_both(0,19,19); expect_both(1,-10,-10);
        expect_both(2,-43,-43); expect_both(3,14,14);

        $display("TEST 4: per-channel bias and requantization shift");
        load_positive_example(); write_bias(0,5); write_bias(1,-10);
        launch_and_wait(1,1);
        // C32=[[24,12],[48,40]], C8=C32>>>1
        expect_both(0,24,12); expect_both(1,12,6);
        expect_both(2,48,24); expect_both(3,40,20);

        $display("TEST 5: positive INT8 saturation");
        write_bias(0,0); write_bias(1,0);
        write_buffer(0,0,127); write_buffer(0,1,127);
        write_buffer(0,2,127); write_buffer(0,3,127);
        write_buffer(1,0,127); write_buffer(1,1,127);
        write_buffer(1,2,127); write_buffer(1,3,127);
        launch_and_wait(1,0);
        expect_both(0,32258,127); expect_both(1,32258,127);
        expect_both(2,32258,127); expect_both(3,32258,127);

        $display("TEST 6: negative INT8 saturation");
        write_buffer(1,0,-127); write_buffer(1,1,-127);
        write_buffer(1,2,-127); write_buffer(1,3,-127);
        launch_and_wait(0,0);
        expect_both(0,-32258,-128); expect_both(1,-32258,-128);
        expect_both(2,-32258,-128); expect_both(3,-32258,-128);

        $display("TEST 7: 20 randomized architecture regressions");
        random_errors_before=errors;
        for(test_index=0; test_index<20; test_index=test_index+1) begin
            a0=$urandom_range(15,0)-8; a1=$urandom_range(15,0)-8;
            a2=$urandom_range(15,0)-8; a3=$urandom_range(15,0)-8;
            b0=$urandom_range(15,0)-8; b1=$urandom_range(15,0)-8;
            b2=$urandom_range(15,0)-8; b3=$urandom_range(15,0)-8;
            rbias0=$urandom_range(15,0)-8; rbias1=$urandom_range(15,0)-8;
            random_relu=$urandom_range(1,0); random_shift=$urandom_range(2,0);
            write_buffer(0,0,a0); write_buffer(0,1,a1);
            write_buffer(0,2,a2); write_buffer(0,3,a3);
            write_buffer(1,0,b0); write_buffer(1,1,b1);
            write_buffer(1,2,b2); write_buffer(1,3,b3);
            write_bias(0,rbias0); write_bias(1,rbias1);
            e0=a0*b0+a1*b2+rbias0; e1=a0*b1+a1*b3+rbias1;
            e2=a2*b0+a3*b2+rbias0; e3=a2*b1+a3*b3+rbias1;
            launch_and_wait(random_relu,random_shift);
            check_both_quiet(0,full_expected(e0,random_relu),quant_expected(e0,random_shift,random_relu));
            check_both_quiet(1,full_expected(e1,random_relu),quant_expected(e1,random_shift,random_relu));
            check_both_quiet(2,full_expected(e2,random_relu),quant_expected(e2,random_shift,random_relu));
            check_both_quiet(3,full_expected(e3,random_relu),quant_expected(e3,random_shift,random_relu));
        end
        if(errors==random_errors_before) $display("  PASS 20 randomized cases");

        $display("TEST 8: illegal write attack is blocked and logged");
        secure_wipe(); load_positive_example(); write_bias(0,0); write_bias(1,0);
        @(negedge clk); relu_enable=1; requant_shift=0; start=1;
        @(negedge clk); start=0;
        wait(busy==1'b1);
        @(negedge clk); cfg_we=1; cfg_sel=1; cfg_addr=0; cfg_wdata=8'sd99;
        @(negedge clk); cfg_we=0;
        wait(done==1'b1); #1;
        if (security_fault && fault_code==2'd1)
            $display("  PASS illegal write blocked, fault_code=1");
        else begin
            $display("  FAIL illegal write was not reported"); errors=errors+1;
        end
        // The blocked write must not corrupt the in-flight result.
        expect_both(0,19,19); expect_both(1,22,22);
        expect_both(2,43,43); expect_both(3,50,50);

        $display("TEST 9: parity detects weight fault injection");
        secure_wipe(); load_positive_example(); write_bias(0,0); write_bias(1,0);
        @(negedge clk);
        dut.weight_buffer[0] = dut.weight_buffer[0] ^ 8'h01;
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        repeat(2) @(negedge clk);
        if (security_fault && fault_code==2'd2 && !busy && !done)
            $display("  PASS weight tamper detected, execution refused");
        else begin
            $display("  FAIL integrity attack was not blocked"); errors=errors+1;
        end

        $display("TEST 10: secure_clear zeroizes architectural state");
        secure_wipe(); result_addr=0; #1;
        if (!security_fault && fault_code==0 && result_data==0 &&
            quant_result_data==0 && dut.weight_buffer[0]==0 &&
            dut.bias_buffer[0]==0)
            $display("  PASS secure zeroization complete");
        else begin
            $display("  FAIL secure zeroization incomplete"); errors=errors+1;
        end

        if(errors==0) $display("ALL TESTS PASSED");
        else $display("TESTS FAILED: %0d error(s)",errors);
        #20; $finish;
    end
endmodule
