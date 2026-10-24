// tb_cu.v - testbench for cu.v                                         Phase 2
// 1. Loads a memory image:   vvp cu_sim +image=image.hex
//    (made by run_rtl_tests.py; "@xxxx" lines give the word address)
// 2. Starts the CU at 0x10000 and runs until HALT, error, or timeout
// 3. Prints every register, the mask, every EXPORT, and every non-zero memory word,
//    so Python can compare them with simulator.py

`timescale 1ns/1ps

module tb_cu;
    reg clk = 1'b0;
    reg rst = 1'b1;
    reg start = 1'b0;
    always #5 clk = ~clk;                       // 100 MHz clock

    localparam WORDS = 32768;                   // 128 KB: byte addresses 0x00000-0x1FFFF
    reg [31:0] mem [0:WORDS-1];

    wire [31:0] imem_addr, dmem_addr, dmem_wdata, instr_count;
    wire        dmem_we, busy, done, error;
    wire        export_valid;
    wire [7:0]  export_target;
    wire [3:0]  export_mask;
    wire [127:0] export_data;
    wire [31:0] imem_data  = mem[imem_addr[16:2]];
    wire [31:0] dmem_rdata = mem[dmem_addr[16:2]];

    cu dut (
        .clk(clk), .rst(rst), .start(start), .start_pc(32'h0001_0000),
        .cu_id(4'd0), .wave_type(1'b0), .frame(32'd0),
        .init_v(1408'd0), .init_mask(4'hF), .init_s1(32'd0), .init_s2(32'd0),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .dmem_addr(dmem_addr), .dmem_wdata(dmem_wdata), .dmem_we(dmem_we),
        .dmem_rdata(dmem_rdata), .dmem_req(), .dmem_gnt(1'b1),
        .export_valid(export_valid), .export_target(export_target),
        .export_mask(export_mask), .export_data(export_data),
        .tex_start(), .tex_slot(), .tex_u(), .tex_v(), .tex_done(1'b0), .tex_result(128'd0),
        .busy(busy), .done(done), .error(error), .instr_count(instr_count)
    );

    always @(posedge clk)
        if (dmem_we) mem[dmem_addr[16:2]] <= dmem_wdata;

    // Print every EXPORT: target, mask, then lane 0..3
    always @(posedge clk)
        if (export_valid)
            $display("EXPORT %0d %h %h %h %h %h", export_target, export_mask,
                     export_data[31:0], export_data[63:32], export_data[95:64], export_data[127:96]);

    integer i;
    integer cycles;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        $readmemh(image, mem);

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        cycles = 0;
        while (!done && !error && cycles < 200000) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        @(posedge clk);                         // let the last memory write land

        if (done)       $display("STATUS DONE");
        else if (error) $display("STATUS ERROR");
        else            $display("STATUS TIMEOUT");
        $display("COUNT %0d", instr_count);
        $display("CYCLES %0d", cycles);
        $display("MASK %h", dut.mask);
        for (i = 0; i < 16; i = i + 1)
            $display("SREG %0d %h", i, dut.sreg[i]);
        for (i = 0; i < 64; i = i + 1)
            $display("VREG %0d %0d %h", i / 4, i % 4, dut.vreg[i]);
        for (i = 0; i < WORDS; i = i + 1)
            if (mem[i] != 32'd0) $display("MEM %h %h", i * 4, mem[i]);
        $finish;
    end
endmodule
