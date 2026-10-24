// tb_tex.v - testbench for tex_unit.v                                    Phase 6 (early)
//   vvp tex_sim +image=tex.hex
// The memory image (made by run_tex_tests.py) holds:
//   0x0F000          number of requests
//   0x0F010..0x0F01C texture address for slots 0-3
//   0x0F020..0x0F02C texture info for slots 0-3
//   0x0F100 + 64*k   request k: [slot | wave_type << 8], u0..u3, v0..v3
//   textures from 0x100000 (spec section 9)
// Memory is randomly NOT granted about 1 cycle in 3, to test waiting.

`timescale 1ns/1ps

module tb_tex;
    localparam WORDS = 524288;                 // 2 MB: 0x000000-0x1FFFFF

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];

    reg          start = 1'b0;
    reg  [1:0]   slot = 2'd0;
    reg          wave_type = 1'b0;
    reg  [127:0] u = 128'd0, v = 128'd0, tex_addr = 128'd0, tex_info = 128'd0;
    wire         mem_req, done, busy;
    wire [31:0]  mem_addr;
    wire [127:0] result;
    reg          lucky = 1'b1;                 // random memory grant
    wire         mem_gnt = mem_req && lucky;
    wire [31:0]  mem_rdata = mem[mem_addr[20:2]];

    tex_unit dut (
        .clk(clk), .rst(rst), .start(start), .slot(slot), .wave_type(wave_type),
        .u(u), .v(v), .tex_addr(tex_addr), .tex_info(tex_info),
        .mem_req(mem_req), .mem_addr(mem_addr), .mem_gnt(mem_gnt), .mem_rdata(mem_rdata),
        .done(done), .result(result), .busy(busy)
    );

    integer seed = 7;
    always @(posedge clk) lucky <= ($random(seed) % 3) != 0;

    integer bad = 0;
    always @(posedge clk)
        if (mem_req && mem_addr[31:21] != 0) begin
            if (bad < 5) $display("BADADDR %h", mem_addr);
            bad = bad + 1;
        end

    integer i, k, n, base, cycles, waited;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        $readmemh(image, mem);
        n = mem[32'h0F000 >> 2];
        for (i = 0; i < 4; i = i + 1) begin
            tex_addr[32*i +: 32] = mem[(32'h0F010 >> 2) + i];
            tex_info[32*i +: 32] = mem[(32'h0F020 >> 2) + i];
        end

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        cycles = 0;
        for (k = 0; k < n; k = k + 1) begin
            base = (32'h0F100 >> 2) + 16 * k;
            slot      <= mem[base][1:0];
            wave_type <= mem[base][8];
            for (i = 0; i < 4; i = i + 1) begin
                u[32*i +: 32] <= mem[base + 1 + i];
                v[32*i +: 32] <= mem[base + 5 + i];
            end
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            waited = 0;
            while (!done && waited < 10000) begin
                @(posedge clk);
                waited = waited + 1;
            end
            cycles = cycles + waited + 1;
            $display("RESULT %0d %h %h %h %h", k,
                     result[31:0], result[63:32], result[95:64], result[127:96]);
            @(posedge clk);
        end
        $display("CYCLES %0d", cycles);
        $display("BADCOUNT %0d", bad);
        $display("STATUS DONE");
        $finish;
    end
endmodule
