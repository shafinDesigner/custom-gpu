// tb_gpu_core.v - testbench for gpu_core.v                             Phase 3
//   vvp core_sim +image=scene.hex +draws=2
// 1. Loads the memory image (pixel shader at 0x10000, constants at 0x20000 + 0x40 x draw,
//    texture settings at 0x0F010 / 0x0F020, textures from 0x100000)
// 2. Runs each draw: pulse draw_start, wait for draw_done
// 3. Writes exported pixels into a frame buffer, counting writes per pixel
// 4. Prints statistics and every frame buffer pixel for run_phase3.py

`timescale 1ns/1ps

module tb_gpu_core;
    localparam [15:0] W = 16'd64, H = 16'd36;  // small 16:9 screen (fast to simulate)
    localparam WORDS = 524288;                 // 2 MB memory: 0x000000-0x1FFFFF (textures at 0x100000)
    localparam BACKGROUND = 32'hFF1E1E1E;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];
    reg [31:0] fb  [0:W*H-1];                  // frame buffer
    integer    writes [0:W*H-1];               // how often each pixel was written

    reg         draw_start = 1'b0;
    reg  [31:0] const_addr = 32'h0002_0000;
    reg  [127:0] tex_addr = 128'd0, tex_info = 128'd0;
    wire        draw_done, draw_error;
    wire [31:0] imem0_addr, imem1_addr, mem_addr, mem_wdata;
    wire        mem_we;
    wire [31:0] imem0_data = mem[imem0_addr[20:2]];
    wire [31:0] imem1_data = mem[imem1_addr[20:2]];
    wire [31:0] mem_rdata  = mem[mem_addr[20:2]];
    wire        pix0_valid, pix1_valid;
    wire [15:0] pix0_x, pix0_y, pix1_x, pix1_y;
    wire [3:0]  pix0_mask, pix1_mask;
    wire [127:0] pix0_data, pix1_data;
    wire [31:0] waves_cu0, waves_cu1, mem_conflicts;

    gpu_core dut (
        .clk(clk), .rst(rst),
        .draw_start(draw_start), .ps_addr(32'h0001_0000), .const_addr(const_addr),
        .width(W), .height(H), .tex_addr(tex_addr), .tex_info(tex_info),
        .draw_done(draw_done), .draw_error(draw_error),
        .imem0_addr(imem0_addr), .imem0_data(imem0_data),
        .imem1_addr(imem1_addr), .imem1_data(imem1_data),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_we(mem_we), .mem_rdata(mem_rdata),
        .pix0_valid(pix0_valid), .pix0_x(pix0_x), .pix0_y(pix0_y),
        .pix0_mask(pix0_mask), .pix0_data(pix0_data),
        .pix1_valid(pix1_valid), .pix1_x(pix1_x), .pix1_y(pix1_y),
        .pix1_mask(pix1_mask), .pix1_data(pix1_data),
        .waves_cu0(waves_cu0), .waves_cu1(waves_cu1), .mem_conflicts(mem_conflicts)
    );

    always @(posedge clk)
        if (mem_we) mem[mem_addr[20:2]] <= mem_wdata;

    // ---- ALARM: any memory access outside the 256 KB we have ----
    integer bad_addresses = 0;
    always @(posedge clk) begin
        if ((dut.gnt0 || dut.gnt1) && mem_addr[31:21] != 0) begin
            if (bad_addresses < 5) $display("BADADDR data %h", mem_addr);
            bad_addresses = bad_addresses + 1;
        end
        if (dut.cu0.busy && imem0_addr[31:21] != 0) begin
            if (bad_addresses < 5) $display("BADADDR instruction %h", imem0_addr);
            bad_addresses = bad_addresses + 1;
        end
    end

    // ---- simple pixel writer: 4 pixels of a quad per EXPORT COLOR ----
    integer l, px, py, idx;
    always @(posedge clk) begin
        if (pix0_valid)
            for (l = 0; l < 4; l = l + 1) begin
                px = pix0_x + (l % 2);  py = pix0_y + (l / 2);  idx = py * W + px;
                if (pix0_mask[l] && px < W && py < H) begin
                    fb[idx] <= pix0_data[32*l +: 32];
                    writes[idx] = writes[idx] + 1;
                end
            end
        if (pix1_valid)
            for (l = 0; l < 4; l = l + 1) begin
                px = pix1_x + (l % 2);  py = pix1_y + (l / 2);  idx = py * W + px;
                if (pix1_mask[l] && px < W && py < H) begin
                    fb[idx] <= pix1_data[32*l +: 32];
                    writes[idx] = writes[idx] + 1;
                end
            end
    end

    integer i, d, draws, cycles, overdraw;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        for (i = 0; i < W*H; i = i + 1) begin fb[i] = BACKGROUND; writes[i] = 0; end
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        if (!$value$plusargs("draws=%d", draws)) draws = 1;
        $readmemh(image, mem);
        for (i = 0; i < 4; i = i + 1) begin                 // texture slots 0-3
            tex_addr[32*i +: 32] = mem[(32'h0F010 >> 2) + i];
            tex_info[32*i +: 32] = mem[(32'h0F020 >> 2) + i];
        end

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        cycles = 0;
        for (d = 0; d < draws; d = d + 1) begin
            const_addr <= 32'h0002_0000 + 32'h40 * d;     // one constant block per draw
            draw_start <= 1'b1;
            @(posedge clk);
            draw_start <= 1'b0;
            @(posedge clk);
            while (!draw_done && !draw_error && cycles < 2000000) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
        end
        repeat (2) @(posedge clk);

        overdraw = 0;
        for (i = 0; i < W*H; i = i + 1) if (writes[i] > 1) overdraw = overdraw + 1;
        if (draw_error)            $display("STATUS ERROR");
        else if (cycles >= 2000000) $display("STATUS TIMEOUT");
        else                       $display("STATUS DONE");
        $display("CYCLES %0d", cycles);
        $display("WAVES %0d %0d", waves_cu0, waves_cu1);
        $display("CONFLICTS %0d", mem_conflicts);
        $display("OVERDRAW %0d", overdraw);
        $display("SIZE %0d %0d", W, H);
        $display("BADCOUNT %0d", bad_addresses);
        for (i = 0; i < W*H; i = i + 1) $display("FB %0d %h", i, fb[i]);
        $finish;
    end
endmodule
