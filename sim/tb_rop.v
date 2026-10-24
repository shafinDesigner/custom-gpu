// tb_rop.v - testbench for rop.v                                    Phase 5, stage 2
//   vvp rop_sim +image=rop.hex
// Memory image (made by run_rop_tests.py):
//   0x0F000  number of operations, width, height, fb, depth, stencil addresses
//   0x0F100 + 64*k  operation k:
//       word 0: type (0 quad, 1 clear) | depth_mode << 8 | blend_mode << 12 | color_mask << 16
//       quad : x, y, mask, close0-3, color0-3
//       clear: flags, color, depth, stencil
// At the end it prints the whole color, depth and stencil buffers.

`timescale 1ns/1ps

module tb_rop;
    localparam WORDS = 524288;                 // 2 MB

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];

    reg  [15:0]  width = 16'd0, height = 16'd0;
    reg  [31:0]  fb_addr = 32'd0, depth_addr = 32'd0, stencil_addr = 32'd0;
    reg  [1:0]   depth_mode = 2'd0, blend_mode = 2'd0;
    reg  [3:0]   color_mask = 4'hF;
    reg          clear_start = 1'b0;
    reg  [2:0]   clear_flags = 3'd0;
    reg  [31:0]  clear_color = 32'd0;
    reg  [15:0]  clear_depth = 16'd0;
    reg  [7:0]   clear_stencil = 8'd0;
    reg          quad_valid = 1'b0;
    reg  [15:0]  quad_x = 16'd0, quad_y = 16'd0;
    reg  [3:0]   quad_mask = 4'd0;
    reg  [127:0] quad_close = 128'd0, quad_color = 128'd0;
    wire         quad_ready, mem_req, mem_we, busy;
    wire [31:0]  mem_addr, mem_wdata, stat_written, stat_depth_failed;
    reg          lucky = 1'b1;
    wire         mem_gnt = mem_req && lucky;
    wire [31:0]  mem_rdata = mem[mem_addr[20:2]];

    rop dut (
        .clk(clk), .rst(rst), .width(width), .height(height),
        .fb_addr(fb_addr), .depth_addr(depth_addr), .stencil_addr(stencil_addr),
        .depth_mode(depth_mode), .blend_mode(blend_mode), .color_mask(color_mask),
        .clear_start(clear_start), .clear_flags(clear_flags), .clear_color(clear_color),
        .clear_depth(clear_depth), .clear_stencil(clear_stencil),
        .quad_valid(quad_valid), .quad_ready(quad_ready), .quad_x(quad_x), .quad_y(quad_y),
        .quad_mask(quad_mask), .quad_close(quad_close), .quad_color(quad_color),
        .mem_req(mem_req), .mem_addr(mem_addr), .mem_we(mem_we), .mem_wdata(mem_wdata),
        .mem_gnt(mem_gnt), .mem_rdata(mem_rdata),
        .busy(busy), .stat_written(stat_written), .stat_depth_failed(stat_depth_failed)
    );

    always @(posedge clk)
        if (mem_we) mem[mem_addr[20:2]] <= mem_wdata;

    integer seed = 23;
    always @(posedge clk) lucky <= ($random(seed) % 3) != 0;

    integer i, k, n, base, cycles, npix;
    reg [31:0] op;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        $readmemh(image, mem);
        n            = mem[32'h0F000 >> 2];
        width        = mem[(32'h0F000 >> 2) + 1];
        height       = mem[(32'h0F000 >> 2) + 2];
        fb_addr      = mem[(32'h0F000 >> 2) + 3];
        depth_addr   = mem[(32'h0F000 >> 2) + 4];
        stencil_addr = mem[(32'h0F000 >> 2) + 5];

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        cycles = 0;
        for (k = 0; k < n; k = k + 1) begin
            base = (32'h0F100 >> 2) + 16 * k;
            op   = mem[base];
            while (busy) begin @(posedge clk); cycles = cycles + 1; end
            depth_mode <= op[9:8];
            blend_mode <= op[13:12];
            color_mask <= op[19:16];
            if (op[7:0] == 8'd1) begin                   // clear
                clear_flags   <= mem[base + 1];
                clear_color   <= mem[base + 2];
                clear_depth   <= mem[base + 3];
                clear_stencil <= mem[base + 4];
                clear_start   <= 1'b1;
            end else begin                               // quad
                quad_x    <= mem[base + 1];
                quad_y    <= mem[base + 2];
                quad_mask <= mem[base + 3];
                for (i = 0; i < 4; i = i + 1) begin
                    quad_close[32*i +: 32] <= mem[base + 4 + i];
                    quad_color[32*i +: 32] <= mem[base + 8 + i];
                end
                quad_valid <= 1'b1;
            end
            @(posedge clk);                              // the ROP takes it here
            cycles = cycles + 1;
            clear_start <= 1'b0;
            quad_valid  <= 1'b0;
            @(posedge clk);
            cycles = cycles + 1;
        end
        while (busy) begin @(posedge clk); cycles = cycles + 1; end
        repeat (3) @(posedge clk);

        npix = width * height;
        for (i = 0; i < npix; i = i + 1)
            $display("FB %0d %h", i, mem[(fb_addr >> 2) + i]);
        for (i = 0; i < (npix + 1) / 2; i = i + 1)
            $display("DEPTH %0d %h", i, mem[(depth_addr >> 2) + i]);
        for (i = 0; i < (npix + 3) / 4; i = i + 1)
            $display("STENCIL %0d %h", i, mem[(stencil_addr >> 2) + i]);
        $display("STATS %0d %0d %0d", stat_written, stat_depth_failed, cycles);
        $display("STATUS DONE");
        $finish;
    end
endmodule
