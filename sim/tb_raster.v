// tb_raster.v - testbench for rasterizer.v                             Phase 5, stage 1
//   vvp raster_sim +image=raster.hex
// Memory image (made by run_raster_tests.py):
//   0x0F000  number of triangles     0x0F004 width   0x0F008 height
//   0x0F00C  depth mode              0x0F010 depth buffer address
//   0x40000 + 256*t  triangle t: prim, n_attrs, x[a,b,c], y[a,b,c], z[a,b,c],
//                    attrs of a (8), attrs of b (8), attrs of c (8)
// Memory grant and quad_ready are random, to test every kind of waiting.

`timescale 1ns/1ps

module tb_raster;
    localparam WORDS = 524288;                 // 2 MB

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];

    reg          tri_valid = 1'b0;
    reg  [31:0]  tri_prim = 32'd0;
    reg  [95:0]  tri_x = 96'd0, tri_y = 96'd0, tri_z = 96'd0;
    reg  [767:0] tri_attr = 768'd0;
    reg  [3:0]   n_attrs = 4'd0;
    reg  [15:0]  width = 16'd64, height = 16'd36;
    reg  [1:0]   depth_mode = 2'd0;
    reg  [31:0]  depth_addr = 32'd0;
    wire         tri_ready, mem_req, quad_valid, busy;
    wire [31:0]  mem_addr, quad_prim, stat_quads, stat_early_dropped;
    wire [15:0]  quad_x, quad_y;
    wire [3:0]   quad_mask;
    wire [127:0] quad_close;
    wire [1023:0] quad_attr;
    reg          lucky = 1'b1;
    reg          quad_ready = 1'b0;
    wire         mem_gnt = mem_req && lucky;
    wire [31:0]  mem_rdata = mem[mem_addr[20:2]];

    rasterizer dut (
        .clk(clk), .rst(rst),
        .tri_valid(tri_valid), .tri_ready(tri_ready), .tri_prim(tri_prim),
        .tri_x(tri_x), .tri_y(tri_y), .tri_z(tri_z), .tri_attr(tri_attr), .n_attrs(n_attrs),
        .width(width), .height(height), .depth_mode(depth_mode), .depth_addr(depth_addr),
        .mem_req(mem_req), .mem_addr(mem_addr), .mem_gnt(mem_gnt), .mem_rdata(mem_rdata),
        .quad_valid(quad_valid), .quad_ready(quad_ready), .quad_prim(quad_prim),
        .quad_x(quad_x), .quad_y(quad_y), .quad_mask(quad_mask),
        .quad_close(quad_close), .quad_attr(quad_attr),
        .busy(busy), .stat_quads(stat_quads), .stat_early_dropped(stat_early_dropped)
    );

    integer seed = 11;
    always @(posedge clk) begin
        lucky      <= ($random(seed) % 3) != 0;
        quad_ready <= ($random(seed) % 4) != 0;
    end

    // print every quad handed over
    integer a, l;
    always @(posedge clk)
        if (quad_valid && quad_ready) begin
            $write("QUAD %0d %0d %0d %h", quad_prim, quad_x, quad_y, quad_mask);
            for (l = 0; l < 4; l = l + 1) $write(" %h", quad_close[32*l +: 32]);
            for (a = 0; a < 8; a = a + 1)
                for (l = 0; l < 4; l = l + 1) $write(" %h", quad_attr[128*a + 32*l +: 32]);
            $write("\n");
        end

    integer i, t, v, j, n, base, cycles;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        $readmemh(image, mem);
        n          = mem[32'h0F000 >> 2];
        width      = mem[32'h0F004 >> 2];
        height     = mem[32'h0F008 >> 2];
        depth_mode = mem[32'h0F00C >> 2];
        depth_addr = mem[32'h0F010 >> 2];

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        cycles = 0;
        for (t = 0; t < n; t = t + 1) begin
            base = (32'h40000 >> 2) + 64 * t;
            while (!tri_ready) begin @(posedge clk); cycles = cycles + 1; end
            tri_prim <= mem[base];
            n_attrs  <= mem[base + 1];
            for (v = 0; v < 3; v = v + 1) begin
                tri_x[32*v +: 32] <= mem[base + 2 + v];
                tri_y[32*v +: 32] <= mem[base + 5 + v];
                tri_z[32*v +: 32] <= mem[base + 8 + v];
                for (j = 0; j < 8; j = j + 1)
                    tri_attr[256*v + 32*j +: 32] <= mem[base + 11 + 8*v + j];
            end
            tri_valid <= 1'b1;
            @(posedge clk);                     // the rasterizer takes it here
            cycles = cycles + 1;
            tri_valid <= 1'b0;
            @(posedge clk);
            cycles = cycles + 1;
        end
        while (busy || quad_valid) begin @(posedge clk); cycles = cycles + 1; end
        repeat (3) @(posedge clk);

        $display("STATS %0d %0d %0d", stat_quads, stat_early_dropped, cycles);
        $display("STATUS DONE");
        $finish;
    end
endmodule
