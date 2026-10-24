// tb_prim.v - testbench for prim_unit.v                              Phase 5, stage 3
//   vvp prim_sim +image=prim.hex
// Memory image (made by run_prim_tests.py):
//   0x0F000           number of groups (draws)
//   0x0F100 + 64*g    group g: n_tris, first_tri, cx, cy, focal, near_z, cull_mode, n_attrs
//   0x40000 + 256*t   triangle t: x[a,b,c], y[a,b,c], z[a,b,c], attrs a(8), b(8), c(8)
// out_ready is random, to test waiting.

`timescale 1ns/1ps

module tb_prim;
    localparam WORDS = 524288;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];

    reg          draw_start = 1'b0;
    reg  [31:0]  proj_cx = 32'd0, proj_cy = 32'd0, proj_focal = 32'd0, near_z = 32'd0;
    reg  [1:0]   cull_mode = 2'd0;
    reg          in_valid = 1'b0;
    reg  [95:0]  in_x = 96'd0, in_y = 96'd0, in_z = 96'd0;
    reg  [767:0] in_attr = 768'd0;
    reg  [3:0]   n_attrs = 4'd0;
    wire         in_ready, out_valid, busy;
    reg          out_ready = 1'b0;
    wire [31:0]  out_prim, st_in, st_near, st_zero, st_back, st_out;
    wire [95:0]  out_x, out_y, out_z;
    wire [767:0] out_attr;
    wire [3:0]   out_nattrs;

    prim_unit dut (
        .clk(clk), .rst(rst), .draw_start(draw_start),
        .proj_cx(proj_cx), .proj_cy(proj_cy), .proj_focal(proj_focal), .near_z(near_z),
        .cull_mode(cull_mode),
        .in_valid(in_valid), .in_ready(in_ready), .in_x(in_x), .in_y(in_y), .in_z(in_z),
        .in_attr(in_attr), .n_attrs(n_attrs),
        .out_valid(out_valid), .out_ready(out_ready), .out_prim(out_prim),
        .out_x(out_x), .out_y(out_y), .out_z(out_z), .out_attr(out_attr), .out_nattrs(out_nattrs),
        .busy(busy), .st_in(st_in), .st_near(st_near), .st_zero(st_zero),
        .st_back(st_back), .st_out(st_out)
    );

    integer seed = 31;
    always @(posedge clk) out_ready <= ($random(seed) % 3) != 0;

    integer a, v;
    always @(posedge clk)
        if (out_valid && out_ready) begin
            $write("TRI %0d", out_prim);
            for (v = 0; v < 3; v = v + 1) $write(" %h", out_x[32*v +: 32]);
            for (v = 0; v < 3; v = v + 1) $write(" %h", out_y[32*v +: 32]);
            for (v = 0; v < 3; v = v + 1) $write(" %h", out_z[32*v +: 32]);
            for (v = 0; v < 3; v = v + 1)
                for (a = 0; a < 8; a = a + 1) $write(" %h", out_attr[256*v + 32*a +: 32]);
            $write("\n");
        end

    integer i, g, t, j, groups, hdr, base, ntri, first;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        $readmemh(image, mem);
        groups = mem[32'h0F000 >> 2];

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        for (g = 0; g < groups; g = g + 1) begin
            hdr = (32'h0F100 >> 2) + 16 * g;
            while (busy) @(posedge clk);
            ntri       = mem[hdr];
            first      = mem[hdr + 1];
            proj_cx    <= mem[hdr + 2];
            proj_cy    <= mem[hdr + 3];
            proj_focal <= mem[hdr + 4];
            near_z     <= mem[hdr + 5];
            cull_mode  <= mem[hdr + 6];
            n_attrs    <= mem[hdr + 7];
            draw_start <= 1'b1;
            @(posedge clk);
            draw_start <= 1'b0;
            $display("GROUP %0d", g);
            for (t = first; t < first + ntri; t = t + 1) begin
                base = (32'h40000 >> 2) + 64 * t;
                while (!in_ready) @(posedge clk);
                for (v = 0; v < 3; v = v + 1) begin
                    in_x[32*v +: 32] <= mem[base + v];
                    in_y[32*v +: 32] <= mem[base + 3 + v];
                    in_z[32*v +: 32] <= mem[base + 6 + v];
                    for (j = 0; j < 8; j = j + 1)
                        in_attr[256*v + 32*j +: 32] <= mem[base + 9 + 8*v + j];
                end
                in_valid <= 1'b1;
                @(posedge clk);
                in_valid <= 1'b0;
                @(posedge clk);
            end
        end
        while (busy) @(posedge clk);
        repeat (3) @(posedge clk);
        $display("STATS %0d %0d %0d %0d %0d", st_in, st_near, st_zero, st_back, st_out);
        $display("STATUS DONE");
        $finish;
    end
endmodule
