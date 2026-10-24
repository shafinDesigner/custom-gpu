// tb_gpu_top.v - testbench for the complete GPU (gpu_top.v)          Phase 8 + 9
//   vvp top_sim +image=scene.hex
// The memory image holds everything a driver would prepare: the command list at 0x0,
// shaders, constants, vertex and index buffers, textures. The GPU runs the command list
// on its own; at every SWAP_BUFFERS the testbench prints the finished frame.
// It also WATCHES THE VIDEO OUTPUT like a monitor: every time the picture on screen
// changes, it prints the new picture (one line per pixel, "V rrggbb").
// Display timing here: 160 x 90 visible (matching the test viewport) with blanking.

`timescale 1ns/1ps

module tb_gpu_top;
    localparam WORDS = 524288;                 // 2 MB
    localparam DW = 160, DH = 90;              // display size = viewport of the test scenes

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];

    reg          start = 1'b0;
    wire         done, error, swap_valid, mem_we;
    wire [31:0]  imem0_addr, imem1_addr, mem_addr, mem_wdata, swap_addr;
    wire [15:0]  frame_width, frame_height;
    wire [31:0]  st_commands, st_vertex_waves, st_pixel_waves, video_frames;
    wire         video_de, video_hsync, video_vsync, video_underflow;
    wire [7:0]   video_r, video_g, video_b;
    wire [31:0]  imem0_data = mem[imem0_addr[20:2]];
    wire [31:0]  imem1_data = mem[imem1_addr[20:2]];
    wire [31:0]  mem_rdata  = mem[mem_addr[20:2]];

    gpu_top #(.H_ACTIVE(DW), .H_FP(8), .H_SYNC(16), .H_BP(136),
              .V_ACTIVE(DH), .V_FP(2), .V_SYNC(2), .V_BP(6)) dut (
        .clk(clk), .rst(rst), .start(start), .cmd_addr(32'd0), .done(done), .error(error),
        .imem0_addr(imem0_addr), .imem0_data(imem0_data),
        .imem1_addr(imem1_addr), .imem1_data(imem1_data),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_we(mem_we), .mem_rdata(mem_rdata),
        .swap_valid(swap_valid), .swap_addr(swap_addr),
        .frame_width(frame_width), .frame_height(frame_height),
        .video_de(video_de), .video_hsync(video_hsync), .video_vsync(video_vsync),
        .video_r(video_r), .video_g(video_g), .video_b(video_b),
        .video_underflow(video_underflow), .video_frames(video_frames),
        .st_commands(st_commands), .st_vertex_waves(st_vertex_waves), .st_pixel_waves(st_pixel_waves)
    );

    always @(posedge clk)
        if (mem_we) mem[mem_addr[20:2]] <= mem_wdata;

    // ---- alarm: any access outside the 2 MB we have ----
    integer bad = 0;
    always @(posedge clk)
        if (mem_addr[31:21] != 0 && (mem_we || dut.g_disp || dut.g_rop || dut.g_ras || dut.g_side0 ||
                                     dut.g_side1 || dut.g_ia || dut.g_cp)) begin
            if (bad < 5) $display("BADADDR %h", mem_addr);
            bad = bad + 1;
        end

    // ---- print every finished frame ----
    integer frame_no = 0, p;
    always @(posedge clk)
        if (swap_valid) begin
            $display("FRAME %0d %0d %0d", frame_no, frame_width, frame_height);
            for (p = 0; p < frame_width * frame_height; p = p + 1)
                $display("P %h", mem[(swap_addr >> 2) + p]);
            frame_no = frame_no + 1;
        end

    // ---- watch the screen: print the picture whenever it changes ----
    reg [23:0] cur  [0:DW*DH-1];
    reg [23:0] prev [0:DW*DH-1];
    integer vis = 0, shown = 0, vsyncs = 0, q;
    reg     prev_vs = 1'b0, changed;
    always @(negedge clk) begin
        if (video_de && vis < DW*DH) begin
            cur[vis] = {video_r, video_g, video_b};
            vis = vis + 1;
        end
        if (video_vsync && !prev_vs) begin             // a display frame just ended
            vsyncs = vsyncs + 1;
            if (vis == DW*DH) begin
                changed = (shown == 0);
                for (q = 0; q < DW*DH; q = q + 1)
                    if (cur[q] !== prev[q]) changed = 1'b1;
                if (changed) begin
                    $display("SHOWN %0d", shown);
                    for (q = 0; q < DW*DH; q = q + 1) begin
                        $display("V %h", cur[q]);
                        prev[q] = cur[q];
                    end
                    shown = shown + 1;
                end
            end
            vis = 0;
        end
        prev_vs = video_vsync;
    end

    integer cycles;
    reg [8*256-1:0] image;

    initial begin
        for (cycles = 0; cycles < WORDS; cycles = cycles + 1) mem[cycles] = 32'd0;
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
        while (!done && !error && cycles < 20000000) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        // keep the monitor running for 2 more display frames, so the last picture is seen
        begin : more_frames
            integer start_vs;
            start_vs = vsyncs;
            while (done && !error && vsyncs < start_vs + 2 && cycles < 20000000) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
        end
        repeat (3) @(posedge clk);

        $display("VIDEO %0d %0d %0d", shown, video_frames, video_underflow);
        $display("FENCE %h", mem[32'h00F000 >> 2]);
        $display("STATS %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d", cycles, st_commands,
                 st_vertex_waves, st_pixel_waves, dut.ras.stat_quads, dut.ras.stat_early_dropped,
                 dut.rop_u.stat_written, dut.rop_u.stat_depth_failed, dut.prim.st_out, dut.prim.st_back);
        $display("BADCOUNT %0d", bad);
        if (error)                  $display("STATUS ERROR");
        else if (!done)             $display("STATUS TIMEOUT");
        else                        $display("STATUS DONE");
        $finish;
    end
endmodule
