// tb_display.v - testbench for display_engine.v                          Phase 9, stage 5
//   vvp disp_sim +image=disp.hex              -> prints the video signals of every cycle
//   vvp disp_sim +image=disp.hex +summary     -> measures one whole frame (for 720p)
// Memory image (made by run_display_tests.py):
//   0x0F000 cycles to run, number of swaps, initial front buffer, memory deny % (0-99)
//   0x0F100 + 8*k  swap k: cycle number, frame address
// Timing parameters can be changed with  iverilog -P tb_display.H_ACTIVE=32 ...

`timescale 1ns/1ps

module tb_display #(
    parameter H_ACTIVE = 1280, H_FP = 110, H_SYNC = 40, H_BP = 220,
    parameter V_ACTIVE = 720,  V_FP = 5,   V_SYNC = 5,  V_BP = 20
);
    localparam WORDS = 524288;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg [31:0] mem [0:WORDS-1];

    reg         enable = 1'b0;
    reg  [31:0] init_front = 32'd0;
    reg         swap_valid = 1'b0;
    reg  [31:0] swap_addr = 32'd0;
    wire        flipped, mem_req, de, hsync, vsync, underflow;
    wire [31:0] front, mem_addr, frame_count;
    wire [7:0]  r, g, b;
    reg         lucky = 1'b1;
    wire        mem_gnt = mem_req && lucky;
    wire [31:0] mem_rdata = mem[mem_addr[20:2]];

    display_engine #(.H_ACTIVE(H_ACTIVE), .H_FP(H_FP), .H_SYNC(H_SYNC), .H_BP(H_BP),
                     .V_ACTIVE(V_ACTIVE), .V_FP(V_FP), .V_SYNC(V_SYNC), .V_BP(V_BP)) dut (
        .clk(clk), .rst(rst), .enable(enable), .init_front(init_front),
        .swap_valid(swap_valid), .swap_addr(swap_addr), .flipped(flipped), .front(front),
        .mem_req(mem_req), .mem_addr(mem_addr), .mem_gnt(mem_gnt), .mem_rdata(mem_rdata),
        .de(de), .hsync(hsync), .vsync(vsync), .r(r), .g(g), .b(b),
        .underflow(underflow), .frame_count(frame_count)
    );

    integer seed = 41, deny = 0;
    always @(posedge clk) lucky <= ({$random(seed)} % 100) >= deny;

    integer i, n_cycles, n_swaps, next_swap, cycle, flips;
    integer f_cycles, f_de, f_hs, f_vs, f_hrise, vs_rises;
    reg     summary, prev_hs, prev_vs, measuring;
    reg [8*256-1:0] image;

    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 32'd0;
        if (!$value$plusargs("image=%s", image)) begin
            $display("STATUS NO_IMAGE");
            $finish;
        end
        summary = $test$plusargs("summary");
        $readmemh(image, mem);
        n_cycles   = mem[32'h0F000 >> 2];
        n_swaps    = mem[(32'h0F000 >> 2) + 1];
        init_front = mem[(32'h0F000 >> 2) + 2];
        deny       = mem[(32'h0F000 >> 2) + 3];

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
        enable <= 1'b1;
        @(posedge clk);                                  // the engine starts here

        cycle = 0;  next_swap = 0;  flips = 0;
        f_cycles = 0;  f_de = 0;  f_hs = 0;  f_vs = 0;  f_hrise = 0;  vs_rises = 0;
        prev_hs = 1'b0;  prev_vs = 1'b0;  measuring = 1'b0;
        while (cycle < n_cycles) begin
            @(negedge clk);                              // signals are stable mid-cycle
            if (!summary)
                $display("S %b %b %b %h%h%h", de, hsync, vsync, r, g, b);
            else begin                                   // measure exactly one frame
                if (vsync && !prev_vs) begin
                    vs_rises = vs_rises + 1;
                    measuring = (vs_rises == 1);
                end
                if (measuring) begin
                    f_cycles = f_cycles + 1;
                    if (de) f_de = f_de + 1;
                    if (hsync) f_hs = f_hs + 1;
                    if (vsync) f_vs = f_vs + 1;
                    if (hsync && !prev_hs) f_hrise = f_hrise + 1;
                end
                prev_hs = hsync;  prev_vs = vsync;
                if (vs_rises == 2) cycle = n_cycles;     // done
            end
            // a swap request during this cycle (seen by the engine at the next clock edge)
            if (next_swap < n_swaps && mem[(32'h0F100 >> 2) + 2 * next_swap] == cycle) begin
                swap_valid <= 1'b1;
                swap_addr  <= mem[(32'h0F100 >> 2) + 2 * next_swap + 1];
                next_swap   = next_swap + 1;
            end else
                swap_valid <= 1'b0;
            if (flipped) flips = flips + 1;
            cycle = cycle + 1;
        end
        if (summary)
            $display("SUMMARY %0d %0d %0d %0d %0d", f_cycles, f_de, f_hs, f_vs, f_hrise);
        $display("END %0d %0d %0d", underflow, flips, frame_count);
        $finish;
    end
endmodule
