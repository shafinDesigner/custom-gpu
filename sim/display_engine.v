// display_engine.v - display engine (spec section 8.11)                  Phase 9, stage 5
//
// Generates video timing and streams the front buffer out, one pixel per clock.
// Default timing = 720p60 (CEA-861): 1280 x 720 visible, 1650 x 750 total, 74.25 MHz.
//
//   - Two line buffers: while line y is shown, line y + 1 is read from memory.
//     Line 0 is read during the last blanking line.
//   - A finished frame (swap_valid from the GPU) becomes the front buffer at the start
//     of the next vertical blank, so the picture never changes mid-screen.
//   - If a line is not fully read in time: that pixel shows magenta and 'underflow' is set.
//
// Simplification for v1: one clock for everything (pixel clock = core clock). A real chip
// uses an asynchronous FIFO between the memory clock and the 74.25 MHz pixel clock.

`timescale 1ns/1ps

module display_engine #(
    parameter H_ACTIVE = 1280, H_FP = 110, H_SYNC = 40, H_BP = 220,
    parameter V_ACTIVE = 720,  V_FP = 5,   V_SYNC = 5,  V_BP = 20
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,            // 1 = start / keep running
    input  wire [31:0] init_front,        // front buffer when enabled

    input  wire        swap_valid,        // the GPU finished a frame
    input  wire [31:0] swap_addr,
    output reg         flipped,           // 1 for one cycle: a new frame is now the front buffer
    output reg  [31:0] front,

    // memory: reading the front buffer
    output wire        mem_req,
    output wire [31:0] mem_addr,
    input  wire        mem_gnt,
    input  wire [31:0] mem_rdata,

    // video out
    output wire        de,                // visible pixel
    output wire        hsync,
    output wire        vsync,
    output wire [7:0]  r,
    output wire [7:0]  g,
    output wire [7:0]  b,
    output reg         underflow,         // sticky: a line was not ready in time
    output reg  [31:0] frame_count
);

    localparam H_TOTAL = H_ACTIVE + H_FP + H_SYNC + H_BP;
    localparam V_TOTAL = V_ACTIVE + V_FP + V_SYNC + V_BP;

    reg        running;
    reg [15:0] h, v;                     // current pixel position (incl. blanking)

    // ---------------- line buffers ----------------
    reg [31:0] lbuf [0:2*H_ACTIVE-1];    // buffer k holds lines with (line & 1) == k
    reg [15:0] ready_line [0:1];         // which line each buffer holds completely
    reg        fetching;
    reg [15:0] f_line, f_x;
    reg        f_buf;

    // ---------------- buffer flip ----------------
    reg        pending;
    reg [31:0] pending_addr;

    // ---------------- video timing ----------------
    wire visible   = running && (h < H_ACTIVE) && (v < V_ACTIVE);
    wire line_ok   = (ready_line[v[0]] == v);
    wire [31:0] px = lbuf[v[0] * H_ACTIVE + h];
    assign de    = visible;
    assign hsync = running && (h >= H_ACTIVE + H_FP) && (h < H_ACTIVE + H_FP + H_SYNC);
    assign vsync = running && (v >= V_ACTIVE + V_FP) && (v < V_ACTIVE + V_FP + V_SYNC);
    assign r     = !visible ? 8'd0 : line_ok ? px[7:0]   : 8'hFF;      // magenta if not ready
    assign g     = !visible ? 8'd0 : line_ok ? px[15:8]  : 8'h00;
    assign b     = !visible ? 8'd0 : line_ok ? px[23:16] : 8'hFF;

    // ---------------- memory ----------------
    assign mem_req  = fetching;
    assign mem_addr = front + ({16'd0, f_line} * H_ACTIVE + {16'd0, f_x}) * 4;

    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0;  h <= 16'd0;  v <= 16'd0;  fetching <= 1'b0;
            f_line <= 16'd0;  f_x <= 16'd0;  f_buf <= 1'b0;  pending <= 1'b0;  pending_addr <= 32'd0;
            front <= 32'd0;  flipped <= 1'b0;  underflow <= 1'b0;  frame_count <= 32'd0;
            ready_line[0] <= 16'hFFFF;  ready_line[1] <= 16'hFFFF;
        end else if (!running) begin
            if (enable) begin                         // start in vertical blank
                running <= 1'b1;  h <= 16'd0;  v <= V_ACTIVE;  front <= init_front;
            end
        end else begin
            flipped <= 1'b0;

            // ---- position ----
            if (h == H_TOTAL - 1) begin
                h <= 16'd0;
                if (v == V_TOTAL - 1) begin
                    v <= 16'd0;
                    frame_count <= frame_count + 32'd1;
                end else v <= v + 16'd1;
            end else h <= h + 16'd1;

            // ---- underflow: a visible pixel whose line is not ready ----
            if (visible && !line_ok) underflow <= 1'b1;

            // ---- flip at the start of vertical blank ----
            if (h == 0 && v == V_ACTIVE && pending) begin
                front   <= pending_addr;
                pending <= 1'b0;
                flipped <= 1'b1;
            end
            if (swap_valid) begin                      // remember the newest finished frame
                pending      <= 1'b1;
                pending_addr <= swap_addr;
            end

            // ---- fetching the next line ----
            if (fetching && mem_gnt) begin
                lbuf[f_buf * H_ACTIVE + f_x] <= mem_rdata;
                if (f_x == H_ACTIVE - 1) begin
                    fetching <= 1'b0;
                    ready_line[f_buf] <= f_line;
                end else f_x <= f_x + 16'd1;
            end
            if (h == 0 && (v < V_ACTIVE - 1 || v == V_TOTAL - 1)) begin
                // start reading the line shown next (line 0 during the last blank line)
                fetching <= 1'b1;
                f_line   <= (v == V_TOTAL - 1) ? 16'd0 : v + 16'd1;
                f_buf    <= (v == V_TOTAL - 1) ? 1'b0 : ~v[0];
                f_x      <= 16'd0;
                ready_line[(v == V_TOTAL - 1) ? 1'b0 : ~v[0]] <= 16'hFFFF;
            end
        end
    end

endmodule
