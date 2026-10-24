// gpu_core.v - two compute units working together                      Phase 3 + 6
//
//   draw_start --> WORK DISTRIBUTOR --> CU0 (+ texture unit 0) --> pixel port 0
//                  (one 2x2 quad per wave) CU1 (+ texture unit 1) --> pixel port 1
//
//   side 0 = CU0 or texture unit 0, side 1 = CU1 or texture unit 1
//   (a CU waiting for its texture unit never asks for memory itself)
//   side 0 / side 1 --> MEMORY ARBITER (round-robin) --> one data memory port
//
// A "draw" shades every 2x2 quad of the screen once, in raster order.
// Each wave starts with:  v0 = pixel x, v1 = pixel y  (lane 0 TL, 1 TR, 2 BL, 3 BR)
//                         s1 = const_addr,  s2 = wave number
// Spec sections 8.4 (work distributor) and 8.5 (compute units).

`timescale 1ns/1ps

module gpu_core (
    input  wire         clk,
    input  wire         rst,

    // draw control
    input  wire         draw_start,      // 1 for one cycle
    input  wire [31:0]  ps_addr,         // pixel shader program
    input  wire [31:0]  const_addr,      // constant buffer for this draw
    input  wire [15:0]  width,           // screen size in pixels (even numbers)
    input  wire [15:0]  height,
    input  wire [127:0] tex_addr,        // TEX0-3_ADDR: slot 3 | 2 | 1 | 0
    input  wire [127:0] tex_info,        // TEX0-3_INFO
    output reg          draw_done,
    output reg          draw_error,

    // instruction ports: one per CU (like the shared 2-port instruction cache)
    output wire [31:0]  imem0_addr,
    input  wire [31:0]  imem0_data,
    output wire [31:0]  imem1_addr,
    input  wire [31:0]  imem1_data,

    // ONE shared data memory port
    output wire [31:0]  mem_addr,
    output wire [31:0]  mem_wdata,
    output wire         mem_we,
    input  wire [31:0]  mem_rdata,

    // pixel ports: a CU exported COLOR for its quad
    output wire         pix0_valid,
    output reg  [15:0]  pix0_x,          // top-left pixel of the quad
    output reg  [15:0]  pix0_y,
    output wire [3:0]   pix0_mask,
    output wire [127:0] pix0_data,
    output wire         pix1_valid,
    output reg  [15:0]  pix1_x,
    output reg  [15:0]  pix1_y,
    output wire [3:0]   pix1_mask,
    output wire [127:0] pix1_data,

    // statistics
    output reg  [31:0]  waves_cu0,
    output reg  [31:0]  waves_cu1,
    output reg  [31:0]  mem_conflicts    // cycles where both CUs wanted memory
);

    // ------------------------------------------------------------------
    // The two compute units
    // ------------------------------------------------------------------
    reg          start0, start1;
    reg  [127:0] init0_v0, init0_v1, init1_v0, init1_v1;
    reg  [31:0]  init0_s2, init1_s2;

    wire [31:0]  d0_addr, d0_wdata, d1_addr, d1_wdata;
    wire         d0_we, d1_we, d0_req, d1_req, gnt0, gnt1;
    wire         ex0_valid, ex1_valid;
    wire [7:0]   ex0_target, ex1_target;
    wire         busy0, busy1, done0, done1, err0, err1;
    wire [31:0]  count0, count1;
    wire         cu0_gnt, cu1_gnt;
    // texture port wires
    wire         ts0, ts1, td0, td1, t0_req, t1_req, t0_gnt, t1_gnt, tbusy0, tbusy1;
    wire [1:0]   tslot0, tslot1;
    wire [127:0] tu0, tv0, tu1, tv1, tres0, tres1;
    wire [31:0]  t0_addr, t1_addr;

    cu cu0 (
        .clk(clk), .rst(rst), .start(start0), .start_pc(ps_addr),
        .cu_id(4'd0), .wave_type(1'b1), .frame(32'd0),
        .init_v({1152'd0, init0_v1, init0_v0}), .init_mask(4'hF), .init_s1(const_addr), .init_s2(init0_s2),
        .imem_addr(imem0_addr), .imem_data(imem0_data),
        .dmem_addr(d0_addr), .dmem_wdata(d0_wdata), .dmem_we(d0_we), .dmem_rdata(mem_rdata),
        .dmem_req(d0_req), .dmem_gnt(cu0_gnt),
        .export_valid(ex0_valid), .export_target(ex0_target),
        .export_mask(pix0_mask), .export_data(pix0_data),
        .tex_start(ts0), .tex_slot(tslot0), .tex_u(tu0), .tex_v(tv0),
        .tex_done(td0), .tex_result(tres0),
        .busy(busy0), .done(done0), .error(err0), .instr_count(count0)
    );

    cu cu1 (
        .clk(clk), .rst(rst), .start(start1), .start_pc(ps_addr),
        .cu_id(4'd1), .wave_type(1'b1), .frame(32'd0),
        .init_v({1152'd0, init1_v1, init1_v0}), .init_mask(4'hF), .init_s1(const_addr), .init_s2(init1_s2),
        .imem_addr(imem1_addr), .imem_data(imem1_data),
        .dmem_addr(d1_addr), .dmem_wdata(d1_wdata), .dmem_we(d1_we), .dmem_rdata(mem_rdata),
        .dmem_req(d1_req), .dmem_gnt(cu1_gnt),
        .export_valid(ex1_valid), .export_target(ex1_target),
        .export_mask(pix1_mask), .export_data(pix1_data),
        .tex_start(ts1), .tex_slot(tslot1), .tex_u(tu1), .tex_v(tv1),
        .tex_done(td1), .tex_result(tres1),
        .busy(busy1), .done(done1), .error(err1), .instr_count(count1)
    );

    // ------------------------------------------------------------------
    // One texture unit per CU (spec 8.6). Pixel waves only in this core.
    // ------------------------------------------------------------------
    tex_unit tex0 (
        .clk(clk), .rst(rst), .start(ts0), .slot(tslot0), .wave_type(1'b1),
        .u(tu0), .v(tv0), .tex_addr(tex_addr), .tex_info(tex_info),
        .mem_req(t0_req), .mem_addr(t0_addr), .mem_gnt(t0_gnt), .mem_rdata(mem_rdata),
        .done(td0), .result(tres0), .busy(tbusy0)
    );

    tex_unit tex1 (
        .clk(clk), .rst(rst), .start(ts1), .slot(tslot1), .wave_type(1'b1),
        .u(tu1), .v(tv1), .tex_addr(tex_addr), .tex_info(tex_info),
        .mem_req(t1_req), .mem_addr(t1_addr), .mem_gnt(t1_gnt), .mem_rdata(mem_rdata),
        .done(td1), .result(tres1), .busy(tbusy1)
    );

    // Only EXPORT COLOR (target 16) goes to the pixel ports
    assign pix0_valid = ex0_valid && (ex0_target == 8'd16);
    assign pix1_valid = ex1_valid && (ex1_target == 8'd16);

    // ------------------------------------------------------------------
    // Memory arbiter: round-robin between side 0 and side 1. If both ask,
    // the side that did NOT go last time wins; the other waits one cycle.
    // ------------------------------------------------------------------
    reg last_was_1;
    wire        side0_req  = d0_req | t0_req;
    wire        side1_req  = d1_req | t1_req;
    wire [31:0] side0_addr = t0_req ? t0_addr : d0_addr;
    wire [31:0] side1_addr = t1_req ? t1_addr : d1_addr;
    assign gnt0    = side0_req && (!side1_req || last_was_1);
    assign gnt1    = side1_req && (!side0_req || !last_was_1);
    assign cu0_gnt = gnt0 && d0_req;
    assign cu1_gnt = gnt1 && d1_req;
    assign t0_gnt  = gnt0 && t0_req;
    assign t1_gnt  = gnt1 && t1_req;

    assign mem_addr  = gnt1 ? side1_addr : side0_addr;
    assign mem_wdata = gnt1 ? d1_wdata   : d0_wdata;
    assign mem_we    = d0_we | d1_we;               // a CU only writes when granted

    always @(posedge clk) begin
        if (rst) begin
            last_was_1 <= 1'b1;
            mem_conflicts <= 32'd0;
        end else begin
            if (gnt0)      last_was_1 <= 1'b0;
            else if (gnt1) last_was_1 <= 1'b1;
            if (side0_req && side1_req) mem_conflicts <= mem_conflicts + 32'd1;
        end
    end

    // ------------------------------------------------------------------
    // Work distributor
    //   Each CU: FREE -> STARTING (start pulse) -> RUNNING (until busy = 0) -> FREE
    // ------------------------------------------------------------------
    localparam FREE = 2'd0, STARTING = 2'd1, RUNNING = 2'd2;

    reg [1:0]  cu_state0, cu_state1;
    reg [15:0] qx, qy;                 // next quad to give out
    reg        issuing;                // quads still left in this draw
    reg        drawing;
    reg        prefer1;                // round-robin between the CUs
    reg [31:0] wave_number;

    // the 4 pixel positions of quad (x, y): lane 3 | lane 2 | lane 1 | lane 0
    function [127:0] quad_x;
        input [15:0] x;
        quad_x = {16'd0, x + 16'd1, 16'd0, x, 16'd0, x + 16'd1, 16'd0, x};
    endfunction
    function [127:0] quad_y;
        input [15:0] y;
        quad_y = {16'd0, y + 16'd1, 16'd0, y + 16'd1, 16'd0, y, 16'd0, y};
    endfunction

    wire free0 = (cu_state0 == FREE);
    wire free1 = (cu_state1 == FREE);
    wire give0 = issuing && free0 && (!free1 || !prefer1);
    wire give1 = issuing && free1 && !give0;

    wire        last_quad_in_row = (qx + 16'd2 >= width);
    wire        last_quad        = last_quad_in_row && (qy + 16'd2 >= height);

    always @(posedge clk) begin
        if (rst) begin
            cu_state0 <= FREE;  cu_state1 <= FREE;  start0 <= 1'b0;  start1 <= 1'b0;
            qx <= 16'd0;  qy <= 16'd0;  issuing <= 1'b0;  drawing <= 1'b0;  prefer1 <= 1'b0;
            wave_number <= 32'd0;  draw_done <= 1'b0;  draw_error <= 1'b0;
            waves_cu0 <= 32'd0;  waves_cu1 <= 32'd0;
            init0_v0 <= 128'd0;  init0_v1 <= 128'd0;  init1_v0 <= 128'd0;  init1_v1 <= 128'd0;
            init0_s2 <= 32'd0;   init1_s2 <= 32'd0;
            pix0_x <= 16'd0;  pix0_y <= 16'd0;  pix1_x <= 16'd0;  pix1_y <= 16'd0;
        end else begin
            start0 <= 1'b0;
            start1 <= 1'b0;

            // ---- a new draw ----
            if (draw_start) begin
                qx <= 16'd0;  qy <= 16'd0;  issuing <= 1'b1;  drawing <= 1'b1;
                draw_done <= 1'b0;  draw_error <= 1'b0;  wave_number <= 32'd0;
            end

            // ---- give the next quad to a free CU ----
            if (!draw_start && (give0 || give1)) begin
                if (give0) begin
                    start0 <= 1'b1;  cu_state0 <= STARTING;
                    init0_v0 <= quad_x(qx);  init0_v1 <= quad_y(qy);  init0_s2 <= wave_number;
                    pix0_x <= qx;  pix0_y <= qy;
                    waves_cu0 <= waves_cu0 + 32'd1;
                    prefer1 <= 1'b1;
                end else begin
                    start1 <= 1'b1;  cu_state1 <= STARTING;
                    init1_v0 <= quad_x(qx);  init1_v1 <= quad_y(qy);  init1_s2 <= wave_number;
                    pix1_x <= qx;  pix1_y <= qy;
                    waves_cu1 <= waves_cu1 + 32'd1;
                    prefer1 <= 1'b0;
                end
                wave_number <= wave_number + 32'd1;
                if (last_quad) begin
                    issuing <= 1'b0;
                end else if (last_quad_in_row) begin
                    qx <= 16'd0;  qy <= qy + 16'd2;
                end else begin
                    qx <= qx + 16'd2;
                end
            end

            // ---- follow each CU: STARTING -> RUNNING -> FREE ----
            if (cu_state0 == STARTING) cu_state0 <= RUNNING;
            else if (cu_state0 == RUNNING && !busy0) cu_state0 <= FREE;
            if (cu_state1 == STARTING) cu_state1 <= RUNNING;
            else if (cu_state1 == RUNNING && !busy1) cu_state1 <= FREE;

            // ---- errors and the end of the draw ----
            if (err0 || err1) begin
                draw_error <= 1'b1;  issuing <= 1'b0;  drawing <= 1'b0;
            end else if (drawing && !draw_start && !issuing && free0 && free1 && !start0 && !start1) begin
                draw_done <= 1'b1;  drawing <= 1'b0;
            end
        end
    end

endmodule
