// gpu_top.v - the complete GPU: command list in, video out          Phase 8 + 9
// Must produce exactly the same frames as GPU.start() in gpu_model.py.
//
//   COMMAND PROCESSOR --- state registers, CLEAR / DRAW / SWAP / FENCE / JUMP / END
//         |
//   DRAW: INPUT ASSEMBLER -> vertex waves on the CUs -> VERTEX OUTPUT BUFFER
//         -> PRIMITIVE UNIT -> RASTERIZER -> pixel waves on the CUs (+ texture units)
//         -> RENDER OUTPUT UNIT -> memory
//
//   SWAP -> DISPLAY ENGINE: the finished frame is shown from the next vertical blank
//
// Ordering rule (keeps results identical to the golden model): a triangle enters the
// rasterizer only when every pixel of the previous triangle has been written.
// One memory port is shared by 7 users through the MEMORY ARBITER (display first).
// The first SWAP turns the display on. Every later SWAP waits until the display has
// flipped, so the GPU never draws into a buffer that is still on screen (no tearing).

`timescale 1ns/1ps

module gpu_top #(
    parameter MAXV = 256,                    // vertices per draw (vertex output buffer size)
    // display timing (default 720p60); H_ACTIVE x V_ACTIVE must equal the viewport
    parameter H_ACTIVE = 1280, H_FP = 110, H_SYNC = 40, H_BP = 220,
    parameter V_ACTIVE = 720,  V_FP = 5,   V_SYNC = 5,  V_BP = 20
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,               // 1 for one cycle: run the command list
    input  wire [31:0]  cmd_addr,
    output reg          done,                // END reached
    output reg          error,

    // instruction ports (one per CU)
    output wire [31:0]  imem0_addr,
    input  wire [31:0]  imem0_data,
    output wire [31:0]  imem1_addr,
    input  wire [31:0]  imem1_data,

    // the shared memory port
    output wire [31:0]  mem_addr,
    output wire [31:0]  mem_wdata,
    output wire         mem_we,
    input  wire [31:0]  mem_rdata,

    // a frame is finished (SWAP_BUFFERS): its address, for the display / testbench
    output reg          swap_valid,
    output reg  [31:0]  swap_addr,
    output wire [15:0]  frame_width,
    output wire [15:0]  frame_height,

    // video out
    output wire         video_de,
    output wire         video_hsync,
    output wire         video_vsync,
    output wire [7:0]   video_r,
    output wire [7:0]   video_g,
    output wire [7:0]   video_b,
    output wire         video_underflow,
    output wire [31:0]  video_frames,

    // statistics
    output reg  [31:0]  st_commands,
    output reg  [31:0]  st_vertex_waves,
    output reg  [31:0]  st_pixel_waves
);

    // ==================================================================
    // 1. STATE REGISTERS (spec section 7)
    // ==================================================================
    reg [31:0] st [0:63];
    wire [31:0] VS_ADDR = st[0],  PS_ADDR = st[1],  CONST_ADDR = st[2], VB_ADDR = st[3];
    wire [31:0] VB_STRIDE = st[4], VB_ATTRS = st[5], IB_ADDR = st[6],  VS_OUT_ATTRS = st[7];
    wire [31:0] FB_A = st[16], FB_B = st[17], DEPTH_ADDR = st[18], STENCIL_ADDR = st[19];
    wire [15:0] VW = st[20][15:0], VH = st[21][15:0];
    wire [31:0] FOCAL = st[22], CX = st[23], CY = st[24], NEAR_Z = st[25];
    wire [1:0]  CULL = st[32][1:0], DEPTH_MODE = st[33][1:0], BLEND = st[35][1:0];
    wire [3:0]  CMASK = st[36][3:0];
    wire [127:0] TEX_ADDR = {st[51], st[50], st[49], st[48]};
    wire [127:0] TEX_INFO = {st[55], st[54], st[53], st[52]};

    reg         back_sel;                       // 0: drawing into A, 1: into B
    reg  [31:0] frames_done;
    wire [31:0] back_addr = back_sel ? FB_B : FB_A;
    assign frame_width  = VW;
    assign frame_height = VH;

    // ==================================================================
    // 2. MEMORY ARBITER: ROP > rasterizer > CU sides (round-robin) > IA > CP
    // ==================================================================
    wire        rop_req, ras_req, ia_req, cp_req;
    wire [31:0] rop_addr, ras_addr, ia_addr, cp_addr, rop_wdata, cp_wdata;
    wire        rop_we, cp_we_raw;
    wire        d0_req, d1_req, t0_req, t1_req;
    wire [31:0] d0_addr, d1_addr, t0_addr, t1_addr, d0_wdata, d1_wdata;
    wire        d0_we, d1_we;
    reg         last_was_1;

    wire        disp_req;
    wire [31:0] disp_addr;
    wire side0_req = d0_req | t0_req;
    wire side1_req = d1_req | t1_req;
    wire g_disp  = disp_req;                    // the screen can never wait
    wire g_rop   = rop_req & ~g_disp;
    wire g_ras   = ras_req & ~g_rop & ~g_disp;
    wire free_cu = ~g_disp & ~g_rop & ~g_ras;
    wire g_side0 = free_cu & side0_req & (~side1_req | last_was_1);
    wire g_side1 = free_cu & side1_req & (~side0_req | ~last_was_1);
    wire g_ia    = free_cu & ~side0_req & ~side1_req & ia_req;
    wire g_cp    = free_cu & ~side0_req & ~side1_req & ~ia_req & cp_req;
    wire cu0_gnt = g_side0 & d0_req,  tx0_gnt = g_side0 & t0_req;
    wire cu1_gnt = g_side1 & d1_req,  tx1_gnt = g_side1 & t1_req;

    assign mem_addr  = g_disp ? disp_addr : g_rop ? rop_addr : g_ras ? ras_addr :
                       g_side0 ? (t0_req ? t0_addr : d0_addr) :
                       g_side1 ? (t1_req ? t1_addr : d1_addr) :
                       g_ia ? ia_addr : cp_addr;
    assign mem_wdata = g_rop ? rop_wdata : g_side0 ? d0_wdata : g_side1 ? d1_wdata : cp_wdata;
    assign mem_we    = rop_we | d0_we | d1_we | (g_cp & cp_we_raw);

    always @(posedge clk)
        if (rst) last_was_1 <= 1'b1;
        else if (g_side0) last_was_1 <= 1'b0;
        else if (g_side1) last_was_1 <= 1'b1;

    // ==================================================================
    // 3. COMMAND PROCESSOR (spec 8.2)
    // ==================================================================
    localparam C_IDLE = 4'd0, C_HDR = 4'd1, C_PAY = 4'd2, C_EXEC = 4'd3, C_ROP1 = 4'd4,
               C_ROP2 = 4'd5, C_DRAW = 4'd6, C_FENCE = 4'd7, C_FLIP = 4'd8, C_VBLANK = 4'd9;
    reg [3:0]  cstate;
    reg [31:0] cpc, hdr;
    reg [31:0] pay [0:3];
    reg [7:0]  pi;
    wire [7:0]  h_code  = hdr[31:24];
    wire [7:0]  h_words = hdr[23:16];
    wire [15:0] h_param = hdr[15:0];

    reg        clear_go, draw_go;
    reg        cmd_error;                       // unknown command code
    reg        disp_on;                         // display running (after the first SWAP)
    reg [31:0] disp_first;                      // first frame shown
    reg        disp_swap;                       // a SWAP for the running display
    reg [31:0] vbl_frame;                       // WAIT_VBLANK: display frame count at start
    wire       disp_flipped;
    wire [31:0] disp_frame_count;
    wire       rop_busy, draw_busy;

    assign cp_req    = (cstate == C_HDR) || (cstate == C_PAY) || (cstate == C_FENCE);
    assign cp_addr   = (cstate == C_FENCE) ? pay[0] : cpc;
    assign cp_we_raw = (cstate == C_FENCE);
    assign cp_wdata  = pay[1];

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            cstate <= C_IDLE;  done <= 1'b0;  cpc <= 32'd0;  hdr <= 32'd0;  pi <= 8'd0;
            clear_go <= 1'b0;  draw_go <= 1'b0;  swap_valid <= 1'b0;  swap_addr <= 32'd0;
            back_sel <= 1'b0;  frames_done <= 32'd0;  st_commands <= 32'd0;  cmd_error <= 1'b0;
            disp_on <= 1'b0;  disp_first <= 32'd0;  disp_swap <= 1'b0;  vbl_frame <= 32'd0;
            for (i = 0; i < 4; i = i + 1) pay[i] <= 32'd0;
            for (i = 0; i < 64; i = i + 1) st[i] <= 32'd0;
            // defaults (spec section 7 / DEFAULT_STATE in gpu_model.py)
            st[4]  <= 32'd32;          st[5]  <= 32'd3;
            st[16] <= 32'h0050_0000;   st[17] <= 32'h0090_0000;
            st[18] <= 32'h00D0_0000;   st[19] <= 32'h00ED_0000;
            st[20] <= 32'd1280;        st[21] <= 32'd720;
            st[23] <= 32'd640 << 16;   st[24] <= 32'd360 << 16;   st[25] <= 32'h0001_0000;
            st[32] <= 32'd1;           st[33] <= 32'd1;           st[36] <= 32'hF;
        end else begin
            clear_go   <= 1'b0;
            draw_go    <= 1'b0;
            swap_valid <= 1'b0;
            disp_swap  <= 1'b0;
            case (cstate)
            C_IDLE: if (start) begin
                cpc <= cmd_addr;  done <= 1'b0;  cstate <= C_HDR;
            end
            C_HDR: if (g_cp) begin                       // read the header
                hdr <= mem_rdata;  cpc <= cpc + 32'd4;  pi <= 8'd0;
                st_commands <= st_commands + 32'd1;
                cstate <= (mem_rdata[23:16] == 8'd0) ? C_EXEC : C_PAY;
            end
            C_PAY: if (g_cp) begin                       // read the payload words
                if (pi < 8'd4) pay[pi[1:0]] <= mem_rdata;
                cpc <= cpc + 32'd4;  pi <= pi + 8'd1;
                if (pi + 8'd1 == h_words) cstate <= C_EXEC;
            end
            C_EXEC: begin
                cstate <= C_HDR;
                case (h_code)
                8'h00: ;                                                   // NOP
                8'h01: st[h_param[5:0]] <= pay[0];                         // SET_STATE
                8'h02: begin clear_go <= 1'b1;  cstate <= C_ROP1; end      // CLEAR
                8'h03: begin draw_go  <= 1'b1;  cstate <= C_DRAW; end      // DRAW_TRIANGLES
                8'h04: begin                                               // SWAP_BUFFERS
                    swap_valid  <= 1'b1;  swap_addr <= back_addr;
                    back_sel    <= ~back_sel;  frames_done <= frames_done + 32'd1;
                    if (!disp_on) begin                  // first frame: turn the display on
                        disp_on <= 1'b1;  disp_first <= back_addr;
                    end else begin                       // later: wait until it is on screen
                        disp_swap <= 1'b1;  cstate <= C_FLIP;
                    end
                end
                8'h05: cstate <= C_FENCE;                                  // FENCE
                8'h06: if (disp_on) begin                                  // WAIT_VBLANK
                    vbl_frame <= disp_frame_count;  cstate <= C_VBLANK;
                end
                8'h07: cpc <= pay[0];                                      // JUMP
                8'hFF: begin done <= 1'b1;  cstate <= C_IDLE; end          // END
                default: begin cmd_error <= 1'b1;  cstate <= C_IDLE; end
                endcase
            end
            C_ROP1: cstate <= C_ROP2;                    // the ROP starts the clear now
            C_ROP2: if (!rop_busy) cstate <= C_HDR;
            C_DRAW: if (!draw_go && !draw_busy) cstate <= C_HDR;
            C_FENCE: if (g_cp) cstate <= C_HDR;          // the word was written
            C_FLIP:   if (disp_flipped) cstate <= C_HDR;  // new frame is on screen
            C_VBLANK: if (disp_frame_count != vbl_frame) cstate <= C_HDR;
            default: cstate <= C_IDLE;
            endcase
        end
    end

    // ==================================================================
    // 4. COMPUTE UNITS, TEXTURE UNITS and the WORK DISTRIBUTOR
    // ==================================================================
    localparam FREE = 2'd0, STARTING = 2'd1, RUNNING = 2'd2;

    reg  [1:0]    cs0, cs1;                  // CU slot states
    reg           start0, start1;
    reg  [1407:0] iv0, iv1;                  // starting v0-v10
    reg  [3:0]    im0, im1;                  // starting mask
    reg  [31:0]   is2_0, is2_1;              // starting s2
    reg           wt0, wt1;                  // wave type: 0 vertex, 1 pixel
    reg  [31:0]   vbase0, vbase1;            // vertex waves: first vertex number
    reg  [15:0]   px0, py0, px1, py1;        // pixel waves: quad position
    reg  [127:0]  pc0, pc1;                  // pixel waves: closeness per lane

    wire          busy0, busy1, done0, done1, err0, err1;
    wire [31:0]   cnt0, cnt1;
    wire          ex0, ex1;
    wire [7:0]    ext0, ext1;
    wire [3:0]    exm0, exm1;
    wire [127:0]  exd0, exd1;
    wire          ts0, ts1, td0, td1, tb0, tb1;
    wire [1:0]    tsl0, tsl1;
    wire [127:0]  tu0, tv0, tu1, tv1, tr0, tr1;

    cu cu0 (
        .clk(clk), .rst(rst), .start(start0), .start_pc(wt0 ? PS_ADDR : VS_ADDR),
        .cu_id(4'd0), .wave_type(wt0), .frame(frames_done),
        .init_v(iv0), .init_mask(im0), .init_s1(CONST_ADDR), .init_s2(is2_0),
        .imem_addr(imem0_addr), .imem_data(imem0_data),
        .dmem_addr(d0_addr), .dmem_wdata(d0_wdata), .dmem_we(d0_we), .dmem_rdata(mem_rdata),
        .dmem_req(d0_req), .dmem_gnt(cu0_gnt),
        .export_valid(ex0), .export_target(ext0), .export_mask(exm0), .export_data(exd0),
        .tex_start(ts0), .tex_slot(tsl0), .tex_u(tu0), .tex_v(tv0), .tex_done(td0), .tex_result(tr0),
        .busy(busy0), .done(done0), .error(err0), .instr_count(cnt0)
    );
    cu cu1 (
        .clk(clk), .rst(rst), .start(start1), .start_pc(wt1 ? PS_ADDR : VS_ADDR),
        .cu_id(4'd1), .wave_type(wt1), .frame(frames_done),
        .init_v(iv1), .init_mask(im1), .init_s1(CONST_ADDR), .init_s2(is2_1),
        .imem_addr(imem1_addr), .imem_data(imem1_data),
        .dmem_addr(d1_addr), .dmem_wdata(d1_wdata), .dmem_we(d1_we), .dmem_rdata(mem_rdata),
        .dmem_req(d1_req), .dmem_gnt(cu1_gnt),
        .export_valid(ex1), .export_target(ext1), .export_mask(exm1), .export_data(exd1),
        .tex_start(ts1), .tex_slot(tsl1), .tex_u(tu1), .tex_v(tv1), .tex_done(td1), .tex_result(tr1),
        .busy(busy1), .done(done1), .error(err1), .instr_count(cnt1)
    );
    tex_unit tex0 (
        .clk(clk), .rst(rst), .start(ts0), .slot(tsl0), .wave_type(wt0), .u(tu0), .v(tv0),
        .tex_addr(TEX_ADDR), .tex_info(TEX_INFO),
        .mem_req(t0_req), .mem_addr(t0_addr), .mem_gnt(tx0_gnt), .mem_rdata(mem_rdata),
        .done(td0), .result(tr0), .busy(tb0)
    );
    tex_unit tex1 (
        .clk(clk), .rst(rst), .start(ts1), .slot(tsl1), .wave_type(wt1), .u(tu1), .v(tv1),
        .tex_addr(TEX_ADDR), .tex_info(TEX_INFO),
        .mem_req(t1_req), .mem_addr(t1_addr), .mem_gnt(tx1_gnt), .mem_rdata(mem_rdata),
        .done(td1), .result(tr1), .busy(tb1)
    );

    // pixel exports waiting for the ROP (one per CU)
    reg          pend0, pend1;
    reg  [15:0]  ppx0, ppy0, ppx1, ppy1;
    reg  [3:0]   ppm0, ppm1;
    reg  [127:0] ppc0, ppc1, ppd0, ppd1;

    wire free0 = (cs0 == FREE) && !pend0;
    wire free1 = (cs1 == FREE) && !pend1;
    reg  prefer1;

    // ==================================================================
    // 5. DRAW CONTROL + INPUT ASSEMBLER + VERTEX OUTPUT BUFFER
    // ==================================================================
    localparam D_IDLE = 4'd0, D_IDX = 4'd1, D_ATT = 4'd2, D_LAUNCH = 4'd3, D_VWAIT = 4'd4,
               D_TRI = 4'd5, D_DRAIN = 4'd6, D_ERR = 4'd7;
    reg [3:0]  dstate;
    reg [31:0] d_count, d_first, vi, bstart, cur_index, ntri, ti;
    reg        d_indexed;
    reg [1:0]  blane;                        // lane being filled
    reg [2:0]  bfill;                        // vertices in the batch
    reg [3:0]  aj;                           // attribute being read
    reg [255:0] battr [0:3];                 // batch: 8 attributes per lane

    reg [31:0] vob [0:MAXV*11-1];            // per vertex: x, y, z, ATTR0-7

    wire [31:0] ib_byte = IB_ADDR + ((d_first + vi) << 1);
    assign ia_req  = (dstate == D_IDX && d_indexed) || (dstate == D_ATT);
    assign ia_addr = (dstate == D_IDX) ? {ib_byte[31:2], 2'b00} : VB_ADDR + cur_index * VB_STRIDE + {aj, 2'b00};
    wire [15:0] index_half = ib_byte[1] ? mem_rdata[31:16] : mem_rdata[15:0];

    wire vertex_launch = (dstate == D_LAUNCH) && (free0 || free1);

    // primitive unit input: triangle ti from the vertex output buffer
    wire        prim_in_ready, prim_out_valid, prim_busy;
    wire [31:0] prim_out_prim;
    wire [95:0] prim_out_x, prim_out_y, prim_out_z;
    wire [767:0] prim_out_attr;
    wire [3:0]  prim_out_nattrs;
    wire [31:0] p_in, p_near, p_zero, p_back, p_out;
    reg  [95:0]  tri_x, tri_y, tri_z;
    reg  [767:0] tri_attr;
    integer tv, ta;
    always @* begin
        for (tv = 0; tv < 3; tv = tv + 1) begin
            tri_x[32*tv +: 32] = vob[(3*ti + tv) * 11 + 0];
            tri_y[32*tv +: 32] = vob[(3*ti + tv) * 11 + 1];
            tri_z[32*tv +: 32] = vob[(3*ti + tv) * 11 + 2];
            for (ta = 0; ta < 8; ta = ta + 1)
                tri_attr[256*tv + 32*ta +: 32] = vob[(3*ti + tv) * 11 + 3 + ta];
        end
    end
    wire prim_in_valid = (dstate == D_TRI) && (ti < ntri);

    // rasterizer / pixel side
    wire        ras_tri_ready, ras_busy, quad_valid;
    wire [31:0] quad_prim, ras_quads, ras_dropped;
    wire [15:0] quad_x, quad_y;
    wire [3:0]  quad_mask;
    wire [127:0] quad_close;
    wire [1023:0] quad_attr;
    wire        rop_quad_ready, rop_qvalid;
    wire [31:0] rop_written, rop_failed;

    // a triangle may enter the rasterizer only when all earlier pixels are written
    wire drained  = !ras_busy && !quad_valid && (cs0 == FREE || wt0 == 1'b0) &&
                    (cs1 == FREE || wt1 == 1'b0) && !pend0 && !pend1 && !rop_busy && !rop_qvalid;
    wire pix_idle = drained && (cs0 == FREE) && (cs1 == FREE);
    wire quad_ready = (free0 || free1) && (dstate == D_TRI || dstate == D_DRAIN);
    assign draw_busy = (dstate != D_IDLE);

    integer l, k;
    always @(posedge clk) begin
        if (rst) begin
            dstate <= D_IDLE;  vi <= 32'd0;  bstart <= 32'd0;  blane <= 2'd0;  bfill <= 3'd0;  aj <= 4'd0;
            d_count <= 32'd0;  d_first <= 32'd0;  d_indexed <= 1'b0;  cur_index <= 32'd0;
            ntri <= 32'd0;  ti <= 32'd0;  error <= 1'b0;
            cs0 <= FREE;  cs1 <= FREE;  start0 <= 1'b0;  start1 <= 1'b0;  prefer1 <= 1'b0;
            iv0 <= 1408'd0;  iv1 <= 1408'd0;  im0 <= 4'd0;  im1 <= 4'd0;  is2_0 <= 32'd0;  is2_1 <= 32'd0;
            wt0 <= 1'b0;  wt1 <= 1'b0;  vbase0 <= 32'd0;  vbase1 <= 32'd0;
            px0 <= 16'd0;  py0 <= 16'd0;  px1 <= 16'd0;  py1 <= 16'd0;  pc0 <= 128'd0;  pc1 <= 128'd0;
            pend0 <= 1'b0;  pend1 <= 1'b0;  ppx0 <= 16'd0;  ppy0 <= 16'd0;  ppx1 <= 16'd0;  ppy1 <= 16'd0;
            ppm0 <= 4'd0;  ppm1 <= 4'd0;  ppc0 <= 128'd0;  ppc1 <= 128'd0;  ppd0 <= 128'd0;  ppd1 <= 128'd0;
            st_vertex_waves <= 32'd0;  st_pixel_waves <= 32'd0;
            for (l = 0; l < 4; l = l + 1) battr[l] <= 256'd0;
        end else begin
            start0 <= 1'b0;
            start1 <= 1'b0;
            if (err0 || err1 || cmd_error) error <= 1'b1;

            // ---------------- CU slots: STARTING -> RUNNING -> FREE ----------------
            if (cs0 == STARTING) cs0 <= RUNNING;
            else if (cs0 == RUNNING && !busy0) cs0 <= FREE;
            if (cs1 == STARTING) cs1 <= RUNNING;
            else if (cs1 == RUNNING && !busy1) cs1 <= FREE;

            // ---------------- exports ----------------
            if (ex0 && wt0 == 1'b0)                                 // vertex export -> buffer
                for (l = 0; l < 4; l = l + 1)
                    if (exm0[l] && ext0 <= 8'd2)
                        vob[(vbase0 + l) * 11 + ext0] <= exd0[32*l +: 32];
                    else if (exm0[l] && ext0 >= 8'd8 && ext0 <= 8'd15)
                        vob[(vbase0 + l) * 11 + ext0 - 5] <= exd0[32*l +: 32];
            if (ex1 && wt1 == 1'b0)
                for (l = 0; l < 4; l = l + 1)
                    if (exm1[l] && ext1 <= 8'd2)
                        vob[(vbase1 + l) * 11 + ext1] <= exd1[32*l +: 32];
                    else if (exm1[l] && ext1 >= 8'd8 && ext1 <= 8'd15)
                        vob[(vbase1 + l) * 11 + ext1 - 5] <= exd1[32*l +: 32];
            if (ex0 && wt0 == 1'b1 && ext0 == 8'd16) begin          // pixel export -> ROP queue
                pend0 <= 1'b1;  ppx0 <= px0;  ppy0 <= py0;  ppm0 <= exm0;  ppc0 <= pc0;  ppd0 <= exd0;
            end
            if (ex1 && wt1 == 1'b1 && ext1 == 8'd16) begin
                pend1 <= 1'b1;  ppx1 <= px1;  ppy1 <= py1;  ppm1 <= exm1;  ppc1 <= pc1;  ppd1 <= exd1;
            end
            if (rop_qvalid && rop_quad_ready) begin                // the ROP took one
                if (pend0) pend0 <= 1'b0; else pend1 <= 1'b0;
            end

            // ---------------- launch a wave on a free CU ----------------
            if (vertex_launch) begin
                if (free0 && (!free1 || !prefer1)) begin
                    start0 <= 1'b1;  cs0 <= STARTING;  wt0 <= 1'b0;  vbase0 <= bstart;  is2_0 <= bstart;
                    im0 <= (4'd1 << bfill) - 4'd1;
                    iv0 <= 1408'd0;
                    for (k = 0; k < 8; k = k + 1)
                        for (l = 0; l < 4; l = l + 1)
                            iv0[128*k + 32*l +: 32] <= battr[l][32*k +: 32];
                    prefer1 <= 1'b1;
                end else begin
                    start1 <= 1'b1;  cs1 <= STARTING;  wt1 <= 1'b0;  vbase1 <= bstart;  is2_1 <= bstart;
                    im1 <= (4'd1 << bfill) - 4'd1;
                    iv1 <= 1408'd0;
                    for (k = 0; k < 8; k = k + 1)
                        for (l = 0; l < 4; l = l + 1)
                            iv1[128*k + 32*l +: 32] <= battr[l][32*k +: 32];
                    prefer1 <= 1'b0;
                end
                st_vertex_waves <= st_vertex_waves + 32'd1;
            end else if (quad_valid && quad_ready) begin           // pixel wave
                if (free0 && (!free1 || !prefer1)) begin
                    start0 <= 1'b1;  cs0 <= STARTING;  wt0 <= 1'b1;  is2_0 <= quad_prim;
                    im0 <= quad_mask;  px0 <= quad_x;  py0 <= quad_y;  pc0 <= quad_close;
                    iv0 <= {quad_attr, quad_close,
                            16'd0, quad_y + 16'd1, 16'd0, quad_y + 16'd1, 16'd0, quad_y, 16'd0, quad_y,
                            16'd0, quad_x + 16'd1, 16'd0, quad_x, 16'd0, quad_x + 16'd1, 16'd0, quad_x};
                    prefer1 <= 1'b1;
                end else begin
                    start1 <= 1'b1;  cs1 <= STARTING;  wt1 <= 1'b1;  is2_1 <= quad_prim;
                    im1 <= quad_mask;  px1 <= quad_x;  py1 <= quad_y;  pc1 <= quad_close;
                    iv1 <= {quad_attr, quad_close,
                            16'd0, quad_y + 16'd1, 16'd0, quad_y + 16'd1, 16'd0, quad_y, 16'd0, quad_y,
                            16'd0, quad_x + 16'd1, 16'd0, quad_x, 16'd0, quad_x + 16'd1, 16'd0, quad_x};
                    prefer1 <= 1'b0;
                end
                st_pixel_waves <= st_pixel_waves + 32'd1;
            end

            // ---------------- the draw ----------------
            case (dstate)
            D_IDLE: if (draw_go) begin
                d_count <= pay[0];  d_first <= pay[1];  d_indexed <= h_param[0];
                vi <= 32'd0;  bstart <= 32'd0;  blane <= 2'd0;  bfill <= 3'd0;  aj <= 4'd0;
                ntri <= pay[0] / 3;  ti <= 32'd0;
                if (pay[0] > MAXV) begin error <= 1'b1;  dstate <= D_ERR; end
                else dstate <= (pay[0] == 32'd0) ? D_IDLE : D_IDX;
            end

            // input assembler: index of vertex vi
            D_IDX: begin
                if (!d_indexed) begin
                    cur_index <= d_first + vi;  aj <= 4'd0;  dstate <= D_ATT;
                end else if (g_ia) begin
                    cur_index <= {16'd0, index_half};  aj <= 4'd0;  dstate <= D_ATT;
                end
            end

            // input assembler: its attributes, one word per granted cycle
            D_ATT: if (g_ia) begin
                battr[blane][32*aj +: 32] <= mem_rdata;
                if (aj + 4'd1 >= VB_ATTRS[3:0] || aj == 4'd7) begin
                    bfill <= bfill + 3'd1;
                    vi    <= vi + 32'd1;
                    if (blane == 2'd3 || vi + 32'd1 == d_count) dstate <= D_LAUNCH;
                    else begin blane <= blane + 2'd1;  dstate <= D_IDX; end
                end else aj <= aj + 4'd1;
            end

            D_LAUNCH: if (vertex_launch) begin                     // the wave started above
                for (l = 0; l < 4; l = l + 1) battr[l] <= 256'd0;
                blane <= 2'd0;  bfill <= 3'd0;  bstart <= vi;
                dstate <= (vi == d_count) ? D_VWAIT : D_IDX;
            end

            D_VWAIT: if (cs0 == FREE && cs1 == FREE && !start0 && !start1) dstate <= D_TRI;

            // primitive unit: one triangle at a time from the vertex output buffer
            D_TRI: begin
                if (prim_in_valid && prim_in_ready) ti <= ti + 32'd1;
                if (ti >= ntri) dstate <= D_DRAIN;
            end

            D_DRAIN: if (!prim_busy && pix_idle && !start0 && !start1) dstate <= D_IDLE;

            D_ERR: ;                                               // too many vertices: stop
            default: dstate <= D_IDLE;
            endcase
        end
    end

    // ==================================================================
    // 6. PRIMITIVE UNIT, RASTERIZER, RENDER OUTPUT UNIT
    // ==================================================================
    prim_unit prim (
        .clk(clk), .rst(rst), .draw_start(draw_go),
        .proj_cx(CX), .proj_cy(CY), .proj_focal(FOCAL), .near_z(NEAR_Z), .cull_mode(CULL),
        .in_valid(prim_in_valid), .in_ready(prim_in_ready),
        .in_x(tri_x), .in_y(tri_y), .in_z(tri_z), .in_attr(tri_attr), .n_attrs(VS_OUT_ATTRS[3:0]),
        .out_valid(prim_out_valid), .out_ready(ras_tri_ready && drained),
        .out_prim(prim_out_prim), .out_x(prim_out_x), .out_y(prim_out_y), .out_z(prim_out_z),
        .out_attr(prim_out_attr), .out_nattrs(prim_out_nattrs),
        .busy(prim_busy), .st_in(p_in), .st_near(p_near), .st_zero(p_zero), .st_back(p_back), .st_out(p_out)
    );

    rasterizer ras (
        .clk(clk), .rst(rst),
        .tri_valid(prim_out_valid && drained), .tri_ready(ras_tri_ready), .tri_prim(prim_out_prim),
        .tri_x(prim_out_x), .tri_y(prim_out_y), .tri_z(prim_out_z), .tri_attr(prim_out_attr),
        .n_attrs(prim_out_nattrs),
        .width(VW), .height(VH), .depth_mode(DEPTH_MODE), .depth_addr(DEPTH_ADDR),
        .mem_req(ras_req), .mem_addr(ras_addr), .mem_gnt(g_ras), .mem_rdata(mem_rdata),
        .quad_valid(quad_valid), .quad_ready(quad_ready), .quad_prim(quad_prim),
        .quad_x(quad_x), .quad_y(quad_y), .quad_mask(quad_mask), .quad_close(quad_close),
        .quad_attr(quad_attr), .busy(ras_busy), .stat_quads(ras_quads), .stat_early_dropped(ras_dropped)
    );

    assign rop_qvalid = pend0 || pend1;
    rop rop_u (
        .clk(clk), .rst(rst), .width(VW), .height(VH),
        .fb_addr(back_addr), .depth_addr(DEPTH_ADDR), .stencil_addr(STENCIL_ADDR),
        .depth_mode(DEPTH_MODE), .blend_mode(BLEND), .color_mask(CMASK),
        .clear_start(clear_go), .clear_flags(h_param[2:0]), .clear_color(pay[0]),
        .clear_depth(pay[1][15:0]), .clear_stencil(pay[2][7:0]),
        .quad_valid(rop_qvalid), .quad_ready(rop_quad_ready),
        .quad_x(pend0 ? ppx0 : ppx1), .quad_y(pend0 ? ppy0 : ppy1),
        .quad_mask(pend0 ? ppm0 : ppm1), .quad_close(pend0 ? ppc0 : ppc1),
        .quad_color(pend0 ? ppd0 : ppd1),
        .mem_req(rop_req), .mem_addr(rop_addr), .mem_we(rop_we), .mem_wdata(rop_wdata),
        .mem_gnt(g_rop), .mem_rdata(mem_rdata),
        .busy(rop_busy), .stat_written(rop_written), .stat_depth_failed(rop_failed)
    );

    // ==================================================================
    // 7. DISPLAY ENGINE (spec 8.11)
    // ==================================================================
    display_engine #(.H_ACTIVE(H_ACTIVE), .H_FP(H_FP), .H_SYNC(H_SYNC), .H_BP(H_BP),
                     .V_ACTIVE(V_ACTIVE), .V_FP(V_FP), .V_SYNC(V_SYNC), .V_BP(V_BP)) disp (
        .clk(clk), .rst(rst), .enable(disp_on), .init_front(disp_first),
        .swap_valid(disp_swap), .swap_addr(swap_addr), .flipped(disp_flipped), .front(),
        .mem_req(disp_req), .mem_addr(disp_addr), .mem_gnt(g_disp), .mem_rdata(mem_rdata),
        .de(video_de), .hsync(video_hsync), .vsync(video_vsync),
        .r(video_r), .g(video_g), .b(video_b),
        .underflow(video_underflow), .frame_count(disp_frame_count)
    );
    assign video_frames = disp_frame_count;

endmodule
