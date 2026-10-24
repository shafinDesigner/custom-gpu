// rop.v - render output unit (spec section 8.9)                     Phase 5, stage 2
// Must match render_output() and clear() in gpu_model.py exactly.
//
// Quad (4 pixels from pixel shading), one lane at a time:
//   LANE -> DREAD (depth test) -> DWRITE (depth write, mode 1)
//        -> CREAD (old color, only for blending or a color mask) -> CWRITE -> next lane
// Clear: fills the color buffer, then depth, then stencil, one word per granted cycle.
//
// Every memory access is one cycle with mem_gnt = 1; without the grant it waits.
// Depth values are 16 bits, two per word: a depth write reads the word, replaces
// one half and writes the word back.

`timescale 1ns/1ps

module rop (
    input  wire         clk,
    input  wire         rst,

    // pipeline state (only change while busy = 0)
    input  wire [15:0]  width,
    input  wire [15:0]  height,
    input  wire [31:0]  fb_addr,          // back buffer
    input  wire [31:0]  depth_addr,
    input  wire [31:0]  stencil_addr,
    input  wire [1:0]   depth_mode,       // 0 off, 1 test + write, 2 test only
    input  wire [1:0]   blend_mode,       // 0 off, 1 alpha, 2-3 additive
    input  wire [3:0]   color_mask,       // bit i = write channel i (R, G, B, A)

    // clear command
    input  wire         clear_start,      // 1 for one cycle while idle
    input  wire [2:0]   clear_flags,      // bit 0 color, 1 depth, 2 stencil
    input  wire [31:0]  clear_color,
    input  wire [15:0]  clear_depth,
    input  wire [7:0]   clear_stencil,

    // pixels from pixel shading
    input  wire         quad_valid,
    output wire         quad_ready,
    input  wire [15:0]  quad_x,           // top-left pixel of the quad
    input  wire [15:0]  quad_y,
    input  wire [3:0]   quad_mask,        // lanes to write
    input  wire [127:0] quad_close,       // closeness per lane (from the rasterizer)
    input  wire [127:0] quad_color,       // RGBA8 per lane (from EXPORT COLOR)

    // memory port
    output wire         mem_req,
    output wire [31:0]  mem_addr,
    output wire         mem_we,           // only when granted
    output wire [31:0]  mem_wdata,
    input  wire         mem_gnt,
    input  wire [31:0]  mem_rdata,

    output wire         busy,
    output reg  [31:0]  stat_written,
    output reg  [31:0]  stat_depth_failed
);

    localparam R_IDLE = 4'd0, R_LANE = 4'd1, R_DREAD = 4'd2, R_DWRITE = 4'd3,
               R_CREAD = 4'd4, R_CWRITE = 4'd5, R_NEXT = 4'd6, R_CPHASE = 4'd7, R_CLEAR = 4'd8;

    reg [3:0]  state;
    reg [15:0] qx, qy;
    reg [3:0]  qmask;
    reg [31:0] qclose [0:3];
    reg [31:0] qcolor [0:3];
    reg [1:0]  lane;
    reg [31:0] dword;                       // depth word read in DREAD
    reg [31:0] dst;                         // old color read in CREAD

    assign quad_ready = (state == R_IDLE);
    assign busy       = (state != R_IDLE);

    // ------------------------------------------------------------------
    // This lane's addresses and values
    // ------------------------------------------------------------------
    wire [31:0] px      = {16'd0, qx} + {31'd0, lane[0]};
    wire [31:0] py      = {16'd0, qy} + {31'd0, lane[1]};
    wire [31:0] pidx    = py * {16'd0, width} + px;
    wire [31:0] d_byte  = depth_addr + (pidx << 1);
    wire [31:0] c_addr  = fb_addr + (pidx << 2);
    wire [31:0] close_l = qclose[lane];
    wire [15:0] d_new   = (close_l > 32'h0000_FFFF) ? 16'hFFFF : close_l[15:0];
    wire [15:0] d_old   = d_byte[1] ? mem_rdata[31:16] : mem_rdata[15:0];
    wire        need_dst = (blend_mode != 2'd0) || (color_mask != 4'hF);

    // ------------------------------------------------------------------
    // Blending and color mask
    // ------------------------------------------------------------------
    function [7:0] alpha_ch;                // (s * a' + d * (256 - a')) >> 8
        input [7:0] s;
        input [7:0] d;
        input [8:0] ap;
        reg [16:0] t;
        begin
            t = s * ap + d * (9'd256 - ap);
            alpha_ch = t[15:8];
        end
    endfunction

    function [7:0] add_ch;                  // s + d, clamped at 255
        input [7:0] s;
        input [7:0] d;
        reg [8:0] t;
        begin
            t = s + d;
            add_ch = t[8] ? 8'hFF : t[7:0];
        end
    endfunction

    wire [31:0] src = qcolor[lane];
    wire [8:0]  ap  = {1'b0, src[31:24]} + {8'd0, src[31]};     // a + (a >> 7): 0..256
    wire [31:0] blended =
        (blend_mode == 2'd0) ? src :
        (blend_mode == 2'd1) ? {alpha_ch(src[31:24], dst[31:24], ap), alpha_ch(src[23:16], dst[23:16], ap),
                                alpha_ch(src[15:8],  dst[15:8],  ap), alpha_ch(src[7:0],   dst[7:0],   ap)}
                             : {add_ch(src[31:24], dst[31:24]), add_ch(src[23:16], dst[23:16]),
                                add_ch(src[15:8],  dst[15:8]),  add_ch(src[7:0],   dst[7:0])};
    wire [31:0] keep  = {{8{~color_mask[3]}}, {8{~color_mask[2]}}, {8{~color_mask[1]}}, {8{~color_mask[0]}}};
    wire [31:0] final_color = (blended & ~keep) | (dst & keep);

    // ------------------------------------------------------------------
    // Clear engine
    // ------------------------------------------------------------------
    reg [2:0]  cflags;
    reg [31:0] cval_color, cval_depth, cval_stencil;
    reg [1:0]  cphase;                       // 0 color, 1 depth, 2 stencil, 3 done
    reg [31:0] clr_base, clr_value, clr_count, ci;
    wire [31:0] npix = {16'd0, width} * {16'd0, height};

    // ------------------------------------------------------------------
    // Memory port
    // ------------------------------------------------------------------
    wire        req_now = (state == R_DREAD) || (state == R_DWRITE) || (state == R_CREAD) ||
                          (state == R_CWRITE) || (state == R_CLEAR);
    wire        we_now  = (state == R_DWRITE) || (state == R_CWRITE) || (state == R_CLEAR);
    assign mem_req   = req_now;
    assign mem_we    = we_now && mem_gnt;
    assign mem_addr  = (state == R_DREAD || state == R_DWRITE) ? {d_byte[31:2], 2'b00} :
                       (state == R_CLEAR) ? clr_base + (ci << 2) : c_addr;
    assign mem_wdata = (state == R_DWRITE) ? (d_byte[1] ? {d_new, dword[15:0]} : {dword[31:16], d_new}) :
                       (state == R_CLEAR)  ? clr_value : final_color;

    // ------------------------------------------------------------------
    // State machine
    // ------------------------------------------------------------------
    integer l;
    always @(posedge clk) begin
        if (rst) begin
            state <= R_IDLE;  lane <= 2'd0;  qx <= 16'd0;  qy <= 16'd0;  qmask <= 4'd0;
            dword <= 32'd0;  dst <= 32'd0;  stat_written <= 32'd0;  stat_depth_failed <= 32'd0;
            cflags <= 3'd0;  cval_color <= 32'd0;  cval_depth <= 32'd0;  cval_stencil <= 32'd0;
            cphase <= 2'd0;  clr_base <= 32'd0;  clr_value <= 32'd0;  clr_count <= 32'd0;  ci <= 32'd0;
            for (l = 0; l < 4; l = l + 1) begin qclose[l] <= 32'd0;  qcolor[l] <= 32'd0; end
        end else begin
            case (state)
            R_IDLE: begin
                if (clear_start) begin
                    cflags       <= clear_flags;
                    cval_color   <= clear_color;
                    cval_depth   <= {clear_depth, clear_depth};
                    cval_stencil <= {4{clear_stencil}};
                    cphase       <= 2'd0;
                    state        <= R_CPHASE;
                end else if (quad_valid) begin
                    qx <= quad_x;  qy <= quad_y;  qmask <= quad_mask;
                    for (l = 0; l < 4; l = l + 1) begin
                        qclose[l] <= quad_close[32*l +: 32];
                        qcolor[l] <= quad_color[32*l +: 32];
                    end
                    lane  <= 2'd0;
                    state <= R_LANE;
                end
            end

            // ---------------- one pixel ----------------
            R_LANE: begin
                if (!qmask[lane])             state <= R_NEXT;
                else if (depth_mode != 2'd0)  state <= R_DREAD;
                else if (need_dst)            state <= R_CREAD;
                else                          state <= R_CWRITE;
            end

            R_DREAD: if (mem_gnt) begin
                dword <= mem_rdata;
                if (d_new <= d_old) begin                         // not closer: drop it
                    stat_depth_failed <= stat_depth_failed + 32'd1;
                    state <= R_NEXT;
                end else if (depth_mode == 2'd1) state <= R_DWRITE;
                else if (need_dst)               state <= R_CREAD;
                else                             state <= R_CWRITE;
            end

            R_DWRITE: if (mem_gnt) state <= need_dst ? R_CREAD : R_CWRITE;

            R_CREAD: if (mem_gnt) begin
                dst   <= mem_rdata;
                state <= R_CWRITE;
            end

            R_CWRITE: if (mem_gnt) begin
                stat_written <= stat_written + 32'd1;
                state <= R_NEXT;
            end

            R_NEXT: begin
                lane <= lane + 2'd1;
                state <= (lane == 2'd3) ? R_IDLE : R_LANE;
            end

            // ---------------- clear: color, then depth, then stencil ----------------
            R_CPHASE: begin
                ci <= 32'd0;
                if (cphase == 2'd0) begin
                    if (cflags[0] && npix != 0) begin
                        clr_base <= fb_addr;  clr_value <= cval_color;  clr_count <= npix;
                        state <= R_CLEAR;
                    end
                    cphase <= 2'd1;
                end else if (cphase == 2'd1) begin
                    if (cflags[1] && npix != 0) begin
                        clr_base <= depth_addr;  clr_value <= cval_depth;  clr_count <= (npix + 1) >> 1;
                        state <= R_CLEAR;
                    end
                    cphase <= 2'd2;
                end else if (cphase == 2'd2) begin
                    if (cflags[2] && npix != 0) begin
                        clr_base <= stencil_addr;  clr_value <= cval_stencil;  clr_count <= (npix + 3) >> 2;
                        state <= R_CLEAR;
                    end
                    cphase <= 2'd3;
                end else begin
                    state <= R_IDLE;
                end
            end

            R_CLEAR: if (mem_gnt) begin
                ci <= ci + 32'd1;
                if (ci + 32'd1 == clr_count) state <= R_CPHASE;   // next phase
            end

            default: state <= R_IDLE;
            endcase
        end
    end

endmodule
