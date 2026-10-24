// rasterizer.v - hardware rasterizer (spec section 8.8)                Phase 5, stage 1
// Must match rasterizer() in gpu_model.py exactly.
//
// For one triangle:
//   SETUP    winding fix (area < 0), top-left bias per edge, bounding box (even x, y)
//   DIVAREA  1 / area with 48 fraction bits: exact bit-serial divider, 49 cycles
//   then for every 2x2 quad in the bounding box, row by row:
//   EVAL     edge values of the 4 pixel centers -> coverage (skip empty quads)
//   WEIGHT   barycentric weights and closeness (1/z) for all 4 lanes
//   DEPTH    early depth test: read the depth buffer for covered lanes
//   RCP      1 / closeness per lane (exact bit-serial divider, 33 cycles each)
//   ATTR     perspective-correct attributes, one attribute per cycle
//   OUT      hand the quad to pixel shading (waits for quad_ready)
//
// Wide arithmetic keeps results exact for ANY 32-bit input; a later version will
// narrow the widths once the guard band is fixed in the spec.

`timescale 1ns/1ps

module rasterizer (
    input  wire          clk,
    input  wire          rst,

    // triangle input (from the primitive unit). Vertex order: c | b | a
    input  wire          tri_valid,
    output wire          tri_ready,       // 1 = idle, will take a triangle this cycle
    input  wire [31:0]   tri_prim,        // primitive number (goes to s2 of pixel waves)
    input  wire [95:0]   tri_x,           // 12.4 signed screen x per vertex
    input  wire [95:0]   tri_y,
    input  wire [95:0]   tri_z,           // closeness 1/z (0.16) per vertex
    input  wire [767:0]  tri_attr,        // attr x 1/z, 8 per vertex (attr 7 | ... | 0)
    input  wire [3:0]    n_attrs,         // 0-8

    // pipeline state
    input  wire [15:0]   width,
    input  wire [15:0]   height,
    input  wire [1:0]    depth_mode,      // 0 off, 1 test + write, 2 test only
    input  wire [31:0]   depth_addr,

    // memory: depth buffer reads
    output wire          mem_req,
    output wire [31:0]   mem_addr,
    input  wire          mem_gnt,
    input  wire [31:0]   mem_rdata,

    // quad output (to pixel shading)
    output reg           quad_valid,
    input  wire          quad_ready,
    output reg  [31:0]   quad_prim,
    output reg  [15:0]   quad_x,          // top-left pixel of the quad
    output reg  [15:0]   quad_y,
    output reg  [3:0]    quad_mask,       // covered lanes (after early depth)
    output reg  [127:0]  quad_close,      // closeness per lane: lane 3 | 2 | 1 | 0
    output reg  [1023:0] quad_attr,       // attr k at [128k +: 128], lanes inside

    output wire          busy,
    output reg  [31:0]   stat_quads,
    output reg  [31:0]   stat_early_dropped
);

    localparam S_IDLE = 4'd0, S_SETUP = 4'd1, S_SETUP2 = 4'd2, S_DIVAREA = 4'd3,
               S_EVAL = 4'd4, S_WEIGHT = 4'd5, S_DEPTH = 4'd6, S_DCHK = 4'd7,
               S_RCP = 4'd8, S_ATTR = 4'd9, S_OUT = 4'd10, S_NEXT = 4'd11;

    reg [3:0] state;
    assign tri_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);

    // ------------------------------------------------------------------
    // Triangle (vertex 0 = a, 1 = b, 2 = c), after the winding fix
    // ------------------------------------------------------------------
    reg signed [31:0] vx [0:2];
    reg signed [31:0] vy [0:2];
    reg        [31:0] vz [0:2];
    reg       [255:0] va [0:2];
    reg        [3:0]  nat;
    reg        [31:0] prim;
    reg        [67:0] area_neg;              // -area > 0
    reg               tl [0:2];              // edge i is top-left
    reg signed [31:0] mnx, mny, mxx, mxy;    // bounding box (pixels)
    reg signed [31:0] qx, qy;                // current quad
    reg        [48:0] inv_area;              // 2^48 / -area

    // per quad
    reg               [3:0] covmask;
    reg signed [67:0]  E   [0:11];           // edge values, index lane*3 + edge
    reg signed [85:0]  wgt [0:11];           // weights (0.16), index lane*3 + corner
    reg        [31:0]  close [0:3];
    reg        [31:0]  rcp   [0:3];
    reg        [1:0]   lane;
    reg        [3:0]   k;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    function signed [67:0] edgef;            // (x2-x1)(py-y1) - (y2-y1)(px-x1)
        input signed [31:0] x1, y1, x2, y2, px, py;
        reg signed [67:0] dx, dy, ex, ey;
        begin
            dx = x2 - x1;  dy = y2 - y1;  ex = px - x1;  ey = py - y1;
            edgef = dx * ey - dy * ex;
        end
    endfunction

    function is_tl;                          // left edge, or flat top edge (y down)
        input signed [31:0] x1, y1, x2, y2;
        is_tl = (y2 > y1) || ((y2 == y1) && (x2 < x1));
    endfunction

    // ------------------------------------------------------------------
    // Bit-serial divider: (1 << K) / divisor, used for 1/area and 1/closeness
    // ------------------------------------------------------------------
    reg        [68:0] div_rem;
    reg        [48:0] div_quo;
    reg               div_first;
    reg         [6:0] div_cnt;
    reg        [67:0] divisor;
    wire       [68:0] rem_shift = {div_rem[67:0], div_first};
    wire              div_ge    = (rem_shift >= {1'b0, divisor});
    wire       [68:0] rem_next  = div_ge ? (rem_shift - {1'b0, divisor}) : rem_shift;
    wire       [48:0] quo_next  = {div_quo[47:0], div_ge};
    reg         [1:0] rcp_phase;
    reg               rcp_neg;
    wire       [31:0] rcp_mag   = (quo_next > 49'h0_7FFF_FFFF) ? 32'h7FFF_FFFF : quo_next[31:0];

    // ------------------------------------------------------------------
    // Depth buffer address of the current lane
    // ------------------------------------------------------------------
    wire signed [31:0] lane_x   = qx + {31'd0, lane[0]};
    wire signed [31:0] lane_y   = qy + {31'd0, lane[1]};
    wire        [31:0] d_byte   = depth_addr + ((lane_y * {16'd0, width} + lane_x) << 1);
    wire        [15:0] d_stored = d_byte[1] ? mem_rdata[31:16] : mem_rdata[15:0];
    wire        [31:0] lane_close = close[lane];
    wire        [15:0] d_new    = (lane_close > 32'h0000_FFFF) ? 16'hFFFF : lane_close[15:0];

    assign mem_req  = (state == S_DEPTH) && (depth_mode != 2'd0) && covmask[lane];
    assign mem_addr = {d_byte[31:2], 2'b00};

    // ------------------------------------------------------------------
    // Temporaries (blocking, used inside the clocked block only)
    // ------------------------------------------------------------------
    reg signed  [67:0] t_area, t_e;
    reg signed [117:0] t_prod;
    reg signed  [85:0] t_w0, t_w1, t_w2;
    reg signed [120:0] t_csum;
    reg signed [104:0] t_c;
    reg signed [119:0] t_asum;
    reg signed [103:0] t_az;
    reg signed  [63:0] t_mul;
    reg signed  [31:0] t_x, t_y, t_px, t_py, t_min, t_max;
    reg                t_ok;
    reg          [3:0] t_cov;
    integer l, i, p1, p2;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;  quad_valid <= 1'b0;  stat_quads <= 32'd0;  stat_early_dropped <= 32'd0;
            quad_prim <= 32'd0;  quad_x <= 16'd0;  quad_y <= 16'd0;  quad_mask <= 4'd0;
            quad_close <= 128'd0;  quad_attr <= 1024'd0;
            lane <= 2'd0;  k <= 4'd0;  covmask <= 4'd0;  rcp_phase <= 2'd0;  rcp_neg <= 1'b0;
            div_rem <= 69'd0;  div_quo <= 49'd0;  div_first <= 1'b0;  div_cnt <= 7'd0;  divisor <= 68'd0;
        end else begin
            case (state)
            // ---------------- take a triangle ----------------
            S_IDLE: if (tri_valid) begin
                for (i = 0; i < 3; i = i + 1) begin
                    vx[i] <= tri_x[32*i +: 32];
                    vy[i] <= tri_y[32*i +: 32];
                    vz[i] <= tri_z[32*i +: 32];
                    va[i] <= tri_attr[256*i +: 256];
                end
                nat   <= n_attrs;
                prim  <= tri_prim;
                state <= S_SETUP;
            end

            // ---------------- winding: make area < 0 ----------------
            S_SETUP: begin
                t_area = edgef(vx[0], vy[0], vx[1], vy[1], vx[2], vy[2]);
                if (t_area == 0) begin
                    state <= S_IDLE;                         // nothing to draw
                end else begin
                    if (t_area > 0) begin                    // swap b and c
                        vx[1] <= vx[2];  vx[2] <= vx[1];
                        vy[1] <= vy[2];  vy[2] <= vy[1];
                        vz[1] <= vz[2];  vz[2] <= vz[1];
                        va[1] <= va[2];  va[2] <= va[1];
                        area_neg <= t_area;
                    end else begin
                        area_neg <= -t_area;
                    end
                    state <= S_SETUP2;
                end
            end

            // ---------------- top-left bias, bounding box, start 1/area ----------------
            S_SETUP2: begin
                for (i = 0; i < 3; i = i + 1) begin
                    p1 = (i + 1) % 3;  p2 = (i + 2) % 3;     // edge i = (b,c), (c,a), (a,b)
                    tl[i] <= is_tl(vx[p1], vy[p1], vx[p2], vy[p2]);
                end
                t_min = vx[0];  t_max = vx[0];
                for (i = 1; i < 3; i = i + 1) begin
                    if (vx[i] < t_min) t_min = vx[i];
                    if (vx[i] > t_max) t_max = vx[i];
                end
                t_min = t_min >>> 4;  if (t_min < 0) t_min = 0;
                mnx <= t_min & ~32'sd1;
                t_max = (t_max >>> 4) + 1;  if (t_max > $signed({16'd0, width})) t_max = {16'd0, width};
                mxx <= t_max;
                qx  <= t_min & ~32'sd1;
                t_min = vy[0];  t_max = vy[0];
                for (i = 1; i < 3; i = i + 1) begin
                    if (vy[i] < t_min) t_min = vy[i];
                    if (vy[i] > t_max) t_max = vy[i];
                end
                t_min = t_min >>> 4;  if (t_min < 0) t_min = 0;
                mny <= t_min & ~32'sd1;
                t_max = (t_max >>> 4) + 1;  if (t_max > $signed({16'd0, height})) t_max = {16'd0, height};
                mxy <= t_max;
                qy  <= t_min & ~32'sd1;
                div_rem <= 69'd0;  div_quo <= 49'd0;  div_first <= 1'b1;
                div_cnt <= 7'd49;  divisor <= area_neg;
                state <= S_DIVAREA;
            end

            // ---------------- 1/area: one quotient bit per cycle ----------------
            S_DIVAREA: begin
                div_rem <= rem_next;  div_quo <= quo_next;  div_first <= 1'b0;
                div_cnt <= div_cnt - 7'd1;
                if (div_cnt == 7'd1) begin
                    inv_area <= quo_next;
                    state <= (mny >= mxy || mnx >= mxx) ? S_IDLE : S_EVAL;
                end
            end

            // ---------------- coverage of this quad ----------------
            S_EVAL: begin
                t_cov = 4'd0;
                for (l = 0; l < 4; l = l + 1) begin
                    t_x  = qx + (l % 2);
                    t_y  = qy + (l / 2);
                    t_px = (t_x <<< 4) + 8;                  // pixel center, 12.4
                    t_py = (t_y <<< 4) + 8;
                    t_ok = 1'b1;
                    for (i = 0; i < 3; i = i + 1) begin
                        p1 = (i + 1) % 3;  p2 = (i + 2) % 3;
                        t_e = -edgef(vx[p1], vy[p1], vx[p2], vy[p2], t_px, t_py);
                        E[l*3 + i] <= t_e;
                        if ((t_e + (tl[i] ? 0 : -1)) < 0) t_ok = 1'b0;
                    end
                    t_cov[l] = t_ok && (t_x < $signed({16'd0, width})) && (t_y < $signed({16'd0, height}));
                end
                if (t_cov == 4'd0) begin
                    state <= S_NEXT;
                end else begin
                    covmask <= t_cov;
                    stat_quads <= stat_quads + 32'd1;
                    state <= S_WEIGHT;
                end
            end

            // ---------------- weights and closeness, all 4 lanes ----------------
            S_WEIGHT: begin
                for (l = 0; l < 4; l = l + 1) begin
                    t_prod = E[l*3 + 0] * $signed({1'b0, inv_area});  t_w0 = t_prod >>> 32;
                    t_prod = E[l*3 + 1] * $signed({1'b0, inv_area});  t_w1 = t_prod >>> 32;
                    t_prod = E[l*3 + 2] * $signed({1'b0, inv_area});  t_w2 = t_prod >>> 32;
                    wgt[l*3 + 0] <= t_w0;  wgt[l*3 + 1] <= t_w1;  wgt[l*3 + 2] <= t_w2;
                    t_csum = t_w0 * $signed({1'b0, vz[0]}) + t_w1 * $signed({1'b0, vz[1]}) +
                             t_w2 * $signed({1'b0, vz[2]});
                    t_c = t_csum >>> 16;
                    if (t_c < 1)                           close[l] <= 32'd1;
                    else if (t_c > 105'sd4294967295)       close[l] <= 32'hFFFF_FFFF;
                    else                                   close[l] <= t_c[31:0];
                end
                lane  <= 2'd0;
                state <= S_DEPTH;
            end

            // ---------------- early depth test, lane by lane ----------------
            S_DEPTH: begin
                if (depth_mode == 2'd0 || !covmask[lane] || mem_gnt) begin
                    if (depth_mode != 2'd0 && covmask[lane] && (d_new <= d_stored))
                        covmask[lane] <= 1'b0;                 // hidden: never shaded
                    lane <= lane + 2'd1;
                    if (lane == 2'd3) state <= S_DCHK;
                end
            end

            S_DCHK: begin
                if (covmask == 4'd0) begin
                    stat_early_dropped <= stat_early_dropped + 32'd1;
                    state <= S_NEXT;
                end else begin
                    lane <= 2'd0;  rcp_phase <= 2'd0;  state <= S_RCP;
                end
            end

            // ---------------- 1 / closeness per lane (reciprocal unit) ----------------
            S_RCP: begin
                case (rcp_phase)
                2'd0: begin
                    if (lane_close == 32'd0) begin
                        rcp[lane] <= 32'h7FFF_FFFF;  rcp_phase <= 2'd2;
                    end else begin
                        rcp_neg <= lane_close[31];
                        divisor <= {36'd0, lane_close[31] ? (~lane_close + 32'd1) : lane_close};
                        div_rem <= 69'd0;  div_quo <= 49'd0;  div_first <= 1'b1;
                        div_cnt <= 7'd33;  rcp_phase <= 2'd1;
                    end
                end
                2'd1: begin
                    div_rem <= rem_next;  div_quo <= quo_next;  div_first <= 1'b0;
                    div_cnt <= div_cnt - 7'd1;
                    if (div_cnt == 7'd1) begin
                        rcp[lane] <= rcp_neg ? (~rcp_mag + 32'd1) : rcp_mag;
                        rcp_phase <= 2'd2;
                    end
                end
                default: begin
                    rcp_phase <= 2'd0;
                    lane <= lane + 2'd1;
                    if (lane == 2'd3) begin
                        k <= 4'd0;
                        quad_attr <= 1024'd0;
                        state <= S_ATTR;
                    end
                end
                endcase
            end

            // ---------------- perspective-correct attributes ----------------
            S_ATTR: begin
                if (k >= nat || k == 4'd8) begin
                    quad_valid <= 1'b1;
                    quad_prim  <= prim;
                    quad_x     <= qx[15:0];
                    quad_y     <= qy[15:0];
                    quad_mask  <= covmask;
                    quad_close <= {close[3], close[2], close[1], close[0]};
                    state <= S_OUT;
                end else begin
                    for (l = 0; l < 4; l = l + 1) begin
                        t_asum = wgt[l*3 + 0] * $signed(va[0][32*k +: 32]) +
                                 wgt[l*3 + 1] * $signed(va[1][32*k +: 32]) +
                                 wgt[l*3 + 2] * $signed(va[2][32*k +: 32]);
                        t_az  = t_asum >>> 16;
                        t_mul = $signed(t_az[31:0]) * $signed(rcp[l]);
                        quad_attr[128*k + 32*l +: 32] <= t_mul[47:16];
                    end
                    k <= k + 4'd1;
                end
            end

            // ---------------- hand the quad over ----------------
            S_OUT: if (quad_ready) begin
                quad_valid <= 1'b0;
                state <= S_NEXT;
            end

            // ---------------- next quad ----------------
            S_NEXT: begin
                if (qx + 2 >= mxx) begin
                    qx <= mnx;
                    if (qy + 2 >= mxy) state <= S_IDLE;
                    else begin qy <= qy + 2;  state <= S_EVAL; end
                end else begin
                    qx <= qx + 2;  state <= S_EVAL;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
