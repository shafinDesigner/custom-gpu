// prim_unit.v - primitive unit (spec section 8.7)                    Phase 5, stage 3
// Must match primitive_unit() in gpu_model.py exactly.
//
// For each triangle (3 shaded vertices, camera space 16.16):
//   NEAR   any z < NEAR_Z -> discard the triangle
//   RCP    1/z per vertex (exact bit-serial divider, 33 cycles each)
//   PROJ   screen x = CX + (x * FOCAL) * 1/z,  y = CY - (y * FOCAL) * 1/z  -> 12.4
//          attribute x 1/z for perspective-correct interpolation
//   AREA   zero area -> discard; back / front faces culled per CULL_MODE
//   OUT    hand the triangle to the rasterizer with its primitive number

`timescale 1ns/1ps

module prim_unit (
    input  wire          clk,
    input  wire          rst,
    input  wire          draw_start,      // 1 for one cycle: primitive numbers restart at 0

    // pipeline state
    input  wire [31:0]   proj_cx,         // 16.16
    input  wire [31:0]   proj_cy,
    input  wire [31:0]   proj_focal,
    input  wire [31:0]   near_z,
    input  wire [1:0]    cull_mode,       // 0 none, 1 back faces, 2 front faces

    // triangle in: 3 shaded vertices, vertex c | b | a
    input  wire          in_valid,
    output wire          in_ready,
    input  wire [95:0]   in_x,
    input  wire [95:0]   in_y,
    input  wire [95:0]   in_z,
    input  wire [767:0]  in_attr,         // 8 per vertex (attr 7 | ... | 0)
    input  wire [3:0]    n_attrs,

    // triangle out (to the rasterizer)
    output reg           out_valid,
    input  wire          out_ready,
    output reg  [31:0]   out_prim,
    output reg  [95:0]   out_x,           // 12.4
    output reg  [95:0]   out_y,
    output reg  [95:0]   out_z,           // 1/z (0.16)
    output reg  [767:0]  out_attr,        // attr x 1/z
    output reg  [3:0]    out_nattrs,

    output wire          busy,
    output reg  [31:0]   st_in,
    output reg  [31:0]   st_near,
    output reg  [31:0]   st_zero,
    output reg  [31:0]   st_back,
    output reg  [31:0]   st_out
);

    localparam P_IDLE = 3'd0, P_NEAR = 3'd1, P_RCP = 3'd2, P_PROJ = 3'd3,
               P_AREA = 3'd4, P_OUT = 3'd5;

    reg [2:0]   state;
    reg [31:0]  vx [0:2];
    reg [31:0]  vy [0:2];
    reg [31:0]  vz [0:2];
    reg [255:0] va [0:2];
    reg [3:0]   nat;
    reg [1:0]   v;                          // vertex being projected
    reg [31:0]  inv_z;
    reg [31:0]  prim_count;

    assign in_ready = (state == P_IDLE) && !out_valid;
    assign busy     = (state != P_IDLE) || out_valid;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    function [31:0] mulfx32;                 // ((a * b) >> 16), kept to 32 bits
        input [31:0] a;
        input [31:0] b;
        reg signed [63:0] p;
        begin
            p = $signed(a) * $signed(b);
            mulfx32 = p[47:16];
        end
    endfunction

    function signed [67:0] edgef;
        input signed [31:0] x1, y1, x2, y2, px, py;
        reg signed [67:0] dx, dy, ex, ey;
        begin
            dx = x2 - x1;  dy = y2 - y1;  ex = px - x1;  ey = py - y1;
            edgef = dx * ey - dy * ex;
        end
    endfunction

    // ------------------------------------------------------------------
    // Bit-serial reciprocal: 2^32 / |z| with the same rules as V_RCP
    // ------------------------------------------------------------------
    reg  [32:0] div_rem;
    reg  [32:0] div_quo;
    reg         div_first, rcp_neg;
    reg  [5:0]  div_cnt;
    reg  [31:0] divisor;
    reg  [1:0]  rcp_phase;
    wire [32:0] rem_shift = {div_rem[31:0], div_first};
    wire        div_ge    = (rem_shift >= {1'b0, divisor});
    wire [32:0] rem_next  = div_ge ? (rem_shift - {1'b0, divisor}) : rem_shift;
    wire [32:0] quo_next  = {div_quo[31:0], div_ge};
    wire [31:0] rcp_mag   = (quo_next > 33'h0_7FFF_FFFF) ? 32'h7FFF_FFFF : quo_next[31:0];
    wire [31:0] cur_z     = vz[v];

    // ------------------------------------------------------------------
    // Projection of vertex v (uses inv_z)
    // ------------------------------------------------------------------
    wire [31:0] sx = proj_cx + mulfx32(mulfx32(vx[v], proj_focal), inv_z);
    wire [31:0] sy = proj_cy - mulfx32(mulfx32(vy[v], proj_focal), inv_z);

    wire signed [67:0] area = edgef(out_x[31:0], out_y[31:0], out_x[63:32], out_y[63:32],
                                    out_x[95:64], out_y[95:64]);

    integer i, k;
    always @(posedge clk) begin
        if (rst) begin
            state <= P_IDLE;  out_valid <= 1'b0;  prim_count <= 32'd0;  v <= 2'd0;  inv_z <= 32'd0;
            out_prim <= 32'd0;  out_x <= 96'd0;  out_y <= 96'd0;  out_z <= 96'd0;
            out_attr <= 768'd0;  out_nattrs <= 4'd0;  nat <= 4'd0;
            st_in <= 32'd0;  st_near <= 32'd0;  st_zero <= 32'd0;  st_back <= 32'd0;  st_out <= 32'd0;
            div_rem <= 33'd0;  div_quo <= 33'd0;  div_first <= 1'b0;  div_cnt <= 6'd0;
            divisor <= 32'd0;  rcp_phase <= 2'd0;  rcp_neg <= 1'b0;
        end else begin
            if (draw_start) prim_count <= 32'd0;

            case (state)
            P_IDLE: if (out_valid) begin
                if (out_ready) out_valid <= 1'b0;          // the rasterizer took it
            end else if (in_valid) begin
                for (i = 0; i < 3; i = i + 1) begin
                    vx[i] <= in_x[32*i +: 32];
                    vy[i] <= in_y[32*i +: 32];
                    vz[i] <= in_z[32*i +: 32];
                    va[i] <= in_attr[256*i +: 256];
                end
                nat   <= n_attrs;
                st_in <= st_in + 32'd1;
                state <= P_NEAR;
            end

            // ---------------- near plane ----------------
            P_NEAR: begin
                if ($signed(vz[0]) < $signed(near_z) || $signed(vz[1]) < $signed(near_z) ||
                    $signed(vz[2]) < $signed(near_z)) begin
                    st_near <= st_near + 32'd1;
                    state   <= P_IDLE;
                end else begin
                    v <= 2'd0;  rcp_phase <= 2'd0;  out_attr <= 768'd0;
                    state <= P_RCP;
                end
            end

            // ---------------- 1/z of vertex v ----------------
            P_RCP: begin
                case (rcp_phase)
                2'd0: begin
                    if (cur_z == 32'd0) begin
                        inv_z <= 32'h7FFF_FFFF;  state <= P_PROJ;
                    end else begin
                        rcp_neg <= cur_z[31];
                        divisor <= cur_z[31] ? (~cur_z + 32'd1) : cur_z;
                        div_rem <= 33'd0;  div_quo <= 33'd0;  div_first <= 1'b1;
                        div_cnt <= 6'd33;  rcp_phase <= 2'd1;
                    end
                end
                default: begin
                    div_rem <= rem_next;  div_quo <= quo_next;  div_first <= 1'b0;
                    div_cnt <= div_cnt - 6'd1;
                    if (div_cnt == 6'd1) begin
                        inv_z <= rcp_neg ? (~rcp_mag + 32'd1) : rcp_mag;
                        rcp_phase <= 2'd0;
                        state <= P_PROJ;
                    end
                end
                endcase
            end

            // ---------------- project vertex v ----------------
            P_PROJ: begin
                out_x[32*v +: 32] <= $signed(sx) >>> 12;
                out_y[32*v +: 32] <= $signed(sy) >>> 12;
                out_z[32*v +: 32] <= inv_z;
                for (k = 0; k < 8; k = k + 1)
                    if (k < nat)
                        out_attr[256*v + 32*k +: 32] <= mulfx32(va[v][32*k +: 32], inv_z);
                v <= v + 2'd1;
                state <= (v == 2'd2) ? P_AREA : P_RCP;
            end

            // ---------------- zero area and culling ----------------
            P_AREA: begin
                if (area == 0) begin
                    st_zero <= st_zero + 32'd1;
                    state   <= P_IDLE;
                end else if ((cull_mode == 2'd1 && area > 0) || (cull_mode == 2'd2 && area < 0)) begin
                    st_back <= st_back + 32'd1;
                    state   <= P_IDLE;
                end else begin
                    out_prim   <= prim_count;
                    out_nattrs <= nat;
                    prim_count <= prim_count + 32'd1;
                    st_out     <= st_out + 32'd1;
                    out_valid  <= 1'b1;
                    state      <= P_IDLE;                    // waits there until taken
                end
            end

            default: state <= P_IDLE;
            endcase
        end
    end

endmodule
