// tex_unit.v - texture unit (spec section 8.6)                          Phase 6 (early)
// Executes V_TEX for one wave: 4 lanes, each with its own (u, v) in 16.16.
// Must match texture_unit() in gpu_model.py exactly.
//
//   IDLE -> SETUP (pick mip level for the quad) -> for each lane:
//           LANE (texel position + blend fractions) -> FETCH (1 or 4 texel reads,
//           waits for the memory grant) -> BLEND -> ... -> DONE (result ready)
//
// TEXn_INFO bits: 3-0 size as log2, 7-4 mip levels, 8 RGB565, 9 clamp,
//                 10 bilinear, 11 mipmaps on.

`timescale 1ns/1ps

module tex_unit (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,          // 1 for one cycle: sample 4 lanes
    input  wire [1:0]   slot,           // texture slot 0-3
    input  wire         wave_type,      // 1 = pixel wave: mip level from the quad
    input  wire [127:0] u,              // 16.16 per lane: lane 3 | 2 | 1 | 0
    input  wire [127:0] v,
    input  wire [127:0] tex_addr,       // per slot: slot 3 | 2 | 1 | 0
    input  wire [127:0] tex_info,

    output wire         mem_req,        // reading a texel
    output wire [31:0]  mem_addr,
    input  wire         mem_gnt,        // the read happens this cycle (else: wait)
    input  wire [31:0]  mem_rdata,

    output reg          done,           // 1 for one cycle: result is ready
    output reg  [127:0] result,         // RGBA8 per lane
    output wire         busy
);

    localparam T_IDLE = 3'd0, T_SETUP = 3'd1, T_LANE = 3'd2, T_FETCH = 3'd3,
               T_BLEND = 3'd4, T_DONE = 3'd5;

    reg [2:0]  state;
    reg [31:0] r_u [0:3];
    reg [31:0] r_v [0:3];
    reg [31:0] r_base, r_info;
    reg        r_wave;
    reg [3:0]  r_sl;                   // size as log2 at the chosen mip level
    reg [31:0] r_lbase;                // address of the chosen mip level
    reg [1:0]  lane;
    reg [1:0]  tap;                    // which of the 4 texels (bilinear)
    reg signed [31:0] r_x0, r_y0, r_xp, r_yp;
    reg [7:0]  r_fx, r_fy;
    reg [31:0] texel [0:3];

    // ---------------- texture settings ----------------
    wire [3:0] size_log2 = r_info[3:0];
    wire [3:0] levels    = (r_info[7:4] == 4'd0) ? 4'd1 : r_info[7:4];
    wire       rgb565    = r_info[8];
    wire       clamp     = r_info[9];
    wire       bilinear  = r_info[10];
    wire       mipmaps   = r_info[11];

    // ---------------- mip level: one per quad ----------------
    function [32:0] absdiff;                   // |a - b| of two signed 32-bit numbers
        input [31:0] a;
        input [31:0] b;
        reg signed [32:0] d;
        begin
            d = $signed({a[31], a}) - $signed({b[31], b});
            absdiff = d[32] ? -d : d;
        end
    endfunction

    function [32:0] max2;
        input [32:0] a;
        input [32:0] b;
        max2 = (a > b) ? a : b;
    endfunction

    function [5:0] highest_bit;                // position of the highest 1 (= log2)
        input [32:0] x;
        integer i;
        begin
            highest_bit = 6'd0;
            for (i = 0; i < 33; i = i + 1)
                if (x[i]) highest_bit = i;
        end
    endfunction

    wire [32:0] d_max    = max2(max2(absdiff(r_u[1], r_u[0]), absdiff(r_v[1], r_v[0])),
                                max2(absdiff(r_u[2], r_u[0]), absdiff(r_v[2], r_v[0])));
    wire [48:0] d_shift  = {16'd0, d_max} << size_log2;
    wire [32:0] texels   = d_shift[48:16];                 // footprint in whole texels
    wire [5:0]  lvl_raw  = (texels == 33'd0) ? 6'd0 : highest_bit(texels);
    wire [5:0]  lvl_max  = ({2'd0, levels} - 6'd1 < {2'd0, size_log2}) ?
                           ({2'd0, levels} - 6'd1) : {2'd0, size_log2};
    wire [5:0]  lvl_clip = (lvl_raw < lvl_max) ? lvl_raw : lvl_max;
    wire [3:0]  level    = (mipmaps && r_wave) ? lvl_clip[3:0] : 4'd0;

    function [31:0] level_offset;              // bytes before mip level 'lvl'
        input [3:0] sl;
        input [3:0] lvl;
        input       two_bytes;
        integer i;
        begin
            level_offset = 32'd0;
            for (i = 0; i < 15; i = i + 1)
                if (i < lvl)
                    level_offset = level_offset +
                        ((32'd1 << (2 * (sl - i))) << (two_bytes ? 1 : 2));
        end
    endfunction

    // ---------------- this lane: texel position and blend fractions ----------------
    wire [31:0]        lane_u = r_u[lane];
    wire [31:0]        lane_v = r_v[lane];
    wire signed [63:0] uu     = {{32{lane_u[31]}}, lane_u};
    wire signed [63:0] vv     = {{32{lane_v[31]}}, lane_v};
    wire signed [63:0] tx     = (uu <<< r_sl) - 64'sd32768;     // measured from texel centers
    wire signed [63:0] ty     = (vv <<< r_sl) - 64'sd32768;
    wire signed [63:0] x0_64  = tx >>> 16;
    wire signed [63:0] y0_64  = ty >>> 16;
    wire signed [63:0] xp_64  = (uu <<< r_sl) >>> 16;          // point sampling
    wire signed [63:0] yp_64  = (vv <<< r_sl) >>> 16;

    // ---------------- texel address for the current tap ----------------
    function [31:0] wrap_or_clamp;
        input signed [31:0] c;
        input [3:0]  sl;
        input        do_clamp;
        reg   [31:0] size;
        begin
            size = 32'd1 << sl;
            if (do_clamp) begin
                if (c < 0)                            wrap_or_clamp = 32'd0;
                else if (c > $signed(size - 32'd1))  wrap_or_clamp = size - 32'd1;
                else                                  wrap_or_clamp = c;
            end else begin
                wrap_or_clamp = c & (size - 32'd1);   // repeat (size is a power of two)
            end
        end
    endfunction

    wire signed [31:0] cx = !bilinear ? r_xp : (r_x0 + (tap[0] ? 32'sd1 : 32'sd0));
    wire signed [31:0] cy = !bilinear ? r_yp : (r_y0 + (tap[1] ? 32'sd1 : 32'sd0));
    wire [31:0] wx        = wrap_or_clamp(cx, r_sl, clamp);
    wire [31:0] wy        = wrap_or_clamp(cy, r_sl, clamp);
    wire [31:0] index     = (wy << r_sl) | wx;
    wire [31:0] byte_addr = r_lbase + (rgb565 ? (index << 1) : (index << 2));

    assign mem_req  = (state == T_FETCH);
    assign mem_addr = {byte_addr[31:2], 2'b00};
    assign busy     = (state != T_IDLE);

    // RGB565 -> RGBA8 (R = bits 7-0)
    wire [15:0] half    = byte_addr[1] ? mem_rdata[31:16] : mem_rdata[15:0];
    wire [31:0] from565 = {8'hFF, half[4:0], half[4:2], half[10:5], half[10:9],
                           half[15:11], half[15:13]};
    wire [31:0] texel_in = rgb565 ? from565 : mem_rdata;

    // ---------------- bilinear blend ----------------
    function [7:0] lerp_ch;                    // (x * (256 - f) + y * f) >> 8
        input [7:0] x;
        input [7:0] y;
        input [7:0] f;
        reg [16:0] s;
        begin
            s = x * (9'd256 - f) + y * f;
            lerp_ch = s[15:8];
        end
    endfunction

    function [31:0] lerp32;
        input [31:0] a;
        input [31:0] b;
        input [7:0]  f;
        lerp32 = {lerp_ch(a[31:24], b[31:24], f), lerp_ch(a[23:16], b[23:16], f),
                  lerp_ch(a[15:8],  b[15:8],  f), lerp_ch(a[7:0],   b[7:0],   f)};
    endfunction

    wire [31:0] top    = lerp32(texel[0], texel[1], r_fx);
    wire [31:0] bottom = lerp32(texel[2], texel[3], r_fx);
    wire [31:0] mixed  = lerp32(top, bottom, r_fy);

    // ---------------- state machine ----------------
    integer k;
    always @(posedge clk) begin
        if (rst) begin
            state <= T_IDLE;  done <= 1'b0;  result <= 128'd0;
            r_base <= 32'd0;  r_info <= 32'd0;  r_wave <= 1'b0;  r_sl <= 4'd0;  r_lbase <= 32'd0;
            lane <= 2'd0;  tap <= 2'd0;
            r_x0 <= 32'sd0;  r_y0 <= 32'sd0;  r_xp <= 32'sd0;  r_yp <= 32'sd0;
            r_fx <= 8'd0;  r_fy <= 8'd0;
            for (k = 0; k < 4; k = k + 1) begin
                r_u[k] <= 32'd0;  r_v[k] <= 32'd0;  texel[k] <= 32'd0;
            end
        end else begin
            done <= 1'b0;
            case (state)
            T_IDLE: if (start) begin
                for (k = 0; k < 4; k = k + 1) begin
                    r_u[k] <= u[32*k +: 32];
                    r_v[k] <= v[32*k +: 32];
                end
                r_base <= tex_addr[32*slot +: 32];
                r_info <= tex_info[32*slot +: 32];
                r_wave <= wave_type;
                state  <= T_SETUP;
            end

            T_SETUP: begin                     // one mip level for the whole quad
                r_sl    <= size_log2 - level;
                r_lbase <= r_base + level_offset(size_log2, level, rgb565);
                lane    <= 2'd0;
                state   <= T_LANE;
            end

            T_LANE: begin
                r_x0 <= x0_64[31:0];  r_y0 <= y0_64[31:0];
                r_xp <= xp_64[31:0];  r_yp <= yp_64[31:0];
                r_fx <= tx[15:8];     r_fy <= ty[15:8];
                tap   <= 2'd0;
                state <= T_FETCH;
            end

            T_FETCH: if (mem_gnt) begin        // else: wait for memory
                texel[tap] <= texel_in;
                if (!bilinear || tap == 2'd3) state <= T_BLEND;
                else                          tap   <= tap + 2'd1;
            end

            T_BLEND: begin
                result[32*lane +: 32] <= bilinear ? mixed : texel[0];
                lane <= lane + 2'd1;
                state <= (lane == 2'd3) ? T_DONE : T_LANE;
            end

            T_DONE: begin
                done  <= 1'b1;
                state <= T_IDLE;
            end

            default: state <= T_IDLE;
            endcase
        end
    end

endmodule
