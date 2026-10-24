// cu.v - Compute unit, version 4 (non-pipelined)                       Phase 3 + 6 + 5
// Follows spec sections 5 and 6. One wave = 4 lanes.
//
// State machine:
//     FETCH -> EXEC -> FETCH ...
//     vector loads/stores : EXEC -> MEM (one lane per cycle)              -> FETCH
//     V_RCP / V_RSQ       : EXEC -> SFU (exact, bit-by-bit, lane by lane) -> FETCH
//     V_TEX               : EXEC -> TEX (wait for this CU's texture unit)  -> FETCH
//
// Must match simulator.py exactly (registers, mask, memory, exports, instruction count).
// Version 1 adds: special function unit (V_RCP, V_RSQ), color instructions
// (V_CMUL, V_CADD, V_PACK, V_UNPK) and the EXPORT port.
// Version 2 adds (for sharing with a second CU):
//   - dmem_req / dmem_gnt: ask for the data memory, WAIT until granted
//   - init_*: starting register values loaded at start (pixel x, y, constants)
//     All other registers start at 0, like a fresh wave in simulator.py.
// Version 3 adds V_TEX: the CU sends u, v (4 lanes) and the slot to its texture
// unit (tex_unit.v), waits for tex_done, then writes the 4 colors to active lanes.
// ALL 52 instructions of ISA v0.1 are now implemented.
// Version 4: a wave starts with v0-v10 from init_v (pixel waves: x, y, closeness,
// ATTR0-7; vertex waves: input attributes) and with the mask from init_mask.

`timescale 1ns/1ps

module cu (
    input  wire         clk,
    input  wire         rst,          // synchronous reset, active high
    input  wire         start,        // 1 for one cycle: start running at start_pc
    input  wire [31:0]  start_pc,
    input  wire [3:0]   cu_id,        // read by S_SPR 0
    input  wire         wave_type,    // read by S_SPR 1 (0 vertex, 1 pixel)
    input  wire [31:0]  frame,        // read by S_SPR 2
    input  wire [1407:0] init_v,      // v0-v10 at start: v_k at [128k +: 128], lane 3 | 2 | 1 | 0
    input  wire [3:0]   init_mask,    // execution mask at start
    input  wire [31:0]  init_s1,      // s1 at start (constant buffer address)
    input  wire [31:0]  init_s2,      // s2 at start (wave / primitive number)

    // instruction port: combinational read
    output wire [31:0]  imem_addr,
    input  wire [31:0]  imem_data,

    // data port: combinational read, write on the clock edge when dmem_we = 1
    output wire [31:0]  dmem_addr,
    output wire [31:0]  dmem_wdata,
    output wire         dmem_we,
    input  wire [31:0]  dmem_rdata,
    output wire         dmem_req,       // this CU wants the data memory now
    input  wire         dmem_gnt,       // ... and gets it this cycle (else: wait)

    // export port: valid for one cycle per EXPORT instruction
    output wire         export_valid,
    output wire [7:0]   export_target,  // 0-2 POS_X/Y/Z, 8-15 ATTR0-7, 16 COLOR
    output wire [3:0]   export_mask,    // which lanes are active
    output wire [127:0] export_data,    // lane 3 | lane 2 | lane 1 | lane 0

    // texture port: to this CU's texture unit
    output wire         tex_start,      // 1 for one cycle when V_TEX executes
    output wire [1:0]   tex_slot,
    output wire [127:0] tex_u,          // lane 3 | lane 2 | lane 1 | lane 0
    output wire [127:0] tex_v,
    input  wire         tex_done,       // 1 for one cycle: tex_result is ready
    input  wire [127:0] tex_result,

    // status
    output reg          busy,
    output reg          done,
    output reg          error,
    output reg  [31:0]  instr_count
);

    // ------------------------------------------------------------------
    // Opcodes (spec section 6)
    // ------------------------------------------------------------------
    localparam OP_NOP    = 6'h00, OP_HALT  = 6'h01, OP_JMP   = 6'h02, OP_BEQ   = 6'h03,
               OP_BNE    = 6'h04, OP_BLT   = 6'h05, OP_SPR   = 6'h06, OP_KILL  = 6'h07,
               OP_ADDI   = 6'h13, OP_ORI   = 6'h14, OP_LUI   = 6'h15,
               OP_SLOAD  = 6'h18, OP_SSTORE= 6'h19,
               OP_MADFX  = 6'h24, OP_SEL   = 6'h2E, OP_VADDI = 6'h2F, OP_BCAST = 6'h30,
               OP_LANEID = 6'h31, OP_RCP   = 6'h34, OP_RSQ   = 6'h35,
               OP_VLOAD  = 6'h38, OP_VSTORE= 6'h39, OP_TEX   = 6'h3A,
               OP_CMUL   = 6'h3B, OP_CADD  = 6'h3C, OP_PACK  = 6'h3D, OP_UNPK  = 6'h3E,
               OP_EXPORT = 6'h3F;

    localparam ST_IDLE = 3'd0, ST_FETCH = 3'd1, ST_EXEC = 3'd2, ST_MEM = 3'd3, ST_SFU = 3'd4,
               ST_TEX  = 3'd5;

    // ------------------------------------------------------------------
    // Architectural state
    // ------------------------------------------------------------------
    reg [31:0] sreg [0:15];          // scalar registers s0-s15 (s0 always 0)
    reg [31:0] vreg [0:63];          // vector registers: index = {register, lane}
    reg [3:0]  mask;                 // execution mask, bit = lane
    reg [31:0] pc;
    reg [31:0] ir;                   // current instruction
    reg [2:0]  state;
    reg [1:0]  lane;                 // lane counter for vector memory access

    // ------------------------------------------------------------------
    // Decode (spec section 6 formats)
    // ------------------------------------------------------------------
    wire [5:0]  op    = ir[31:26];
    wire [3:0]  rd    = ir[25:22];
    wire [3:0]  ra    = ir[21:18];
    wire [3:0]  rb    = ir[17:14];
    wire        bs    = ir[13];
    wire [3:0]  rc    = ir[12:9];
    wire [31:0] imm18 = {{14{ir[17]}}, ir[17:0]};
    wire [31:0] uimm  = {14'd0, ir[17:0]};
    wire [31:0] off14 = {{18{ir[13]}}, ir[13:0]};
    wire [31:0] off26 = {{6{ir[25]}}, ir[25:0]};

    wire [31:0] s_a = sreg[ra];
    wire [31:0] s_b = sreg[rb];
    wire [31:0] s_d = sreg[rd];

    wire [31:0] pc_next   = pc + 32'd4;
    wire [31:0] pc_branch = pc + 32'd4 + (off14 << 2);
    wire [31:0] pc_jump   = pc + 32'd4 + (off26 << 2);

    // ------------------------------------------------------------------
    // Memory ports
    // ------------------------------------------------------------------
    wire [31:0] s_mem_addr = s_a + imm18;                       // S_LOAD / S_STORE
    wire [31:0] v_mem_addr = vreg[{ra, lane}] + imm18;          // V_LOAD / V_STORE, this lane
    wire [31:0] v_mem_data = vreg[{rd, lane}];

    assign imem_addr  = pc;
    assign dmem_addr  = (state == ST_MEM) ? v_mem_addr : s_mem_addr;
    assign dmem_wdata = (state == ST_MEM) ? v_mem_data : s_d;
    assign dmem_req   = ((state == ST_EXEC) && ((op == OP_SLOAD) || (op == OP_SSTORE))) ||
                        ((state == ST_MEM)  && mask[lane]);
    assign dmem_we    = dmem_gnt && (dmem_addr[1:0] == 2'b00) &&
                        (((state == ST_EXEC) && (op == OP_SSTORE)) ||
                         ((state == ST_MEM)  && (op == OP_VSTORE) && mask[lane]));

    // ------------------------------------------------------------------
    // Export port
    // ------------------------------------------------------------------
    assign export_valid  = (state == ST_EXEC) && (op == OP_EXPORT);
    assign export_target = ir[7:0];
    assign export_mask   = mask;
    assign export_data   = {vreg[{ra, 2'd3}], vreg[{ra, 2'd2}], vreg[{ra, 2'd1}], vreg[{ra, 2'd0}]};

    // ------------------------------------------------------------------
    // Texture port:  V_TEX vd, va (u), vb (v), slot (rc)
    // ------------------------------------------------------------------
    assign tex_start = (state == ST_EXEC) && (op == OP_TEX);
    assign tex_slot  = rc[1:0];
    assign tex_u     = {vreg[{ra, 2'd3}], vreg[{ra, 2'd2}], vreg[{ra, 2'd1}], vreg[{ra, 2'd0}]};
    assign tex_v     = {vreg[{rb, 2'd3}], vreg[{rb, 2'd2}], vreg[{rb, 2'd1}], vreg[{rb, 2'd0}]};

    // ------------------------------------------------------------------
    // ALU: shared by scalar (0x08-0x12) and vector (0x20-0x2D) opcodes
    // ------------------------------------------------------------------
    function [31:0] alu;
        input [5:0]  f_op;
        input [31:0] a;
        input [31:0] b;
        reg signed [63:0] prod;
        begin
            prod = $signed(a) * $signed(b);
            case (f_op)
                6'h08, 6'h20: alu = a + b;                                  // ADD
                6'h09, 6'h21: alu = a - b;                                  // SUB
                6'h0A, 6'h27: alu = a & b;                                  // AND
                6'h0B, 6'h28: alu = a | b;                                  // OR
                6'h0C, 6'h29: alu = a ^ b;                                  // XOR
                6'h0D, 6'h2A: alu = a << b[4:0];                            // SHL
                6'h0E, 6'h2B: alu = a >> b[4:0];                            // SHR (logical)
                6'h0F, 6'h2C: alu = $signed(a) >>> b[4:0];                  // SRA (arithmetic)
                6'h10, 6'h22: alu = prod[31:0];                             // MUL (low 32 bits)
                6'h11, 6'h23: alu = prod[47:16];                            // MULFX (16.16)
                6'h12, 6'h2D: alu = ($signed(a) < $signed(b)) ? 32'd1 : 32'd0;  // SLT
                6'h25:        alu = ($signed(a) < $signed(b)) ? a : b;      // V_MIN
                6'h26:        alu = ($signed(a) > $signed(b)) ? a : b;      // V_MAX
                default:      alu = 32'd0;
            endcase
        end
    endfunction

    function is_scalar_alu;  input [5:0] f_op; is_scalar_alu = (f_op >= 6'h08 && f_op <= 6'h12); endfunction
    function is_vector_alu;  input [5:0] f_op;
        is_vector_alu = (f_op >= 6'h20 && f_op <= 6'h2D) && (f_op != 6'h24);
    endfunction

    // ------------------------------------------------------------------
    // Color helpers. RGBA8: R = bits 7-0, G = 15-8, B = 23-16, A = 31-24
    // ------------------------------------------------------------------
    function [7:0] ch_mul;                    // (x * (y + 1)) >> 8: white keeps the color
        input [7:0] x;
        input [7:0] y;
        reg [16:0] p;
        begin
            p = x * ({1'b0, y} + 9'd1);
            ch_mul = p[15:8];
        end
    endfunction

    function [7:0] ch_add;                    // add, clamped at 255
        input [7:0] x;
        input [7:0] y;
        reg [8:0] s;
        begin
            s = x + y;
            ch_add = s[8] ? 8'hFF : s[7:0];
        end
    endfunction

    function [7:0] clamp255;                  // signed 32-bit -> 0..255
        input [31:0] x;
        begin
            if ($signed(x) < 0)        clamp255 = 8'd0;
            else if ($signed(x) > 255) clamp255 = 8'hFF;
            else                       clamp255 = x[7:0];
        end
    endfunction

    function [31:0] cmul;
        input [31:0] a;
        input [31:0] b;
        cmul = {ch_mul(a[31:24], b[31:24]), ch_mul(a[23:16], b[23:16]),
                ch_mul(a[15:8],  b[15:8]),  ch_mul(a[7:0],   b[7:0])};
    endfunction

    function [31:0] cadd;
        input [31:0] a;
        input [31:0] b;
        cadd = {ch_add(a[31:24], b[31:24]), ch_add(a[23:16], b[23:16]),
                ch_add(a[15:8],  b[15:8]),  ch_add(a[7:0],   b[7:0])};
    endfunction

    // ------------------------------------------------------------------
    // Special function unit (algorithms proven bit-exact in sfu_check.py)
    //   V_RCP: 2^32 / |x|       restoring divider, 33 cycles, then sign
    //   V_RSQ: sqrt(2^48 / x)   divider 49 cycles + square root 25 cycles
    // ------------------------------------------------------------------
    reg [1:0]  sfu_lane;
    reg [1:0]  sfu_phase;            // 0 setup, 1 divide, 2 square root, 3 write
    reg [5:0]  sfu_cnt;
    reg        sfu_first;            // the dividend is a 1 followed by zeros
    reg        sfu_neg;
    reg [31:0] sfu_div;
    reg [49:0] sfu_rem;
    reg [48:0] sfu_quo;
    reg [49:0] sq_n, sq_res, sq_bit;
    reg [31:0] sfu_result;

    wire [31:0] sfu_x         = vreg[{ra, sfu_lane}];
    wire [31:0] sfu_abs       = sfu_x[31] ? (~sfu_x + 32'd1) : sfu_x;
    wire [49:0] sfu_rem_shift = {sfu_rem[48:0], sfu_first};
    wire        sfu_ge        = (sfu_rem_shift >= {18'd0, sfu_div});
    wire [49:0] sfu_rem_next  = sfu_ge ? (sfu_rem_shift - {18'd0, sfu_div}) : sfu_rem_shift;
    wire [48:0] sfu_quo_next  = {sfu_quo[47:0], sfu_ge};
    wire [31:0] rcp_mag       = (sfu_quo_next > 49'h0_7FFF_FFFF) ? 32'h7FFF_FFFF : sfu_quo_next[31:0];
    wire [49:0] sq_trial      = sq_res + sq_bit;
    wire        sq_ge         = (sq_n >= sq_trial);
    wire [49:0] sq_res_next   = sq_ge ? ((sq_res >> 1) + sq_bit) : (sq_res >> 1);

    // ------------------------------------------------------------------
    // State machine
    // ------------------------------------------------------------------
    integer l;
    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;  busy <= 1'b0;  done <= 1'b0;  error <= 1'b0;
            instr_count <= 32'd0;  pc <= 32'd0;  ir <= 32'd0;  mask <= 4'hF;  lane <= 2'd0;
            sfu_lane <= 2'd0;  sfu_phase <= 2'd0;  sfu_cnt <= 6'd0;  sfu_first <= 1'b0;
            sfu_neg <= 1'b0;  sfu_div <= 32'd0;  sfu_rem <= 50'd0;  sfu_quo <= 49'd0;
            sq_n <= 50'd0;  sq_res <= 50'd0;  sq_bit <= 50'd0;  sfu_result <= 32'd0;
            for (l = 0; l < 16; l = l + 1) sreg[l] <= 32'd0;
            for (l = 0; l < 64; l = l + 1) vreg[l] <= 32'd0;
        end else begin
            case (state)
            // ---------------- wait for start ----------------
            ST_IDLE: begin
                if (start) begin
                    pc <= start_pc;  busy <= 1'b1;  done <= 1'b0;  error <= 1'b0;
                    mask <= init_mask;  instr_count <= 32'd0;  state <= ST_FETCH;
                    // fresh wave: every register 0, then the starting values
                    for (l = 0; l < 16; l = l + 1) sreg[l] <= 32'd0;
                    for (l = 0; l < 64; l = l + 1) vreg[l] <= 32'd0;
                    sreg[1] <= init_s1;
                    sreg[2] <= init_s2;
                    for (l = 0; l < 44; l = l + 1)      // v0-v10, 4 lanes each
                        vreg[l] <= init_v[32*l +: 32];
                end
            end

            // ---------------- fetch ----------------
            ST_FETCH: begin
                ir    <= imem_data;
                state <= ST_EXEC;
            end

            // ---------------- execute ----------------
            ST_EXEC: if (dmem_req && !dmem_gnt) begin
                // wait: the other CU is using the data memory this cycle
            end else begin
                pc          <= pc_next;
                state       <= ST_FETCH;
                instr_count <= instr_count + 32'd1;

                if (is_scalar_alu(op)) begin
                    if (rd != 4'd0) sreg[rd] <= alu(op, s_a, s_b);
                end else if (is_vector_alu(op)) begin
                    for (l = 0; l < 4; l = l + 1)
                        if (mask[l]) vreg[rd*4 + l] <= alu(op, vreg[ra*4 + l], bs ? s_b : vreg[rb*4 + l]);
                end else begin
                    case (op)
                    OP_NOP:  ;
                    OP_HALT: begin state <= ST_IDLE;  busy <= 1'b0;  done <= 1'b1; end
                    OP_JMP:  pc <= pc_jump;
                    OP_BEQ:  if (s_a == s_b)                   pc <= pc_branch;
                    OP_BNE:  if (s_a != s_b)                   pc <= pc_branch;
                    OP_BLT:  if ($signed(s_a) < $signed(s_b))  pc <= pc_branch;
                    OP_SPR:  if (rd != 4'd0) begin
                                 case (uimm)
                                     32'd0:   sreg[rd] <= {28'd0, cu_id};
                                     32'd1:   sreg[rd] <= {31'd0, wave_type};
                                     32'd2:   sreg[rd] <= frame;
                                     32'd3:   sreg[rd] <= 32'd4;
                                     default: sreg[rd] <= 32'd0;
                                 endcase
                             end
                    OP_KILL: for (l = 0; l < 4; l = l + 1)
                                 if (vreg[ra*4 + l] != 32'd0) mask[l] <= 1'b0;
                    OP_ADDI: if (rd != 4'd0) sreg[rd] <= s_a + imm18;
                    OP_ORI:  if (rd != 4'd0) sreg[rd] <= s_a | {16'd0, ir[15:0]};
                    OP_LUI:  if (rd != 4'd0) sreg[rd] <= {ir[15:0], 16'd0};
                    OP_SLOAD: begin
                                 if (dmem_addr[1:0] != 2'b00) begin
                                     error <= 1'b1;  busy <= 1'b0;  state <= ST_IDLE;
                                 end else if (rd != 4'd0) sreg[rd] <= dmem_rdata;
                             end
                    OP_SSTORE: begin
                                 if (dmem_addr[1:0] != 2'b00) begin
                                     error <= 1'b1;  busy <= 1'b0;  state <= ST_IDLE;
                                 end
                             end
                    OP_MADFX: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= vreg[rc*4 + l] +
                                     alu(6'h23, vreg[ra*4 + l], bs ? s_b : vreg[rb*4 + l]);
                    OP_SEL:  for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= (vreg[rc*4 + l] != 32'd0) ?
                                     vreg[ra*4 + l] : (bs ? s_b : vreg[rb*4 + l]);
                    OP_VADDI: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= vreg[ra*4 + l] + imm18;
                    OP_BCAST: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= s_a;
                    OP_LANEID: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= l;
                    OP_CMUL: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= cmul(vreg[ra*4 + l], vreg[rb*4 + l]);
                    OP_CADD: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= cadd(vreg[ra*4 + l], vreg[rb*4 + l]);
                    OP_PACK: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= {8'hFF, clamp255(vreg[rc*4 + l]),
                                     clamp255(vreg[rb*4 + l]), clamp255(vreg[ra*4 + l])};
                    OP_UNPK: for (l = 0; l < 4; l = l + 1)
                                 if (mask[l]) vreg[rd*4 + l] <= (vreg[ra*4 + l] >> {ir[1:0], 3'b000}) & 32'hFF;
                    OP_EXPORT: ;                         // the export port does the work
                    OP_RCP, OP_RSQ: begin
                                 pc        <= pc;        // pc moves on when the SFU finishes
                                 sfu_lane  <= 2'd0;
                                 sfu_phase <= 2'd0;
                                 state     <= ST_SFU;
                             end
                    OP_TEX:  begin
                                 pc    <= pc;            // pc moves on when the texture unit is done
                                 state <= ST_TEX;        // tex_start goes to the texture unit now
                             end
                    OP_VLOAD, OP_VSTORE: begin
                                 pc    <= pc;            // pc moves on when MEM finishes
                                 lane  <= 2'd0;
                                 state <= ST_MEM;
                             end
                    default: begin                       // reserved opcode
                                 error <= 1'b1;  busy <= 1'b0;  state <= ST_IDLE;
                             end
                    endcase
                end
            end

            // ---------------- vector memory: one lane per cycle ----------------
            ST_MEM: if (dmem_req && !dmem_gnt) begin
                // wait for the data memory
            end else begin
                if (mask[lane] && (dmem_addr[1:0] != 2'b00)) begin
                    error <= 1'b1;  busy <= 1'b0;  state <= ST_IDLE;
                end else begin
                    if (mask[lane] && (op == OP_VLOAD)) vreg[{rd, lane}] <= dmem_rdata;
                    lane <= lane + 2'd1;
                    if (lane == 2'd3) begin
                        pc    <= pc_next;
                        state <= ST_FETCH;
                    end
                end
            end

            // ---------------- special function unit: lane by lane ----------------
            ST_SFU: begin
                case (sfu_phase)
                2'd0: begin                                  // set up this lane
                    if (!mask[sfu_lane]) begin
                        sfu_phase <= 2'd3;                   // inactive lane: nothing to do
                    end else if (op == OP_RCP) begin
                        if (sfu_x == 32'd0) begin
                            sfu_result <= 32'h7FFF_FFFF;  sfu_phase <= 2'd3;
                        end else begin
                            sfu_neg <= sfu_x[31];  sfu_div <= sfu_abs;
                            sfu_rem <= 50'd0;  sfu_quo <= 49'd0;  sfu_first <= 1'b1;
                            sfu_cnt <= 6'd33;  sfu_phase <= 2'd1;
                        end
                    end else begin                           // OP_RSQ
                        if ($signed(sfu_x) <= 0) begin
                            sfu_result <= 32'h7FFF_FFFF;  sfu_phase <= 2'd3;
                        end else begin
                            sfu_neg <= 1'b0;  sfu_div <= sfu_x;
                            sfu_rem <= 50'd0;  sfu_quo <= 49'd0;  sfu_first <= 1'b1;
                            sfu_cnt <= 6'd49;  sfu_phase <= 2'd1;
                        end
                    end
                end
                2'd1: begin                                  // divider: one quotient bit per cycle
                    sfu_rem   <= sfu_rem_next;
                    sfu_quo   <= sfu_quo_next;
                    sfu_first <= 1'b0;
                    sfu_cnt   <= sfu_cnt - 6'd1;
                    if (sfu_cnt == 6'd1) begin
                        if (op == OP_RCP) begin
                            sfu_result <= sfu_neg ? (~rcp_mag + 32'd1) : rcp_mag;
                            sfu_phase  <= 2'd3;
                        end else begin
                            sq_n      <= {1'b0, sfu_quo_next};
                            sq_res    <= 50'd0;
                            sq_bit    <= 50'd1 << 48;
                            sfu_cnt   <= 6'd25;
                            sfu_phase <= 2'd2;
                        end
                    end
                end
                2'd2: begin                                  // square root: one result bit per cycle
                    if (sq_ge) sq_n <= sq_n - sq_trial;
                    sq_res  <= sq_res_next;
                    sq_bit  <= sq_bit >> 2;
                    sfu_cnt <= sfu_cnt - 6'd1;
                    if (sfu_cnt == 6'd1) begin
                        sfu_result <= (sq_res_next > 50'h7FFF_FFFF) ? 32'h7FFF_FFFF : sq_res_next[31:0];
                        sfu_phase  <= 2'd3;
                    end
                end
                2'd3: begin                                  // write this lane, then the next
                    if (mask[sfu_lane]) vreg[{rd, sfu_lane}] <= sfu_result;
                    sfu_phase <= 2'd0;
                    sfu_lane  <= sfu_lane + 2'd1;
                    if (sfu_lane == 2'd3) begin
                        pc    <= pc_next;
                        state <= ST_FETCH;
                    end
                end
                endcase
            end

            // ---------------- wait for the texture unit ----------------
            ST_TEX: begin
                if (tex_done) begin
                    for (l = 0; l < 4; l = l + 1)
                        if (mask[l]) vreg[rd*4 + l] <= tex_result[32*l +: 32];
                    pc    <= pc_next;
                    state <= ST_FETCH;
                end
            end

            default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
