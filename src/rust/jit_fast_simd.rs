// Fast page-batched SIMD/MMX/streaming loop compiler for tight single-BB flat
// 32-bit CPL3 loops under Vine common memory.
//
// Invariant: executes at most `min(budget_iters - 1, counter_iters)` full
// iterations inside a verified zero-fault, zero-page-crossing inner loop,
// leaving at least one iteration (or any boundary/faulting iteration) for the
// standard basic block epilogue, which preserves exact architectural flags,
// page faults, SMC barriers, and instruction budget accounting.
use crate::codegen;
use crate::cpu::global_pointers as gp;
use crate::cpu_context::CpuContext;
use crate::jit::{BasicBlock, BasicBlockType, JitContext};
use crate::modrm;
use crate::regs;
use crate::wasmgen::wasm_builder::{WasmBuilder, WasmLocal, WasmLocalV128};

const MAX_LOOP_OPS: usize = 48;
const MAX_STREAMS: usize = 6;
const MAX_INV_LOADS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Address {
    coefficients: [i32; 8],
    inv_coeffs: [i32; MAX_INV_LOADS],
    offset: i32,
}

impl Address {
    fn constant(offset: i32) -> Self {
        Self {
            coefficients: [0; 8],
            inv_coeffs: [0; MAX_INV_LOADS],
            offset,
        }
    }

    fn register(r: usize) -> Self {
        let mut a = Self::constant(0);
        a.coefficients[r] = 1;
        a
    }

    fn inv_load(slot: usize) -> Self {
        let mut a = Self::constant(0);
        a.inv_coeffs[slot] = 1;
        a
    }

    fn add_scaled(mut self, other: Self, scale: i32) -> Option<Self> {
        self.offset = self.offset.checked_add(other.offset.checked_mul(scale)?)?;
        for (a, b) in self.coefficients.iter_mut().zip(other.coefficients) {
            *a = a.checked_add(b.checked_mul(scale)?)?;
        }
        for (a, b) in self.inv_coeffs.iter_mut().zip(other.inv_coeffs) {
            *a = a.checked_add(b.checked_mul(scale)?)?;
        }
        Some(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct InvLoad {
    coefficients: [i32; 8],
    offset: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VecReg {
    Xmm(u8),
    Mm(u8),
}

#[derive(Clone, Copy)]
enum SimdBinOp {
    AddI8,
    AddI16,
    AddI32,
    AddI64,
    AddSatS8,
    AddSatS16,
    AddSatU8,
    AddSatU16,
    SubI8,
    SubI16,
    SubI32,
    SubI64,
    SubSatS8,
    SubSatS16,
    SubSatU8,
    SubSatU16,
    MulLoI16,
    MulHiS16,
    MulHiU16,
    Pmaddwd,
    And,
    AndNot,
    Or,
    Xor,
    EqI8,
    EqI16,
    EqI32,
    GtSI8,
    GtSI16,
    GtSI32,
    MinU8,
    MaxU8,
    MinS16,
    MaxS16,
    AvgU8,
    AvgU16,
    UnpackLo8,
    UnpackLo16,
    UnpackLo32,
    UnpackLo64,
    UnpackHi8,
    UnpackHi16,
    UnpackHi32,
    UnpackHi64,
    PackSSWB,
    PackUSWB,
    PackSSDW,
    AddF32,
    SubF32,
    MulF32,
    DivF32,
}

#[derive(Clone, Copy)]
enum ShiftKind {
    Shl16,
    ShrU16,
    ShrS16,
    Shl32,
    ShrU32,
    ShrS32,
    Shl64,
    ShrU64,
    ByteShr,
    ByteShl,
}

#[derive(Clone, Copy)]
enum ShuffleKind {
    Pshufw,
    Pshufd,
    Pshuflw,
    Pshufhw,
}

#[derive(Clone, Copy)]
struct MemRef {
    stream: usize,
    offset: i32,
}

#[derive(Clone, Copy)]
enum Op {
    Nop,
    // SIMD / MMX loads & stores
    Load128(u8, MemRef),
    Store128(MemRef, u8),
    Load64Zero(VecReg, MemRef),
    Store64(MemRef, VecReg),
    Load32Zero(VecReg, MemRef),
    Store32(MemRef, VecReg),
    // SIMD / MMX register moves & conversions
    MovVec(VecReg, VecReg),
    MovqXmmReg(u8, u8),
    Movq2dq(u8, u8),
    Movdq2q(u8, u8),
    MovdFromGpr(VecReg, u8),
    MovdToGpr(u8, VecReg),
    // SIMD / MMX compute
    BinReg(SimdBinOp, VecReg, VecReg),
    BinMem(SimdBinOp, VecReg, MemRef),
    ShiftImm(ShiftKind, VecReg, u8),
    ShuffleReg(ShuffleKind, VecReg, VecReg, u8),
    ShuffleMem(ShuffleKind, VecReg, MemRef, u8),
    // Integer ops
    GprLoad32(u8, MemRef),
    GprLoadInv32(u8, usize),
    GprLoadZx8(u8, MemRef),
    GprLoadZx16(u8, MemRef),
    GprStore32(MemRef, u8),
    GprMovReg(u8, u8),
    GprMovImm(u8, i32),
    GprAddImm(u8, i32),
    GprOrImm(u8, i32),
    GprAndImm(u8, i32),
    GprXorImm(u8, i32),
    GprAddReg(u8, u8),
    GprSubReg(u8, u8),
    GprXorReg(u8, u8),
    GprShlImm(u8, u8),
    GprShrUImm(u8, u8),
    GprShrSImm(u8, u8),
    GprLea(u8, Option<u8>, Option<u8>, u8, i32),
    // Induction/compare marker (no emission needed when deferred)
    DeferredInd(u8, i32),
    CmpMarker,
    // Head-of-loop skip test `test reg, imm32`: computes `reg & imm` for the
    // conditional body region (no emission in loops without a skip body).
    SkipTest(u8, i32),
    // Read-only table load `[base + index*scale + disp]`: the index is provably
    // bounded, so the whole region is guarded once instead of per batch.
    Load64TableZero(VecReg, usize, u8),
}

struct Stream {
    coefficients: [i32; 8],
    inv_coeffs: [i32; MAX_INV_LOADS],
    min_offset: i32,
    max_end: i64,
    is_write: bool,
    stride: i32,
    used: bool,
}

impl Stream {
    fn span_bytes(&self) -> u32 { (self.max_end - self.min_offset as i64) as u32 }
}

#[derive(Clone, Copy)]
enum FlagOp {
    Dec(u8),
    CmpReg(u8, u8),
    CmpImm(u8, i32),
    TestReg(u8, u8),
    Other,
}

#[derive(Clone, Copy)]
enum LoopCond {
    // dec cnt; jne -> runs `cnt.wrapping_sub(1)` fast iters
    DecJne(u8),
    // inc cnt; cmp cnt, lim_reg; jl -> runs `if cnt <s lim { (lim - cnt) as u32 - 1 } else { 0 }`
    IncCmpRegJl(u8, u8),
    // inc cnt; cmp cnt, lim_imm; jl
    IncCmpImmJl(u8, i32),
    // inc cnt; cmp cnt, lim_reg; jne -> `(lim - cnt).wrapping_sub(1)`
    IncCmpRegJne(u8, u8),
    // inc cnt; cmp cnt, lim_imm; jne
    IncCmpImmJne(u8, i32),
    // dec cnt; cmp cnt, lim_reg; jne -> `(cnt - lim).wrapping_sub(1)`
    DecCmpRegJne(u8, u8),
    // dec cnt; cmp cnt, lim_imm; jne
    DecCmpImmJne(u8, i32),
    // `dec cnt` outside the body, tail ends `test cnt, cnt; jne head`
    // -> same iteration semantics as DecJne (runs `cnt - 1` fast iters)
    DecTestJne(u8),
}

// Read-only table addressed by a provably bounded index: `[base + idx*scale +
// disp]` where `base` is a loop-invariant GPR and `idx < entries` because the
// most recent write to `idx` is `shr idx, imm` with imm >= 20.
struct TableRegion {
    base_reg: u8,
    disp: i32,
    shift: u8,
    entries: u32,
}

const MAX_TABLES: usize = 2;

struct FastSimdLoop {
    ops: Vec<Op>,
    streams: Vec<Stream>,
    inv_loads: Vec<InvLoad>,
    cond: LoopCond,
    deltas: [i32; 8],
    deferred_gpr_mask: u8,
    xmm_read_before_write: u8,
    xmm_written: u8,
    mm_read_before_write: u8,
    mm_written: u8,
    touched_mmx: bool,
    needs_cr0_check: bool,
    num_instructions: u32,
    // Head/body/tail skip loop: ops[body_start..body_end] are the conditional
    // body ops; body_instructions is the body's guest instruction count.
    body_start: usize,
    body_end: usize,
    body_instructions: u32,
    tables: Vec<TableRegion>,
    mmx_in_head_tail: bool,
}

fn op_uses_mmx(op: &Op) -> bool {
    let mm = |r: &VecReg| matches!(r, VecReg::Mm(_));
    match *op {
        Op::Load64Zero(d, _)
        | Op::Load32Zero(d, _)
        | Op::Load64TableZero(d, _, _)
        | Op::MovdFromGpr(d, _)
        | Op::ShiftImm(_, d, _)
        | Op::ShuffleMem(_, d, _, _)
        | Op::BinMem(_, d, _) => mm(&d),
        Op::Store64(_, s) | Op::Store32(_, s) | Op::MovdToGpr(_, s) => mm(&s),
        Op::MovVec(a, b) | Op::BinReg(_, a, b) | Op::ShuffleReg(_, a, b, _) => mm(&a) || mm(&b),
        Op::Movq2dq(..) | Op::Movdq2q(..) => true,
        _ => false,
    }
}

fn flat_address(
    base: Option<u32>,
    index: Option<u32>,
    shift: u8,
    offset: i32,
    registers: &[Option<Address>; 8],
) -> Option<Address> {
    let mut a = Address::constant(offset);
    if let Some(r) = base {
        a = a.add_scaled(registers[r as usize]?, 1)?;
    }
    if let Some(r) = index {
        a = a.add_scaled(registers[r as usize]?, 1 << shift)?;
    }
    Some(a)
}

fn effective_address(
    c: &mut CpuContext,
    byte: u8,
    registers: &[Option<Address>; 8],
) -> Option<Address> {
    let (base, index, shift, offset) = modrm::decode(c, byte).flat_parts()?;
    flat_address(base, index, shift, offset, registers)
}

fn add_stream_access(
    streams: &mut Vec<Stream>,
    addr: Address,
    bytes: u32,
    is_write: bool,
) -> Option<MemRef> {
    let end = addr.offset as i64 + bytes as i64;
    for (idx, s) in streams.iter_mut().enumerate() {
        if s.coefficients == addr.coefficients && s.inv_coeffs == addr.inv_coeffs {
            let new_min = s.min_offset.min(addr.offset);
            let new_max = s.max_end.max(end);
            if new_max - new_min as i64 <= 4096 {
                s.min_offset = new_min;
                s.max_end = new_max;
                s.is_write |= is_write;
                return Some(MemRef {
                    stream: idx,
                    offset: addr.offset,
                });
            }
        }
    }
    if streams.len() >= MAX_STREAMS {
        return None;
    }
    let idx = streams.len();
    streams.push(Stream {
        coefficients: addr.coefficients,
        inv_coeffs: addr.inv_coeffs,
        min_offset: addr.offset,
        max_end: end,
        is_write,
        stride: 0,
        used: true,
    });
    Some(MemRef {
        stream: idx,
        offset: addr.offset,
    })
}

fn decode_simd_binop(opcode: u8, is_sse: bool) -> Option<SimdBinOp> {
    use SimdBinOp::*;
    Some(match opcode {
        0x60 => UnpackLo8,
        0x61 => UnpackLo16,
        0x62 => UnpackLo32,
        0x63 => PackSSWB,
        0x64 => GtSI8,
        0x65 => GtSI16,
        0x66 => GtSI32,
        0x67 => PackUSWB,
        0x68 => UnpackHi8,
        0x69 => UnpackHi16,
        0x6A => UnpackHi32,
        0x6B => PackSSDW,
        0x6C if is_sse => UnpackLo64,
        0x6D if is_sse => UnpackHi64,
        0x74 => EqI8,
        0x75 => EqI16,
        0x76 => EqI32,
        0xD4 => AddI64,
        0xD5 => MulLoI16,
        0xD8 => SubSatU8,
        0xD9 => SubSatU16,
        0xDA => MinU8,
        0xDB => And,
        0xDC => AddSatU8,
        0xDD => AddSatU16,
        0xDE => MaxU8,
        0xDF => AndNot,
        0xE0 => AvgU8,
        0xE3 => AvgU16,
        0xE4 => MulHiU16,
        0xE5 => MulHiS16,
        0xE8 => SubSatS8,
        0xE9 => SubSatS16,
        0xEA => MinS16,
        0xEB => Or,
        0xEC => AddSatS8,
        0xED => AddSatS16,
        0xEE => MaxS16,
        0xEF => Xor,
        0xF5 => Pmaddwd,
        0xF8 => SubI8,
        0xF9 => SubI16,
        0xFA => SubI32,
        0xFB => SubI64,
        0xFC => AddI8,
        0xFD => AddI16,
        0xFE => AddI32,
        _ => return None,
    })
}

fn scan(
    cpu: &CpuContext,
    block: &BasicBlock,
    fallthrough_loop: Option<&BasicBlock>,
    skip_loop: Option<(&BasicBlock, &BasicBlock)>,
) -> Option<FastSimdLoop> {
    if !cpu.state_flags.is_32() || !cpu.has_flat_segmentation() || !cpu.cpl3() || block.has_sti {
        return None;
    }
    // A skip loop is three basic blocks: head `block` is a ConditionalJump whose
    // taken target is the tail and whose fallthrough is the body, the body is a
    // Normal block falling into the tail, and the tail is a ConditionalJump
    // back to the head. `skip_bounds` carries (body_addr, tail_addr).
    let (loop_last_addr, condition, num_instructions, skip_bounds) =
        if let Some((body, tail)) = skip_loop {
            if body.has_sti || tail.has_sti {
                return None;
            }
            let BasicBlockType::ConditionalJump {
                next_block_addr: Some(next),
                next_block_branch_taken_addr: Some(taken),
                condition: head_condition,
                ..
            } = block.ty
            else {
                return None;
            };
            // Only `je` (condition 0x4) may terminate the head; the terminating
            // `test r32, imm32` is validated while decoding.
            if next != body.addr
                || taken != tail.addr
                || !matches!(head_condition & 0xF0, 0x00 | 0x70 | 0x80)
                || head_condition & 0xF != 0x4
            {
                return None;
            }
            if !matches!(
                body.ty,
                BasicBlockType::Normal {
                    next_block_addr: Some(next),
                    jump_offset: 0,
                    ..
                } if next == tail.addr
            ) {
                return None;
            }
            let BasicBlockType::ConditionalJump {
                next_block_branch_taken_addr: Some(tail_taken),
                condition: tail_condition,
                ..
            } = tail.ty
            else {
                return None;
            };
            if tail_taken != block.addr {
                return None;
            }
            (
                tail.last_instruction_addr,
                tail_condition,
                block.number_of_instructions
                    + body.number_of_instructions
                    + tail.number_of_instructions,
                Some((body.addr, tail.addr)),
            )
        }
        else if let Some(nb) = fallthrough_loop {
            if nb.has_sti {
                return None;
            }
            let BasicBlockType::ConditionalJump {
                next_block_branch_taken_addr: Some(taken),
                condition,
                ..
            } = nb.ty
            else {
                return None;
            };
            if taken != block.addr {
                return None;
            }
            (
                nb.last_instruction_addr,
                condition,
                block.number_of_instructions + nb.number_of_instructions,
                None,
            )
        }
        else {
            let BasicBlockType::ConditionalJump {
                next_block_branch_taken_addr: Some(taken),
                condition,
                ..
            } = block.ty
            else {
                return None;
            };
            if taken != block.addr {
                return None;
            }
            (
                block.last_instruction_addr,
                condition,
                block.number_of_instructions,
                None,
            )
        };
    if num_instructions < 2 || num_instructions as usize > MAX_LOOP_OPS {
        return None;
    }
    if !matches!(condition & 0xF0, 0x00 | 0x70 | 0x80) {
        return None;
    }
    let cond_low = condition & 0xF;
    if cond_low != 0x5 && cond_low != 0xC {
        return None;
    }

    let mut c = cpu.clone();
    c.eip = block.addr;
    c.prefixes = 0;

    let mut registers: [Option<Address>; 8] = std::array::from_fn(|r| Some(Address::register(r)));
    let mut gpr_data_read_mask: u8 = 0;
    let mut gpr_written_non_ind_mask: u8 = 0;
    let mut gpr_ind_delta: [i32; 8] = [0; 8];
    let mut gpr_ind_touched: u8 = 0;

    let mut ops: Vec<Op> = Vec::with_capacity(num_instructions as usize);
    let mut streams: Vec<Stream> = Vec::new();
    let mut inv_candidates: Vec<InvLoad> = Vec::new();
    let mut tables: Vec<TableRegion> = Vec::new();

    // Skip-loop state: ops[body_start..body_end] are emitted inside a wasm `if`.
    let mut body_start = 0usize;
    let mut body_end = 0usize;
    let mut in_body = false;
    // Induction writes inside the conditional body cannot fold into batch
    // stream-stride/deferred-counter arithmetic.
    let mut gpr_ind_touched_in_body: u8 = 0;
    // Per-GPR bound proof: value is < 2^(32 - imm) after `shr reg, imm` (imm>=20).
    let mut bounded_shr = [0u8; 8];

    let mut xmm_read_before_write: u8 = 0;
    let mut xmm_written: u8 = 0;
    let mut mm_read_before_write: u8 = 0;
    let mut mm_written: u8 = 0;
    let mut touched_mmx = false;
    let mut needs_cr0_check = false;
    let mut has_simd_or_stream = false;
    let mut last_flag_op: Option<FlagOp> = None;

    let mark_vec_read = |reg: VecReg,
                         xmm_rbw: &mut u8,
                         xmm_w: u8,
                         mm_rbw: &mut u8,
                         mm_w: u8,
                         touched_mmx: &mut bool,
                         needs_cr0: &mut bool| {
        *needs_cr0 = true;
        match reg {
            VecReg::Xmm(r) => {
                if xmm_w & (1 << r) == 0 {
                    *xmm_rbw |= 1 << r;
                }
            },
            VecReg::Mm(r) => {
                *touched_mmx = true;
                if mm_w & (1 << r) == 0 {
                    *mm_rbw |= 1 << r;
                }
            },
        }
    };
    let mark_vec_write = |reg: VecReg,
                          xmm_w: &mut u8,
                          mm_w: &mut u8,
                          touched_mmx: &mut bool,
                          needs_cr0: &mut bool| {
        *needs_cr0 = true;
        match reg {
            VecReg::Xmm(r) => *xmm_w |= 1 << r,
            VecReg::Mm(r) => {
                *touched_mmx = true;
                *mm_w |= 1 << r;
            },
        }
    };

    while c.eip < loop_last_addr {
        if let Some((body_addr, tail_addr)) = skip_bounds {
            if c.eip == block.last_instruction_addr {
                // The head's `je` terminator: consume its bytes; execution must
                // land exactly on the body afterwards.
                loop {
                    let jb = c.read_imm8();
                    if jb == 0x2E || jb == 0x3E {
                        continue;
                    }
                    if jb == 0x74 {
                        c.read_imm8();
                        break;
                    }
                    if jb == 0x0F && c.read_imm8() == 0x84 {
                        c.read_imm32();
                        break;
                    }
                    return None;
                }
                if c.eip != body_addr {
                    return None;
                }
            }
            if c.eip == body_addr {
                in_body = true;
                body_start = ops.len();
            }
            if c.eip == tail_addr {
                in_body = false;
                body_end = ops.len();
            }
        }
        if c.eip & 0xFFF >= 0xFF0 {
            return None;
        }
        c.prefixes = 0;
        let mut prefix_66 = false;
        let mut prefix_f2 = false;
        let mut prefix_f3 = false;
        let mut b = c.read_imm8();
        loop {
            match b {
                0x66 => {
                    prefix_66 = true;
                    b = c.read_imm8();
                },
                0xF2 => {
                    prefix_f2 = true;
                    b = c.read_imm8();
                },
                0xF3 => {
                    prefix_f3 = true;
                    b = c.read_imm8();
                },
                0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x67 | 0xF0 => return None,
                _ => break,
            }
        }
        if (prefix_66 as u8 + prefix_f2 as u8 + prefix_f3 as u8) > 1 {
            return None;
        }

        if b == 0x0F {
            let op2 = c.read_imm8();
            match op2 {
                // PREFETCHh m8 / NOP
                0x18 if !prefix_66 && !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    if modrm_byte < 0xC0 {
                        let _ = modrm::decode(&mut c, modrm_byte);
                    }
                    ops.push(Op::Nop);
                },
                // MOVNTPS / MOVNTPD m128, xmm
                0x2B if !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    if modrm_byte >= 0xC0 {
                        return None;
                    }
                    let r = (modrm_byte >> 3) & 7;
                    let addr = effective_address(&mut c, modrm_byte, &registers)?;
                    let mem = add_stream_access(&mut streams, addr, 16, true)?;
                    mark_vec_read(
                        VecReg::Xmm(r),
                        &mut xmm_read_before_write,
                        xmm_written,
                        &mut mm_read_before_write,
                        mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    has_simd_or_stream = true;
                    ops.push(Op::Store128(mem, r));
                },
                // MOVAPS/MOVAPD/MOVUPS/MOVUPD xmm, xmm/m128
                0x28 | 0x10 if !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        let mem = add_stream_access(&mut streams, addr, 16, false)?;
                        mark_vec_write(
                            VecReg::Xmm(r),
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::Load128(r, mem));
                    }
                    else {
                        let rm = modrm_byte & 7;
                        mark_vec_read(
                            VecReg::Xmm(rm),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        mark_vec_write(
                            VecReg::Xmm(r),
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::MovVec(VecReg::Xmm(r), VecReg::Xmm(rm)));
                    }
                    has_simd_or_stream = true;
                },
                // MOVAPS/MOVAPD/MOVUPS/MOVUPD xmm/m128, xmm
                0x29 | 0x11 if !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        let mem = add_stream_access(&mut streams, addr, 16, true)?;
                        mark_vec_read(
                            VecReg::Xmm(r),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::Store128(mem, r));
                    }
                    else {
                        let rm = modrm_byte & 7;
                        mark_vec_read(
                            VecReg::Xmm(r),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        mark_vec_write(
                            VecReg::Xmm(rm),
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::MovVec(VecReg::Xmm(rm), VecReg::Xmm(r)));
                    }
                    has_simd_or_stream = true;
                },
                // ADDPS / SUBPS / MULPS / DIVPS (no prefix only)
                0x58 | 0x59 | 0x5C | 0x5E if !prefix_66 && !prefix_f2 && !prefix_f3 => {
                    let binop = match op2 {
                        0x58 => SimdBinOp::AddF32,
                        0x59 => SimdBinOp::MulF32,
                        0x5C => SimdBinOp::SubF32,
                        0x5E => SimdBinOp::DivF32,
                        _ => unreachable!(),
                    };
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    let dst = VecReg::Xmm(r);
                    mark_vec_read(
                        dst,
                        &mut xmm_read_before_write,
                        xmm_written,
                        &mut mm_read_before_write,
                        mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        let mem = add_stream_access(&mut streams, addr, 16, false)?;
                        ops.push(Op::BinMem(binop, dst, mem));
                    }
                    else {
                        let src = VecReg::Xmm(modrm_byte & 7);
                        mark_vec_read(
                            src,
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::BinReg(binop, dst, src));
                    }
                    mark_vec_write(
                        dst,
                        &mut xmm_written,
                        &mut mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    has_simd_or_stream = true;
                },
                // MOVD mm/xmm, r/m32
                0x6E if !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    let dst = if prefix_66 { VecReg::Xmm(r) } else { VecReg::Mm(r) };
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        let mem = add_stream_access(&mut streams, addr, 4, false)?;
                        ops.push(Op::Load32Zero(dst, mem));
                    }
                    else {
                        let rm = modrm_byte & 7;
                        gpr_data_read_mask |= 1 << rm;
                        ops.push(Op::MovdFromGpr(dst, rm));
                    }
                    mark_vec_write(
                        dst,
                        &mut xmm_written,
                        &mut mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    has_simd_or_stream = true;
                },
                // MOVD r/m32, mm/xmm OR F3 0F 7E MOVQ xmm, xmm/m64
                0x7E if !prefix_f2 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    if prefix_f3 {
                        let dst = VecReg::Xmm(r);
                        if modrm_byte < 0xC0 {
                            let addr = effective_address(&mut c, modrm_byte, &registers)?;
                            let mem = add_stream_access(&mut streams, addr, 8, false)?;
                            ops.push(Op::Load64Zero(dst, mem));
                        }
                        else {
                            let rm = modrm_byte & 7;
                            mark_vec_read(
                                VecReg::Xmm(rm),
                                &mut xmm_read_before_write,
                                xmm_written,
                                &mut mm_read_before_write,
                                mm_written,
                                &mut touched_mmx,
                                &mut needs_cr0_check,
                            );
                            ops.push(Op::MovqXmmReg(r, rm));
                        }
                        mark_vec_write(
                            dst,
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                    }
                    else {
                        let src = if prefix_66 { VecReg::Xmm(r) } else { VecReg::Mm(r) };
                        mark_vec_read(
                            src,
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        if modrm_byte < 0xC0 {
                            let addr = effective_address(&mut c, modrm_byte, &registers)?;
                            let mem = add_stream_access(&mut streams, addr, 4, true)?;
                            ops.push(Op::Store32(mem, src));
                        }
                        else {
                            let rm = modrm_byte & 7;
                            registers[rm as usize] = None;
                            gpr_written_non_ind_mask |= 1 << rm;
                            bounded_shr[rm as usize] = 0;
                            ops.push(Op::MovdToGpr(rm, src));
                        }
                    }
                    has_simd_or_stream = true;
                },
                // MOVQ mm, mm/m64 | MOVDQA (66) | MOVDQU (F3)
                0x6F if !prefix_f2 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    let is_sse = prefix_66 || prefix_f3;
                    let dst = if is_sse { VecReg::Xmm(r) } else { VecReg::Mm(r) };
                    if modrm_byte < 0xC0 {
                        let (m_base, m_index, m_shift, m_disp) =
                            modrm::decode(&mut c, modrm_byte).flat_parts()?;
                        let addr = flat_address(m_base, m_index, m_shift, m_disp, &registers);
                        if is_sse {
                            let mem = add_stream_access(&mut streams, addr?, 16, false)?;
                            ops.push(Op::Load128(r, mem));
                        }
                        else {
                            match addr {
                                Some(addr) => {
                                    let mem = add_stream_access(&mut streams, addr, 8, false)?;
                                    ops.push(Op::Load64Zero(dst, mem));
                                },
                                None => {
                                    // Bounded-index read-only table
                                    // `[base + idx*scale + disp]`: idx's most recent
                                    // write must be `shr idx, imm` with imm >= 20 so
                                    // idx < entries = 2^(32-imm), and base must be a
                                    // single loop-invariant GPR.
                                    let (Some(br), Some(ir)) = (m_base, m_index)
                                    else {
                                        return None;
                                    };
                                    let bound = bounded_shr[ir as usize];
                                    if bound < 20 {
                                        return None;
                                    }
                                    let entries = 1u32 << (32 - bound as u32);
                                    let base_addr = registers[br as usize]?;
                                    if base_addr.inv_coeffs != [0; MAX_INV_LOADS]
                                        || base_addr.offset != 0
                                        || base_addr.coefficients[br as usize] != 1
                                        || base_addr
                                            .coefficients
                                            .iter()
                                            .enumerate()
                                            .any(|(i, &co)| co != 0 && i != br as usize)
                                        || (gpr_written_non_ind_mask | gpr_ind_touched) & (1 << br)
                                            != 0
                                    {
                                        return None;
                                    }
                                    let table = match tables.iter().position(|t| {
                                        t.base_reg == br as u8
                                            && t.disp == m_disp
                                            && t.shift == m_shift
                                            && t.entries == entries
                                    }) {
                                        Some(t) => t,
                                        None => {
                                            if tables.len() >= MAX_TABLES {
                                                return None;
                                            }
                                            tables.push(TableRegion {
                                                base_reg: br as u8,
                                                disp: m_disp,
                                                shift: m_shift,
                                                entries,
                                            });
                                            tables.len() - 1
                                        },
                                    };
                                    gpr_data_read_mask |= (1 << br) | (1 << ir);
                                    ops.push(Op::Load64TableZero(dst, table, ir as u8));
                                },
                            }
                        }
                    }
                    else {
                        let rm = modrm_byte & 7;
                        let src = if is_sse { VecReg::Xmm(rm) } else { VecReg::Mm(rm) };
                        mark_vec_read(
                            src,
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::MovVec(dst, src));
                    }
                    mark_vec_write(
                        dst,
                        &mut xmm_written,
                        &mut mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    has_simd_or_stream = true;
                },
                // MOVQ mm/m64, mm | MOVDQA (66) | MOVDQU (F3)
                0x7F if !prefix_f2 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    let is_sse = prefix_66 || prefix_f3;
                    let src = if is_sse { VecReg::Xmm(r) } else { VecReg::Mm(r) };
                    mark_vec_read(
                        src,
                        &mut xmm_read_before_write,
                        xmm_written,
                        &mut mm_read_before_write,
                        mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        if is_sse {
                            let mem = add_stream_access(&mut streams, addr, 16, true)?;
                            ops.push(Op::Store128(mem, r));
                        }
                        else {
                            let mem = add_stream_access(&mut streams, addr, 8, true)?;
                            ops.push(Op::Store64(mem, src));
                        }
                    }
                    else {
                        let rm = modrm_byte & 7;
                        let dst = if is_sse { VecReg::Xmm(rm) } else { VecReg::Mm(rm) };
                        mark_vec_write(
                            dst,
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::MovVec(dst, src));
                    }
                    has_simd_or_stream = true;
                },
                // MOVNTQ m64, mm | 66 MOVNTDQ m128, xmm
                0xE7 if !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    if modrm_byte >= 0xC0 {
                        return None;
                    }
                    let r = (modrm_byte >> 3) & 7;
                    let addr = effective_address(&mut c, modrm_byte, &registers)?;
                    if prefix_66 {
                        let mem = add_stream_access(&mut streams, addr, 16, true)?;
                        mark_vec_read(
                            VecReg::Xmm(r),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::Store128(mem, r));
                    }
                    else {
                        let mem = add_stream_access(&mut streams, addr, 8, true)?;
                        mark_vec_read(
                            VecReg::Mm(r),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::Store64(mem, VecReg::Mm(r)));
                    }
                    has_simd_or_stream = true;
                },
                // 66 0F D6 MOVQ xmm/m64, xmm | F3 0F D6 MOVQ2DQ xmm, mm | F2 0F D6 MOVDQ2Q mm, xmm
                0xD6 if prefix_66 || prefix_f2 || prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    if prefix_f3 {
                        if modrm_byte < 0xC0 {
                            return None;
                        }
                        let rm = modrm_byte & 7;
                        mark_vec_read(
                            VecReg::Mm(rm),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        mark_vec_write(
                            VecReg::Xmm(r),
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::Movq2dq(r, rm));
                    }
                    else if prefix_f2 {
                        if modrm_byte < 0xC0 {
                            return None;
                        }
                        let rm = modrm_byte & 7;
                        mark_vec_read(
                            VecReg::Xmm(rm),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        mark_vec_write(
                            VecReg::Mm(r),
                            &mut xmm_written,
                            &mut mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::Movdq2q(r, rm));
                    }
                    else {
                        mark_vec_read(
                            VecReg::Xmm(r),
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        if modrm_byte < 0xC0 {
                            let addr = effective_address(&mut c, modrm_byte, &registers)?;
                            let mem = add_stream_access(&mut streams, addr, 8, true)?;
                            ops.push(Op::Store64(mem, VecReg::Xmm(r)));
                        }
                        else {
                            let rm = modrm_byte & 7;
                            mark_vec_write(
                                VecReg::Xmm(rm),
                                &mut xmm_written,
                                &mut mm_written,
                                &mut touched_mmx,
                                &mut needs_cr0_check,
                            );
                            ops.push(Op::MovqXmmReg(rm, r));
                        }
                    }
                    has_simd_or_stream = true;
                },
                // PSHUFW / PSHUFD / PSHUFLW / PSHUFHW
                0x70 => {
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    let (kind, is_sse) = if prefix_66 {
                        (ShuffleKind::Pshufd, true)
                    }
                    else if prefix_f2 {
                        (ShuffleKind::Pshuflw, true)
                    }
                    else if prefix_f3 {
                        (ShuffleKind::Pshufhw, true)
                    }
                    else {
                        (ShuffleKind::Pshufw, false)
                    };
                    let dst = if is_sse { VecReg::Xmm(r) } else { VecReg::Mm(r) };
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        let imm8 = c.read_imm8();
                        let mem = add_stream_access(
                            &mut streams,
                            addr,
                            if is_sse { 16 } else { 8 },
                            false,
                        )?;
                        ops.push(Op::ShuffleMem(kind, dst, mem, imm8));
                    }
                    else {
                        let rm = modrm_byte & 7;
                        let imm8 = c.read_imm8();
                        let src = if is_sse { VecReg::Xmm(rm) } else { VecReg::Mm(rm) };
                        mark_vec_read(
                            src,
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::ShuffleReg(kind, dst, src, imm8));
                    }
                    mark_vec_write(
                        dst,
                        &mut xmm_written,
                        &mut mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    has_simd_or_stream = true;
                },
                // Packed immediate shifts (0F 71, 0F 72, 0F 73)
                0x71 | 0x72 | 0x73 if !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    if modrm_byte < 0xC0 {
                        return None;
                    }
                    let group = (modrm_byte >> 3) & 7;
                    let rm = modrm_byte & 7;
                    let imm8 = c.read_imm8();
                    let kind = match (op2, group, prefix_66) {
                        (0x71, 2, _) => ShiftKind::ShrU16,
                        (0x71, 4, _) => ShiftKind::ShrS16,
                        (0x71, 6, _) => ShiftKind::Shl16,
                        (0x72, 2, _) => ShiftKind::ShrU32,
                        (0x72, 4, _) => ShiftKind::ShrS32,
                        (0x72, 6, _) => ShiftKind::Shl32,
                        (0x73, 2, _) => ShiftKind::ShrU64,
                        (0x73, 6, _) => ShiftKind::Shl64,
                        (0x73, 3, true) => ShiftKind::ByteShr,
                        (0x73, 7, true) => ShiftKind::ByteShl,
                        _ => return None,
                    };
                    let dst = if prefix_66 { VecReg::Xmm(rm) } else { VecReg::Mm(rm) };
                    mark_vec_read(
                        dst,
                        &mut xmm_read_before_write,
                        xmm_written,
                        &mut mm_read_before_write,
                        mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    mark_vec_write(
                        dst,
                        &mut xmm_written,
                        &mut mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    ops.push(Op::ShiftImm(kind, dst, imm8));
                    has_simd_or_stream = true;
                },
                // MOVNTI m32, r32
                0xC3 if !prefix_66 && !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    if modrm_byte >= 0xC0 {
                        return None;
                    }
                    let r = (modrm_byte >> 3) & 7;
                    let addr = effective_address(&mut c, modrm_byte, &registers)?;
                    let mem = add_stream_access(&mut streams, addr, 4, true)?;
                    gpr_data_read_mask |= 1 << r;
                    has_simd_or_stream = true;
                    ops.push(Op::GprStore32(mem, r));
                },
                // MOVZX r32, r/m8 | MOVZX r32, r/m16
                0xB6 | 0xB7 if !prefix_66 && !prefix_f2 && !prefix_f3 => {
                    let modrm_byte = c.read_imm8();
                    if modrm_byte >= 0xC0 {
                        return None;
                    }
                    let r = (modrm_byte >> 3) & 7;
                    let addr = effective_address(&mut c, modrm_byte, &registers)?;
                    let bytes = if op2 == 0xB6 { 1 } else { 2 };
                    let mem = add_stream_access(&mut streams, addr, bytes, false)?;
                    registers[r as usize] = None;
                    gpr_written_non_ind_mask |= 1 << r;
                    bounded_shr[r as usize] = 0;
                    ops.push(if op2 == 0xB6 {
                        Op::GprLoadZx8(r, mem)
                    }
                    else {
                        Op::GprLoadZx16(r, mem)
                    });
                },
                // General packed binary MMX/SSE2 instructions
                _ if !prefix_f2 && !prefix_f3 => {
                    let binop = decode_simd_binop(op2, prefix_66)?;
                    let modrm_byte = c.read_imm8();
                    let r = (modrm_byte >> 3) & 7;
                    let dst = if prefix_66 { VecReg::Xmm(r) } else { VecReg::Mm(r) };
                    if modrm_byte < 0xC0 {
                        let addr = effective_address(&mut c, modrm_byte, &registers)?;
                        let mem = add_stream_access(
                            &mut streams,
                            addr,
                            if prefix_66 { 16 } else { 8 },
                            false,
                        )?;
                        mark_vec_read(
                            dst,
                            &mut xmm_read_before_write,
                            xmm_written,
                            &mut mm_read_before_write,
                            mm_written,
                            &mut touched_mmx,
                            &mut needs_cr0_check,
                        );
                        ops.push(Op::BinMem(binop, dst, mem));
                    }
                    else {
                        let rm = modrm_byte & 7;
                        let src = if prefix_66 { VecReg::Xmm(rm) } else { VecReg::Mm(rm) };
                        let self_zero =
                            matches!(binop, SimdBinOp::Xor | SimdBinOp::AndNot) && r == rm;
                        if !self_zero {
                            mark_vec_read(
                                dst,
                                &mut xmm_read_before_write,
                                xmm_written,
                                &mut mm_read_before_write,
                                mm_written,
                                &mut touched_mmx,
                                &mut needs_cr0_check,
                            );
                            mark_vec_read(
                                src,
                                &mut xmm_read_before_write,
                                xmm_written,
                                &mut mm_read_before_write,
                                mm_written,
                                &mut touched_mmx,
                                &mut needs_cr0_check,
                            );
                        }
                        ops.push(Op::BinReg(binop, dst, src));
                    }
                    mark_vec_write(
                        dst,
                        &mut xmm_written,
                        &mut mm_written,
                        &mut touched_mmx,
                        &mut needs_cr0_check,
                    );
                    has_simd_or_stream = true;
                },
                _ => return None,
            }
            continue;
        }

        // Single-byte integer opcodes (reject prefixes)
        if prefix_66 || prefix_f2 || prefix_f3 {
            return None;
        }
        match b {
            0x90 => ops.push(Op::Nop),
            // INC r32
            0x40..=0x47 => {
                let r = b & 7;
                registers[r as usize] =
                    registers[r as usize].and_then(|a| a.add_scaled(Address::constant(1), 1));
                gpr_ind_delta[r as usize] = gpr_ind_delta[r as usize].checked_add(1)?;
                gpr_ind_touched |= 1 << r;
                bounded_shr[r as usize] = 0;
                if in_body {
                    gpr_ind_touched_in_body |= 1 << r;
                }
                last_flag_op = Some(FlagOp::Other);
                ops.push(Op::DeferredInd(r, 1));
            },
            // DEC r32
            0x48..=0x4F => {
                let r = b & 7;
                registers[r as usize] =
                    registers[r as usize].and_then(|a| a.add_scaled(Address::constant(-1), 1));
                gpr_ind_delta[r as usize] = gpr_ind_delta[r as usize].checked_add(-1)?;
                gpr_ind_touched |= 1 << r;
                bounded_shr[r as usize] = 0;
                if in_body {
                    gpr_ind_touched_in_body |= 1 << r;
                }
                last_flag_op = Some(FlagOp::Dec(r));
                ops.push(Op::DeferredInd(r, -1));
            },
            // ADD/OR/AND/SUB/XOR/CMP r32, imm8s/imm32
            0x81 | 0x83 => {
                let modrm_byte = c.read_imm8();
                let group = (modrm_byte >> 3) & 7;
                if modrm_byte < 0xC0 || !matches!(group, 0 | 1 | 4 | 5 | 6 | 7) {
                    return None;
                }
                let r = modrm_byte & 7;
                let imm = if b == 0x83 { c.read_imm8s() as i32 } else { c.read_imm32() as i32 };
                if group == 7 {
                    last_flag_op = Some(FlagOp::CmpImm(r, imm));
                    ops.push(Op::CmpMarker);
                }
                else if matches!(group, 0 | 5) {
                    let delta = if group == 0 { imm } else { imm.checked_neg()? };
                    registers[r as usize] = registers[r as usize]
                        .and_then(|a| a.add_scaled(Address::constant(delta), 1));
                    gpr_ind_delta[r as usize] = gpr_ind_delta[r as usize].checked_add(delta)?;
                    gpr_ind_touched |= 1 << r;
                    bounded_shr[r as usize] = 0;
                    if in_body {
                        gpr_ind_touched_in_body |= 1 << r;
                    }
                    last_flag_op = Some(FlagOp::Other);
                    ops.push(Op::DeferredInd(r, delta));
                }
                else {
                    gpr_data_read_mask |= 1 << r;
                    gpr_written_non_ind_mask |= 1 << r;
                    registers[r as usize] = None;
                    bounded_shr[r as usize] = 0;
                    last_flag_op = Some(FlagOp::Other);
                    ops.push(match group {
                        1 => Op::GprOrImm(r, imm),
                        4 => Op::GprAndImm(r, imm),
                        6 => Op::GprXorImm(r, imm),
                        _ => unreachable!(),
                    });
                }
            },
            // ADD/OR/AND/SUB/XOR/CMP eax, imm32
            0x05 | 0x0D | 0x25 | 0x2D | 0x35 | 0x3D => {
                let imm = c.read_imm32() as i32;
                if b == 0x3D {
                    last_flag_op = Some(FlagOp::CmpImm(0, imm));
                    ops.push(Op::CmpMarker);
                }
                else if matches!(b, 0x05 | 0x2D) {
                    let delta = if b == 0x05 { imm } else { imm.checked_neg()? };
                    registers[0] =
                        registers[0].and_then(|a| a.add_scaled(Address::constant(delta), 1));
                    gpr_ind_delta[0] = gpr_ind_delta[0].checked_add(delta)?;
                    gpr_ind_touched |= 1;
                    bounded_shr[0] = 0;
                    if in_body {
                        gpr_ind_touched_in_body |= 1;
                    }
                    last_flag_op = Some(FlagOp::Other);
                    ops.push(Op::DeferredInd(0, delta));
                }
                else {
                    gpr_data_read_mask |= 1;
                    gpr_written_non_ind_mask |= 1;
                    registers[0] = None;
                    bounded_shr[0] = 0;
                    last_flag_op = Some(FlagOp::Other);
                    ops.push(match b {
                        0x0D => Op::GprOrImm(0, imm),
                        0x25 => Op::GprAndImm(0, imm),
                        0x35 => Op::GprXorImm(0, imm),
                        _ => unreachable!(),
                    });
                }
            },
            // CMP r32, r32
            0x39 | 0x3B => {
                let modrm_byte = c.read_imm8();
                if modrm_byte < 0xC0 {
                    return None;
                }
                let rm = modrm_byte & 7;
                let r = (modrm_byte >> 3) & 7;
                let (lhs, rhs) = if b == 0x39 { (rm, r) } else { (r, rm) };
                last_flag_op = Some(FlagOp::CmpReg(lhs, rhs));
                ops.push(Op::CmpMarker);
            },
            // TEST r/m32, r32 (register form): flag-only, used by tail loop conditions.
            0x85 => {
                let modrm_byte = c.read_imm8();
                if modrm_byte < 0xC0 {
                    return None;
                }
                let rm = modrm_byte & 7;
                let r = (modrm_byte >> 3) & 7;
                last_flag_op = Some(FlagOp::TestReg(rm, r));
                ops.push(Op::CmpMarker);
            },
            // TEST eax, imm32
            0xA9 => {
                let imm = c.read_imm32() as i32;
                gpr_data_read_mask |= 1;
                last_flag_op = Some(FlagOp::Other);
                ops.push(Op::SkipTest(0, imm));
            },
            // Group 0xF7 register form: only `test r32, imm32` (/0) is accepted.
            0xF7 => {
                let modrm_byte = c.read_imm8();
                if modrm_byte < 0xC0 || (modrm_byte >> 3) & 7 != 0 {
                    return None;
                }
                let rm = modrm_byte & 7;
                let imm = c.read_imm32() as i32;
                gpr_data_read_mask |= 1 << rm;
                last_flag_op = Some(FlagOp::Other);
                ops.push(Op::SkipTest(rm, imm));
            },
            // ADD / SUB r32, r32
            0x01 | 0x03 | 0x29 | 0x2B => {
                let modrm_byte = c.read_imm8();
                if modrm_byte < 0xC0 {
                    return None;
                }
                let rm = modrm_byte & 7;
                let r = (modrm_byte >> 3) & 7;
                let (dst, src) = if b & 2 == 0 { (rm, r) } else { (r, rm) };
                gpr_data_read_mask |= (1 << dst) | (1 << src);
                gpr_written_non_ind_mask |= 1 << dst;
                registers[dst as usize] = None;
                bounded_shr[dst as usize] = 0;
                last_flag_op = Some(FlagOp::Other);
                ops.push(if matches!(b, 0x01 | 0x03) {
                    Op::GprAddReg(dst, src)
                }
                else {
                    Op::GprSubReg(dst, src)
                });
            },
            // XOR r32, r32
            0x31 | 0x33 => {
                let modrm_byte = c.read_imm8();
                if modrm_byte < 0xC0 {
                    return None;
                }
                let rm = modrm_byte & 7;
                let r = (modrm_byte >> 3) & 7;
                let (dst, src) = if b == 0x31 { (rm, r) } else { (r, rm) };
                if dst != src {
                    gpr_data_read_mask |= (1 << dst) | (1 << src);
                }
                gpr_written_non_ind_mask |= 1 << dst;
                registers[dst as usize] =
                    if dst == src { Some(Address::constant(0)) } else { None };
                bounded_shr[dst as usize] = 0;
                last_flag_op = Some(FlagOp::Other);
                ops.push(Op::GprXorReg(dst, src));
            },
            // SHL/SHR/SAR r32, imm8
            0xC1 => {
                let modrm_byte = c.read_imm8();
                let group = (modrm_byte >> 3) & 7;
                if modrm_byte < 0xC0 || !matches!(group, 4 | 5 | 7) {
                    return None;
                }
                let rm = modrm_byte & 7;
                let imm = c.read_imm8() & 31;
                gpr_data_read_mask |= 1 << rm;
                gpr_written_non_ind_mask |= 1 << rm;
                registers[rm as usize] = None;
                bounded_shr[rm as usize] = if group == 5 { imm } else { 0 };
                last_flag_op = Some(FlagOp::Other);
                ops.push(match group {
                    4 => Op::GprShlImm(rm, imm),
                    5 => Op::GprShrUImm(rm, imm),
                    7 => Op::GprShrSImm(rm, imm),
                    _ => unreachable!(),
                });
            },
            // MOV r32, imm32
            0xB8..=0xBF => {
                let r = b & 7;
                let imm = c.read_imm32() as i32;
                gpr_written_non_ind_mask |= 1 << r;
                registers[r as usize] = Some(Address::constant(imm));
                bounded_shr[r as usize] = 0;
                ops.push(Op::GprMovImm(r, imm));
            },
            // MOV r/m32, r32 | MOV r32, r/m32
            0x89 | 0x8B => {
                let modrm_byte = c.read_imm8();
                let r = (modrm_byte >> 3) & 7;
                if modrm_byte < 0xC0 {
                    let addr = effective_address(&mut c, modrm_byte, &registers)?;
                    if b == 0x8B {
                        let mem = add_stream_access(&mut streams, addr, 4, false)?;
                        registers[r as usize] = if addr.inv_coeffs == [0; MAX_INV_LOADS] {
                            let cand = InvLoad {
                                coefficients: addr.coefficients,
                                offset: addr.offset,
                            };
                            if let Some(pos) = inv_candidates.iter().position(|x| *x == cand) {
                                Some(Address::inv_load(pos))
                            }
                            else if inv_candidates.len() < MAX_INV_LOADS {
                                let pos = inv_candidates.len();
                                inv_candidates.push(cand);
                                Some(Address::inv_load(pos))
                            }
                            else {
                                None
                            }
                        }
                        else {
                            None
                        };
                        gpr_written_non_ind_mask |= 1 << r;
                        bounded_shr[r as usize] = 0;
                        ops.push(Op::GprLoad32(r, mem));
                    }
                    else {
                        let mem = add_stream_access(&mut streams, addr, 4, true)?;
                        gpr_data_read_mask |= 1 << r;
                        ops.push(Op::GprStore32(mem, r));
                    }
                }
                else {
                    let rm = modrm_byte & 7;
                    let (dst, src) = if b == 0x89 { (rm, r) } else { (r, rm) };
                    gpr_data_read_mask |= 1 << src;
                    gpr_written_non_ind_mask |= 1 << dst;
                    registers[dst as usize] = registers[src as usize];
                    bounded_shr[dst as usize] = 0;
                    ops.push(Op::GprMovReg(dst, src));
                }
            },
            // LEA r32, m
            0x8D => {
                let modrm_byte = c.read_imm8();
                if modrm_byte >= 0xC0 {
                    return None;
                }
                let r = (modrm_byte >> 3) & 7;
                let decoded = modrm::decode(&mut c, modrm_byte);
                if decoded.is_nop(r as u32) {
                    ops.push(Op::Nop);
                }
                else {
                    let (base, index, shift, disp) = decoded.flat_parts()?;
                    let mut a = Address::constant(disp);
                    if let Some(br) = base {
                        gpr_data_read_mask |= 1 << br;
                        a = a.add_scaled(registers[br as usize]?, 1)?;
                    }
                    if let Some(ir) = index {
                        gpr_data_read_mask |= 1 << ir;
                        a = a.add_scaled(registers[ir as usize]?, 1 << shift)?;
                    }
                    gpr_written_non_ind_mask |= 1 << r;
                    registers[r as usize] = Some(a);
                    bounded_shr[r as usize] = 0;
                    ops.push(Op::GprLea(
                        r,
                        base.map(|x| x as u8),
                        index.map(|x| x as u8),
                        shift,
                        disp,
                    ));
                }
            },
            _ => return None,
        }
    }

    if c.eip != loop_last_addr || streams.is_empty() {
        return None;
    }

    let has_body = skip_bounds.is_some();
    if has_body {
        // The skip test must be the head's last instruction (immediately before
        // `je`), exactly once, and the body must contain at least one op.
        if body_start == 0
            || body_end <= body_start
            || !matches!(ops[body_start - 1], Op::SkipTest(..))
            || ops
                .iter()
                .filter(|op| matches!(op, Op::SkipTest(..)))
                .count()
                != 1
        {
            return None;
        }
        // An induction write inside the conditional body would only execute on
        // non-skipped iterations, so it must not feed a stream's batch stride.
        if gpr_ind_touched_in_body != 0 {
            for s in &streams {
                if !s.used {
                    continue;
                }
                for r in 0..8 {
                    if s.coefficients[r] != 0 && gpr_ind_touched_in_body & (1 << r) != 0 {
                        return None;
                    }
                }
            }
        }
    }

    // Compact loop-invariant scalar pointer loads that are referenced by stream addresses.
    let mut inv_loads: Vec<InvLoad> = Vec::new();
    let mut inv_remap: [Option<usize>; MAX_INV_LOADS] = [None; MAX_INV_LOADS];
    for s in &streams {
        for k in 0..MAX_INV_LOADS {
            if s.inv_coeffs[k] != 0 && inv_remap[k].is_none() {
                let cand = inv_candidates[k];
                for r in 0..8 {
                    if cand.coefficients[r] != 0
                        && (gpr_written_non_ind_mask | gpr_ind_touched) & (1 << r) != 0
                    {
                        return None;
                    }
                }
                let new_idx = inv_loads.len();
                inv_loads.push(cand);
                inv_remap[k] = Some(new_idx);
            }
        }
    }
    for s in &mut streams {
        let old = s.inv_coeffs;
        s.inv_coeffs = [0; MAX_INV_LOADS];
        for k in 0..MAX_INV_LOADS {
            if old[k] != 0 {
                let new_idx = inv_remap[k].unwrap();
                s.inv_coeffs[new_idx] = old[k];
            }
        }
    }

    // Hoist GprLoad32 ops that match a verified inv_load into GprLoadInv32,
    // and mark only streams still referenced by remaining ops as used.
    if !inv_loads.is_empty() {
        for op in &mut ops {
            if let Op::GprLoad32(r, mem) = *op {
                let s = &streams[mem.stream];
                if !s.is_write && s.inv_coeffs == [0; MAX_INV_LOADS] {
                    let cand = InvLoad {
                        coefficients: s.coefficients,
                        offset: mem.offset,
                    };
                    if let Some(slot) = inv_loads.iter().position(|x| *x == cand) {
                        *op = Op::GprLoadInv32(r, slot);
                    }
                }
            }
        }
    }
    for s in &mut streams {
        s.used = false;
    }
    for op in &ops {
        let mem = match *op {
            Op::Load128(_, m)
            | Op::Store128(m, _)
            | Op::Load64Zero(_, m)
            | Op::Store64(m, _)
            | Op::Load32Zero(_, m)
            | Op::Store32(m, _)
            | Op::BinMem(_, _, m)
            | Op::ShuffleMem(_, _, m, _)
            | Op::GprLoad32(_, m)
            | Op::GprLoadZx8(_, m)
            | Op::GprLoadZx16(_, m)
            | Op::GprStore32(m, _) => Some(m),
            _ => None,
        };
        if let Some(m) = mem {
            streams[m.stream].used = true;
        }
    }

    // Verify all GPRs used as active stream bases are invariant or linear induction.
    for s in &mut streams {
        if !s.used {
            continue;
        }
        let mut stride: i32 = 0;
        for r in 0..8 {
            let coeff = s.coefficients[r];
            if coeff == 0 {
                continue;
            }
            if gpr_written_non_ind_mask & (1 << r) != 0 {
                return None;
            }
            stride = stride.checked_add(coeff.checked_mul(gpr_ind_delta[r])?)?;
        }
        if stride == i32::MIN {
            return None;
        }
        s.stride = stride;
    }

    // Must have either SIMD/MMX/streaming instruction OR a strided memory loop.
    if !has_simd_or_stream && streams.iter().all(|s| !s.used || s.stride == 0) {
        return None;
    }

    // Resolve loop condition from last_flag_op and backedge condition.
    let cond = match (cond_low, last_flag_op?) {
        (0x5, FlagOp::Dec(cnt))
            if gpr_written_non_ind_mask & (1 << cnt) == 0
                && gpr_ind_touched_in_body & (1 << cnt) == 0
                && gpr_ind_delta[cnt as usize] == -1 =>
        {
            LoopCond::DecJne(cnt)
        },
        (0xC, FlagOp::CmpReg(cnt, lim))
            if cnt != lim
                && gpr_written_non_ind_mask & ((1 << cnt) | (1 << lim)) == 0
                && gpr_ind_touched_in_body & (1 << cnt) == 0
                && gpr_ind_delta[cnt as usize] == 1
                && gpr_ind_delta[lim as usize] == 0 =>
        {
            LoopCond::IncCmpRegJl(cnt, lim)
        },
        (0xC, FlagOp::CmpImm(cnt, lim))
            if gpr_written_non_ind_mask & (1 << cnt) == 0
                && gpr_ind_touched_in_body & (1 << cnt) == 0
                && gpr_ind_delta[cnt as usize] == 1 =>
        {
            LoopCond::IncCmpImmJl(cnt, lim)
        },
        (0x5, FlagOp::CmpReg(cnt, lim))
            if cnt != lim
                && gpr_written_non_ind_mask & ((1 << cnt) | (1 << lim)) == 0
                && gpr_ind_touched_in_body & (1 << cnt) == 0
                && gpr_ind_delta[lim as usize] == 0
                && matches!(gpr_ind_delta[cnt as usize], 1 | -1) =>
        {
            if gpr_ind_delta[cnt as usize] == 1 {
                LoopCond::IncCmpRegJne(cnt, lim)
            }
            else {
                LoopCond::DecCmpRegJne(cnt, lim)
            }
        },
        (0x5, FlagOp::CmpImm(cnt, lim))
            if gpr_written_non_ind_mask & (1 << cnt) == 0
                && gpr_ind_touched_in_body & (1 << cnt) == 0
                && matches!(gpr_ind_delta[cnt as usize], 1 | -1) =>
        {
            if gpr_ind_delta[cnt as usize] == 1 {
                LoopCond::IncCmpImmJne(cnt, lim)
            }
            else {
                LoopCond::DecCmpImmJne(cnt, lim)
            }
        },
        // `dec cnt` outside the conditional body; tail is `test cnt, cnt; jne`.
        (0x5, FlagOp::TestReg(cnt, other))
            if cnt == other
                && gpr_written_non_ind_mask & (1 << cnt) == 0
                && gpr_ind_touched_in_body & (1 << cnt) == 0
                && gpr_ind_delta[cnt as usize] == -1 =>
        {
            LoopCond::DecTestJne(cnt)
        },
        _ => return None,
    };

    // Any induction GPR that is never read as non-address data and never
    // overwritten by a non-induction op can be deferred until after the inner
    // batch (updated once via `gpr += batch * delta`).
    let deferred_gpr_mask = gpr_ind_touched
        & !gpr_data_read_mask
        & !gpr_written_non_ind_mask
        & !gpr_ind_touched_in_body;
    // Convert non-deferred induction ops into explicit in-loop adds so data
    // readers always observe the exact intermediate GPR value.
    for op in &mut ops {
        if let Op::DeferredInd(r, d) = *op {
            if deferred_gpr_mask & (1 << r) == 0 {
                *op = Op::GprAddImm(r, d);
            }
        }
    }

    let mmx_in_head_tail = if has_body {
        ops[..body_start]
            .iter()
            .chain(&ops[body_end..])
            .any(op_uses_mmx)
    }
    else {
        touched_mmx
    };

    Some(FastSimdLoop {
        ops,
        streams,
        inv_loads,
        cond,
        deltas: gpr_ind_delta,
        deferred_gpr_mask,
        xmm_read_before_write,
        xmm_written,
        mm_read_before_write,
        mm_written,
        touched_mmx,
        needs_cr0_check,
        num_instructions,
        body_start,
        body_end,
        body_instructions: if has_body {
            skip_loop
                .map(|(body, _)| body.number_of_instructions)
                .unwrap_or(0)
        }
        else {
            0
        },
        tables,
        mmx_in_head_tail,
    })
}

fn emit_min_u32(b: &mut WasmBuilder, a: &WasmLocal, c: &WasmLocal) {
    // Leaves min_u32(a, c) on stack.
    b.get_local(a);
    b.get_local(c);
    b.get_local(a);
    b.get_local(c);
    b.ltu_i32();
    b.select();
}

fn emit_simd_binop(
    b: &mut WasmBuilder,
    op: SimdBinOp,
    is_sse: bool,
    dst: &WasmLocalV128,
    src_push: impl Fn(&mut WasmBuilder),
) {
    use SimdBinOp::*;
    match op {
        AndNot => {
            // x86 PANDN dst, src -> src & (~dst) = v128.andnot(src, dst)
            src_push(b);
            b.get_local_v128(dst);
            b.andnot_v128();
        },
        MulHiS16 | MulHiU16 => {
            let signed = matches!(op, MulHiS16);
            let src_tmp = {
                src_push(b);
                b.set_new_local_v128()
            };
            b.get_local_v128(dst);
            b.get_local_v128(&src_tmp);
            if signed {
                b.extmul_low_i16x8_s();
            }
            else {
                b.extmul_low_i16x8_u();
            }
            if is_sse {
                b.get_local_v128(dst);
                b.get_local_v128(&src_tmp);
                if signed {
                    b.extmul_high_i16x8_s();
                }
                else {
                    b.extmul_high_i16x8_u();
                }
                b.i8x16_shuffle([2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31]);
            }
            else {
                b.v128_const([0; 16]);
                b.i8x16_shuffle([2, 3, 6, 7, 10, 11, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16]);
            }
            b.free_local_v128(src_tmp);
        },
        PackSSWB | PackUSWB | PackSSDW => {
            b.get_local_v128(dst);
            src_push(b);
            if !is_sse {
                b.i8x16_shuffle([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]);
                b.v128_const([0; 16]);
            }
            match op {
                PackSSWB => b.i8x16_narrow_i16x8_s(),
                PackUSWB => b.i8x16_narrow_i16x8_u(),
                PackSSDW => b.i16x8_narrow_i32x4_s(),
                _ => unreachable!(),
            }
        },
        _ => {
            b.get_local_v128(dst);
            src_push(b);
            match op {
                AddI8 => b.add_i8x16(),
                AddI16 => b.add_i16x8(),
                AddI32 => b.add_i32x4(),
                AddI64 => b.add_i64x2(),
                AddSatS8 => b.add_sat_s_i8x16(),
                AddSatS16 => b.add_sat_s_i16x8(),
                AddSatU8 => b.add_sat_u_i8x16(),
                AddSatU16 => b.add_sat_u_i16x8(),
                SubI8 => b.sub_i8x16(),
                SubI16 => b.sub_i16x8(),
                SubI32 => b.sub_i32x4(),
                SubI64 => b.sub_i64x2(),
                SubSatS8 => b.sub_sat_s_i8x16(),
                SubSatS16 => b.sub_sat_s_i16x8(),
                SubSatU8 => b.sub_sat_u_i8x16(),
                SubSatU16 => b.sub_sat_u_i16x8(),
                MulLoI16 => b.mul_i16x8(),
                Pmaddwd => b.dot_i16x8_s(),
                And => b.and_v128(),
                Or => b.or_v128(),
                Xor => b.xor_v128(),
                EqI8 => b.eq_i8x16(),
                EqI16 => b.eq_i16x8(),
                EqI32 => b.eq_i32x4(),
                GtSI8 => b.gt_s_i8x16(),
                GtSI16 => b.gt_s_i16x8(),
                GtSI32 => b.gt_s_i32x4(),
                MinU8 => b.min_u_i8x16(),
                MaxU8 => b.max_u_i8x16(),
                MinS16 => b.min_s_i16x8(),
                MaxS16 => b.max_s_i16x8(),
                AvgU8 => b.avgr_u_i8x16(),
                AvgU16 => b.avgr_u_i16x8(),
                UnpackLo8 => {
                    b.i8x16_shuffle([0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23])
                },
                UnpackLo16 => {
                    b.i8x16_shuffle([0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23])
                },
                UnpackLo32 => {
                    b.i8x16_shuffle([0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23])
                },
                UnpackLo64 => {
                    b.i8x16_shuffle([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23])
                },
                UnpackHi8 => {
                    if is_sse {
                        b.i8x16_shuffle([
                            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
                        ]);
                    }
                    else {
                        b.i8x16_shuffle([4, 20, 5, 21, 6, 22, 7, 23, 0, 0, 0, 0, 0, 0, 0, 0]);
                    }
                },
                UnpackHi16 => {
                    if is_sse {
                        b.i8x16_shuffle([
                            8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31,
                        ]);
                    }
                    else {
                        b.i8x16_shuffle([4, 5, 20, 21, 6, 7, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0]);
                    }
                },
                UnpackHi32 => {
                    if is_sse {
                        b.i8x16_shuffle([
                            8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31,
                        ]);
                    }
                    else {
                        b.i8x16_shuffle([4, 5, 6, 7, 20, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0]);
                    }
                },
                UnpackHi64 => {
                    b.i8x16_shuffle([8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31])
                },
                AddF32 => b.add_f32x4(),
                SubF32 => b.sub_f32x4(),
                MulF32 => b.mul_f32x4(),
                DivF32 => b.div_f32x4(),
                AndNot | MulHiS16 | MulHiU16 | PackSSWB | PackUSWB | PackSSDW => unreachable!(),
            }
        },
    }
    b.set_local_v128(dst);
}

fn emit_shuffle(b: &mut WasmBuilder, kind: ShuffleKind, imm8: u8) {
    let mut lanes = [0u8; 16];
    match kind {
        ShuffleKind::Pshufw | ShuffleKind::Pshuflw => {
            for i in 0..4 {
                let src = ((imm8 >> (2 * i)) & 3) * 2;
                lanes[i * 2] = src;
                lanes[i * 2 + 1] = src + 1;
            }
            for i in 8..16 {
                lanes[i] = i as u8;
            }
        },
        ShuffleKind::Pshufhw => {
            for i in 0..8 {
                lanes[i] = i as u8;
            }
            for i in 0..4 {
                let src = 8 + ((imm8 >> (2 * i)) & 3) * 2;
                lanes[8 + i * 2] = src;
                lanes[8 + i * 2 + 1] = src + 1;
            }
        },
        ShuffleKind::Pshufd => {
            for i in 0..4 {
                let src = ((imm8 >> (2 * i)) & 3) * 4;
                lanes[i * 4] = src;
                lanes[i * 4 + 1] = src + 1;
                lanes[i * 4 + 2] = src + 2;
                lanes[i * 4 + 3] = src + 3;
            }
        },
    }
    b.i8x16_shuffle(lanes);
}

pub fn generate_loop(
    ctx: &mut JitContext,
    block: &BasicBlock,
    fallthrough_loop: Option<&BasicBlock>,
    skip_loop: Option<(&BasicBlock, &BasicBlock)>,
) {
    if cfg!(feature = "profiler") {
        return;
    }
    let Some(limit_local) = ctx.instruction_limit.as_ref().map(|l| l.unsafe_clone())
    else {
        return;
    };
    let Some((common_base, permissions)) = (unsafe { crate::cpu::vine::common_memory() })
    else {
        return;
    };
    // With the memarg fold the arena base rides in the memarg of every
    // access instead of being added into the pointer locals.
    let simd_extra = if codegen::common_memarg_base(ctx).is_some() { common_base } else { 0 };
    let Some(plan) = scan(ctx.cpu, block, fallthrough_loop, skip_loop)
    else {
        return;
    };

    let xmm_used = plan.xmm_read_before_write | plan.xmm_written;
    let mm_used = plan.mm_read_before_write | plan.mm_written;
    let active_streams = plan.streams.iter().filter(|s| s.used).count() as u8;
    let need_v128 = (xmm_used.count_ones() + mm_used.count_ones()) as u8 + 2;
    let need_i32 =
        active_streams + (plan.inv_loads.len() as u8) * 2 + (plan.tables.len() as u8) * 5 + 14;
    if !ctx
        .builder
        .can_allocate_typed_locals(need_i32, 0, need_v128, 0)
    {
        return;
    }

    let skip_fast = ctx.builder.block_void();

    // 1. Check CR0_EM | CR0_TS once before entering the fast loop.
    if plan.needs_cr0_check {
        let cr0_offset = gp::get_creg_offset(0);
        ctx.builder.load_fixed_u8(cr0_offset);
        ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
        ctx.builder.and_i32();
        ctx.builder.br_if(skip_fast);
    }

    // 2. Compute budget_iters = (limit - instruction_counter) / num_instructions.
    // Need at least 2 iterations of budget so at least 1 remains for the standard BB.
    ctx.builder.get_local(&limit_local);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.leu_i32();
    ctx.builder.br_if(skip_fast);

    ctx.builder.get_local(&limit_local);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(plan.num_instructions as i32);
    ctx.builder.div_u_i32();
    let budget_iters = ctx.builder.tee_new_local();
    ctx.builder.const_i32(2);
    ctx.builder.ltu_i32();
    ctx.builder.br_if(skip_fast);

    ctx.builder.get_local(&budget_iters);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    ctx.builder.set_local(&budget_iters);

    // 3. Compute counter_iters (strictly leaving >= 1 iteration before loop exit).
    match plan.cond {
        LoopCond::DecJne(cnt) | LoopCond::DecTestJne(cnt) => {
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
        LoopCond::IncCmpRegJl(cnt, lim) => {
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.get_local(&ctx.register_locals[lim as usize]);
            ctx.builder.ge_i32();
            ctx.builder.br_if(skip_fast);
            ctx.builder.get_local(&ctx.register_locals[lim as usize]);
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
        LoopCond::IncCmpImmJl(cnt, lim) => {
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.const_i32(lim);
            ctx.builder.ge_i32();
            ctx.builder.br_if(skip_fast);
            ctx.builder.const_i32(lim);
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
        LoopCond::IncCmpRegJne(cnt, lim) => {
            ctx.builder.get_local(&ctx.register_locals[lim as usize]);
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
        LoopCond::IncCmpImmJne(cnt, lim) => {
            ctx.builder.const_i32(lim);
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
        LoopCond::DecCmpRegJne(cnt, lim) => {
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.get_local(&ctx.register_locals[lim as usize]);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
        LoopCond::DecCmpImmJne(cnt, lim) => {
            ctx.builder.get_local(&ctx.register_locals[cnt as usize]);
            ctx.builder.const_i32(lim);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
        },
    }
    let counter_iters = ctx.builder.tee_new_local();
    ctx.builder.eqz_i32();
    ctx.builder.br_if(skip_fast);

    emit_min_u32(ctx.builder, &budget_iters, &counter_iters);
    let remaining = ctx.builder.set_new_local();
    ctx.builder.free_local(budget_iters);
    ctx.builder.free_local(counter_iters);

    let mut inv_val_locals: Vec<WasmLocal> = Vec::with_capacity(plan.inv_loads.len());
    let mut inv_page_locals: Vec<WasmLocal> = Vec::with_capacity(plan.inv_loads.len());
    for inv in &plan.inv_loads {
        ctx.builder.const_i32(inv.offset);
        for (r, &coeff) in inv.coefficients.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            ctx.builder.get_local(&ctx.register_locals[r]);
            if coeff != 1 {
                if (coeff as u32).is_power_of_two() {
                    ctx.builder.const_i32(coeff.trailing_zeros() as i32);
                    ctx.builder.shl_i32();
                }
                else {
                    ctx.builder.const_i32(coeff);
                    ctx.builder.mul_i32();
                }
            }
            ctx.builder.add_i32();
        }
        let inv_va = ctx.builder.tee_new_local();
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(4096 - 4);
        ctx.builder.gtu_i32();
        ctx.builder.br_if(skip_fast);

        ctx.builder.get_local(&inv_va);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        let inv_page = ctx.builder.tee_new_local();
        ctx.builder.load_u8(permissions);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.eqz_i32();
        ctx.builder.br_if(skip_fast);

        ctx.builder.get_local(&inv_va);
        if simd_extra == 0 {
            ctx.builder.const_i32(common_base as i32);
            ctx.builder.add_i32();
            ctx.builder.load_unaligned_i32(0);
        }
        else {
            ctx.builder.load_unaligned_i32(simd_extra);
        }
        inv_val_locals.push(ctx.builder.set_new_local());
        inv_page_locals.push(inv_page);
        ctx.builder.free_local(inv_va);
    }

    // Bounded-index read-only table regions are loop-invariant: guard first and
    // last byte of each region once (read permission on one or two pages) and
    // reject regions spanning more than two pages.
    let mut table_ptr_locals: Vec<WasmLocal> = Vec::with_capacity(plan.tables.len());
    let mut table_page_locals: Vec<WasmLocal> = Vec::with_capacity(plan.tables.len() * 2);
    for t in &plan.tables {
        ctx.builder
            .get_local(&ctx.register_locals[t.base_reg as usize]);
        ctx.builder.const_i32(t.disp);
        ctx.builder.add_i32();
        let region_lo = ctx.builder.set_new_local();
        let last_offset = (((t.entries - 1) << t.shift) + 7) as i32;
        ctx.builder.get_local(&region_lo);
        ctx.builder.const_i32(last_offset);
        ctx.builder.add_i32();
        let region_hi = ctx.builder.tee_new_local();

        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        ctx.builder.get_local(&region_lo);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        let lo_page = ctx.builder.tee_new_local();
        ctx.builder.sub_i32();
        ctx.builder.const_i32(1);
        ctx.builder.gtu_i32();
        ctx.builder.br_if(skip_fast);

        ctx.builder.get_local(&lo_page);
        ctx.builder.load_u8(permissions);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.eqz_i32();
        ctx.builder.br_if(skip_fast);

        ctx.builder.get_local(&region_hi);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        let hi_page = ctx.builder.tee_new_local();
        ctx.builder.load_u8(permissions);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.eqz_i32();
        ctx.builder.br_if(skip_fast);

        table_page_locals.push(lo_page);
        table_page_locals.push(hi_page);

        ctx.builder.get_local(&region_lo);
        if simd_extra == 0 {
            ctx.builder.const_i32(common_base as i32);
            ctx.builder.add_i32();
        }
        table_ptr_locals.push(ctx.builder.set_new_local());
        ctx.builder.free_local(region_lo);
        ctx.builder.free_local(region_hi);
    }

    ctx.builder.const_i32(0);
    let done_iters = ctx.builder.set_new_local();
    // Skip loop: `skip_local` holds `(reg & imm) != 0` per iteration and
    // `body_iters` counts iterations that actually executed the body (for exact
    // instruction accounting and conditional MMX side effects).
    let (skip_local, body_iters) = if plan.body_instructions > 0 {
        ctx.builder.const_i32(0);
        let skip_local = ctx.builder.set_new_local();
        ctx.builder.const_i32(0);
        let body_iters = ctx.builder.set_new_local();
        (Some(skip_local), Some(body_iters))
    }
    else {
        (None, None)
    };

    // 4. Allocate and load live-in XMM / MMX Wasm v128 locals.
    let mut xmm_locals: [Option<WasmLocalV128>; 8] = Default::default();
    let mut mm_locals: [Option<WasmLocalV128>; 8] = Default::default();
    for r in 0..8u8 {
        if xmm_used & (1 << r) != 0 {
            let local = ctx.builder.alloc_local_v128();
            if plan.xmm_read_before_write & (1 << r) != 0 {
                ctx.builder
                    .load_fixed_v128(gp::get_reg_xmm_offset(r as u32));
                ctx.builder.set_local_v128(&local);
            }
            xmm_locals[r as usize] = Some(local);
        }
        if mm_used & (1 << r) != 0 {
            let local = ctx.builder.alloc_local_v128();
            if plan.mm_read_before_write & (1 << r) != 0 {
                ctx.builder
                    .load_fixed_v128_zero64(gp::get_reg_mmx_offset(r as u32));
                ctx.builder.set_local_v128(&local);
            }
            mm_locals[r as usize] = Some(local);
        }
    }

    let mut ptr_locals: Vec<Option<WasmLocal>> = Vec::with_capacity(plan.streams.len());
    for s in &plan.streams {
        if s.used {
            ctx.builder.const_i32(0);
            ptr_locals.push(Some(ctx.builder.set_new_local()));
        }
        else {
            ptr_locals.push(None);
        }
    }
    ctx.builder.const_i32(0);
    let batch = ctx.builder.set_new_local();
    ctx.builder.const_i32(0);
    let inner_i = ctx.builder.set_new_local();

    // 5. Outer page-batch loop.
    let outer_done = ctx.builder.block_void();
    let outer_loop = ctx.builder.loop_void();

    ctx.builder.get_local(&remaining);
    ctx.builder.set_local(&batch);

    for (s_idx, s) in plan.streams.iter().enumerate() {
        if !s.used {
            continue;
        }
        let span = s.span_bytes();
        // Compute starting virtual address for this stream's min_offset.
        ctx.builder.const_i32(s.min_offset);
        for (r, &coeff) in s.coefficients.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            ctx.builder.get_local(&ctx.register_locals[r]);
            if coeff != 1 {
                if (coeff as u32).is_power_of_two() {
                    ctx.builder.const_i32(coeff.trailing_zeros() as i32);
                    ctx.builder.shl_i32();
                }
                else {
                    ctx.builder.const_i32(coeff);
                    ctx.builder.mul_i32();
                }
            }
            ctx.builder.add_i32();
        }
        for (k, &coeff) in s.inv_coeffs.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            ctx.builder.get_local(&inv_val_locals[k]);
            if coeff != 1 {
                if (coeff as u32).is_power_of_two() {
                    ctx.builder.const_i32(coeff.trailing_zeros() as i32);
                    ctx.builder.shl_i32();
                }
                else {
                    ctx.builder.const_i32(coeff);
                    ctx.builder.mul_i32();
                }
            }
            ctx.builder.add_i32();
        }
        let va = ctx.builder.tee_new_local();

        // Starting access must not cross a 4 KiB page boundary.
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        let page_off = ctx.builder.tee_new_local();
        ctx.builder.const_i32((4096 - span) as i32);
        ctx.builder.gtu_i32();
        ctx.builder.br_if(outer_done);

        // Page must have required permissions (bit 1 is cleared for executable pages).
        ctx.builder.get_local(&va);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        let s_page = if s.is_write && (!inv_page_locals.is_empty() || !table_page_locals.is_empty())
        {
            Some(ctx.builder.tee_new_local())
        }
        else {
            None
        };
        ctx.builder.load_u8(permissions);
        if s.is_write {
            ctx.builder.const_i32(3);
            ctx.builder.and_i32();
            ctx.builder.const_i32(3);
            ctx.builder.ne_i32();
        }
        else {
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
            ctx.builder.eqz_i32();
        }
        ctx.builder.br_if(outer_done);

        // Prove write stream cannot alias any hoisted loop-invariant load or
        // read-only table region page.
        if let Some(s_page) = s_page {
            for inv_page in inv_page_locals.iter().chain(&table_page_locals) {
                ctx.builder.get_local(&s_page);
                ctx.builder.get_local(inv_page);
                ctx.builder.eq_i32();
                ctx.builder.br_if(outer_done);
            }
            ctx.builder.free_local(s_page);
        }

        // Bound `batch` to iterations staying within this single verified page.
        if s.stride > 0 {
            let stride_u = s.stride as u32;
            ctx.builder.const_i32((4096 - span) as i32);
            ctx.builder.get_local(&page_off);
            ctx.builder.sub_i32();
            if stride_u.is_power_of_two() {
                if stride_u > 1 {
                    ctx.builder.const_i32(stride_u.trailing_zeros() as i32);
                    ctx.builder.shr_u_i32();
                }
            }
            else {
                ctx.builder.const_i32(stride_u as i32);
                ctx.builder.div_u_i32();
            }
            ctx.builder.const_i32(1);
            ctx.builder.add_i32();
            let page_iters = ctx.builder.set_new_local();
            emit_min_u32(ctx.builder, &batch, &page_iters);
            ctx.builder.set_local(&batch);
            ctx.builder.free_local(page_iters);
        }
        else if s.stride < 0 {
            let neg_u = (-s.stride) as u32;
            ctx.builder.get_local(&page_off);
            if neg_u.is_power_of_two() {
                if neg_u > 1 {
                    ctx.builder.const_i32(neg_u.trailing_zeros() as i32);
                    ctx.builder.shr_u_i32();
                }
            }
            else {
                ctx.builder.const_i32(neg_u as i32);
                ctx.builder.div_u_i32();
            }
            ctx.builder.const_i32(1);
            ctx.builder.add_i32();
            let page_iters = ctx.builder.set_new_local();
            emit_min_u32(ctx.builder, &batch, &page_iters);
            ctx.builder.set_local(&batch);
            ctx.builder.free_local(page_iters);
        }

        // Convert to Wasm linear memory pointer.
        ctx.builder.get_local(&va);
        if simd_extra == 0 {
            ctx.builder.const_i32(common_base as i32);
            ctx.builder.add_i32();
        }
        ctx.builder.set_local(ptr_locals[s_idx].as_ref().unwrap());

        ctx.builder.free_local(page_off);
        ctx.builder.free_local(va);
    }

    // 6. Inner tight SIMD loop: `batch` verified iterations with zero guards/flags.
    ctx.builder.get_local(&batch);
    ctx.builder.set_local(&inner_i);

    let inner_loop = ctx.builder.loop_void();
    // Track which MMX locals are guaranteed to have zero upper 64 bits within this iteration.
    let mut mm_zero_hi: u8 = 0;

    let vec_local = |reg: VecReg| -> &WasmLocalV128 {
        match reg {
            VecReg::Xmm(r) => xmm_locals[r as usize].as_ref().unwrap(),
            VecReg::Mm(r) => mm_locals[r as usize].as_ref().unwrap(),
        }
    };

    // The conditional body (ops[body_start..body_end]) runs inside `if
    // (skip_local)` so skipped iterations keep every register's previous value.
    let mut mm_zero_before_body: u8 = 0;
    for (op_index, &op) in plan.ops.iter().enumerate() {
        if plan.body_instructions > 0 && op_index == plan.body_start {
            ctx.builder.get_local(skip_local.as_ref().unwrap());
            ctx.builder.if_void();
            ctx.builder.get_local(body_iters.as_ref().unwrap());
            ctx.builder.const_i32(1);
            ctx.builder.add_i32();
            ctx.builder.set_local(body_iters.as_ref().unwrap());
            mm_zero_before_body = mm_zero_hi;
        }
        if plan.body_instructions > 0 && op_index == plan.body_end {
            ctx.builder.block_end();
            // Zero-high tracking after the conditional region is the merge of
            // "body ran" (current tracking) and "body skipped" (entry state).
            mm_zero_hi &= mm_zero_before_body;
        }
        match op {
            Op::Nop | Op::DeferredInd(..) | Op::CmpMarker => {},
            Op::SkipTest(reg, imm) => {
                if let Some(skip_local) = skip_local.as_ref() {
                    ctx.builder.get_local(&ctx.register_locals[reg as usize]);
                    ctx.builder.const_i32(imm);
                    ctx.builder.and_i32();
                    ctx.builder.set_local(skip_local);
                }
            },
            Op::Load64TableZero(dst, table, index) => {
                let t = &plan.tables[table];
                ctx.builder.get_local(&table_ptr_locals[table]);
                ctx.builder.get_local(&ctx.register_locals[index as usize]);
                if t.shift != 0 {
                    ctx.builder.const_i32(t.shift as i32);
                    ctx.builder.shl_i32();
                }
                ctx.builder.add_i32();
                ctx.builder.load_unaligned_v128_zero64(simd_extra);
                ctx.builder.set_local_v128(vec_local(dst));
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi |= 1 << r;
                }
            },
            Op::Load128(r, mem) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.load_unaligned_v128(rel + simd_extra);
                ctx.builder
                    .set_local_v128(xmm_locals[r as usize].as_ref().unwrap());
            },
            Op::Store128(mem, r) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder
                    .get_local_v128(xmm_locals[r as usize].as_ref().unwrap());
                ctx.builder.store_unaligned_v128(rel + simd_extra);
            },
            Op::Load64Zero(dst, mem) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.load_unaligned_v128_zero64(rel + simd_extra);
                ctx.builder.set_local_v128(vec_local(dst));
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi |= 1 << r;
                }
            },
            Op::Store64(mem, src) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.get_local_v128(vec_local(src));
                ctx.builder.store_unaligned_v128_lane64(rel + simd_extra, 0);
            },
            Op::Load32Zero(dst, mem) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.load_unaligned_v128_zero32(rel + simd_extra);
                ctx.builder.set_local_v128(vec_local(dst));
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi |= 1 << r;
                }
            },
            Op::Store32(mem, src) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.get_local_v128(vec_local(src));
                ctx.builder.extract_i32x4_lane(0);
                ctx.builder.store_unaligned_i32(rel + simd_extra);
            },
            Op::MovVec(dst, src) => {
                if dst != src {
                    ctx.builder.get_local_v128(vec_local(src));
                    ctx.builder.set_local_v128(vec_local(dst));
                    if let VecReg::Mm(dr) = dst {
                        if let VecReg::Mm(sr) = src {
                            if mm_zero_hi & (1 << sr) != 0 {
                                mm_zero_hi |= 1 << dr;
                            }
                            else {
                                mm_zero_hi &= !(1 << dr);
                            }
                        }
                        else {
                            mm_zero_hi &= !(1 << dr);
                        }
                    }
                }
            },
            Op::MovqXmmReg(dst, src) => {
                ctx.builder
                    .get_local_v128(xmm_locals[src as usize].as_ref().unwrap());
                ctx.builder.const_i64(0);
                ctx.builder.replace_i64x2_lane(1);
                ctx.builder
                    .set_local_v128(xmm_locals[dst as usize].as_ref().unwrap());
            },
            Op::Movq2dq(dst, src) => {
                ctx.builder
                    .get_local_v128(mm_locals[src as usize].as_ref().unwrap());
                if mm_zero_hi & (1 << src) == 0 {
                    ctx.builder.const_i64(0);
                    ctx.builder.replace_i64x2_lane(1);
                }
                ctx.builder
                    .set_local_v128(xmm_locals[dst as usize].as_ref().unwrap());
            },
            Op::Movdq2q(dst, src) => {
                ctx.builder
                    .get_local_v128(xmm_locals[src as usize].as_ref().unwrap());
                ctx.builder
                    .set_local_v128(mm_locals[dst as usize].as_ref().unwrap());
                mm_zero_hi &= !(1 << dst);
            },
            Op::MovdFromGpr(dst, gpr) => {
                ctx.builder.v128_const([0; 16]);
                ctx.builder.get_local(&ctx.register_locals[gpr as usize]);
                ctx.builder.replace_i32x4_lane(0);
                ctx.builder.set_local_v128(vec_local(dst));
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi |= 1 << r;
                }
            },
            Op::MovdToGpr(gpr, src) => {
                ctx.builder.get_local_v128(vec_local(src));
                ctx.builder.extract_i32x4_lane(0);
                ctx.builder.set_local(&ctx.register_locals[gpr as usize]);
            },
            Op::BinReg(binop, dst, src) => {
                let is_sse = matches!(dst, VecReg::Xmm(_));
                if matches!(binop, SimdBinOp::Xor | SimdBinOp::AndNot) && dst == src {
                    ctx.builder.v128_const([0; 16]);
                    ctx.builder.set_local_v128(vec_local(dst));
                    if let VecReg::Mm(r) = dst {
                        mm_zero_hi |= 1 << r;
                    }
                }
                else {
                    let src_l = vec_local(src);
                    let dst_l = vec_local(dst);
                    emit_simd_binop(ctx.builder, binop, is_sse, dst_l, |b| {
                        b.get_local_v128(src_l)
                    });
                    if let VecReg::Mm(r) = dst {
                        mm_zero_hi &= !(1 << r);
                    }
                }
            },
            Op::BinMem(binop, dst, mem) => {
                let is_sse = matches!(dst, VecReg::Xmm(_));
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                let ptr_l = ptr_locals[mem.stream].as_ref().unwrap();
                let dst_l = vec_local(dst);
                emit_simd_binop(ctx.builder, binop, is_sse, dst_l, |b| {
                    b.get_local(ptr_l);
                    if is_sse {
                        b.load_unaligned_v128(rel + simd_extra);
                    }
                    else {
                        b.load_unaligned_v128_zero64(rel + simd_extra);
                    }
                });
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi &= !(1 << r);
                }
            },
            Op::ShiftImm(kind, dst, imm8) => {
                let dst_l = vec_local(dst);
                let imm = imm8 as u32;
                match kind {
                    ShiftKind::Shl16 => {
                        if imm >= 16 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.const_i32(imm as i32);
                            ctx.builder.shl_i16x8();
                        }
                    },
                    ShiftKind::ShrU16 => {
                        if imm >= 16 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.const_i32(imm as i32);
                            ctx.builder.shr_u_i16x8();
                        }
                    },
                    ShiftKind::ShrS16 => {
                        ctx.builder.get_local_v128(dst_l);
                        ctx.builder.const_i32(imm.min(15) as i32);
                        ctx.builder.shr_s_i16x8();
                    },
                    ShiftKind::Shl32 => {
                        if imm >= 32 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.const_i32(imm as i32);
                            ctx.builder.shl_i32x4();
                        }
                    },
                    ShiftKind::ShrU32 => {
                        if imm >= 32 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.const_i32(imm as i32);
                            ctx.builder.shr_u_i32x4();
                        }
                    },
                    ShiftKind::ShrS32 => {
                        ctx.builder.get_local_v128(dst_l);
                        ctx.builder.const_i32(imm.min(31) as i32);
                        ctx.builder.shr_s_i32x4();
                    },
                    ShiftKind::Shl64 => {
                        if imm >= 64 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.const_i32(imm as i32);
                            ctx.builder.shl_i64x2();
                        }
                    },
                    ShiftKind::ShrU64 => {
                        if imm >= 64 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.const_i32(imm as i32);
                            ctx.builder.shr_u_i64x2();
                        }
                    },
                    ShiftKind::ByteShr => {
                        if imm == 0 {
                            ctx.builder.get_local_v128(dst_l);
                        }
                        else if imm >= 16 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.v128_const([0; 16]);
                            let mut lanes = [16u8; 16];
                            for i in 0..(16 - imm as usize) {
                                lanes[i] = (i + imm as usize) as u8;
                            }
                            ctx.builder.i8x16_shuffle(lanes);
                        }
                    },
                    ShiftKind::ByteShl => {
                        if imm == 0 {
                            ctx.builder.get_local_v128(dst_l);
                        }
                        else if imm >= 16 {
                            ctx.builder.v128_const([0; 16]);
                        }
                        else {
                            ctx.builder.get_local_v128(dst_l);
                            ctx.builder.v128_const([0; 16]);
                            let mut lanes = [16u8; 16];
                            for i in (imm as usize)..16 {
                                lanes[i] = (i - imm as usize) as u8;
                            }
                            ctx.builder.i8x16_shuffle(lanes);
                        }
                    },
                }
                ctx.builder.set_local_v128(dst_l);
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi &= !(1 << r);
                }
            },
            Op::ShuffleReg(kind, dst, src, imm8) => {
                let src_l = vec_local(src);
                ctx.builder.get_local_v128(src_l);
                ctx.builder.get_local_v128(src_l);
                emit_shuffle(ctx.builder, kind, imm8);
                ctx.builder.set_local_v128(vec_local(dst));
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi &= !(1 << r);
                }
            },
            Op::ShuffleMem(kind, dst, mem, imm8) => {
                let is_sse = matches!(dst, VecReg::Xmm(_));
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                if is_sse {
                    ctx.builder.load_unaligned_v128(rel + simd_extra);
                }
                else {
                    ctx.builder.load_unaligned_v128_zero64(rel + simd_extra);
                }
                let v = ctx.builder.tee_new_local_v128();
                ctx.builder.get_local_v128(&v);
                emit_shuffle(ctx.builder, kind, imm8);
                ctx.builder.set_local_v128(vec_local(dst));
                ctx.builder.free_local_v128(v);
                if let VecReg::Mm(r) = dst {
                    mm_zero_hi &= !(1 << r);
                }
            },
            Op::GprLoad32(r, mem) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.load_unaligned_i32(rel + simd_extra);
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprLoadInv32(r, slot) => {
                ctx.builder.get_local(&inv_val_locals[slot]);
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprLoadZx8(r, mem) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.load_u8(rel + simd_extra);
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprLoadZx16(r, mem) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.load_unaligned_u16(rel + simd_extra);
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprStore32(mem, r) => {
                let rel = (mem.offset as i64 - plan.streams[mem.stream].min_offset as i64) as u32;
                ctx.builder
                    .get_local(ptr_locals[mem.stream].as_ref().unwrap());
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.store_unaligned_i32(rel + simd_extra);
            },
            Op::GprMovReg(dst, src) => {
                if dst != src {
                    ctx.builder.get_local(&ctx.register_locals[src as usize]);
                    ctx.builder.set_local(&ctx.register_locals[dst as usize]);
                }
            },
            Op::GprMovImm(r, imm) => {
                ctx.builder.const_i32(imm);
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprAddImm(r, d) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(d);
                ctx.builder.add_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprOrImm(r, imm) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(imm);
                ctx.builder.or_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprAndImm(r, imm) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(imm);
                ctx.builder.and_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprXorImm(r, imm) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(imm);
                ctx.builder.xor_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprAddReg(dst, src) => {
                ctx.builder.get_local(&ctx.register_locals[dst as usize]);
                ctx.builder.get_local(&ctx.register_locals[src as usize]);
                ctx.builder.add_i32();
                ctx.builder.set_local(&ctx.register_locals[dst as usize]);
            },
            Op::GprSubReg(dst, src) => {
                ctx.builder.get_local(&ctx.register_locals[dst as usize]);
                ctx.builder.get_local(&ctx.register_locals[src as usize]);
                ctx.builder.sub_i32();
                ctx.builder.set_local(&ctx.register_locals[dst as usize]);
            },
            Op::GprXorReg(dst, src) => {
                if dst == src {
                    ctx.builder.const_i32(0);
                }
                else {
                    ctx.builder.get_local(&ctx.register_locals[dst as usize]);
                    ctx.builder.get_local(&ctx.register_locals[src as usize]);
                    ctx.builder.xor_i32();
                }
                ctx.builder.set_local(&ctx.register_locals[dst as usize]);
            },
            Op::GprShlImm(r, imm) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(imm as i32);
                ctx.builder.shl_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprShrUImm(r, imm) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(imm as i32);
                ctx.builder.shr_u_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprShrSImm(r, imm) => {
                ctx.builder.get_local(&ctx.register_locals[r as usize]);
                ctx.builder.const_i32(imm as i32);
                ctx.builder.shr_s_i32();
                ctx.builder.set_local(&ctx.register_locals[r as usize]);
            },
            Op::GprLea(dst, base, index, shift, disp) => {
                ctx.builder.const_i32(disp);
                if let Some(br) = base {
                    ctx.builder.get_local(&ctx.register_locals[br as usize]);
                    ctx.builder.add_i32();
                }
                if let Some(ir) = index {
                    ctx.builder.get_local(&ctx.register_locals[ir as usize]);
                    if shift != 0 {
                        ctx.builder.const_i32(shift as i32);
                        ctx.builder.shl_i32();
                    }
                    ctx.builder.add_i32();
                }
                ctx.builder.set_local(&ctx.register_locals[dst as usize]);
            },
        }
    }
    if plan.body_instructions > 0 && plan.body_end == plan.ops.len() {
        ctx.builder.block_end();
    }

    // Advance stream pointers for next inner iteration.
    for (s_idx, s) in plan.streams.iter().enumerate() {
        if s.used && s.stride != 0 {
            let ptr = ptr_locals[s_idx].as_ref().unwrap();
            ctx.builder.get_local(ptr);
            ctx.builder.const_i32(s.stride);
            ctx.builder.add_i32();
            ctx.builder.set_local(ptr);
        }
    }

    // inner_i -= 1; br_if inner_loop
    ctx.builder.get_local(&inner_i);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    ctx.builder.tee_local(&inner_i);
    ctx.builder.br_if(inner_loop);
    ctx.builder.block_end(); // inner_loop

    // 7. Apply deferred induction GPR updates once per page batch.
    for r in 0..8 {
        if plan.deferred_gpr_mask & (1 << r) != 0 && plan.deltas[r] != 0 {
            let d = plan.deltas[r];
            ctx.builder.get_local(&ctx.register_locals[r]);
            ctx.builder.get_local(&batch);
            if d == 1 {
                ctx.builder.add_i32();
            }
            else if d == -1 {
                ctx.builder.sub_i32();
            }
            else if d > 0 && (d as u32).is_power_of_two() {
                ctx.builder.const_i32(d.trailing_zeros() as i32);
                ctx.builder.shl_i32();
                ctx.builder.add_i32();
            }
            else if d < 0 && ((-d) as u32).is_power_of_two() {
                ctx.builder.const_i32((-d).trailing_zeros() as i32);
                ctx.builder.shl_i32();
                ctx.builder.sub_i32();
            }
            else {
                ctx.builder.const_i32(d);
                ctx.builder.mul_i32();
                ctx.builder.add_i32();
            }
            ctx.builder.set_local(&ctx.register_locals[r]);
        }
    }

    // done_iters += batch; remaining -= batch; if remaining != 0 br outer_loop
    ctx.builder.get_local(&done_iters);
    ctx.builder.get_local(&batch);
    ctx.builder.add_i32();
    ctx.builder.set_local(&done_iters);

    ctx.builder.get_local(&remaining);
    ctx.builder.get_local(&batch);
    ctx.builder.sub_i32();
    ctx.builder.tee_local(&remaining);
    ctx.builder.br_if(outer_loop);

    ctx.builder.block_end(); // outer_loop
    ctx.builder.block_end(); // outer_done

    // 8. Flush architectural XMM, MMX, MMX tag/TOP, and retired instruction count once.
    ctx.builder.get_local(&done_iters);
    ctx.builder.if_void();
    {
        for r in 0..8u8 {
            if plan.xmm_written & (1 << r) != 0 {
                ctx.builder
                    .const_i32(gp::get_reg_xmm_offset(r as u32) as i32);
                ctx.builder
                    .get_local_v128(xmm_locals[r as usize].as_ref().unwrap());
                ctx.builder.store_aligned_v128(0);
            }
        }
        // MMX register/tag/TOP side effects only exist if an MMX instruction
        // actually executed: unconditional when MMX ops live in head/tail, or
        // gated on iterations that ran a body containing the MMX ops.
        if plan.touched_mmx {
            if plan.mmx_in_head_tail {
                ctx.builder.get_local(&done_iters);
            }
            else {
                ctx.builder.get_local(body_iters.as_ref().unwrap());
            }
            ctx.builder.if_void();
            for r in 0..8u8 {
                if plan.mm_written & (1 << r) != 0 {
                    let dest = gp::get_reg_mmx_offset(r as u32);
                    ctx.builder.const_i32(dest as i32);
                    ctx.builder
                        .get_local_v128(mm_locals[r as usize].as_ref().unwrap());
                    ctx.builder.store_aligned_v128_lane64(0, 0);
                    ctx.builder.const_i32(0);
                    ctx.builder.const_i32(0xFFFF);
                    ctx.builder.store_aligned_u16(dest + 8);
                }
            }
            ctx.builder.const_i32(0);
            ctx.builder.const_i32(0);
            ctx.builder.store_u8(gp::fpu_stack_empty as u32);
            ctx.builder.const_i32(0);
            ctx.builder.const_i32(0);
            ctx.builder.store_u8(gp::fpu_stack_ptr as u32);
            ctx.builder.block_end();
        }
        // Charge head+tail instructions for every fast iteration plus the
        // body's instructions for each iteration that ran it.
        ctx.builder.get_local(&ctx.instruction_counter);
        ctx.builder.get_local(&done_iters);
        ctx.builder
            .const_i32((plan.num_instructions - plan.body_instructions) as i32);
        ctx.builder.mul_i32();
        ctx.builder.add_i32();
        if plan.body_instructions > 0 {
            ctx.builder.get_local(body_iters.as_ref().unwrap());
            ctx.builder.const_i32(plan.body_instructions as i32);
            ctx.builder.mul_i32();
            ctx.builder.add_i32();
        }
        ctx.builder.set_local(&ctx.instruction_counter);
    }
    ctx.builder.block_end();

    ctx.builder.block_end(); // skip_fast

    // Free all temporary locals.
    ctx.builder.free_local(inner_i);
    ctx.builder.free_local(batch);
    for l in ptr_locals.into_iter().flatten() {
        ctx.builder.free_local(l);
    }
    for l in mm_locals.into_iter().flatten() {
        ctx.builder.free_local_v128(l);
    }
    for l in xmm_locals.into_iter().flatten() {
        ctx.builder.free_local_v128(l);
    }
    if let Some(l) = body_iters {
        ctx.builder.free_local(l);
    }
    if let Some(l) = skip_local {
        ctx.builder.free_local(l);
    }
    ctx.builder.free_local(done_iters);
    for l in table_ptr_locals {
        ctx.builder.free_local(l);
    }
    for l in table_page_locals {
        ctx.builder.free_local(l);
    }
    for l in inv_page_locals {
        ctx.builder.free_local(l);
    }
    for l in inv_val_locals {
        ctx.builder.free_local(l);
    }
    ctx.builder.free_local(remaining);
}
