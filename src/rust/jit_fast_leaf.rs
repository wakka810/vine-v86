// General pre-guarded tiny leaf CALL rel32 inliner for flat 32-bit CPL3 code
// under Vine common memory.
//
// Invariant: Zero architectural state (GPRs, guest memory, EFLAGS, lazy flags,
// instruction_counter, EIP) is mutated before all guards succeed. On any failed
// guard (instruction budget, leaf page map/execute, return slot writable,
// memory access permissions/page-crossing, return-slot/write aliasing, or cold
// branch), execution branches to the slow path, which executes the original
// CALL rel32 and chains/exits identically to non-inlined code.
use crate::codegen::{self, ConditionNegate};
use crate::cpu::cpu::{
    FLAGS_ALL, FLAG_ADJUST, FLAG_CARRY, FLAG_OVERFLOW, FLAG_PARITY, FLAG_SUB, OPSIZE_32,
};
use crate::cpu::global_pointers;
use crate::cpu::memory;
use crate::cpu_context::CpuContext;
use crate::jit::{
    gen_chain_to_cached_entry, is_near_end_of_page, BasicBlock, Instruction, JitContext,
};
use crate::jit_instructions;
use crate::modrm;
use crate::page::Page;
use crate::regs;
use crate::state_flags::CachedStateFlags;
use crate::wasmgen::wasm_builder::{Label, WasmLocal};

const MAX_LEAF_INSNS: usize = 14;
const MAX_LEAF_BYTES: u32 = 96;
const MAX_PENDING_WRITES: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
struct MemAddr {
    base: Option<u8>,
    index: Option<u8>,
    shift: u8,
    disp: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MemKey {
    base: Option<(u8, u8)>,
    index: Option<(u8, u8)>,
    shift: u8,
    disp: i32,
    size: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AluOp {
    Add,
    Sub,
    Cmp,
    And,
    Or,
    Xor,
    Test,
}

impl AluOp {
    fn writes_dest(self) -> bool { !matches!(self, Self::Cmp | Self::Test) }
}

#[derive(Clone, Copy)]
enum LeafOp {
    MovRegReg32 {
        dst: u8,
        src: u8,
    },
    MovRegImm32 {
        dst: u8,
        imm: i32,
    },
    MovRegMem32 {
        dst: u8,
        addr: MemAddr,
    },
    MovMemReg32 {
        addr: MemAddr,
        src: u8,
    },
    MovMemImm32 {
        addr: MemAddr,
        imm: i32,
    },
    MovReg8Mem8 {
        dst8: u8,
        addr: MemAddr,
    },
    MovMem8Reg8 {
        addr: MemAddr,
        src8: u8,
    },
    MovReg8Imm8 {
        dst8: u8,
        imm: u8,
    },
    MovReg8Reg8 {
        dst8: u8,
        src8: u8,
    },
    Movzx8 {
        dst: u8,
        src_mem: Option<MemAddr>,
        src_reg8: u8,
    },
    Movzx16 {
        dst: u8,
        src_mem: Option<MemAddr>,
        src_reg16: u8,
    },
    Lea32 {
        dst: u8,
        addr: MemAddr,
    },
    IncReg32 {
        dst: u8,
    },
    DecReg32 {
        dst: u8,
    },
    AluRegReg32 {
        op: AluOp,
        dst: u8,
        src: u8,
    },
    AluRegImm32 {
        op: AluOp,
        dst: u8,
        imm: i32,
    },
    AluRegMem32 {
        op: AluOp,
        dst: u8,
        addr: MemAddr,
    },
    AluMemReg32 {
        op: AluOp,
        addr: MemAddr,
        src: u8,
    },
    AluMemImm32 {
        op: AluOp,
        addr: MemAddr,
        imm: i32,
    },
    SbbSelf32 {
        dst: u8,
    },
    ColdJcc {
        cond: u8,
    },
    SkipDiamond {
        cond: u8,
        skipped_len: u8,
    },
}

impl LeafOp {
    fn is_pure_read_only_no_flags(self) -> bool {
        matches!(
            self,
            Self::MovRegReg32 { .. }
                | Self::MovRegImm32 { .. }
                | Self::MovRegMem32 { .. }
                | Self::MovReg8Mem8 { .. }
                | Self::MovReg8Imm8 { .. }
                | Self::MovReg8Reg8 { .. }
                | Self::Movzx8 { .. }
                | Self::Movzx16 { .. }
                | Self::Lea32 { .. }
        )
    }
}

#[derive(Clone)]
pub struct LeafPlan {
    ops: Vec<LeafOp>,
    max_insns: u32,
    modified_regs_mask: u8,
    has_diamond: bool,
}

#[derive(Clone)]
pub struct InlinedCall {
    pub leaf_phys: u32,
    pub call_rel32: i32,
    pub plan: LeafPlan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AnalyzeFlagKind {
    None,
    SubCmp,
    Bitwise,
    AddIncDec,
    SbbSelf,
}

fn jcc_supported(kind: AnalyzeFlagKind, cond: u8) -> bool {
    match kind {
        AnalyzeFlagKind::SubCmp => {
            matches!(cond, 0x72..=0x79 | 0x7c..=0x7f)
        },
        AnalyzeFlagKind::Bitwise => {
            matches!(cond, 0x74..=0x79 | 0x7c..=0x7f)
        },
        AnalyzeFlagKind::AddIncDec => {
            matches!(cond, 0x74 | 0x75 | 0x78 | 0x79)
        },
        AnalyzeFlagKind::None | AnalyzeFlagKind::SbbSelf => false,
    }
}

fn decode_mem_no_esp(c: &mut CpuContext, modrm_byte: u8) -> Option<MemAddr> {
    let (base, index, shift, disp) = modrm::decode(c, modrm_byte).flat_parts()?;
    let base = base.map(|r| r as u8);
    let index = index.map(|r| r as u8);
    if base == Some(regs::ESP as u8) || index == Some(regs::ESP as u8) {
        return None;
    }
    Some(MemAddr {
        base,
        index,
        shift,
        disp,
    })
}

fn alu_from_group(group: u8) -> Option<AluOp> {
    match group {
        0 => Some(AluOp::Add),
        1 => Some(AluOp::Or),
        4 => Some(AluOp::And),
        5 => Some(AluOp::Sub),
        6 => Some(AluOp::Xor),
        7 => Some(AluOp::Cmp),
        _ => None,
    }
}

pub fn analyze(leaf_phys: u32, call_rel32: i32, cpu: &CpuContext) -> Option<InlinedCall> {
    if cfg!(feature = "profiler")
        || !cpu.state_flags.is_32()
        || !cpu.ssize_32()
        || !cpu.has_flat_segmentation()
        || !cpu.cpl3()
    {
        return None;
    }
    let (common_base, _) = unsafe { crate::cpu::vine::common_memory() }?;
    let mem_base = unsafe { memory::mem8 } as u32;
    if common_base < mem_base || leaf_phys < common_base - mem_base {
        return None;
    }

    let start_page = Page::page_of(leaf_phys);
    let mut c = CpuContext {
        eip: leaf_phys,
        prefixes: 0,
        ..cpu.clone()
    };

    let mut ops: Vec<LeafOp> = Vec::with_capacity(MAX_LEAF_INSNS);
    let mut op_starts: Vec<u32> = Vec::with_capacity(MAX_LEAF_INSNS);
    let mut modified_regs_mask: u8 = 0;
    let mut write_count: usize = 0;
    let mut flag_kind = AnalyzeFlagKind::None;
    let mut has_leaf_cf = false;
    let mut pending_jcc: Option<(usize, u32, u8)> = None; // (op_idx, target_eip, cond)
    let mut ret_end_eip: Option<u32> = None;

    for _ in 0..MAX_LEAF_INSNS {
        if Page::page_of(c.eip) != start_page
            || is_near_end_of_page(c.eip)
            || c.eip - leaf_phys >= MAX_LEAF_BYTES
        {
            return None;
        }
        let insn_start = c.eip;
        let opcode = c.read_imm8();

        if opcode == 0xC3 {
            ret_end_eip = Some(c.eip);
            break;
        }
        if opcode == 0xF3 {
            if c.read_imm8() == 0xC3 {
                ret_end_eip = Some(c.eip);
                break;
            }
            return None;
        }

        let op = match opcode {
            0x40..=0x47 => {
                let dst = opcode & 7;
                if dst == regs::ESP as u8 {
                    return None;
                }
                modified_regs_mask |= 1 << dst;
                flag_kind = AnalyzeFlagKind::AddIncDec;
                LeafOp::IncReg32 { dst }
            },
            0x48..=0x4F => {
                let dst = opcode & 7;
                if dst == regs::ESP as u8 {
                    return None;
                }
                modified_regs_mask |= 1 << dst;
                flag_kind = AnalyzeFlagKind::AddIncDec;
                LeafOp::DecReg32 { dst }
            },
            0xB0..=0xB7 => {
                let dst8 = opcode & 7;
                let imm = c.read_imm8();
                modified_regs_mask |= 1 << (dst8 & 3);
                LeafOp::MovReg8Imm8 { dst8, imm }
            },
            0xB8..=0xBF => {
                let dst = opcode & 7;
                if dst == regs::ESP as u8 {
                    return None;
                }
                let imm = c.read_imm32() as i32;
                modified_regs_mask |= 1 << dst;
                LeafOp::MovRegImm32 { dst, imm }
            },
            0x88 | 0x8A => {
                let byte = c.read_imm8();
                let r = (byte >> 3) & 7;
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    if opcode == 0x8A {
                        modified_regs_mask |= 1 << (r & 3);
                        LeafOp::MovReg8Mem8 { dst8: r, addr }
                    }
                    else {
                        write_count += 1;
                        LeafOp::MovMem8Reg8 { addr, src8: r }
                    }
                }
                else {
                    let rm = byte & 7;
                    let (dst8, src8) = if opcode == 0x88 { (rm, r) } else { (r, rm) };
                    modified_regs_mask |= 1 << (dst8 & 3);
                    LeafOp::MovReg8Reg8 { dst8, src8 }
                }
            },
            0x89 | 0x8B => {
                let byte = c.read_imm8();
                let r = (byte >> 3) & 7;
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    if opcode == 0x8B {
                        if r == regs::ESP as u8 {
                            return None;
                        }
                        modified_regs_mask |= 1 << r;
                        LeafOp::MovRegMem32 { dst: r, addr }
                    }
                    else {
                        write_count += 1;
                        LeafOp::MovMemReg32 { addr, src: r }
                    }
                }
                else {
                    let rm = byte & 7;
                    let (dst, src) = if opcode == 0x89 { (rm, r) } else { (r, rm) };
                    if dst == regs::ESP as u8 {
                        return None;
                    }
                    modified_regs_mask |= 1 << dst;
                    LeafOp::MovRegReg32 { dst, src }
                }
            },
            0xC7 => {
                let byte = c.read_imm8();
                if (byte >> 3) & 7 != 0 {
                    return None;
                }
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    let imm = c.read_imm32() as i32;
                    write_count += 1;
                    LeafOp::MovMemImm32 { addr, imm }
                }
                else {
                    let dst = byte & 7;
                    if dst == regs::ESP as u8 {
                        return None;
                    }
                    let imm = c.read_imm32() as i32;
                    modified_regs_mask |= 1 << dst;
                    LeafOp::MovRegImm32 { dst, imm }
                }
            },
            0x8D => {
                let byte = c.read_imm8();
                if byte >= 0xC0 {
                    return None;
                }
                let dst = (byte >> 3) & 7;
                if dst == regs::ESP as u8 {
                    return None;
                }
                let addr = decode_mem_no_esp(&mut c, byte)?;
                modified_regs_mask |= 1 << dst;
                LeafOp::Lea32 { dst, addr }
            },
            0x0F => {
                let op2 = c.read_imm8();
                match op2 {
                    0xB6 | 0xB7 => {
                        let byte = c.read_imm8();
                        let dst = (byte >> 3) & 7;
                        if dst == regs::ESP as u8 {
                            return None;
                        }
                        modified_regs_mask |= 1 << dst;
                        if byte < 0xC0 {
                            let addr = decode_mem_no_esp(&mut c, byte)?;
                            if op2 == 0xB6 {
                                LeafOp::Movzx8 {
                                    dst,
                                    src_mem: Some(addr),
                                    src_reg8: 0,
                                }
                            }
                            else {
                                LeafOp::Movzx16 {
                                    dst,
                                    src_mem: Some(addr),
                                    src_reg16: 0,
                                }
                            }
                        }
                        else {
                            let rm = byte & 7;
                            if op2 == 0xB6 {
                                LeafOp::Movzx8 {
                                    dst,
                                    src_mem: None,
                                    src_reg8: rm,
                                }
                            }
                            else {
                                LeafOp::Movzx16 {
                                    dst,
                                    src_mem: None,
                                    src_reg16: rm,
                                }
                            }
                        }
                    },
                    _ => return None,
                }
            },
            0x01 | 0x03 | 0x09 | 0x0B | 0x21 | 0x23 | 0x29 | 0x2B | 0x31 | 0x33 | 0x39 | 0x3B => {
                let alu = alu_from_group((opcode >> 3) & 7)?;
                let dir_reg_dst = (opcode & 2) != 0;
                let byte = c.read_imm8();
                let r = (byte >> 3) & 7;
                flag_kind = match alu {
                    AluOp::Sub | AluOp::Cmp => AnalyzeFlagKind::SubCmp,
                    AluOp::Add => AnalyzeFlagKind::AddIncDec,
                    AluOp::And | AluOp::Or | AluOp::Xor | AluOp::Test => AnalyzeFlagKind::Bitwise,
                };
                has_leaf_cf = true;
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    if dir_reg_dst {
                        if alu.writes_dest() {
                            if r == regs::ESP as u8 {
                                return None;
                            }
                            modified_regs_mask |= 1 << r;
                        }
                        LeafOp::AluRegMem32 {
                            op: alu,
                            dst: r,
                            addr,
                        }
                    }
                    else {
                        if alu.writes_dest() {
                            write_count += 1;
                        }
                        LeafOp::AluMemReg32 {
                            op: alu,
                            addr,
                            src: r,
                        }
                    }
                }
                else {
                    let rm = byte & 7;
                    let (dst, src) = if dir_reg_dst { (r, rm) } else { (rm, r) };
                    if alu.writes_dest() {
                        if dst == regs::ESP as u8 {
                            return None;
                        }
                        modified_regs_mask |= 1 << dst;
                    }
                    LeafOp::AluRegReg32 { op: alu, dst, src }
                }
            },
            0x05 | 0x0D | 0x25 | 0x2D | 0x35 | 0x3D | 0xA9 => {
                let alu =
                    if opcode == 0xA9 { AluOp::Test } else { alu_from_group((opcode >> 3) & 7)? };
                let imm = c.read_imm32() as i32;
                if alu.writes_dest() {
                    modified_regs_mask |= 1 << regs::EAX;
                }
                flag_kind = match alu {
                    AluOp::Sub | AluOp::Cmp => AnalyzeFlagKind::SubCmp,
                    AluOp::Add => AnalyzeFlagKind::AddIncDec,
                    AluOp::And | AluOp::Or | AluOp::Xor | AluOp::Test => AnalyzeFlagKind::Bitwise,
                };
                has_leaf_cf = true;
                LeafOp::AluRegImm32 {
                    op: alu,
                    dst: regs::EAX as u8,
                    imm,
                }
            },
            0x81 | 0x83 => {
                let byte = c.read_imm8();
                let alu = alu_from_group((byte >> 3) & 7)?;
                flag_kind = match alu {
                    AluOp::Sub | AluOp::Cmp => AnalyzeFlagKind::SubCmp,
                    AluOp::Add => AnalyzeFlagKind::AddIncDec,
                    AluOp::And | AluOp::Or | AluOp::Xor | AluOp::Test => AnalyzeFlagKind::Bitwise,
                };
                has_leaf_cf = true;
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    let imm =
                        if opcode == 0x83 { c.read_imm8s() as i32 } else { c.read_imm32() as i32 };
                    if alu.writes_dest() {
                        write_count += 1;
                    }
                    LeafOp::AluMemImm32 { op: alu, addr, imm }
                }
                else {
                    let dst = byte & 7;
                    let imm =
                        if opcode == 0x83 { c.read_imm8s() as i32 } else { c.read_imm32() as i32 };
                    if alu.writes_dest() {
                        if dst == regs::ESP as u8 {
                            return None;
                        }
                        modified_regs_mask |= 1 << dst;
                    }
                    LeafOp::AluRegImm32 { op: alu, dst, imm }
                }
            },
            0x85 => {
                let byte = c.read_imm8();
                let r = (byte >> 3) & 7;
                flag_kind = AnalyzeFlagKind::Bitwise;
                has_leaf_cf = true;
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    LeafOp::AluRegMem32 {
                        op: AluOp::Test,
                        dst: r,
                        addr,
                    }
                }
                else {
                    let rm = byte & 7;
                    LeafOp::AluRegReg32 {
                        op: AluOp::Test,
                        dst: rm,
                        src: r,
                    }
                }
            },
            0xF7 => {
                let byte = c.read_imm8();
                if (byte >> 3) & 7 != 0 {
                    return None;
                }
                flag_kind = AnalyzeFlagKind::Bitwise;
                has_leaf_cf = true;
                if byte < 0xC0 {
                    let addr = decode_mem_no_esp(&mut c, byte)?;
                    let imm = c.read_imm32() as i32;
                    LeafOp::AluMemImm32 {
                        op: AluOp::Test,
                        addr,
                        imm,
                    }
                }
                else {
                    let dst = byte & 7;
                    let imm = c.read_imm32() as i32;
                    LeafOp::AluRegImm32 {
                        op: AluOp::Test,
                        dst,
                        imm,
                    }
                }
            },
            0x19 | 0x1B => {
                let byte = c.read_imm8();
                if byte < 0xC0 || !has_leaf_cf {
                    return None;
                }
                let r = (byte >> 3) & 7;
                let rm = byte & 7;
                if r != rm || r == regs::ESP as u8 {
                    return None;
                }
                modified_regs_mask |= 1 << r;
                flag_kind = AnalyzeFlagKind::SbbSelf;
                LeafOp::SbbSelf32 { dst: r }
            },
            0x70..=0x79 | 0x7C..=0x7F => {
                if pending_jcc.is_some() || !jcc_supported(flag_kind, opcode) {
                    return None;
                }
                let rel8 = c.read_imm8s() as i32;
                let target = c.eip.wrapping_add(rel8 as u32);
                pending_jcc = Some((ops.len(), target, opcode));
                LeafOp::ColdJcc { cond: opcode }
            },
            _ => return None,
        };

        if write_count > MAX_PENDING_WRITES {
            return None;
        }
        op_starts.push(insn_start);
        ops.push(op);
    }

    let ret_end = ret_end_eip?;
    let ret_start = op_starts
        .last()
        .map(|_| {
            // Compute start of RET from end minus length (1 for C3, 2 for F3 C3).
            if ret_end >= 2
                && memory::read8(ret_end - 2) == 0xF3
                && memory::read8(ret_end - 1) == 0xC3
            {
                ret_end - 2
            }
            else {
                ret_end - 1
            }
        })
        .unwrap_or(leaf_phys);

    let mut has_diamond = false;
    if let Some((jcc_idx, target, cond)) = pending_jcc {
        if target < leaf_phys || target >= ret_end {
            // Cold exit outside the leaf (e.g. __security_check_cookie failure).
            ops[jcc_idx] = LeafOp::ColdJcc { cond };
        }
        else if target == ret_start {
            let skipped = ops.len() - (jcc_idx + 1);
            if skipped > 3
                || !ops[jcc_idx + 1..]
                    .iter()
                    .all(|op| op.is_pure_read_only_no_flags())
            {
                return None;
            }
            if skipped > 0 {
                ops[jcc_idx] = LeafOp::SkipDiamond {
                    cond,
                    skipped_len: skipped as u8,
                };
                has_diamond = true;
            }
            else {
                ops[jcc_idx] = LeafOp::SkipDiamond {
                    cond,
                    skipped_len: 0,
                };
            }
        }
        else if let Some(target_idx) = op_starts.iter().position(|&addr| addr == target) {
            if target_idx <= jcc_idx {
                return None;
            }
            let skipped = target_idx - (jcc_idx + 1);
            if skipped > 3
                || !ops[jcc_idx + 1..target_idx]
                    .iter()
                    .all(|op| op.is_pure_read_only_no_flags())
            {
                return None;
            }
            ops[jcc_idx] = LeafOp::SkipDiamond {
                cond,
                skipped_len: skipped as u8,
            };
            has_diamond = skipped > 0;
        }
        else {
            return None;
        }
    }

    let max_insns = (ops.len() + 1) as u32; // +1 for RET
    Some(InlinedCall {
        leaf_phys,
        call_rel32,
        plan: LeafPlan {
            ops,
            max_insns,
            modified_regs_mask,
            has_diamond,
        },
    })
}

struct PendingWrite {
    addr: WasmLocal,
    val: WasmLocal,
    size: u8,
    key: MemKey,
}

enum CfState {
    Caller,
    Const(bool),
    Local(WasmLocal),
}

enum ShadowFlagState {
    None,
    Add32 {
        op1: WasmLocal,
        res: WasmLocal,
    },
    SubCmp32 {
        op1: WasmLocal,
        op2: WasmLocal,
        res: WasmLocal,
    },
    Inc32 {
        op1: WasmLocal,
        res: WasmLocal,
    },
    Dec32 {
        op1: WasmLocal,
        res: WasmLocal,
    },
    Bitwise32 {
        res: WasmLocal,
    },
    SbbSelf32 {
        cf: WasmLocal,
    },
}

fn emit_slow_call(
    ctx: &mut JitContext,
    block: &BasicBlock,
    ic: &InlinedCall,
    state_flags: CachedStateFlags,
) {
    ctx.start_of_current_instruction = block.last_instruction_addr;
    ctx.cpu.eip = block.last_instruction_addr;
    ctx.cpu.prefixes = 0;
    let mut flags = 0;
    jit_instructions::jit_instruction(ctx, &mut flags);
    dbg_assert!(ctx.cpu.eip == block.end_addr);
    codegen::gen_set_eip_low_bits_and_jump_rel32(
        ctx.builder,
        block.end_addr as i32 & 0xFFF,
        ic.call_rel32,
    );
    codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
    gen_chain_to_cached_entry(ctx, state_flags);
    ctx.builder.br(ctx.exit_label);
}

fn get_reg32_shadow(ctx: &mut JitContext, shadow: &[Option<WasmLocal>; 8], r: u8) {
    if let Some(l) = &shadow[r as usize] {
        ctx.builder.get_local(l);
    }
    else {
        ctx.builder.get_local(&ctx.register_locals[r as usize]);
    }
}

fn get_reg8_shadow(ctx: &mut JitContext, shadow: &[Option<WasmLocal>; 8], r8: u8) {
    get_reg32_shadow(ctx, shadow, r8 & 3);
    if r8 >= 4 {
        ctx.builder.const_i32(8);
        ctx.builder.shr_u_i32();
    }
    ctx.builder.const_i32(0xFF);
    ctx.builder.and_i32();
}

fn set_reg8_shadow(ctx: &mut JitContext, shadow: &[Option<WasmLocal>; 8], r8: u8) {
    // Expects masked 0..255 value on top of Wasm stack.
    let dst = shadow[(r8 & 3) as usize].as_ref().unwrap();
    if r8 < 4 {
        ctx.builder.get_local(dst);
        ctx.builder.const_i32(!0xFF);
        ctx.builder.and_i32();
        ctx.builder.or_i32();
        ctx.builder.set_local(dst);
    }
    else {
        ctx.builder.const_i32(8);
        ctx.builder.shl_i32();
        ctx.builder.get_local(dst);
        ctx.builder.const_i32(!0xFF00);
        ctx.builder.and_i32();
        ctx.builder.or_i32();
        ctx.builder.set_local(dst);
    }
}

fn make_mem_key(addr: MemAddr, reg_vers: &[u8; 8], size: u8) -> MemKey {
    MemKey {
        base: addr.base.map(|r| (r, reg_vers[r as usize])),
        index: addr.index.map(|r| (r, reg_vers[r as usize])),
        shift: addr.shift,
        disp: addr.disp,
        size,
    }
}

fn emit_addr(ctx: &mut JitContext, shadow: &[Option<WasmLocal>; 8], addr: MemAddr) {
    let mut have_term = false;
    if let Some(b) = addr.base {
        get_reg32_shadow(ctx, shadow, b);
        have_term = true;
    }
    if let Some(idx) = addr.index {
        get_reg32_shadow(ctx, shadow, idx);
        if addr.shift != 0 {
            ctx.builder.const_i32(addr.shift as i32);
            ctx.builder.shl_i32();
        }
        if have_term {
            ctx.builder.add_i32();
        }
        have_term = true;
    }
    if !have_term {
        ctx.builder.const_i32(addr.disp);
    }
    else if addr.disp != 0 {
        ctx.builder.const_i32(addr.disp);
        ctx.builder.add_i32();
    }
}

fn emit_common_guard(
    ctx: &mut JitContext,
    addr_local: &WasmLocal,
    size: u8,
    write: bool,
    permissions: u32,
    slow: Label,
) {
    if size > 1 {
        ctx.builder.get_local(addr_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(4096 - size as i32);
        ctx.builder.gtu_i32();
        ctx.builder.br_if(slow);
    }
    ctx.builder.get_local(addr_local);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.load_u8(permissions);
    let mask = if write { 3 } else { 1 };
    ctx.builder.const_i32(mask);
    ctx.builder.and_i32();
    if write {
        ctx.builder.const_i32(3);
        ctx.builder.ne_i32();
    }
    else {
        ctx.builder.eqz_i32();
    }
    ctx.builder.br_if(slow);
}

fn emit_non_overlap_check(
    ctx: &mut JitContext,
    a_local: &WasmLocal,
    a_size: u8,
    b_local: &WasmLocal,
    b_size: u8,
    slow: Label,
) {
    // Overlap iff ((a - b + (a_size - 1)) as u32) < (a_size + b_size - 1)
    ctx.builder.get_local(a_local);
    ctx.builder.get_local(b_local);
    ctx.builder.sub_i32();
    if a_size > 1 {
        ctx.builder.const_i32((a_size - 1) as i32);
        ctx.builder.add_i32();
    }
    ctx.builder.const_i32((a_size + b_size - 1) as i32);
    ctx.builder.ltu_i32();
    ctx.builder.br_if(slow);
}

#[derive(Clone, Copy)]
enum StaticAlias {
    Exact,
    Disjoint,
    Partial,
    Unknown,
}

fn classify_alias(a: MemKey, b: MemKey) -> StaticAlias {
    if a.base == b.base && a.index == b.index && a.shift == b.shift {
        if a.disp == b.disp && a.size == b.size {
            return StaticAlias::Exact;
        }
        let diff = (a.disp.wrapping_sub(b.disp)).wrapping_add(a.size as i32 - 1) as u32;
        if diff >= (a.size + b.size - 1) as u32 {
            return StaticAlias::Disjoint;
        }
        return StaticAlias::Partial;
    }
    StaticAlias::Unknown
}

fn emit_guarded_read(
    ctx: &mut JitContext,
    shadow: &[Option<WasmLocal>; 8],
    reg_vers: &[u8; 8],
    pending_writes: &[PendingWrite],
    ret_slot_addr: &WasmLocal,
    addr: MemAddr,
    size: u8,
    common_base: u32,
    permissions: u32,
    slow: Label,
) {
    // Leaves the loaded value (u8 / u16 / i32) on the Wasm stack.
    let key = make_mem_key(addr, reg_vers, size);
    // Check most recent pending write first for static exact forwarding!
    let mut forwarded: Option<&WasmLocal> = None;
    let mut dynamic_checks: Vec<usize> = Vec::new();
    for (idx, pw) in pending_writes.iter().enumerate().rev() {
        match classify_alias(key, pw.key) {
            StaticAlias::Exact => {
                forwarded = Some(&pw.val);
                break;
            },
            StaticAlias::Disjoint => {},
            StaticAlias::Partial => {
                ctx.builder.br(slow);
                ctx.builder.const_i32(0);
                return;
            },
            StaticAlias::Unknown => {
                dynamic_checks.push(idx);
            },
        }
    }

    if let Some(val_local) = forwarded {
        if dynamic_checks.is_empty() {
            // Statically exact match with the most recent write and no newer unknown writes!
            ctx.builder.get_local(val_local);
            return;
        }
    }

    emit_addr(ctx, shadow, addr);
    let addr_local = ctx.builder.set_new_local();
    emit_common_guard(ctx, &addr_local, size, false, permissions, slow);
    // Ensure leaf read does not observe the uncommitted CALL return address slot [ESP-4..ESP).
    emit_non_overlap_check(ctx, &addr_local, size, ret_slot_addr, 4, slow);
    for idx in dynamic_checks {
        let pw = &pending_writes[idx];
        emit_non_overlap_check(ctx, &addr_local, size, &pw.addr, pw.size, slow);
    }

    if let Some(val_local) = forwarded {
        ctx.builder.get_local(val_local);
    }
    else {
        ctx.builder.get_local(&addr_local);
        ctx.builder.const_i32(common_base as i32);
        ctx.builder.add_i32();
        match size {
            1 => ctx.builder.load_u8(0),
            2 => ctx.builder.load_unaligned_u16(0),
            4 => ctx.builder.load_unaligned_i32(0),
            _ => unreachable!(),
        }
    }
    ctx.builder.free_local(addr_local);
}

fn emit_eval_jcc(ctx: &mut JitContext, flag_state: &ShadowFlagState, cond: u8) {
    // Pushes 1 if branch condition `cond` is TRUE, 0 if FALSE.
    match flag_state {
        ShadowFlagState::SubCmp32 { op1, op2, res } => match cond {
            0x72 => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.ltu_i32();
            },
            0x73 => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.geu_i32();
            },
            0x74 => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.eq_i32();
            },
            0x75 => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.ne_i32();
            },
            0x76 => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.leu_i32();
            },
            0x77 => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.gtu_i32();
            },
            0x78 => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.lt_i32();
            },
            0x79 => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.ge_i32();
            },
            0x7C => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.lt_i32();
            },
            0x7D => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.ge_i32();
            },
            0x7E => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.le_i32();
            },
            0x7F => {
                ctx.builder.get_local(op1);
                ctx.builder.get_local(op2);
                ctx.builder.gt_i32();
            },
            _ => unreachable!(),
        },
        ShadowFlagState::Bitwise32 { res } => match cond {
            0x74 | 0x76 => {
                ctx.builder.get_local(res);
                ctx.builder.eqz_i32();
            },
            0x75 | 0x77 => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.ne_i32();
            },
            0x78 | 0x7C => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.lt_i32();
            },
            0x79 | 0x7D => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.ge_i32();
            },
            0x7E => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.le_i32();
            },
            0x7F => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.gt_i32();
            },
            _ => unreachable!(),
        },
        ShadowFlagState::Add32 { res, .. }
        | ShadowFlagState::Inc32 { res, .. }
        | ShadowFlagState::Dec32 { res, .. } => match cond {
            0x74 => {
                ctx.builder.get_local(res);
                ctx.builder.eqz_i32();
            },
            0x75 => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.ne_i32();
            },
            0x78 => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.lt_i32();
            },
            0x79 => {
                ctx.builder.get_local(res);
                ctx.builder.const_i32(0);
                ctx.builder.ge_i32();
            },
            _ => unreachable!(),
        },
        ShadowFlagState::None | ShadowFlagState::SbbSelf32 { .. } => unreachable!(),
    }
}

fn free_flag_state(ctx: &mut JitContext, state: ShadowFlagState) {
    match state {
        ShadowFlagState::None => {},
        ShadowFlagState::Add32 { op1, res }
        | ShadowFlagState::Inc32 { op1, res }
        | ShadowFlagState::Dec32 { op1, res } => {
            ctx.builder.free_local(op1);
            ctx.builder.free_local(res);
        },
        ShadowFlagState::SubCmp32 { op1, op2, res } => {
            ctx.builder.free_local(op1);
            ctx.builder.free_local(op2);
            ctx.builder.free_local(res);
        },
        ShadowFlagState::Bitwise32 { res } => {
            ctx.builder.free_local(res);
        },
        ShadowFlagState::SbbSelf32 { cf } => {
            ctx.builder.free_local(cf);
        },
    }
}

fn free_cf_state(ctx: &mut JitContext, state: CfState) {
    if let CfState::Local(l) = state {
        ctx.builder.free_local(l);
    }
}

fn emit_pure_op(
    ctx: &mut JitContext,
    op: LeafOp,
    shadow: &[Option<WasmLocal>; 8],
    reg_vers: &mut [u8; 8],
    pending_writes: &[PendingWrite],
    ret_slot_addr: &WasmLocal,
    common_base: u32,
    permissions: u32,
    slow: Label,
) {
    match op {
        LeafOp::MovRegReg32 { dst, src } => {
            get_reg32_shadow(ctx, shadow, src);
            ctx.builder
                .set_local(shadow[dst as usize].as_ref().unwrap());
            reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
        },
        LeafOp::MovRegImm32 { dst, imm } => {
            ctx.builder.const_i32(imm);
            ctx.builder
                .set_local(shadow[dst as usize].as_ref().unwrap());
            reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
        },
        LeafOp::MovRegMem32 { dst, addr } => {
            emit_guarded_read(
                ctx,
                shadow,
                reg_vers,
                pending_writes,
                ret_slot_addr,
                addr,
                4,
                common_base,
                permissions,
                slow,
            );
            ctx.builder
                .set_local(shadow[dst as usize].as_ref().unwrap());
            reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
        },
        LeafOp::MovReg8Mem8 { dst8, addr } => {
            emit_guarded_read(
                ctx,
                shadow,
                reg_vers,
                pending_writes,
                ret_slot_addr,
                addr,
                1,
                common_base,
                permissions,
                slow,
            );
            set_reg8_shadow(ctx, shadow, dst8);
            reg_vers[(dst8 & 3) as usize] = reg_vers[(dst8 & 3) as usize].wrapping_add(1);
        },
        LeafOp::MovReg8Imm8 { dst8, imm } => {
            ctx.builder.const_i32(imm as i32);
            set_reg8_shadow(ctx, shadow, dst8);
            reg_vers[(dst8 & 3) as usize] = reg_vers[(dst8 & 3) as usize].wrapping_add(1);
        },
        LeafOp::MovReg8Reg8 { dst8, src8 } => {
            get_reg8_shadow(ctx, shadow, src8);
            set_reg8_shadow(ctx, shadow, dst8);
            reg_vers[(dst8 & 3) as usize] = reg_vers[(dst8 & 3) as usize].wrapping_add(1);
        },
        LeafOp::Movzx8 {
            dst,
            src_mem,
            src_reg8,
        } => {
            if let Some(addr) = src_mem {
                emit_guarded_read(
                    ctx,
                    shadow,
                    reg_vers,
                    pending_writes,
                    ret_slot_addr,
                    addr,
                    1,
                    common_base,
                    permissions,
                    slow,
                );
            }
            else {
                get_reg8_shadow(ctx, shadow, src_reg8);
            }
            ctx.builder
                .set_local(shadow[dst as usize].as_ref().unwrap());
            reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
        },
        LeafOp::Movzx16 {
            dst,
            src_mem,
            src_reg16,
        } => {
            if let Some(addr) = src_mem {
                emit_guarded_read(
                    ctx,
                    shadow,
                    reg_vers,
                    pending_writes,
                    ret_slot_addr,
                    addr,
                    2,
                    common_base,
                    permissions,
                    slow,
                );
            }
            else {
                get_reg32_shadow(ctx, shadow, src_reg16);
                ctx.builder.const_i32(0xFFFF);
                ctx.builder.and_i32();
            }
            ctx.builder
                .set_local(shadow[dst as usize].as_ref().unwrap());
            reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
        },
        LeafOp::Lea32 { dst, addr } => {
            emit_addr(ctx, shadow, addr);
            ctx.builder
                .set_local(shadow[dst as usize].as_ref().unwrap());
            reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
        },
        _ => unreachable!(),
    }
}

fn emit_alu_compute(
    ctx: &mut JitContext,
    alu: AluOp,
    lhs_local: &WasmLocal,
    rhs_local: &WasmLocal,
    flag_state: &mut ShadowFlagState,
    cf_state: &mut CfState,
    last_op1_preserved: &mut Option<WasmLocal>,
) -> WasmLocal {
    // Returns a new WasmLocal holding the ALU result (caller frees or stores).
    ctx.builder.get_local(lhs_local);
    ctx.builder.get_local(rhs_local);
    match alu {
        AluOp::Add => ctx.builder.add_i32(),
        AluOp::Sub | AluOp::Cmp => ctx.builder.sub_i32(),
        AluOp::And | AluOp::Test => ctx.builder.and_i32(),
        AluOp::Or => ctx.builder.or_i32(),
        AluOp::Xor => ctx.builder.xor_i32(),
    }
    let res = ctx.builder.set_new_local();

    let old_flag = std::mem::replace(flag_state, ShadowFlagState::None);
    free_flag_state(ctx, old_flag);
    let old_cf = std::mem::replace(cf_state, CfState::Caller);
    free_cf_state(ctx, old_cf);

    match alu {
        AluOp::Add => {
            ctx.builder.get_local(lhs_local);
            let op1 = ctx.builder.set_new_local();
            ctx.builder.get_local(&res);
            let res_copy = ctx.builder.set_new_local();
            // CF = (res as u32) < (op1 as u32)
            ctx.builder.get_local(&res);
            ctx.builder.get_local(&op1);
            ctx.builder.ltu_i32();
            *cf_state = CfState::Local(ctx.builder.set_new_local());
            if let Some(old) = last_op1_preserved.take() {
                ctx.builder.free_local(old);
            }
            ctx.builder.get_local(&op1);
            *last_op1_preserved = Some(ctx.builder.set_new_local());
            *flag_state = ShadowFlagState::Add32 { op1, res: res_copy };
        },
        AluOp::Sub | AluOp::Cmp => {
            ctx.builder.get_local(lhs_local);
            let op1 = ctx.builder.set_new_local();
            ctx.builder.get_local(rhs_local);
            let op2 = ctx.builder.set_new_local();
            ctx.builder.get_local(&res);
            let res_copy = ctx.builder.set_new_local();
            // CF = (op1 as u32) < (op2 as u32)
            ctx.builder.get_local(&op1);
            ctx.builder.get_local(&op2);
            ctx.builder.ltu_i32();
            *cf_state = CfState::Local(ctx.builder.set_new_local());
            if let Some(old) = last_op1_preserved.take() {
                ctx.builder.free_local(old);
            }
            ctx.builder.get_local(&op1);
            *last_op1_preserved = Some(ctx.builder.set_new_local());
            *flag_state = ShadowFlagState::SubCmp32 {
                op1,
                op2,
                res: res_copy,
            };
        },
        AluOp::And | AluOp::Or | AluOp::Xor | AluOp::Test => {
            ctx.builder.get_local(&res);
            let res_copy = ctx.builder.set_new_local();
            *cf_state = CfState::Const(false);
            *flag_state = ShadowFlagState::Bitwise32 { res: res_copy };
        },
    }
    res
}

pub fn generate(
    ctx: &mut JitContext,
    block: &BasicBlock,
    ic: &InlinedCall,
    state_flags: CachedStateFlags,
) {
    let common = unsafe { crate::cpu::vine::common_memory() }.filter(|_| {
        !cfg!(feature = "profiler")
            && ctx.cpu.cpl3()
            && ctx.cpu.state_flags.is_32()
            && ctx.cpu.ssize_32()
            && ctx.cpu.has_flat_segmentation()
    });
    let Some((common_base, permissions)) = common
    else {
        emit_slow_call(ctx, block, ic, state_flags);
        return;
    };
    let mem_base = unsafe { memory::mem8 } as u32;
    if common_base < mem_base || ic.leaf_phys < common_base - mem_base {
        emit_slow_call(ctx, block, ic, state_flags);
        return;
    }
    let guest_offset = common_base - mem_base;
    let expected_leaf_virt = ic.leaf_phys - guest_offset;

    let needed_locals = 18 + ic.plan.modified_regs_mask.count_ones() as u8;
    if !ctx.builder.can_allocate_temporary_locals(needed_locals) {
        emit_slow_call(ctx, block, ic, state_flags);
        return;
    }

    let done = ctx.builder.block_void();
    let slow = ctx.builder.block_void();

    // Guard 1: Exact instruction budget for all instructions in the leaf (including RET).
    // Note: The caller block's instruction_counter already includes CALL itself.
    if let Some(limit) = &ctx.instruction_limit {
        ctx.builder.get_local(&ctx.instruction_counter);
        ctx.builder.const_i32(ic.plan.max_insns as i32);
        ctx.builder.add_i32();
        ctx.builder.get_local(limit);
        ctx.builder.gtu_i32();
        ctx.builder.br_if(slow);
    }

    // Compute live return virtual address: (EIP & !0xFFF) | (block.end_addr & 0xFFF)
    codegen::gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(block.end_addr as i32 & 0xFFF);
    ctx.builder.or_i32();
    let ret_virt_addr = ctx.builder.tee_new_local();

    // Guard 2: Live target virtual address matches compiled leaf AND leaf page is readable+executable.
    ctx.builder.const_i32(ic.call_rel32);
    ctx.builder.add_i32();
    ctx.builder.const_i32(expected_leaf_virt as i32);
    ctx.builder.ne_i32();
    if Page::page_of(ic.leaf_phys) != Page::page_of(block.addr) {
        let leaf_page = expected_leaf_virt >> 12;
        let (exec_enabled_ptr, exec_bitmap_ptr) = crate::cpu::vine::execute_protection_pointers();
        ctx.builder.load_fixed_u8(permissions + leaf_page);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        ctx.builder.eqz_i32();
        ctx.builder.or_i32();
        ctx.builder.load_fixed_u8(exec_enabled_ptr);
        ctx.builder
            .load_fixed_u8(exec_bitmap_ptr + (leaf_page >> 3));
        ctx.builder.const_i32(1 << (leaf_page & 7));
        ctx.builder.and_i32();
        ctx.builder.eqz_i32();
        ctx.builder.and_i32();
        ctx.builder.or_i32();
    }
    ctx.builder.br_if(slow);

    // Guard 3: Return address slot [ESP - 4 .. ESP) is writable and within a single page.
    ctx.builder
        .get_local(&ctx.register_locals[regs::ESP as usize]);
    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();
    let ret_slot_addr = ctx.builder.set_new_local();
    emit_common_guard(ctx, &ret_slot_addr, 4, true, permissions, slow);

    // Allocate shadow locals ONLY for modified GPRs.
    let mut shadow: [Option<WasmLocal>; 8] = std::array::from_fn(|_| None);
    for r in 0..8usize {
        if (ic.plan.modified_regs_mask & (1 << r)) != 0 {
            ctx.builder.get_local(&ctx.register_locals[r]);
            shadow[r] = Some(ctx.builder.set_new_local());
        }
    }

    let actual_insns = if ic.plan.has_diamond {
        // Initialized to total instructions executed when the diamond branch IS taken (skips body).
        let skipped_total: u32 = ic
            .plan
            .ops
            .iter()
            .filter_map(|op| match *op {
                LeafOp::SkipDiamond { skipped_len, .. } => Some(skipped_len as u32),
                _ => None,
            })
            .sum();
        ctx.builder
            .const_i32((ic.plan.max_insns - skipped_total) as i32);
        Some(ctx.builder.set_new_local())
    }
    else {
        None
    };

    let mut reg_vers = [0u8; 8];
    let mut pending_writes: Vec<PendingWrite> = Vec::with_capacity(MAX_PENDING_WRITES);
    let mut flag_state = ShadowFlagState::None;
    let mut cf_state = CfState::Caller;
    let mut last_op1_preserved: Option<WasmLocal> = None;

    let ops = &ic.plan.ops;
    let mut i = 0;
    while i < ops.len() {
        let op = ops[i];
        match op {
            LeafOp::MovRegReg32 { .. }
            | LeafOp::MovRegImm32 { .. }
            | LeafOp::MovRegMem32 { .. }
            | LeafOp::MovReg8Mem8 { .. }
            | LeafOp::MovReg8Imm8 { .. }
            | LeafOp::MovReg8Reg8 { .. }
            | LeafOp::Movzx8 { .. }
            | LeafOp::Movzx16 { .. }
            | LeafOp::Lea32 { .. } => {
                emit_pure_op(
                    ctx,
                    op,
                    &shadow,
                    &mut reg_vers,
                    &pending_writes,
                    &ret_slot_addr,
                    common_base,
                    permissions,
                    slow,
                );
            },
            LeafOp::MovMemReg32 { addr, src } => {
                let key = make_mem_key(addr, &reg_vers, 4);
                emit_addr(ctx, &shadow, addr);
                let addr_local = ctx.builder.set_new_local();
                emit_common_guard(ctx, &addr_local, 4, true, permissions, slow);
                emit_non_overlap_check(ctx, &addr_local, 4, &ret_slot_addr, 4, slow);
                get_reg32_shadow(ctx, &shadow, src);
                let val_local = ctx.builder.set_new_local();
                pending_writes.push(PendingWrite {
                    addr: addr_local,
                    val: val_local,
                    size: 4,
                    key,
                });
            },
            LeafOp::MovMemImm32 { addr, imm } => {
                let key = make_mem_key(addr, &reg_vers, 4);
                emit_addr(ctx, &shadow, addr);
                let addr_local = ctx.builder.set_new_local();
                emit_common_guard(ctx, &addr_local, 4, true, permissions, slow);
                emit_non_overlap_check(ctx, &addr_local, 4, &ret_slot_addr, 4, slow);
                ctx.builder.const_i32(imm);
                let val_local = ctx.builder.set_new_local();
                pending_writes.push(PendingWrite {
                    addr: addr_local,
                    val: val_local,
                    size: 4,
                    key,
                });
            },
            LeafOp::MovMem8Reg8 { addr, src8 } => {
                let key = make_mem_key(addr, &reg_vers, 1);
                emit_addr(ctx, &shadow, addr);
                let addr_local = ctx.builder.set_new_local();
                emit_common_guard(ctx, &addr_local, 1, true, permissions, slow);
                emit_non_overlap_check(ctx, &addr_local, 1, &ret_slot_addr, 4, slow);
                get_reg8_shadow(ctx, &shadow, src8);
                let val_local = ctx.builder.set_new_local();
                pending_writes.push(PendingWrite {
                    addr: addr_local,
                    val: val_local,
                    size: 1,
                    key,
                });
            },
            LeafOp::IncReg32 { dst } | LeafOp::DecReg32 { dst } => {
                let is_inc = matches!(op, LeafOp::IncReg32 { .. });
                let dst_local = shadow[dst as usize].as_ref().unwrap();
                ctx.builder.get_local(dst_local);
                let op1 = ctx.builder.set_new_local();
                ctx.builder.get_local(&op1);
                ctx.builder.const_i32(1);
                if is_inc {
                    ctx.builder.add_i32();
                }
                else {
                    ctx.builder.sub_i32();
                }
                ctx.builder.tee_local(dst_local);
                let res = ctx.builder.set_new_local();
                reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);

                let old_flag = std::mem::replace(&mut flag_state, ShadowFlagState::None);
                free_flag_state(ctx, old_flag);
                if let Some(old) = last_op1_preserved.take() {
                    ctx.builder.free_local(old);
                }
                ctx.builder.get_local(&op1);
                last_op1_preserved = Some(ctx.builder.set_new_local());
                flag_state = if is_inc {
                    ShadowFlagState::Inc32 { op1, res }
                }
                else {
                    ShadowFlagState::Dec32 { op1, res }
                };
                // NOTE: INC/DEC intentionally preserve `cf_state`!
            },
            LeafOp::AluRegReg32 { op: alu, dst, src } => {
                get_reg32_shadow(ctx, &shadow, dst);
                let lhs = ctx.builder.set_new_local();
                get_reg32_shadow(ctx, &shadow, src);
                let rhs = ctx.builder.set_new_local();
                let res = emit_alu_compute(
                    ctx,
                    alu,
                    &lhs,
                    &rhs,
                    &mut flag_state,
                    &mut cf_state,
                    &mut last_op1_preserved,
                );
                if alu.writes_dest() {
                    ctx.builder.get_local(&res);
                    ctx.builder
                        .set_local(shadow[dst as usize].as_ref().unwrap());
                    reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
                }
                ctx.builder.free_local(res);
                ctx.builder.free_local(rhs);
                ctx.builder.free_local(lhs);
            },
            LeafOp::AluRegImm32 { op: alu, dst, imm } => {
                get_reg32_shadow(ctx, &shadow, dst);
                let lhs = ctx.builder.set_new_local();
                ctx.builder.const_i32(imm);
                let rhs = ctx.builder.set_new_local();
                let res = emit_alu_compute(
                    ctx,
                    alu,
                    &lhs,
                    &rhs,
                    &mut flag_state,
                    &mut cf_state,
                    &mut last_op1_preserved,
                );
                if alu.writes_dest() {
                    ctx.builder.get_local(&res);
                    ctx.builder
                        .set_local(shadow[dst as usize].as_ref().unwrap());
                    reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
                }
                ctx.builder.free_local(res);
                ctx.builder.free_local(rhs);
                ctx.builder.free_local(lhs);
            },
            LeafOp::AluRegMem32 { op: alu, dst, addr } => {
                get_reg32_shadow(ctx, &shadow, dst);
                let lhs = ctx.builder.set_new_local();
                emit_guarded_read(
                    ctx,
                    &shadow,
                    &reg_vers,
                    &pending_writes,
                    &ret_slot_addr,
                    addr,
                    4,
                    common_base,
                    permissions,
                    slow,
                );
                let rhs = ctx.builder.set_new_local();
                let res = emit_alu_compute(
                    ctx,
                    alu,
                    &lhs,
                    &rhs,
                    &mut flag_state,
                    &mut cf_state,
                    &mut last_op1_preserved,
                );
                if alu.writes_dest() {
                    ctx.builder.get_local(&res);
                    ctx.builder
                        .set_local(shadow[dst as usize].as_ref().unwrap());
                    reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);
                }
                ctx.builder.free_local(res);
                ctx.builder.free_local(rhs);
                ctx.builder.free_local(lhs);
            },
            LeafOp::AluMemReg32 { op: alu, addr, src } => {
                let key = make_mem_key(addr, &reg_vers, 4);
                let writes = alu.writes_dest();
                emit_addr(ctx, &shadow, addr);
                let addr_local = ctx.builder.set_new_local();
                emit_common_guard(ctx, &addr_local, 4, writes, permissions, slow);
                emit_non_overlap_check(ctx, &addr_local, 4, &ret_slot_addr, 4, slow);
                // Read old value from pending writes or guest memory.
                let mut forwarded: Option<&WasmLocal> = None;
                let mut dynamic_checks: Vec<usize> = Vec::new();
                for (idx, pw) in pending_writes.iter().enumerate().rev() {
                    match classify_alias(key, pw.key) {
                        StaticAlias::Exact => {
                            forwarded = Some(&pw.val);
                            break;
                        },
                        StaticAlias::Disjoint => {},
                        StaticAlias::Partial => {
                            ctx.builder.br(slow);
                            break;
                        },
                        StaticAlias::Unknown => dynamic_checks.push(idx),
                    }
                }
                for idx in dynamic_checks {
                    let pw = &pending_writes[idx];
                    emit_non_overlap_check(ctx, &addr_local, 4, &pw.addr, pw.size, slow);
                }
                if let Some(val_local) = forwarded {
                    ctx.builder.get_local(val_local);
                }
                else {
                    ctx.builder.get_local(&addr_local);
                    ctx.builder.const_i32(common_base as i32);
                    ctx.builder.add_i32();
                    ctx.builder.load_unaligned_i32(0);
                }
                let lhs = ctx.builder.set_new_local();
                get_reg32_shadow(ctx, &shadow, src);
                let rhs = ctx.builder.set_new_local();
                let res = emit_alu_compute(
                    ctx,
                    alu,
                    &lhs,
                    &rhs,
                    &mut flag_state,
                    &mut cf_state,
                    &mut last_op1_preserved,
                );
                ctx.builder.free_local(rhs);
                ctx.builder.free_local(lhs);
                if writes {
                    pending_writes.push(PendingWrite {
                        addr: addr_local,
                        val: res,
                        size: 4,
                        key,
                    });
                }
                else {
                    ctx.builder.free_local(res);
                    ctx.builder.free_local(addr_local);
                }
            },
            LeafOp::AluMemImm32 { op: alu, addr, imm } => {
                let key = make_mem_key(addr, &reg_vers, 4);
                let writes = alu.writes_dest();
                emit_addr(ctx, &shadow, addr);
                let addr_local = ctx.builder.set_new_local();
                emit_common_guard(ctx, &addr_local, 4, writes, permissions, slow);
                emit_non_overlap_check(ctx, &addr_local, 4, &ret_slot_addr, 4, slow);
                let mut forwarded: Option<&WasmLocal> = None;
                let mut dynamic_checks: Vec<usize> = Vec::new();
                for (idx, pw) in pending_writes.iter().enumerate().rev() {
                    match classify_alias(key, pw.key) {
                        StaticAlias::Exact => {
                            forwarded = Some(&pw.val);
                            break;
                        },
                        StaticAlias::Disjoint => {},
                        StaticAlias::Partial => {
                            ctx.builder.br(slow);
                            break;
                        },
                        StaticAlias::Unknown => dynamic_checks.push(idx),
                    }
                }
                for idx in dynamic_checks {
                    let pw = &pending_writes[idx];
                    emit_non_overlap_check(ctx, &addr_local, 4, &pw.addr, pw.size, slow);
                }
                if let Some(val_local) = forwarded {
                    ctx.builder.get_local(val_local);
                }
                else {
                    ctx.builder.get_local(&addr_local);
                    ctx.builder.const_i32(common_base as i32);
                    ctx.builder.add_i32();
                    ctx.builder.load_unaligned_i32(0);
                }
                let lhs = ctx.builder.set_new_local();
                ctx.builder.const_i32(imm);
                let rhs = ctx.builder.set_new_local();
                let res = emit_alu_compute(
                    ctx,
                    alu,
                    &lhs,
                    &rhs,
                    &mut flag_state,
                    &mut cf_state,
                    &mut last_op1_preserved,
                );
                ctx.builder.free_local(rhs);
                ctx.builder.free_local(lhs);
                if writes {
                    pending_writes.push(PendingWrite {
                        addr: addr_local,
                        val: res,
                        size: 4,
                        key,
                    });
                }
                else {
                    ctx.builder.free_local(res);
                    ctx.builder.free_local(addr_local);
                }
            },
            LeafOp::SbbSelf32 { dst } => {
                // dst = 0 - cf (0 if CF==0, -1 if CF==1)
                match &cf_state {
                    CfState::Const(b) => ctx.builder.const_i32(i32::from(*b)),
                    CfState::Local(l) => ctx.builder.get_local(l),
                    CfState::Caller => unreachable!(),
                }
                let cf = ctx.builder.set_new_local();
                ctx.builder.const_i32(0);
                ctx.builder.get_local(&cf);
                ctx.builder.sub_i32();
                ctx.builder
                    .set_local(shadow[dst as usize].as_ref().unwrap());
                reg_vers[dst as usize] = reg_vers[dst as usize].wrapping_add(1);

                let old_flag = std::mem::replace(&mut flag_state, ShadowFlagState::None);
                free_flag_state(ctx, old_flag);
                let old_cf = std::mem::replace(&mut cf_state, CfState::Caller);
                free_cf_state(ctx, old_cf);

                ctx.builder.get_local(&cf);
                cf_state = CfState::Local(ctx.builder.set_new_local());
                flag_state = ShadowFlagState::SbbSelf32 { cf };
            },
            LeafOp::ColdJcc { cond } => {
                emit_eval_jcc(ctx, &flag_state, cond);
                ctx.builder.br_if(slow);
            },
            LeafOp::SkipDiamond { cond, skipped_len } => {
                let len = skipped_len as usize;
                if len > 0 {
                    emit_eval_jcc(ctx, &flag_state, cond);
                    ctx.builder.eqz_i32(); // Enter body when branch is NOT taken!
                    ctx.builder.if_void();
                    let mut inner_vers = reg_vers;
                    for &inner_op in &ops[i + 1..i + 1 + len] {
                        emit_pure_op(
                            ctx,
                            inner_op,
                            &shadow,
                            &mut inner_vers,
                            &pending_writes,
                            &ret_slot_addr,
                            common_base,
                            permissions,
                            slow,
                        );
                    }
                    if let Some(cnt) = &actual_insns {
                        ctx.builder.get_local(cnt);
                        ctx.builder.const_i32(len as i32);
                        ctx.builder.add_i32();
                        ctx.builder.set_local(cnt);
                    }
                    ctx.builder.block_end();
                    // Invalidate SSA version for any register that could be modified inside diamond.
                    for r in 0..8 {
                        if inner_vers[r] != reg_vers[r] {
                            reg_vers[r] = inner_vers[r].wrapping_add(1);
                        }
                    }
                }
                i += len;
            },
        }
        i += 1;
    }

    // === COMMIT PHASE: All guards passed! No branches to `slow` beyond this point. ===

    // 1. Commit return address word [ESP - 4] = ret_virt_addr (architectural red-zone bytes).
    ctx.builder.get_local(&ret_slot_addr);
    ctx.builder.const_i32(common_base as i32);
    ctx.builder.add_i32();
    ctx.builder.get_local(&ret_virt_addr);
    ctx.builder.store_unaligned_i32(0);

    // 2. Commit buffered leaf writes in exact program order.
    for pw in pending_writes.drain(..) {
        ctx.builder.get_local(&pw.addr);
        ctx.builder.const_i32(common_base as i32);
        ctx.builder.add_i32();
        ctx.builder.get_local(&pw.val);
        match pw.size {
            1 => ctx.builder.store_u8(0),
            4 => ctx.builder.store_unaligned_i32(0),
            _ => unreachable!(),
        }
        ctx.builder.free_local(pw.val);
        ctx.builder.free_local(pw.addr);
    }

    // 3. Commit exact architectural EFLAGS / lazy flags BEFORE clobbering register_locals
    //    (and reset previous_instruction so gen_getcf reads authoritative memory lazy flags).
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    match &flag_state {
        ShadowFlagState::None => {},
        ShadowFlagState::Add32 { op1, res } => {
            codegen::gen_set_last_op1(ctx.builder, op1);
            codegen::gen_set_last_result(ctx.builder, res);
            codegen::gen_set_last_op_size_and_flags_changed(ctx.builder, OPSIZE_32, FLAGS_ALL);
        },
        ShadowFlagState::SubCmp32 { op1, res, .. } => {
            codegen::gen_set_last_op1(ctx.builder, op1);
            codegen::gen_set_last_result(ctx.builder, res);
            codegen::gen_set_last_op_size_and_flags_changed(
                ctx.builder,
                OPSIZE_32,
                FLAGS_ALL | FLAG_SUB,
            );
        },
        ShadowFlagState::Inc32 { op1, res } | ShadowFlagState::Dec32 { op1, res } => {
            // Preserve CF (from cf_state, or materialize caller's live CF before clobbering last_*).
            ctx.builder.const_i32(global_pointers::flags as i32);
            codegen::gen_get_flags(ctx.builder);
            ctx.builder.const_i32(!FLAG_CARRY);
            ctx.builder.and_i32();
            match &cf_state {
                CfState::Caller => {
                    codegen::gen_getcf(ctx, ConditionNegate::False);
                    ctx.builder.or_i32();
                },
                CfState::Const(false) => {},
                CfState::Const(true) => {
                    ctx.builder.const_i32(FLAG_CARRY);
                    ctx.builder.or_i32();
                },
                CfState::Local(cf) => {
                    ctx.builder.get_local(cf);
                    ctx.builder.or_i32();
                },
            }
            ctx.builder.store_aligned_i32(0);

            codegen::gen_set_last_op1(ctx.builder, op1);
            codegen::gen_set_last_result(ctx.builder, res);
            let fc = if matches!(&flag_state, ShadowFlagState::Inc32 { .. }) {
                FLAGS_ALL & !1
            }
            else {
                FLAGS_ALL & !1 | FLAG_SUB
            };
            codegen::gen_set_last_op_size_and_flags_changed(ctx.builder, OPSIZE_32, fc);
        },
        ShadowFlagState::Bitwise32 { res } => {
            if let Some(op1) = &last_op1_preserved {
                codegen::gen_set_last_op1(ctx.builder, op1);
            }
            codegen::gen_set_last_result(ctx.builder, res);
            codegen::gen_set_last_op_size_and_flags_changed(
                ctx.builder,
                OPSIZE_32,
                FLAGS_ALL & !FLAG_CARRY & !FLAG_OVERFLOW & !FLAG_ADJUST,
            );
            codegen::gen_clear_flags_bits(ctx.builder, FLAG_CARRY | FLAG_OVERFLOW | FLAG_ADJUST);
        },
        ShadowFlagState::SbbSelf32 { cf } => {
            // Exact flags for `sbb r, r`:
            // CF = cf, PF = 1, AF = cf, ZF = !cf, SF = cf, OF = 0
            ctx.builder.const_i32(global_pointers::flags as i32);
            codegen::gen_get_flags(ctx.builder);
            ctx.builder.const_i32(!FLAGS_ALL);
            ctx.builder.and_i32();
            ctx.builder.const_i32(FLAG_PARITY);
            ctx.builder.or_i32();
            ctx.builder.get_local(cf);
            // If cf == 1 -> CF | AF | SF (0x91); if cf == 0 -> ZF (0x40)
            ctx.builder.if_i32();
            ctx.builder.const_i32(FLAG_CARRY | FLAG_ADJUST | 128);
            ctx.builder.else_();
            ctx.builder.const_i32(64);
            ctx.builder.block_end();
            ctx.builder.or_i32();
            ctx.builder.store_aligned_i32(0);

            ctx.builder.const_i32(global_pointers::flags_changed as i32);
            ctx.builder.const_i32(0);
            ctx.builder.store_aligned_i32(0);
        },
    }

    // 4. Commit modified shadow GPRs back to live register_locals.
    for r in 0..8usize {
        if let Some(l) = shadow[r].take() {
            ctx.builder.get_local(&l);
            ctx.builder.set_local(&ctx.register_locals[r]);
            ctx.builder.free_local(l);
        }
    }

    // 5. Commit exact executed leaf instruction count.
    ctx.builder.get_local(&ctx.instruction_counter);
    if let Some(cnt) = &actual_insns {
        ctx.builder.get_local(cnt);
    }
    else {
        ctx.builder.const_i32(ic.plan.max_insns as i32);
    }
    ctx.builder.add_i32();
    ctx.builder.set_local(&ctx.instruction_counter);

    ctx.builder.br(done);
    ctx.builder.block_end(); // end `slow`

    // Free all temporary locals from the fast path before emitting the slow path
    // so slow path codegen can reuse them safely.
    if let Some(old) = last_op1_preserved.take() {
        ctx.builder.free_local(old);
    }
    free_flag_state(ctx, flag_state);
    free_cf_state(ctx, cf_state);
    if let Some(cnt) = actual_insns {
        ctx.builder.free_local(cnt);
    }
    ctx.builder.free_local(ret_slot_addr);
    ctx.builder.free_local(ret_virt_addr);

    // === SLOW PATH ===
    emit_slow_call(ctx, block, ic, state_flags);

    ctx.builder.block_end(); // end `done`

    ctx.cpu.eip = block.end_addr;
    ctx.cpu.prefixes = 0;
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    ctx.mmx_task_switch_call_count = None;
}
