// Combine the RAM guards of a straight-line run that uses one affine base.
// No speculative guest access or architectural mutation precedes these guards.
use crate::cpu::cpu;
use crate::cpu_context::CpuContext;
use crate::jit::{BasicBlock, BasicBlockType, Instruction, JitContext};
use crate::wasmgen::wasm_builder::{Label, WasmLocal};
use crate::{codegen, jit_instructions, modrm};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Address {
    coefficients: [i32; 8],
    offset: i32,
}
impl Address {
    fn constant(offset: i32) -> Self {
        Self {
            coefficients: [0; 8],
            offset,
        }
    }
    fn register(r: usize) -> Self {
        let mut a = Self::constant(0);
        a.coefficients[r] = 1;
        a
    }
    fn add_scaled(mut self, other: Self, scale: i32) -> Self {
        self.offset = self.offset.wrapping_add(other.offset.wrapping_mul(scale));
        for (a, b) in self.coefficients.iter_mut().zip(other.coefficients) {
            *a = a.wrapping_add(b.wrapping_mul(scale));
        }
        self
    }
}
fn address(c: &mut CpuContext, byte: u8, registers: &[Option<Address>; 8]) -> Option<Address> {
    let (base, index, shift, offset) = modrm::decode(c, byte).flat_parts()?;
    let mut a = Address::constant(offset);
    if let Some(r) = base {
        a = a.add_scaled(registers[r as usize]?, 1);
    }
    if let Some(r) = index {
        a = a.add_scaled(registers[r as usize]?, 1 << shift);
    }
    Some(a)
}
#[derive(Clone, Copy)]
enum Op {
    Load(u32, Address),
    Store(u32, Address),
    StoreImm(i32, Address),
    Integer,
}
impl Op {
    fn memory(self) -> Option<(Address, bool)> {
        match self {
            Self::Load(_, a) => Some((a, false)),
            Self::Store(_, a) | Self::StoreImm(_, a) => Some((a, true)),
            Self::Integer => None,
        }
    }
}
fn decode(c: &mut CpuContext, registers: &mut [Option<Address>; 8]) -> Option<Op> {
    if c.eip & 0xfff >= 0xff0 {
        return None;
    }
    let opcode = c.read_imm8();
    match opcode {
        0x89 | 0x8b => {
            let byte = c.read_imm8();
            let r = (byte >> 3 & 7) as u32;
            if byte < 0xc0 {
                let a = address(c, byte, registers)?;
                if opcode == 0x8b {
                    registers[r as usize] = None;
                    return Some(Op::Load(r, a));
                }
                return Some(Op::Store(r, a));
            }
            let rm = (byte & 7) as usize;
            if opcode == 0x89 {
                registers[rm] = registers[r as usize];
            }
            else {
                registers[r as usize] = registers[rm];
            }
        },
        0xc7 => {
            let byte = c.read_imm8();
            if byte >= 0xc0 || byte >> 3 & 7 != 0 {
                return None;
            }
            let a = address(c, byte, registers)?;
            return Some(Op::StoreImm(c.read_imm32() as i32, a));
        },
        0x40..=0x4f => {
            let r = (opcode & 7) as usize;
            registers[r] = registers[r]
                .map(|a| a.add_scaled(Address::constant(if opcode < 0x48 { 1 } else { -1 }), 1));
        },
        0xb8..=0xbf => {
            registers[(opcode & 7) as usize] = Some(Address::constant(c.read_imm32() as i32))
        },
        0x8d => {
            let byte = c.read_imm8();
            if byte >= 0xc0 {
                return None;
            }
            registers[(byte >> 3 & 7) as usize] = Some(address(c, byte, registers)?);
        },
        0x01 | 0x03 | 0x29 | 0x2b | 0x39 | 0x3b => {
            let byte = c.read_imm8();
            if byte < 0xc0 {
                return None;
            }
            let (rm, r) = ((byte & 7) as usize, (byte >> 3 & 7) as usize);
            let (dest, source) = if opcode & 2 == 0 { (rm, r) } else { (r, rm) };
            if !matches!(opcode, 0x39 | 0x3b) {
                registers[dest] = registers[dest].zip(registers[source]).map(|(a, b)| {
                    a.add_scaled(b, if matches!(opcode, 0x01 | 0x03) { 1 } else { -1 })
                });
            }
        },
        0x81 | 0x83 => {
            let byte = c.read_imm8();
            let group = byte >> 3 & 7;
            if byte < 0xc0 || !matches!(group, 0 | 5 | 7) {
                return None;
            }
            let imm = if opcode == 0x83 { c.read_imm8s() as i32 } else { c.read_imm32() as i32 };
            let dest = (byte & 7) as usize;
            if group != 7 {
                registers[dest] = registers[dest]
                    .map(|a| a.add_scaled(Address::constant(imm), if group == 0 { 1 } else { -1 }));
            }
        },
        0x05 | 0x2d | 0x3d => {
            let imm = c.read_imm32() as i32;
            if opcode != 0x3d {
                registers[0] = registers[0].map(|a| {
                    a.add_scaled(Address::constant(imm), if opcode == 0x05 { 1 } else { -1 })
                });
            }
        },
        0x84 | 0x85 => {
            if c.read_imm8() < 0xc0 {
                return None;
            }
        },
        _ => return None,
    }
    Some(Op::Integer)
}
struct Run {
    ops: Vec<(u32, u32, Op)>,
    base: Address,
    bytes: u32,
    write: bool,
    end: u32,
}
fn scan(cpu: &CpuContext, block: &BasicBlock) -> Option<Run> {
    if !cpu.state_flags.is_32() || !cpu.has_flat_segmentation() {
        return None;
    }
    let limit = if matches!(block.ty, BasicBlockType::Normal { .. }) {
        block.end_addr
    }
    else {
        block.last_instruction_addr
    };
    let mut c = cpu.clone();
    c.prefixes = 0;
    let mut registers = std::array::from_fn(|r| Some(Address::register(r)));
    let mut run: Option<Run> = None;
    let mut high = 0i64;
    let mut accesses = 0;
    for _ in 0..16 {
        if c.eip >= limit {
            break;
        }
        let mut next = c.clone();
        let mut regs = registers;
        let Some(op) = decode(&mut next, &mut regs)
        else {
            break;
        };
        if next.eip > limit || next.eip >> 12 != cpu.eip >> 12 {
            break;
        }
        if let Some((a, write)) = op.memory() {
            if let Some(p) = &mut run {
                if p.base.coefficients != a.coefficients {
                    break;
                }
                let low = p.base.offset.min(a.offset);
                let next_high = high.max(a.offset as i64 + 4);
                if next_high - low as i64 > 4096 {
                    break;
                }
                p.base.offset = low;
                high = next_high;
                p.write |= write;
                p.bytes = (high - low as i64) as u32;
            }
            else {
                high = a.offset as i64 + 4;
                run = Some(Run {
                    ops: Vec::new(),
                    base: a,
                    bytes: 4,
                    write,
                    end: next.eip,
                });
            }
            accesses += 1;
        }
        else if run.is_none() {
            return None;
        }
        let p = run.as_mut().unwrap();
        p.ops.push((c.eip, next.eip, op));
        p.end = next.eip;
        c = next;
        registers = regs;
    }
    if accesses < 2 {
        return None;
    }
    run
}
fn guarded_base(ctx: &mut JitContext, p: &Run, fail: Label) -> WasmLocal {
    ctx.builder.const_i32(p.base.offset);
    for (r, coefficient) in p.base.coefficients.into_iter().enumerate() {
        if coefficient == 0 {
            continue;
        }
        ctx.builder.get_local(&ctx.register_locals[r]);
        if coefficient != 1 {
            ctx.builder.const_i32(coefficient);
            ctx.builder.mul_i32();
        }
        ctx.builder.add_i32();
    }
    let address = ctx.builder.set_new_local();
    ctx.builder.get_local(&address);
    ctx.builder.const_i32(0xfff);
    ctx.builder.and_i32();
    ctx.builder.const_i32((4096 - p.bytes) as i32);
    ctx.builder.gtu_i32();
    ctx.builder.br_if(fail);
    if let Some((base, permissions)) =
        unsafe { crate::cpu::vine::common_memory() }.filter(|_| ctx.cpu.cpl3())
    {
        ctx.builder.get_local(&address);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        ctx.builder.load_u8(permissions);
        let mask = if p.write { 3 } else { 1 };
        ctx.builder.const_i32(mask);
        ctx.builder.and_i32();
        ctx.builder.const_i32(mask);
        ctx.builder.ne_i32();
        ctx.builder.br_if(fail);
        ctx.builder.get_local(&address);
        ctx.builder.const_i32(base as i32);
        ctx.builder.add_i32();
        ctx.builder.set_local(&address);
        return address;
    }
    ctx.builder.get_local(&address);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();
    ctx.builder
        .load_aligned_i32(unsafe { &cpu::tlb_data[0] as *const i32 as u32 });
    let entry = ctx.builder.tee_new_local();
    let mask = 0xfff
        & !cpu::TLB_GLOBAL
        & !(if ctx.cpu.cpl3() { 0 } else { cpu::TLB_NO_USER })
        & !(if p.write { 0 } else { cpu::TLB_READONLY | cpu::TLB_HAS_CODE });
    ctx.builder.const_i32(mask);
    ctx.builder.and_i32();
    ctx.builder.const_i32(cpu::TLB_VALID);
    ctx.builder.ne_i32();
    ctx.builder.br_if(fail);
    ctx.builder.get_local(&entry);
    ctx.builder.const_i32(!0xfff);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address);
    ctx.builder.xor_i32();
    ctx.builder.set_local(&address);
    ctx.builder.free_local(entry);
    address
}
pub fn generate(ctx: &mut JitContext, block: &BasicBlock) -> bool {
    if cfg!(feature = "profiler") {
        return false;
    }
    let Some(p) = scan(ctx.cpu, block)
    else {
        return false;
    };
    if !ctx.builder.can_allocate_temporary_locals(8) {
        return false;
    }
    let saved_cpu = ctx.cpu.clone();
    let done = ctx.builder.block_void();
    let fail = ctx.builder.block_void();
    let base = guarded_base(ctx, &p, fail);
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    for &(pc, end, op) in &p.ops {
        ctx.start_of_current_instruction = pc;
        ctx.cpu.eip = pc;
        let calls = ctx.builder.imported_call_count();
        match op {
            Op::Load(r, a) => {
                ctx.builder.get_local(&base);
                ctx.builder
                    .load_unaligned_i32((a.offset as i64 - p.base.offset as i64) as u32);
                codegen::gen_set_reg32(ctx, r);
            },
            Op::Store(r, a) => {
                ctx.builder.get_local(&base);
                codegen::gen_get_reg32(ctx, r);
                ctx.builder
                    .store_unaligned_i32((a.offset as i64 - p.base.offset as i64) as u32);
            },
            Op::StoreImm(value, a) => {
                ctx.builder.get_local(&base);
                ctx.builder.const_i32(value);
                ctx.builder
                    .store_unaligned_i32((a.offset as i64 - p.base.offset as i64) as u32);
            },
            Op::Integer => {
                let mut flags = 0;
                jit_instructions::jit_instruction(ctx, &mut flags);
                assert!(flags == 0 && ctx.cpu.eip == end);
            },
        }
        assert!(ctx.builder.imported_call_count() == calls);
        ctx.cpu.eip = end;
        ctx.previous_instruction =
            std::mem::replace(&mut ctx.current_instruction, Instruction::Other);
    }
    ctx.builder.free_local(base);
    ctx.builder.br(done);
    ctx.builder.block_end();
    *ctx.cpu = saved_cpu;
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    while ctx.cpu.eip < p.end {
        ctx.start_of_current_instruction = ctx.cpu.eip;
        let mut flags = 0;
        jit_instructions::jit_instruction(ctx, &mut flags);
        assert!(flags == 0 && ctx.cpu.eip <= p.end);
        ctx.previous_instruction =
            std::mem::replace(&mut ctx.current_instruction, Instruction::Other);
    }
    ctx.builder.block_end();
    ctx.cpu.prefixes = 0;
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    ctx.mmx_task_switch_call_count = None;
    true
}
