//! Opt-in, same-BB x87 value runs. No architectural stack state is hidden
//! across an instruction that can fault, call the host, or leave this run.
//! Pure RAM/TLB and stack guards precede all mutation. Ordered RAM accesses
//! retain aliases; the slow arm is the original per-instruction compiler.
use crate::cpu::{cpu, fpu, global_pointers as gp};
use crate::cpu_context::CpuContext;
use crate::jit::{BasicBlock, BasicBlockType, Instruction, JitContext};
use crate::softfloat::F80;
use crate::wasmgen::wasm_builder::{Label, WasmLocal, WasmLocalF64, WasmLocalI64};
use crate::{codegen, fast_x87, jit_instructions, modrm};

extern "C" {
    static mut softfloat_exceptionFlags: u8;
}

static mut COMPLETED_INSTRUCTIONS: u32 = 0;
// A private value ABI, not the architectural SSE/FPU scratch register. The
// caller consumes HIGH immediately after the non-reentrant numeric helper.
static mut HIGH: u32 = 0;
const SHADOW: u32 = 0x1_0000;
// Experiment limits, tunable through jit config 11 and 12.
fn max_ops() -> usize { unsafe { crate::jit::JIT_X87_MAX_OPS as usize } }
fn max_addresses() -> usize { unsafe { crate::jit::JIT_X87_MAX_ADDRESSES as usize } }
#[no_mangle]
pub unsafe fn vine_x87_fast_instructions() -> u32 { COMPLETED_INSTRUCTIONS }

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
#[derive(Clone, Copy)]
enum Math {
    Add,
    Mul,
    Sub,
    ReverseSub,
}
#[derive(Clone, Copy)]
enum FloatWidth {
    F32,
    F64,
}
impl FloatWidth {
    fn bytes(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}
#[derive(Clone, Copy)]
enum Op {
    Load(usize, FloatWidth),
    Store(usize, FloatWidth, bool),
    IntegerStore(usize, bool, bool),
    Copy(u8),
    Exchange(u8),
    Negate,
    Register(Math, u8, u8, bool),
    Memory(Math, usize, FloatWidth),
    StoreRegister(u8, bool),
    CompareRegister(u8, u8, bool),
    CompareMemory(usize, FloatWidth, bool),
    Status,
    Integer(u32),
    IntegerMemory(usize, bool),
}
struct Plan {
    ops: Vec<Op>,
    addresses: Vec<(Address, bool, u32)>,
    end: u32,
    required_full: u8,
    required_empty: u8,
    load: u8,
    written: u8,
    final_tags: [Option<bool>; 8],
    top: u8,
    pushes: bool,
}

fn math(group: u8) -> Option<Math> {
    match group {
        0 => Some(Math::Add),
        1 => Some(Math::Mul),
        4 => Some(Math::Sub),
        5 => Some(Math::ReverseSub),
        _ => None,
    }
}
fn effective_address(
    c: &mut CpuContext,
    byte: u8,
    registers: &[Option<Address>; 8],
) -> Option<Address> {
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
fn memory_operand(
    addresses: &mut Vec<(Address, bool, u32)>,
    address: Address,
    write: bool,
    bytes: u32,
    limit: usize,
) -> Option<usize> {
    if let Some(i) = addresses.iter().position(|(a, _, _)| *a == address) {
        addresses[i].1 |= write;
        addresses[i].2 = addresses[i].2.max(bytes);
        Some(i)
    }
    else {
        if addresses.len() == limit {
            return None;
        }
        addresses.push((address, write, bytes));
        Some(addresses.len() - 1)
    }
}

// These register-only instructions cannot fault or observe hidden x87 state.
// Resolve their effects on future EAs as wrapping affine expressions in the
// entry GPRs, so every RAM guard still precedes all architectural mutation.
// Actual instructions, including lazy integer flags, use the existing emitter.
fn decode_integer(
    c: &mut CpuContext,
    opcode: u8,
    registers: &mut [Option<Address>; 8],
) -> Option<()> {
    match opcode {
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
            registers[(byte >> 3 & 7) as usize] = Some(effective_address(c, byte, registers)?);
        },
        0x89 | 0x8b | 0x01 | 0x03 | 0x29 | 0x2b | 0x39 | 0x3b => {
            let byte = c.read_imm8();
            if byte < 0xc0 {
                return None;
            }
            let (rm, r) = ((byte & 7) as usize, (byte >> 3 & 7) as usize);
            let (dest, source) = if opcode & 2 == 0 { (rm, r) } else { (r, rm) };
            match opcode {
                0x89 | 0x8b => registers[dest] = registers[source],
                0x01 | 0x03 | 0x29 | 0x2b => {
                    registers[dest] = registers[dest].zip(registers[source]).map(|(a, b)| {
                        a.add_scaled(b, if matches!(opcode, 0x01 | 0x03) { 1 } else { -1 })
                    })
                },
                _ => (), // CMP changes only flags.
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
        0xf6 => {
            let byte = c.read_imm8();
            if byte < 0xc0 || byte >> 3 & 7 != 0 {
                return None;
            }
            c.read_imm8(); // TEST r8,imm8 changes only integer flags.
        },
        _ => return None,
    }
    Some(())
}
fn decode(
    c: &mut CpuContext,
    addresses: &mut Vec<(Address, bool, u32)>,
    registers: &mut [Option<Address>; 8],
    address_limit: usize,
) -> Option<Op> {
    // At most opcode/ModRM/SIB/disp32. No prefix is accepted.
    if c.eip & 0xfff >= 0xff0 {
        return None;
    }
    let pc = c.eip;
    let opcode = c.read_imm8();
    if opcode == 0xff {
        let byte = c.read_imm8();
        let group = byte >> 3 & 7;
        if byte >= 0xc0 || group > 1 {
            return None;
        }
        let address = effective_address(c, byte, registers)?;
        let index = memory_operand(addresses, address, true, 4, address_limit)?;
        return Some(Op::IntegerMemory(index, group == 1));
    }
    if !matches!(opcode, 0xd8..=0xdf) {
        decode_integer(c, opcode, registers)?;
        return Some(Op::Integer(pc));
    }
    let byte = c.read_imm8();
    let group = byte >> 3 & 7;
    let r = byte & 7;
    if byte >= 0xc0 {
        if opcode == 0xdf && byte == 0xe0 {
            // FNSTSW changes AX. Future EAs using its value cannot be guarded
            // from entry GPRs; a subsequent full register MOV can restore one.
            registers[0] = None;
            return Some(Op::Status);
        }
        return match opcode {
            0xd8 | 0xdc if matches!(group, 2 | 3) => {
                Some(Op::CompareRegister(r, u8::from(group == 3), false))
            },
            0xde if byte == 0xd9 => Some(Op::CompareRegister(1, 2, false)),
            0xda if byte == 0xe9 => Some(Op::CompareRegister(1, 2, true)),
            0xd8 => Some(Op::Register(math(group)?, 0, r, false)),
            0xdc | 0xde => Some(Op::Register(math(group)?, r, r, opcode == 0xde)),
            0xd9 => match byte {
                0xc0..=0xc7 => Some(Op::Copy(r)),
                0xc8..=0xcf => Some(Op::Exchange(r)),
                0xe0 => Some(Op::Negate),
                _ => None,
            },
            0xdd => match group {
                2 | 3 => Some(Op::StoreRegister(r, group == 3)),
                4 | 5 => Some(Op::CompareRegister(r, u8::from(group == 5), true)),
                _ => None,
            },
            _ => None,
        };
    }
    let arithmetic = matches!(opcode, 0xd8 | 0xdc);
    let compare = arithmetic && matches!(group, 2 | 3);
    let integer_store = opcode == 0xdb && matches!(group, 1 | 2 | 3);
    if !arithmetic
        && !integer_store
        && !(matches!(opcode, 0xd9 | 0xdd) && matches!(group, 0 | 2 | 3))
    {
        return None;
    }
    if arithmetic && !compare {
        math(group)?;
    }
    let address = effective_address(c, byte, registers)?;
    let width = if matches!(opcode, 0xdc | 0xdd) { FloatWidth::F64 } else { FloatWidth::F32 };
    let write = !arithmetic && group != 0;
    let i = memory_operand(addresses, address, write, width.bytes(), address_limit)?;
    if compare {
        Some(Op::CompareMemory(i, width, group == 3))
    }
    else if arithmetic {
        Some(Op::Memory(math(group)?, i, width))
    }
    else if integer_store {
        Some(Op::IntegerStore(i, group != 2, group == 1))
    }
    else if group == 0 {
        Some(Op::Load(i, width))
    }
    else {
        Some(Op::Store(i, width, group == 3))
    }
}

fn require(p: &mut Plan, slot: u8, empty: bool) -> Option<()> {
    match p.final_tags[slot as usize] {
        Some(known) if known != empty => return None,
        Some(_) => (),
        None => {
            let bit = 1 << slot;
            if empty {
                if p.required_full & bit != 0 {
                    return None;
                }
                p.required_empty |= bit;
            }
            else {
                if p.required_empty & bit != 0 {
                    return None;
                }
                p.required_full |= bit;
            }
        },
    }
    if !empty && p.written & (1 << slot) == 0 {
        p.load |= 1 << slot;
    }
    Some(())
}
fn plan(
    cpu: &CpuContext,
    block: &BasicBlock,
    op_limit: usize,
    address_limit: usize,
) -> Option<Plan> {
    if !cpu.has_flat_segmentation() || !cpu.state_flags.is_32() || block.has_sti {
        return None;
    }
    let mut p = Plan {
        ops: Vec::new(),
        addresses: Vec::new(),
        end: cpu.eip,
        required_full: 0,
        required_empty: 0,
        load: 0,
        written: 0,
        final_tags: [None; 8],
        top: 0,
        pushes: false,
    };
    let mut c = cpu.clone();
    c.prefixes = 0;
    let mut registers = std::array::from_fn(|r| Some(Address::register(r)));
    // Normal fallthrough has no last-instruction EIP epilogue. Include its
    // final arithmetic/store, while exits keep the existing precise epilogue.
    // Interior entries keep their original BB and form independent suffixes.
    let limit = if matches!(block.ty, BasicBlockType::Normal { .. }) {
        block.end_addr
    }
    else {
        block.last_instruction_addr
    };
    while p.ops.len() < op_limit && c.eip < limit {
        let mut next = c.clone();
        let mut addresses = p.addresses.clone();
        let Some(op) = decode(&mut next, &mut addresses, &mut registers, address_limit)
        else {
            break;
        };
        if p.ops.is_empty() && matches!(op, Op::Integer(..) | Op::IntegerMemory(..)) {
            return None;
        }
        if next.eip > limit || next.eip >> 12 != cpu.eip >> 12 {
            break;
        }
        let t = p.top;
        match op {
            Op::Integer(_) | Op::IntegerMemory(..) | Op::Status => (),
            Op::Load(..) | Op::Copy(_) => {
                if let Op::Copy(r) = op {
                    require(&mut p, (t + r) & 7, false)?;
                }
                let dest = t.wrapping_sub(1) & 7;
                require(&mut p, dest, true)?;
                p.final_tags[dest as usize] = Some(false);
                p.written |= 1 << dest;
                p.top = dest;
                p.pushes = true;
            },
            Op::Store(_, _, pop) | Op::IntegerStore(_, pop, _) => {
                require(&mut p, t, false)?;
                if pop {
                    p.final_tags[t as usize] = Some(true);
                    p.top = (t + 1) & 7;
                }
            },
            Op::StoreRegister(r, pop) => {
                require(&mut p, t, false)?;
                let dest = (t + r) & 7;
                p.written |= 1 << dest;
                p.final_tags[dest as usize] = Some(false);
                if pop {
                    p.final_tags[t as usize] = Some(true);
                    p.top = (t + 1) & 7;
                }
            },
            Op::CompareRegister(r, pops, _) => {
                require(&mut p, t, false)?;
                require(&mut p, (t + r) & 7, false)?;
                for _ in 0..pops {
                    p.final_tags[p.top as usize] = Some(true);
                    p.top = (p.top + 1) & 7;
                }
            },
            Op::CompareMemory(_, _, pop) => {
                require(&mut p, t, false)?;
                if pop {
                    p.final_tags[t as usize] = Some(true);
                    p.top = (t + 1) & 7;
                }
            },
            Op::Exchange(r) => {
                require(&mut p, t, false)?;
                require(&mut p, (t + r) & 7, false)?;
                p.written |= 1 << t | 1 << ((t + r) & 7);
            },
            Op::Negate | Op::Memory(..) => {
                require(&mut p, t, false)?;
                p.written |= 1 << t;
            },
            Op::Register(_, dest, r, pop) => {
                require(&mut p, t, false)?;
                require(&mut p, (t + r) & 7, false)?;
                p.written |= 1 << ((t + dest) & 7);
                if pop {
                    p.final_tags[t as usize] = Some(true);
                    p.top = (t + 1) & 7;
                }
            },
        }
        p.addresses = addresses;
        p.ops.push(op);
        c = next;
        p.end = c.eip;
    }
    // Load/copy/sign/store runs also save repeated raw80 stack traffic. Their
    // value representation and guards are identical to arithmetic runs.
    let count = p
        .ops
        .iter()
        .filter(|op| !matches!(op, Op::Integer(..) | Op::IntegerMemory(..)))
        .count();
    // A separate entry at the eventual store can split FLD/arithmetic/FSTP.
    // Keep the existing block and budget boundary, but retain the loaded
    // value through arithmetic even when only these two x87 operations fit.
    let load_math_pair = count == 2
        && matches!(p.ops.first(), Some(Op::Load(..)))
        && p.ops
            .iter()
            .any(|op| matches!(op, Op::Memory(..) | Op::Register(..)));
    if count < 3 && !load_math_pair {
        return None;
    }
    Some(p)
}

fn virtual_address(ctx: &mut JitContext, a: Address) {
    ctx.builder.const_i32(a.offset);
    for (r, coefficient) in a.coefficients.into_iter().enumerate() {
        if coefficient == 0 {
            continue;
        }
        ctx.builder.get_local(&ctx.register_locals[r]);
        if coefficient != 1 {
            if (coefficient as u32).is_power_of_two() {
                ctx.builder.const_i32(coefficient.trailing_zeros() as i32);
                ctx.builder.shl_i32();
            }
            else {
                ctx.builder.const_i32(coefficient);
                ctx.builder.mul_i32();
            }
        }
        ctx.builder.add_i32();
    }
}
// Returns the wasm-memory operand local plus the extra memarg offset: under
// the memarg fold the local holds the guest address and the arena base rides
// in the memarg.
fn physical_address(
    ctx: &mut JitContext,
    a: Address,
    write: bool,
    bytes: u32,
    fail: Label,
) -> (WasmLocal, u32) {
    virtual_address(ctx, a);
    let address = ctx.builder.set_new_local();
    ctx.builder.get_local(&address);
    ctx.builder.const_i32(0xfff);
    ctx.builder.and_i32();
    ctx.builder.const_i32((4096 - bytes) as i32);
    ctx.builder.gtu_i32();
    ctx.builder.br_if(fail);
    if let Some((base, permissions)) = unsafe { crate::cpu::vine::common_memory() }
        .filter(|_| ctx.cpu.cpl3() && !cfg!(feature = "profiler"))
    {
        // Ordinary common-memory JIT accesses do not fill the v86 TLB. Retained
        // x87 runs must validate the same Runtime table or they stay cold forever.
        // Require read + write for stores as integer RMWs can share this address.
        let mask = if write { 3 } else { 1 };
        ctx.builder.get_local(&address);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        ctx.builder.load_u8(permissions);
        ctx.builder.const_i32(mask);
        ctx.builder.and_i32();
        ctx.builder.const_i32(mask);
        ctx.builder.ne_i32();
        ctx.builder.br_if(fail);
        if codegen::common_memarg_base(ctx).is_some() {
            return (address, base);
        }
        ctx.builder.get_local(&address);
        ctx.builder.const_i32(base as i32);
        ctx.builder.add_i32();
        ctx.builder.set_local(&address);
        return (address, 0);
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
        & !(if write { 0 } else { cpu::TLB_READONLY | cpu::TLB_HAS_CODE });
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
    (address, 0)
}
// Addresses with identical register coefficients whose constant offsets lie
// within 64 bytes of the first ungrouped member's offset share one page
// guard: their union extent must lie on one page and the permission test is
// the union of the members' needs (write if any member writes). Member
// addresses are the shared base plus a constant; anchoring at i's offset
// keeps every index inside its own group.
fn emit_address_guards(ctx: &mut JitContext, p: &Plan, fail: Label) -> Vec<(WasmLocal, u32)> {
    let mut result: Vec<Option<(WasmLocal, u32)>> = (0..p.addresses.len()).map(|_| None).collect();
    let mut used = vec![false; p.addresses.len()];
    for i in 0..p.addresses.len() {
        if used[i] {
            continue;
        }
        let coefficients = p.addresses[i].0.coefficients;
        let min_offset = p.addresses[i].0.offset;
        let group: Vec<usize> = (0..p.addresses.len())
            .filter(|&j| {
                !used[j]
                    && p.addresses[j].0.coefficients == coefficients
                    && (p.addresses[j].0.offset as i64 - min_offset as i64) <= 64
                    && p.addresses[j].0.offset as i64 >= min_offset as i64
            })
            .collect();
        for &j in &group {
            used[j] = true;
        }
        if group.len() == 1 {
            let (a, w, bytes) = p.addresses[group[0]];
            result[group[0]] = Some(physical_address(ctx, a, w, bytes, fail));
            continue;
        }
        let end = group
            .iter()
            .map(|&j| p.addresses[j].0.offset + p.addresses[j].2 as i32)
            .max()
            .unwrap();
        let write = group.iter().any(|&j| p.addresses[j].1);
        let (base, extra) = physical_address(
            ctx,
            p.addresses[i].0,
            write,
            (end - min_offset) as u32,
            fail,
        );
        let mut base_member = None;
        for &j in &group {
            let delta = p.addresses[j].0.offset - min_offset;
            if delta == 0 {
                base_member = Some(j);
            }
            else {
                ctx.builder.get_local(&base);
                ctx.builder.const_i32(delta);
                ctx.builder.add_i32();
                result[j] = Some((ctx.builder.set_new_local(), extra));
            }
        }
        result[base_member.unwrap()] = Some((base, extra));
    }
    result.into_iter().map(Option::unwrap).collect()
}

fn slot_index(ctx: &mut JitContext, top: &WasmLocal, relative: u8) {
    ctx.builder.get_local(top);
    ctx.builder.const_i32(relative as i32);
    ctx.builder.add_i32();
    ctx.builder.const_i32(7);
    ctx.builder.and_i32();
}
fn slot_address(ctx: &mut JitContext, top: &WasmLocal, relative: u8) -> WasmLocal {
    slot_index(ctx, top, relative);
    ctx.builder.const_i32(4);
    ctx.builder.shl_i32();
    ctx.builder.const_i32(gp::fpu_st as i32);
    ctx.builder.add_i32();
    ctx.builder.set_new_local()
}
struct Value {
    low: WasmLocalI64,
    high: WasmLocal,
}
impl Value {
    fn zero(ctx: &mut JitContext) -> Self {
        ctx.builder.const_i64(0);
        let low = ctx.builder.set_new_local_i64();
        ctx.builder.const_i32(0);
        let high = ctx.builder.set_new_local();
        Self { low, high }
    }
    fn copy_from(&self, ctx: &mut JitContext, other: &Self) {
        ctx.builder.get_local_i64(&other.low);
        ctx.builder.set_local_i64(&self.low);
        ctx.builder.get_local(&other.high);
        ctx.builder.set_local(&self.high);
    }
    fn free(self, ctx: &mut JitContext) {
        ctx.builder.free_local_i64(self.low);
        ctx.builder.free_local(self.high);
    }
}

// Values materialized by a preceding fast run are still stored as raw80 at
// every architectural boundary. Recover their exact binary64 representation
// once on entry. Do not round arbitrary extended values, alter exception
// state, or reinterpret noncanonical/exceptional encodings here.
fn recover_native(ctx: &mut JitContext, value: &Value) {
    let done = ctx.builder.block_void();
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(0x7fff);
    ctx.builder.and_i32();
    let exponent = ctx.builder.set_new_local();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.get_local(&exponent);
    ctx.builder.eqz_i32();
    ctx.builder.and_i32();
    ctx.builder.if_void();
    ctx.builder.get_local(&value.high);
    ctx.builder.extend_unsigned_i32_to_i64();
    ctx.builder.const_i64(48);
    ctx.builder.shl_i64();
    ctx.builder.set_local_i64(&value.low);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.set_local(&value.high);
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.builder.get_local(&exponent);
    ctx.builder.const_i32(0x3c01);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(0x7fe);
    ctx.builder.geu_i32();
    ctx.builder.br_if(done);
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(i64::MIN | 2047);
    ctx.builder.and_i64();
    ctx.builder.const_i64(i64::MIN);
    ctx.builder.ne_i64();
    ctx.builder.br_if(done);
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(11);
    ctx.builder.shr_u_i64();
    ctx.builder.const_i64((1i64 << 52) - 1);
    ctx.builder.and_i64();
    ctx.builder.get_local(&exponent);
    ctx.builder.const_i32(0x3c00);
    ctx.builder.sub_i32();
    ctx.builder.extend_unsigned_i32_to_i64();
    ctx.builder.const_i64(52);
    ctx.builder.shl_i64();
    ctx.builder.or_i64();
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(0x8000);
    ctx.builder.and_i32();
    ctx.builder.extend_unsigned_i32_to_i64();
    ctx.builder.const_i64(48);
    ctx.builder.shl_i64();
    ctx.builder.or_i64();
    ctx.builder.set_local_i64(&value.low);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.set_local(&value.high);
    ctx.builder.block_end();
    ctx.builder.free_local(exponent);
}

// Publish only the ten meaningful bytes. Shadow values are canonical normal
// binary64 or signed zero; no numerical conversion or flags change occurs.
fn store_value(ctx: &mut JitContext, address: &WasmLocal, extra: u32, value: &Value) {
    ctx.builder.get_local(address);
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.eq_i32();
    ctx.builder.if_i64();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.if_i64();
    ctx.builder.const_i64(0);
    ctx.builder.else_();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(11);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(i64::MIN);
    ctx.builder.or_i64();
    ctx.builder.block_end();
    ctx.builder.else_();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.block_end();
    ctx.builder.store_aligned_i64(extra);
    ctx.builder.get_local(address);
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.eq_i32();
    ctx.builder.if_i32();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(48);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x8000);
    ctx.builder.and_i32();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.if_i32();
    ctx.builder.const_i32(0);
    ctx.builder.else_();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(52);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x7ff);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x3c00);
    ctx.builder.add_i32();
    ctx.builder.block_end();
    ctx.builder.or_i32();
    ctx.builder.else_();
    ctx.builder.get_local(&value.high);
    ctx.builder.block_end();
    ctx.builder.store_aligned_u16(8 + extra);
}

fn load_f32(ctx: &mut JitContext, address: &WasmLocal, extra: u32, out: &Value) {
    ctx.builder.get_local(address);
    ctx.builder.load_unaligned_i32(extra);
    let bits = ctx.builder.set_new_local();
    ctx.builder.get_local(&bits);
    ctx.builder.const_i32(0x7fff_ffff);
    ctx.builder.and_i32();
    let magnitude = ctx.builder.tee_new_local();
    ctx.builder.const_i32(0x0080_0000);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(0x7f00_0000);
    ctx.builder.ltu_i32();
    ctx.builder.get_local(&magnitude);
    ctx.builder.eqz_i32();
    ctx.builder.or_i32();
    ctx.builder.if_void();
    // Guarded widening is exact, including the sign of zero. A conversion
    // clears raw flags even when its result has no exception to merge into SW.
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
    ctx.builder.get_local(&bits);
    ctx.builder.reinterpret_i32_as_f32();
    ctx.builder.promote_f32_to_f64();
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.set_local_i64(&out.low);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.set_local(&out.high);
    ctx.builder.else_();
    // This path stays inside the value ABI and cannot fault or access the
    // guest. It preserves NaN payloads, subnormals, and conversion exceptions.
    ctx.builder.get_local(&bits);
    ctx.builder.extend_unsigned_i32_to_i64();
    ctx.builder.const_i32(0);
    ctx.builder.call_fn2_i64_i32_ret_i64("fpu_value_from_f32");
    ctx.builder.set_local_i64(&out.low);
    ctx.builder.load_fixed_i32(std::ptr::addr_of!(HIGH) as u32);
    ctx.builder.set_local(&out.high);
    ctx.builder.block_end();
    ctx.builder.free_local(magnitude);
    ctx.builder.free_local(bits);
}
fn load_f64(ctx: &mut JitContext, address: &WasmLocal, extra: u32, out: &Value) {
    ctx.builder.get_local(address);
    ctx.builder.load_unaligned_i64(extra);
    let bits = ctx.builder.set_new_local_i64();
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(52);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x7ff);
    ctx.builder.and_i32();
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(2046);
    ctx.builder.ltu_i32();
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.or_i32();
    ctx.builder.if_void();
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
    ctx.builder.get_local_i64(&bits);
    ctx.builder.set_local_i64(&out.low);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.set_local(&out.high);
    ctx.builder.else_();
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i32(0);
    ctx.builder.call_fn2_i64_i32_ret_i64("fpu_value_from_f64");
    ctx.builder.set_local_i64(&out.low);
    ctx.builder.load_fixed_i32(std::ptr::addr_of!(HIGH) as u32);
    ctx.builder.set_local(&out.high);
    ctx.builder.block_end();
    ctx.builder.free_local_i64(bits);
}
fn load_float(
    ctx: &mut JitContext,
    address: &WasmLocal,
    extra: u32,
    out: &Value,
    width: FloatWidth,
) {
    match width {
        FloatWidth::F32 => load_f32(ctx, address, extra, out),
        FloatWidth::F64 => load_f64(ctx, address, extra, out),
    }
}
fn binary(ctx: &mut JitContext, kind: Math, a: &Value, b: &Value, out: &Value) {
    let (a, b) = if matches!(kind, Math::ReverseSub) { (b, a) } else { (a, b) };
    let done = ctx.builder.block_void();
    let fallback = ctx.builder.block_void();
    ctx.builder.get_local(&a.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.ne_i32();
    ctx.builder.br_if(fallback);
    ctx.builder.get_local(&b.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.ne_i32();
    ctx.builder.br_if(fallback);
    ctx.builder.get_local_i64(&a.low);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.get_local_i64(&b.low);
    ctx.builder.reinterpret_i64_as_f64();
    match kind {
        Math::Add => ctx.builder.add_f64(),
        Math::Mul => ctx.builder.mul_f64(),
        Math::Sub | Math::ReverseSub => ctx.builder.sub_f64(),
    }
    ctx.builder.reinterpret_f64_as_i64();
    // Operands can alias the destination. Keep them intact until the native
    // result is accepted, including underflow/overflow fallback.
    let result = ctx.builder.set_new_local_i64();
    ctx.builder.get_local_i64(&result);
    ctx.builder.const_i64(52);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x7ff);
    ctx.builder.and_i32();
    let exponent = ctx.builder.tee_new_local();
    ctx.builder.const_i32(0x7ff);
    ctx.builder.eq_i32();
    ctx.builder.br_if(fallback);
    ctx.builder.get_local(&exponent);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    ctx.builder.get_local_i64(&result);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.ne_i64();
    ctx.builder.br_if(fallback);
    if matches!(kind, Math::Mul) {
        ctx.builder.get_local_i64(&a.low);
        ctx.builder.const_i64(1);
        ctx.builder.shl_i64();
        ctx.builder.const_i64(0);
        ctx.builder.ne_i64();
        ctx.builder.get_local_i64(&b.low);
        ctx.builder.const_i64(1);
        ctx.builder.shl_i64();
        ctx.builder.const_i64(0);
        ctx.builder.ne_i64();
        ctx.builder.and_i32();
        ctx.builder.br_if(fallback);
    }
    ctx.builder.block_end();
    if matches!(kind, Math::Add) {
        ctx.builder
            .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
        ctx.builder.const_i32(0);
        ctx.builder.store_u8(0);
    }
    ctx.builder.get_local_i64(&result);
    ctx.builder.set_local_i64(&out.low);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.set_local(&out.high);
    ctx.builder.free_local(exponent);
    ctx.builder.free_local_i64(result);
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.builder.get_local_i64(&a.low);
    ctx.builder.get_local(&a.high);
    ctx.builder.get_local_i64(&b.low);
    ctx.builder.get_local(&b.high);
    ctx.builder.call_fn4_i64_i32_i64_i32_ret_i64(match kind {
        Math::Add => "fpu_value_add",
        Math::Mul => "fpu_value_mul",
        Math::Sub | Math::ReverseSub => "fpu_value_sub",
    });
    ctx.builder.set_local_i64(&out.low);
    ctx.builder.load_fixed_i32(std::ptr::addr_of!(HIGH) as u32);
    ctx.builder.set_local(&out.high);
    ctx.builder.block_end();
}
fn store_f32(ctx: &mut JitContext, address: &WasmLocal, extra: u32, value: &Value) {
    ctx.builder.const_i32(0);
    let result = ctx.builder.set_new_local();
    let done = ctx.builder.block_void();
    let fallback = ctx.builder.block_void();
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.ne_i32();
    ctx.builder.br_if(fallback);
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.demote_f64_to_f32();
    ctx.builder.reinterpret_f32_as_i32();
    ctx.builder.set_local(&result);
    ctx.builder.get_local(&result);
    ctx.builder.const_i32(0x7fff_ffff);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x0080_0000);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(0x7f00_0000);
    ctx.builder.ltu_i32();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
    ctx.builder.br_if(fallback);
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.get_local(&value.high);
    ctx.builder.call_fn2_i64_i32_ret("fpu_value_to_f32");
    ctx.builder.set_local(&result);
    ctx.builder.block_end();
    ctx.builder.get_local(address);
    ctx.builder.get_local(&result);
    ctx.builder.store_unaligned_i32(extra);
    ctx.builder.free_local(result);
}
fn store_f64(ctx: &mut JitContext, address: &WasmLocal, extra: u32, value: &Value) {
    ctx.builder.get_local(address);
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.eq_i32();
    ctx.builder.if_i64();
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.else_();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.get_local(&value.high);
    ctx.builder.call_fn2_i64_i32_ret_i64("fpu_value_to_f64");
    ctx.builder.block_end();
    ctx.builder.store_unaligned_i64(extra);
}
fn store_i32(ctx: &mut JitContext, address: &WasmLocal, extra: u32, value: &Value, truncate: bool) {
    ctx.builder.const_i32(0);
    let result = ctx.builder.set_new_local();
    let done = ctx.builder.block_void();
    let fallback = ctx.builder.block_void();
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.ne_i32();
    ctx.builder.br_if(fallback);
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.reinterpret_i64_as_f64();
    if truncate {
        ctx.builder.trunc_f64();
    }
    else {
        ctx.builder.nearest_f64();
    }
    ctx.builder.reinterpret_f64_as_i64();
    let rounded = ctx.builder.set_new_local_i64();
    // Check the rounded value before Wasm's trapping conversion. Invalid
    // input retains the original indefinite integer and exception handling.
    ctx.builder.get_local_i64(&rounded);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.const_i64((-2147483648.0f64).to_bits() as i64);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.ge_f64();
    ctx.builder.get_local_i64(&rounded);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.const_i64(2147483648.0f64.to_bits() as i64);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.lt_f64();
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.br_if(fallback);
    ctx.builder.get_local_i64(&rounded);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.trunc_signed_f64_to_i32();
    ctx.builder.set_local(&result);
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
    ctx.builder.free_local_i64(rounded);
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.get_local(&value.high);
    ctx.builder.call_fn2_i64_i32_ret(if truncate {
        "fpu_value_truncate_i32"
    }
    else {
        "fpu_value_to_i32"
    });
    ctx.builder.set_local(&result);
    ctx.builder.block_end();
    ctx.builder.get_local(address);
    ctx.builder.get_local(&result);
    ctx.builder.store_unaligned_i32(extra);
    ctx.builder.free_local(result);
}
fn negate(ctx: &mut JitContext, value: &Value) {
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.eq_i32();
    ctx.builder.if_void();
    ctx.builder.get_local_i64(&value.low);
    ctx.builder.const_i64(i64::MIN);
    ctx.builder.xor_i64();
    ctx.builder.set_local_i64(&value.low);
    ctx.builder.else_();
    ctx.builder.get_local(&value.high);
    ctx.builder.const_i32(0x8000);
    ctx.builder.xor_i32();
    ctx.builder.set_local(&value.high);
    ctx.builder.block_end();
}

fn compare(ctx: &mut JitContext, a: &Value, b: &Value, quiet: bool) {
    ctx.builder.get_local(&a.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.eq_i32();
    ctx.builder.get_local(&b.high);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.if_void();
    // Shadows contain finite normal binary64 or signed zero. Their comparison
    // is exact; raw80, NaN, infinities and subnormals retain SoftFloat behavior.
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
    ctx.builder.const_i32(gp::fpu_status_word as i32);
    ctx.builder.load_fixed_u16(gp::fpu_status_word as u32);
    ctx.builder.const_i32(!0x4700);
    ctx.builder.and_i32();
    ctx.builder.get_local_i64(&a.low);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.get_local_i64(&b.low);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.lt_f64();
    ctx.builder.const_i32(8);
    ctx.builder.shl_i32();
    ctx.builder.or_i32();
    ctx.builder.get_local_i64(&a.low);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.get_local_i64(&b.low);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.eq_f64();
    ctx.builder.const_i32(14);
    ctx.builder.shl_i32();
    ctx.builder.or_i32();
    ctx.builder.store_aligned_u16(0);
    ctx.builder.else_();
    ctx.builder.get_local_i64(&a.low);
    ctx.builder.get_local(&a.high);
    ctx.builder.get_local_i64(&b.low);
    ctx.builder.get_local(&b.high);
    ctx.builder.call_fn4_i64_i32_i64_i32(if quiet {
        "fpu_value_compare_quiet"
    }
    else {
        "fpu_value_compare"
    });
    ctx.builder.block_end();
}

// Plan-derived facts shared by the dynamic and typed emissions. They mirror
// the codegen-time flags the op loop used to track: a scratch write, an
// exchange temporary write, a SoftFloat flag clear in the fast arm, or a
// point where the typed twin can bail to the dynamic sequence.
fn op_writes_scratch(op: &Op) -> bool {
    match *op {
        Op::Load(..)
        | Op::Copy(_)
        | Op::Store(..)
        | Op::IntegerStore(..)
        | Op::CompareMemory(..)
        | Op::Memory(..)
        | Op::Register(..) => true,
        Op::CompareRegister(_, _, quiet) => !quiet,
        _ => false,
    }
}
fn op_clears_flags(op: &Op) -> bool {
    match *op {
        Op::Load(..)
        | Op::Memory(..)
        | Op::CompareMemory(..)
        | Op::CompareRegister(..)
        | Op::Store(..)
        | Op::IntegerStore(..) => true,
        Op::Register(kind, ..) => matches!(kind, Math::Add),
        _ => false,
    }
}
fn op_can_bail(op: &Op) -> bool {
    match *op {
        Op::Load(..) | Op::Memory(..) | Op::CompareMemory(..) | Op::IntegerStore(..) => true,
        Op::Store(_, FloatWidth::F32, _) | Op::Register(..) => true,
        _ => false,
    }
}

fn emit_flag_clear(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(std::ptr::addr_of!(softfloat_exceptionFlags) as i32);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(0);
}

struct DynValues<'a> {
    slots: &'a [Option<Value>],
    scratch: &'a Value,
    temporary: &'a Value,
    addresses: &'a [(WasmLocal, u32)],
    top: &'a WasmLocal,
    tags: &'a WasmLocal,
}
impl<'a> DynValues<'a> {
    fn slot(&self, i: usize) -> &'a Value { self.slots[i].as_ref().unwrap() }
}

// Op::Status, Op::Integer and Op::IntegerMemory are identical in both twins:
// they have no value representation of their own.
fn emit_nonvalue_op(ctx: &mut JitContext, dv: &DynValues, op: &Op) -> bool {
    match *op {
        Op::IntegerMemory(a, decrement) => {
            // The complete read/write access was guarded before mutation.
            // Execute in order so overlapping integer and float EAs see
            // each other's writes, using the original lazy-flag emitter.
            ctx.previous_instruction = Instruction::Other;
            ctx.current_instruction = Instruction::Other;
            ctx.builder.get_local(&dv.addresses[a].0);
            ctx.builder.load_unaligned_i32(dv.addresses[a].1);
            let value = ctx.builder.set_new_local();
            if decrement {
                jit_instructions::gen_dec32(ctx, &value);
            }
            else {
                jit_instructions::gen_inc32(ctx, &value);
            }
            ctx.builder.get_local(&dv.addresses[a].0);
            ctx.builder.get_local(&value);
            ctx.builder.store_unaligned_i32(dv.addresses[a].1);
            ctx.builder.free_local(value);
            ctx.current_instruction = Instruction::Other;
        },
        Op::Integer(pc) => {
            ctx.cpu.eip = pc;
            ctx.cpu.prefixes = 0;
            ctx.start_of_current_instruction = pc;
            ctx.previous_instruction = Instruction::Other;
            ctx.current_instruction = Instruction::Other;
            let mut flags = 0;
            jit_instructions::jit_instruction(ctx, &mut flags);
            dbg_assert!(flags == 0);
        },
        _ => return false,
    }
    true
}

fn emit_status(ctx: &mut JitContext, top: &WasmLocal, t: usize, pushed: bool) {
    ctx.builder.load_fixed_u16(gp::fpu_status_word as u32);
    ctx.builder
        .const_i32(!(0x3800 | if pushed { 0x200 } else { 0 }));
    ctx.builder.and_i32();
    slot_index(ctx, top, t as u8);
    ctx.builder.const_i32(11);
    ctx.builder.shl_i32();
    ctx.builder.or_i32();
    codegen::gen_set_reg16(ctx, 0);
}

fn emit_dynamic_op(
    ctx: &mut JitContext,
    dv: &DynValues,
    op: &Op,
    t: &mut usize,
    pushed: &mut bool,
) {
    if matches!(*op, Op::Status) {
        emit_status(ctx, dv.top, *t, *pushed);
        return;
    }
    if emit_nonvalue_op(ctx, dv, op) {
        return;
    }
    match *op {
        Op::IntegerMemory(..) | Op::Integer(_) | Op::Status => unreachable!(),
        Op::Load(a, width) => {
            load_float(
                ctx,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                dv.scratch,
                width,
            );
            *t = t.wrapping_sub(1) & 7;
            dv.slot(*t).copy_from(ctx, dv.scratch);
            *pushed = true;
        },
        Op::Copy(r) => {
            dv.scratch.copy_from(ctx, dv.slot((*t + r as usize) & 7));
            *t = t.wrapping_sub(1) & 7;
            dv.slot(*t).copy_from(ctx, dv.scratch);
            *pushed = true;
        },
        Op::Store(a, width, pop) => {
            dv.scratch.copy_from(ctx, dv.slot(*t));
            match width {
                FloatWidth::F32 => {
                    store_f32(ctx, &dv.addresses[a].0, dv.addresses[a].1, dv.scratch)
                },
                FloatWidth::F64 => {
                    store_f64(ctx, &dv.addresses[a].0, dv.addresses[a].1, dv.scratch)
                },
            }
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::IntegerStore(a, pop, truncate) => {
            dv.scratch.copy_from(ctx, dv.slot(*t));
            store_i32(
                ctx,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                dv.scratch,
                truncate,
            );
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::StoreRegister(r, pop) => {
            dv.slot((*t + r as usize) & 7).copy_from(ctx, dv.slot(*t));
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::CompareMemory(a, width, pop) => {
            load_float(
                ctx,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                dv.scratch,
                width,
            );
            compare(ctx, dv.slot(*t), dv.scratch, false);
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::CompareRegister(r, pops, quiet) => {
            let other = dv.slot((*t + r as usize) & 7);
            // The ordered JIT uses get_sti's architectural SSE scratch;
            // quiet comparisons call a helper without changing scratch.
            if !quiet {
                dv.scratch.copy_from(ctx, other);
            }
            compare(ctx, dv.slot(*t), other, quiet);
            *t = (*t + pops as usize) & 7;
        },
        Op::Exchange(r) => {
            let other = (*t + r as usize) & 7;
            dv.temporary.copy_from(ctx, dv.slot(*t));
            dv.slot(*t).copy_from(ctx, dv.slot(other));
            dv.slot(other).copy_from(ctx, dv.temporary);
        },
        Op::Negate => negate(ctx, dv.slot(*t)),
        Op::Memory(kind, a, width) => {
            load_float(
                ctx,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                dv.scratch,
                width,
            );
            let st = dv.slot(*t);
            binary(ctx, kind, st, dv.scratch, st);
        },
        Op::Register(kind, dest, r, pop) => {
            dv.scratch.copy_from(ctx, dv.slot((*t + r as usize) & 7));
            binary(
                ctx,
                kind,
                dv.slot(*t),
                dv.scratch,
                dv.slot((*t + dest as usize) & 7),
            );
            if pop {
                *t = (*t + 1) & 7;
            }
        },
    }
}

// Tags, TOP, the push status bit and the mode-3 instruction counter are the
// same for either representation.
fn emit_run_state(ctx: &mut JitContext, p: &Plan, top: &WasmLocal, tags: &WasmLocal) {
    // The final tags are compile-time constants relative to TOP, so one
    // rotated clear/set mask pair replaces a shift/and/or per slot. The masks
    // are byte-replicated so a 32-bit rotation stays inside the tag byte.
    let mut changed = 0u32;
    let mut set = 0u32;
    for i in 0..8 {
        if let Some(empty) = p.final_tags[i as usize] {
            changed |= 1 << i;
            if empty {
                set |= 1 << i;
            }
        }
    }
    if changed != 0 {
        ctx.builder.get_local(tags);
        ctx.builder.const_i32((changed * 0x0101_0101) as i32);
        ctx.builder.get_local(top);
        ctx.builder.const_i32(7);
        ctx.builder.and_i32();
        ctx.builder.rotl_i32();
        ctx.builder.const_i32(0xff);
        ctx.builder.and_i32();
        ctx.builder.const_i32(-1);
        ctx.builder.xor_i32();
        ctx.builder.and_i32();
        ctx.builder.const_i32((set * 0x0101_0101) as i32);
        ctx.builder.get_local(top);
        ctx.builder.const_i32(7);
        ctx.builder.and_i32();
        ctx.builder.rotl_i32();
        ctx.builder.const_i32(0xff);
        ctx.builder.and_i32();
        ctx.builder.or_i32();
        ctx.builder.set_local(tags);
    }
    ctx.builder.const_i32(gp::fpu_stack_empty as i32);
    ctx.builder.get_local(tags);
    ctx.builder.store_u8(0);
    ctx.builder.const_i32(gp::fpu_stack_ptr as i32);
    slot_index(ctx, top, p.top);
    ctx.builder.store_u8(0);
    if p.pushes {
        ctx.builder.const_i32(gp::fpu_status_word as i32);
        ctx.builder.load_fixed_u16(gp::fpu_status_word as u32);
        ctx.builder.const_i32(!0x200);
        ctx.builder.and_i32();
        ctx.builder.store_aligned_u16(0);
    }
    if unsafe { fast_x87::vine_get_x87_mode() == 3 } {
        let address = std::ptr::addr_of!(COMPLETED_INSTRUCTIONS) as u32;
        ctx.builder.const_i32(address as i32);
        ctx.builder.load_fixed_i32(address);
        ctx.builder.const_i32(
            p.ops
                .iter()
                .filter(|op| !matches!(op, Op::Integer(..) | Op::IntegerMemory(..)))
                .count() as i32,
        );
        ctx.builder.add_i32();
        let sum = ctx.builder.tee_new_local();
        ctx.builder.load_fixed_i32(address);
        ctx.builder.ltu_i32();
        ctx.builder.if_i32();
        ctx.builder.const_i32(-1);
        ctx.builder.else_();
        ctx.builder.get_local(&sum);
        ctx.builder.block_end();
        ctx.builder.store_aligned_i32(0);
        ctx.builder.free_local(sum);
    }
}

fn emit_dynamic_epilogue(ctx: &mut JitContext, p: &Plan, dv: &DynValues) {
    for i in 0..8 {
        if p.written & (1 << i) != 0 {
            let address = slot_address(ctx, dv.top, i);
            store_value(ctx, &address, 0, dv.slot(i as usize));
            ctx.builder.free_local(address);
        }
    }
    if p.ops.iter().any(op_writes_scratch) {
        ctx.builder.const_i32(gp::sse_scratch_register as i32);
        let address = ctx.builder.set_new_local();
        store_value(ctx, &address, 0, dv.scratch);
        ctx.builder.free_local(address);
    }
    emit_run_state(ctx, p, dv.top, dv.tags);
}

// The typed twin carries every participating slot plus the scratch and
// exchange values as f64 locals; all of them stay ordinary binary64 (finite
// normal or signed zero) or the twin bails to the dynamic sequence.
struct TypedValues<'a> {
    slots: &'a [Option<WasmLocalF64>],
    scratch: &'a WasmLocalF64,
    temporary: &'a WasmLocalF64,
    bail: Label,
}
impl<'a> TypedValues<'a> {
    fn slot(&self, i: usize) -> &'a WasmLocalF64 { self.slots[i].as_ref().unwrap() }
}

fn materialize_typed(ctx: &mut JitContext, dynamic: &Value, typed: &WasmLocalF64) {
    ctx.builder.get_local_f64(typed);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.set_local_i64(&dynamic.low);
    ctx.builder.const_i32(SHADOW as i32);
    ctx.builder.set_local(&dynamic.high);
}

// A bail runs before the op has mutated anything architectural: the typed
// f64 state is published back into the dynamic (low, SHADOW) locals and the
// dispatch below restarts the op with the original per-instruction semantics.
// All bail sites share this one block; resume is written unconditionally
// before each bail-capable op.
fn emit_typed_materialization(ctx: &mut JitContext, dv: &DynValues, tv: &TypedValues) {
    for i in 0..8 {
        if let (Some(dynamic), Some(typed)) = (dv.slots[i].as_ref(), tv.slots[i].as_ref()) {
            materialize_typed(ctx, dynamic, typed);
        }
    }
    materialize_typed(ctx, dv.scratch, tv.scratch);
    materialize_typed(ctx, dv.temporary, tv.temporary);
}

// Loads the memory operand into `out`. The hot path tests only the normal
// range; the rare block accepts a signed zero and bails otherwise, matching
// the dynamic fast arm's ordinary-value set exactly.
fn typed_load(
    ctx: &mut JitContext,
    tv: &TypedValues,
    address: &WasmLocal,
    extra: u32,
    width: FloatWidth,
    out: &WasmLocalF64,
) {
    let done = ctx.builder.block_void();
    match width {
        FloatWidth::F32 => {
            ctx.builder.get_local(address);
            ctx.builder.load_unaligned_i32(extra);
            let bits = ctx.builder.set_new_local();
            ctx.builder.get_local(&bits);
            ctx.builder.const_i32(0x7fff_ffff);
            ctx.builder.and_i32();
            let magnitude = ctx.builder.set_new_local();
            ctx.builder.get_local(&magnitude);
            ctx.builder.const_i32(0x0080_0000);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(0x7f00_0000);
            ctx.builder.ltu_i32();
            ctx.builder.if_void();
            ctx.builder.get_local(&bits);
            ctx.builder.reinterpret_i32_as_f32();
            ctx.builder.promote_f32_to_f64();
            ctx.builder.set_local_f64(out);
            ctx.builder.br(done);
            ctx.builder.block_end();
            ctx.builder.get_local(&magnitude);
            ctx.builder.eqz_i32();
            ctx.builder.if_void();
            ctx.builder.get_local(&bits);
            ctx.builder.reinterpret_i32_as_f32();
            ctx.builder.promote_f32_to_f64();
            ctx.builder.set_local_f64(out);
            ctx.builder.br(done);
            ctx.builder.block_end();
            ctx.builder.br(tv.bail);
            ctx.builder.block_end();
            ctx.builder.free_local(magnitude);
            ctx.builder.free_local(bits);
        },
        FloatWidth::F64 => {
            ctx.builder.get_local(address);
            ctx.builder.load_unaligned_i64(extra);
            let bits = ctx.builder.set_new_local_i64();
            ctx.builder.get_local_i64(&bits);
            ctx.builder.const_i64(52);
            ctx.builder.shr_u_i64();
            ctx.builder.wrap_i64_to_i32();
            ctx.builder.const_i32(0x7ff);
            ctx.builder.and_i32();
            let exponent = ctx.builder.set_new_local();
            ctx.builder.get_local(&exponent);
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
            ctx.builder.const_i32(2046);
            ctx.builder.ltu_i32();
            ctx.builder.if_void();
            ctx.builder.get_local_i64(&bits);
            ctx.builder.reinterpret_i64_as_f64();
            ctx.builder.set_local_f64(out);
            ctx.builder.br(done);
            ctx.builder.block_end();
            ctx.builder.get_local_i64(&bits);
            ctx.builder.const_i64(1);
            ctx.builder.shl_i64();
            ctx.builder.const_i64(0);
            ctx.builder.eq_i64();
            ctx.builder.if_void();
            ctx.builder.get_local_i64(&bits);
            ctx.builder.reinterpret_i64_as_f64();
            ctx.builder.set_local_f64(out);
            ctx.builder.br(done);
            ctx.builder.block_end();
            ctx.builder.br(tv.bail);
            ctx.builder.block_end();
            ctx.builder.free_local(exponent);
            ctx.builder.free_local_i64(bits);
        },
    }
}

// The dynamic fast arm's acceptance tests: the result may not overflow to an
// inf/NaN exponent, may not be a nonzero subnormal, and for multiplication may
// not underflow to zero from two nonzero operands. The hot path is a single
// range test; the rare block decides exactly and bails before `out` changes.
fn typed_binary(
    ctx: &mut JitContext,
    tv: &TypedValues,
    kind: Math,
    a: &WasmLocalF64,
    b: &WasmLocalF64,
    out: &WasmLocalF64,
) {
    let (a, b) = if matches!(kind, Math::ReverseSub) { (b, a) } else { (a, b) };
    ctx.builder.get_local_f64(a);
    ctx.builder.get_local_f64(b);
    match kind {
        Math::Add => ctx.builder.add_f64(),
        Math::Mul => ctx.builder.mul_f64(),
        Math::Sub | Math::ReverseSub => ctx.builder.sub_f64(),
    }
    let result = ctx.builder.set_new_local_f64();
    ctx.builder.get_local_f64(&result);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.const_i64(52);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x7ff);
    ctx.builder.and_i32();
    let exponent = ctx.builder.set_new_local();
    ctx.builder.get_local(&exponent);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(0x7fe);
    ctx.builder.geu_i32();
    ctx.builder.if_void();
    ctx.builder.get_local(&exponent);
    ctx.builder.const_i32(0x7ff);
    ctx.builder.eq_i32();
    ctx.builder.br_if(tv.bail);
    ctx.builder.get_local(&exponent);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    ctx.builder.get_local_f64(&result);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.ne_i64();
    ctx.builder.br_if(tv.bail);
    if matches!(kind, Math::Mul) {
        ctx.builder.get_local_f64(a);
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.const_i64(1);
        ctx.builder.shl_i64();
        ctx.builder.const_i64(0);
        ctx.builder.ne_i64();
        ctx.builder.get_local_f64(b);
        ctx.builder.reinterpret_f64_as_i64();
        ctx.builder.const_i64(1);
        ctx.builder.shl_i64();
        ctx.builder.const_i64(0);
        ctx.builder.ne_i64();
        ctx.builder.and_i32();
        ctx.builder.br_if(tv.bail);
    }
    ctx.builder.block_end();
    ctx.builder.block_end();
    ctx.builder.get_local_f64(&result);
    ctx.builder.set_local_f64(out);
    ctx.builder.free_local(exponent);
    ctx.builder.free_local_f64(result);
}

fn typed_store_f32(
    ctx: &mut JitContext,
    tv: &TypedValues,
    address: &WasmLocal,
    extra: u32,
    source: &WasmLocalF64,
) {
    ctx.builder.get_local_f64(source);
    ctx.builder.demote_f64_to_f32();
    ctx.builder.reinterpret_f32_as_i32();
    let result = ctx.builder.set_new_local();
    let done = ctx.builder.block_void();
    ctx.builder.get_local(&result);
    ctx.builder.const_i32(0x7fff_ffff);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x0080_0000);
    ctx.builder.sub_i32();
    ctx.builder.const_i32(0x7f00_0000);
    ctx.builder.ltu_i32();
    ctx.builder.if_void();
    ctx.builder.get_local(address);
    ctx.builder.get_local(&result);
    ctx.builder.store_unaligned_i32(extra);
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.builder.get_local_f64(source);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.if_void();
    ctx.builder.get_local(address);
    ctx.builder.get_local(&result);
    ctx.builder.store_unaligned_i32(extra);
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.builder.br(tv.bail);
    ctx.builder.block_end();
    ctx.builder.free_local(result);
}

fn typed_store_f64(ctx: &mut JitContext, address: &WasmLocal, extra: u32, source: &WasmLocalF64) {
    ctx.builder.get_local(address);
    ctx.builder.get_local_f64(source);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.store_unaligned_i64(extra);
}

fn typed_store_i32(
    ctx: &mut JitContext,
    tv: &TypedValues,
    address: &WasmLocal,
    extra: u32,
    source: &WasmLocalF64,
    truncate: bool,
) {
    ctx.builder.get_local_f64(source);
    if truncate {
        ctx.builder.trunc_f64();
    }
    else {
        ctx.builder.nearest_f64();
    }
    let rounded = ctx.builder.set_new_local_f64();
    ctx.builder.get_local_f64(&rounded);
    ctx.builder.const_i64((-2147483648.0f64).to_bits() as i64);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.ge_f64();
    ctx.builder.get_local_f64(&rounded);
    ctx.builder.const_i64(2147483648.0f64.to_bits() as i64);
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.lt_f64();
    ctx.builder.and_i32();
    ctx.builder.eqz_i32();
    ctx.builder.br_if(tv.bail);
    ctx.builder.get_local(address);
    ctx.builder.get_local_f64(&rounded);
    ctx.builder.trunc_signed_f64_to_i32();
    ctx.builder.store_unaligned_i32(extra);
    ctx.builder.free_local_f64(rounded);
}

fn typed_compare(ctx: &mut JitContext, a: &WasmLocalF64, b: &WasmLocalF64) {
    ctx.builder.const_i32(gp::fpu_status_word as i32);
    ctx.builder.load_fixed_u16(gp::fpu_status_word as u32);
    ctx.builder.const_i32(!0x4700);
    ctx.builder.and_i32();
    ctx.builder.get_local_f64(a);
    ctx.builder.get_local_f64(b);
    ctx.builder.lt_f64();
    ctx.builder.const_i32(8);
    ctx.builder.shl_i32();
    ctx.builder.or_i32();
    ctx.builder.get_local_f64(a);
    ctx.builder.get_local_f64(b);
    ctx.builder.eq_f64();
    ctx.builder.const_i32(14);
    ctx.builder.shl_i32();
    ctx.builder.or_i32();
    ctx.builder.store_aligned_u16(0);
}

fn typed_negate(ctx: &mut JitContext, value: &WasmLocalF64) {
    ctx.builder.get_local_f64(value);
    ctx.builder.reinterpret_f64_as_i64();
    ctx.builder.const_i64(i64::MIN);
    ctx.builder.xor_i64();
    ctx.builder.reinterpret_i64_as_f64();
    ctx.builder.set_local_f64(value);
}

fn emit_typed_op(
    ctx: &mut JitContext,
    dv: &DynValues,
    tv: &TypedValues,
    op: &Op,
    t: &mut usize,
    pushed: &mut bool,
) {
    if matches!(*op, Op::Status) {
        emit_status(ctx, dv.top, *t, *pushed);
        return;
    }
    if emit_nonvalue_op(ctx, dv, op) {
        return;
    }
    match *op {
        Op::IntegerMemory(..) | Op::Integer(_) | Op::Status => unreachable!(),
        Op::Load(a, width) => {
            typed_load(
                ctx,
                tv,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                width,
                tv.scratch,
            );
            *t = t.wrapping_sub(1) & 7;
            ctx.builder.get_local_f64(tv.scratch);
            ctx.builder.set_local_f64(tv.slot(*t));
            *pushed = true;
        },
        Op::Copy(r) => {
            ctx.builder.get_local_f64(tv.slot((*t + r as usize) & 7));
            ctx.builder.set_local_f64(tv.scratch);
            *t = t.wrapping_sub(1) & 7;
            ctx.builder.get_local_f64(tv.scratch);
            ctx.builder.set_local_f64(tv.slot(*t));
            *pushed = true;
        },
        Op::Store(a, width, pop) => {
            ctx.builder.get_local_f64(tv.slot(*t));
            ctx.builder.set_local_f64(tv.scratch);
            match width {
                FloatWidth::F32 => {
                    typed_store_f32(ctx, tv, &dv.addresses[a].0, dv.addresses[a].1, tv.scratch)
                },
                FloatWidth::F64 => {
                    typed_store_f64(ctx, &dv.addresses[a].0, dv.addresses[a].1, tv.scratch)
                },
            }
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::IntegerStore(a, pop, truncate) => {
            ctx.builder.get_local_f64(tv.slot(*t));
            ctx.builder.set_local_f64(tv.scratch);
            typed_store_i32(
                ctx,
                tv,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                tv.scratch,
                truncate,
            );
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::StoreRegister(r, pop) => {
            ctx.builder.get_local_f64(tv.slot(*t));
            ctx.builder.set_local_f64(tv.slot((*t + r as usize) & 7));
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::CompareMemory(a, width, pop) => {
            typed_load(
                ctx,
                tv,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                width,
                tv.scratch,
            );
            typed_compare(ctx, tv.slot(*t), tv.scratch);
            if pop {
                *t = (*t + 1) & 7;
            }
        },
        Op::CompareRegister(r, pops, quiet) => {
            if !quiet {
                ctx.builder.get_local_f64(tv.slot((*t + r as usize) & 7));
                ctx.builder.set_local_f64(tv.scratch);
            }
            typed_compare(ctx, tv.slot(*t), tv.slot((*t + r as usize) & 7));
            *t = (*t + pops as usize) & 7;
        },
        Op::Exchange(r) => {
            let other = (*t + r as usize) & 7;
            ctx.builder.get_local_f64(tv.slot(*t));
            ctx.builder.set_local_f64(tv.temporary);
            ctx.builder.get_local_f64(tv.slot(other));
            ctx.builder.set_local_f64(tv.slot(*t));
            ctx.builder.get_local_f64(tv.temporary);
            ctx.builder.set_local_f64(tv.slot(other));
        },
        Op::Negate => typed_negate(ctx, tv.slot(*t)),
        Op::Memory(kind, a, width) => {
            typed_load(
                ctx,
                tv,
                &dv.addresses[a].0,
                dv.addresses[a].1,
                width,
                tv.scratch,
            );
            typed_binary(ctx, tv, kind, tv.slot(*t), tv.scratch, tv.slot(*t));
        },
        Op::Register(kind, dest, r, pop) => {
            ctx.builder.get_local_f64(tv.slot((*t + r as usize) & 7));
            ctx.builder.set_local_f64(tv.scratch);
            typed_binary(
                ctx,
                tv,
                kind,
                tv.slot(*t),
                tv.scratch,
                tv.slot((*t + dest as usize) & 7),
            );
            if pop {
                *t = (*t + 1) & 7;
            }
        },
    }
}

// The SHADOW arm of store_value specialised to an f64 local: binary64 bits
// become the exact raw80 encoding without rounding or flag changes.
fn store_typed_value(ctx: &mut JitContext, address: &WasmLocal, extra: u32, value: &WasmLocalF64) {
    ctx.builder.get_local_f64(value);
    ctx.builder.reinterpret_f64_as_i64();
    let bits = ctx.builder.set_new_local_i64();
    ctx.builder.get_local(address);
    ctx.builder.const_i64(0);
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(11);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(i64::MIN);
    ctx.builder.or_i64();
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.select();
    ctx.builder.store_aligned_i64(extra);
    ctx.builder.get_local(address);
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(48);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x8000);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0);
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(52);
    ctx.builder.shr_u_i64();
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0x7ff);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x3c00);
    ctx.builder.add_i32();
    ctx.builder.get_local_i64(&bits);
    ctx.builder.const_i64(1);
    ctx.builder.shl_i64();
    ctx.builder.const_i64(0);
    ctx.builder.eq_i64();
    ctx.builder.select();
    ctx.builder.or_i32();
    ctx.builder.store_aligned_u16(8 + extra);
    ctx.builder.free_local_i64(bits);
}

fn emit_typed_epilogue(ctx: &mut JitContext, p: &Plan, dv: &DynValues, tv: &TypedValues) {
    for i in 0..8 {
        if p.written & (1 << i) != 0 {
            let address = slot_address(ctx, dv.top, i);
            store_typed_value(ctx, &address, 0, tv.slot(i as usize));
            ctx.builder.free_local(address);
        }
    }
    if p.ops.iter().any(op_writes_scratch) {
        ctx.builder.const_i32(gp::sse_scratch_register as i32);
        let address = ctx.builder.set_new_local();
        store_typed_value(ctx, &address, 0, tv.scratch);
        ctx.builder.free_local(address);
    }
    emit_run_state(ctx, p, dv.top, dv.tags);
}

fn fast(ctx: &mut JitContext, p: &Plan, fail: Label, typed: bool) {
    ctx.builder.load_fixed_i32(gp::cr as u32);
    ctx.builder.const_i32(cpu::CR0_EM | cpu::CR0_TS);
    ctx.builder.and_i32();
    ctx.builder.br_if(fail);
    ctx.builder.load_fixed_u16(gp::fpu_control_word as u32);
    ctx.builder.const_i32(0xc3f);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x3f);
    ctx.builder.ne_i32();
    ctx.builder.br_if(fail);
    ctx.builder.load_fixed_u8(gp::fpu_stack_ptr as u32);
    let top = ctx.builder.tee_new_local();
    ctx.builder.const_i32(8);
    ctx.builder.geu_i32();
    ctx.builder.br_if(fail);
    ctx.builder.load_fixed_u8(gp::fpu_stack_empty as u32);
    let tags = ctx.builder.set_new_local();
    for i in 0..8 {
        let full = p.required_full & (1 << i) != 0;
        let empty = p.required_empty & (1 << i) != 0;
        if !full && !empty {
            continue;
        }
        ctx.builder.get_local(&tags);
        slot_index(ctx, &top, i);
        ctx.builder.shr_u_i32();
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
        if empty {
            ctx.builder.eqz_i32();
        }
        ctx.builder.br_if(fail);
    }
    let addresses = emit_address_guards(ctx, p, fail);
    // After this point all instructions are nontrapping value operations or
    // proven ordinary-RAM loads/stores. No helper touches TOP/tags/F80 slots.
    let slots: Vec<_> = (0..8)
        .map(|i| {
            if (p.load | p.written) & (1 << i) == 0 {
                return None;
            }
            let v = Value::zero(ctx);
            if p.load & (1 << i) != 0 {
                let address = slot_address(ctx, &top, i);
                ctx.builder.get_local(&address);
                ctx.builder.load_aligned_i64(0);
                ctx.builder.set_local_i64(&v.low);
                ctx.builder.get_local(&address);
                ctx.builder.load_aligned_u16(8);
                ctx.builder.set_local(&v.high);
                ctx.builder.free_local(address);
                recover_native(ctx, &v);
            }
            Some(v)
        })
        .collect();
    let scratch = Value::zero(ctx);
    let temporary = Value::zero(ctx);
    let dv = DynValues {
        slots: &slots,
        scratch: &scratch,
        temporary: &temporary,
        addresses: &addresses,
        top: &top,
        tags: &tags,
    };
    if !typed {
        let mut t = 0usize;
        let mut pushed = false;
        for op in &p.ops {
            emit_dynamic_op(ctx, &dv, op, &mut t, &mut pushed);
        }
        dbg_assert!(t == p.top as usize);
        emit_dynamic_epilogue(ctx, p, &dv);
    }
    else {
        // The typed twin runs first on f64 locals. Op 0 is always resumable;
        // every op that can fail its typed checks adds a resume point.
        ctx.builder.const_i32(0);
        let resume = ctx.builder.set_new_local();
        let mut resume_points: Vec<usize> = vec![0];
        resume_points.extend(
            p.ops
                .iter()
                .enumerate()
                .filter(|(_, op)| op_can_bail(op))
                .map(|(k, _)| k),
        );
        let done = ctx.builder.block_void();
        let after_typed = ctx.builder.block_void();
        let bail = ctx.builder.block_void();
        let typed_slots: Vec<_> = (0..8)
            .map(|i| {
                if (p.load | p.written) & (1 << i) == 0 {
                    return None;
                }
                let dynamic = slots[i].as_ref().unwrap();
                ctx.builder.get_local_i64(&dynamic.low);
                ctx.builder.reinterpret_i64_as_f64();
                Some(ctx.builder.set_new_local_f64())
            })
            .collect();
        ctx.builder.get_local_i64(&scratch.low);
        ctx.builder.reinterpret_i64_as_f64();
        let typed_scratch = ctx.builder.set_new_local_f64();
        ctx.builder.get_local_i64(&temporary.low);
        ctx.builder.reinterpret_i64_as_f64();
        let typed_temporary = ctx.builder.set_new_local_f64();
        let tv = TypedValues {
            slots: &typed_slots,
            scratch: &typed_scratch,
            temporary: &typed_temporary,
            bail,
        };
        // Every slot loaded at entry must already be an ordinary value; the
        // twin never re-checks SHADOW tags. Otherwise resume at op 0.
        for i in 0..8 {
            if p.load & (1 << i) == 0 {
                continue;
            }
            ctx.builder.get_local(&slots[i].as_ref().unwrap().high);
            ctx.builder.const_i32(SHADOW as i32);
            ctx.builder.ne_i32();
            ctx.builder.br_if(after_typed);
        }
        let mut t = 0usize;
        let mut pushed = false;
        let mut next_case = 1usize;
        for op in &p.ops {
            if op_can_bail(op) {
                ctx.builder.const_i32(next_case as i32);
                ctx.builder.set_local(&resume);
                next_case += 1;
            }
            emit_typed_op(ctx, &dv, &tv, op, &mut t, &mut pushed);
            // A flag-clearing op stores 0 in its fast arm; replay it in the
            // twin so the flags byte has the same state at every point.
            if op_clears_flags(op) {
                emit_flag_clear(ctx);
            }
        }
        dbg_assert!(t == p.top as usize);
        emit_typed_epilogue(ctx, p, &dv, &tv);
        ctx.builder.br(done);
        // Shared bail body. Unread slots are unobservable either way: a
        // written-but-unloaded slot is never read before the dynamic sequence
        // writes it, and an untouched slot materialises its entry value. The
        // same holds for scratch and the exchange temporary: every dynamic
        // read follows a write inside the same op, and an untouched (0,0)
        // pair serialises to the same raw80 bytes as materialised (0,SHADOW).
        ctx.builder.block_end();
        emit_typed_materialization(ctx, &dv, &tv);
        ctx.builder.block_end();
        // Resume dispatch: nested blocks end just before each resumable op.
        // The block for a later resume point encloses the earlier one, so
        // opening them in reverse leaves each end at its op's position.
        let mut resume_labels: Vec<Label> = Vec::with_capacity(resume_points.len());
        for _ in 0..resume_points.len() {
            resume_labels.push(ctx.builder.block_void());
        }
        resume_labels.reverse();
        ctx.builder.get_local(&resume);
        ctx.builder.brtable(fail, &mut resume_labels.iter());
        ctx.builder.block_end();
        let mut next = 1usize;
        let mut t = 0usize;
        let mut pushed = false;
        for (k, op) in p.ops.iter().enumerate() {
            if next < resume_points.len() && k == resume_points[next] {
                ctx.builder.block_end();
                next += 1;
            }
            emit_dynamic_op(ctx, &dv, op, &mut t, &mut pushed);
        }
        dbg_assert!(t == p.top as usize);
        emit_dynamic_epilogue(ctx, p, &dv);
        ctx.builder.block_end();
        ctx.builder.free_local(resume);
        for v in typed_slots.into_iter().flatten() {
            ctx.builder.free_local_f64(v);
        }
        ctx.builder.free_local_f64(typed_scratch);
        ctx.builder.free_local_f64(typed_temporary);
    }
    for v in slots.into_iter().flatten() {
        v.free(ctx);
    }
    scratch.free(ctx);
    temporary.free(ctx);
    for a in addresses {
        ctx.builder.free_local(a.0);
    }
    ctx.builder.free_local(top);
    ctx.builder.free_local(tags);
}

pub fn generate(ctx: &mut JitContext, block: &BasicBlock) -> bool {
    if unsafe { fast_x87::vine_get_x87_mode() < 2 } || cfg!(feature = "profiler") {
        return false;
    }
    let mut op_limit = max_ops();
    let mut address_limit = max_addresses();
    let mut candidate = plan(&ctx.cpu, block, op_limit, address_limit);
    let (p, typed) = loop {
        let Some(p) = candidate
        else {
            return false;
        };
        let slot_count = (p.load | p.written).count_ones() as u8;
        let need_i32 = p.addresses.len() as u8 + slot_count + 14;
        let need_i64 = slot_count + 6;
        // The typed twin adds an f64 mirror of every participating slot plus
        // scratch/temporary, one resume i32 and a few transient locals. Runs
        // that cannot afford it keep the single dynamic sequence.
        let typed = unsafe { crate::jit::JIT_X87_TYPED_RUNS }
            && ctx
                .builder
                .can_allocate_typed_locals(need_i32 + 4, need_i64 + 4, 0, slot_count + 6);
        if typed
            || ctx
                .builder
                .can_allocate_typed_locals(need_i32, need_i64, 0, 0)
        {
            break (p, typed);
        }
        // Plans made with raised limits can outgrow the local budget
        // entirely; retry once at the fallback 32 ops / 16 addresses rather
        // than losing the run.
        if op_limit <= 32 && address_limit <= 16 {
            return false;
        }
        op_limit = 32;
        address_limit = 16;
        candidate = plan(&ctx.cpu, block, op_limit, address_limit);
    };
    let saved_stop_addr = ctx.block_stop_addr;
    let saved_guarded = ctx.in_guarded_mem_run;
    let done = ctx.builder.block_void();
    let fail = ctx.builder.block_void();
    let saved_cpu = ctx.cpu.clone();
    ctx.block_stop_addr = 0;
    ctx.in_guarded_mem_run = false;
    fast(ctx, &p, fail, typed);
    *ctx.cpu = saved_cpu;
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    ctx.builder.br(done);
    ctx.builder.block_end();
    ctx.block_stop_addr = p.end;
    while ctx.cpu.eip < p.end {
        let before = ctx.cpu.eip;
        ctx.start_of_current_instruction = before;
        let mut flags = 0;
        jit_instructions::jit_instruction(ctx, &mut flags);
        dbg_assert!(flags == 0 && ctx.cpu.eip > before && ctx.cpu.eip <= p.end);
        dbg_assert!(ctx.cpu.eip - before < 15);
        ctx.advance_instruction();
    }
    ctx.builder.block_end();
    ctx.block_stop_addr = saved_stop_addr;
    ctx.in_guarded_mem_run = saved_guarded;
    ctx.cpu.prefixes = 0;
    ctx.previous_instruction = Instruction::Other;
    ctx.current_instruction = Instruction::Other;
    ctx.mmx_task_switch_call_count = None;
    true
}

fn raw(low: u64, high: u32) -> F80 {
    if high != SHADOW {
        return F80 {
            mantissa: low,
            sign_exponent: high as u16,
        };
    }
    let exponent = (low >> 52 & 0x7ff) as u16;
    F80 {
        mantissa: if exponent == 0 { 0 } else { 1 << 63 | low << 11 },
        sign_exponent: (low >> 48) as u16 & 0x8000
            | if exponent == 0 { 0 } else { exponent + 0x3c00 },
    }
}
unsafe fn value_binary(a: u64, ah: u32, b: u64, bh: u32, kind: Math) -> u64 {
    let a = raw(a, ah);
    let b = raw(b, bh);
    let out = match kind {
        Math::Add => a + b,
        Math::Mul => a * b,
        Math::Sub => a - b,
        Math::ReverseSub => unreachable!(),
    };
    // Normal helper results rejoin the native value chain. Rare raw80 values
    // stay raw until a later operation makes them representable again.
    if let Some(value) = fast_x87::to_native(out) {
        if let Some(back) = fast_x87::from_native(value) {
            if back.mantissa == out.mantissa && back.sign_exponent == out.sign_exponent {
                HIGH = SHADOW;
                return value.to_bits();
            }
        }
    }
    HIGH = out.sign_exponent as u32;
    out.mantissa
}
#[no_mangle]
pub unsafe fn fpu_value_add(a: u64, ah: u32, b: u64, bh: u32) -> u64 {
    F80::clear_exception_flags();
    let result = value_binary(a, ah, b, bh, Math::Add);
    *gp::fpu_status_word |= F80::get_exception_flags() as u16;
    result
}
#[no_mangle]
pub unsafe fn fpu_value_mul(a: u64, ah: u32, b: u64, bh: u32) -> u64 {
    value_binary(a, ah, b, bh, Math::Mul)
}
#[no_mangle]
pub unsafe fn fpu_value_sub(a: u64, ah: u32, b: u64, bh: u32) -> u64 {
    value_binary(a, ah, b, bh, Math::Sub)
}
#[no_mangle]
pub unsafe fn fpu_value_from_f32(bits: u64, _unused: u32) -> u64 {
    let value = fpu::f32_to_f80(bits as i32);
    HIGH = value.sign_exponent as u32;
    value.mantissa
}
#[no_mangle]
pub unsafe fn fpu_value_to_f32(low: u64, high: u32) -> i32 { fpu::f80_to_f32(raw(low, high)) }
#[no_mangle]
pub unsafe fn fpu_value_from_f64(bits: u64, _unused: u32) -> u64 {
    let value = fpu::f64_to_f80(bits);
    HIGH = value.sign_exponent as u32;
    value.mantissa
}
#[no_mangle]
pub unsafe fn fpu_value_to_f64(low: u64, high: u32) -> u64 { fpu::f80_to_f64(raw(low, high)) }
#[no_mangle]
pub unsafe fn fpu_value_to_i32(low: u64, high: u32) -> i32 {
    fpu::fpu_convert_to_i32(raw(low, high))
}
#[no_mangle]
pub unsafe fn fpu_value_truncate_i32(low: u64, high: u32) -> i32 {
    fpu::fpu_truncate_to_i32(raw(low, high))
}

unsafe fn value_compare(a: u64, ah: u32, b: u64, bh: u32, quiet: bool) {
    let a = raw(a, ah);
    let b = raw(b, bh);
    F80::clear_exception_flags();
    let order = if quiet { a.partial_cmp_quiet(&b) } else { a.partial_cmp(&b) };
    let flags = match order {
        Some(std::cmp::Ordering::Greater) => 0,
        Some(std::cmp::Ordering::Less) => 0x100,
        Some(std::cmp::Ordering::Equal) => 0x4000,
        None => 0x4500,
    };
    *gp::fpu_status_word =
        *gp::fpu_status_word & !0x4700 | flags | F80::get_exception_flags() as u16;
}
#[no_mangle]
pub unsafe fn fpu_value_compare(a: u64, ah: u32, b: u64, bh: u32) {
    value_compare(a, ah, b, bh, false);
}
#[no_mangle]
pub unsafe fn fpu_value_compare_quiet(a: u64, ah: u32, b: u64, bh: u32) {
    value_compare(a, ah, b, bh, true);
}
