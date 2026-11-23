use crate::cpu::cpu::{
    tlb_data, FLAG_CARRY, FLAG_OVERFLOW, FLAG_SIGN, FLAG_ZERO, OPSIZE_16, OPSIZE_32, OPSIZE_8,
    TLB_GLOBAL, TLB_HAS_CODE, TLB_NO_USER, TLB_READONLY, TLB_VALID,
};
use crate::cpu::global_pointers;
use crate::cpu::memory;
use crate::jit::{Instruction, InstructionOperand, InstructionOperandDest, JitContext};
use crate::modrm;
use crate::modrm::ModrmByte;
use crate::opstats;
use crate::page::Page;
use crate::profiler;
use crate::regs;
use crate::wasmgen::wasm_builder::{WasmBuilder, WasmLocal, WasmLocalI64, WasmLocalV128};

pub fn gen_add_cs_offset(ctx: &mut JitContext) {
    if !ctx.cpu.has_flat_segmentation() {
        ctx.builder
            .load_fixed_i32(global_pointers::get_seg_offset(regs::CS));
        ctx.builder.add_i32();
    }
}

pub fn gen_get_eip(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::instruction_pointer as u32);
}

pub fn gen_set_eip_to_after_current_instruction(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(ctx.cpu.eip as i32 & 0xFFF);
    ctx.builder.or_i32();
    ctx.builder.store_aligned_i32(0);
}

pub fn gen_set_previous_eip_offset_from_eip_with_low_bits(
    builder: &mut WasmBuilder,
    low_bits: i32,
) {
    // previous_ip = instruction_pointer & ~0xFFF | low_bits;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::previous_ip as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_eip_low_bits(builder: &mut WasmBuilder, low_bits: i32) {
    // instruction_pointer = instruction_pointer & ~0xFFF | low_bits;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_eip_low_bits_and_jump_rel32(builder: &mut WasmBuilder, low_bits: i32, n: i32) {
    // instruction_pointer = (instruction_pointer & ~0xFFF | low_bits) + n;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    if n != 0 {
        builder.const_i32(n);
        builder.add_i32();
    }
    builder.store_aligned_i32(0);
}

pub fn gen_relative_jump(builder: &mut WasmBuilder, n: i32) {
    // add n to instruction_pointer
    if n != 0 {
        builder.const_i32(global_pointers::instruction_pointer as i32);
        gen_get_eip(builder);
        builder.const_i32(n);
        builder.add_i32();
        builder.store_aligned_i32(0);
    }
}

pub fn gen_page_switch_check(
    ctx: &mut JitContext,
    next_block_addr: u32,
    last_instruction_addr: u32,
) {
    // After switching a page while in jitted code, check if the page mapping still holds

    if let Some((base, permissions)) = unsafe { crate::cpu::vine::common_memory() }.filter(|_| {
        ctx.cpu.cpl3() && ctx.cpu.has_flat_segmentation() && !cfg!(feature = "profiler")
    }) {
        let mem_base = unsafe { memory::mem8 } as u32;
        if base >= mem_base {
            let guest_offset = base - mem_base;
            if next_block_addr >= guest_offset {
                let expected_virt_addr = next_block_addr - guest_offset;
                let page = expected_virt_addr >> 12;
                let (exec_enabled_ptr, exec_bitmap_ptr) =
                    crate::cpu::vine::execute_protection_pointers();

                gen_get_eip(ctx.builder);
                ctx.builder.const_i32(expected_virt_addr as i32);
                ctx.builder.ne_i32();
                ctx.builder.load_fixed_u8(permissions + page);
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();
                ctx.builder.eqz_i32();
                ctx.builder.or_i32();
                ctx.builder.load_fixed_u8(exec_enabled_ptr);
                ctx.builder.load_fixed_u8(exec_bitmap_ptr + (page >> 3));
                ctx.builder.const_i32(1 << (page & 7));
                ctx.builder.and_i32();
                ctx.builder.eqz_i32();
                ctx.builder.and_i32();
                ctx.builder.or_i32();

                if cfg!(debug_assertions) {
                    ctx.builder.if_void();
                    gen_profiler_stat_increment(ctx.builder, profiler::stat::FAILED_PAGE_CHANGE);
                    gen_debug_track_jit_exit(ctx.builder, last_instruction_addr);
                    ctx.builder.br(ctx.exit_label);
                    ctx.builder.block_end();
                }
                else {
                    ctx.builder.br_if(ctx.exit_label);
                }
                return;
            }
        }
    }

    gen_get_eip(ctx.builder);
    let address_local = ctx.builder.set_new_local();
    gen_get_phys_eip_plus_mem(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // The host may move RAM while the CPU is stopped without retiring this
    // module. Compare against the current allocation, never its former base.
    ctx.builder.const_i32(next_block_addr as i32);
    ctx.builder.load_fixed_i32((&raw const memory::mem8) as u32);
    ctx.builder.add_i32();
    ctx.builder.ne_i32();

    if cfg!(debug_assertions) {
        ctx.builder.if_void();
        gen_profiler_stat_increment(ctx.builder, profiler::stat::FAILED_PAGE_CHANGE);
        gen_debug_track_jit_exit(ctx.builder, last_instruction_addr);
        ctx.builder.br(ctx.exit_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if(ctx.exit_label);
    }
}

pub fn gen_update_instruction_counter(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(global_pointers::instruction_counter as i32);
    ctx.builder
        .load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.add_i32();
    ctx.builder.store_aligned_i32(0);
}

pub fn gen_get_reg8(ctx: &mut JitContext, r: u32) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(8);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        _ => assert!(false),
    }
}

/// Return a new local referencing one of the 8 bit registers or a direct reference to one of the
/// register locals. Higher bits might be garbage (suitable for gen_cmp8 etc.). Must be freed with
/// gen_free_reg8_or_alias.
pub fn gen_get_reg8_or_alias_to_reg32(ctx: &mut JitContext, r: u32) -> WasmLocal {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => ctx.register_locals[r as usize].unsafe_clone(),
        regs::AH | regs::CH | regs::DH | regs::BH => {
            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(8);
            ctx.builder.shr_u_i32();
            ctx.builder.set_new_local()
        },
        _ => panic!(),
    }
}

pub fn gen_free_reg8_or_alias(ctx: &mut JitContext, r: u32, local: WasmLocal) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {},
        regs::AH | regs::CH | regs::DH | regs::BH => ctx.builder.free_local(local),
        _ => panic!(),
    }
}

pub fn gen_get_reg16(ctx: &mut JitContext, r: u32) {
    ctx.builder.get_local(&ctx.register_locals[r as usize]);
    ctx.builder.const_i32(0xFFFF);
    ctx.builder.and_i32();
}

pub fn gen_get_reg32(ctx: &mut JitContext, r: u32) {
    ctx.builder.get_local(&ctx.register_locals[r as usize]);
}

pub fn gen_set_reg8(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            // reg32[r] = stack_value & 0xFF | reg32[r] & ~0xFF
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();

            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(!0xFF);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder.set_local(&ctx.register_locals[r as usize]);
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            // reg32[r] = stack_value << 8 & 0xFF00 | reg32[r] & ~0xFF00
            ctx.builder.const_i32(8);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(0xFF00);
            ctx.builder.and_i32();

            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(!0xFF00);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder
                .set_local(&ctx.register_locals[(r - 4) as usize]);
        },
        _ => assert!(false),
    }
}

pub fn gen_set_reg8_unmasked(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    if cfg!(debug_assertions) {
        let val = ctx.builder.set_new_local();
        ctx.builder.get_local(&val);
        ctx.builder.const_i32(!0xFF);
        ctx.builder.and_i32();
        ctx.builder.if_void();
        ctx.builder.unreachable();
        ctx.builder.block_end();
        ctx.builder.get_local(&val);
        ctx.builder.free_local(val);
    }

    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            // reg32[r] = stack_value | reg32[r] & ~0xFF
            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(!0xFF);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder.set_local(&ctx.register_locals[r as usize]);
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            // reg32[r] = stack_value << 8 | reg32[r] & ~0xFF00
            ctx.builder.const_i32(8);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(0xFF00);
            ctx.builder.and_i32();

            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(!0xFF00);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder
                .set_local(&ctx.register_locals[(r - 4) as usize]);
        },
        _ => assert!(false),
    }
}

pub fn gen_set_reg16(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    gen_set_reg16_local(ctx.builder, &ctx.register_locals[r as usize]);
}

pub fn gen_set_reg16_unmasked(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    if cfg!(debug_assertions) {
        let val = ctx.builder.set_new_local();
        ctx.builder.get_local(&val);
        ctx.builder.const_i32(!0xFFFF);
        ctx.builder.and_i32();
        ctx.builder.if_void();
        ctx.builder.unreachable();
        ctx.builder.block_end();
        ctx.builder.get_local(&val);
        ctx.builder.free_local(val);
    }

    ctx.builder.get_local(&ctx.reg(r));
    ctx.builder.const_i32(!0xFFFF);
    ctx.builder.and_i32();
    ctx.builder.or_i32();
    ctx.builder.set_local(&ctx.reg(r));
}

pub fn gen_set_reg16_local(builder: &mut WasmBuilder, local: &WasmLocal) {
    // reg32[r] = v & 0xFFFF | reg32[r] & ~0xFFFF
    builder.const_i32(0xFFFF);
    builder.and_i32();
    builder.get_local(local);
    builder.const_i32(!0xFFFF);
    builder.and_i32();
    builder.or_i32();
    builder.set_local(local);
}

pub fn gen_set_reg32(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    ctx.builder.set_local(&ctx.register_locals[r as usize]);
}

pub fn decr_exc_asize(ctx: &mut JitContext) {
    gen_get_reg32(ctx, regs::ECX);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    if ctx.cpu.asize_32() {
        gen_set_reg32(ctx, regs::ECX);
    }
    else {
        gen_set_reg16(ctx, regs::CX);
    }
}

pub fn gen_read_reg_xmm128_into_scratch(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    ctx.builder
        .load_fixed_v128(global_pointers::get_reg_xmm_offset(r));
    ctx.builder.store_aligned_v128(0);
}

pub fn gen_vec_cache_active(ctx: &JitContext) -> bool {
    ctx.mmx_tags_dirty
        || ctx.mmx_cache.iter().any(|x| x.is_some())
        || ctx.xmm_cache.iter().any(|x| x.is_some())
}

pub fn gen_vec_has_dirty(ctx: &JitContext) -> bool {
    ctx.mmx_dirty != 0 || ctx.mmx_tags_dirty || ctx.xmm_dirty != 0
}

fn gen_mmx_transition_now(ctx: &mut JitContext) {
    ctx.builder.const_i32(0);
    ctx.builder.const_i32(0);
    ctx.builder
        .store_u8(global_pointers::fpu_stack_empty as u32);
    ctx.builder.const_i32(0);
    ctx.builder.const_i32(0);
    ctx.builder.store_u8(global_pointers::fpu_stack_ptr as u32);
}

pub fn gen_vec_flush_for_cold_path(ctx: &mut JitContext) {
    if !gen_vec_has_dirty(ctx) {
        return;
    }
    for r in 0..8u32 {
        if ctx.mmx_dirty & (1 << r) != 0 {
            if let Some(ref local) = ctx.mmx_cache[r as usize] {
                let dest = global_pointers::get_reg_mmx_offset(r);
                ctx.builder.const_i32(dest as i32);
                ctx.builder.get_local_v128(local);
                ctx.builder.store_aligned_v128_lane64(0, 0);
                ctx.builder.const_i32(0);
                ctx.builder.const_i32(0xFFFF);
                ctx.builder.store_aligned_u16(dest + 8);
            }
        }
        if ctx.xmm_dirty & (1 << r) != 0 {
            if let Some(ref local) = ctx.xmm_cache[r as usize] {
                ctx.builder
                    .const_i32(global_pointers::get_reg_xmm_offset(r) as i32);
                ctx.builder.get_local_v128(local);
                ctx.builder.store_aligned_v128(0);
            }
        }
    }
    if ctx.mmx_tags_dirty {
        gen_mmx_transition_now(ctx);
    }
}

pub fn gen_vec_flush_and_clear(ctx: &mut JitContext) {
    gen_vec_flush_for_cold_path(ctx);
    for slot in &mut ctx.mmx_cache {
        if let Some(local) = slot.take() {
            ctx.builder.free_local_v128(local);
        }
    }
    for slot in &mut ctx.xmm_cache {
        if let Some(local) = slot.take() {
            ctx.builder.free_local_v128(local);
        }
    }
    ctx.mmx_dirty = 0;
    ctx.mmx_tags_dirty = false;
    ctx.xmm_dirty = 0;
}

pub fn gen_get_mmx_v128(ctx: &mut JitContext, r: u32) {
    if let Some(ref local) = ctx.mmx_cache[r as usize] {
        ctx.builder.get_local_v128(local);
        return;
    }
    ctx.builder
        .load_fixed_v128_zero64(global_pointers::get_reg_mmx_offset(r));
    if !cfg!(feature = "profiler") && ctx.builder.can_allocate_typed_locals(0, 0, 12, 0) {
        let local = ctx.builder.tee_new_local_v128();
        ctx.mmx_cache[r as usize] = Some(local);
    }
}

pub fn gen_set_mmx_v128(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    if let Some(ref local) = ctx.mmx_cache[r as usize] {
        ctx.builder.set_local_v128(local);
        ctx.mmx_dirty |= 1 << r;
        ctx.mmx_tags_dirty = true;
        return;
    }
    if !cfg!(feature = "profiler") && ctx.builder.can_allocate_typed_locals(0, 0, 12, 0) {
        let local = ctx.builder.set_new_local_v128();
        ctx.mmx_cache[r as usize] = Some(local);
        ctx.mmx_dirty |= 1 << r;
        ctx.mmx_tags_dirty = true;
        return;
    }
    let tmp = ctx.builder.set_new_local_v128();
    let dest = global_pointers::get_reg_mmx_offset(r);
    ctx.builder.const_i32(dest as i32);
    ctx.builder.get_local_v128(&tmp);
    ctx.builder.store_aligned_v128_lane64(0, 0);
    ctx.builder.free_local_v128(tmp);
    ctx.builder.const_i32(0);
    ctx.builder.const_i32(0xFFFF);
    ctx.builder.store_aligned_u16(dest + 8);
    gen_mmx_transition_now(ctx);
}

pub fn gen_mmx_mark_transition(ctx: &mut JitContext) {
    if !cfg!(feature = "profiler") {
        ctx.mmx_tags_dirty = true;
    }
    else {
        gen_mmx_transition_now(ctx);
    }
}

pub fn gen_get_xmm_v128(ctx: &mut JitContext, r: u32) {
    if let Some(ref local) = ctx.xmm_cache[r as usize] {
        ctx.builder.get_local_v128(local);
        return;
    }
    ctx.builder
        .load_fixed_v128(global_pointers::get_reg_xmm_offset(r));
    if !cfg!(feature = "profiler") && ctx.builder.can_allocate_typed_locals(0, 0, 12, 0) {
        let local = ctx.builder.tee_new_local_v128();
        ctx.xmm_cache[r as usize] = Some(local);
    }
}

pub fn gen_set_xmm_v128(ctx: &mut JitContext, r: u32) {
    ctx.instr_state_dirty = true;
    if let Some(ref local) = ctx.xmm_cache[r as usize] {
        ctx.builder.set_local_v128(local);
        ctx.xmm_dirty |= 1 << r;
        return;
    }
    if !cfg!(feature = "profiler") && ctx.builder.can_allocate_typed_locals(0, 0, 12, 0) {
        let local = ctx.builder.set_new_local_v128();
        ctx.xmm_cache[r as usize] = Some(local);
        ctx.xmm_dirty |= 1 << r;
        return;
    }
    let tmp = ctx.builder.set_new_local_v128();
    ctx.builder
        .const_i32(global_pointers::get_reg_xmm_offset(r) as i32);
    ctx.builder.get_local_v128(&tmp);
    ctx.builder.store_aligned_v128(0);
    ctx.builder.free_local_v128(tmp);
}

pub fn gen_get_sreg(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .load_fixed_u16(global_pointers::get_sreg_offset(r))
}

pub fn gen_get_ss_offset(ctx: &mut JitContext) {
    ctx.builder
        .load_fixed_i32(global_pointers::get_seg_offset(regs::SS));
}

pub fn gen_get_flags(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::flags as u32);
}
pub(crate) fn gen_get_flags_changed(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::flags_changed as u32);
}
fn gen_get_last_result(builder: &mut WasmBuilder, previous_instruction: &Instruction) {
    match previous_instruction {
        Instruction::Add {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::AdcSbb {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::Sub {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::Bitwise {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
        }
        | Instruction::NonZeroShift {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
        } => builder.get_local(&l),
        Instruction::Test {
            dest: InstructionOperandDest::WasmLocal(d),
            source: InstructionOperand::Immediate(i),
            opsize: OPSIZE_32,
        } => {
            builder.get_local(d);
            builder.const_i32(*i);
            builder.and_i32();
        },
        Instruction::Test {
            dest: InstructionOperandDest::WasmLocal(d),
            source: InstructionOperand::WasmLocal(s),
            opsize: OPSIZE_32,
        } => {
            builder.get_local(d);
            builder.get_local(s);
            builder.and_i32();
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(l),
            source,
            opsize: OPSIZE_32,
        } => {
            if source.is_zero() {
                builder.get_local(&l)
            }
            else {
                builder.load_fixed_i32(global_pointers::last_result as u32)
            }
        },
        _ => builder.load_fixed_i32(global_pointers::last_result as u32),
    }
}
fn gen_get_last_op_size(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::last_op_size as u32);
}
fn gen_get_last_op1(builder: &mut WasmBuilder, previous_instruction: &Instruction) {
    match previous_instruction {
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(l),
            source: _,
            opsize: OPSIZE_32,
        } => builder.get_local(&l),
        _ => builder.load_fixed_i32(global_pointers::last_op1 as u32),
    }
}

pub fn gen_readable_or_pagefault(ctx: &mut JitContext, address_local: &WasmLocal, size: i32) {
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(size);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    ctx.builder.call_fn3_ret("readable_or_pagefault_jit");
    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else if gen_vec_has_dirty(ctx) {
        ctx.builder.if_void();
        gen_vec_flush_for_cold_path(ctx);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if(ctx.exit_with_fault_label);
    }
}

pub fn gen_writable_or_pagefault(ctx: &mut JitContext, address_local: &WasmLocal, size: i32) {
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(size);
    // packed lower bits of eip and wasm table index
    ctx.builder.const_i32(
        ctx.start_of_current_instruction as i32 & 0xFFF
            | (ctx.wasm_table_index.to_u16() as i32) << 16,
    );
    ctx.builder.call_fn3_ret("writable_or_pagefault_jit");
    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else if gen_vec_has_dirty(ctx) {
        ctx.builder.if_void();
        gen_vec_flush_for_cold_path(ctx);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if(ctx.exit_with_fault_label);
    }
}

/// sign-extend a byte value on the stack and leave it on the stack
pub fn sign_extend_i8(builder: &mut WasmBuilder) {
    builder.const_i32(24);
    builder.shl_i32();
    builder.const_i32(24);
    builder.shr_s_i32();
}

/// sign-extend a two byte value on the stack and leave it on the stack
pub fn sign_extend_i16(builder: &mut WasmBuilder) {
    builder.const_i32(16);
    builder.shl_i32();
    builder.const_i32(16);
    builder.shr_s_i32();
}

pub fn gen_fn0_const(builder: &mut WasmBuilder, name: &str) { builder.call_fn0(name) }
pub fn gen_fn1_const(builder: &mut WasmBuilder, name: &str, arg0: u32) {
    builder.const_i32(arg0 as i32);
    builder.call_fn1(name);
}
pub fn gen_fn2_const(builder: &mut WasmBuilder, name: &str, arg0: u32, arg1: u32) {
    builder.const_i32(arg0 as i32);
    builder.const_i32(arg1 as i32);
    builder.call_fn2(name);
}

// helper functions for gen/generate_jit.js
pub fn gen_modrm_fn0(builder: &mut WasmBuilder, name: &str) {
    // generates: fn( _ )
    builder.call_fn1(name);
}
pub fn gen_modrm_fn1(builder: &mut WasmBuilder, name: &str, arg0: u32) {
    // generates: fn( _, arg0 )
    builder.const_i32(arg0 as i32);
    builder.call_fn2(name);
}

pub fn gen_modrm_resolve(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    modrm::gen(ctx, modrm_byte, 0)
}
pub fn gen_modrm_resolve_with_local(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    gen: &dyn Fn(&mut JitContext, &WasmLocal),
) {
    if let Some(r) = modrm::get_as_reg_index_if_possible(ctx, &modrm_byte) {
        gen(ctx, &ctx.reg(r));
    }
    else {
        gen_modrm_resolve(ctx, modrm_byte);
        let address = ctx.builder.set_new_local();
        gen(ctx, &address);
        ctx.builder.free_local(address);
    }
}
pub fn gen_modrm_resolve_with_esp_offset(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    esp_offset: i32,
) {
    modrm::gen(ctx, modrm_byte, esp_offset)
}

pub fn gen_set_reg8_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg8[r_dest] = reg8[r_src]
    if src != dest {
        gen_get_reg8(ctx, src);
        gen_set_reg8_unmasked(ctx, dest);
    }
}
pub fn gen_set_reg16_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg16[r_dest] = reg16[r_src]
    if src != dest {
        gen_get_reg16(ctx, src);
        gen_set_reg16_unmasked(ctx, dest);
    }
}
pub fn gen_set_reg32_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg32[r_dest] = reg32[r_src]
    if src != dest {
        gen_get_reg32(ctx, src);
        gen_set_reg32(ctx, dest);
    }
}

pub fn gen_modrm_resolve_safe_read8(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read8(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read16(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read16(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read32(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read64(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read128(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    where_to_write: u32,
) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| {
        gen_safe_read128(ctx, addr, where_to_write)
    });
}
pub fn gen_modrm_resolve_safe_read128_v128(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| {
        gen_safe_read128_v128(ctx, addr)
    });
}

pub fn gen_safe_read8(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::BYTE, address_local, None);
}
pub fn gen_safe_read16(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::WORD, address_local, None);
}
pub fn gen_safe_read32(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::DWORD, address_local, None);
}
pub fn gen_safe_read64(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::QWORD, &address_local, None);
}
pub fn gen_safe_read128(ctx: &mut JitContext, address_local: &WasmLocal, where_to_write: u32) {
    gen_safe_read(ctx, BitSize::DQWORD, &address_local, Some(where_to_write));
}
pub fn gen_safe_read128_v128(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::DQWORD, address_local, None);
}

// only used internally for gen_safe_write
#[allow(dead_code)]
enum GenSafeWriteValue<'a> {
    I32(&'a WasmLocal),
    I64(&'a WasmLocalI64),
    TwoI64s(&'a WasmLocalI64, &'a WasmLocalI64),
    FixedV128(u32),
    V128(&'a WasmLocalV128),
}

enum GenSafeReadWriteValue {
    I32(WasmLocal),
    I64(WasmLocalI64),
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum BitSize {
    BYTE,
    WORD,
    DWORD,
    QWORD,
    DQWORD,
}
impl BitSize {
    pub fn bytes(&self) -> u32 {
        match self {
            BitSize::BYTE => 1,
            BitSize::WORD => 2,
            BitSize::DWORD => 4,
            BitSize::QWORD => 8,
            BitSize::DQWORD => 16,
        }
    }
}

pub fn gen_safe_write8(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::BYTE,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write16(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::WORD,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write32(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::DWORD,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write64(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local: &WasmLocalI64,
) {
    gen_safe_write(
        ctx,
        BitSize::QWORD,
        address_local,
        GenSafeWriteValue::I64(value_local),
    )
}

#[allow(dead_code)]
pub fn gen_safe_write128(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local_low: &WasmLocalI64,
    value_local_high: &WasmLocalI64,
) {
    gen_safe_write(
        ctx,
        BitSize::DQWORD,
        address_local,
        GenSafeWriteValue::TwoI64s(value_local_low, value_local_high),
    )
}

#[allow(dead_code)]
pub fn gen_safe_write128_from_fixed(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    source_fixed_addr: u32,
) {
    gen_safe_write(
        ctx,
        BitSize::DQWORD,
        address_local,
        GenSafeWriteValue::FixedV128(source_fixed_addr),
    )
}

pub fn gen_safe_write128_v128(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local: &WasmLocalV128,
) {
    gen_safe_write(
        ctx,
        BitSize::DQWORD,
        address_local,
        GenSafeWriteValue::V128(value_local),
    )
}

// Emits `entry = address + base` and loads the shared permission byte of the
// address' page into a fresh local for the Vine common arena. The caller
// tests the base bits (1 = read, 2 = write) directly for single-byte accesses
// or the span bits (4 = read on this and the next page, 8 = write on both) for
// wider accesses. Returns None when the common arena does not apply and the
// caller must use gen_memory_guard's tlb_data path instead.
// With config 18 bit 1, common-memory accesses fold the arena base
// into the wasm memarg -- `load/store offset=base (address)` -- instead of
// materialising `address + base` in a local. Memarg addressing uses
// `zext(address) + base` without a 32-bit wrap; every address the guard
// accepts lies inside the arena, so the sum stays within the wasm memory
// and cannot wrap: the effective address equals the old wrapping add for
// every accepted access.
fn cg_memarg() -> bool { (unsafe { crate::jit::JIT_CG_OPTIONS } & 2) != 0 }

// Headroom so every small positive access offset added on top of the folded
// base (runs, x87 byte counts, SIMD streams) still fits the u32 memarg.
const MEMARG_HEADROOM: u32 = 1 << 20;

/// The arena base if common memory applies and the memarg fold is enabled
/// for it; `None` means emit the `address + base` local.
pub(crate) fn common_memarg_base(ctx: &JitContext) -> Option<u32> {
    let (base, _) = unsafe { crate::cpu::vine::common_memory() }
        .filter(|_| ctx.cpu.cpl3() && !cfg!(feature = "profiler"))?;
    if cg_memarg() && base <= u32::MAX - MEMARG_HEADROOM {
        Some(base)
    }
    else {
        None
    }
}

/// Access descriptor produced by a common-memory guard.
pub(crate) enum MemOperand {
    /// `local` holds the resolved wasm address; accesses use `offset=0`.
    Resolved(WasmLocal),
    /// Accesses use `get_local(address)` and `offset=base`.
    Guest { base: u32 },
}
impl MemOperand {
    pub fn extra(&self) -> u32 {
        match self {
            MemOperand::Resolved(_) => 0,
            MemOperand::Guest { base } => *base,
        }
    }
}

fn gen_common_memory_guard(
    ctx: &mut JitContext,
    address: &WasmLocal,
) -> Option<(MemOperand, WasmLocal)> {
    let (base, permissions) = unsafe { crate::cpu::vine::common_memory() }
        .filter(|_| ctx.cpu.cpl3() && !cfg!(feature = "profiler"))?;
    let operand = if common_memarg_base(ctx).is_some() {
        MemOperand::Guest { base }
    }
    else {
        ctx.builder.get_local(address);
        ctx.builder.const_i32(base as i32);
        ctx.builder.add_i32();
        MemOperand::Resolved(ctx.builder.set_new_local())
    };
    ctx.builder.get_local(address);
    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.load_u8(permissions);
    let perm = ctx.builder.set_new_local();
    Some((operand, perm))
}

// Emits `(perm & mask) == mask` on the operand stack.
fn gen_perm_mask_test(ctx: &mut JitContext, perm: &WasmLocal, mask: i32) {
    ctx.builder.get_local(perm);
    ctx.builder.const_i32(mask);
    ctx.builder.and_i32();
    if mask != 1 {
        ctx.builder.const_i32(mask);
        ctx.builder.eq_i32();
    }
}

// Emits `(address & 0xfff) <= 0x1000 - bytes` on the operand stack.
fn gen_within_page_test(ctx: &mut JitContext, address: &WasmLocal, bytes: i32) {
    ctx.builder.get_local(address);
    ctx.builder.const_i32(0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(0x1000 - bytes);
    ctx.builder.le_i32();
}

// Common memory is restricted to Runtime-owned user mappings. Its permission
// table covers every u32 address; denied/unmapped/high pages take the existing
// precise slow path. Writable executable pages never use direct stores, so the
// original SMC barrier and interpreter retry remain in force.
pub(crate) fn gen_memory_guard(
    ctx: &mut JitContext,
    address: &WasmLocal,
    read: bool,
    write: bool,
) -> (MemOperand, bool) {
    if let Some((base, permissions)) = unsafe { crate::cpu::vine::common_memory() }
        .filter(|_| ctx.cpu.cpl3() && !cfg!(feature = "profiler"))
    {
        let operand = if common_memarg_base(ctx).is_some() {
            MemOperand::Guest { base }
        }
        else {
            ctx.builder.get_local(address);
            ctx.builder.const_i32(base as i32);
            ctx.builder.add_i32();
            MemOperand::Resolved(ctx.builder.set_new_local())
        };
        ctx.builder.get_local(address);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        ctx.builder.load_u8(permissions);
        let mask = i32::from(read) | i32::from(write) << 1;
        ctx.builder.const_i32(mask);
        ctx.builder.and_i32();
        if mask != 1 {
            ctx.builder.const_i32(mask);
            ctx.builder.eq_i32();
        }
        (operand, true)
    }
    else {
        ctx.builder.get_local(address);
        ctx.builder.const_i32(12);
        ctx.builder.shr_u_i32();
        ctx.builder.const_i32(2);
        ctx.builder.shl_i32();
        ctx.builder
            .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
        let entry = ctx.builder.tee_new_local();
        let mask = 0xfff
            & !TLB_GLOBAL
            & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })
            & !(if write { 0 } else { TLB_READONLY | TLB_HAS_CODE });
        ctx.builder.const_i32(mask);
        ctx.builder.and_i32();
        ctx.builder.const_i32(TLB_VALID);
        ctx.builder.eq_i32();
        (MemOperand::Resolved(entry), false)
    }
}

pub(crate) fn gen_resolved_memory_address(
    ctx: &mut JitContext,
    entry: &WasmLocal,
    address: &WasmLocal,
    common: bool,
) {
    ctx.builder.get_local(entry);
    if !common {
        ctx.builder.const_i32(!0xfff);
        ctx.builder.and_i32();
        ctx.builder.get_local(address);
        ctx.builder.xor_i32();
    }
}

/// Guard exits (set_jit_config 14): a failed common-memory guard
/// exits the generated module instead of calling the rejoining slow helper;
/// the main loop then interprets the instruction, which performs the access
/// exactly as for uncompiled code. Eligible only while the whole instruction
/// can still be re-executed exactly:
/// - fault_precharge is Some, so the interpreter's retired-instruction
///   accounting stays exact;
/// - the instruction is on the generated block's own page (block_stop_addr is
///   its end and is cleared for non-block contexts like emit_slow_call), so
///   publishing eip with gen_set_eip_low_bits restores its start;
/// - the block is not on a copied or inlined leaf page whose guest bytes may
///   be rewritten before they run, and not inside a guarded memory run;
/// - nothing emitted for the instruction so far changed architectural state
///   other than a guest-memory access: no register local write
///   (instr_state_dirty), no vector cache write and no imported call
///   (callers like fpu_* or io helpers could mutate state or be observable
///   on re-execution).
fn gen_guard_bail_eligible(ctx: &mut JitContext) -> bool {
    (unsafe { crate::jit::JIT_GUARD_EXIT }) & 1 != 0
        && ctx.fault_precharge.is_some()
        && !ctx.in_guarded_mem_run
        && !ctx.copied_leaf_page
        && ctx.block_stop_addr != 0
        && Page::page_of(ctx.start_of_current_instruction) == Page::page_of(ctx.block_stop_addr)
        && !ctx.instr_state_dirty
        && ctx.builder.imported_call_count() == ctx.instr_calls_base
        && !ctx
            .guard_exit_disabled
            .contains(&Page::page_of(ctx.start_of_current_instruction))
}

/// Emit the guard-exit arm for a failed common-memory guard: flush the
/// vector cache, undo the block's pre-charge for the current and later
/// instructions (the interpreter retires them), publish EIP at the
/// instruction start, flag the bail and take the normal exit arm, which
/// spills the register locals and publishes the instruction counter.
fn gen_guard_bail(ctx: &mut JitContext) {
    gen_vec_flush_for_cold_path(ctx);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.const_i32(ctx.fault_precharge.unwrap() as i32);
    ctx.builder.sub_i32();
    ctx.builder.set_local(&ctx.instruction_counter);
    gen_set_eip_low_bits(ctx.builder, ctx.start_of_current_instruction as i32 & 0xFFF);
    ctx.builder
        .const_i32(&raw mut crate::cpu::cpu::vine_guard_bail_pending as i32);
    ctx.builder.const_i32(1);
    ctx.builder.store_u8(0);
    ctx.builder.br(ctx.exit_label);
}

fn gen_safe_read(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    where_to_write: Option<u32>,
) {
    // Execute a virtual memory read. All slow paths (memory-mapped IO, tlb miss, page fault and
    // read across page boundary are handled in safe_read_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   if entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes: goto fast
    //   entry <- safe_read_jit_slow(addr, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: mem[(entry & ~0xFFF) ^ addr]

    // With the memarg fold the guard-passed fast path uses
    // `load offset=base (address)`; the slow path resolves an absolute wasm
    // address (possibly outside the arena) and keeps offset=0, so the two
    // merge at a result block instead of sharing a tail.
    let done = common_memarg_base(ctx).map(|_| match bits {
        BitSize::DQWORD if where_to_write.is_some() => ctx.builder.block_void(),
        BitSize::DQWORD => ctx.builder.block_v128(),
        BitSize::QWORD => ctx.builder.block_i64(),
        _ => ctx.builder.block_i32(),
    });
    let cont = ctx.builder.block_void();
    let saved_mmx_cr0 = ctx.mmx_task_switch_call_count;
    let calls_before = ctx.builder.imported_call_count();
    let (operand, common, bail) =
        if let Some((operand, perm)) = gen_common_memory_guard(ctx, address_local) {
            let bail = gen_guard_bail_eligible(ctx);
            if bits == BitSize::BYTE {
                // A single byte never crosses a page; read permission suffices.
                gen_perm_mask_test(ctx, &perm, 1);
            }
            else {
                // Span bit: this and the next page are both readable, so no
                // page-boundary check is needed at all.
                ctx.builder.get_local(&perm);
                ctx.builder.const_i32(4);
                ctx.builder.and_i32();
                ctx.builder.br_if(cont);
                gen_perm_mask_test(ctx, &perm, 1);
                gen_within_page_test(ctx, address_local, bits.bytes() as i32);
                ctx.builder.and_i32();
            }
            ctx.builder.br_if(cont);
            ctx.builder.free_local(perm);
            (operand, true, bail)
        }
        else {
            let (operand, common) = gen_memory_guard(ctx, address_local, true, false);
            if bits != BitSize::BYTE {
                gen_within_page_test(ctx, address_local, bits.bytes() as i32);
                ctx.builder.and_i32();
            }
            ctx.builder.br_if(cont);
            (operand, common, false)
        };
    let entry_local = match &operand {
        MemOperand::Resolved(l) => l.unsafe_clone(),
        MemOperand::Guest { .. } => ctx.builder.alloc_local(),
    };
    // where_to_write is only used by dqword
    dbg_assert!(where_to_write.is_none() || bits == BitSize::DQWORD);
    let emit_load = |ctx: &mut JitContext, offset: u32| match bits {
        BitSize::BYTE => ctx.builder.load_u8(offset),
        BitSize::WORD => ctx.builder.load_unaligned_u16(offset),
        BitSize::DWORD => ctx.builder.load_unaligned_i32(offset),
        BitSize::QWORD => ctx.builder.load_unaligned_i64(offset),
        BitSize::DQWORD => {
            if let Some(where_to_write) = where_to_write {
                let virt_address_local = ctx.builder.set_new_local();
                ctx.builder.const_i32(where_to_write as i32);
                ctx.builder.get_local(&virt_address_local);
                ctx.builder.load_unaligned_v128(offset);
                ctx.builder.store_aligned_v128(0);
                ctx.builder.free_local(virt_address_local);
            }
            else {
                ctx.builder.load_unaligned_v128(offset);
            }
        },
    };

    if common && bail {
        unsafe { crate::jit::GUARD_EXIT_SITES = crate::jit::GUARD_EXIT_SITES.wrapping_add(1) };
        gen_guard_bail(ctx);
    }
    else {
        if common && (unsafe { crate::jit::JIT_GUARD_EXIT }) & 1 != 0 {
            unsafe { crate::jit::GUARD_KEPT_SITES = crate::jit::GUARD_KEPT_SITES.wrapping_add(1) };
        }
        if cfg!(feature = "profiler") {
            ctx.builder.get_local(&address_local);
            ctx.builder.get_local(&entry_local);
            ctx.builder.call_fn2("report_safe_read_jit_slow");
        }

        ctx.builder.get_local(&address_local);
        ctx.builder
            .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn2_ret("safe_read8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn2_ret("safe_read16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn2_ret("safe_read32s_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder.call_fn2_ret("safe_read64s_slow_jit");
            },
            BitSize::DQWORD => {
                ctx.builder.call_fn2_ret("safe_read128s_slow_jit");
            },
        }
        ctx.builder.tee_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();

        if cfg!(feature = "profiler") {
            ctx.builder.if_void();
            gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
            ctx.builder.block_end();

            ctx.builder.get_local(&entry_local);
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        }

        if gen_vec_has_dirty(ctx) || ctx.fault_precharge.is_some() {
            ctx.builder.if_void();
            gen_vec_flush_for_cold_path(ctx);
            if let Some(k) = ctx.fault_precharge {
                // On a real CPU exception (bit 1 of the slow helper's exit code)
                // undo the pre-charge for the instructions after the faulting one;
                // the interpreter counts the faulting instruction as retired. A
                // bail (bit 1 clear, e.g. SMC) keeps the full pre-charge exactly as
                // before. gen_update_instruction_counter publishes the local.
                if k > 1 {
                    ctx.builder.get_local(&ctx.instruction_counter);
                    ctx.builder.const_i32(k as i32 - 1);
                    ctx.builder.get_local(&entry_local);
                    ctx.builder.const_i32(1);
                    ctx.builder.shr_u_i32();
                    ctx.builder.const_i32(1);
                    ctx.builder.and_i32();
                    ctx.builder.mul_i32();
                    ctx.builder.sub_i32();
                    ctx.builder.set_local(&ctx.instruction_counter);
                }
            }
            ctx.builder.br(ctx.exit_with_fault_label);
            ctx.builder.block_end();
        }
        else {
            ctx.builder.br_if(ctx.exit_with_fault_label);
        }
        if common {
            gen_resolved_memory_address(ctx, &entry_local, address_local, false);
            ctx.builder.set_local(&entry_local);
            if saved_mmx_cr0 == Some(calls_before) {
                ctx.mmx_task_switch_call_count = Some(ctx.builder.imported_call_count());
            }
        }
        if let Some(done) = done {
            ctx.builder.get_local(&entry_local);
            emit_load(ctx, 0);
            ctx.builder.br(done);
        }
    }
    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST); // XXX: Both fast and slow

    match &operand {
        MemOperand::Guest { base } => {
            ctx.builder.get_local(address_local);
            emit_load(ctx, *base);
            ctx.builder.block_end();
        },
        MemOperand::Resolved(_) => {
            gen_resolved_memory_address(ctx, &entry_local, address_local, common);
            emit_load(ctx, 0);
        },
    }

    ctx.builder.free_local(entry_local);
}

pub fn gen_get_phys_eip_plus_mem(ctx: &mut JitContext, address_local: &WasmLocal) {
    // Similar to gen_safe_read, but return the physical eip + memory::mem rather than reading from memory
    // In functions that need to use this value we need to fix it by substracting memory::mem
    // this is done in order to remove one instruction from the fast path of memory accesses (no need to add
    // memory::mem anymore ).
    // We need to account for this in gen_page_switch_check and we compare with next_block_addr + memory::mem8
    // We cannot the same while processing an AbsoluteEip flow control change so there we need to fix the value
    // by subscracting memory::mem. Overall, since AbsoluteEip is encountered less often than memory accesses so
    // this ends up improving perf.
    // Does not (need to) handle mapped memory
    // XXX: Currently does not use ctx.start_of_current_instruction, but rather assumes that eip is
    //      already correct (pointing at the current instruction)

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder.const_i32(
        (0xFFF
            & !TLB_READONLY
            & !TLB_GLOBAL
            & !TLB_HAS_CODE
            & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32,
    );
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    ctx.builder.br_if(cont);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_jit_slow");
    }

    ctx.builder.get_local(&address_local);
    ctx.builder.call_fn1_ret("get_phys_eip_slow_jit");

    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction); // XXX
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    ctx.builder.br_if(ctx.exit_with_fault_label);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    ctx.builder.free_local(entry_local);
}

/// Emit `<push value>; store offset` for an operand already on the stack
/// (resolved `entry` without the memarg fold, `get_local(address)` with
/// `offset = base` under the memarg fold).
fn gen_write_value_tail(
    ctx: &mut JitContext,
    bits: BitSize,
    value_local: &GenSafeWriteValue,
    offset: u32,
) {
    match *value_local {
        GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
        GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            assert!(bits == BitSize::DQWORD);

            let virt_address_local = ctx.builder.tee_new_local();
            ctx.builder.get_local_i64(local1);
            ctx.builder.store_unaligned_i64(offset);

            ctx.builder.get_local(&virt_address_local);
            ctx.builder.get_local_i64(local2);
            ctx.builder.store_unaligned_i64(offset + 8);
            ctx.builder.free_local(virt_address_local);
        },
        GenSafeWriteValue::FixedV128(source_fixed_addr) => {
            assert!(bits == BitSize::DQWORD);
            ctx.builder.load_fixed_v128(source_fixed_addr);
            ctx.builder.store_unaligned_v128(offset);
        },
        GenSafeWriteValue::V128(local) => {
            assert!(bits == BitSize::DQWORD);
            ctx.builder.get_local_v128(local);
            ctx.builder.store_unaligned_v128(offset);
        },
    }
    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(offset);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(offset);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(offset);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(offset);
        },
        BitSize::DQWORD => {}, // handled above
    }
}

fn gen_safe_write(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    value_local: GenSafeWriteValue,
) {
    // Execute a virtual memory write. All slow paths (memory-mapped IO, tlb miss, page fault,
    // write across page boundary and page containing jitted code are handled in safe_write_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   if entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes: goto fast
    //   entry <- safe_write_jit_slow(addr, value, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: mem[(entry & ~0xFFF) ^ addr] <- value

    let done = common_memarg_base(ctx).map(|_| ctx.builder.block_void());
    let cont = ctx.builder.block_void();
    let saved_mmx_cr0 = ctx.mmx_task_switch_call_count;
    let calls_before = ctx.builder.imported_call_count();
    let (operand, common, bail) =
        if let Some((operand, perm)) = gen_common_memory_guard(ctx, address_local) {
            let bail = gen_guard_bail_eligible(ctx);
            if bits == BitSize::BYTE {
                // A single byte never crosses a page; write permission suffices.
                gen_perm_mask_test(ctx, &perm, 2);
            }
            else {
                // Span bit: this and the next page are both writable, so no
                // page-boundary check is needed at all.
                ctx.builder.get_local(&perm);
                ctx.builder.const_i32(8);
                ctx.builder.and_i32();
                ctx.builder.br_if(cont);
                gen_perm_mask_test(ctx, &perm, 2);
                gen_within_page_test(ctx, address_local, bits.bytes() as i32);
                ctx.builder.and_i32();
            }
            ctx.builder.br_if(cont);
            ctx.builder.free_local(perm);
            (operand, true, bail)
        }
        else {
            let (operand, common) = gen_memory_guard(ctx, address_local, false, true);
            if bits != BitSize::BYTE {
                gen_within_page_test(ctx, address_local, bits.bytes() as i32);
                ctx.builder.and_i32();
            }
            ctx.builder.br_if(cont);
            (operand, common, false)
        };
    let entry_local = match &operand {
        MemOperand::Resolved(l) => l.unsafe_clone(),
        MemOperand::Guest { .. } => ctx.builder.alloc_local(),
    };

    if common && bail {
        unsafe { crate::jit::GUARD_EXIT_SITES = crate::jit::GUARD_EXIT_SITES.wrapping_add(1) };
        gen_guard_bail(ctx);
    }
    else {
        if common && (unsafe { crate::jit::JIT_GUARD_EXIT }) & 1 != 0 {
            unsafe { crate::jit::GUARD_KEPT_SITES = crate::jit::GUARD_KEPT_SITES.wrapping_add(1) };
        }
        if cfg!(feature = "profiler") {
            ctx.builder.get_local(&address_local);
            ctx.builder.get_local(&entry_local);
            ctx.builder.call_fn2("report_safe_write_jit_slow");
        }

        ctx.builder.get_local(&address_local);
        match value_local {
            GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
            GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
            GenSafeWriteValue::TwoI64s(local1, local2) => {
                ctx.builder.get_local_i64(local1);
                ctx.builder.get_local_i64(local2)
            },
            GenSafeWriteValue::FixedV128(source_fixed_addr) => {
                ctx.builder.load_fixed_i64(source_fixed_addr);
                ctx.builder.load_fixed_i64(source_fixed_addr + 8);
            },
            GenSafeWriteValue::V128(local) => {
                ctx.builder.get_local_v128(local);
                ctx.builder.extract_i64x2_lane(0);
                ctx.builder.get_local_v128(local);
                ctx.builder.extract_i64x2_lane(1);
            },
        }

        // packed lower bits of eip and wasm table index
        ctx.builder.const_i32(
            ctx.start_of_current_instruction as i32 & 0xFFF
                | (ctx.wasm_table_index.to_u16() as i32) << 16,
        );

        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn3_ret("safe_write8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn3_ret("safe_write16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn3_ret("safe_write32_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder
                    .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
            },
            BitSize::DQWORD => {
                ctx.builder
                    .call_fn4_i32_i64_i64_i32_ret("safe_write128_slow_jit");
            },
        }
        ctx.builder.tee_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();

        if cfg!(feature = "profiler") {
            ctx.builder.if_void();
            gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
            ctx.builder.block_end();

            ctx.builder.get_local(&entry_local);
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        }

        if gen_vec_has_dirty(ctx) || ctx.fault_precharge.is_some() {
            ctx.builder.if_void();
            gen_vec_flush_for_cold_path(ctx);
            if let Some(k) = ctx.fault_precharge {
                // On a real CPU exception (bit 1 of the slow helper's exit code)
                // undo the pre-charge for the instructions after the faulting one;
                // the interpreter counts the faulting instruction as retired. A
                // bail (bit 1 clear, e.g. SMC) keeps the full pre-charge exactly as
                // before. gen_update_instruction_counter publishes the local.
                if k > 1 {
                    ctx.builder.get_local(&ctx.instruction_counter);
                    ctx.builder.const_i32(k as i32 - 1);
                    ctx.builder.get_local(&entry_local);
                    ctx.builder.const_i32(1);
                    ctx.builder.shr_u_i32();
                    ctx.builder.const_i32(1);
                    ctx.builder.and_i32();
                    ctx.builder.mul_i32();
                    ctx.builder.sub_i32();
                    ctx.builder.set_local(&ctx.instruction_counter);
                }
            }
            ctx.builder.br(ctx.exit_with_fault_label);
            ctx.builder.block_end();
        }
        else {
            ctx.builder.br_if(ctx.exit_with_fault_label);
        }
        if common {
            gen_resolved_memory_address(ctx, &entry_local, address_local, false);
            ctx.builder.set_local(&entry_local);
            if saved_mmx_cr0 == Some(calls_before) {
                ctx.mmx_task_switch_call_count = Some(ctx.builder.imported_call_count());
            }
        }
        if let Some(done) = done {
            ctx.builder.get_local(&entry_local);
            gen_write_value_tail(ctx, bits, &value_local, 0);
            ctx.builder.br(done);
        }
    }
    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_WRITE_FAST); // XXX: Both fast and slow

    match &operand {
        MemOperand::Guest { base } => {
            ctx.builder.get_local(address_local);
            gen_write_value_tail(ctx, bits, &value_local, *base);
            ctx.builder.block_end();
        },
        MemOperand::Resolved(_) => {
            gen_resolved_memory_address(ctx, &entry_local, address_local, common);
            gen_write_value_tail(ctx, bits, &value_local, 0);
        },
    }

    ctx.builder.free_local(entry_local);
}

/// The `if !can_use_fast_path { safe_write*_slow_jit(addr, value, eip) }`
/// arm of a read+write access.
fn gen_rmw_write_slow_check(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    value_local: &GenSafeReadWriteValue,
    can_use_fast_path_local: &WasmLocal,
) {
    ctx.builder.get_local(can_use_fast_path_local);
    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    {
        ctx.builder.get_local(address_local);

        match value_local {
            GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
            GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
        }

        // packed lower bits of eip and wasm table index
        ctx.builder.const_i32(
            ctx.start_of_current_instruction as i32 & 0xFFF
                | (ctx.wasm_table_index.to_u16() as i32) << 16,
        );

        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn3_ret("safe_write8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn3_ret("safe_write16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn3_ret("safe_write32_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder
                    .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
            },
            BitSize::DQWORD => {
                dbg_assert!(false);
            },
        }

        if cfg!(debug_assertions) {
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.if_void();
            {
                // handled above
                ctx.builder.const_i32(match bits {
                    BitSize::BYTE => 8,
                    BitSize::WORD => 16,
                    BitSize::DWORD => 32,
                    BitSize::QWORD => 64,
                    _ => {
                        dbg_assert!(false);
                        0
                    },
                });
                ctx.builder.get_local(address_local);
                ctx.builder.call_fn2("bug_gen_safe_read_write_page_fault");
            }
            ctx.builder.block_end();
        }
        else {
            ctx.builder.drop_();
        }
    }
    ctx.builder.block_end();
}

/// Emit `value <- f(mem[operand+offset]); mem[operand+offset] <- value`.
/// The operand emission is `get_local(operand)`; the memarg fold passes the
/// guest address local and `offset = base`. `can_use_fast_path_local` is
/// `Some` on slow-path completions, where the write-back check is dynamic;
/// it is `None` on the guard-passed fast path, where the common guard makes
/// it statically true.
fn gen_rmw_tail(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    operand: &WasmLocal,
    offset: u32,
    can_use_fast_path_local: Option<&WasmLocal>,
    f: &dyn Fn(&mut JitContext),
) {
    ctx.builder.get_local(operand);

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(offset);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(offset);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(offset);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(offset);
        },
        BitSize::DQWORD => assert!(false), // not used
    }

    // value is now on stack

    f(ctx);

    // TODO: Could get rid of this local by returning one from f
    let value_local = if bits == BitSize::QWORD {
        GenSafeReadWriteValue::I64(ctx.builder.set_new_local_i64())
    }
    else {
        GenSafeReadWriteValue::I32(ctx.builder.set_new_local())
    };

    if let Some(can_use_fast_path_local) = can_use_fast_path_local {
        gen_rmw_write_slow_check(
            ctx,
            bits,
            address_local,
            &value_local,
            can_use_fast_path_local,
        );
    }

    ctx.builder.get_local(operand);
    match &value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
    }

    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(offset);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(offset);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(offset);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(offset);
        },
        BitSize::DQWORD => {
            dbg_assert!(false);
        },
    }

    match value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.free_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.free_local_i64(l),
    }
}

pub fn gen_safe_read_write(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    f: &dyn Fn(&mut JitContext),
) {
    // Execute a virtual memory read+write. All slow paths (memory-mapped IO, tlb miss, page fault,
    // write across page boundary and page containing jitted code are handled in
    // safe_read_write_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   can_use_fast_path <- entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes
    //   if can_use_fast_path: goto fast
    //   entry <- safe_read_write_jit_slow(addr, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: value <- f(mem[(entry & ~0xFFF) ^ addr])
    //   if !can_use_fast_path { safe_write_jit_slow(addr, value, instruction_pointer) }
    //   mem[(entry & ~0xFFF) ^ addr] <- value

    let done = common_memarg_base(ctx).map(|_| ctx.builder.block_void());
    let cont = ctx.builder.block_void();
    let (operand, common, can_use_fast_path_local, bail) =
        if let Some((operand, perm)) = gen_common_memory_guard(ctx, address_local) {
            let bail = gen_guard_bail_eligible(ctx);
            let can_use_fast_path_local = if bits == BitSize::BYTE {
                // A single byte never crosses a page; read+write permission suffices.
                gen_perm_mask_test(ctx, &perm, 3);
                ctx.builder.tee_new_local()
            }
            else {
                // Span bits: this and the next page are both readable+writable,
                // so no page-boundary check is needed at all.
                gen_perm_mask_test(ctx, &perm, 12);
                let l = ctx.builder.tee_new_local();
                ctx.builder.br_if(cont);
                gen_perm_mask_test(ctx, &perm, 3);
                gen_within_page_test(ctx, address_local, bits.bytes() as i32);
                ctx.builder.and_i32();
                ctx.builder.tee_local(&l);
                l
            };
            ctx.builder.br_if(cont);
            ctx.builder.free_local(perm);
            (operand, true, can_use_fast_path_local, bail)
        }
        else {
            let (operand, common) = gen_memory_guard(ctx, address_local, true, true);
            if bits != BitSize::BYTE {
                gen_within_page_test(ctx, address_local, bits.bytes() as i32);
                ctx.builder.and_i32();
            }
            let can_use_fast_path_local = ctx.builder.tee_new_local();
            ctx.builder.br_if(cont);
            (operand, common, can_use_fast_path_local, false)
        };
    let entry_local = match &operand {
        MemOperand::Resolved(l) => l.unsafe_clone(),
        MemOperand::Guest { .. } => ctx.builder.alloc_local(),
    };

    if common && bail {
        unsafe { crate::jit::GUARD_EXIT_SITES = crate::jit::GUARD_EXIT_SITES.wrapping_add(1) };
        gen_guard_bail(ctx);
    }
    else {
        if common && (unsafe { crate::jit::JIT_GUARD_EXIT }) & 1 != 0 {
            unsafe { crate::jit::GUARD_KEPT_SITES = crate::jit::GUARD_KEPT_SITES.wrapping_add(1) };
        }
        if cfg!(feature = "profiler") {
            ctx.builder.get_local(&address_local);
            ctx.builder.get_local(&entry_local);
            ctx.builder.call_fn2("report_safe_read_write_jit_slow");
        }

        ctx.builder.get_local(&address_local);

        // packed lower bits of eip and wasm table index
        ctx.builder.const_i32(
            ctx.start_of_current_instruction as i32 & 0xFFF
                | (ctx.wasm_table_index.to_u16() as i32) << 16,
        );

        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn2_ret("safe_read_write8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn2_ret("safe_read_write16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn2_ret("safe_read_write32s_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder.call_fn2_ret("safe_read_write64_slow_jit");
            },
            BitSize::DQWORD => {
                dbg_assert!(false);
            },
        }
        ctx.builder.tee_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();

        if cfg!(feature = "profiler") {
            ctx.builder.if_void();
            gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
            ctx.builder.block_end();

            ctx.builder.get_local(&entry_local);
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();
        }

        if gen_vec_has_dirty(ctx) || ctx.fault_precharge.is_some() {
            ctx.builder.if_void();
            gen_vec_flush_for_cold_path(ctx);
            if let Some(k) = ctx.fault_precharge {
                // On a real CPU exception (bit 1 of the slow helper's exit code)
                // undo the pre-charge for the instructions after the faulting one;
                // the interpreter counts the faulting instruction as retired. A
                // bail (bit 1 clear, e.g. SMC) keeps the full pre-charge exactly as
                // before. gen_update_instruction_counter publishes the local.
                if k > 1 {
                    ctx.builder.get_local(&ctx.instruction_counter);
                    ctx.builder.const_i32(k as i32 - 1);
                    ctx.builder.get_local(&entry_local);
                    ctx.builder.const_i32(1);
                    ctx.builder.shr_u_i32();
                    ctx.builder.const_i32(1);
                    ctx.builder.and_i32();
                    ctx.builder.mul_i32();
                    ctx.builder.sub_i32();
                    ctx.builder.set_local(&ctx.instruction_counter);
                }
            }
            ctx.builder.br(ctx.exit_with_fault_label);
            ctx.builder.block_end();
        }
        else {
            ctx.builder.br_if(ctx.exit_with_fault_label);
        }
        if common {
            gen_resolved_memory_address(ctx, &entry_local, address_local, false);
            ctx.builder.set_local(&entry_local);
        }
        if let Some(done) = done {
            gen_rmw_tail(
                ctx,
                bits,
                address_local,
                &entry_local,
                0,
                Some(&can_use_fast_path_local),
                f,
            );
            ctx.builder.br(done);
        }
    }
    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_WRITE_FAST); // XXX: Also slow

    match &operand {
        MemOperand::Guest { base } => {
            gen_rmw_tail(ctx, bits, address_local, address_local, *base, None, f);
            ctx.builder.block_end();

            ctx.builder.free_local(can_use_fast_path_local);
            ctx.builder.free_local(entry_local);
            return;
        },
        MemOperand::Resolved(_) => {},
    }

    gen_resolved_memory_address(ctx, &entry_local, address_local, common);

    ctx.builder.free_local(entry_local);
    let phys_addr_local = ctx.builder.tee_new_local();

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(0);
        },
        BitSize::DQWORD => assert!(false), // not used
    }

    // value is now on stack

    f(ctx);

    // TODO: Could get rid of this local by returning one from f
    let value_local = if bits == BitSize::QWORD {
        GenSafeReadWriteValue::I64(ctx.builder.set_new_local_i64())
    }
    else {
        GenSafeReadWriteValue::I32(ctx.builder.set_new_local())
    };

    ctx.builder.get_local(&can_use_fast_path_local);

    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    {
        ctx.builder.get_local(&address_local);

        match &value_local {
            GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
            GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
        }

        // packed lower bits of eip and wasm table index
        ctx.builder.const_i32(
            ctx.start_of_current_instruction as i32 & 0xFFF
                | (ctx.wasm_table_index.to_u16() as i32) << 16,
        );

        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn3_ret("safe_write8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn3_ret("safe_write16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn3_ret("safe_write32_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder
                    .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
            },
            BitSize::DQWORD => {
                dbg_assert!(false);
            },
        }

        if cfg!(debug_assertions) {
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.if_void();
            {
                // handled above
                ctx.builder.const_i32(match bits {
                    BitSize::BYTE => 8,
                    BitSize::WORD => 16,
                    BitSize::DWORD => 32,
                    BitSize::QWORD => 64,
                    _ => {
                        dbg_assert!(false);
                        0
                    },
                });
                ctx.builder.get_local(&address_local);
                ctx.builder.call_fn2("bug_gen_safe_read_write_page_fault");
            }
            ctx.builder.block_end();
        }
        else {
            ctx.builder.drop_();
        }
    }
    ctx.builder.block_end();

    ctx.builder.get_local(&phys_addr_local);
    match &value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
    }

    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(0);
        },
        BitSize::DQWORD => {
            dbg_assert!(false);
        },
    }

    match value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.free_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.free_local_i64(l),
    }
    ctx.builder.free_local(can_use_fast_path_local);
    ctx.builder.free_local(phys_addr_local);
}

#[cfg(debug_assertions)]
#[no_mangle]
pub fn bug_gen_safe_read_write_page_fault(bits: i32, addr: u32) {
    dbg_log!("bug: gen_safe_read_write_page_fault {} {:x}", bits, addr);
    dbg_assert!(false);
}

pub fn gen_jmp_rel16(builder: &mut WasmBuilder, rel16: u16) {
    let cs_offset_addr = global_pointers::get_seg_offset(regs::CS);
    builder.load_fixed_i32(cs_offset_addr);
    let local = builder.set_new_local();

    // generate:
    // *instruction_pointer = cs_offset + ((*instruction_pointer - cs_offset + rel16) & 0xFFFF);
    {
        builder.const_i32(global_pointers::instruction_pointer as i32);

        gen_get_eip(builder);
        builder.get_local(&local);
        builder.sub_i32();

        builder.const_i32(rel16 as i32);
        builder.add_i32();

        builder.const_i32(0xFFFF);
        builder.and_i32();

        builder.get_local(&local);
        builder.add_i32();

        builder.store_aligned_i32(0);
    }
    builder.free_local(local);
}

pub fn gen_pop16_ss16(ctx: &mut JitContext) {
    // sp = segment_offsets[SS] + reg16[SP] (or just reg16[SP] if has_flat_segmentation)
    gen_get_reg16(ctx, regs::SP);

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    // result = safe_read16(sp)
    let address_local = ctx.builder.set_new_local();
    gen_safe_read16(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg16[SP] += 2;
    gen_get_reg16(ctx, regs::SP);
    ctx.builder.const_i32(2);
    ctx.builder.add_i32();
    gen_set_reg16(ctx, regs::SP);

    // return value is already on stack
}

pub fn gen_pop16_ss32(ctx: &mut JitContext) {
    // esp = segment_offsets[SS] + reg32[ESP] (or just reg32[ESP] if has_flat_segmentation)
    gen_get_reg32(ctx, regs::ESP);

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    // result = safe_read16(esp)
    let address_local = ctx.builder.set_new_local();
    gen_safe_read16(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg32[ESP] += 2;
    gen_get_reg32(ctx, regs::ESP);
    ctx.builder.const_i32(2);
    ctx.builder.add_i32();
    gen_set_reg32(ctx, regs::ESP);

    // return value is already on stack
}

pub fn gen_pop16(ctx: &mut JitContext) {
    if ctx.cpu.ssize_32() {
        gen_pop16_ss32(ctx);
    }
    else {
        gen_pop16_ss16(ctx);
    }
}

pub fn gen_pop32s_ss16(ctx: &mut JitContext) {
    // sp = reg16[SP]
    gen_get_reg16(ctx, regs::SP);

    // result = safe_read32s(segment_offsets[SS] + sp) (or just sp if has_flat_segmentation)
    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    let address_local = ctx.builder.set_new_local();
    gen_safe_read32(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg16[SP] = sp + 4;
    gen_get_reg16(ctx, regs::SP);
    ctx.builder.const_i32(4);
    ctx.builder.add_i32();
    gen_set_reg16(ctx, regs::SP);

    // return value is already on stack
}

pub fn gen_pop32s_ss32(ctx: &mut JitContext) {
    if !ctx.cpu.has_flat_segmentation() {
        gen_get_reg32(ctx, regs::ESP);
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
        let address_local = ctx.builder.set_new_local();
        gen_safe_read32(ctx, &address_local);
        ctx.builder.free_local(address_local);
    }
    else {
        let reg = ctx.register_locals[regs::ESP as usize].unsafe_clone();
        gen_safe_read32(ctx, &reg);
    }

    gen_get_reg32(ctx, regs::ESP);
    ctx.builder.const_i32(4);
    ctx.builder.add_i32();
    gen_set_reg32(ctx, regs::ESP);

    // return value is already on stack
}

pub fn gen_pop32s(ctx: &mut JitContext) {
    if ctx.cpu.ssize_32() {
        gen_pop32s_ss32(ctx);
    }
    else {
        gen_pop32s_ss16(ctx);
    }
}

pub fn gen_adjust_stack_reg(ctx: &mut JitContext, offset: u32) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
        ctx.builder.const_i32(offset as i32);
        ctx.builder.add_i32();
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
        ctx.builder.const_i32(offset as i32);
        ctx.builder.add_i32();
        gen_set_reg16(ctx, regs::SP);
    }
}

pub fn gen_leave(ctx: &mut JitContext, os32: bool) {
    // [e]bp = safe_read{16,32}([e]bp)

    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::EBP);
    }
    else {
        gen_get_reg16(ctx, regs::BP);
    }

    let old_vbp = ctx.builder.tee_new_local();

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }
    if os32 {
        let address_local = ctx.builder.set_new_local();
        gen_safe_read32(ctx, &address_local);
        ctx.builder.free_local(address_local);
        gen_set_reg32(ctx, regs::EBP);
    }
    else {
        let address_local = ctx.builder.set_new_local();
        gen_safe_read16(ctx, &address_local);
        ctx.builder.free_local(address_local);
        gen_set_reg16(ctx, regs::BP);
    }

    // [e]sp = [e]bp + (os32 ? 4 : 2)

    if ctx.cpu.ssize_32() {
        ctx.builder.get_local(&old_vbp);
        ctx.builder.const_i32(if os32 { 4 } else { 2 });
        ctx.builder.add_i32();
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        ctx.builder.get_local(&old_vbp);
        ctx.builder.const_i32(if os32 { 4 } else { 2 });
        ctx.builder.add_i32();
        gen_set_reg16(ctx, regs::SP);
    }

    ctx.builder.free_local(old_vbp);
}

pub fn gen_task_switch_test(ctx: &mut JitContext) {
    // generate if(cr[0] & (CR0_EM | CR0_TS)) { task_switch_test_jit(); goto exit_with_fault; }
    let cr0_offset = global_pointers::get_creg_offset(0);

    dbg_assert!(regs::CR0_EM | regs::CR0_TS <= 0xFF);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
    ctx.builder.and_i32();

    ctx.builder.if_void();
    {
        gen_vec_flush_for_cold_path(ctx);
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        gen_fn1_const(
            ctx.builder,
            "task_switch_test_jit",
            ctx.start_of_current_instruction & 0xFFF,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
}

pub fn gen_task_switch_test_mmx(ctx: &mut JitContext) {
    // Within this basic block CR0 can change only through an imported helper.
    // Even a cold memory/MMIO helper invalidates the previously emitted check.
    if ctx.mmx_task_switch_call_count == Some(ctx.builder.imported_call_count()) {
        return;
    }
    // generate if(cr[0] & (CR0_EM | CR0_TS)) { task_switch_test_mmx_jit(); goto exit_with_fault; }
    let cr0_offset = global_pointers::get_creg_offset(0);

    dbg_assert!(regs::CR0_EM | regs::CR0_TS <= 0xFF);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
    ctx.builder.and_i32();

    ctx.builder.if_void();
    {
        gen_vec_flush_for_cold_path(ctx);
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        gen_fn1_const(
            ctx.builder,
            "task_switch_test_mmx_jit",
            ctx.start_of_current_instruction & 0xFFF,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
    ctx.mmx_task_switch_call_count = Some(ctx.builder.imported_call_count());
}

pub fn gen_push16(ctx: &mut JitContext, value_local: &WasmLocal) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(2);
    ctx.builder.sub_i32();

    let reg_updated_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let reg_updated_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();
        gen_safe_write16(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&reg_updated_local);
        reg_updated_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus two
        let reg_updated_local = ctx.builder.tee_new_local();
        gen_safe_write16(ctx, &reg_updated_local, &value_local);
        reg_updated_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(reg_updated_local);
}

pub fn gen_push32(ctx: &mut JitContext, value_local: &WasmLocal) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();

    let new_sp_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let new_sp_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();

        gen_safe_write32(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&new_sp_local);
        new_sp_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus four
        let new_sp_local = ctx.builder.tee_new_local();
        gen_safe_write32(ctx, &new_sp_local, &value_local);
        new_sp_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(new_sp_local);
}

pub fn gen_push32_sreg(ctx: &mut JitContext, reg: u32) {
    gen_get_sreg(ctx, reg);
    let value_local = ctx.builder.set_new_local();

    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();

    let new_sp_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let new_sp_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();

        gen_safe_write16(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&new_sp_local);
        new_sp_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus four
        let new_sp_local = ctx.builder.tee_new_local();
        gen_safe_write16(ctx, &new_sp_local, &value_local);
        new_sp_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(new_sp_local);
    ctx.builder.free_local(value_local);
}

pub fn gen_get_real_eip(ctx: &mut JitContext) {
    gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(ctx.cpu.eip as i32 & 0xFFF);
    ctx.builder.or_i32();
    if !ctx.cpu.has_flat_segmentation() {
        ctx.builder
            .load_fixed_i32(global_pointers::get_seg_offset(regs::CS));
        ctx.builder.sub_i32();
    }
}

pub fn gen_set_last_op1(builder: &mut WasmBuilder, source: &WasmLocal) {
    builder.const_i32(global_pointers::last_op1 as i32);
    builder.get_local(&source);
    builder.store_aligned_i32(0);
}

pub fn gen_set_last_result(builder: &mut WasmBuilder, source: &WasmLocal) {
    builder.const_i32(global_pointers::last_result as i32);
    builder.get_local(&source);
    builder.store_aligned_i32(0);
}

pub fn gen_clear_flags_changed_bits(builder: &mut WasmBuilder, bits_to_clear: i32) {
    builder.const_i32(global_pointers::flags_changed as i32);
    gen_get_flags_changed(builder);
    builder.const_i32(!bits_to_clear);
    builder.and_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_last_op_size_and_flags_changed(
    builder: &mut WasmBuilder,
    last_op_size: i32,
    flags_changed: i32,
) {
    dbg_assert!(last_op_size == OPSIZE_8 || last_op_size == OPSIZE_16 || last_op_size == OPSIZE_32);
    dbg_assert!(global_pointers::last_op_size as i32 % 8 == 0);
    dbg_assert!(global_pointers::last_op_size as i32 + 4 == global_pointers::flags_changed as i32);
    builder.const_i32(global_pointers::last_op_size as i32);
    builder.const_i64(last_op_size as u32 as i64 | (flags_changed as u32 as i64) << 32);
    builder.store_aligned_i64(0);
}

pub fn gen_set_flags_bits(builder: &mut WasmBuilder, bits_to_set: i32) {
    builder.const_i32(global_pointers::flags as i32);
    gen_get_flags(builder);
    builder.const_i32(bits_to_set);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_clear_flags_bits(builder: &mut WasmBuilder, bits_to_clear: i32) {
    builder.const_i32(global_pointers::flags as i32);
    gen_get_flags(builder);
    builder.const_i32(!bits_to_clear);
    builder.and_i32();
    builder.store_aligned_i32(0);
}

#[derive(PartialEq)]
pub enum ConditionNegate {
    True,
    False,
}

pub fn gen_getzf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::WasmLocal(source),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            ctx.builder.get_local(source);
            if negate == ConditionNegate::False {
                ctx.builder.eq_i32();
            }
            else {
                ctx.builder.ne_i32();
            }
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::Immediate(0),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::Immediate(i),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            ctx.builder.const_i32(*i);
            if negate == ConditionNegate::False {
                ctx.builder.eq_i32();
            }
            else {
                ctx.builder.ne_i32();
            }
        },
        Instruction::Cmp { .. }
        | Instruction::Sub { .. }
        | Instruction::Add { .. }
        | Instruction::AdcSbb { .. }
        | Instruction::NonZeroShift { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Bitwise { opsize, .. } | Instruction::Test { opsize, .. } => {
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // Note: Necessary because test{8,16} don't mask either last_result or any of their operands
            // TODO: Use local instead of last_result for 8-bit/16-bit
            if opsize == OPSIZE_32 {
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            }
            else if opsize == OPSIZE_16 {
                ctx.builder
                    .load_fixed_u16(global_pointers::last_result as u32);
            }
            else if opsize == OPSIZE_8 {
                ctx.builder
                    .load_fixed_u8(global_pointers::last_result as u32);
            }
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other | Instruction::Preserve { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            ctx.builder.const_i32(FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.if_i32();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            let last_result = ctx.builder.tee_new_local();
            ctx.builder.const_i32(-1);
            ctx.builder.xor_i32();
            ctx.builder.get_local(&last_result);
            ctx.builder.free_local(last_result);
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
            ctx.builder.and_i32();
            gen_get_last_op_size(ctx.builder);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.else_();
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.block_end();

            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getcf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp { source, opsize, .. }
        | Instruction::Sub {
            source,
            opsize,
            is_dec: false,
            ..
        } => {
            // Note: x < y and x < x - y can be used interchangeably (see getcf)
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => gen_get_last_result(ctx.builder, &ctx.previous_instruction),
            }
            if negate == ConditionNegate::True {
                ctx.builder.geu_i32();
            }
            else {
                ctx.builder.ltu_i32();
            }
        },
        Instruction::Add {
            source,
            opsize,
            is_inc: false,
            ..
        } => {
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => gen_get_last_op1(ctx.builder, &ctx.previous_instruction),
            }
            if negate == ConditionNegate::True {
                ctx.builder.geu_i32();
            }
            else {
                ctx.builder.ltu_i32();
            }
        },
        Instruction::Add { is_inc: true, .. } | Instruction::Sub { is_dec: true, .. } => {
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Bitwise { .. } | Instruction::Test { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder
                .const_i32(if negate == ConditionNegate::True { 1 } else { 0 });
        },
        Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other | Instruction::Preserve { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);

            gen_get_flags_changed(ctx.builder);
            let flags_changed = ctx.builder.tee_new_local();
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            ctx.builder.if_i32();

            ctx.builder.get_local(&flags_changed);
            ctx.builder.const_i32(31);
            ctx.builder.shr_s_i32();
            ctx.builder.free_local(flags_changed);
            let sub_mask = ctx.builder.set_new_local();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.get_local(&sub_mask);
            ctx.builder.xor_i32();

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            ctx.builder.get_local(&sub_mask);
            ctx.builder.xor_i32();

            ctx.builder.ltu_i32();

            ctx.builder.else_();
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            ctx.builder.block_end();

            ctx.builder.free_local(sub_mask);

            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getsf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp { opsize, .. }
        | Instruction::Sub { opsize, .. }
        | Instruction::Add { opsize, .. }
        | Instruction::AdcSbb { opsize, .. }
        | Instruction::Bitwise { opsize, .. }
        | Instruction::Test { opsize, .. }
        | Instruction::NonZeroShift { opsize, .. } => {
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            if opsize == OPSIZE_32 {
                ctx.builder.const_i32(0);
                if negate == ConditionNegate::True {
                    ctx.builder.ge_i32();
                }
                else {
                    ctx.builder.lt_i32();
                }
            }
            else {
                // TODO: use register (see get_last_result)
                ctx.builder
                    .const_i32(if opsize == OPSIZE_16 { 0x8000 } else { 0x80 });
                ctx.builder.and_i32();
                if negate == ConditionNegate::True {
                    ctx.builder.eqz_i32();
                }
            }
        },
        Instruction::Other | Instruction::Preserve { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            ctx.builder.const_i32(FLAG_SIGN);
            ctx.builder.and_i32();
            ctx.builder.if_i32();
            {
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                gen_get_last_op_size(ctx.builder);
                ctx.builder.shr_u_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();
            }
            ctx.builder.else_();
            {
                gen_get_flags(ctx.builder);
                ctx.builder.const_i32(FLAG_SIGN);
                ctx.builder.and_i32();
            }
            ctx.builder.block_end();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getof(ctx: &mut JitContext) {
    match &ctx.previous_instruction {
        Instruction::Cmp { opsize, .. } | Instruction::Sub { opsize, .. } => {
            // TODO: a better formula might be possible
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.xor_i32();

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.sub_i32();
            ctx.builder.xor_i32();
            ctx.builder.and_i32();

            ctx.builder.const_i32(if opsize == OPSIZE_32 {
                0x8000_0000u32 as i32
            }
            else if opsize == OPSIZE_16 {
                0x8000
            }
            else {
                0x80
            });
            ctx.builder.and_i32();
        },
        Instruction::Add { opsize, .. } => {
            // TODO: a better formula might be possible
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.xor_i32();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            ctx.builder.sub_i32();
            ctx.builder.xor_i32();
            ctx.builder.and_i32();

            ctx.builder.const_i32(if opsize == OPSIZE_32 {
                0x8000_0000u32 as i32
            }
            else if opsize == OPSIZE_16 {
                0x8000
            }
            else {
                0x80
            });
            ctx.builder.and_i32();
        },
        Instruction::Bitwise { .. } | Instruction::Test { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.const_i32(0);
        },
        Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_OVERFLOW);
            ctx.builder.and_i32();
        },
        Instruction::Other | Instruction::Preserve { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            let flags_changed = ctx.builder.tee_new_local();
            ctx.builder.const_i32(FLAG_OVERFLOW);
            ctx.builder.and_i32();
            ctx.builder.if_i32();
            {
                gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                let last_op1 = ctx.builder.tee_new_local();
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                let last_result = ctx.builder.tee_new_local();
                ctx.builder.xor_i32();

                ctx.builder.get_local(&last_result);
                ctx.builder.get_local(&last_op1);
                ctx.builder.sub_i32();
                gen_get_flags_changed(ctx.builder);
                ctx.builder.const_i32(31);
                ctx.builder.shr_u_i32();
                ctx.builder.sub_i32();

                ctx.builder.get_local(&last_result);
                ctx.builder.xor_i32();

                ctx.builder.and_i32();

                gen_get_last_op_size(ctx.builder);
                ctx.builder.shr_u_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();

                ctx.builder.free_local(last_op1);
                ctx.builder.free_local(last_result);
            }
            ctx.builder.else_();
            {
                gen_get_flags(ctx.builder);
                ctx.builder.const_i32(FLAG_OVERFLOW);
                ctx.builder.and_i32();
            }
            ctx.builder.block_end();
            ctx.builder.free_local(flags_changed);
        },
    }
}

pub fn gen_test_be(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    dbg_assert!(*opsize != OPSIZE_8 || i >= 0 && i < 0x100);
                    dbg_assert!(*opsize != OPSIZE_16 || i >= 0 && i < 0x10000);
                    ctx.builder.const_i32(i);
                },
            }

            if negate == ConditionNegate::True {
                ctx.builder.gtu_i32();
            }
            else {
                ctx.builder.leu_i32();
            }
        },
        Instruction::Sub {
            opsize,
            source,
            is_dec: false,
            ..
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
            }

            if negate == ConditionNegate::True {
                ctx.builder.gtu_i32();
            }
            else {
                ctx.builder.leu_i32();
            }
        },
        &Instruction::Bitwise { .. } | &Instruction::Test { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_getzf(ctx, negate);
        },
        &Instruction::Add { .. } | &Instruction::Sub { is_dec: true, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // not the best code generation, but reasonable for this fairly uncommon case
            gen_getcf(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other
        | Instruction::Preserve { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_getcf(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_l(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    ctx.builder.const_i32(i);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.ge_i32();
            }
            else {
                ctx.builder.lt_i32();
            }
        },
        Instruction::Sub { opsize, source, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                ctx.builder
                    .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                ctx.builder.shl_i32();
            }
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(
                    i << if *opsize == OPSIZE_32 {
                        0
                    }
                    else if *opsize == OPSIZE_16 {
                        16
                    }
                    else {
                        24
                    },
                ),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.ge_i32();
            }
            else {
                ctx.builder.lt_i32();
            }
        },
        &Instruction::Bitwise { .. } | &Instruction::Test { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_getsf(ctx, negate);
        },
        &Instruction::Other
        | Instruction::Preserve { .. }
        | Instruction::Add { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            if let Instruction::Add { .. } = ctx.previous_instruction {
                gen_profiler_stat_increment(
                    ctx.builder,
                    profiler::stat::CONDITION_UNOPTIMISED_UNHANDLED_L,
                );
            }
            gen_getsf(ctx, ConditionNegate::False);
            ctx.builder.eqz_i32();
            gen_getof(ctx);
            ctx.builder.eqz_i32();
            ctx.builder.xor_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_le(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    ctx.builder.const_i32(i);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.gt_i32();
            }
            else {
                ctx.builder.le_i32();
            }
        },
        Instruction::Sub { opsize, source, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                ctx.builder
                    .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                ctx.builder.shl_i32();
            }
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(
                    i << if *opsize == OPSIZE_32 {
                        0
                    }
                    else if *opsize == OPSIZE_16 {
                        16
                    }
                    else {
                        24
                    },
                ),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.gt_i32();
            }
            else {
                ctx.builder.le_i32();
            }
        },
        &Instruction::Bitwise { .. } | &Instruction::Test { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // TODO: Could probably be improved (<= 0)
            gen_test_l(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other
        | Instruction::Preserve { .. }
        | Instruction::Add { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            if let Instruction::Add { .. } = ctx.previous_instruction {
                gen_profiler_stat_increment(
                    ctx.builder,
                    profiler::stat::CONDITION_UNOPTIMISED_UNHANDLED_LE,
                );
            }
            gen_test_l(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_loopnz(ctx: &mut JitContext, is_asize_32: bool) {
    gen_test_loop(ctx, is_asize_32);
    ctx.builder.eqz_i32();
    gen_getzf(ctx, ConditionNegate::False);
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
}
pub fn gen_test_loopz(ctx: &mut JitContext, is_asize_32: bool) {
    gen_test_loop(ctx, is_asize_32);
    ctx.builder.eqz_i32();
    gen_getzf(ctx, ConditionNegate::False);
    ctx.builder.eqz_i32();
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
}
pub fn gen_test_loop(ctx: &mut JitContext, is_asize_32: bool) {
    if is_asize_32 {
        gen_get_reg32(ctx, regs::ECX);
    }
    else {
        gen_get_reg16(ctx, regs::CX);
    }
}
pub fn gen_test_jcxz(ctx: &mut JitContext, is_asize_32: bool) {
    if is_asize_32 {
        gen_get_reg32(ctx, regs::ECX);
    }
    else {
        gen_get_reg16(ctx, regs::CX);
    }
    ctx.builder.eqz_i32();
}

pub fn gen_fpu_get_sti(ctx: &mut JitContext, i: u32) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    ctx.builder.const_i32(i as i32);
    ctx.builder.call_fn2("fpu_get_sti_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_fpu_load_m64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read64(ctx, modrm_byte);
    ctx.builder.call_fn2_i32_i64("f64_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_fpu_load_i16(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read16(ctx, modrm_byte);
    sign_extend_i16(ctx.builder);
    ctx.builder.call_fn2("i32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}
pub fn gen_fpu_load_i32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read32(ctx, modrm_byte);
    ctx.builder.call_fn2("i32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}
pub fn gen_fpu_load_i64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read64(ctx, modrm_byte);
    ctx.builder.call_fn2_i32_i64("i64_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_trigger_de(ctx: &mut JitContext) {
    gen_vec_flush_for_cold_path(ctx);
    gen_fn1_const(
        ctx.builder,
        "trigger_de_jit",
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_trigger_ud(ctx: &mut JitContext) {
    gen_vec_flush_for_cold_path(ctx);
    gen_fn1_const(
        ctx.builder,
        "trigger_ud_jit",
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_trigger_gp(ctx: &mut JitContext, error_code: u32) {
    gen_vec_flush_for_cold_path(ctx);
    gen_fn2_const(
        ctx.builder,
        "trigger_gp_jit",
        error_code,
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_condition_fn_negated(ctx: &mut JitContext, condition: u8) {
    gen_condition_fn(ctx, condition ^ 1)
}

pub fn gen_condition_fn(ctx: &mut JitContext, condition: u8) {
    if condition & 0xF0 == 0x00 || condition & 0xF0 == 0x70 || condition & 0xF0 == 0x80 {
        match condition & 0xF {
            0x0 => {
                gen_getof(ctx);
            },
            0x1 => {
                gen_getof(ctx);
                ctx.builder.eqz_i32();
            },
            0x2 => {
                gen_getcf(ctx, ConditionNegate::False);
            },
            0x3 => {
                gen_getcf(ctx, ConditionNegate::True);
            },
            0x4 => {
                gen_getzf(ctx, ConditionNegate::False);
            },
            0x5 => {
                gen_getzf(ctx, ConditionNegate::True);
            },
            0x6 => {
                gen_test_be(ctx, ConditionNegate::False);
            },
            0x7 => {
                gen_test_be(ctx, ConditionNegate::True);
            },
            0x8 => {
                gen_getsf(ctx, ConditionNegate::False);
            },
            0x9 => {
                gen_getsf(ctx, ConditionNegate::True);
            },
            0xA => {
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED_PF);
                ctx.builder.call_fn0_ret("test_p");
            },
            0xB => {
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED_PF);
                ctx.builder.call_fn0_ret("test_np");
            },
            0xC => {
                gen_test_l(ctx, ConditionNegate::False);
            },
            0xD => {
                gen_test_l(ctx, ConditionNegate::True);
            },
            0xE => {
                gen_test_le(ctx, ConditionNegate::False);
            },
            0xF => {
                gen_test_le(ctx, ConditionNegate::True);
            },
            _ => {
                dbg_assert!(false);
            },
        }
    }
    else {
        // loop, loopnz, loopz, jcxz
        dbg_assert!(condition & !0x3 == 0xE0);
        if condition == 0xE0 {
            gen_test_loopnz(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE1 {
            gen_test_loopz(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE2 {
            gen_test_loop(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE3 {
            gen_test_jcxz(ctx, ctx.cpu.asize_32());
        }
    }
}

pub fn gen_move_registers_from_locals_to_memory(ctx: &mut JitContext) {
    if cfg!(feature = "profiler") {
        let instruction = memory::read32s(ctx.start_of_current_instruction) as u32;
        opstats::gen_opstat_unguarded_register(ctx.builder, instruction);
    }

    for i in 0..8 {
        ctx.builder
            .const_i32(global_pointers::get_reg32_offset(i as u32) as i32);
        ctx.builder.get_local(&ctx.register_locals[i]);
        ctx.builder.store_aligned_i32(0);
    }
}
pub fn gen_move_registers_from_memory_to_locals(ctx: &mut JitContext) {
    if cfg!(feature = "profiler") {
        let instruction = memory::read32s(ctx.start_of_current_instruction) as u32;
        opstats::gen_opstat_unguarded_register(ctx.builder, instruction);
    }

    for i in 0..8 {
        ctx.builder
            .const_i32(global_pointers::get_reg32_offset(i as u32) as i32);
        ctx.builder.load_aligned_i32(0);
        ctx.builder.set_local(&ctx.register_locals[i]);
    }
}

pub fn gen_profiler_stat_increment(builder: &mut WasmBuilder, stat: profiler::stat) {
    if !cfg!(feature = "profiler") {
        return;
    }
    let addr = unsafe { &raw mut profiler::stat_array[stat as usize] } as u32;
    builder.increment_fixed_i64(addr, 1)
}

pub fn gen_debug_track_jit_exit(builder: &mut WasmBuilder, address: u32) {
    if cfg!(feature = "profiler") {
        gen_fn1_const(builder, "track_jit_exit", address);
    }
}
