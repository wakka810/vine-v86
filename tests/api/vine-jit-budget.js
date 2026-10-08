#!/usr/bin/env node

// Run make vine-jit-budget-test, or use an existing build with:
// node tests/api/vine-jit-budget.js
// V86_WASM can select another build. This is a correctness test, not a benchmark:
// its Wasm entry wrapper and getter instrumentation deliberately add host calls.
// V86_BASELINE_WASM additionally compares all CPU state words, guest memory,
// dirty-page reports and retired counters with an unchanged build.
import assert from "node:assert/strict";
import fs from "node:fs";

const wasm_path = process.env.V86_WASM || new URL("../../build/v86-fallback.wasm", import.meta.url);
const module = new WebAssembly.Module(fs.readFileSync(wasm_path));
const baseline_module = process.env.V86_BASELINE_WASM &&
    new WebAssembly.Module(fs.readFileSync(process.env.V86_BASELINE_WASM));
const START = 0x1000;
const DATA = 0x8000;
const MEMORY_SIZE = 1024 * 1024;
const TABLE_OFFSET = 1024;
const VIRTUAL_BASE = 0x400000;
const PAGE_DIRECTORY = 0x10000;
const PAGE_TABLE = 0x11000;

// (module
//   (import "e" "f" (func $f (param i32)))
//   (import "e" "enter" (func $enter))
//   (func (export "f") (param i32) call $enter local.get 0 call $f))
const entry_wrapper = new WebAssembly.Module(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0,
    1, 8, 2, 96, 1, 127, 0, 96, 0, 0,
    2, 17, 2, 1, 101, 1, 102, 0, 0, 1, 101, 5, 101, 110, 116, 101, 114, 0, 1,
    3, 2, 1, 0,
    7, 5, 1, 1, 102, 0, 2,
    10, 10, 1, 8, 0, 16, 1, 32, 0, 16, 0, 11,
]));

function create_cpu(code, {
    disable_jit = false, exact = true, cpu_module = module, paging = false, asynchronous = false,
    on_finalize = null, on_mmap = null,
} = {})
{
    const table = new WebAssembly.Table({ element: "anyfunc", initial: TABLE_OFFSET + 900 });
    let wasm;
    let jit_imports;
    let last_entry;
    let last_limit;
    const counts = { entries: 0, reads: 0, modules: 0, invalidations: 0,
        jit_slow_writes: 0, jit_slow_reads: 0, mmx_guards: 0, mmap: 0, fpu_imports: new Set() };
    const virtual_base = paging ? VIRTUAL_BASE : 0;
    const env = { __indirect_function_table: table };
    for(const imported of WebAssembly.Module.imports(cpu_module))
    {
        assert.equal(imported.module, "env");
        if(imported.kind === "table") continue;
        assert.equal(imported.kind, "function");
        env[imported.name] = (...args) => {
            switch(imported.name)
            {
                case "microtick":
                case "get_rand_int":
                case "cpu_event_halt":
                case "stop_idling":
                case "run_hardware_timers":
                    return 0;
                case "jit_clear_func":
                    table.set(TABLE_OFFSET + args[0], null);
                    counts.invalidations++;
                    return;
                case "console_log_from_wasm":
                    throw new Error(new TextDecoder().decode(
                        new Uint8Array(wasm.memory.buffer, args[0], args[1])));
                case "mmap_read8":
                case "mmap_read32":
                case "mmap_write8":
                case "mmap_write16":
                case "mmap_write32":
                    assert(on_mmap, "MMIO requires an explicit test adapter");
                    counts.mmap++;
                    return on_mmap(imported.name, args, wasm);
                case "codegen_finalize": {
                    if(on_finalize) on_finalize(wasm);
                    const [index, start, state_flags, pointer, length] = args;
                    const bytes = Buffer.from(wasm.memory.buffer, pointer, length);
                    // Count the generated CR0 byte-load/mask/if sequence. The
                    // focused fixture also witnesses its compiled entry and
                    // compares architectural state; this count proves elision.
                    const guard = Buffer.from([0x41, 0xc4, 0x04, 0x2d, 0, 0, 0x41, 12, 0x71, 4, 0x40]);
                    counts.mmx_guards = 0;
                    for(let index = bytes.indexOf(guard); index >= 0; index = bytes.indexOf(guard, index + 1))
                        counts.mmx_guards++;
                    const generated = new WebAssembly.Module(bytes);
                    for(const dependency of WebAssembly.Module.imports(generated))
                        if(/^(fpu_|f32_to_f80)/.test(dependency.name)) counts.fpu_imports.add(dependency.name);
                    const f = new WebAssembly.Instance(generated, { e: jit_imports }).exports.f;
                    const wrapped = new WebAssembly.Instance(entry_wrapper, {
                        e: { f, enter: () => counts.entries++ },
                    }).exports.f;
                    table.set(TABLE_OFFSET + index, wrapped);
                    last_entry = wrapped;
                    if(!exact) assert(!WebAssembly.Module.imports(generated).some(
                        dependency => dependency.name === "vine_jit_instruction_limit"));
                    counts.modules++;
                    if(asynchronous) globalThis.queueMicrotask(() =>
                        wasm.codegen_finalize_finished(index, start, state_flags));
                    return;
                }
                default:
                    throw new Error(`Unexpected host import ${imported.name}`);
            }
        };
    }
    wasm = new WebAssembly.Instance(cpu_module, { env }).exports;
    jit_imports = { ...wasm, m: wasm.memory };
    jit_imports.vine_jit_instruction_limit = () => {
        counts.reads++;
        last_limit = wasm.vine_jit_instruction_limit() >>> 0;
        return last_limit;
    };
    jit_imports.safe_write8_slow_jit = (...args) => {
        counts.jit_slow_writes++;
        return wasm.safe_write8_slow_jit(...args);
    };
    jit_imports.safe_read32s_slow_jit = (...args) => {
        counts.jit_slow_reads++;
        return wasm.safe_read32s_slow_jit(...args);
    };
    wasm.rust_init();
    assert.equal(wasm.vine_set_memory_size(MEMORY_SIZE), 0);
    const physical_base = wasm.allocate_memory(MEMORY_SIZE);
    wasm.reset_cpu();
    wasm.set_jit_config(0, Number(disable_jit));
    wasm.set_jit_config(4, Number(exact));
    wasm.set_jit_config(5, 1);
    wasm.set_jit_config(6, Number(!asynchronous));
    new Uint8Array(wasm.memory.buffer, physical_base + START, code.length).set(code);

    wasm.vine_cpu_state_save();
    const state = new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153);
    state[3] = virtual_base + START; // ecx: indirect loop target
    state[6] = 0x90000; // esp
    state[8] = 3; // esi: divisor
    state[9] = virtual_base + DATA; // edi: writable data
    state[10] = state[11] = virtual_base + START;
    state[19] = state[20] = state[21] = 1; // protected mode, 32-bit code/stack
    state.fill(0, 34, 42); // flat segments
    state.fill(0xFFFFFFFF, 42, 50); // segment limits
    state[58] = 1; // CR0.PE, without paging
    if(paging)
    {
        const physical = new DataView(wasm.memory.buffer, physical_base, MEMORY_SIZE);
        physical.setUint32(PAGE_DIRECTORY + (VIRTUAL_BASE >>> 22) * 4, PAGE_TABLE | 7, true);
        for(let page = 0; page < MEMORY_SIZE >>> 12; page++)
        {
            physical.setUint32(PAGE_TABLE + page * 4, page << 12 | 7, true);
        }
        state[18] = 3; // user-mode paging permissions
        state[58] = 0x80010001; // CR0.PG | WP | PE
        state[61] = PAGE_DIRECTORY; // CR3
    }
    assert.equal(wasm.vine_cpu_state_restore(), 0);

    function retired(prefix)
    {
        return BigInt(wasm[`${prefix}_high`]() >>> 0) << 32n |
            BigInt(wasm[`${prefix}_low`]() >>> 0);
    }

    return {
        wasm, counts,
        get physical() { return new DataView(wasm.memory.buffer, physical_base, MEMORY_SIZE); },
        resume(eax = null, address = START, esp = virtual_base + 0x90000) {
            if(typeof wasm.vine_cpu_apply_stopped_resume === "function")
            {
                assert.equal(wasm.vine_cpu_apply_stopped_resume(
                    eax ?? 0, Number(eax !== null), esp, virtual_base + address), 0);
            }
            else
            {
                wasm.vine_cpu_state_save();
                const saved = new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153);
                if(eax !== null) saved[2] = eax;
                saved[6] = esp;
                saved[10] = saved[11] = virtual_base + address;
                saved[22] = 0;
                assert.equal(wasm.vine_cpu_state_restore(), 0);
            }
        },
        clear_dirty(addresses) {
            assert(paging);
            for(const address of addresses)
            {
                const pte = PAGE_TABLE + (address >>> 12) * 4;
                this.physical.setUint32(pte, this.physical.getUint32(pte, true) & ~0x40, true);
            }
            wasm.vine_reset_dirty_pages();
            if(typeof wasm.vine_rearm_dirty_page === "function")
            {
                for(const address of addresses)
                    assert.equal(wasm.vine_rearm_dirty_page(virtual_base + address), 0);
            }
            else wasm.full_clear_tlb();
        },
        restart_at_entry() {
            this.restart_at(START);
        },
        restart_at(address, clear_tlb = true) {
            new DataView(wasm.memory.buffer).setUint32(556, virtual_base + address, true);
            if(clear_tlb) wasm.full_clear_tlb();
        },
        protect_data_page() {
            assert(paging);
            const physical = new DataView(wasm.memory.buffer, physical_base, MEMORY_SIZE);
            const pte = PAGE_TABLE + (DATA >>> 12) * 4;
            physical.setUint32(pte, physical.getUint32(pte, true) & ~2, true);
            wasm.full_clear_tlb();
        },
        set_register(index, value) {
            new DataView(wasm.memory.buffer).setUint32(64 + index * 4, value, true);
        },
        set_counter(value) {
            new DataView(wasm.memory.buffer).setUint32(664, value, true);
        },
        execute(budget) {
            const before = wasm.vine_instruction_counter();
            const jit_before = retired("vine_jit_retired_instructions");
            const interpreted_before = retired("vine_interpreted_retired_instructions");
            const reason = wasm.vine_execute_budget(budget);
            const instructions = (wasm.vine_instruction_counter() - before) >>> 0;
            assert.equal(retired("vine_jit_retired_instructions") - jit_before +
                retired("vine_interpreted_retired_instructions") - interpreted_before,
                BigInt(instructions), "retired accounting must match the slice");
            assert.equal(wasm.vine_jit_instruction_limit() >>> 0, 0xFFFFFFFF,
                "the limit is unbounded outside Vine execution");
            wasm.vine_cpu_state_save();
            const saved = new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153);
            return {
                reason,
                error: wasm.vine_get_stop_error_code(),
                instructions,
                registers: [...saved.slice(2, 10)],
                eip: saved[10],
                flags: wasm.get_eflags() >>> 0,
                data: [...new Uint8Array(wasm.memory.buffer, physical_base + DATA, 16)],
            };
        },
        run_unbounded_entry() {
            assert(last_entry);
            last_entry(0);
            assert.equal(last_limit, 0xFFFFFFFF);
        },
        full_snapshot() {
            wasm.vine_cpu_state_save();
            return {
                state: [...new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153)],
                memory: Buffer.from(new Uint8Array(wasm.memory.buffer, physical_base, MEMORY_SIZE)),
                dirty: [...new Uint32Array(wasm.memory.buffer, wasm.vine_dirty_pages_ptr(),
                    wasm.vine_dirty_page_count())],
                dirty_overflow: wasm.vine_dirty_pages_overflowed(),
                counter: wasm.vine_instruction_counter(),
                jit: retired("vine_jit_retired_instructions"),
                interpreted: retired("vine_interpreted_retired_instructions"),
            };
        },
        assert_hoisted() {
            assert(counts.entries > 0, "test must execute compiled code");
            assert.equal(counts.reads, exact ? counts.entries : 0,
                "read the budget once per JIT invocation, and never with checks disabled");
        },
    };
}

function create_pair(code, options = {}, {
    compare_retired_partition = true, allow_budget_previous_eip = false,
} = {})
{
    const jit = create_cpu(code, options);
    const interpreted = create_cpu(code, { ...options, disable_jit: true });
    const baseline = baseline_module && create_cpu(code, { ...options, cpu_module: baseline_module });
    return {
        jit,
        baseline,
        cpus: [jit, interpreted, ...(baseline ? [baseline] : [])],
        execute(budget, description, { compare_instructions = true } = {}) {
            const actual = jit.execute(budget);
            if(baseline)
            {
                assert.deepEqual(actual, baseline.execute(budget), `${description}: baseline result`);
                const actual_snapshot = jit.full_snapshot();
                const expected_snapshot = baseline.full_snapshot();
                if(compare_retired_partition)
                {
                    assert.deepEqual(actual_snapshot, expected_snapshot,
                        `${description}: complete CPU state, guest memory and retirement counters`);
                }
                else
                {
                    const { jit: actual_jit, interpreted: actual_interpreted, ...actual_state } = actual_snapshot;
                    const { jit: expected_jit, interpreted: expected_interpreted, ...expected_state } = expected_snapshot;
                    if(allow_budget_previous_eip && actual.reason === 1)
                    {
                        // Cached split blocks need not publish the previous IP
                        // at a successful budget exit. Fault/HLT/SMC exits keep
                        // every raw state word strict, including PREVIOUS_EIP.
                        actual_state.state = actual_state.state.filter((_, index) => index !== 11);
                        expected_state.state = expected_state.state.filter((_, index) => index !== 11);
                    }
                    assert.deepEqual(actual_state, expected_state,
                        `${description}: all CPU state words, memory, dirty pages and counter`);
                    assert.equal(actual_jit + actual_interpreted, expected_jit + expected_interpreted,
                        `${description}: total retired instructions`);
                }
            }
            const expected = interpreted.execute(budget);
            if(compare_instructions)
            {
                assert.deepEqual(actual, expected, `${description}: interpreter oracle`);
            }
            else
            {
                // Existing JIT fault/SMC exits charge the remainder of their basic
                // block. At a fixed exit, compare architectural state to the
                // interpreter and preserve exact baseline accounting above.
                const { instructions: actual_count, ...actual_state } = actual;
                const { instructions: expected_count, ...expected_state } = expected;
                assert.deepEqual(actual_state, expected_state, `${description}: interpreter state`);
                assert.deepEqual(jit.full_snapshot().memory, interpreted.full_snapshot().memory,
                    `${description}: interpreter memory`);
                console.log(`${description}: JIT=${actual_count}, interpreter=${expected_count} instructions`);
            }
            return actual;
        },
    };
}

const budgets = [131072, 0, 1, 4095, 4096, 4097, 8193, 131072, 65537];
const loops = [
    ["direct", [0x40, 0xEB, 0xFD]], // inc eax; jmp start
    ["indirect-memory", [0x40, 0xFF, 0x07, 0xFF, 0xE1]], // inc eax; inc [edi]; jmp ecx
    ["indirect-helper", [0x40, 0x99, 0xF7, 0xFE, 0x43, 0xFF, 0xE1]], // inc; cdq; idiv; inc; jmp
    ["multiple-entries", [0x40, 0xA8, 1, 0x74, 3, 0x43, 0xFF, 0xE1, 0x4B, 0xFF, 0xE1]],
];

function assert_budget_architecture(jit, interpreted, description)
{
    const { state, jit: compiled, interpreted: fallback, ...actual } = jit.full_snapshot();
    const { state: expected_state, jit: expected_compiled,
        interpreted: expected_fallback, ...expected } = interpreted.full_snapshot();
    assert.deepEqual(actual, expected, `${description}: memory, dirty pages and counter`);
    assert.deepEqual(state.filter((_, index) => index !== 11),
        expected_state.filter((_, index) => index !== 11),
        `${description}: all state words except successful-budget previous IP`);
    assert.equal(jit.wasm.get_eflags() >>> 0, interpreted.wasm.get_eflags() >>> 0,
        `${description}: resolved EFLAGS`);
    assert.equal(compiled + fallback, expected_compiled + expected_fallback,
        `${description}: total retirement`);
}

for(const [name, code] of loops)
{
    const { jit, cpus, execute } = create_pair(code, {}, {
        compare_retired_partition: false, allow_budget_previous_eip: true,
    });
    for(const budget of budgets)
    {
        const actual = execute(budget, `${name}: budget ${budget}`);
        assert.equal(actual.reason, 1);
        assert.equal(actual.instructions, budget);
    }
    for(const cpu of cpus) cpu.set_counter(0xFFFFF000);
    execute(131072, `${name}: counter wrap`);

    // Replace INC EAX with DEC EAX through the CPU write path: invalidate and recompile.
    for(const cpu of cpus) cpu.wasm.write8(START, 0x48);
    execute(131072, `${name}: code invalidation`);
    jit.assert_hoisted();
    console.log(`${name}: exact budgets, wrap, invalidation, getter count passed`);
}

// An aligned warmup leaves a single 258-instruction block: no interior entry
// learned from a partial tail. A budget of one must reject it once, execute one
// NOP interpreted and terminate. Cold TLB discovery must not redispatch forever.
for(const [name, options, cold] of [
    ["large block hot tail", {}, false],
    ["large block paged deferred cold tail", { paging: true, asynchronous: true }, true],
])
{
    const code = new Uint8Array(262).fill(0x90);
    code[256] = 0x40;
    code[257] = 0xE9;
    new DataView(code.buffer).setInt32(258, -262, true);
    const { jit, cpus, execute } = create_pair(code, options, {
        compare_retired_partition: false, allow_budget_previous_eip: true,
    });
    execute(258 * 400, `${name}: aligned warmup`);
    await Promise.resolve();
    assert.equal(jit.counts.modules, 1, "warmup must leave one unsplit cached block");
    for(const cpu of cpus) cpu.restart_at(START, cold);
    const before = jit.full_snapshot();
    const entries = jit.counts.entries;
    const tiny = execute(1, `${name}: zero-retired rejection`);
    assert.equal(tiny.reason, 1);
    assert.equal(tiny.instructions, 1);
    assert.equal(tiny.eip, (options.paging ? VIRTUAL_BASE : 0) + START + 1);
    assert.equal(jit.counts.entries - entries, 1, "oversized block is attempted once");
    const after = jit.full_snapshot();
    assert.equal(after.jit - before.jit, 0n);
    assert.equal(after.interpreted - before.interpreted, 1n);
    assert_budget_architecture(jit, cpus[1], `${name}: tiny fallback`);
    for(const budget of [0, 2, 257, 258, 259, 4095])
    {
        for(const cpu of cpus) cpu.restart_at(START, cold);
        const result = execute(budget, `${name}: budget ${budget}`);
        assert.equal(result.reason, 1);
        assert.equal(result.instructions, budget);
        assert_budget_architecture(jit, cpus[1], `${name}: budget ${budget}`);
        await Promise.resolve();
    }
    for(const cpu of cpus) cpu.set_counter(0xFFFFFFFE);
    execute(259, `${name}: tiny tail across counter wrap`);
    assert_budget_architecture(jit, cpus[1], `${name}: wrapped counter`);
    jit.assert_hoisted();
    console.log(`${name}: bounded fallback, exact budgets, full state and wrap passed`);
}

// Starting on the JMP inside a compiled two-instruction block should interpret
// that one instruction, then hand its remaining budget of two back to the cache.
{
    const { jit, cpus, execute } = create_pair(loops[0][1], {}, {
        compare_retired_partition: false, allow_budget_previous_eip: true,
    });
    execute(100004, "tiny backward reentry: aligned warmup");
    for(const cpu of cpus) cpu.restart_at(START + 1, false);
    const before = jit.full_snapshot();
    const result = execute(3, "tiny backward reentry");
    assert.equal(result.reason, 1);
    assert.equal(result.instructions, 3);
    const after = jit.full_snapshot();
    assert.equal(after.jit - before.jit, 2n);
    assert.equal(after.interpreted - before.interpreted, 1n);
    assert_budget_architecture(jit, cpus[1], "tiny backward reentry");
    jit.assert_hoisted();
    console.log("tiny backward reentry: one interpreted jump and two cached instructions passed");
}

// Rejecting a too-large compiled block must retry a self-modifying store exactly
// once. The target is in the same executable page but outside the running loop.
{
    const target = START + 0x100;
    const code = [0xA2, 0, 0, 0, 0, 0x40, 0xFF, 0xE1];
    code[1] = target & 0xFF;
    code[2] = target >>> 8;
    const { jit, cpus, execute } = create_pair(code);
    for(const cpu of cpus) cpu.set_register(0, 0x12);
    execute(3, "tiny SMC fallback: warmup");
    const invalidations = jit.counts.invalidations;
    assert.equal(jit.counts.modules, 1);
    for(const cpu of cpus) cpu.set_register(0, 0x34);
    const result = execute(1, "tiny SMC fallback");
    assert.equal(result.reason, 1);
    assert.equal(result.instructions, 1);
    assert.equal(result.eip, START + 5);
    assert.equal(jit.full_snapshot().state[11], START);
    assert.equal(jit.full_snapshot().memory[target], 0x34);
    assert(jit.counts.invalidations > invalidations);
    console.log("tiny SMC fallback: single store, precise previous IP and invalidation passed");
}

for(const [name, ending, reason] of [["halt", [0xF4], 2], ["invalid-opcode", [0x0F, 0x0B], 0x106]])
{
    // inc eax; dec edx; jnz start; ending. Warm the loop, then reach its early exit.
    const code = [0x40, 0x4A, 0x75, 0xFC, ...ending];
    const { jit, cpus, execute } = create_pair(code);
    for(const cpu of cpus) cpu.set_register(2, 1000000);
    execute(131072, `${name}: warmup`);
    for(const cpu of cpus) cpu.set_register(2, 7);
    const actual = execute(131072, name);
    assert.equal(actual.reason, reason);
    jit.assert_hoisted();
    console.log(`${name}: compiled early exit passed`);
}

for(const [name, ending, reason] of [["halt", [0xF4], 2], ["invalid-opcode", [0x0F, 0x0B], 0x106]])
{
    const code = [0x40, 0x4A, 0x75, 0xFC, ...ending];
    const { jit, cpus, execute } = create_pair(code, {}, { compare_retired_partition: false });
    for(const cpu of cpus) cpu.set_register(2, 1000000);
    execute(131072, `tiny ${name}: warmup`);
    for(const cpu of cpus) cpu.set_register(2, 7);
    const result = execute(64, `tiny ${name}`);
    assert.equal(result.reason, reason);
    const expected_eip = START + (name === "halt" ? 5 : 4);
    for(const cpu of cpus)
    {
        const state = cpu.full_snapshot().state;
        assert.equal(state[10], expected_eip, `tiny ${name}: terminal EIP`);
        assert.equal(state[11], START + 4, `tiny ${name}: terminal previous IP`);
    }
    jit.assert_hoisted();
    console.log(`tiny ${name}: exact terminal state and previous IP passed`);
}

// After warming, execute the byte store from the compiled module it modifies.
// The JIT must bail before the store, retry it interpreted, then recompile DEC EAX.
{
    const code = [
        0x40, 0x4A, 0x75, 0xFC, // inc eax; dec edx; jnz start
        0x85, 0xDB, 0x75, 0x0F, // test ebx, ebx; jnz halt
        0x43, // inc ebx: modify code only on the first pass
        0xC6, 0x05, 0x00, 0x10, 0x00, 0x00, 0x48, // mov byte [START], DEC EAX
        0xBA, 0x07, 0x00, 0x00, 0x00, // mov edx, 7
        0xFF, 0xE1, // jmp ecx
        0xF4, // halt after the modified loop
    ];
    const { jit, cpus, execute } = create_pair(code);
    for(const cpu of cpus) cpu.set_register(2, 1000000);
    execute(131072, "in-flight SMC: warmup");
    const before = { ...jit.counts };
    for(const cpu of cpus)
    {
        cpu.set_register(2, 7);
        cpu.restart_at_entry();
    }
    // The original JIT precharges three unexecuted instructions at the SMC
    // bailout; running to HLT lets the architectural interpreter oracle stay exact.
    const result = execute(131072, "in-flight SMC", { compare_instructions: false });
    assert.equal(result.reason, 2);
    assert(jit.counts.jit_slow_writes > before.jit_slow_writes,
        "self-modifying store must reach the compiled write helper");
    assert(jit.counts.invalidations > before.invalidations);
    assert(jit.counts.modules > before.modules, "modified code must recompile");
    assert.equal(jit.full_snapshot().memory[START], 0x48);
    jit.assert_hoisted();
    console.log("in-flight SMC: compiled bail, interpreted store and recompile passed");
}

// Exercise virtual-to-physical translation and the production-style deferred
// finalize callback, then reject a write from warmed compiled code.
{
    const { jit, cpus, execute } = create_pair(loops[1][1], { paging: true, asynchronous: true },
        { compare_retired_partition: false, allow_budget_previous_eip: true });
    for(const budget of budgets)
    {
        const actual = execute(budget, `paged asynchronous: budget ${budget}`);
        assert.equal(actual.reason, 1);
        assert.equal(actual.instructions, budget);
        await Promise.resolve();
    }
    jit.assert_hoisted();
    const entries_before = jit.counts.entries;
    for(const cpu of cpus)
    {
        cpu.protect_data_page();
        cpu.restart_at_entry();
    }
    const fault = execute(131072, "paged write protection");
    assert.equal(fault.reason, 0x10E);
    assert.equal(fault.error, 7); // present | write | user
    assert.equal(jit.full_snapshot().state[60], VIRTUAL_BASE + DATA); // CR2
    assert(jit.counts.entries > entries_before, "write fault must exit compiled code");
    jit.assert_hoisted();
    console.log("paged asynchronous: exact budgets and compiled write-protection fault passed");
}

const unchecked = create_cpu(loops[0][1], { exact: false });
unchecked.execute(131072);
unchecked.execute(131072);
unchecked.assert_hoisted();
if(baseline_module)
{
    const baseline = create_cpu(loops[0][1], { exact: false, cpu_module: baseline_module });
    baseline.execute(131072);
    baseline.execute(131072);
    assert.deepEqual(unchecked.full_snapshot(), baseline.full_snapshot());
}
console.log("disabled exact-budget mode: no getter passed");

{
    const unchecked = create_cpu(loops[0][1], { exact: false });
    const baseline = baseline_module && create_cpu(loops[0][1], {
        exact: false, cpu_module: baseline_module,
    });
    unchecked.execute(100004);
    if(baseline) baseline.execute(100004);
    for(const budget of [0, 1, 31, 4095])
    {
        unchecked.restart_at_entry();
        if(baseline) baseline.restart_at_entry();
        const entries = unchecked.counts.entries;
        const result = unchecked.execute(budget);
        assert.equal(result.reason, 1);
        assert.equal(result.instructions, budget);
        assert.equal(unchecked.counts.entries, entries,
            "unchecked mode must retain the 4096-instruction cache cutoff");
        if(baseline)
        {
            assert.deepEqual(result, baseline.execute(budget));
            assert.deepEqual(unchecked.full_snapshot(), baseline.full_snapshot());
        }
    }
    console.log("unchecked tiny tails: original cutoff and all 153 state words passed");
}

// A same-context API return must preserve every field outside its five outputs,
// even when the host staging buffer no longer contains the current CPU state.
{
    const cpu = create_cpu([0x90]);
    const before = cpu.full_snapshot();
    const staging = new Uint32Array(cpu.wasm.memory.buffer, cpu.wasm.vine_cpu_state_ptr(), 153);
    for(let index = 2; index < staging.length; index++) staging[index] = 0xa5a50000 + index;
    cpu.resume(null, START + 1, 0x8fffc);
    const expected = { ...before, state: [...before.state] };
    expected.state[6] = 0x8fffc;
    expected.state[10] = expected.state[11] = START + 1;
    expected.state[22] = 0;
    assert.deepEqual(cpu.full_snapshot(), expected, "partial resume preserves all 153 architectural words");
    cpu.resume(0xfedcba98, START, 0x90000);
    expected.state[2] = 0xfedcba98;
    expected.state[6] = 0x90000;
    expected.state[10] = expected.state[11] = START;
    assert.deepEqual(cpu.full_snapshot(), expected, "partial resume updates EAX only when requested");
    assert.equal(cpu.wasm.vine_cpu_apply_stopped_resume(1, 2, 0, 0), 1);
    assert.equal(cpu.wasm.vine_rearm_dirty_page(DATA + 1), 1);
    assert.deepEqual(cpu.full_snapshot(), expected, "invalid scalar arguments do not mutate CPU or memory");
    console.log("stopped resume: poisoned staging, all 153 words and argument rejection passed");
}

{
    let cpu;
    let rejected = 0;
    cpu = create_cpu([0x90, 0xeb, 0xfd], { on_finalize(wasm) {
        const before = cpu.full_snapshot();
        assert.equal(wasm.vine_cpu_apply_stopped_resume(1, 1, 0, 0), 1);
        assert.equal(wasm.vine_rearm_dirty_page(DATA), 1);
        assert.deepEqual(cpu.full_snapshot(), before, "active execution rejects both host mutations");
        rejected++;
    } });
    cpu.execute(4096);
    assert(rejected > 0, "validation must run during an actual JIT-finalize host call");
    console.log("active CPU: resume and dirty rearm reject without mutation passed");
}

// Keep hot read/code translations while rearming writes. The unchanged CPU uses
// full restore/full flush, and every differential snapshot remains fully strict.
{
    const bytes = new Uint8Array(256);
    const imm32 = value => [value & 255, value >>> 8 & 255, value >>> 16 & 255, value >>> 24];
    bytes.set([0xa1, ...imm32(VIRTUAL_BASE + DATA), 0xeb, 0xf9]);
    bytes.set([0xa3, ...imm32(VIRTUAL_BASE + DATA), 0xeb, 0xf9], 0x40);
    bytes.set([0xa3, ...imm32(VIRTUAL_BASE + DATA),
        0xa3, ...imm32(VIRTUAL_BASE + DATA + 4096), 0xeb, 0xf4], 0x80);
    const jit = create_cpu(bytes, { paging: true });
    const baseline = baseline_module && create_cpu(bytes, { paging: true, cpu_module: baseline_module });
    const cpus = [jit, ...(baseline ? [baseline] : [])];
    function execute(budget, description) {
        const actual = jit.execute(budget);
        if(baseline)
        {
            assert.deepEqual(actual, baseline.execute(budget), `${description}: result`);
            assert.deepEqual(jit.full_snapshot(), baseline.full_snapshot(),
                `${description}: all 153 state words, memory, dirty pages and retired partition`);
        }
        return actual;
    }
    for(const cpu of cpus)
    {
        cpu.physical.setUint32(DATA, 0x11223344, true);
        cpu.physical.setUint32(PAGE_TABLE + (DATA >>> 12) * 4 + 4, DATA | 7, true);
    }
    execute(4096, "warm read loop");
    execute(4096, "cached read loop");
    for(const cpu of cpus) cpu.wasm.full_clear_tlb();
    execute(64, "cold JIT read translation");
    assert(jit.counts.entries > 0 && jit.counts.jit_slow_reads > 0);
    const slow_reads = jit.counts.jit_slow_reads;
    for(const cpu of cpus)
    {
        cpu.clear_dirty([DATA]);
        cpu.resume();
    }
    const read = execute(64, "rearmed cached reads");
    assert.equal(read.reason, 1);
    assert.equal(read.registers[0], 0x11223344);
    assert.equal(jit.counts.jit_slow_reads, slow_reads, "READONLY rearm retains the hot JIT read translation");
    assert.equal(jit.wasm.vine_rearm_dirty_page(VIRTUAL_BASE + DATA + 8192), 0,
        "a never-cached virtual page remains invalid");

    for(const cpu of cpus) cpu.resume(0x12345678, START + 0x40);
    execute(4096, "warm write loop");
    execute(4096, "cached write loop");
    const entries = jit.counts.entries;
    for(const value of [0x22334455, 0x33445566])
    {
        for(const cpu of cpus)
        {
            cpu.clear_dirty([DATA]);
            cpu.resume(value, START + 0x40);
        }
        const result = execute(2, "successive hot writes");
        assert.equal(result.reason, 1);
        assert.equal(result.instructions, 2);
        assert.equal(jit.physical.getUint32(DATA, true), value);
        assert.deepEqual(jit.full_snapshot().dirty, [VIRTUAL_BASE + DATA]);
        assert(jit.physical.getUint32(PAGE_TABLE + (DATA >>> 12) * 4, true) & 0x40);
    }
    assert(jit.counts.entries > entries, "both successive writes must execute cached code");

    for(const cpu of cpus) cpu.resume(0x44556677, START + 0x80);
    execute(4095, "warm aliased writes");
    execute(4095, "cached aliased writes");
    for(const cpu of cpus)
    {
        cpu.clear_dirty([DATA, DATA + 4096]);
        cpu.resume(0x55667788, START + 0x80);
    }
    execute(3, "rearmed aliased writes");
    assert.deepEqual(jit.full_snapshot().dirty, [VIRTUAL_BASE + DATA, VIRTUAL_BASE + DATA + 4096]);
    assert.equal(jit.physical.getUint32(DATA, true), 0x55667788);
    for(const cpu of cpus)
    {
        cpu.clear_dirty([DATA, DATA + 4096]);
        cpu.protect_data_page();
        cpu.resume(0xdeadbeef, START + 0x40);
    }
    const fault = execute(16, "rearmed read-only fault");
    assert.equal(fault.reason, 0x10e);
    assert.equal(fault.error, 7);
    assert.equal(jit.full_snapshot().state[60], VIRTUAL_BASE + DATA);
    assert.equal(jit.full_snapshot().state[11], VIRTUAL_BASE + START + 0x40);
    assert.equal(jit.physical.getUint32(DATA, true), 0x55667788);
    assert.deepEqual(jit.full_snapshot().dirty, []);
    console.log("dirty rearm: cached reads, successive JIT writes, aliases and strict write fault passed");
}

{
    const target = START + 0x100;
    const address = VIRTUAL_BASE + target;
    const code = [0xa2, address & 255, address >>> 8 & 255, address >>> 16 & 255, address >>> 24,
        0x40, 0xff, 0xe1];
    const jit = create_cpu(code, { paging: true });
    const baseline = baseline_module && create_cpu(code, { paging: true, cpu_module: baseline_module });
    const cpus = [jit, ...(baseline ? [baseline] : [])];
    for(const cpu of cpus)
    {
        cpu.resume(0x12);
        cpu.execute(3);
        cpu.clear_dirty([START]);
        cpu.resume(0x34);
    }
    const writes = jit.counts.jit_slow_writes;
    const result = jit.execute(4);
    if(baseline)
    {
        assert.deepEqual(result, baseline.execute(4));
        assert.deepEqual(jit.full_snapshot(), baseline.full_snapshot(), "SMC keeps all 153 words and exact precharge");
    }
    assert.equal(result.reason, 1);
    assert.equal(result.eip, VIRTUAL_BASE + START + 5);
    assert.equal(jit.full_snapshot().state[11], VIRTUAL_BASE + START);
    assert.equal(jit.full_snapshot().memory[target], 0x34);
    assert(jit.counts.jit_slow_writes > writes, "rearmed executable page reaches the compiled SMC helper");
    assert.deepEqual(jit.full_snapshot().dirty, [VIRTUAL_BASE + START]);
    console.log("dirty rearm: compiled SMC bail and single interpreted retry passed");
}

// Full opaque restoration still replaces the address space and every parked
// thread field; a subsequent partial return must use that restored context.
{
    const address = VIRTUAL_BASE + DATA;
    const code = [0xa1, address & 255, address >>> 8 & 255, address >>> 16 & 255, address >>> 24,
        0xeb, 0xf9];
    const jit = create_cpu(code, { paging: true });
    const baseline = baseline_module && create_cpu(code, { paging: true, cpu_module: baseline_module });
    const cpus = [jit, ...(baseline ? [baseline] : [])];
    const parked = [];
    for(const cpu of cpus)
    {
        cpu.physical.setUint32(DATA, 0x66778899, true);
        cpu.execute(4096);
        cpu.execute(4096);
        parked.push(cpu.full_snapshot().state);
        const memory = new Uint8Array(cpu.physical.buffer, cpu.physical.byteOffset, MEMORY_SIZE);
        memory.copyWithin(0x12000, PAGE_DIRECTORY, PAGE_DIRECTORY + 4096);
        memory.copyWithin(0x13000, PAGE_TABLE, PAGE_TABLE + 4096);
        cpu.physical.setUint32(0x12000 + (VIRTUAL_BASE >>> 22) * 4, 0x13000 | 7, true);
        cpu.physical.setUint32(0x13000 + (DATA >>> 12) * 4, 0xa000 | 7, true);
        cpu.physical.setUint32(0xa000, 0x778899aa, true);
        const state = new Uint32Array(cpu.wasm.memory.buffer, cpu.wasm.vine_cpu_state_ptr(), 153);
        state[61] = 0x12000;
        state[5] = 0x12344321;
        state[38] = 0x4000; // the parked thread's FS base
        assert.equal(cpu.wasm.vine_cpu_state_restore(), 0);
        cpu.resume();
    }
    const switched = jit.execute(2);
    if(baseline)
    {
        assert.deepEqual(switched, baseline.execute(2));
        assert.deepEqual(jit.full_snapshot(), baseline.full_snapshot(), "CR3 switch remains fully strict");
    }
    assert.equal(switched.registers[0], 0x778899aa, "full restore flushes the previous data translation");
    assert.equal(switched.registers[3], 0x12344321);
    for(let index = 0; index < cpus.length; index++)
    {
        const cpu = cpus[index];
        new Uint32Array(cpu.wasm.memory.buffer, cpu.wasm.vine_cpu_state_ptr(), 153).set(parked[index]);
        assert.equal(cpu.wasm.vine_cpu_state_restore(), 0);
        assert.deepEqual(cpu.full_snapshot().state, parked[index], "all parked thread words restore exactly");
        cpu.resume();
    }
    const restored = jit.execute(2);
    if(baseline)
    {
        assert.deepEqual(restored, baseline.execute(2));
        assert.deepEqual(jit.full_snapshot(), baseline.full_snapshot(), "restored parked context remains strict");
    }
    assert.equal(restored.registers[0], 0x66778899);
    console.log("stopped resume: full CR3/thread restoration retains flush and all 153 words passed");
}

function update_cpu_words(cpu, update)
{
    cpu.wasm.vine_cpu_state_save();
    update(new Uint32Array(cpu.wasm.memory.buffer, cpu.wasm.vine_cpu_state_ptr(), 153));
    assert.equal(cpu.wasm.vine_cpu_state_restore(), 0);
}

// INC; MOVQ mm1,mm0; MOVDQA xmm2,xmm1; JMP. The first SIMD guard
// remains after INC, and two register-only instructions share its success.
const register_simd = [0x40, 0x0f, 0x6f, 0xc8, 0x66, 0x0f, 0x6f, 0xd1, 0xeb, 0xf6];
function seed_simd(cpu)
{
    update_cpu_words(cpu, state => {
        state[62] |= 0x200; // CR4.OSFXSR
        state[97] = 0x12345678; // MM0 mantissa
        state[98] = 0x9abcdef0;
        state[99] = 0xffff;
        state.set([0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00], 125); // XMM1
    });
}

{
    const { jit, baseline, cpus, execute } = create_pair(register_simd);
    for(const cpu of cpus) seed_simd(cpu);
    execute(132000, "register SIMD aligned warmup");
    assert.equal(jit.counts.modules, 1);
    assert.equal(jit.counts.mmx_guards, 1, "a compiled register run retains only its first guard");
    if(baseline) assert.equal(baseline.counts.mmx_guards, 1);
    const entries = jit.counts.entries;
    for(const budget of [0, 1, 2, 3, 4, 5, 4095, 4096, 4097])
        assert.equal(execute(budget, `register SIMD budget ${budget}`).instructions, budget);
    assert(jit.counts.entries > entries, "the reduced guard sequence must execute cached code");
    const state = jit.full_snapshot().state;
    assert.deepEqual(state.slice(100, 102), [0x12345678, 0x9abcdef0]); // MM1
    assert.deepEqual(state.slice(129, 133), state.slice(125, 129)); // XMM2
    console.log("SIMD register guards: actual elision, exact budgets and strict baseline state passed");
}

for(const [name, bits, reason] of [["EM", 4, 0x106], ["TS", 8, 0x107], ["EM+TS", 12, 0x106]])
{
    const { jit, cpus, execute } = create_pair(register_simd);
    for(const cpu of cpus) seed_simd(cpu);
    execute(132000, `${name}: warm SIMD`);
    for(const cpu of cpus)
    {
        update_cpu_words(cpu, state => { state[58] |= bits; });
        cpu.resume();
    }
    const entries = jit.counts.entries;
    const result = execute(64, `${name}: first SIMD fault`, { compare_instructions: false });
    assert.equal(result.reason, reason);
    assert.equal(result.eip, START + 1, "INC executes before the first SIMD exception");
    assert(jit.counts.entries > entries, "the restored CR0 must be checked by cached code");
    assert.equal(jit.full_snapshot().state[11], START + 1, "JIT publishes the faulting IP");
    assert.equal(cpus[1].full_snapshot().state[11], START + 1, "interpreter publishes the faulting IP");
}

// A real MMIO read callback changes CR0.TS inside the first SIMD memory
// instruction. The following register instruction must perform a fresh check.
{
    let armed = false;
    const code = [0x40, 0x0f, 0x6f, 0xc8, 0x0f, 0x6f, 0x07, 0x0f, 0x6f, 0xd8, 0xeb, 0xf4];
    const { jit, baseline, cpus, execute } = create_pair(code, {
        on_mmap(name, _args, wasm) {
            assert.equal(name, "mmap_read32");
            if(armed)
            {
                const memory = new DataView(wasm.memory.buffer);
                memory.setUint32(580, memory.getUint32(580, true) | 8, true);
            }
            return 0x11223344;
        },
    });
    for(const cpu of cpus)
    {
        seed_simd(cpu);
        cpu.set_register(7, 0xa0000);
    }
    execute(132000, "MMIO SIMD aligned warmup");
    assert.equal(jit.counts.mmx_guards, 2, "memory helper emission breaks guard reuse");
    if(baseline) assert.equal(baseline.counts.mmx_guards, 2);
    for(const cpu of cpus) cpu.resume();
    armed = true;
    const entries = jit.counts.entries;
    const accesses = jit.counts.mmap;
    const result = execute(64, "MMIO changes TS before next SIMD", { compare_instructions: false });
    assert.equal(result.reason, 0x107);
    assert.equal(result.eip, START + 7);
    assert(jit.counts.entries > entries && jit.counts.mmap > accesses,
        "the compiled memory instruction must invoke actual MMIO");
    assert.equal(jit.full_snapshot().state[11], START + 7);
    assert.equal(cpus[1].full_snapshot().state[11], START + 7);
    console.log("SIMD guards: restored EM/TS fault position and MMIO context mutation passed");
}

// A compiled read can fault before the block's final instruction publishes its
// previous IP. Repair the mapping and retry the saved faulting instruction.
{
    const { jit, cpus, execute } = create_pair([0x40, 0x8B, 0x1F, 0xEB, 0xFB], { paging: true });
    for(const cpu of cpus) cpu.resume();
    execute(132000, "read PF: warm");
    for(const cpu of cpus)
    {
        cpu.physical.setUint32(PAGE_TABLE + (DATA >>> 12) * 4, DATA | 6, true);
        cpu.wasm.full_clear_tlb();
        cpu.resume();
    }
    const entries = jit.counts.entries;
    const fault = execute(64, "read PF: compiled absent page", { compare_instructions: false });
    assert.equal(fault.reason, 0x10E);
    assert.equal(fault.error, 4);
    assert.equal(fault.eip, VIRTUAL_BASE + START + 1);
    assert(jit.counts.entries > entries);
    assert.equal(jit.full_snapshot().state[11], fault.eip);
    assert.equal(jit.full_snapshot().state[60], VIRTUAL_BASE + DATA);
    for(const cpu of cpus)
    {
        cpu.physical.setUint32(PAGE_TABLE + (DATA >>> 12) * 4, DATA | 7, true);
        cpu.wasm.full_clear_tlb();
    }
    const retry = execute(1, "read PF: repair and retry saved context");
    assert.equal(retry.registers[0], fault.registers[0], "retry does not duplicate the preceding INC");
    assert.equal(retry.eip, VIRTUAL_BASE + START + 3);
    execute(64, "read PF: continue after retry");
}

function seed_fpu(cpu, bits = 0x3fc00000, { empty = false, full = false, cw = 0x37f } = {})
{
    const memory = new DataView(cpu.wasm.memory.buffer);
    memory.setUint8(816, full ? 0 : empty ? 0xff : 0xf7);
    memory.setUint8(1032, 3);
    memory.setUint16(1040, 0x4200, true);
    for(let register = 0; register < 8; register++)
    {
        memory.setBigUint64(1152 + register * 16, 0x8000000000000000n, true);
        memory.setUint16(1160 + register * 16, 0x3fff, true);
    }
    cpu.wasm.set_control_word(cw);
    new Uint8Array(cpu.wasm.memory.buffer, 1136, 16).fill(0xa5);
    cpu.physical.setUint32(DATA, bits, true);
    cpu.resume(0);
}

function assert_fpu_scratch(jit, baseline, description)
{
    if(baseline) assert.deepEqual(
        Buffer.from(jit.wasm.memory.buffer, 1136, 10),
        Buffer.from(baseline.wasm.memory.buffer, 1136, 10),
        `${description}: converted F80 scratch fields`);
}

// Only m32 memory operations are fused. Exercise every D8 /n and FLD with
// a cached module, exact C18 architectural/retirement comparisons, and scratch
// fields that the generated caller formerly reloaded after conversion.
for(const operation of [null, 0, 1, 2, 3, 4, 5, 6, 7])
{
    const name = operation === null ? "FLD m32" : `D8 /${operation} m32`;
    const code = [0x40, operation === null ? 0xd9 : 0xd8,
        0x07 | (operation ?? 0) << 3, 0xeb, 0xfb];
    const { jit, baseline, cpus, execute } = create_pair(code);
    for(const cpu of cpus) seed_fpu(cpu);
    execute(132000, `${name}: warm`);
    assert(jit.counts.entries > 0);
    assert(jit.counts.fpu_imports.has(operation === null ? "fpu_fld_m32_jit" : "fpu_op_m32_jit"));
    assert(!jit.counts.fpu_imports.has("f32_to_f80_jit"), "one generated native import replaces the pair");
    if(baseline) assert(baseline.counts.fpu_imports.has("f32_to_f80_jit") ||
        baseline.counts.fpu_imports.has(operation === null ? "fpu_fld_m32_jit" : "fpu_op_m32_jit"));
    for(const bits of [0x3fc00000, 0xbf000000, 0, 0x80000000, 1, 0x7f7fffff, 0x7fc01234, 0x7f800001])
    {
        for(const cpu of cpus) seed_fpu(cpu, bits);
        const entries = jit.counts.entries;
        execute(3, `${name}: input ${bits.toString(16)}`);
        assert(jit.counts.entries > entries, "the fused operation must actually execute generated code");
        assert_fpu_scratch(jit, baseline, name);
    }
    for(const state of [{ empty: true }, { full: true }])
    {
        for(const cpu of cpus) seed_fpu(cpu, 0x7f800001, state);
        execute(3, `${name}: stack ${state.empty ? "empty" : "full"}`);
        assert_fpu_scratch(jit, baseline, name);
    }
    if(operation === 0 || operation === 6)
        for(const precision of [0, 2, 3]) for(const rounding of [0, 1, 2, 3])
        {
            const cw = 0x7f | precision << 8 | rounding << 10;
            for(const cpu of cpus) seed_fpu(cpu, 0x3dcccccd, { cw });
            execute(3, `${name}: precision ${precision}, rounding ${rounding}`);
            assert_fpu_scratch(jit, baseline, name);
        }
    for(const budget of [0, 1, 2, 4095, 4096, 4097])
    {
        for(const cpu of cpus) seed_fpu(cpu);
        execute(budget, `${name}: budget ${budget}`);
        assert_fpu_scratch(jit, baseline, name);
    }
    console.log(`${name}: strict cached conversion, stack, flags, scratch and budgets passed`);
}

// The #NM guard stays before memory access. On PF neither conversion nor push
// may run, and repairing the mapping must retry only the saved instruction.
for(const opcode of [0xd9, 0xd8])
{
    const name = opcode === 0xd9 ? "FLD" : "FADD";
    const { jit, baseline, cpus, execute } = create_pair([0x40, opcode, 7, 0xeb, 0xfb], { paging: true });
    for(const cpu of cpus) seed_fpu(cpu);
    execute(132000, `${name}: paged warm`);
    for(const bits of [4, 8, 12, 0])
    {
        for(const cpu of cpus)
        {
            seed_fpu(cpu);
            cpu.physical.setUint32(PAGE_TABLE + (DATA >>> 12) * 4, DATA | 6, true);
            const memory = new DataView(cpu.wasm.memory.buffer);
            memory.setUint32(580, 0x80010001 | bits, true);
            cpu.wasm.full_clear_tlb();
        }
        const result = execute(64, `${name}: absent page, EM/TS ${bits}`, { compare_instructions: false });
        assert.equal(result.reason, bits ? 0x107 : 0x10e);
        assert.equal(result.eip, VIRTUAL_BASE + START + 1);
        assert.equal(jit.full_snapshot().state[11], result.eip);
        assert(Buffer.from(jit.wasm.memory.buffer, 1136, 10).every(byte => byte === 0xa5));
        assert_fpu_scratch(jit, baseline, name);
    }
    for(const cpu of cpus)
    {
        cpu.physical.setUint32(PAGE_TABLE + (DATA >>> 12) * 4, DATA | 7, true);
        cpu.wasm.full_clear_tlb();
    }
    const retry = execute(1, `${name}: repair read PF`);
    assert.equal(retry.registers[0], 1, "fault retry must not duplicate preceding INC");
    assert.equal(retry.eip, VIRTUAL_BASE + START + 3);
    assert_fpu_scratch(jit, baseline, name);
}

// A real MMIO callback updates FPU state and CR0 after the instruction's guard.
// Complete that first arithmetic operation; the next x87 instruction raises #NM.
{
    let armed = false;
    const { jit, baseline, cpus, execute } = create_pair([0x40, 0xd8, 7, 0xd8, 7, 0xeb, 0xf9], {
        on_mmap(name, _args, wasm) {
            assert.equal(name, "mmap_read32");
            if(armed)
            {
                const memory = new DataView(wasm.memory.buffer);
                memory.setUint8(1032, 5);
                memory.setUint8(816, 0xdf);
                memory.setUint16(1040, 0x400, true);
                memory.setBigUint64(1152 + 5 * 16, 0x8000000000000000n, true);
                memory.setUint16(1160 + 5 * 16, 0x3ffe, true);
                memory.setUint32(580, memory.getUint32(580, true) | 8, true);
            }
            return 0x3fc00000;
        },
    });
    for(const cpu of cpus) { seed_fpu(cpu); cpu.set_register(7, 0xa0000); }
    execute(132000, "FPU MMIO warm");
    for(const cpu of cpus) seed_fpu(cpu);
    armed = true;
    const entries = jit.counts.entries, accesses = jit.counts.mmap;
    const result = execute(64, "FPU MMIO state and next guard", { compare_instructions: false });
    assert.equal(result.reason, 0x107);
    assert.equal(result.eip, START + 3);
    assert(jit.counts.entries > entries && jit.counts.mmap > accesses);
    const memory = new DataView(jit.wasm.memory.buffer);
    assert.equal(memory.getUint8(1032), 5);
    assert.equal(memory.getBigUint64(1152 + 5 * 16, true), 0x8000000000000000n);
    assert.equal(memory.getUint16(1160 + 5 * 16, true), 0x4000, "first operation completed to 2.0 after MMIO");
    assert_fpu_scratch(jit, baseline, "FPU MMIO");
}

const ordinary = create_cpu(loops[0][1]);
ordinary.execute(131072);
ordinary.run_unbounded_entry();
ordinary.assert_hoisted();
if(baseline_module)
{
    const baseline = create_cpu(loops[0][1], { cpu_module: baseline_module });
    baseline.execute(131072);
    baseline.run_unbounded_entry();
    assert.deepEqual(ordinary.full_snapshot(), baseline.full_snapshot());
}
console.log("ordinary non-Vine JIT invocation: unbounded getter passed");

// SBB32 same-register lowering preserves raw lazy flags and subsequent reads.
{
    const values = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF];
    const fixtures = [
        { alias: true, code: [0x1B, 0xC0, 0xFF, 0xE1] },
        { alias: false, code: [0x1B, 0xC3, 0xFF, 0xE1] },
    ];
    let cases = 0;
    for(const fixture of fixtures)
    {
        const pair = create_pair(Uint8Array.from(fixture.code));
        for(let warm = 0; warm < 258; warm++) pair.execute(4096, "SBB warm entry");
        assert(pair.jit.counts.entries > 0);
        if(pair.baseline) assert(pair.baseline.counts.entries > 0);
        for(const dest of values) for(const carry of [0, 1])
        {
            const source = fixture.alias ? dest : values[(values.indexOf(dest) + 2) % values.length];
            for(const cpu of pair.cpus)
            {
                cpu.resume(dest);
                cpu.set_register(3, source);
                const fields = new DataView(cpu.wasm.memory.buffer);
                fields.setUint32(96, 31, true);
                fields.setUint32(100, 0, true);
                fields.setUint32(104, 0xDEADBEEF, true);
                fields.setUint32(112, 0xCAFE1234, true);
                fields.setUint32(120, 0x202 | 0x8D4 | carry, true);
            }
            const before = pair.jit.full_snapshot();
            const result = pair.execute(2, "SBB compiled alias/nonalias with seeded carry");
            const after = pair.jit.full_snapshot();
            assert.equal(after.jit - before.jit, 2n, "case actually executes the compiled two-instruction BB");
            const value = Number(BigInt.asUintN(32, BigInt(dest) - BigInt(source) - BigInt(carry)));
            assert.equal(result.registers[0], value);
            assert.equal(result.flags & 1, Number(BigInt(dest) < BigInt(source) + BigInt(carry)));
            assert.equal(result.flags & 16, (dest ^ source ^ value) & 16);
            assert.equal(result.flags & 2048, ((dest ^ source) & (dest ^ value) & 0x80000000) ? 2048 : 0);
            assert.equal(new DataView(pair.jit.wasm.memory.buffer).getUint32(104, true), 0xDEADBEEF,
                "SBB does not write last_op1");
            cases++;
        }
        pair.jit.assert_hoisted();
        if(pair.baseline) pair.baseline.assert_hoisted();
    }
    console.log(`SBB alias focused: ${cases} compiled cases, alias/nonalias and both carry inputs.`);
}
