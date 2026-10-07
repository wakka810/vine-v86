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
} = {})
{
    const table = new WebAssembly.Table({ element: "anyfunc", initial: TABLE_OFFSET + 900 });
    let wasm;
    let jit_imports;
    let last_entry;
    let last_limit;
    const counts = { entries: 0, reads: 0, modules: 0, invalidations: 0, jit_slow_writes: 0 };
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
                case "codegen_finalize": {
                    const [index, start, state_flags, pointer, length] = args;
                    const generated = new WebAssembly.Module(
                        new Uint8Array(wasm.memory.buffer, pointer, length));
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
