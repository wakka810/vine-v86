#!/usr/bin/env node

// Architectural differential checks, not timings: both entry wrappers call JS.
import assert from "node:assert/strict";
import { create_cpu } from "./vine-jit-budget.js";

const A = 0x1000, B = 0x3000, C = 0x5000;
const u32 = value => [value & 255, value >>> 8 & 255, value >>> 16 & 255, value >>> 24];
const jump = (from, to, opcode = 0xe9) => [opcode, ...u32(to - from - 5)];
const call = (from, to) => jump(from, to, 0xe8);
function program(parts) {
    const bytes = new Uint8Array(0x5000).fill(0x90);
    for(const [address, instructions] of parts) bytes.set(instructions, address - A);
    return bytes;
}
function pair(parts, options = {}) {
    const bytes = program(parts);
    const chained = create_cpu(bytes, { ...options, chaining: true });
    const plain = create_cpu(bytes, { ...options, chaining: false });
    for(const cpu of [chained, plain]) cpu.wasm.set_jit_config(1, 1);
    return { chained, plain, cpus: [chained, plain] };
}
function equal_state(chained, plain, budget_exit, description) {
    const { jit, interpreted, ...actual } = chained.full_snapshot();
    const { jit: reference_jit, interpreted: reference_interpreted, ...expected } = plain.full_snapshot();
    // PREVIOUS_EIP is not architectural at a successful budget exit. At faults
    // and other stops it must still match, as must every other state word.
    if(budget_exit) {
        actual.state = actual.state.filter((_, index) => index !== 11);
        expected.state = expected.state.filter((_, index) => index !== 11);
    }
    assert.deepEqual(actual, expected, description);
    assert.equal(jit + interpreted, reference_jit + reference_interpreted, description);
}
function run(both, budget, description) {
    const actual = both.chained.execute(budget);
    assert.deepEqual(actual, both.plain.execute(budget), description);
    equal_state(both.chained, both.plain, actual.reason === 1, description);
    return actual;
}

const loops = [
    ["direct CALL/RET", [[A, [0x40, ...call(A + 1, B), 0x43, ...jump(A + 7, A)]],
        [B, [0x01, 0xc2, 0xc3]]]],
    ["indirect CALL/RET", [[A, [0xb9, ...u32(B), 0xff, 0xd1, 0x40, ...jump(A + 8, A)]],
        [B, [0x43, 0xc3]]]],
    ["direct JMP", [[A, [0x40, ...jump(A + 1, B)]], [B, [0x43, ...jump(B + 1, A)]]]],
    ["indirect JMP", [[A, [0x40, 0xba, ...u32(B), 0xff, 0xe2]],
        [B, [0x43, 0xba, ...u32(A), 0xff, 0xe2]]]],
    ["conditional exits", [[A, [0x40, 0xa8, 1, 0x0f, 0x84, ...u32(B - (A + 9)), ...jump(A + 9, C)]],
        [B, [0x83, 0xc3, 3, ...jump(B + 3, A)]], [C, [0x83, 0xeb, 2, ...jump(C + 3, A)]]]],
    ["nested CALL/RET", [[A, [...call(A, B), ...jump(A + 5, A)]],
        [B, [0x40, ...call(B + 1, C), 0x43, 0xc3]], [C, [0x42, 0xc3]]]],
];
for(const [name, parts] of loops) {
    const both = pair(parts);
    for(let index = 0; index < 8; index++) run(both, 131072, `${name}: warm ${index}`);
    const before = { ...both.chained.counts };
    run(both, 1_000_003, `${name}: deep bounded native chain`);
    const transfers = both.chained.counts.chain_entries - before.chain_entries;
    const entries = both.chained.counts.entries - before.entries;
    assert(transfers > 100000, `${name}: must exercise a chain too deep for ordinary calls`);
    assert(entries < 40, `${name}: native chaining must avoid the CPU dispatcher`);
    for(const budget of [0, 1, 2, 5, 7, 4095, 4096, 4097, 131072])
        run(both, budget, `${name}: exact budget ${budget}`);
    for(const cpu of both.cpus) cpu.set_counter(0xfffffff0);
    run(both, 4097, `${name}: wrapping global counter`);
    both.chained.assert_hoisted();
    console.log(`${name}: ${transfers} native transfers, ${entries} CPU entries; full state passed`);
}

{
    const both = pair([
        [A, [0x85, 0xf6, 0x75, 12, 0xc6, 0x05, ...u32(B), 0x48,
            0xbe, ...u32(1), ...jump(A + 16, B)]],
        [B, [0x40, ...jump(B + 1, A)]],
    ]);
    for(let index = 0; index < 6; index++) run(both, 131072, "SMC: warm");
    const invalidations = both.chained.counts.invalidations;
    for(const cpu of both.cpus) { cpu.set_register(6, 0); cpu.resume(); }
    run(both, 4096, "SMC: another cached module changes before a chain");
    assert(both.chained.counts.invalidations > invalidations);
    assert.equal(both.chained.physical.getUint8(B), 0x48);
    run(both, 131072, "SMC: changed target is recompiled");
    console.log("cross-module self modification: stale entry retired and changed code executed");
}

for(const [name, fault, options, reason] of [
    ["UD2", [0x0f, 0x0b], {}, 0x106],
    ["divide zero", [0xf7, 0xf5], {}, 0x100],
    ["page fault", [0xa1, ...u32(0x500000)], { paging: true }, 0x10e],
]) {
    const both = pair([[A, [0x40, ...call(A + 1, B), ...jump(A + 6, A)]],
        [B, [0x85, 0xf6, 0x74, 1, 0xc3, ...fault]]], options);
    for(let index = 0; index < 6; index++) run(both, 131072, `${name}: warm`);
    for(const cpu of both.cpus) { cpu.set_register(6, 0); cpu.resume(); }
    const stopped = run(both, 4096, `${name}: chained fault`);
    assert.equal(stopped.reason, reason);
    console.log(`${name}: all 153 state words, dirty pages and retired accounting passed`);
}

{
    let active_rejections = 0;
    const cpu = create_cpu(program(loops[2][1]), { chaining: true, asynchronous: true,
        on_finalize(wasm) {
            assert.equal(wasm.vine_set_jit_chaining(0), 1);
            active_rejections++;
        },
    });
    cpu.wasm.set_jit_config(1, 1);
    for(let index = 0; index < 10; index++) { cpu.execute(131072); await Promise.resolve(); }
    assert(active_rejections > 0 && cpu.counts.chain_entries > 0);
    const snapshot = cpu.full_snapshot();
    assert.equal(cpu.wasm.vine_set_jit_chaining(2), 1);
    assert.deepEqual(cpu.full_snapshot(), snapshot);
    assert.equal(cpu.wasm.vine_set_jit_chaining(0), 0);
    const transfers = cpu.counts.chain_entries;
    for(let index = 0; index < 6; index++) { cpu.execute(131072); await Promise.resolve(); }
    assert.equal(cpu.counts.chain_entries, transfers);
    assert.equal(cpu.wasm.vine_set_jit_chaining(1), 0);
    for(let index = 0; index < 6; index++) { cpu.execute(131072); await Promise.resolve(); }
    assert(cpu.counts.chain_entries > transfers);
    cpu.assert_hoisted();
    console.log("asynchronous publication and mode change: live ABI retired, active mutation rejected");
}

for(const enabled of [false, true]) {
    const queued = [];
    const original_queue = globalThis.queueMicrotask;
    const cpu = create_cpu(program(loops[2][1]), { chaining: enabled, asynchronous: true });
    const reference = create_cpu(program(loops[2][1]), { chaining: false });
    cpu.wasm.set_jit_config(1, 1);
    reference.wasm.set_jit_config(1, 1);
    try {
        globalThis.queueMicrotask = callback => queued.push(callback);
        const both = { chained: cpu, plain: reference };
        run(both, 4096, "pending mode change: initial run");
        assert(queued.length > 0, "an old-ABI completion must actually be pending");
        assert.equal(cpu.wasm.vine_set_jit_chaining(Number(!enabled)), 0);
        while(queued.length) queued.shift()();
        for(let index = 0; index < 6; index++) {
            run(both, 131072, "pending mode change: no stale target is callable");
            while(queued.length) queued.shift()();
        }
        assert.equal(cpu.wasm.vine_get_jit_chaining(), Number(!enabled));
        if(!enabled) assert(cpu.counts.chain_entries > 0);
    } finally {
        globalThis.queueMicrotask = original_queue;
    }
}
console.log("pending compilation: both ABI mode changes reject stale finalization");

// The fused indirect-jump tail chains cross-module transfers
// directly with the state and table index it already loaded. The same
// indirect call/ret between two pages must produce identical architectural
// state, identical retired accounting and the same native transfers under
// every setting of config 18; enabling bit 0 must change the generated module
// (the dead re-dispatch and duplicate tlb_code/state lookup are gone).
{
    const parts = [[A, [0xb9, ...u32(B), 0xff, 0xd1, 0x40, ...jump(A + 8, A)]],
        [B, [0x43, 0xc3]]];
    const budgets = [131072, 131072, 131072, 131072, 4096, 1, 7, 131072];
    const runs = {};
    for(const cg of [0, 1, 2, 3]) {
        const interpreter = create_cpu(program(parts), { disable_jit: true });
        const captured = [];
        const cpu = create_cpu(program(parts), { chaining: true,
            on_finalize(w, args) {
                captured.push(Buffer.from(w.memory.buffer, args[3], args[4]).toString("hex"));
            } });
        cpu.wasm.set_jit_config(18, cg);
        assert.equal(cpu.wasm.get_jit_config(18), cg, "config 18 must round-trip");
        for(const budget of budgets) {
            assert.equal(cpu.execute(budget).reason, 1, `cg=${cg}: budget exit`);
            interpreter.execute(budget);
        }
        const { jit, interpreted, ...actual } = cpu.full_snapshot();
        const { jit: jit_b, interpreted: interpreted_b, ...expected } = interpreter.full_snapshot();
        // PREVIOUS_EIP is not architectural at a budget exit.
        actual.state = actual.state.filter((_, index) => index !== 11);
        expected.state = expected.state.filter((_, index) => index !== 11);
        assert.deepEqual(actual, expected, `cg=${cg}: state must match the interpreter`);
        assert(cpu.counts.chain_entries > 0, `cg=${cg}: indirect transfers must chain`);
        assert(cpu.counts.entries < 40,
            `cg=${cg}: native chaining must avoid the CPU dispatcher`);
        runs[cg] = { modules: captured.slice().sort(), transfers: cpu.counts.chain_entries };
        cpu.assert_hoisted();
    }
    assert.equal(runs[1].transfers, runs[0].transfers,
        "bit 0 must chain the same transfers");
    assert.equal(runs[3].transfers, runs[0].transfers,
        "both bits must chain the same transfers");
    assert.notDeepEqual(runs[1].modules, runs[0].modules,
        "bit 0 must change the generated module");
    console.log("direct fused chaining: identical state and transfers across config 18 = 0..3");
}
