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
function equalState(chained, plain, budgetExit, description) {
    const { jit, interpreted, ...actual } = chained.full_snapshot();
    const { jit: referenceJit, interpreted: referenceInterpreted, ...expected } = plain.full_snapshot();
    // PREVIOUS_EIP is not architectural at a successful budget exit. At faults
    // and other stops it must still match, as must every other state word.
    if(budgetExit) {
        actual.state = actual.state.filter((_, index) => index !== 11);
        expected.state = expected.state.filter((_, index) => index !== 11);
    }
    assert.deepEqual(actual, expected, description);
    assert.equal(jit + interpreted, referenceJit + referenceInterpreted, description);
}
function run(both, budget, description) {
    const actual = both.chained.execute(budget);
    assert.deepEqual(actual, both.plain.execute(budget), description);
    equalState(both.chained, both.plain, actual.reason === 1, description);
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
    let activeRejections = 0;
    const cpu = create_cpu(program(loops[2][1]), { chaining: true, asynchronous: true,
        on_finalize(wasm) {
            assert.equal(wasm.vine_set_jit_chaining(0), 1);
            activeRejections++;
        },
    });
    cpu.wasm.set_jit_config(1, 1);
    for(let index = 0; index < 10; index++) { cpu.execute(131072); await Promise.resolve(); }
    assert(activeRejections > 0 && cpu.counts.chain_entries > 0);
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
    const originalQueue = globalThis.queueMicrotask;
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
        globalThis.queueMicrotask = originalQueue;
    }
}
console.log("pending compilation: both ABI mode changes reject stale finalization");
