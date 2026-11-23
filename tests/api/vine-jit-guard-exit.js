#!/usr/bin/env node

// Run make vine-jit-guard-exit-test, or use an existing build with:
// node tests/api/vine-jit-guard-exit.js
// V86_WASM can select another build. Correctness test for guard exits
// (jit config 14): a failed common-memory guard exits the generated
// module before the instruction and the interpreter performs the access,
// instead of calling the rejoining slow helper. Knob off must behave exactly
// like the existing path.
import assert from "node:assert/strict";
import fs from "node:fs";
import { create_cpu } from "./vine-jit-budget.js";

const wasm_path = process.env.V86_WASM || new URL("../../build/v86-fallback.wasm", import.meta.url);
const module = new WebAssembly.Module(fs.readFileSync(wasm_path));
const V = 0x400000;
const A = 0x1000;
const DATA = 0x8000;
const PERMS = 0xff000;
const PAGE_TABLE = 0x11000;

const u32 = n => [n & 255, n >>> 8 & 255, n >>> 16 & 255, n >>> 24 & 255];
const loop = bytes => Uint8Array.from([...bytes, 0xE9, ...u32(-bytes.length - 5)]);

function set_perm(cpu, vaddr, bits)
{
    const base = cpu.physical.byteOffset;
    new Uint8Array(cpu.wasm.memory.buffer)[base + PERMS + (vaddr >>> 12)] = bits;
}

function pte(cpu, address, value)
{
    cpu.physical.setUint32(PAGE_TABLE + (address >>> 12) * 4, value, true);
    // Invalidate only the data page's translations: a full_clear_tlb would
    // drop the code page's compiled entries too and the bail could never fire.
    if(typeof cpu.wasm.vine_invalidate_tlb_range === "function")
        assert.equal(cpu.wasm.vine_invalidate_tlb_range(V + address, 0x1000), 0);
    else cpu.wasm.full_clear_tlb();
}

// Two independent instances per scenario: one JIT instance under test and one
// interpreter oracle, both with common memory bound so the JIT side emits
// common-memory guards.
function make(code, { guard_exit = false, chaining = false } = {})
{
    const jit = create_cpu(code, { paging: true, cpu_module: module, chaining });
    const oracle = create_cpu(code, { paging: true, disable_jit: true, cpu_module: module });
    for(const c of [jit, oracle])
    {
        const base = c.physical.byteOffset;
        const perms = base + PERMS;
        // read|write for every page in the fixture's 4 MiB window
        new Uint8Array(c.wasm.memory.buffer, perms + (V >>> 12), 0x100).fill(3);
        assert.equal(c.wasm.vine_set_common_memory((base - V) >>> 0, perms), 0);
    }
    jit.wasm.set_jit_config(14, Number(guard_exit));
    oracle.wasm.set_jit_config(14, Number(guard_exit));
    // Keep the deopt limit out of the way of the other scenarios; the deopt
    // scenario sets its own.
    jit.wasm.set_jit_config(16, 1_000_000_000);
    oracle.wasm.set_jit_config(16, 1_000_000_000);
    const bail_count = () => typeof jit.wasm.vine_guard_bail_count === "function" ?
        jit.wasm.vine_guard_bail_count() : 0;
    const sites = () => typeof jit.wasm.vine_guard_exit_sites === "function" ?
        { bail: jit.wasm.vine_guard_exit_sites(), kept: jit.wasm.vine_guard_exit_kept() } : null;

    function compare(budget, label, { compare_instructions = true } = {})
    {
        const actual = jit.execute(budget);
        const expected = oracle.execute(budget);
        if(compare_instructions)
        {
            assert.deepEqual(actual, expected, `${label}: interpreter oracle`);
        }
        else
        {
            const { instructions: ai, ...as_ } = actual;
            const { instructions: ei, ...es } = expected;
            assert.deepEqual(as_, es, `${label}: interpreter state`);
        }
        const a = jit.full_snapshot();
        const e = oracle.full_snapshot();
        // Cached split blocks need not publish the previous IP at a
        // successful budget exit (same allowance as the budget test).
        const strip = (s, reason) => reason === 1 ? s.state.filter((_, i) => i !== 11) : s.state;
        assert.deepEqual(
            { state: strip(a, actual.reason), memory: a.memory, dirty: a.dirty,
                counter: a.counter, retired: a.jit + a.interpreted },
            { state: strip(e, actual.reason), memory: e.memory, dirty: e.dirty,
                counter: e.counter, retired: e.jit + e.interpreted },
            `${label}: full CPU state, guest memory, dirty pages and retired total`);
        return actual;
    }
    return { jit, oracle, compare, bail_count, sites };
}

// ---------------------------------------------------------------------------
// Knob off: compiled loops touching denied pages go through the existing slow
// helper. This path must stay bit-identical to the guard-exit-off shape.
// ---------------------------------------------------------------------------
for(const chaining of [false, true])
{
    // inc dword [edi+8]; mov eax,[edi]; mov [edi+4],eax; jmp back
    const code = loop([0xFF, 0x47, 0x08, 0x8B, 0x07, 0x89, 0x47, 0x04]);
    const p = make(code, { guard_exit: false, chaining });
    p.compare(1 << 20, `${chaining}: knob off warm`);
    set_perm(p.jit, V + DATA, 0); // deny all accesses in the permission table
    set_perm(p.oracle, V + DATA, 0);
    const before = p.bail_count();
    p.compare(1 << 18, `${chaining}: knob off denied access`);
    assert.equal(p.bail_count() - before, 0, "knob off must never guard-bail");
    assert(p.jit.counts.jit_slow_reads > 0,
        "knob off must still reach the slow helpers");
}

// ---------------------------------------------------------------------------
// Knob on: denied common-memory accesses bail to the interpreter; state,
// memory and retired totals stay identical to the oracle.
// ---------------------------------------------------------------------------
for(const chaining of [false, true])
{
    // read bail: mov eax,[edi]; add ebx,eax; mov [edi+4],eax
    const code = loop([0x8B, 0x07, 0x01, 0xD8, 0x89, 0x47, 0x04]);
    const p = make(code, { guard_exit: true, chaining });
    p.compare(1 << 20, `${chaining}: read warm`);
    assert(p.jit.counts.entries > 0, "warm-up must execute compiled code");
    const sites = p.sites();
    assert(sites === null || sites.bail > 0, "guards must be emitted with the bail arm");

    // Deny the read on the data page; the real PTE stays readable, so the
    // interpreter performs the access exactly.
    set_perm(p.jit, V + DATA, 2);
    set_perm(p.oracle, V + DATA, 2); // inert for the interpreter, keeps parity
    const before = p.bail_count();
    const slow_before = p.jit.counts.jit_slow_reads;
    p.compare(1 << 18, `${chaining}: read bail`);
    assert(p.bail_count() - before > 0, "failed read guard must bail");
    assert.equal(p.jit.counts.jit_slow_reads, slow_before,
        "the bailing read must not call the slow helper");
    console.log(`${chaining}: read bail ${p.bail_count() - before} times`);
}

{
    // write bail: mov [edi+4],eax; inc eax; inc dword [edi]
    const code = loop([0x89, 0x47, 0x04, 0x40, 0xFF, 0x07]);
    const p = make(code, { guard_exit: true });
    p.compare(1 << 20, "write warm");
    set_perm(p.jit, V + DATA, 1); // runtime table denies writes, PTE still allows
    set_perm(p.oracle, V + DATA, 1);
    const before = p.bail_count();
    p.compare(1 << 18, "write and read-modify-write bail");
    assert(p.bail_count() - before > 0, "failed write guard must bail");
    assert.equal(p.jit.physical.getUint32(DATA + 4, true),
        p.oracle.physical.getUint32(DATA + 4, true), "written bytes identical");
    assert.equal(p.jit.physical.getUint32(DATA, true),
        p.oracle.physical.getUint32(DATA, true), "rmw bytes identical");
    console.log(`write/rmw bail ${p.bail_count() - before} times`);
}

{
    // page fault: runtime table and the PTE both deny the access; the
    // interpreter raises #PF exactly as it does for uncompiled code.
    const code = loop([0x8B, 0x07, 0x8B, 0x5F, 0x04]);
    for(const guard_exit of [false, true])
    {
        const p = make(code, { guard_exit });
        p.compare(1 << 20, `fault warm (${guard_exit})`);
        set_perm(p.jit, V + DATA, 0);
        set_perm(p.oracle, V + DATA, 0);
        for(const c of [p.jit, p.oracle]) pte(c, DATA, 0);
        // Resume at the compiled loop entry so the JIT side reaches the denied
        // access in generated code and bails into the faulting interpreter.
        for(const c of [p.jit, p.oracle]) c.resume(0, A);
        const actual = p.jit.execute(64);
        const expected = p.oracle.execute(64);
        assert.equal(actual.reason, 0x10E, `fault reason (${guard_exit})`);
        assert.equal(expected.reason, 0x10E);
        const { instructions: ai, ...as_ } = actual;
        const { instructions: ei, ...es } = expected;
        assert.deepEqual(as_, es, `fault state (${guard_exit})`);
        if(guard_exit) assert(p.bail_count() > 0, "faulting access must bail first");
    }
}

{
    // Budget stepping: every slice size must keep retired accounting exact on
    // both knobs, including slices that end immediately after a bail.
    const code = loop([0x8B, 0x07, 0x89, 0x47, 0x04, 0x40]);
    for(const guard_exit of [false, true])
    {
        const p = make(code, { guard_exit });
        p.compare(1 << 20, `budget warm (${guard_exit})`);
        set_perm(p.jit, V + DATA, 0);
        set_perm(p.oracle, V + DATA, 0);
        for(const budget of [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 64, 255, 4096, 131072])
            p.compare(budget, `budget ${budget} (${guard_exit})`);
    }
}

{
    // Bail on the first instruction of a compiled block: the loop entry is a
    // JIT entry point, so denying the page bails on its first instruction and
    // exercises the zero-retired-delta path.
    const code = loop([0x8B, 0x47, 0x08, 0x89, 0x47, 0x04, 0x40]);
    const p = make(code, { guard_exit: true });
    p.compare(1 << 20, "block-head warm");
    p.compare(1 << 16, "block-head compile");
    set_perm(p.jit, V + DATA, 0);
    set_perm(p.oracle, V + DATA, 0);
    const before = p.bail_count();
    p.compare(1 << 16, "block-head bail");
    assert(p.bail_count() - before > 0, "first-instruction guard must bail");
}

{
    // Bails must not register an entry point or add heat: the number of
    // compiled modules stays constant no matter how often guards bail.
    const code = loop([0x8B, 0x07, 0x89, 0x47, 0x04]);
    const p = make(code, { guard_exit: true });
    p.compare(1 << 20, "no-recompile warm");
    set_perm(p.jit, V + DATA, 0);
    set_perm(p.oracle, V + DATA, 0);
    const modules = p.jit.counts.modules;
    const before = p.bail_count();
    p.compare(1 << 18, "no-recompile bail");
    assert(p.bail_count() - before > 0, "non-vacuous bail run");
    assert.equal(p.jit.counts.modules, modules,
        "guard bails must not grow the compile count");
    console.log(`no-recompile: bail ${p.bail_count() - before} times, ` +
        `modules ${modules} -> ${p.jit.counts.modules}`);
}

{
    // Second-access bails: the instruction's first memory access passes its
    // guard and a later one fails; the interpreter re-executes the entire
    // instruction, so earlier accesses must be idempotent under re-execution.
    const SRC = 0x7000;
    const STACK_PAGE = 0x8f000; // esp - 4 lives here (esp = V + 0x90000)
    const esp_init = [0xBC, ...u32(V + 0x90000)]; // mov esp, V + 0x90000
    const cases = [
        // pop dword [edi]: stack read passes, write to denied edi fails;
        // esp is reloaded each iteration so the stack stays in place
        ["pop [edi]", loop([...esp_init, 0x8F, 0x07]), V + DATA, 1],
        // push dword [edi]: read of denied edi fails before the stack write
        ["push [edi]", loop([...esp_init, 0xFF, 0x37]), V + DATA, 2],
        // xchg eax, [edi]: read passes, write to denied edi fails
        ["xchg eax, [edi]", loop([0x87, 0x07]), V + DATA, 1],
        // add [edi], eax: read passes, write to denied edi fails
        ["add [edi], eax", loop([0x01, 0x07]), V + DATA, 1],
        // movsd: esi reloaded each iteration; write to denied edi fails
        ["movsd", loop([0xBF, ...u32(V + DATA), 0xBE, ...u32(V + SRC), 0xA5]), V + DATA, 1],
        // call dword [edi]: target read passes, return-address push into the
        // denied stack page fails second
        ["call [edi]", loop([...esp_init, 0xFF, 0x17]), V + STACK_PAGE, 1],
    ];
    for(const [name, code, denied_page, denied_perm] of cases)
    {
        const p = make(code, { guard_exit: true });
        // call [edi] needs a valid target at DATA
        p.jit.physical.setUint32(DATA, V + A, true);
        p.oracle.physical.setUint32(DATA, V + A, true);
        p.compare(1 << 20, `${name} warm`);
        set_perm(p.jit, denied_page, denied_perm);
        set_perm(p.oracle, denied_page, denied_perm);
        const before = p.bail_count();
        p.compare(1 << 15, `${name} denied access`);
        console.log(`${name}: bail ${p.bail_count() - before} times, sites ${JSON.stringify(p.sites())}`);
    }
}

{
    // Accesses spanning a page boundary into a denied neighbour page: compare
    // with the oracle whether the site bails or the slow helper handles it.
    // mov eax, [edi]; mov [edi+4], eax with edi = DATA + 0xFFE straddles
    // pages 0x8000/0x9000.
    const code = loop([0x8B, 0x07, 0x89, 0x47, 0x04]);
    const p = make(code, { guard_exit: true });
    p.jit.set_register(9, V + DATA + 0xFFE);
    p.oracle.set_register(9, V + DATA + 0xFFE);
    p.compare(1 << 18, "crossing warm");
    set_perm(p.jit, V + DATA + 0x1000, 0);
    set_perm(p.oracle, V + DATA + 0x1000, 0);
    const before = p.bail_count();
    p.compare(1 << 16, "crossing denied neighbour");
    console.log(`crossing: bail ${p.bail_count() - before} times`);
}

{
    // Write to a read-only page: the runtime table allows the write but the
    // PTE is read-only; the guard bails and the interpreter raises #PF.
    const code = loop([0x89, 0x07]); // mov [edi], eax
    for(const guard_exit of [false, true])
    {
        const p = make(code, { guard_exit });
        p.compare(1 << 20, `ro-write warm (${guard_exit})`);
        set_perm(p.jit, V + DATA, 1); // runtime table denies writes
        set_perm(p.oracle, V + DATA, 1);
        for(const c of [p.jit, p.oracle]) pte(c, DATA, DATA | 5); // P|US, no RW
        for(const c of [p.jit, p.oracle]) c.resume(0, A);
        const actual = p.jit.execute(64);
        const expected = p.oracle.execute(64);
        assert.equal(actual.reason, 0x10E, `ro write fault reason (${guard_exit})`);
        assert.equal(expected.reason, 0x10E);
        const { instructions: ai, ...as_ } = actual;
        const { instructions: ei, ...es_ } = expected;
        assert.deepEqual(as_, es_, `ro write fault state (${guard_exit})`);
        if(guard_exit) assert(p.bail_count() > 0, "ro write must bail before faulting");
    }
}

{
    // Self-modifying code: the loop patches a byte in its own compiled page.
    // Whether the code-page write traps through the guard (bail) or another
    // mechanism, the executed result must match the interpreter oracle.
    // mov byte [esi], 0x40 (inc eax) | nop <- patched | jmp back
    const code = loop([0xC6, 0x06, 0x40, 0x90]);
    for(const guard_exit of [false, true])
    {
        const p = make(code, { guard_exit });
        p.jit.set_register(8, V + A + 3); // esi -> the nop byte
        p.oracle.set_register(8, V + A + 3);
        p.compare(1 << 20, `smc warm (${guard_exit})`);
        const before = p.bail_count();
        const modules = p.jit.counts.modules;
        p.compare(1 << 16, `smc bail (${guard_exit})`);
        console.log(`smc (${guard_exit}): bail ${p.bail_count() - before}, ` +
            `modules ${modules} -> ${p.jit.counts.modules}`);
    }
}

{
    // Deopt (config 16): the guards of a page that bail past the limit get
    // the page invalidated once; its next compile keeps the rejoining
    // slow-helper shape and results still match the oracle. A second page
    // in the same run keeps its guard exits.
    //   A+0:  mov ecx, 300000 | mov edx, V+A+0x1000
    //   loopA (A+10): mov eax,[edi]; mov [edi+4],eax; dec ecx; jnz loopA
    //   A+18: jmp edx                      (indirect: page B stays a
    //                                     separate module)
    //   loopB (A+0x1000): mov eax,[esi]; mov [esi+4],eax; jmp loopB
    const DATA2 = 0xA000;
    const code = Uint8Array.from([
        0xB9, ...u32(300000),
        0xBA, ...u32(V + A + 0x1000),
        0x8B, 0x07, 0x89, 0x47, 0x04, 0x49, 0x75, 0xF8, 0xFF, 0xE2,
        ...new Array(0x1000 - 20).fill(0x90),
        0x8B, 0x06, 0x89, 0x46, 0x04, 0xEB, 0xF9,
    ]);
    const p = make(code, { guard_exit: true });
    p.jit.wasm.set_jit_config(16, 64);
    p.oracle.wasm.set_jit_config(16, 64);
    p.jit.set_register(8, V + DATA2); // esi
    p.oracle.set_register(8, V + DATA2);
    p.compare(1 << 20, "deopt warm");
    // Compile the second page's module too, then restart loopA.
    for(const c of [p.jit, p.oracle]) c.resume(0, A + 0x1000);
    p.compare(1 << 20, "deopt page B warm");
    for(const c of [p.jit, p.oracle]) c.resume(0, A);
    const modules = p.jit.counts.modules;
    const slow_before =
        p.jit.counts.jit_slow_reads + p.jit.counts.jit_slow_writes;

    set_perm(p.jit, V + DATA, 0);
    set_perm(p.oracle, V + DATA, 0);
    const before = p.bail_count();
    // Each budget slice gets at most a couple of bails: after one the
    // interpreter runs the rest of the slice. Slice the run so page A's
    // guards bail until the limit, then the page is invalidated.
    for(let i = 0; i < 80; i++) p.compare(8, "deopt page A");
    const bails_a = p.bail_count() - before;
    assert.equal(bails_a, 64,
        `page A bails stop at the limit, got ${bails_a}`);

    // The invalidated page goes hot in the interpreter and recompiles once,
    // with the rejoining slow-helper shape.
    p.compare(1 << 21, "deopt recompile");
    assert.equal(p.jit.counts.modules, modules + 1,
        `exactly one recompile of the bailing page (${modules} -> ${p.jit.counts.modules})`);
    assert(
        p.jit.counts.jit_slow_reads + p.jit.counts.jit_slow_writes > slow_before,
        "recompiled page uses the slow-helper path");

    // Deny the second data page and let loopB bail without reaching the
    // limit: its page keeps the guard-exit shape.
    set_perm(p.jit, V + DATA2, 0);
    set_perm(p.oracle, V + DATA2, 0);
    const before2 = p.bail_count();
    const modules2 = p.jit.counts.modules;
    for(let i = 0; i < 40; i++) p.compare(8, "deopt page B keeps guard exits");
    assert(p.bail_count() - before2 > 0,
        "the second page keeps its guard exits");
    assert.equal(p.jit.counts.modules, modules2,
        "the second page is not deopted");
    console.log(`deopt: page A bails ${bails_a}, page B bails ${p.bail_count() - before2}`);
}

console.log("guard-exit: knob off/on common-memory bails match the interpreter oracle");
