#!/usr/bin/env node

// Instructions that end at or cross a page boundary are stepped through
// vine_jit_boundary_step as a block-final instruction instead of bouncing
// to the interpreter (set_jit_config 17, default on).
// node tests/api/vine-jit-boundary.js ; V86_WASM selects another build.
import assert from "node:assert/strict";
import fs from "node:fs";

const wasm_path = process.env.V86_WASM || new URL("../../build/v86-fallback.wasm", import.meta.url);
const module = new WebAssembly.Module(fs.readFileSync(wasm_path));
const START = 0x1000;
const DATA = 0x8000;
const MEMORY_SIZE = 1024 * 1024;
const TABLE_OFFSET = 1024;
const CHAIN_OFFSET = TABLE_OFFSET + 900;
const ITERATIONS = 200;
const CHAINING = process.env.V86_TEST_CHAINING === "1";

// A native tail wrapper counts chained entries without adding a stack frame
// (same import shape as vine-jit-budget.js).
const section = (id, bytes) => [id, bytes.length, ...bytes];
const chain_body = [0, 0x10, 1, ...Array.from({ length: 11 }, (_, i) => [0x20, i]).flat(), 0x12, 0, 11];
const chain_wrapper = new WebAssembly.Module(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0,
    ...section(1, [2, 0x60, 11, ...Array(11).fill(0x7f), 0, 0x60, 0, 0]),
    ...section(2, [2, 1, 101, 1, 103, 0, 0, 1, 101, 5, 101, 110, 116, 101, 114, 0, 1]),
    ...section(3, [1, 0]), ...section(7, [1, 1, 103, 0, 2]),
    ...section(10, [1, chain_body.length, ...chain_body]),
]));

function create_cpu(code, { disable_jit = false, boundary = null, eip = null, ecx = null,
    pages = 3, chaining = CHAINING, // multi-page programs: the non-Vine default kept explicit
    cg = process.env.V86_TEST_CG === undefined ? null : Number(process.env.V86_TEST_CG) } = {})
{
    const table = new WebAssembly.Table({ element: "anyfunc", initial: CHAIN_OFFSET + 900 });
    const counts = { chain_entries: 0 };
    let wasm;
    let jit_imports;
    const env = { __indirect_function_table: table };
    for(const imported of WebAssembly.Module.imports(module))
    {
        assert.equal(imported.module, "env");
        if(imported.kind === "table") continue;
        assert.equal(imported.kind, "function");
        env[imported.name] = (...args) => {
            switch(imported.name)
            {
                case "microtick": case "get_rand_int": case "cpu_event_halt":
                case "stop_idling": case "run_hardware_timers":
                case "io_port_read8": case "io_port_read16": case "io_port_read32":
                case "io_port_write8": case "io_port_write16": case "io_port_write32":
                case "mmap_read8": case "mmap_read32": case "mmap_write8":
                case "mmap_write16": case "mmap_write32": case "mmap_write64":
                case "mmap_write128":
                    return 0;
                case "jit_clear_func":
                    table.set(TABLE_OFFSET + args[0], null);
                    table.set(CHAIN_OFFSET + args[0], null);
                    return;
                case "cpu_exception_hook": case "dbg_trace_from_wasm": case "log_from_wasm":
                    return 0;
                case "console_log_from_wasm":
                    throw new Error(new TextDecoder().decode(
                        new Uint8Array(wasm.memory.buffer, args[0], args[1])));
                case "codegen_finalize": {
                    const [index, _start, _state_flags, pointer, length] = args;
                    const generated = new WebAssembly.Module(
                        Buffer.from(wasm.memory.buffer, pointer, length));
                    const { f, g } = new WebAssembly.Instance(generated, { e: jit_imports }).exports;
                    table.set(TABLE_OFFSET + index, f);
                    table.set(CHAIN_OFFSET + index, g === undefined ? null :
                        new WebAssembly.Instance(chain_wrapper, {
                            e: { g, enter: () => counts.chain_entries++ },
                        }).exports.g);
                    return;
                }
                default:
                    throw new Error(`Unexpected host import ${imported.name}`);
            }
        };
    }
    wasm = new WebAssembly.Instance(module, { env }).exports;
    jit_imports = { ...wasm, m: wasm.memory, t: table };
    wasm.rust_init();
    if(typeof wasm.vine_set_jit_chaining === "function")
        assert.equal(wasm.vine_set_jit_chaining(Number(chaining)), 0);
    assert.equal(wasm.vine_set_memory_size(MEMORY_SIZE), 0);
    const physical_base = wasm.allocate_memory(MEMORY_SIZE);
    wasm.reset_cpu();
    wasm.set_jit_config(0, Number(disable_jit));
    if(boundary !== null) wasm.set_jit_config(17, boundary);
    if(cg !== null) wasm.set_jit_config(18, cg);
    wasm.set_jit_config(4, 1);
    wasm.set_jit_config(5, 1);
    wasm.set_jit_config(6, 1);
    if(pages !== null) wasm.set_jit_config(1, pages);
    new Uint8Array(wasm.memory.buffer, physical_base + START, code.length).set(code);

    wasm.vine_cpu_state_save();
    const state = new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153);
    state[6] = 0x90000; // esp
    if(ecx !== null) state[3] = ecx;
    state[9] = DATA; // edi
    state[10] = state[11] = eip ?? START;
    state[19] = state[20] = state[21] = 1;
    state.fill(0, 34, 42);
    state.fill(0xFFFFFFFF, 42, 50);
    state[58] = 1;
    assert.equal(wasm.vine_cpu_state_restore(), 0);

    function retired(prefix)
    {
        return BigInt(wasm[`${prefix}_high`]() >>> 0) << 32n |
            BigInt(wasm[`${prefix}_low`]() >>> 0);
    }

    return {
        wasm,
        get physical() { return new DataView(wasm.memory.buffer, physical_base, MEMORY_SIZE); },
        get steps() { return wasm.vine_jit_boundary_step_count(); },
        get interpreted() { return retired("vine_interpreted_retired_instructions"); },
        get compiled() { return retired("vine_jit_retired_instructions"); },
        get chain_entries() { return counts.chain_entries; },
        execute(budget) {
            const reason = wasm.vine_execute_budget(budget);
            wasm.vine_cpu_state_save();
            const saved = new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153);
            return {
                reason,
                instructions: wasm.vine_instruction_counter() >>> 0,
                registers: [...saved.slice(2, 10)],
                eip: saved[10],
                flags: wasm.get_eflags() >>> 0,
                data: [...new Uint8Array(wasm.memory.buffer, physical_base + DATA, 64)],
            };
        },
    };
}

// --- program builder -------------------------------------------------------
// 0x1000: jmp 0x1F00; 0x1F00: mov ecx,ITER (once); NOPs to 0x1FF0; LOOP:
// inc dword [edi] at 0x1FF0, NOP fill to `boundary_off` (>=0x1FF6), the
// boundary instruction, then the tail on 0x2000+: dec ecx; jnz rel32 0x1FF0;
// hlt. Loop body ~20 instructions.
const LOOP = 0x1FF0;

function make_program(boundary_off, boundary_bytes, tail_bytes = null, pre = [])
{
    const code = new Uint8Array(0x4100 - START);
    const put = (addr, bytes) => code.set(bytes, addr - START);
    put(START, [0xE9, 0xFB, 0x0E, 0, 0]);       // jmp 0x1F00
    put(0x1F00, [0xB9, ITERATIONS & 0xFF, ITERATIONS >> 8, 0, 0]); // mov ecx,ITER
    put(0x1F05, pre.length ? pre : [0x90]);
    for(let a = 0x1F05 + Math.max(1, pre.length); a < LOOP; a++) put(a, [0x90]);
    let p = LOOP;
    put(p, [0xFF, 0x07]); p += 2;              // inc dword [edi] (iteration count)
    for(let a = p; a < boundary_off; a++) put(a, [0x90]); // NOP fill
    put(boundary_off, boundary_bytes);
    const tail = tail_bytes || default_tail(boundary_off + boundary_bytes.length);
    put(boundary_off + boundary_bytes.length, tail);
    return code;
}

// default tail starting at `at`: dec ecx; jnz rel32 -> LOOP; hlt
function default_tail(at)
{
    const jz_at = at + 1, after = jz_at + 6;
    const rel = LOOP - after;
    return [0x49, 0x0F, 0x85, rel & 0xFF, (rel >> 8) & 0xFF, (rel >> 16) & 0xFF,
        (rel >> 24) & 0xFF, 0xF4];
}

// Run one case on three cpus: jit+steps (17=1), jit without (17=0), interpreter.
function run_case(name, code, { expect_steps = true, cg = null } = {})
{
    const jit1 = create_cpu(code, { boundary: 1, cg });
    const jit0 = create_cpu(code, { boundary: 0, cg });
    const interp = create_cpu(code, { disable_jit: true });
    const strip = o => ({ ...o, instructions: undefined });
    for(let i = 0; i < 200; i++)
    {
        const a = jit1.execute(97);
        const b = jit0.execute(97);
        const c = interp.execute(97);
        assert.deepEqual(strip(a), strip(b), `${name}: 17=1 vs 17=0 slice ${i}`);
        assert.deepEqual(strip(a), strip(c), `${name}: 17=1 vs interp slice ${i}`);
        if(a.reason !== 1) break;
    }
    const f1 = jit1.execute(1e9), f0 = jit0.execute(1e9), fi = interp.execute(1e9);
    assert.deepEqual(strip(f1), strip(f0), `${name}: final 17=1 vs 17=0`);
    assert.deepEqual(strip(f1), strip(fi), `${name}: final vs interp`);
    if(expect_steps)
    {
        assert.ok(jit1.steps > 0, `${name}: boundary steps counted (got ${jit1.steps})`);
        assert.equal(jit0.steps, 0, `${name}: 17=0 performs no steps`);
    }
    return { jit1, jit0, interp };
}

// --- class E: last byte is the last byte of the page ------------------------
run_case("E: mov eax,imm32 ending at 0xFFF",
    make_program(0x1FFB, [0xB8, 0x78, 0x56, 0x34, 0x12]));

run_case("E: add [mem],imm8 ending at 0xFFF",
    make_program(0x1FF9, [0x80, 0x05, DATA & 0xFF, DATA >> 8, 0, 0, 0x2A]));

// E-class terminators are stepped too: `call rel32` at 0x1FF6 calls a lone
// `ret` at 0x1FFF (a step-only block entry: short_leaf rejects near-end so it
// is never inlined). The call returns to 0x1FFB, which jumps over the callee
// to the tail at 0x2000.
run_case("E: call rel32 + ret at 0xFFF (stepped call and step-only ret)",
    make_program(0x1FF6, [0xE8, 0x04, 0, 0, 0, 0xEB, 0x03, 0x90, 0x90, 0xC3],
        default_tail(0x2000)));

run_case("E: jmp rel8 at 0xFFE..0xFFF into next page",
    make_program(0x1FFE, [0xEB, 0x01],
        [0x90, ...default_tail(0x2001)])); // jmp +1 -> 0x2001

run_case("E: call rel32 at 0xFFB..0xFFF + callee on next page",
    (() => {
        const code = make_program(0x1FFB, [0xE8, 0x0B, 0, 0, 0], default_tail(0x2000));
        code.set([0xB8, 0x11, 0x22, 0x33, 0x44, 0xC3], 0x200B - START);
        return code;
    })());

// --- class S: instruction bytes straddle the boundary -----------------------
for(const off of [0x1FFC, 0x1FFD, 0x1FFE, 0x1FFF])
{
    run_case(`S: mov eax,imm32 split at 0x${off.toString(16)}`,
        make_program(off, [0xB8, 0xEF, 0xBE, 0xAD, 0xDE]));
}

// jcc rel32 straddling: flag setup inside the page, 6-byte jz at 0x1FFC, rel32=0
// so taken and not-taken both land on the tail at 0x2002.
for(const [name, flag_bytes] of [["taken", [0x31, 0xC0]], ["not-taken", [0x0C, 0x01]]])
{
    run_case(`S: jz rel32 ${name} straddling`,
        make_program(0x1FFA, [...flag_bytes, 0x0F, 0x84, 0x00, 0, 0, 0],
            default_tail(0x2002)));
}

run_case("S: call rel32 straddling + callee on next page",
    (() => {
        const code = make_program(0x1FFD, [0xE8, 0x09, 0, 0, 0], default_tail(0x2002));
        code.set([0xB8, 0x99, 0x88, 0x77, 0x66, 0xC3], 0x200B - START);
        return code;
    })());

run_case("S: jmp rel32 straddling into next page",
    // jmp at 0x1FFD -> 0x2005; NOPs pad so the tail physically starts there.
    make_program(0x1FFD, [0xE9, 0x03, 0, 0, 0, 0x90, 0x90, 0x90],
        default_tail(0x2005)));

// ModRM+SIB+disp32: mov eax,[ebx*1+edi+disp32]; disp32 straddles the boundary.
// Encoding 8B 84 1D <disp32>; ebx=0, edi=DATA -> reads [DATA].
run_case("S: mov eax,[ebx+edi+disp32] disp32 straddling",
    make_program(0x1FFB, [0x8B, 0x84, 0x1D, 0x00, 0x00, 0x00, 0x00]));

// prefixed instruction: 66 B8 imm16 — prefix byte in P, opcode byte in P+1
run_case("S: mov ax,imm16 prefix/opcode split",
    make_program(0x1FFF, [0x66, 0xB8, 0x34, 0x12]));

// rep movsb straddling: F3 at 0x1FFF, A4 on the next page; each stepped
// iteration copies 8 bytes (ecx is preset inside the loop body).
// rep movsb straddling: F3 at 0x1FFF, A4 on the next page. ecx is re-armed
// inside the loop body (rep consumes it); the outer counter is edx.
run_case("S: rep movsb straddling",
    (() => {
        const code = new Uint8Array(0x4100 - START);
        const put = (addr, bytes) => code.set(bytes, addr - START);
        put(START, [0xE9, 0xFB, 0x0E, 0, 0]);            // jmp 0x1F00
        put(0x1F00, [0xBA, ITERATIONS & 0xFF, ITERATIONS >> 8, 0, 0]); // mov edx,ITER
        put(0x1F05, [0xBE, (DATA + 32) & 0xFF, (DATA + 32) >> 8, 0, 0]); // mov esi
        for(let a = 0x1F0A; a < 0x1FF0; a++) put(a, [0x90]);
        put(0x1FF0, [0xFF, 0x07]);                      // inc dword [edi]
        put(0x1FF2, [0xB9, 8, 0, 0, 0]);                // mov ecx,8
        for(let a = 0x1FF7; a < 0x1FFF; a++) put(a, [0x90]);
        put(0x1FFF, [0xF3, 0xA4]);                      // rep movsb straddles
        const rel = LOOP - (0x2001 + 1 + 6);            // jnz at 0x2002 ends 0x2008
        put(0x2001, [0x4A, 0x0F, 0x85, rel & 0xFF, (rel >> 8) & 0xFF,
            (rel >> 16) & 0xFF, (rel >> 24) & 0xFF, 0xF4]); // dec edx; jnz LOOP; hlt
        return code;
    })());

// --- boundary instruction as a compiled call target (step-only entry) -------
// call at 0x1FF6 -> straddling mov eax,imm32 at 0x1FFF (step-only entry),
// which continues to a ret at 0x2004; return addr 0x1FFB jumps to the tail.
run_case("step-only: compiled call lands on a boundary instruction",
    make_program(0x1FF6, [0xE8, 0x04, 0, 0, 0, 0xEB, 0x08, 0x90, 0x90,
        0xB8, 0xAA, 0xBB, 0xCC, 0xDD, 0xC3],
        default_tail(0x2005)));

// --- SMC: host writes to bytes of the hot straddling instruction ------------
// The stepped instruction is decoded at run time, so writes in P or in P+1
// take effect on the next iteration without any invalidation.
{
    const code = make_program(0x1FFC, [0xB8, 0x11, 0x22, 0x33, 0x44]);
    const jit1 = create_cpu(code, { boundary: 1 });
    const jit0 = create_cpu(code, { boundary: 0 });
    const interp = create_cpu(code, { disable_jit: true });
    // warm up only part of the loop so later iterations observe the writes
    for(let i = 0; i < 40; i++) { jit1.execute(64); jit0.execute(64); interp.execute(64); }
    // mov eax,imm32 at 0x1FFC: bytes B8 11 22 33 44 over 0x1FFC..0x2000.
    // Rewrite 0x22->0x55 at 0x1FFE (in P) and 0x44->0x66 at 0x2000 (in P+1):
    // the immediate becomes 11 55 33 66 -> eax = 0x66335511.
    for(const c of [jit1, jit0, interp])
    {
        const mem = new Uint8Array(c.wasm.memory.buffer);
        const base = c.physical.byteOffset;
        mem[base + 0x1FFE] = 0x55;
        mem[base + 0x2000] = 0x66;
    }
    const strip = o => ({ ...o, instructions: undefined });
    const r1 = jit1.execute(1e9), r0 = jit0.execute(1e9), ri = interp.execute(1e9);
    assert.equal(r1.registers[0] & 0xFFFFFFFF, 0x66335511, "SMC: new immediate");
    assert.deepEqual(strip(r1), strip(r0), "SMC: 17=1 vs 17=0");
    assert.deepEqual(strip(r1), strip(ri), "SMC: 17=1 vs interpreter");
}

// --- exact budgets: retire exactly at / around the boundary instruction -----
{
    const code = make_program(0x1FFC, [0xB8, 0x77, 0x66, 0x55, 0x44]);
    const j1 = create_cpu(code, { boundary: 1 });
    const i0 = create_cpu(code, { disable_jit: true });
    const strip = o => ({ ...o, instructions: undefined });
    for(const budget of [9, 11, 1, 1, 1, 1, 2, 17, 97, 97])
    {
        const a = j1.execute(budget), b = i0.execute(budget);
        assert.deepEqual(strip(a), strip(b),
            `exact budget ${budget}: 17=1 jit vs interpreter`);
        assert.deepEqual(a.instructions, b.instructions,
            `exact budget ${budget}: retired count`);
    }
}

// --- retired accounting: total and the split --------------------------------
{
    const code = make_program(0x1FFC, [0xB8, 0x77, 0x66, 0x55, 0x44]);
    const j1 = create_cpu(code, { boundary: 1 });
    const j0 = create_cpu(code, { boundary: 0 });
    const i0 = create_cpu(code, { disable_jit: true });
    j1.execute(1e9); j0.execute(1e9); i0.execute(1e9);
    const total1 = j1.compiled + j1.interpreted;
    const total0 = j0.compiled + j0.interpreted;
    const totali = i0.interpreted;
    assert.equal(total1, total0, "17=1 and 17=0 retire the same total");
    assert.equal(total1, totali, "jit total matches the interpreter total");
    // the split is identical: stepped instructions retire on the interpreted
    // side in both modes; only the step counter distinguishes them
    assert.ok(j1.steps > 0, "17=1 steps the boundary instruction");
    assert.equal(j0.steps, 0, "17=0 performs no steps");
    assert.equal(j1.compiled, j0.compiled, "compiled counts match");
    assert.equal(j1.interpreted, j0.interpreted,
        "interpreted counts match (steps included on both sides)");
}

// --- module entry on a boundary instruction --------------------------------
// Execution starts directly on the straddling mov: the registered entry is a
// step-only block. ecx is preset so the tail loop still terminates.
{
    const code = make_program(0x1FFC, [0xB8, 0x77, 0x66, 0x55, 0x44]);
    const jit1 = create_cpu(code, { boundary: 1, eip: 0x1FFC, ecx: 60 });
    const jit0 = create_cpu(code, { boundary: 0, eip: 0x1FFC, ecx: 60 });
    const interp = create_cpu(code, { disable_jit: true, eip: 0x1FFC, ecx: 60 });
    const strip = o => ({ ...o, instructions: undefined });
    for(let i = 0; i < 40; i++)
    {
        const a = jit1.execute(97), b = jit0.execute(97), c = interp.execute(97);
        assert.deepEqual(strip(a), strip(b), `entry@${i}: 17=1 vs 17=0`);
        assert.deepEqual(strip(a), strip(c), `entry@${i}: 17=1 vs interp`);
        if(a.reason !== 1) break;
    }
    const f1 = jit1.execute(1e9), f0 = jit0.execute(1e9), fi = interp.execute(1e9);
    assert.deepEqual(strip(f1), strip(f0), "entry: final 17=1 vs 17=0");
    assert.deepEqual(strip(f1), strip(fi), "entry: final vs interp");
    assert.ok(jit1.steps > 0, "module entry on a boundary instruction steps");
}

// --- "stays in generated code" observable -----------------------------------
// Interpreter-entry count: with 17=1 the hot boundary instruction is handled
// inside generated code, so after warm-up interpreter entries stop growing;
// with 17=0 every loop iteration enters the interpreter once for it.
// 50k iterations keeps the measurement in steady state past JIT_THRESHOLD.
{
    const code = make_program(0x1FFC, [0xB8, 0x77, 0x66, 0x55, 0x44]);
    code[0x1F01 - START] = 0x50; code[0x1F02 - START] = 0xC3; // mov ecx,50000
    const jit1 = create_cpu(code, { boundary: 1 });
    const jit0 = create_cpu(code, { boundary: 0 });
    for(let i = 0; i < 80; i++) { jit1.execute(4096); jit0.execute(4096); }
    const ent1a = jit1.wasm.vine_interpreted_entry_count();
    const ent0a = jit0.wasm.vine_interpreted_entry_count();
    const st1a = jit1.steps;
    for(let i = 0; i < 50; i++) { jit1.execute(4096); jit0.execute(4096); }
    const d1 = jit1.wasm.vine_interpreted_entry_count() - ent1a;
    const d0 = jit0.wasm.vine_interpreted_entry_count() - ent0a;
    const ds = jit1.steps - st1a;
    assert.ok(ds > 1000, `17=1 steps in steady state (${ds})`);
    assert.ok(d0 > 5000,
        `17=0 enters the interpreter per boundary instruction (${d0})`);
    assert.ok(d1 < d0 / 20,
        `17=1 stays in generated code (entries ${d1} vs ${d0})`);
}

// --- execute protection on P+1 ----------------------------------------------
// Remove execute permission from the second page while the boundary
// instruction is hot: the next fetch on 0x2000 must fault identically in all
// modes (stop reason, error code, cr2 and eip at the failing instruction).
{
    const code = make_program(0x1FFC, [0xB8, 0x77, 0x66, 0x55, 0x44]);
    const jit1 = create_cpu(code, { boundary: 1 });
    const jit0 = create_cpu(code, { boundary: 0 });
    const interp = create_cpu(code, { disable_jit: true });
    for(let i = 0; i < 40; i++)
    {
        jit1.execute(64); jit0.execute(64); interp.execute(64);
    }
    for(const c of [jit1, jit0, interp])
    {
        c.wasm.vine_reset_execute_permissions();
        assert.equal(c.wasm.vine_set_page_executable(0x1000, 1), 0);
    }
    const strip = o => ({ ...o, instructions: undefined, data: undefined });
    const r1 = jit1.execute(1e9), r0 = jit0.execute(1e9), ri = interp.execute(1e9);
    const err1 = jit1.wasm.vine_get_stop_error_code();
    const err0 = jit0.wasm.vine_get_stop_error_code();
    const erri = interp.wasm.vine_get_stop_error_code();
    const state_of = c => [...new Uint32Array(
        c.wasm.memory.buffer, c.wasm.vine_cpu_state_ptr(), 64)];
    assert.equal(r1.reason, ri.reason, "exec-prot: 17=1 stop reason matches interp");
    assert.equal(r0.reason, ri.reason, "exec-prot: 17=0 stop reason matches interp");
    assert.equal(r1.eip, ri.eip, "exec-prot: 17=1 eip at fault matches interp");
    assert.equal(r0.eip, ri.eip, "exec-prot: 17=0 eip at fault matches interp");
    assert.equal(err1, erri, "exec-prot: 17=1 error code matches interp");
    assert.equal(err0, erri, "exec-prot: 17=0 error code matches interp");
    assert.deepEqual(state_of(jit1), state_of(interp), "exec-prot: state 17=1");
    assert.deepEqual(state_of(jit0), state_of(interp), "exec-prot: state 17=0");
}

// --- fault inside a stepped instruction: div by zero ------------------------
// `div ebx` straddles at 0x1FFF (F7 in P, F3 in P+1). ebx is set to ecx-2 in
// the tail so the divisor hits 0 on the last iteration while the instruction
// is stepped.
{
    const code = new Uint8Array(0x4100 - START);
    const put = (addr, bytes) => code.set(bytes, addr - START);
    put(START, [0xE9, 0xFB, 0x0E, 0, 0]);            // jmp 0x1F00
    put(0x1F00, [0xB9, ITERATIONS & 0xFF, ITERATIONS >> 8, 0, 0]); // mov ecx,ITER
    put(0x1F05, [0xBB, 1, 0, 0, 0]);                 // mov ebx,1
    for(let a = 0x1F0A; a < LOOP; a++) put(a, [0x90]);
    put(LOOP, [0xFF, 0x07]);                          // inc dword [edi]
    for(let a = 0x1FF2; a < 0x1FFF; a++) put(a, [0x90]);
    put(0x1FFF, [0xF7, 0xF3]);                        // div ebx straddles
    const rel = LOOP - (0x2001 + 3 + 1 + 6);          // jnz at 0x2005 ends 0x200B
    put(0x2001, [0x8D, 0x59, 0xFE,                    // lea ebx,[ecx-2]
        0x49,                                          // dec ecx
        0x0F, 0x85, rel & 0xFF, (rel >> 8) & 0xFF,
        (rel >> 16) & 0xFF, (rel >> 24) & 0xFF,        // jnz LOOP
        0xF4]);                                        // hlt
    const jit1 = create_cpu(code, { boundary: 1 });
    const jit0 = create_cpu(code, { boundary: 0 });
    const interp = create_cpu(code, { disable_jit: true });
    const r1 = jit1.execute(1e9), r0 = jit0.execute(1e9), ri = interp.execute(1e9);
    assert.equal(r1.reason, ri.reason, "div-fault: 17=1 reason matches interp");
    assert.equal(r0.reason, ri.reason, "div-fault: 17=0 reason matches interp");
    assert.equal(r1.eip, ri.eip, "div-fault: 17=1 eip at fault matches interp");
    assert.equal(r0.eip, ri.eip, "div-fault: 17=0 eip at fault matches interp");
    assert.equal(jit1.wasm.vine_get_stop_error_code(),
        interp.wasm.vine_get_stop_error_code(), "div-fault: error code");
    assert.ok(jit1.steps > 0, "div-fault: the faulting instruction was stepped");
}

// --- exact budgets inside a single generated module -------------------------
// Regression for the limit-local defect: a loop whose only module exit is the
// boundary-step terminator must still retire exactly each requested budget.
// The straddling back-edge jmp at 0x1FFC keeps every iteration inside one
// module, so only the reduced limit stops the run.
// 0x1000: jmp 0x1F00; 0x1F00: dec ecx; jz 0x2010; inc dword [edi];
// jmp 0x1FFC; 0x1FFC: jmp 0x1F00 (straddles); 0x2010: hlt.
function single_module_loop_program()
{
    const code = new Uint8Array(0x2011 - START).fill(0x90);
    const put = (at, bytes) => code.set(bytes, at - START);
    put(0x1000, [0xE9, 0xFB, 0x0E, 0x00, 0x00]);       // jmp 0x1F00
    put(0x1F00, [0x49]);                               // dec ecx
    put(0x1F01, [0x0F, 0x84, 0x09, 0x01, 0x00, 0x00]); // jz 0x2010
    put(0x1F07, [0xFF, 0x07]);                         // inc dword [edi]
    put(0x1F09, [0xE9, 0xEE, 0x00, 0x00, 0x00]);       // jmp 0x1FFC
    put(0x1FFC, [0xE9, 0xFF, 0xFE, 0xFF, 0xFF]);       // jmp 0x1F00 (straddles)
    put(0x2010, [0xF4]);                               // hlt
    return code;
}

// Same loop shape, but the entire loop body between the head and the back
// edge consists of boundary instructions: two step-only blocks on two pages.
// 0x1FFC: jmp 0x2FFC (straddles 0x1FFF); 0x2FFC: jmp 0x1F00 (straddles
// 0x2FFF). The dead `jo 0x2FFC` in the head (OF is always clear here) is only
// there so analysis registers 0x2FFC as a step-only block in the same module;
// it is never taken. Exit via jz -> 0x3010 hlt.
function two_step_only_loop_program()
{
    const code = new Uint8Array(0x3011 - START).fill(0x90);
    const put = (at, bytes) => code.set(bytes, at - START);
    put(0x1000, [0xE9, 0xFB, 0x0E, 0x00, 0x00]);       // jmp 0x1F00
    put(0x1F00, [0x49]);                               // dec ecx
    put(0x1F01, [0x0F, 0x84, 0x09, 0x11, 0x00, 0x00]); // jz 0x3010
    put(0x1F07, [0xFF, 0x07]);                         // inc dword [edi]
    put(0x1F09, [0x0F, 0x80, 0xED, 0x0F, 0x00, 0x00]); // jo 0x2FFC (never taken)
    put(0x1F0F, [0xE9, 0xE8, 0x00, 0x00, 0x00]);       // jmp 0x1FFC
    put(0x1FFC, [0xE9, 0xFB, 0x0F, 0x00, 0x00]);       // jmp 0x2FFC (straddles)
    put(0x2FFC, [0xE9, 0xFF, 0xEE, 0xFF, 0xFF]);       // jmp 0x1F00 (straddles)
    put(0x3010, [0xF4]);                               // hlt
    return code;
}

// Two modules chained: MAX_PAGES=1 keeps the page-0x2000 tail out of the
// first module. The straddling back-edge jmp steps in module 1, the tail
// module's jmp chains back to the loop head entry.
// 0x1000: jmp 0x1F00; 0x1F00: dec ecx; jz 0x2010; inc dword [edi];
// jmp 0x1FFC; 0x1FFC: jmp 0x2000 (straddles); 0x2000: jmp 0x1F00; 0x2010: hlt.
function step_chain_loop_program()
{
    const code = new Uint8Array(0x2011 - START).fill(0x90);
    const put = (at, bytes) => code.set(bytes, at - START);
    put(0x1000, [0xE9, 0xFB, 0x0E, 0x00, 0x00]);       // jmp 0x1F00
    put(0x1F00, [0x49]);                               // dec ecx
    put(0x1F01, [0x0F, 0x84, 0x09, 0x01, 0x00, 0x00]); // jz 0x2010
    put(0x1F07, [0xFF, 0x07]);                         // inc dword [edi]
    put(0x1F09, [0xE9, 0xEE, 0x00, 0x00, 0x00]);       // jmp 0x1FFC
    put(0x1FFC, [0xE9, 0x04, 0x00, 0x00, 0x00]);       // jmp 0x2005 (straddles)
    put(0x2005, [0xE9, 0xF6, 0xFE, 0xFF, 0xFF]);       // jmp 0x1F00
    put(0x2010, [0xF4]);                               // hlt
    return code;
}

// Run fixed budget slices; each slice must retire exactly its budget (or the
// cpu stopped early on a fault/hlt). Compares 17=1 vs 17=0 vs interpreter on
// registers, eip, flags, memory and cumulative retired count per slice, then
// checks the compiled+interpreted accounting across modes.
function budget_case(name, code, { budgets, ecx = 3000000, pages = null,
    expect_steps = true, expect_chain = false, cg = null } = {})
{
    const opts = { ecx, cg };
    if(pages !== null) opts.pages = pages;
    const jit1 = create_cpu(code, { ...opts, boundary: 1 });
    const jit0 = create_cpu(code, { ...opts, boundary: 0 });
    const interp = create_cpu(code, { ...opts, disable_jit: true });
    const strip = o => ({ ...o, instructions: undefined });
    const last = new Map([[jit1, 0], [jit0, 0], [interp, 0]]);
    for(let i = 0; i < budgets.length; i++)
    {
        const rs = [jit1, jit0, interp].map(c => c.execute(budgets[i]));
        for(const [cpu, r] of [[jit1, rs[0]], [jit0, rs[1]], [interp, rs[2]]])
        {
            const delta = r.instructions - last.get(cpu);
            last.set(cpu, r.instructions);
            assert.ok(delta === budgets[i] || r.reason !== 1,
                `${name}: slice ${i} retired ${delta}, expected ${budgets[i]}` +
                ` (reason ${r.reason}, mode ${cpu === jit1 ? "17=1" : cpu === jit0 ? "17=0" : "interp"})`);
        }
        assert.deepEqual(strip(rs[0]), strip(rs[1]),
            `${name}: 17=1 vs 17=0 slice ${i}`);
        assert.deepEqual(strip(rs[0]), strip(rs[2]),
            `${name}: 17=1 vs interp slice ${i}`);
        assert.equal(rs[0].instructions, rs[1].instructions,
            `${name}: retired count 17=1 vs 17=0 slice ${i}`);
        if(rs[0].reason !== 1) break;
    }
    const total1 = jit1.compiled + jit1.interpreted;
    const total0 = jit0.compiled + jit0.interpreted;
    const totali = interp.interpreted;
    assert.equal(total1, total0, `${name}: 17=1/17=0 retired totals match`);
    assert.equal(total1, totali, `${name}: jit total matches interpreter`);
    if(expect_steps)
    {
        assert.ok(jit1.steps > 0, `${name}: boundary steps counted (got ${jit1.steps})`);
        assert.equal(jit0.steps, 0, `${name}: 17=0 performs no steps`);
    }
    if(expect_chain)
    {
        assert.ok(jit1.chain_entries > 0,
            `${name}: chain hop used (got ${jit1.chain_entries})`);
    }
    return { jit1, jit0, interp };
}

// Regression for the AbsoluteEip arm leaking one wasm local per
// ret/indirect-jump block: a module with more ret blocks than the 128-local
// budget panicked at generation time ("jit local index overflow"). 160
// callees of >MAX_LEAF_INSNS instructions each stay non-inlinable, so the
// page's module contains 160 real AbsoluteEip blocks.
function many_ret_blocks_program()
{
    const N = 160;
    const code = new Uint8Array(0x2000 - START);
    for(let i = 0; i < N; i++)
    {
        const at = START + i * 5;
        const callee = 0x1400 + i * 16;
        const rel = callee - (at + 5);
        code.set([0xE8, rel & 0xFF, rel >> 8 & 0xFF, rel >> 16 & 0xFF,
            rel >> 24 & 0xFF], at - START);
        code.set([0x40, 0x48, 0x40, 0x48, 0x40, 0x48, 0x40, 0x48,
            0x40, 0x48, 0x40, 0x48, 0x40, 0x48, 0x90, 0xC3], callee - START);
    }
    code[N * 5] = 0xF4; // hlt after the last call returns
    return code;
}
for(const cg of [0, 1, 2, 3])
    run_case(`160 AbsoluteEip ret blocks in one module (config 18 = ${cg})`,
        many_ret_blocks_program(), { expect_steps: false, cg });

const BUDGETS_A = [2000, 2000, 2000, 333, 7, 50000];
const BUDGETS_B = [1, 1, 1, 2, 3, 97, 4096, 4097, 131072];

budget_case("in-module budget loop (list A)", single_module_loop_program(),
    { budgets: BUDGETS_A });
budget_case("in-module budget loop (list B)", single_module_loop_program(),
    { budgets: BUDGETS_B });
budget_case("two step-only blocks (list A)", two_step_only_loop_program(),
    { budgets: BUDGETS_A });
budget_case("two step-only blocks (list B)", two_step_only_loop_program(),
    { budgets: BUDGETS_B });
budget_case("step+chain across two modules (list A)", step_chain_loop_program(),
    { budgets: BUDGETS_A, pages: 1, expect_chain: CHAINING });
budget_case("step+chain across two modules (list B)", step_chain_loop_program(),
    { budgets: BUDGETS_B, pages: 1, expect_chain: CHAINING });

console.log(`vine-jit-boundary: all cases passed (chaining=${CHAINING})`);
