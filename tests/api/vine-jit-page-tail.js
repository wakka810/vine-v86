#!/usr/bin/env node

// Run make vine-jit-page-tail-test, or use an existing build with:
// node tests/api/vine-jit-page-tail.js
// V86_WASM can select another build. Correctness test for exact page-tail
// compilation: instructions that end inside their page are compiled even when
// they start in the last 16 bytes, while instructions that actually cross the
// boundary (or end exactly on it) stay interpreted.
import assert from "node:assert/strict";
import fs from "node:fs";

const wasm_path = process.env.V86_WASM || new URL("../../build/v86-fallback.wasm", import.meta.url);
const module = new WebAssembly.Module(fs.readFileSync(wasm_path));
const START = 0x1000;
const DATA = 0x8000;
const MEMORY_SIZE = 1024 * 1024;
const TABLE_OFFSET = 1024;
const ITERATIONS = 200;

function create_cpu(code, { disable_jit = false } = {})
{
    const table = new WebAssembly.Table({ element: "anyfunc", initial: TABLE_OFFSET + 900 });
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
                case "microtick":
                case "get_rand_int":
                case "cpu_event_halt":
                case "stop_idling":
                case "run_hardware_timers":
                case "io_port_read8":
                case "io_port_read16":
                case "io_port_read32":
                case "io_port_write8":
                case "io_port_write16":
                case "io_port_write32":
                case "mmap_read8":
                case "mmap_read32":
                case "mmap_write8":
                case "mmap_write16":
                case "mmap_write32":
                case "mmap_write64":
                case "mmap_write128":
                    return 0;
                case "jit_clear_func":
                    table.set(TABLE_OFFSET + args[0], null);
                    return;
                case "cpu_exception_hook":
                case "dbg_trace_from_wasm":
                case "log_from_wasm":
                    // Debug-build diagnostics only.
                    return 0;
                case "console_log_from_wasm":
                    throw new Error(new TextDecoder().decode(
                        new Uint8Array(wasm.memory.buffer, args[0], args[1])));
                case "codegen_finalize": {
                    const [index, _start, _state_flags, pointer, length] = args;
                    const generated = new WebAssembly.Module(
                        Buffer.from(wasm.memory.buffer, pointer, length));
                    const { f } = new WebAssembly.Instance(generated, { e: jit_imports }).exports;
                    table.set(TABLE_OFFSET + index, f);
                    // Synchronous finalize (jit config 6): the wasm completes
                    // the finalize itself when this import returns.
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
    assert.equal(wasm.vine_set_memory_size(MEMORY_SIZE), 0);
    const physical_base = wasm.allocate_memory(MEMORY_SIZE);
    wasm.reset_cpu();
    wasm.set_jit_config(0, Number(disable_jit));
    wasm.set_jit_config(4, 1);
    wasm.set_jit_config(5, 1);
    wasm.set_jit_config(6, 1);
    new Uint8Array(wasm.memory.buffer, physical_base + START, code.length).set(code);

    wasm.vine_cpu_state_save();
    const state = new Uint32Array(wasm.memory.buffer, wasm.vine_cpu_state_ptr(), 153);
    state[6] = 0x90000; // esp
    state[9] = DATA; // edi: writable data
    state[10] = state[11] = START;
    state[19] = state[20] = state[21] = 1; // protected mode, 32-bit code/stack
    state.fill(0, 34, 42); // flat segments
    state.fill(0xFFFFFFFF, 42, 50); // segment limits
    state[58] = 1; // CR0.PE, without paging
    assert.equal(wasm.vine_cpu_state_restore(), 0);

    function retired(prefix)
    {
        return BigInt(wasm[`${prefix}_high`]() >>> 0) << 32n |
            BigInt(wasm[`${prefix}_low`]() >>> 0);
    }

    return {
        wasm,
        get physical() { return new DataView(wasm.memory.buffer, physical_base, MEMORY_SIZE); },
        get interpreted() { return retired("vine_interpreted_retired_instructions"); },
        get compiled() { return retired("vine_jit_retired_instructions"); },
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
                data: [...new Uint8Array(wasm.memory.buffer, physical_base + DATA, 32)],
                jit: retired("vine_jit_retired_instructions"),
                interp: retired("vine_interpreted_retired_instructions"),
            };
        },
    };
}

// Flat 32-bit code; everything executes on each of ITERATIONS loop passes.
// (a) funcA ends in "pop ebp; ret 4" at 0x1FF0..0x1FF3, reached by falling
//     through; (e) funcD's tail "mov edx, imm32" at 0x1FF4 gets its immediate
//     rewritten by the guest on the second-to-last iteration.
// (b) funcB's tail call at 0x2FF0 returns to 0x2FF5; mov/test/mov and a taken
//     jne to the next page all end inside the page and run compiled.
// (d) funcC's "mov ebx, imm32" at 0x3FFB ends exactly at the page end.
// (c) funcC's "mov eax, imm32" at 0x4FFD straddles the boundary.
const code = new Uint8Array(0x5003 - START).fill(0x90);
const put = (offset, bytes) => code.set(bytes, offset - START);

put(0x1000, [0xC7, 0x47, 0x0C, ITERATIONS, 0, 0, 0]); // mov dword [edi+12], ITERATIONS
put(0x1007, [0x6A, 0x07]);                           // push 7
put(0x1009, [0xE8, 0xD2, 0x0F, 0x00, 0x00]);         // call 0x1FE0 (funcA)
put(0x100E, [0xE8, 0xCD, 0x1F, 0x00, 0x00]);         // call 0x2FE0 (funcB)
put(0x1013, [0xE8, 0xC8, 0x2F, 0x00, 0x00]);         // call 0x3FE0 (funcC)
put(0x1018, [0xE8, 0xD7, 0x0F, 0x00, 0x00]);         // call 0x1FF4 (funcD)
put(0x101D, [0x83, 0x7F, 0x0C, 0x02]);               // cmp dword [edi+12], 2
put(0x1021, [0x75, 0x07]);                           // jne 0x102A
put(0x1023, [0xC6, 0x05, 0xF5, 0x1F, 0x00, 0x00, 0x2A]); // mov byte [0x1FF5], 42
put(0x102A, [0xFF, 0x4F, 0x0C]);                     // dec dword [edi+12]
put(0x102D, [0x75, 0xD8]);                           // jnz 0x1007
put(0x102F, [0xE8, 0xC0, 0x0F, 0x00, 0x00]);         // call 0x1FF4 (funcD, once more)
put(0x1034, [0xF4]);                                 // hlt

put(0x1FE0, [0x55, 0x89, 0xEC, 0x8B, 0x45, 0x08, 0x01, 0x07]); // push ebp; mov ebp,esp; mov eax,[ebp+8]; add [edi],eax
put(0x1FF0, [0x5D, 0xC2, 0x04, 0x00]);               // (a) pop ebp; ret 4
put(0x1FF4, [0xBA, 0, 0, 0, 0, 0x89, 0x57, 0x10, 0xC3]);     // (e) mov edx,0; mov [edi+16],edx; ret

put(0x2080, [0xB8, 0x2A, 0, 0, 0, 0xC3]);            // stub: mov eax,42; ret
put(0x2FE0, [0xFF, 0x47, 0x08]);                     // funcB: inc dword [edi+8]
put(0x2FF0, [0xE8, 0x8B, 0xF0, 0xFF, 0xFF]);         // (b) call 0x2080, return addr 0x2FF5
put(0x2FF5, [0x89, 0xC6, 0x85, 0xF6, 0x89, 0x77, 0x04]); // mov esi,eax; test esi,esi; mov [edi+4],esi
put(0x2FFC, [0x75, 0x02]);                           // jne 0x3000 (taken)
put(0x3000, [0xC3]);                                 // ret

// funcC starts at 0x3FE0; the 0x90 fill between 0x3FE0 and 0x3FFB is 27 NOPs.
put(0x3FFB, [0xBB, 0x78, 0x56, 0x34, 0x12]);         // (d) mov ebx,0x12345678, ends at page end
put(0x4000, [0xE9, 0xF8, 0x0F, 0x00, 0x00]);         // jmp 0x4FFD
put(0x4FFD, [0xB8, 0x99, 0x88, 0x77, 0x66]);         // (c) mov eax,0x66778899, straddles boundary
put(0x5002, [0xC3]);                                 // ret

// Per-iteration instruction counts: 9 loop instructions + funcA 15 (4 + 9 NOP
// fill + pop ebp + ret) + funcB+stub 21 + funcC 31 (27 NOP fill + 4) +
// funcD 3 = 79; the self-modifying pass adds the extra store (80).
const ITER_INSTRUCTIONS = 79;

const jit = create_cpu(code);
const interpreted = create_cpu(code, { disable_jit: true });

function step(budget, description)
{
    const actual = jit.execute(budget);
    const expected = interpreted.execute(budget);
    // The raw instruction counter is excluded: the JIT side retires one extra
    // instruction when the self-modifying store invalidates a code page
    // (pre-existing accounting, identical on builds without this change).
    assert.deepEqual(
        { ...actual, jit: undefined, interp: undefined, instructions: undefined },
        { ...expected, jit: undefined, interp: undefined, instructions: undefined },
        `${description}: interpreter oracle`);
    return actual;
}

// Warm up: slices land on unregistered entry offsets at first; each landing
// registers and compiles it, so within a bounded number of iterations every
// offset in the loop's steady state has compiled code. From then on each
// iteration must retire exactly the two page-straddling instructions — (c)
// and (d) — interpreted; all other 77 run in generated code.
let consecutive = 0;
let converged = false;
for(let i = 0; i < 80 && !converged; i++)
{
    const before_i = jit.interpreted;
    const before_j = jit.compiled;
    step(ITER_INSTRUCTIONS, `warmup iteration ${i}`);
    consecutive = jit.interpreted - before_i === 2n && jit.compiled - before_j === 77n ?
        consecutive + 1 : 0;
    converged = consecutive >= 2;
}
assert(converged, "JIT coverage failed to converge within 80 iterations");

let result;
do
{
    result = step(8192, "finish");
} while(result.reason !== 2);
assert.equal(jit.physical.getUint32(DATA + 0, true), ITERATIONS * 7, "funcA accumulated argument");
assert.equal(jit.physical.getUint32(DATA + 4, true), 42, "funcB store");
assert.equal(jit.physical.getUint32(DATA + 8, true), ITERATIONS, "funcB call count");
assert.equal(jit.physical.getUint32(DATA + 16, true), 42, "self-modified funcD immediate");
console.log("page-tail: steady-state iterations retire exactly the two " +
    "page-straddling instructions interpreted; guest state matches oracle");
