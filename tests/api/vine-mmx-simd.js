// Direct MMX lowering: independent lane oracle, full boundary comparison,
// exact budgets, RAM width/faults and MMIO callbacks. Not a timing harness.
// Optionally compare an unchanged build with V86_MMX_BASELINE_WASM=/path/to/old.wasm.
import assert from "node:assert/strict";
import fs from "node:fs";
import { create_cpu } from "./vine-jit-budget.js";

const old_path = process.env.V86_MMX_BASELINE_WASM;
const baseline = old_path ? new WebAssembly.Module(fs.readFileSync(old_path)) : undefined;
const mask = (1n << 64n) - 1n;
const values = [0n, mask, 1n, 0x8000000000000000n, 0x000100ff0100ffffn,
    0x7fff8000fffeffffn, 0x0123456789abcdefn, 0xfedcba9876543210n,
    0x00ff010000fe0001n, 0x80017ffe00000080n, 0x8000800080008000n,
    0x7fff7fff7fff7fffn, 0x800000007fffffffn, 0x00008000ffff7fffn];
let random = 0x123456789abcdef0n;
for(let i = 0; i < 22; i++) {
    random ^= random << 13n; random ^= random >> 7n; random ^= random << 17n;
    random &= mask; values.push(random);
}
const u16 = (x, i) => Number(x >> BigInt(i * 16) & 65535n);
const signed = x => x & 0x8000 ? x - 65536 : x;
const words = a => a.reduce((v, x, i) => v | BigInt(x & 65535) << BigInt(i * 16), 0n);
const dwords = a => a.reduce((v, x, i) => v | BigInt(x >>> 0) << BigInt(i * 32), 0n);
const i32 = (x, i) => Number(x >> BigInt(i * 32) & 0xffffffffn) | 0;
function oracle(op, a, b, shift = 0) {
    if(op === 0xdb) return a & b;
    if(op === 0xef) return a ^ b;
    if(op === 0xfe) return dwords([i32(a,0)+i32(b,0),i32(a,1)+i32(b,1)]);
    if(op === 0x6b) return words([i32(a,0),i32(a,1),i32(b,0),i32(b,1)]
        .map(x=>Math.max(-32768,Math.min(32767,x))));
    if(op === 0xf5) return dwords([0,2].map(i=>
        signed(u16(a,i))*signed(u16(b,i))+signed(u16(a,i+1))*signed(u16(b,i+1))));
    if(op === 0x724) return dwords([i32(a,0)>>Math.min(shift,31),i32(a,1)>>Math.min(shift,31)]);
    if(op === 0x60) {
        let r = 0n;
        for(let i = 0; i < 4; i++) r |= ((a >> BigInt(8*i) & 255n) |
            (b >> BigInt(8*i) & 255n) << 8n) << BigInt(16*i);
        return r;
    }
    if(op === 0x61) return words([u16(a,0),u16(b,0),u16(a,1),u16(b,1)]);
    if(op === 0x67) {
        let r = 0n;
        for(let i = 0; i < 8; i++) {
            const v = Math.max(0, Math.min(255, signed(u16(i < 4 ? a : b, i & 3))));
            r |= BigInt(v) << BigInt(8*i);
        }
        return r;
    }
    return words(Array.from({length:4}, (_,i) => {
        const x=u16(a,i),y=u16(b,i);
        if(op===0xd5) return Math.imul(x,y);
        if(op===0xe5) return Math.imul(signed(x),signed(y)) >> 16;
        if(op===0xf9) return x-y;
        if(op===0xfd) return x+y;
        if(op===0x712) return shift>=16 ? 0 : x>>>shift;
        if(op===0x716) return shift>=16 ? 0 : x<<shift;
        assert.equal(op,0x71); return signed(x) >> Math.min(shift,15);
    }));
}
function update(cpu, fn) {
    cpu.wasm.vine_cpu_state_save();
    const state=new Uint32Array(cpu.wasm.memory.buffer,cpu.wasm.vine_cpu_state_ptr(),153);
    fn(state);assert.equal(cpu.wasm.vine_cpu_state_restore(),0);
}
function seed(cpu, a, b, dest=1, source=0, address=0x8000, paging=false) {
    cpu.resume();
    update(cpu,s=>{
        s[88]=0xa5;s[89]=7;s[91]=0x0241;
        for(let r=0;r<8;r++) {
            const v=(values[r]+BigInt(r))&mask;
            s[97+r*3]=Number(v&0xffffffffn);s[98+r*3]=Number(v>>32n);s[99+r*3]=0x1234+r;
        }
        for(const [r,v] of [[source,b],[dest,a]]) {
            s[97+r*3]=Number(v&0xffffffffn);s[98+r*3]=Number(v>>32n);
        }
        s[9]=address+(paging?0x400000:0);
    });
    cpu.physical.setBigUint64(address,b,true);
}
function compare(a,b,label,{fault=false,interpreter=false}={}) {
    const sa=a.full_snapshot(),sb=b.full_snapshot();
    // The ordinary JIT may charge a faulting instruction differently from
    // the interpreter; retain all raw words for the unchanged-JIT comparison.
    if(fault&&interpreter) assert.deepEqual(sa.state.slice(88),sb.state.slice(88),label+" FP/SIMD state");
    else assert.deepEqual(sa.state.filter((_,i)=>fault||i!==11),sb.state.filter((_,i)=>fault||i!==11),label+" raw state");
    assert.equal(a.wasm.get_eflags()>>>0,b.wasm.get_eflags()>>>0,label+" flags");
    assert.deepEqual(sa.memory,sb.memory,label+" memory");
    assert.deepEqual(sa.dirty,sb.dirty,label+" dirty");
}
const forbidden=new Set(["instr_0F60","instr_0F61","instr_0F67","instr_0FD5","instr_0FF9","instr_0FFD","instr_0F71_4_reg","instr_0F6E","instr_0F7E",
    "instr_0F6B","instr_0FDB","instr_0FEF","instr_0FE5","instr_0FF5","instr_0FFE",
    "instr_0F71_2_reg","instr_0F71_6_reg","instr_0F72_4_reg"]);
function group(code,options={}) {
    const modules=[];
    const on_finalize=(wasm,args)=>{
        const bytes=Buffer.from(wasm.memory.buffer,args[3],args[4]);
        const names=WebAssembly.Module.imports(new WebAssembly.Module(bytes)).map(i=>i.name);
        assert(!names.some(n=>forbidden.has(n)),"selected MMX helpers must be absent");modules.push(names);
    };
    const cpus=[create_cpu(code,{...options,on_finalize}),create_cpu(code,{...options,cpu_module:baseline}),
        create_cpu(code,{...options,disable_jit:true})];
    for(const c of cpus){seed(c,values[4],values[5],1,0,0x8000,options.paging);c.execute(12000);c.resume();}
    assert(modules.length>0);assert(cpus[0].counts.entries>0);
    return cpus;
}
let checked=0;
for(const op of [0x6e,0x7e]) for(const memory of [false,true]) {
    const cpus=group([0x0f,op,memory?0x0f:0xc8,0xff,0xe1],{paging:true});
    for(const address of memory?[0x8000,0x8001,0x8ffc,0x8fff]:[0x8000]) {
        for(const value of values) {
            for(const c of cpus) {
                seed(c,value,value,1,0,address,true);
                update(c,s=>{s[2]=Number(value&0xffffffffn);});
            }
            const results=cpus.map(c=>c.execute(2));
            assert.deepEqual(results[0],results[1]);assert.deepEqual(results[0],results[2]);
            compare(cpus[0],cpus[1],"MOVD reference JIT");compare(cpus[0],cpus[2],"MOVD interpreter");
            const s=cpus[0].full_snapshot().state;
            if(op===0x6e){assert.equal(s[100],Number(value&0xffffffffn));assert.equal(s[101],0);assert.equal(s[102],65535);}
            else if(!memory)assert.equal(s[2],Number(value&0xffffffffn));
            else assert.equal(cpus[0].physical.getUint32(address,true),Number(value&0xffffffffn));
            assert.equal(s[88],0);assert.equal(s[89],0);checked++;
        }
    }
    if(memory) {
        for(const c of cpus){seed(c,values[7],values[7],1,0,0x8fff,true);c.physical.setUint32(0x11000+9*4,0,true);c.wasm.full_clear_tlb();}
        const results=cpus.map(c=>c.execute(2));assert.equal(results[0].reason,0x10e);
        assert.deepEqual(results[0],results[1]);compare(cpus[0],cpus[1],"MOVD second-page fault",{fault:true});
        compare(cpus[0],cpus[2],"MOVD interpreter fault",{fault:true,interpreter:true});
    }
}
const binary_ops=[0x60,0x61,0x67,0xd5,0xf9,0xfd,0x6b,0xdb,0xef,0xe5,0xf5,0xfe];
for(const op of binary_ops) for(const [dest,source] of [[1,0],[7,3],[5,5]]) {
    const cpus=group([0x0f,op,0xc0|dest<<3|source,0xff,0xe1]);
    for(let i=0;i<values.length;i++) {
        const a=values[i],b=values[(i*7+3)%values.length];
        for(const c of cpus)seed(c,a,b,dest,source);
        const before=cpus[0].counts.entries;
        const results=cpus.map(c=>c.execute(2));
        assert.deepEqual(results[0],results[1]);assert.deepEqual(results[0],results[2]);
        assert(cpus[0].counts.entries>before,"compiled direct operation executes");
        compare(cpus[0],cpus[1],"reference JIT");compare(cpus[0],cpus[2],"interpreter");
        const s=cpus[0].full_snapshot().state;
        const result=BigInt(s[97+dest*3])|BigInt(s[98+dest*3])<<32n;
        assert.equal(result,oracle(op,a,source===dest?a:b),"independent lane oracle");
        assert.equal(s[99+dest*3],65535);assert.equal(s[88],0);assert.equal(s[89],0);checked++;
    }
    for(const budget of [0,1,2,3,7,4095,4096,4097,131072]) {
        const results=cpus.map(c=>c.execute(budget));assert.deepEqual(results[0],results[1]);assert.deepEqual(results[0],results[2]);
        compare(cpus[0],cpus[1],"budget reference JIT");compare(cpus[0],cpus[2],"budget interpreter");
    }
}
for(const [opcode,extension,op] of [[0x71,4,0x71],[0x71,2,0x712],[0x71,6,0x716],[0x72,4,0x724]])
for(const shift of [0,1,7,14,15,16,17,30,31,32,63,127,255]) {
    const cpus=group([0x0f,opcode,0xc1|extension<<3,shift,0xff,0xe1]);
    for(const a of values) {
        for(const c of cpus)seed(c,a,0n);
        const results=cpus.map(c=>c.execute(2));assert.deepEqual(results[0],results[1]);assert.deepEqual(results[0],results[2]);
        compare(cpus[0],cpus[1],"shift reference JIT");compare(cpus[0],cpus[2],"shift interpreter");
        const s=cpus[0].full_snapshot().state;
        assert.equal(BigInt(s[100])|BigInt(s[101])<<32n,oracle(op,a,0n,shift));checked++;
    }
}
for(const op of binary_ops) {
    const cpus=group([0x0f,op,0x0f,0xff,0xe1],{paging:true});
    for(const address of [0x8000,0x8001,0x8ff8,0x8ffc,0x8fff]) {
        for(const c of cpus)seed(c,values[4],values[5],1,0,address,true);
        const results=cpus.map(c=>c.execute(2));assert.deepEqual(results[0],results[1]);assert.deepEqual(results[0],results[2]);
        compare(cpus[0],cpus[1],"RAM width reference JIT");compare(cpus[0],cpus[2],"RAM width interpreter");
    }
    // Low-unpack may read exactly four bytes when the following page is absent.
    if(op===0x60||op===0x61) {
        for(const c of cpus) {
            seed(c,values[4],values[5],1,0,0x8ffc,true);
            c.physical.setUint32(0x11000+9*4,0,true);c.wasm.full_clear_tlb();
        }
        const edge=cpus.map(c=>c.execute(2));assert.equal(edge[0].reason,1);
        assert.deepEqual(edge[0],edge[1]);assert.deepEqual(edge[0],edge[2]);
        compare(cpus[0],cpus[1],"m32 exact page edge");compare(cpus[0],cpus[2],"m32 edge interpreter");
        for(const c of cpus){c.physical.setUint32(0x11000+9*4,0x9007,true);c.wasm.full_clear_tlb();}
    }
    // Now force an actual crossing into the absent page for each read width.
    const address=op===0x60||op===0x61?0x8ffd:0x8ffc;
    for(const c of cpus) {
        seed(c,values[4],values[5],1,0,address,true);
        c.physical.setUint32(0x11000+9*4,0,true);c.wasm.full_clear_tlb();
    }
    const results=cpus.map(c=>c.execute(2));assert.deepEqual(results[0],results[1]);assert.equal(results[0].reason,0x10e);
    compare(cpus[0],cpus[1],"fault reference JIT",{fault:true});compare(cpus[0],cpus[2],"fault interpreter",{fault:true,interpreter:true});
    for(const c of cpus){c.physical.setUint32(0x11000+9*4,0x9007,true);c.wasm.full_clear_tlb();c.resume();}
    cpus.forEach(c=>c.execute(2));compare(cpus[0],cpus[1],"fault repair reference JIT");
}
// Memory helpers may invoke MMIO callbacks that change CR0. The next MMX
// instruction must check again despite the preceding direct SIMD operation.
{
    let armed=false;
    const cpus=group([0x0f,0x60,0x0f,0x0f,0xfd,0xd1,0xff,0xe1],{
        on_mmap(name,_args,wasm){assert.equal(name,"mmap_read32");if(armed){const m=new DataView(wasm.memory.buffer);m.setUint32(580,m.getUint32(580,true)|8,true);} return 0x01020304;},
    });
    for(const c of cpus){seed(c,values[4],values[5]);c.set_register(7,0xa0000);}
    armed=true;
    const result=cpus.map(c=>c.execute(8));assert.deepEqual(result[0],result[1]);
    assert.equal(result[0].reason,0x107);assert.equal(result[0].eip,0x1003);
    compare(cpus[0],cpus[1],"MMIO changes CR0",{fault:true});
    assert(cpus[0].counts.mmap>0);
}
console.log(`MMX SIMD: ${checked} independent lane/alias/shift cases, reference JIT/interpreter boundaries, exact budgets, RAM width/fault/repair and MMIO guard refresh passed`);
