// Common-helper/JIT equivalence is a boundary check, not a numerical oracle.
// Ordinary numerical results below also use independent JS binary64 arithmetic.
import assert from "node:assert/strict";
import fs from "node:fs";
import { create_cpu } from "./vine-jit-budget.js";

const DATA = 0x8000;
const code = [0xd9,7,0xd8,0x4f,4,0xd8,0x47,8,0xd9,0x5f,12,0xff,0xe1];
function seed(cpu, values = [1.25,1.8,0.125], control = 0x37f) {
    cpu.resume(); cpu.wasm.set_control_word(control);
    const view = new DataView(cpu.wasm.memory.buffer);
    view.setUint8(816,255); view.setUint8(1032,7); view.setUint16(1040,0,true);
    for(let i=0;i<values.length;i++) cpu.physical.setFloat32(DATA+i*4,values[i],true);
}
const cpus = [create_cpu(code,{disable_jit:true}),create_cpu(code),create_cpu(code)];
for(let i=0;i<cpus.length;i++) {
    assert.equal(cpus[i].wasm.vine_get_x87_mode(),0);
    assert.equal(cpus[i].wasm.vine_set_x87_mode(i===2?3:1),0);
    seed(cpus[i]);
}
for(const budget of [100000,0,1,2,3,4,5,6,4095,4096,4097,131072]) {
    const results=cpus.map(cpu=>cpu.execute(budget));
    assert.deepEqual(results[1],results[0],"helper JIT/interpreter boundary");
    assert.deepEqual(results[2],results[0],"value JIT/interpreter boundary");
    assert.equal(results[0].instructions,budget);
}
for(const cpu of cpus) {
    const actual=cpu.physical.getFloat32(DATA+12,true);
    assert.equal(actual,Math.fround(Math.fround(1.25)*Math.fround(1.8)+Math.fround(0.125)));
}
assert(cpus[1].counts.entries>0); assert(cpus[2].counts.entries>0);
assert(cpus[2].wasm.vine_x87_fast_instructions()>0,"value runs actually execute");

// Runtime control changes must be observed by already installed native code.
for(const control of [0x37f,0x27f,0x7f,0x77f,0xb7f,0xf7f,0x35f]) {
    for(const cpu of cpus) seed(cpu,[1.234567,-0.75,0.3125],control);
    const results=cpus.map(cpu=>cpu.execute(100000));
    assert.deepEqual(results[1],results[0]); assert.deepEqual(results[2],results[0]);
}
// Signed zero, subnormal, NaN and overflow must remain executable and bounded.
for(const values of [[-0,1,0],[1e-40,0.5,0],[Infinity,0,1],[NaN,1,0],[3.4e38,3.4e38,1]]) {
    for(const cpu of cpus) seed(cpu,values);
    const results=cpus.map(cpu=>cpu.execute(100000));
    assert.deepEqual(results[1],results[0]); assert.deepEqual(results[2],results[0]);
}

const mode_cpu=cpus[2];
seed(mode_cpu);
mode_cpu.execute(100000);
const before=mode_cpu.full_snapshot();
assert.equal(mode_cpu.wasm.vine_set_x87_mode(9),1);
assert.equal(mode_cpu.wasm.vine_get_x87_mode(),3);
assert.deepEqual(mode_cpu.full_snapshot(),before,"invalid policy is not a partial mutation");
assert.equal(mode_cpu.wasm.vine_set_x87_mode(0),0);
assert.equal(mode_cpu.wasm.vine_get_x87_mode(),0);
seed(mode_cpu); mode_cpu.execute(100000);

// A store fault must not publish a rounded value or pop the architectural stack.
const fault_code=[0xd9,7,0xd8,0x4f,4,0xd9,0x1b,0xff,0xe1];
for(const mode of [1,2]) {
    const jit=create_cpu(fault_code,{paging:true});
    const interpreted=create_cpu(fault_code,{paging:true,disable_jit:true});
    for(const cpu of [jit,interpreted]) {
        cpu.wasm.vine_set_x87_mode(mode); seed(cpu);
        cpu.set_register(3,0x400000+DATA+12);
        cpu.execute(120000);
        seed(cpu); cpu.set_register(3,0x70000000);
    }
    const a=jit.execute(50),b=interpreted.execute(50);
    assert.equal(a.reason,0x10e); assert.equal(b.reason,0x10e);
    assert.deepEqual(a.data,b.data); assert.deepEqual(a.registers,b.registers);
    assert.equal(a.eip,b.eip);
    assert.deepEqual(jit.full_snapshot().state.slice(88,121),interpreted.full_snapshot().state.slice(88,121));
}
console.log("x87 performance: interpreter/JIT budgets, control changes, numeric oracle, exceptional values, policy and store fault passed");

// Exercise each decoded register arithmetic form, rotating TOP, stack copies,
// exchanges and MMX aliasing. Compare raw architectural values after every exit.
let register_cases=0;
for(const opcode of [0xd8,0xdc,0xde]) for(const group of [0,1,4,5]) {
    const pop=opcode===0xde;
    const bytes=[0xd9,7,0xd9,0x47,4,0xd9,0xc9,0xd9,0xe0,
        opcode,0xc1|group<<3,0xd9,0x5f,12,...(pop?[]:[0xd9,0x5f,16]),0xff,0xe1];
    const a=create_cpu(bytes),b=create_cpu(bytes,{disable_jit:true});
    a.wasm.vine_set_x87_mode(3);b.wasm.vine_set_x87_mode(1);
    for(const top of [0,3,7]) for(const values of [[1.234567,-0.75],[1e-40,-2],[NaN,Infinity]]) {
        for(const cpu of [a,b]){seed(cpu,values);new DataView(cpu.wasm.memory.buffer).setUint8(1032,top);}
        const x=a.execute(100000),y=b.execute(100000);assert.deepEqual(x,y);
        const sa=a.full_snapshot(),sb=b.full_snapshot();
        assert.deepEqual(sa.state.slice(88),sb.state.slice(88));
        assert.deepEqual(sa.memory,sb.memory);register_cases++;
    }
    assert(a.wasm.vine_x87_fast_instructions()>0,"register run reached");
}

// Keep guest accesses in program order even when distinct EAs overlap. The
// cross-page and rearmed-dirty cases must decline the inline memory path.
const alias_code=[0xd9,7,0xd8,0x4f,4,0xd9,0x13,0xd8,0x03,0xd9,0x5f,12,0xff,0xe1];
for(const output of [DATA+8,DATA+12,DATA+4094]) {
    const a=create_cpu(alias_code,{paging:true}),b=create_cpu(alias_code,{paging:true,disable_jit:true});
    a.wasm.vine_set_x87_mode(3);b.wasm.vine_set_x87_mode(1);
    for(const cpu of [a,b]){seed(cpu);cpu.set_register(3,0x400000+output);}
    for(let i=0;i<4;i++) {
        assert.deepEqual(a.execute(120000),b.execute(120000));
        const sa=a.full_snapshot(),sb=b.full_snapshot();
        assert.deepEqual(sa.state.slice(88),sb.state.slice(88));
        assert.deepEqual(sa.memory,sb.memory);assert.deepEqual(sa.dirty,sb.dirty);
        for(const cpu of [a,b])cpu.clear_dirty([DATA,DATA+4096]);
    }
}

// EMMS leaves the shared raw MMX/x87 bank intact; the next value run must read
// the correct tag/TOP state and publish raw80 before subsequent MMX observes it.
const mmx_code=[0x0f,0xef,0xc0,0x0f,0x77,...code.slice(0,-2),0x0f,0x6f,0xc8,0x0f,0x77,0xff,0xe1];
const ma=create_cpu(mmx_code),mb=create_cpu(mmx_code,{disable_jit:true});
for(const [cpu,mode] of [[ma,3],[mb,1]]){cpu.wasm.vine_set_x87_mode(mode);seed(cpu);}
for(const budget of [100000,1,2,3,4097]){
 assert.deepEqual(ma.execute(budget),mb.execute(budget));
 assert.deepEqual(ma.full_snapshot().state.slice(88),mb.full_snapshot().state.slice(88));
}

let active_rejections=0;
const active=create_cpu(code,{on_finalize(wasm){
 const mode=wasm.vine_get_x87_mode();
 assert.equal(wasm.vine_set_x87_mode(0),1);assert.equal(wasm.vine_get_x87_mode(),mode);active_rejections++;
}});
active.wasm.vine_set_x87_mode(2);seed(active);active.execute(100000);
assert(active_rejections>0,"active policy mutation was exercised");
console.log(`x87 value runs: ${register_cases} register/TOP/exception cases, RAM aliases, cross-page access, dirty rearm, MMX sharing and active policy rejection passed`);

// Mixed runs must use each effective address at its original program point.
// Integer flags remain visible to the following conditional/ADC instructions.
const mixed_cases=[
 [0xd9,7,0xd8,0x4f,4,0x8d,0x5f,12,0xd9,0x1b], // LEA between MUL/store
 [0xd9,7,0x83,0xc7,4,0xd8,7,0x83,0xef,4,0xd9,0x5f,12], // successive EDI versions
 [0xd9,7,0x8b,0xdf,0xd8,0x4b,4,0x83,0xc3,8,0xd9,0x13,0x83,0xeb,4,
  0xd8,0x43,4,0xd9,0x5f,12], // aliased store/reload through changed EBX
 [0xd9,7,0xb8,0xfc,0xff,0xff,0xff,0x03,0xc7,0xd8,0x48,8,
  0x8d,0x5c,0x00,4,0x2b,0xd8,0x83,0xc3,8,0xd9,0x1b], // affine wrap and SIB
 [0xd9,7,0x89,0xf8,0x05,4,0,0,0,0xd8,0x08,0x2d,4,0,0,0,0xd9,0x58,12],
 [0xd9,7,0xd8,0x4f,4,0x48,0x42,0x81,0xc2,0xff,0xff,0xff,0x7f,0xd9,0x5f,12],
 [0xd9,7,0xd8,0x4f,4,0x3b,0xdf,0xd9,0x5f,12],
 [0xd9,7,0x81,0xfb,0,0,0,0x80,0xd8,0x4f,4,0x3d,1,0,0,0,0xd9,0x5f,12],
];
for(const [caseIndex,body] of mixed_cases.entries()) {
 const bytes=[...body,0x83,0xd6,0,0xff,0xe1]; // ADC esi,0 consumes CF after run
 const a=create_cpu(bytes,{paging:true}),b=create_cpu(bytes,{paging:true,disable_jit:true});
 for(const [cpu,mode] of [[a,3],[b,1]]){cpu.wasm.vine_set_x87_mode(mode);seed(cpu);}
 for(const budget of [130000,1,2,7,4097,131072]) {
  assert.deepEqual(a.execute(budget),b.execute(budget),`mixed case ${caseIndex}, budget ${budget}`);
  const sa=a.full_snapshot(),sb=b.full_snapshot();
  // PREVIOUS_EIP is unspecified at a budget exit. Existing ADC lowering omits
  // last_op1 when CF/AF/OF are already materialized; only then it is dead.
  for(const s of [sa,sb]) if((s.state[13]&0x811)===0)s.state[15]=0;
  assert.deepEqual(sa.state.filter((_,i)=>i!==11),sb.state.filter((_,i)=>i!==11),
   `mixed case ${caseIndex} full raw state except budget PREVIOUS_EIP`);
  assert.deepEqual(sa.memory,sb.memory);assert.deepEqual(sa.dirty,sb.dirty);
 }
 assert(a.wasm.vine_x87_fast_instructions()>0,`mixed case ${caseIndex} executes value run`);
}

// A failed future RAM guard must leave integer registers untouched until the
// slow arm executes them in order. Check both cold and already compiled code.
for(const destination of [DATA+12,DATA+4094,0x70000000]) {
 const bytes=[0xd9,7,0xd8,0x4f,4,0x8d,0x53,4,0xd9,0x1a,0xff,0xe1];
 const a=create_cpu(bytes,{paging:true}),b=create_cpu(bytes,{paging:true,disable_jit:true});
 for(const [cpu,mode] of [[a,3],[b,1]]){
  cpu.wasm.vine_set_x87_mode(mode);seed(cpu);cpu.set_register(3,0x400000+DATA+8);
  cpu.execute(130000);seed(cpu);
  cpu.set_register(3,destination>=0x70000000?destination-4:0x400000+destination-4);
 }
 const x=a.execute(100),y=b.execute(100);
 assert.equal(x.reason,y.reason);assert.equal(x.eip,y.eip);
 assert.deepEqual(x.registers,y.registers);assert.deepEqual(x.data,y.data);
 const sa=a.full_snapshot(),sb=b.full_snapshot();
 assert.deepEqual(sa.state.slice(88),sb.state.slice(88));assert.deepEqual(sa.memory,sb.memory);
}
console.log("x87 mixed runs: wrapping/changed/aliased EAs, integer flags, budgets and late store fault passed");

// The untaken branch creates an interior entry immediately after FSTP. The
// preceding block therefore ends with FSTP, not a jump. A normal fallthrough
// must retain that final store without skipping it or decoding past the BB.
const tail_body=[0xd9,7,0xd8,0x4f,4,0x8d,0x53,4,0xd9,0x1a];
const tail_code=[0x31,0xc0,0x85,0xc0,0x75,tail_body.length,...tail_body,0xff,0xe1];
const ta=create_cpu(tail_code,{paging:true}),tb=create_cpu(tail_code,{paging:true,disable_jit:true});
for(const [cpu,mode] of [[ta,3],[tb,1]]){
 cpu.wasm.vine_set_x87_mode(mode);seed(cpu);cpu.set_register(3,0x400000+DATA+8);
}
for(const budget of [130000,1,2,5,4097,131072]){
 assert.deepEqual(ta.execute(budget),tb.execute(budget));
 const a=ta.full_snapshot(),b=tb.full_snapshot();
 assert.deepEqual(a.state.filter((_,i)=>i!==11),b.state.filter((_,i)=>i!==11));
 assert.deepEqual(a.memory,b.memory);assert.deepEqual(a.dirty,b.dirty);
}
assert(ta.wasm.vine_x87_fast_instructions()>0,"last-instruction FSTP is fused");
for(const cpu of [ta,tb]){seed(cpu);cpu.set_register(3,0x70000000-4);cpu.restart_at_entry();}
const tx=ta.execute(100),ty=tb.execute(100);
assert.equal(tx.reason,0x10e);assert.equal(tx.reason,ty.reason);
assert.equal(tx.eip,ty.eip);assert.deepEqual(tx.registers,ty.registers);
assert.deepEqual(ta.full_snapshot().state,tb.full_snapshot().state,"precise fallthrough store fault");
assert.deepEqual(ta.full_snapshot().memory,tb.full_snapshot().memory);
console.log("x87 final normal-BB store: executed fast path, exact budgets and all raw fault state passed");

// Retain binary64 spills and integer stores in the same value run. The helper
// JIT is the boundary reference, and an optional unchanged CPU checks against
// accidentally changing both sides. Ordinary results have a separate JS oracle.
const conversion_baseline=process.env.V86_X87_BASELINE_WASM &&
 new WebAssembly.Module(fs.readFileSync(process.env.V86_X87_BASELINE_WASM));
function conversion_cpus(bytes,paging=true) {
 const a=create_cpu(bytes,{paging}),b=create_cpu(bytes,{paging});
 const all=[a,b];a.wasm.vine_set_x87_mode(3);b.wasm.vine_set_x87_mode(1);
 if(conversion_baseline){const c=create_cpu(bytes,{paging,cpu_module:conversion_baseline});c.wasm.vine_set_x87_mode(2);all.push(c);}
 return all;
}
function compare_conversion(all,budget,label,fullFault=false) {
 const results=all.map(c=>c.execute(budget));
 const snapshots=all.map(c=>c.full_snapshot());
 for(let i=1;i<all.length;i++) {
  assert.deepEqual(results[0],results[i],label+" boundary");
  const state=s=>fullFault?s.state:s.state.filter((_,i)=>i!==11);
  assert.deepEqual(state(snapshots[0]),state(snapshots[i]),label+" raw state");
  assert.deepEqual(snapshots[0].memory,snapshots[i].memory,label+" memory");
  assert.deepEqual(snapshots[0].dirty,snapshots[i].dirty,label+" dirty");
 }
 return results[0];
}
function integer_oracle(x,truncate) {
 const f=Math.floor(x),d=x-f;
 const rounded=truncate?Math.trunc(x):d<0.5?f:d>0.5?f+1:f%2===0?f:f+1;
 return rounded>=-2147483648&&rounded<2147483648?rounded|0:-2147483648;
}
const conversion_values=[-0,0,0.5,-0.5,1.5,-1.5,2.5,-2.5,1.25,12345.75,
 2147483647.49,2147483647.5,2147483648,-2147483648,-2147483648.5,-2147483648.75,
 Number.MIN_VALUE,-Number.MIN_VALUE,1e-307,1e300,Infinity,-Infinity,NaN];
let conversion_cases=0;
for(const group of [1,2,3]) {
 const body=[0xdd,0x47,16,0xdc,0x4f,24,0xdd,0x5f,32,0xdd,0x47,32,
  0xdb,0x47|group<<3,48,...(group===2?[0xdd,0x5f,40]:[])];
 const all=conversion_cpus([...body,0xff,0xe1]);
 for(const value of conversion_values) {
  for(const cpu of all){seed(cpu);cpu.physical.setFloat64(DATA+16,value,true);cpu.physical.setFloat64(DATA+24,1,true);}
  compare_conversion(all,120000,"m64 / FIST group "+group+" value "+value);
  // Execute one whole loop from a fresh architectural state with warm code.
  for(const cpu of all){seed(cpu);cpu.physical.setFloat64(DATA+16,value,true);cpu.physical.setFloat64(DATA+24,1,true);}
  compare_conversion(all,group===2?7:6,"m64 conversion oracle");
  assert.equal(all[0].physical.getInt32(DATA+48,true),integer_oracle(value,group===1));
  if(!Number.isNaN(value))assert(Object.is(all[0].physical.getFloat64(DATA+32,true),value),"m64 spill retains value/sign");
  conversion_cases++;
 }
 for(const cw of [0x37f,0x27f,0x7f,0x77f,0xb7f,0xf7f,0x35f]) {
  for(const cpu of all){seed(cpu,undefined,cw);cpu.physical.setFloat64(DATA+16,2.5,true);cpu.physical.setFloat64(DATA+24,1,true);}
  compare_conversion(all,120000,"control "+cw+" group "+group);
 }
 for(const cpu of all){seed(cpu);cpu.physical.setFloat64(DATA+16,1.25,true);cpu.physical.setFloat64(DATA+24,1.5,true);}
 for(const budget of [120000,1,2,3,4,5,7,4097,131072])compare_conversion(all,budget,"conversion exact budget "+budget);
 assert(all[0].wasm.vine_x87_fast_instructions()>0,"m64/integer conversion run actually executes");
}

// Every m64 arithmetic direction, payload-preserving exceptional conversion,
// and overlapping m32/m64 EAs. An 8-byte guard must cover the whole access.
for(const group of [0,1,4,5]) {
 const all=conversion_cpus([0xdd,0x47,16,0xdc,0x47|group<<3,24,0xdd,0x5f,32,0xff,0xe1]);
 for(const cpu of all){seed(cpu);cpu.physical.setFloat64(DATA+16,1.25,true);cpu.physical.setFloat64(DATA+24,1.5,true);}
 compare_conversion(all,120000,"m64 arithmetic "+group);
 const expected=group===0?2.75:group===1?1.875:group===4?-0.25:0.25;
 assert.equal(all[0].physical.getFloat64(DATA+32,true),expected);
 for(const bits of [0x7ff0000000000001n,0xfff8000000001234n,0x0000000000000001n]){
  for(const cpu of all){seed(cpu);cpu.physical.setBigUint64(DATA+16,bits,true);cpu.physical.setFloat64(DATA+24,1,true);}
  compare_conversion(all,120000,"m64 exceptional payload "+bits);
 }
}
const spill_code=[0xd9,7,0xd8,0x4f,4,0xdd,0x1b,0xdd,3,0xdb,0x1a,0xff,0xe1];
for(const spill of [DATA+32,DATA,DATA+4,DATA+4092]) {
 const all=conversion_cpus(spill_code);
 for(const cpu of all){seed(cpu);cpu.set_register(3,0x400000+spill);cpu.set_register(2,0x400000+DATA+48);}
 for(let round=0;round<3;round++){
  compare_conversion(all,120000,"mixed-width alias "+spill);
  for(const cpu of all)cpu.clear_dirty([DATA,DATA+4096]);
 }
}
for(const fault of ["spill","integer","readonly","cross-page-unmapped"]) {
 const all=conversion_cpus(spill_code);
 for(const cpu of all){
  seed(cpu);cpu.set_register(3,0x400000+DATA+32);cpu.set_register(2,0x400000+DATA+48);
  cpu.execute(120000);seed(cpu);cpu.restart_at_entry();
  if(fault==="spill")cpu.set_register(3,0x70000000);
  if(fault==="integer")cpu.set_register(2,0x70000000);
  if(fault==="readonly")cpu.protect_data_page();
  if(fault==="cross-page-unmapped"){
   cpu.set_register(3,0x400000+DATA+4092);
   cpu.physical.setUint32(0x11000+((DATA+4096)>>>12)*4,0,true);cpu.wasm.full_clear_tlb();
  }
 }
 assert.equal(compare_conversion(all,100,"ordered conversion fault "+fault,true).reason,0x10e);
}
console.log(`x87 m64/integer runs: ${conversion_cases} values, ties/range oracle, controls, aliases, whole-access guards, exact budgets and ordered fault state passed`);

// Pure load/copy/sign/store runs have no binary arithmetic. Keep the source
// sign and copied bits, including signed zero and exceptional payloads.
const copy_code=[0xd9,7,0x83,0xc7,4,0xd9,0x57,8,0xd9,0xe0,0xd9,0x5f,12,
 0x83,0xef,4,0xff,0xe1];
const copy_cpus=conversion_cpus(copy_code);
for(const bits of [0,0x80000000,0x3fa00000,0xbfa00000,1,0x80000001,
 0x7f7fffff,0x7f800000,0xff800000,0x7fc01234,0x7f801234]) {
 for(const cpu of copy_cpus){seed(cpu);cpu.physical.setUint32(DATA,bits,true);}
 compare_conversion(copy_cpus,120000,"pure copy/sign "+bits.toString(16));
 for(const cpu of copy_cpus){seed(cpu);cpu.physical.setUint32(DATA,bits,true);}
 compare_conversion(copy_cpus,7,"copy/sign numeric oracle");
 if((bits&0x7fffffff)<=0x7f800000){
  assert.equal(copy_cpus[0].physical.getUint32(DATA+12,true),bits>>>0);
  assert.equal(copy_cpus[0].physical.getUint32(DATA+16,true),(bits^0x80000000)>>>0);
 }
}
for(const cpu of copy_cpus)seed(cpu);
for(const budget of [120000,0,1,2,3,4,5,7,4095,4096,4097,131072])
 compare_conversion(copy_cpus,budget,"copy/sign budget "+budget);
assert(copy_cpus[0].wasm.vine_x87_fast_instructions()>0,"pure copy/sign value run executes");
console.log("x87 copy/sign runs: numeric/sign oracle, exceptional payloads and exact budgets passed");

// Comparison keeps x87 condition codes distinct from lazy integer flags. The
// helper and unchanged CPU are references; normal comparisons and quiet-NaN
// behavior also have an independent ordering/status oracle.
const comparison_forms=[
 ["FCOM",[0xd8,0xd1],0,false],["FCOMP",[0xd8,0xd9],1,false],
 ["FCOM alias",[0xdc,0xd1],0,false],["FCOMP alias",[0xdc,0xd9],1,false],
 ["FCOMPP",[0xde,0xd9],2,false],["FUCOM",[0xdd,0xe1],0,true],
 ["FUCOMP",[0xdd,0xe9],1,true],["FUCOMPP",[0xda,0xe9],2,true],
 ["FCOM m32",[0xd8,0x57,16],0,false,4],["FCOMP m32",[0xd8,0x5f,16],1,false,4],
 ["FCOM m64",[0xdc,0x57,16],0,false,8],["FCOMP m64",[0xdc,0x5f,16],1,false,8],
];
const comparison_values=[[1.5,2.5],[-2,1],[2.5,1.5],[1.5,1.5],[-0,0],
 [Number.MIN_VALUE,-Number.MIN_VALUE],[Infinity,Infinity],[Infinity,-Infinity],
 [NaN,1],[1,NaN],[NaN,NaN]];
let comparison_cases=0;
for(const [name,op,pops,quiet,width] of comparison_forms){
 const cleanup=Array.from({length:2-pops},()=>[0xdd,0xd8]).flat();
 const body=[0xdd,7,0xdd,0x47,8,...op,0xdf,0xe0,0x89,0xc3,...cleanup,0xff,0xe1];
 const all=conversion_cpus(body);
 function initialize(cpu,a,b,top=7){
  seed(cpu);cpu.physical.setFloat64(DATA,a,true);cpu.physical.setFloat64(DATA+8,b,true);
  if(width===4)cpu.physical.setFloat32(DATA+16,a,true);else cpu.physical.setFloat64(DATA+16,a,true);
  const view=new DataView(cpu.wasm.memory.buffer);view.setUint8(1032,top);view.setUint16(1040,0x280,true);
 }
 for(const [a,b] of comparison_values){
  for(const cpu of all)initialize(cpu,a,b);
  compare_conversion(all,120000,name+" "+a+" "+b);
  for(const top of [0,3,7]){
   for(const cpu of all)initialize(cpu,a,b,top);
   const result=compare_conversion(all,8-pops,name+" oracle TOP "+top);
   const other=width===4?Math.fround(a):a;
   const unordered=Number.isNaN(b)||Number.isNaN(other);
   const condition=unordered?0x4500:b<other?0x100:b===other?0x4000:0;
   const expected=0x80|condition|((top+6+pops)&7)<<11|(unordered&&!quiet?1:0);
   assert.equal(result.registers[3]&0xffff,expected,name+" independent status oracle");
   comparison_cases++;
  }
 }
 for(const cpu of all)initialize(cpu,1.25,2.5);
 for(const budget of [120000,0,1,2,3,4,5,7,4095,4096,4097,131072])
  compare_conversion(all,budget,name+" exact budget "+budget);
 assert(all[0].wasm.vine_x87_fast_instructions()>0,name+" executes a value run");
}

// Two comparisons with FNSTSW/TEST between them: the final integer branch
// observes the first comparison while AX observes the second one.
const status_code=[0xd9,7,0xd9,0x47,4,0xd9,0xc1,0xd8,0x5f,8,
 0xdf,0xe0,0x89,0xc3,0xd8,0x57,8,0xf6,0xc4,0x41,0xdf,0xe0,
 0xdd,0xd8,0xdd,0xd8,0x0f,0x95,0xc2,0xff,0xe1];
const status_cpus=conversion_cpus(status_code);
for(const [a,b] of [[-2,3],[3,-2],[0,0],[NaN,1],[1,NaN]]){
 for(const cpu of status_cpus)seed(cpu,[a,b,0]);
 compare_conversion(status_cpus,120000,"two comparison status "+a+" "+b);
 for(const cpu of status_cpus)seed(cpu,[a,b,0]);
 const r=compare_conversion(status_cpus,13,"two comparison independent condition");
 const condition=x=>Number.isNaN(x)?0x4500:x<0?0x100:x===0?0x4000:0;
 assert.equal(r.registers[3]&0x4500,condition(a));
 assert.equal(r.registers[0]&0x4500,condition(b));
 assert.equal(r.registers[2]&255,Number((condition(a)&0x4100)!==0));
}
// AX can be observed before the first push clears C1, then after it. A status
// write invalidates any affine EA using EAX until a full known MOV replaces it.
const status_ea_bodies=[
 [0xdf,0xe0,0x89,0xc3,0xd9,7,0xd9,0xe0,0xd9,0x5f,12,0xdf,0xe0],
 [0xd9,7,0xd9,0x47,4,0xd8,0xd9,0xdf,0xe0,0xd9,0x18],
 [0xd9,7,0xd9,0x47,4,0xd8,0xd9,0xdf,0xe0,0x89,0xc3,0xd9,0x1b],
 [0xd9,7,0xd9,0x47,4,0xd8,0xd9,0xdf,0xe0,0x83,0xc0,8,0xd9,0x18],
 [0xd9,7,0xd9,0x47,4,0xd8,0xd9,0xdf,0xe0,0x89,0xf8,0xd9,0x58,12],
];
for(const [index,body] of status_ea_bodies.entries()){
 const all=conversion_cpus([...body,0xff,0xe1]);
 for(const cpu of all){seed(cpu);cpu.set_register(0,0x408000);new DataView(cpu.wasm.memory.buffer).setUint16(1040,0x200,true);}
 for(const budget of [120000,1,2,4,7,4097])compare_conversion(all,budget,"status EA "+index);
 if(index===0){
  for(const cpu of all){seed(cpu);new DataView(cpu.wasm.memory.buffer).setUint16(1040,0x200,true);}
  const r=compare_conversion(all,7,"status before/after push oracle");
  assert.equal(r.registers[3]&0xffff,0x3a00);assert.equal(r.registers[0]&0xffff,0x3800);
 }
}

// FST/FSTP ST(i) may fill an empty destination and may alias ST(0).
for(const pop of [false,true]) for(let r=0;r<8;r++){
 const all=conversion_cpus([0xd9,7,0xd9,0x47,4,0xdd,(pop?0xd8:0xd0)|r,
  0xdf,0xe0,0xdd,0xd8,...(pop?[]:[0xdd,0xd8]),0xff,0xe1]);
 for(const cpu of all)seed(cpu);
 compare_conversion(all,120000,"stack copy "+r+" pop "+pop);
 for(const top of [0,3,7]){
  for(const cpu of all){seed(cpu);new DataView(cpu.wasm.memory.buffer).setUint8(1032,top);}
  compare_conversion(all,pop?6:7,"stack copy TOP "+top);
 }
 assert(all[0].wasm.vine_x87_fast_instructions()>0,"stack copy value run executes");
}

// A later comparison read fault must preserve all earlier loads and status.
// Both narrow and page-straddling operands use the original slow instruction.
for(const width of [4,8]) for(const address of [0x70000000,0x408ffd]){
 const all=conversion_cpus([0xd9,7,0xd9,0x47,4,width===4?0xd8:0xdc,0x1b,
  0xdf,0xe0,0xdd,0xd8,0xff,0xe1]);
 for(const cpu of all){
  seed(cpu);cpu.set_register(3,0x408008);cpu.execute(120000);seed(cpu);cpu.set_register(3,address);
  if(address!==0x70000000){cpu.physical.setUint32(0x11000+9*4,0,true);cpu.wasm.full_clear_tlb();}
 }
 assert.equal(compare_conversion(all,100,"comparison ordered fault "+width+" "+address,true).reason,0x10e);
}
console.log(`x87 comparison/stack runs: ${comparison_cases} ordering/status oracles, quiet NaNs, TOP/tags, scratch, AX/flags/EAs, budgets and ordered faults passed`);

// A preexisting raw80 value must not be rounded to binary64 merely to compare
// it. This includes a value one extended-precision unit above exactly one.
for(const quiet of [false,true]){
 // FUCOM (DD /4) is an existing block boundary because the same group also
 // contains FRSTOR. FUCOMP can retain a copied raw80 operand in one value run.
 const all=conversion_cpus([0xd9,0xc0,quiet?0xdd:0xd8,quiet?0xea:0xda,
  0xdf,0xe0,0xdd,0xd0,0xff,0xe1]);
 const encodings=[[0x8000000000000001n,0x3fff],[0x8000000000000000n,0xbfff],
  [1n,0],[0x8000000000000000n,0x7fff],[0xc000000000001234n,0x7fff],
  [0x8000000000001234n,0xffff],[0x4000000000000000n,0x3fff]];
 for(const [mantissa,exponent] of encodings){
  for(const cpu of all){
   seed(cpu);const v=new DataView(cpu.wasm.memory.buffer);v.setUint8(816,0xe7);v.setUint8(1032,3);
   v.setBigUint64(1152+3*16,mantissa,true);v.setUint16(1160+3*16,exponent,true);
   v.setBigUint64(1152+4*16,0x8000000000000000n,true);v.setUint16(1160+4*16,0x3fff,true);
  }
  const r=compare_conversion(all,120000,"raw80 compare "+quiet+" "+mantissa+" "+exponent);
  if(mantissa===0x8000000000000001n&&exponent===0x3fff)
   assert.equal(r.registers[0]&0x4500,0,"raw80 above-one stays greater, not rounded equal");
 }
 assert(all[0].wasm.vine_x87_fast_instructions()>0,"raw80 comparison value run executes");
}
for(const cw of [0x37f,0x27f,0x7f,0x77f,0xb7f,0xf7f,0x35f]){
 for(const cpu of status_cpus)seed(cpu,[-2,3,0],cw);
 compare_conversion(status_cpus,120000,"cached comparison control "+cw);
}
console.log("x87 comparison fallback: raw80 precision/encodings and runtime control changes passed");

// RAM INC/DEC must retain the intervening x87 values without changing integer
// flags, aliases, dirty tracking or the first fault. The unchanged CPU and
// helper-only JIT retain the original memory path; flags have an independent
// arithmetic oracle as well.
function seed_rmw(cpu,value,carry,address=DATA+64) {
 seed(cpu);cpu.set_register(3,0x400000+address);
 cpu.physical.setUint32(address,value,true);
 const v=new DataView(cpu.wasm.memory.buffer);
 v.setUint32(100,0,true);v.setUint32(120,0x202|carry,true);
}
function rmw_flags(value,decrement,carry) {
 const result=(value+(decrement?-1:1))>>>0;
 let parity=0;for(let i=0;i<8;i++)parity^=result>>>i&1;
 return carry|((parity^1)<<2)|(((value^result^1)&16))|
  (result===0?64:0)|(result>>>31?128:0)|
  (value===(decrement?0x80000000:0x7fffffff)?2048:0);
}
let rmw_cases=0;
for(const decrement of [false,true]) {
 const bytes=[0xd9,7,0xd8,0x4f,4,0xff,decrement?0x0b:0x03,0xd9,0x5f,12,0xff,0xe1];
 const all=conversion_cpus(bytes);
 for(const cpu of all)seed_rmw(cpu,123,1);
 compare_conversion(all,120000,"RAM RMW warmup");
 for(const value of [0,1,15,16,0xff,0x100,0x7fffffff,0x80000000,0xffffffff]) for(const carry of [0,1]) {
  for(const cpu of all)seed_rmw(cpu,value,carry);
  const before=all[0].wasm.vine_x87_fast_instructions();
  const result=compare_conversion(all,5,"RAM RMW oracle "+decrement+" "+value+" "+carry);
  assert.equal(all[0].physical.getUint32(DATA+64,true),(value+(decrement?-1:1))>>>0);
  assert.equal(result.flags&0x8d5,rmw_flags(value,decrement,carry));
  assert.equal(all[0].wasm.vine_x87_fast_instructions()-before,3,"three x87 instructions stay in value run");
  rmw_cases++;
 }
 for(const budget of [1,2,3,4,7,4095,4096,4097,131072])compare_conversion(all,budget,"RAM RMW budget "+budget);
 // Overlap before and after the integer write, including m64 high halves.
 const alias=conversion_cpus([0xdd,0x47,16,0xdd,0x57,32,0xff,decrement?0x0b:0x03,
  0xdd,0x47,32,0xde,0xc1,0xdd,0x5f,40,0xff,0xe1]);
 for(const address of [DATA+32,DATA+36,DATA+33,DATA+40,DATA+4092,DATA+4093]) {
  for(const cpu of alias){seed_rmw(cpu,2,1,address);cpu.physical.setFloat64(DATA+16,1.25,true);}
  compare_conversion(alias,120000,"RAM RMW overlap "+address);
  for(const cpu of alias)cpu.clear_dirty([DATA,DATA+4096]);
  compare_conversion(alias,120000,"RAM RMW dirty rearm "+address);
 }
 for(const fault of ["rmw-unmapped","rmw-readonly","rmw-cross-page","later-store"]) {
  const group=conversion_cpus([0xd9,7,0xd8,0x4f,4,0xff,decrement?0x0b:0x03,0xd9,0x1a,0xff,0xe1]);
  for(const cpu of group){seed_rmw(cpu,2,1);cpu.set_register(2,0x400000+DATA+12);}
  compare_conversion(group,120000,"RAM RMW fault warmup");
  for(const cpu of group){
   seed_rmw(cpu,2,1,fault==="rmw-cross-page"?DATA+4093:DATA+64);
   if(fault==="rmw-unmapped")cpu.set_register(3,0x70000000);
   if(fault==="rmw-readonly")cpu.protect_data_page();
   if(fault==="rmw-cross-page"){
    cpu.physical.setUint32(0x11000+9*4,0,true);cpu.wasm.full_clear_tlb();
   }
   if(fault==="later-store")cpu.set_register(2,0x70000000);
  }
  const count=group[0].wasm.vine_x87_fast_instructions();
  assert.equal(compare_conversion(group,100,"RAM RMW ordered fault "+fault,true).reason,0x10e);
  assert.equal(group[0].wasm.vine_x87_fast_instructions(),count,"fault must decline entire value run");
 }
}
// A code-page write must still retire generated code, even when it does not
// touch this run's instructions. An MMIO callback must see already published
// x87 state and any CR0 change must be checked by the next x87 instruction.
{
 const bytes=[0xd9,7,0xd8,0x4f,4,0xff,0x0b,0xd9,0x5f,12,0xff,0xe1];
 const group=conversion_cpus(bytes);
 for(const c of group)seed_rmw(c,2,1);
 compare_conversion(group,120000,"RAM RMW code-page warmup");
 const count=group[0].wasm.vine_x87_fast_instructions(),invalidations=group[0].counts.invalidations;
 for(const c of group)seed_rmw(c,2,1,0x1800);
 compare_conversion(group,5,"RAM RMW code-page write");
 assert.equal(group[0].wasm.vine_x87_fast_instructions(),count);
 assert(group[0].counts.invalidations>invalidations,"code write retires dependent modules");
 for(const change_control of [false,true]) {
  const notes=[];
  const cpus=[3,1,...(conversion_baseline?[2]:[])].map((mode,index)=>{
   notes[index]=[];
   const c=create_cpu(bytes,{...(index===2?{cpu_module:conversion_baseline}:{}),
    on_mmap(name,args,wasm){
     const view=new DataView(wasm.memory.buffer);
     notes[index].push({name,args,top:view.getUint8(1032),tags:view.getUint8(816),
      stack:Array.from(new Uint8Array(wasm.memory.buffer,1152,128))});
     if(change_control&&name==="mmap_write32")view.setUint32(580,view.getUint32(580,true)|8,true);
     return name==="mmap_read32"?7:undefined;
    }});
   c.wasm.vine_set_x87_mode(mode);return c;
  });
  for(const c of cpus){seed(c);c.set_register(3,DATA+64);c.physical.setUint32(DATA+64,7,true);}
  compare_conversion(cpus,120000,"RAM RMW MMIO warmup");
  for(const c of cpus){seed(c);c.set_register(3,0xa0000);}
  const count=cpus[0].wasm.vine_x87_fast_instructions();
  const result=compare_conversion(cpus,5,"RAM RMW MMIO "+change_control,change_control);
  assert.equal(result.reason,change_control?0x107:1);
  assert.equal(cpus[0].wasm.vine_x87_fast_instructions(),count);
  assert(notes[0].some(n=>n.name==="mmap_write32"));
  for(const n of notes.slice(1))assert.deepEqual(notes[0],n,"MMIO observes architectural x87 state");
 }
}
console.log(`x87 RAM INC/DEC: ${rmw_cases} independent flag/value cases, aliases, page edge, dirty rearm, budgets, ordered faults, code retirement and MMIO publication/control passed`);

// An untaken edge to FSTP splits a load/arithmetic/store sequence after the
// arithmetic. Both entry points and their original budget/fault boundaries
// must remain usable when the first two operations retain their value.
{
 const bytes=[0x31,0xc0,0x85,0xc0,0x75,5,0xd9,7,0xd8,0x4f,4,0xd9,0x1b,0xff,0xe1];
 const cpus=conversion_cpus(bytes);
 for(const c of cpus){seed(c);c.set_register(3,0x400000+DATA+12);}
 compare_conversion(cpus,120000,"split load/arithmetic warmup");
 for(const c of cpus){seed(c);c.set_register(3,0x400000+DATA+12);}
 const count=cpus[0].wasm.vine_x87_fast_instructions();
 compare_conversion(cpus,7,"split load/arithmetic one loop");
 assert.equal(cpus[0].wasm.vine_x87_fast_instructions()-count,2);
 for(const budget of [1,2,3,4,5,6,7,4097,131072])compare_conversion(cpus,budget,"split load/arithmetic budget "+budget);
 for(const c of cpus)seed(c);
 compare_conversion(cpus,5,"split run before interior entry");
 for(const c of cpus)c.resume(null,0x100b);
 compare_conversion(cpus,2,"split run direct FSTP entry");
 for(const c of cpus){seed(c);c.set_register(3,0x70000000);}
 assert.equal(compare_conversion(cpus,100,"split run later store fault",true).reason,0x10e);
 for(const c of cpus){
  seed(c);c.set_register(7,0x400000+DATA+4092);
  c.physical.setFloat32(DATA+4092,1.25,true);
  c.physical.setUint32(0x11000+9*4,0,true);c.wasm.full_clear_tlb();
 }
 assert.equal(compare_conversion(cpus,100,"split run second load fault",true).reason,0x10e);
}
console.log("x87 split load/arithmetic: both entries, exact budgets, second read and later store faults passed");

// ===== Typed f64 twin (JIT config index 10) =====
// The typed twin must produce bit-identical architectural state to the
// dynamic value-run sequence and the helper/interpreter paths for every
// input, including every bail kind at every position inside a run.
assert.equal(create_cpu([0xff,0xe1]).wasm.get_jit_config(10),1,"typed twin defaults on");
function typed_cpus(bytes,paging=true){
 const all=conversion_cpus(bytes,paging);
 const off=create_cpu(bytes,{paging});
 off.wasm.set_jit_config(10,0);
 off.wasm.vine_set_x87_mode(3);
 all.push(off);
 return all;
}
// Write a raw80 slot holding an ordinary binary64 so entry recovery publishes
// the SHADOW form the typed twin accepts.
function slot80(view,register,x){
 const bits=new BigUint64Array(new Float64Array([x]).buffer)[0];
 const exponent=Number(bits>>52n&0x7ffn);
 if(exponent===0)view.setBigUint64(1152+register*16,0n,true);
 else view.setBigUint64(1152+register*16,(bits&0xfffffffffffffn)<<11n|1n<<63n,true);
 view.setUint16(1160+register*16,Number(bits>>63n)<<15|(exponent===0?0:exponent+0x3c00),true);
}
function seed_full(cpu,st,top=4){
 seed(cpu);
 const view=new DataView(cpu.wasm.memory.buffer);
 view.setUint8(1032,top);view.setUint8(816,0xff&~(1<<top));
 slot80(view,top,st);
}

// MDCT-shaped butterflies: fld/fsub/fadd/fmul/fstp on m32 operands, repeated
// in one straight-line run.
const butterfly=[];
for(let i=0;i<4;i++){
 const b=i*20;
 butterfly.push(
  0xd9,0x47,b,      // fld  dword [bx+b]
  0xd8,0x67,b+4,    // fsub dword [bx+b+4]   (D8 /4)
  0xd8,0x47,b+8,    // fadd dword [bx+b+8]   (D8 /0)
  0xd8,0x4f,b+12,   // fmul dword [bx+b+12]  (D8 /1)
  0xd9,0x5f,b+16,   // fstp dword [bx+b+16]  (D9 /3)
 );
}
{
 const all=typed_cpus([...butterfly,0xff,0xe1]);
 for(const c of all){
  seed(c);c.set_register(3,0x400000+DATA);
  for(let i=0;i<20;i++)c.physical.setFloat32(DATA+i*4,(i%7-3)*0.5+0.125*i,true);
 }
 for(const budget of [120000,1,2,3,4,5,7,9,13,17,21,4097])
  compare_conversion(all,budget,"typed butterfly budget "+budget);
 assert(all[0].wasm.vine_x87_fast_instructions()>0,"butterfly typed run executes");
}
console.log("x87 typed twin: m32 butterfly run, knob off parity and exact budgets passed");

// Non-ordinary memory operands bail before the op mutates anything; the
// trigger can sit at the first, a middle or the last op of the run.
const m32_body=[0xd9,0x47,0,0xd8,0x47,4,0xd8,0x47,8,0xd8,0x47,12,0xff,0xe1];
const m64_body=[0xdd,0x47,0,0xdc,0x47,8,0xdc,0x47,16,0xdc,0x47,24,0xff,0xe1];
const m32_values=[Infinity,-Infinity,NaN,1e-40,0,-0];
const m64_values=[Infinity,-Infinity,NaN,5e-324,0,-0];
let typed_bail_cases=0;
for(const offset of [0,4,8,12]){
 const all=typed_cpus(m32_body);
 for(const value of m32_values){
  for(const c of all){
   seed(c);c.set_register(3,0x400000+DATA);
   for(let i=0;i<16;i+=4)c.physical.setFloat32(DATA+i,1.25+i*0.5,true);
   c.physical.setFloat32(DATA+offset,value,true);
  }
  compare_conversion(all,120000,"typed m32 operand bail @"+offset+" "+value);
  typed_bail_cases++;
 }
}
for(const offset of [0,8,16,24]){
 const all=typed_cpus(m64_body);
 for(const value of m64_values){
  for(const c of all){
   seed(c);c.set_register(3,0x400000+DATA);
   for(let i=0;i<32;i+=8)c.physical.setFloat64(DATA+i,1.25+i*0.25,true);
   c.physical.setFloat64(DATA+offset,value,true);
  }
  compare_conversion(all,120000,"typed m64 operand bail @"+offset+" "+value);
  typed_bail_cases++;
 }
}
// Overflow to infinity, nonzero subnormal results and underflow-to-zero in
// multiplication bail at the first, a middle and the last arithmetic op.
const add_body=[0xdc,0x47,0,0xdc,0x47,8,0xdc,0x47,16,0xdd,0x5f,24,0xff,0xe1];
const mul_body=[0xdc,0x4f,0,0xdc,0x4f,8,0xdc,0x4f,16,0xdd,0x5f,24,0xff,0xe1];
for(const [name,body,st,vals] of [
 ["add overflow",add_body,1e308,[[1e308,1,1],[1,1e308,1],[1,1,1e308]]],
 ["mul overflow",mul_body,1e200,[[1e200,1,1],[1,1e200,1],[1,1,1e200]]],
 ["mul subnormal",mul_body,2.2250738585072014e-308,[[0.5,1,1],[1,0.5,1],[1,1,0.5]]],
 ["mul zero",mul_body,1e-200,[[1e-200,1,1],[1,1e-200,1],[1,1,1e-200]]],
]){
 const all=typed_cpus(body);
 for(const [v0,v1,v2] of vals){
  for(const c of all){
   seed_full(c,st);c.set_register(3,0x400000+DATA);
   c.physical.setFloat64(DATA,v0,true);c.physical.setFloat64(DATA+8,v1,true);
   c.physical.setFloat64(DATA+16,v2,true);
  }
  compare_conversion(all,120000,"typed "+name+" "+v0+" "+v1+" "+v2);
  typed_bail_cases++;
 }
}
// f32 store overflow/underflow and out-of-range integer stores bail before the
// guest store; first, middle and last store positions all resume exactly.
const store_body=[0xdd,0x47,0,0xd9,0x5f,8,0xdd,0x47,16,0xd9,0x5f,24,0xff,0xe1];
for(const [label,v0,v1] of [["overflow",1e300,2],["underflow",1e-300,2],
 ["first",1e300,2],["middle",2,1e300]]){
 const all=typed_cpus(store_body);
 for(const c of all){
  seed(c);c.set_register(3,0x400000+DATA);
  c.physical.setFloat64(DATA,v0,true);c.physical.setFloat64(DATA+16,v1,true);
 }
 compare_conversion(all,120000,"typed f32 store "+label);
 typed_bail_cases++;
}
// A store bail at literal op 0: the stack starts full and the demote fails.
{
 const all=typed_cpus([0xd9,0x5f,0,0xdd,0x47,8,0xdd,0x5f,16,0xff,0xe1]);
 for(const c of all){
  seed_full(c,1e300);c.set_register(3,0x400000+DATA);c.physical.setFloat64(DATA+8,2,true);
 }
 compare_conversion(all,120000,"typed f32 store bail at first op");
 typed_bail_cases++;
}
const fist_body=[0xdd,0x47,0,0xdb,0x57,8,0xdc,0x47,16,0xdb,0x5f,24,0xff,0xe1];
for(const [label,v0,v1] of [["middle",3e9,2],["last",2,3e9]]){
 const all=typed_cpus(fist_body);
 for(const c of all){
  seed(c);c.set_register(3,0x400000+DATA);
  c.physical.setFloat64(DATA,v0,true);c.physical.setFloat64(DATA+16,v1,true);
 }
 compare_conversion(all,120000,"typed fist bail "+label);
 typed_bail_cases++;
}
{
 const all=typed_cpus([0xdb,0x57,0,0xdd,0x47,8,0xdd,0x5f,16,0xff,0xe1]);
 for(const c of all){
  seed_full(c,3e9);c.set_register(3,0x400000+DATA);c.physical.setFloat64(DATA+8,2,true);
 }
 compare_conversion(all,120000,"typed fist bail at first op");
 typed_bail_cases++;
}
// A page fault inside a typed run and budget stops at every op boundary keep
// the original slow-path behaviour.
{
 const all=typed_cpus(m32_body);
 for(const c of all){seed(c);c.set_register(3,0x400000+DATA);}
 for(const budget of [1,2,3,4,5,6])compare_conversion(all,budget,"typed m32 budget "+budget);
}
{
 const all=typed_cpus([0xd9,0x47,0,0xd8,0x47,4,0xd9,0x1b,0xff,0xe1]);
 for(const c of all){
  seed(c);c.set_register(3,0x400000+DATA);c.execute(120000);
  seed(c);c.set_register(3,0x70000000);
 }
 assert.equal(compare_conversion(all,100,"typed interior store fault",true).reason,0x10e);
}
{
 const all=typed_cpus([0xd9,0x47,0,0xd8,0x47,4,0xd8,0x1b,0xff,0xe1]);
 for(const c of all){
  seed(c);c.set_register(3,0x400000+DATA);c.execute(120000);
  seed(c);c.set_register(3,0x70000000);
 }
 assert.equal(compare_conversion(all,100,"typed interior load fault",true).reason,0x10e);
}
console.log(`x87 typed twin: ${typed_bail_cases} bail cases at first/middle/last ops, knob-off parity, budgets and interior faults passed`);

// ===== Round 2: shared guards, plan-limit knobs and tag masks =====

// Two addresses with identical register coefficients inside one 64-byte
// window share a single page guard. A group whose union extent straddles a
// page boundary must fail the shared guard exactly like an individual guard
// failure, and a write member makes the whole group require write access.
{
 // fld [ebx+8] fadd [ebx+24] fstp [ebx+40]: the union 0x2ff8..0x301c
 // straddles the page boundary at 0x3000, so the run cannot start.
 const all=typed_cpus([0xd9,0x43,8,0xd8,0x43,24,0xd9,0x5b,40,0xff,0xe1],false);
 for(const c of all){
  seed(c);c.set_register(3,0x2ff0);
  c.physical.setFloat32(0x2ff8,1.5,true);c.physical.setFloat32(0x3004,2.25,true);
 }
 compare_conversion(all,120000,"shared guard straddling pair");
}
{
 // fld [ebx+8] fstp [ebx+24] both inside the data page; protecting the page
 // must fail the shared guard's union write check, never a silent store.
 const all=typed_cpus([0xd9,0x43,8,0xd9,0x5b,24,0xff,0xe1],true);
 for(const c of all){
  seed(c);c.set_register(3,0x400000+DATA-8);
  c.physical.setFloat32(DATA,1.5,true);c.physical.setFloat32(DATA+16,0.5,true);
  c.protect_data_page();
 }
 assert.equal(compare_conversion(all,100,"shared guard non-writable member",true).reason,0x10e);
}
console.log("x87 shared guards: straddling group and non-writable member fall back exactly");

// Plan limits are runtime knobs: index 11 = maximum ops per value run
// (default 64, clamp 3..128), index 12 = maximum distinct addresses
// (default 24, clamp 1..40).
{
 const probe=create_cpu([0xff,0xe1]);
 assert.equal(probe.wasm.get_jit_config(11),64,"ops limit defaults to 64");
 assert.equal(probe.wasm.get_jit_config(12),24,"address limit defaults to 24");
 probe.wasm.set_jit_config(11,1);assert.equal(probe.wasm.get_jit_config(11),3,"ops clamp low");
 probe.wasm.set_jit_config(11,200);assert.equal(probe.wasm.get_jit_config(11),128,"ops clamp high");
 probe.wasm.set_jit_config(12,0);assert.equal(probe.wasm.get_jit_config(12),1,"addresses clamp low");
 probe.wasm.set_jit_config(12,99);assert.equal(probe.wasm.get_jit_config(12),40,"addresses clamp high");
}
function limit_cpus(bytes,ops_limit,address_limit){
 const all=typed_cpus(bytes,true);
 for(const c of all){c.wasm.set_jit_config(11,ops_limit);c.wasm.set_jit_config(12,address_limit);}
 return all;
}
function long_body(pairs,addresses){
 const body=[];
 for(let i=0;i<pairs;i++){const o=(i%addresses)*4;body.push(0xd9,0x43,o,0xd9,0x5b,o);}
 body.push(0xff,0xe1);return body;
}
function seed_long(c,addresses,base=0x40000){
 seed(c);c.set_register(3,0x400000+base);
 for(let i=0;i<addresses*4;i+=4)c.physical.setFloat32(base+i,(i%9-4)*0.25+1,true);
}
// Runs longer than the default 32 ops need the raised limits: 64 ops over 24
// distinct addresses and 96 ops over 32, exact budgets at every boundary.
for(const [pairs,addresses,label] of [[32,24,"64 ops/24 addresses"],[48,32,"96 ops/32 addresses"]]){
 const all=limit_cpus(long_body(pairs,addresses),pairs*2,addresses);
 for(const c of all)seed_long(c,addresses);
 for(const budget of [131072,1,5,7,33,65,4097])
  compare_conversion(all,budget,"typed long "+label+" budget "+budget);
 assert(all[0].wasm.vine_x87_fast_instructions()>0,label+" executes a fast run");
}
// A bail at the literal last op of a full run: the final fmul overflows to
// infinity from two nonzero operands.
for(const [pairs,addresses,label] of [[32,24,"64 ops"],[48,32,"96 ops"]]){
 const last_offset=(addresses-1)*4;
 const body=long_body(pairs-1,addresses);
 body.splice(-2,0,0xdd,0x43,last_offset,0xd8,0xc9); // fld qword [ebx+o]; fmul st1
 const all=limit_cpus(body,pairs*2,addresses);
 for(const c of all){
  seed_long(c,addresses);seed_full(c,1e308);
  c.physical.setFloat64(0x40000+last_offset,1e308,true);
 }
 compare_conversion(all,131072,"typed long last-op bail "+label);
}
// A page fault inside a long run: the second half addresses a register that
// is repointed at unmapped memory after compilation.
{
 const body=[];
 for(let i=0;i<12;i++){const o=(i%12)*4;body.push(0xd9,0x43,o,0xd9,0x5b,o);}
 for(let i=0;i<12;i++){const o=(i%12)*4;body.push(0xd9,0x47,o,0xd9,0x5f,o);}
 body.push(0xff,0xe1);
 const all=limit_cpus(body,64,24);
 for(const c of all){
  seed_long(c,12);c.set_register(7,0x400000+0x60000);
  for(let i=0;i<48;i+=4)c.physical.setFloat32(0x60000+i,1.5,true);
  c.execute(131072);
  seed_long(c,12);c.set_register(7,0x70000000);
 }
 assert.equal(compare_conversion(all,100,"typed long interior fault",true).reason,0x10e);
}
console.log("x87 plan limits: 64-op/24-address and 96-op/32-address runs, last-op bail, budgets and interior fault passed");

// A plan that does not fit the local budget re-plans once at 32 ops / 16
// addresses rather than losing the run. The fallback cannot be forced
// through this surface — the largest plan (40 addresses, 8 slots) still
// needs under ~100 of the 128 local slots — so this case pins down the
// observable contract: a run wider than the fallback limits still executes
// a fast plan with exact parity. The re-plan branch itself was exercised
// with a diagnostic build during development.
{
 const body=long_body(24,20); // 48 ops, 20 distinct addresses: >16/<32-op fallback
 const all=typed_cpus(body);
 for(const c of all)seed_long(c,20);
 for(const budget of [131072,1,3,33,4097])
  compare_conversion(all,budget,"typed re-plan parity budget "+budget);
 assert(all[0].wasm.vine_x87_fast_instructions()>0,"re-plan candidate still executes a fast run");
}

// The rotated tag mask must be identical for every initial TOP: seed two
// adjacent full slots and run pops, pushes and exchanges on top of them.
function seed_two(cpu,a,b,top){
 seed(cpu);
 const view=new DataView(cpu.wasm.memory.buffer);
 view.setUint8(1032,top);view.setUint8(816,0xff&~(1<<top)&~(1<<((top+1)&7)));
 slot80(view,top,a);slot80(view,(top+1)&7,b);
}
let tag_cases=0;
for(const [name,body] of [
 ["pop+exchange",[0xdd,0xd8,0xd9,0xc9,0xff,0xe1]],              // fstp st0; fxch st1
 ["push+pop",[0xd9,0xc1,0xde,0xc1,0xff,0xe1]],                  // fld st1; faddp st1
 ["push+exchange+pop",[0xdd,0x47,8,0xd9,0xc9,0xde,0xc9,0xff,0xe1]], // fld [edi+8]; fxch st1; fmulp
]){
 for(let top=0;top<8;top++){
  const all=typed_cpus(body);
  for(const c of all){seed_two(c,1.25,2.5,top);c.physical.setFloat64(DATA+8,3.5,true);}
  compare_conversion(all,120000,"typed tag parity "+name+" TOP "+top);
  tag_cases++;
 }
}
console.log(`x87 tag masks: ${tag_cases} pop/push/exchange parity cases over TOP 0..7 passed`);
