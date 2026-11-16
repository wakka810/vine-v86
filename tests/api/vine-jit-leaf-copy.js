import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {create_cpu} from './vine-jit-budget.js';

const referenceModule=process.env.V86_BASELINE_WASM ? new WebAssembly.Module(readFileSync(process.env.V86_BASELINE_WASM)) : undefined;
const V=0x400000,A=0x1000,B=0x3000,HALT=0x1e00,DATA=0x8000;
const u32=n=>[n&255,n>>>8&255,n>>>16&255,n>>>24];
function fixture(leaf=[0x40,0xc3]) {
 const code=new Uint8Array(0x2040).fill(0x90);
 code.set([0xe8,...u32(B-A-5),0x43,0xeb,0xf8]);
 code[HALT-A]=0xf4;
 code.set(leaf,B-A);code.set([0x83,0xc0,7,0xc3],B-A+0x20);
 return code;
}
function make(options,leaf,code=fixture(leaf)) {
 const records=[[],[]];
 const cpus=[undefined,referenceModule].map((cpu_module,i)=>create_cpu(code,{
  ...options,paging:true,...(cpu_module?{cpu_module}:{}),
  on_finalize(_wasm,args){records[i].push({index:args[0],start:args[1],flags:args[2]});},
 }));
 // A one-page build provides the no-copy comparison when no frozen CPU is supplied.
 if(!referenceModule)cpus[1].wasm.set_jit_config(1,1);
 function cached(i,address) {
  return [...new Set(records[i].filter(({index,flags})=>
   cpus[i].wasm.jit_find_cache_entry_in_page(V+address,index,flags)>=0).map(r=>r.index))];
 }
 function compare(budget,label) {
  const results=cpus.map(c=>c.execute(budget));
  assert.deepEqual(results[0],results[1],label+' returned CPU');
  const snapshots=cpus.map(c=>c.full_snapshot());
  const normalized=s=>{const{jit,interpreted,...r}=s;if(results[0].reason===1)r.state=r.state.filter((_,i)=>i!==11);return r;};
  assert.deepEqual(normalized(snapshots[0]),normalized(snapshots[1]),label+' state/memory/dirty');
  assert.equal(snapshots[0].jit+snapshots[0].interpreted,snapshots[1].jit+snapshots[1].interpreted);
  return results[0];
 }
 async function resume(address,budget=256) {
  for(const c of cpus){c.physical.setUint32(0x90000,V+HALT,true);c.resume(0,address);}
  const result=compare(budget,'resume '+address.toString(16));await Promise.resolve();return result;
 }
 return{cpus,cached,compare,resume};
}

for(const asynchronous of [false,true])for(const chaining of [false,true]) {
 const pair=make({asynchronous,chaining});const{cpus,cached,compare,resume}=pair;
 await resume(B);await resume(B+0x20);
 const target=cached(0,B);assert.equal(target.length,1);assert.deepEqual(cached(0,B+0x20),target);
 await resume(A,1);assert.deepEqual(cached(0,B),target,'caller compilation preserves callee primary');
 assert.deepEqual(cached(0,B+0x20),target,'unrelated warmed entry retains identity');
 await resume(A,100000);
 for(const c of cpus)c.resume(0,A);
 const before=cpus.map(c=>({...c.counts}));compare(100000,'warm copied leaf');
 const entries=cpus.map((c,i)=>c.counts.entries-before[i].entries);
 const transfers=cpus.map((c,i)=>c.counts.chain_entries-before[i].chain_entries);
 if(chaining){assert.equal(transfers[0],0,'copied callee stays in the caller Wasm');assert(transfers[1]>20000);}
 else{assert(entries[0]<20);assert(entries[1]>20000);}
 for(const budget of [0,1,2,3,4,5,6,7,4095,4096,4097])await resume(A,budget);
 for(const c of cpus)c.set_counter(0xfffffff0);
 await resume(A,4097);
 for(const c of cpus)c.wasm.jit_dirty_cache(A,A+4096);
 assert.deepEqual(cached(0,B),target,'retiring caller preserves callee');
 await resume(B+0x20);await resume(A,100000);
 const modules=cpus.map(c=>c.counts.modules);
 for(let i=0;i<4;i++){await resume(B);await resume(B+0x20);await resume(A,4096);}
 assert.deepEqual(cpus.map(c=>c.counts.modules),modules,'alternation does not rebuild entries');
 for(const c of cpus){c.wasm.jit_dirty_cache(B,B+4096);c.physical.setUint8(B,0x48);}
 assert.equal(cached(0,A).length,0,'callee write retires copied caller');
 await resume(B);await resume(A,100000);
 console.log(JSON.stringify({asynchronous,chaining,entries,transfers,passed:'identity, bounded copy, exact budgets, counter wrap, caller/callee retirement, warm reuse, all CPU/memory'}));
}

for(const asynchronous of [false,true]) {
 const pair=make({asynchronous,chaining:true},[0x8b,0x07,0xc3]);
 await pair.resume(B);await pair.resume(A,100000);
 for(const c of pair.cpus){
  c.physical.setUint32(0x11000+(DATA>>>12)*4,0,true);
  c.wasm.full_clear_tlb();c.resume(0,A);
 }
 const stopped=pair.compare(4096,'copied leaf read fault');
 assert.equal(stopped.reason,0x10e);assert.equal(stopped.eip,V+B);
 console.log('Copied leaf read fault preserves all 153 state words, error and dirty bytes (async='+asynchronous+')');
}

for(const asynchronous of [false,true]) {
 const pair=make({asynchronous,chaining:true},[0x8b,0x07,0x89,0x04,0x24,0xc3]);
 for(const c of pair.cpus)c.physical.setUint32(DATA,V+A+5,true);
 await pair.resume(A,100000);
 await pair.resume(A,100000);
 // A callee may rewrite its actual stack return after the normal continuation
 // was compiled. The specialized edge must not override that live address.
 for(const c of pair.cpus)c.physical.setUint32(DATA,V+HALT,true);
 const result=await pair.resume(A,4096);
 assert.equal(result.reason,0x10d,'the CPL3 target reaches its HLT protection fault');
 assert.equal(result.eip,V+HALT,'unexpected return follows the actual stack');
 console.log('Copied leaf rewritten RET: actual target and full state match (async='+asynchronous+')');
}

{
 const pair=make({asynchronous:true,chaining:true});
 await pair.resume(B);await pair.resume(B+0x20);
 const queued=[],original=globalThis.queueMicrotask;
 try {
  globalThis.queueMicrotask=fn=>queued.push(fn);
  for(const c of pair.cpus)c.resume(0,A);
  pair.compare(1,'pending copied leaf');assert(queued.length>=2,'both finalizations are held');
  for(const c of pair.cpus){c.wasm.jit_dirty_cache(B,B+4096);c.physical.setUint8(B,0x48);}
  while(queued.length)queued.shift()();
 } finally {globalThis.queueMicrotask=original;}
 assert.equal(pair.cached(0,A).length,0,'written dependency cancels pending caller');
 await pair.resume(B);await pair.resume(A,100000);
 console.log('Pending copied callee write: stale finalization discarded, changed code executed');
}

{
 const C=0x5000,code=new Uint8Array(0x4100).fill(0x90);
 code.set([0xe8,...u32(B+0x20-A-5),0x43,0xeb,0xf8]);code[HALT-A]=0xf4;
 code.set([0xe8,...u32(C-B-5),0xc3],B-A);
 code.set([0x40,0xc3],B-A+0x20);code.set([0x40,0xc3],C-A);
 const pair=make({asynchronous:true,chaining:true},undefined,code);
 await pair.resume(B);await pair.resume(B+0x20);
 assert(pair.cached(0,B+0x20).length>0,'callee primary exists before pending copy');
 const queued=[],original=globalThis.queueMicrotask;
 try {
  globalThis.queueMicrotask=fn=>queued.push(fn);
  for(const c of pair.cpus)c.resume(0,A);
  pair.compare(1,'pending copy before other dependency retires');assert(queued.length>=2);
  for(const c of pair.cpus)c.wasm.jit_dirty_cache(C,C+4096);
  assert.equal(pair.cached(0,B+0x20).length,0,'other page actually retires the old callee primary');
  while(queued.length)queued.shift()();
 } finally {globalThis.queueMicrotask=original;}
 assert(pair.cached(0,A).length>0,'unchanged copied bytes allow caller publication');
 assert.equal(pair.cached(0,B+0x20).length,0,'no copied entry is published as callee primary');
 await pair.resume(A,100000);await pair.resume(B+0x20);
 assert(pair.cached(0,A).length>0,'warming the callee preserves the copied caller');
 for(const c of pair.cpus){c.wasm.jit_dirty_cache(B,B+4096);c.physical.setUint8(B+0x20,0x48);}
 assert.equal(pair.cached(0,A).length,0,'new callee primary still retains caller byte dependency');
 await pair.resume(B+0x20);await pair.resume(A,100000);
 console.log('Pending primary loss via another page: caller publishes without ghost callee entries, later writes retire it');
}
