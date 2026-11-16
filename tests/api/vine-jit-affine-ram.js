import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {create_cpu} from './vine-jit-budget.js';
const reference=process.env.V86_BASELINE_WASM &&
 new WebAssembly.Module(readFileSync(process.env.V86_BASELINE_WASM));
const V=0x400000,A=0x1000,DATA=0x8000;
const u32=n=>[n&255,n>>>8&255,n>>>16&255,n>>>24];
function loop(bytes){return Uint8Array.from([...bytes,0xe9,...u32(-bytes.length-5)]);}
const fieldRun=loop([0x8b,0x47,0x7c,0x89,0x47,0x78,0x8b,0x57,0x78,0x40,
 0x89,0x47,0x7c,0x8b,0x4f,0x2c]);
function make(code=fieldRun,options={}) {
 const traces=[[],[]];
 const cpus=[undefined,reference].map((cpu_module,i)=>create_cpu(code,{paging:true,...options,
  ...(cpu_module?{cpu_module}:i===1?{disable_jit:true}:{}),on_mmap(name,args){
   traces[i].push([name,...args]);
   return name.startsWith('mmap_read') ? (0x12345678^args[0]^traces[i].length)|0 : 0;
  }}));
 function compare(budget,label){
  const results=cpus.map(c=>c.execute(budget));
  if(reference || results[0].reason===1)assert.deepEqual(results[0],results[1],label+': returned state');
  else {
   // JIT fault exits retain the existing whole-block retirement accounting.
   const normalized=results.map(({instructions,...r})=>r);
   assert.deepEqual(normalized[0],normalized[1],label+': interpreter state');
  }
  const snapshots=cpus.map(c=>{
   const {jit,interpreted,...r}=c.full_snapshot();
   if(results[0].reason===1)r.state=r.state.filter((_,i)=>i!==11);
   return {...r,retired:jit+interpreted};
  });
  if(reference)assert.deepEqual(snapshots[0],snapshots[1],label+': full CPU, memory, dirty and counters');
  else {
   assert.deepEqual(snapshots[0].memory,snapshots[1].memory,label+': interpreter memory');
   assert.deepEqual(snapshots[0].dirty,snapshots[1].dirty,label+': interpreter dirty pages');
  }
  assert.deepEqual(traces[0],traces[1],label+': MMIO order/values');
  return results[0];
 }
 async function warm(){compare(131072,'warm');await Promise.resolve();compare(131072,'warm compiled');await Promise.resolve();assert(cpus[0].counts.entries>0);}
 function point(offset){for(const c of cpus){c.resume(0,A);c.set_register(7,V+offset);}}
 function pte(address,value){for(const c of cpus){c.physical.setUint32(0x11000+(address>>>12)*4,value,true);c.wasm.full_clear_tlb();}}
 return{cpus,compare,warm,point,pte,traces};
}
for(const asynchronous of [false,true])for(const chaining of [false,true]){
 const p=make(fieldRun,{asynchronous,chaining});await p.warm();
 for(const budget of [0,1,2,3,4,5,6,7,8,4095,4096,4097,131072])p.compare(budget,'field run '+budget);
 for(const c of p.cpus)c.set_counter(0xfffffff0);
 p.compare(4097,'counter wrap');
 for(const c of p.cpus)c.clear_dirty([DATA]);
 p.compare(256,'first write after dirty rearm');
 assert(p.cpus[0].full_snapshot().dirty.includes(V+DATA),'writes still report the dirty virtual page');
 p.point(DATA+0xf80);p.compare(256,'span crosses a page although individual accesses fit');
 p.point(0xa0000);p.compare(48,'MMIO remains ordered');assert(p.traces[0].length>0);
 p.point(DATA);p.pte(DATA,DATA|5);const fault=p.compare(64,'read before readonly write');
 assert.equal(fault.reason,0x10e);assert.equal(fault.eip,V+A+3);
 console.log(JSON.stringify({asynchronous,chaining,passed:'budgets/wrap/dirty/page span/MMIO/readonly'}));
}
{
 const p=make(loop([0x89,0x07,0x8b,0x57,4]));await p.warm();
 p.point(DATA+0xffc);p.pte(DATA+4096,0);
 for(const c of p.cpus)c.set_register(0,0x76543210);
 const fault=p.compare(32,'earlier store commits before later read fault');
 assert.equal(fault.reason,0x10e);assert.equal(fault.eip,V+A+2);
 assert.equal(p.cpus[0].physical.getUint32(DATA+0xffc,true),0x76543210);
}
{
 const p=make(loop([0x8b,0x07,0x83,0xc7,4,0x89,0xfe,0x8b,0x16,
  0x83,0xef,4,0x89,0x47,8]));
 await p.warm();p.compare(131072,'affine base changes and register copies');
}
{
 // The second load depends on a value read from memory, so it must not use
 // the entry value of EAX to precheck or address that load.
 const p=make(loop([0x8b,0x07,0x8b,0x10,0x89,0x57,8]));
 for(const c of p.cpus){c.physical.setUint32(DATA,V+DATA+4,true);c.physical.setUint32(DATA+4,0x11223344,true);}
 await p.warm();p.compare(131072,'loaded pointer is not an entry register');
 assert.equal(p.cpus[0].physical.getUint32(DATA+8,true),0x11223344);
}
{
 // Stop at HLT after a fixed number of iterations: the existing SMC bailout
 // charges a whole JIT block, so a budget exit need not match the interpreter PC.
 const code=Uint8Array.from([0x8b,0x07,0x89,0x47,4,0x8b,0x57,4,0x4b,0x75,0xf5,0xf4]);
 const p=make(code);for(const c of p.cpus)c.set_register(3,1000000);await p.warm();
 p.point(A+0x100);
 for(const c of p.cpus){c.set_register(3,1);c.physical.setUint32(A+0x100,0x11223344,true);}
 p.compare(64,'write to compiled source page follows code retirement');
 assert(p.cpus[0].counts.invalidations>0);
}
{
 const p=make(loop([0x8b,0x07,0x8b,0x57,4,0x66,0x8b,0x4f,8]));
 await p.warm();p.compare(131072,'operand size prefix ends the run');
}
{
 const p=make(loop([0x8b,0x47,0x20,0x8b,0x57,0x24,0x89,0x47,0x28]));
 await p.warm();
 // An entry register near 2^32 wraps to a valid low virtual page. The span
 // proof uses the wrapped address, while each physical access stays bounded.
 for(const c of p.cpus){
  c.physical.setUint32(0x10000,0x12007,true);
  c.physical.setUint32(0x12000,DATA|7,true);
  c.wasm.full_clear_tlb();
 }
 p.point(0xffbffff0);p.compare(256,'wrapping affine address');
}
console.log('Affine RAM run checks passed against '+(reference?'the frozen CPU':'the interpreter'));
