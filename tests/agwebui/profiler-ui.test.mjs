import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadModule} from './module-loader.mjs';
const {renderProfiler,handleProfilerTableEvent,resetProfilerTables}=await loadModule('profiler-ui.js');
const call=(i,kind='tool')=>({id:String(i),name:'bash',agent:'a',kind,start:i,duration:i===305?400:1,outcome:'success',model:'x',tokens:100,output_tokens:10});
const run={id:'fixture',status:'completed',duration:2000,actions:Array.from({length:1000},(_,i)=>call(i)),agents:[{id:'a',label:'A',status:'completed'}],counters:{'workload:cpu_pct':[[0,10],[100,80]],'workload:memory_mb':[[0,20],[100,40]],'workload:io_r':[[0,0],[100,0]],'proc:12-100:rss_mb':[[0,10],[100,20]]},coverage:{},intervals:[]};
test('each tab preserves evidence access and bounds large detail tables',()=>{
 for(const category of ['Overview','LLM','Tools','Systems']){
  resetProfilerTables();const html=renderProfiler({...run,actions:category==='LLM'?run.actions.map(a=>({...a,kind:'model'})):run.actions},category);
  assert.match(html,new RegExp(category+' profiler'));assert.ok((html.match(/<tr>/g)||[]).length<230);
 }
});
test('primary latency chart bounds marks and preserves slow request inspector identity',()=>{
 const html=renderProfiler({...run,actions:run.actions.map(a=>({...a,kind:'model'}))},'LLM');
 assert.match(html,/Request latency over time/);assert.match(html,/data-prof-operation="305"/);
 const charts=html.match(/<svg[^>]*>[\s\S]*?<\/svg>/g);for(const chart of charts)assert.ok((chart.match(/<circle/g)||[]).length<=480);
 assert.ok(html.indexOf('Request latency over time')<html.indexOf('Secondary analytics'));
});
test('zero charts hide by default but recorded zeros and virtual memory remain inspectable',()=>{
 const r={...run,counters:{...run.counters,'proc:12-100:vms_mb':[[0,500]]}};
 const html=renderProfiler(r,'Systems');assert.doesNotMatch(html,/Disk utilization over time/);assert.match(html,/workload:io_r/);assert.match(html,/vms_mb/);
 assert.match(renderProfiler(r,'Systems',{showZero:true}),/Disk utilization over time/);
 assert.match(renderProfiler(r,'Systems',{process:'12-100'}),/Virtual address space; not physical memory/);
});
test('tools have ranked duration bars, sparse estimates and unresolved call sections',()=>{
 const html=renderProfiler({...run,actions:[call(0),{...call(1),outcome:'running'}]},'Tools');
 assert.match(html,/summed completed duration/);assert.match(html,/Sparse tail estimate/);assert.match(html,/Missing completion/);assert.match(html,/Unknown \/ incomplete/);
});
test('table paging, search and sort retain bounded state and update actual rows',()=>{
 resetProfilerTables();let renders=0;
 const send=dataset=>handleProfilerTableEvent({target:{closest:()=>({dataset,value:'305'})}},()=>renders++);
 assert.equal(send({profPage:'tools',delta:'1'}),true);assert.match(renderProfiler(run,'Tools'),/data-prof-operation="949"/);
 send({profSearch:'tools'});const html=renderProfiler(run,'Tools');assert.match(html,/data-prof-operation="305"/);assert.doesNotMatch(html,/data-prof-operation="999">bash/);assert.equal(renders,2);
 resetProfilerTables();
});
test('all 420 identical tool executions remain separate across pages, excluding syscalls',()=>{
 resetProfilerTables();
 const actions=Array.from({length:420},(_,i)=>({...call(i),id:`true-${i}`,name:'Bash',arguments:{command:'true'}}));
 const r={...run,actions:[...actions,{...call(421,'syscall'),id:'kernel-exec',name:'execve'},{...call(422,'syscall'),id:'kernel-send',name:'sendto'}]};
 const seen=[];
 for(let page=0;page<9;page++){
  const html=renderProfiler(r,'Tools');
  const body=html.slice(html.indexOf('aria-label="Search tools"')).match(/<tbody>([\s\S]*?)<\/tbody>/)[1];
  const ids=[...body.matchAll(/data-prof-operation="(true-\d+)">Bash/g)].map(m=>m[1]);
  assert.equal(ids.length,page===8?20:50);
  assert.equal((body.match(/&quot;command&quot;: &quot;true&quot;/g)||[]).length,ids.length);
  assert.doesNotMatch(body,/kernel-exec|kernel-send|>x ↗/);
  seen.push(...ids);
  handleProfilerTableEvent({target:{closest:()=>({dataset:{profPage:'tools',delta:'1'}})}},()=>{});
 }
 assert.equal(new Set(seen).size,420);
 assert.deepEqual(new Set(seen),new Set(actions.map(a=>a.id)));
 resetProfilerTables();
});
test('tool input is searchable beyond its preview and rendered without HTML execution',()=>{
 resetProfilerTables();
 const command='printf '+ 'x'.repeat(500)+' <script>needle</script>';
 const r={...run,actions:[{...call(1),arguments:JSON.stringify({cmd:command,workdir:'/workspace/proxy-case'})},{...call(2),arguments:{command:'true'}}]};
 const search=value=>handleProfilerTableEvent({target:{closest:()=>({dataset:{profSearch:'tools'},value})}},()=>{});
 for(const query of ['needle','/workspace/proxy-case']){
  search(query);
  const html=renderProfiler(r,'Tools');
  const body=html.slice(html.indexOf('aria-label="Search tools"')).match(/<tbody>([\s\S]*?)<\/tbody>/)[1];
  assert.match(body,/data-prof-operation="1"/);
  assert.doesNotMatch(body,/data-prof-operation="2"|<script>/);
  assert.ok(body.includes(command.replace('<script>','&lt;script&gt;').replace('</script>','&lt;/script&gt;')));
 }
 resetProfilerTables();
});
test('agent resource ownership is not inferred from PID or process command',()=>{
 const html=renderProfiler(run,'Systems',{agent:'a'});assert.match(html,/No reliably owned resource series/);assert.doesNotMatch(html,/data-prof-process="12-100"/);
});
test('empty resource windows remain unavailable rather than measured zero',()=>{
 const html=renderProfiler(run,'Overview',{range:[1500,1600]});
 assert.match(html,/<small>Peak workload memory<\/small><strong>Unavailable/);
 assert.match(renderProfiler(run,'Systems',{range:[1500,1600]}),/<small>Peak memory<\/small><strong>Unavailable/);
});
test('unreliable timestamps stay out of token growth charts but usage remains inspectable',()=>{
 const html=renderProfiler({...run,actions:[{...call(0,'model'),source:'timestamp_missing'}]},'LLM');
 assert.doesNotMatch(html,/data-prof-chart/);assert.match(html,/Start missing/);assert.match(html,/100/);
});
