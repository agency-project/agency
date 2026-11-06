import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const base=new URL('../../agency/observability/agwebui/static/',import.meta.url);
const url=s=>`data:text/javascript;base64,${Buffer.from(s).toString('base64')}`;
const model=url(await readFile(new URL('investigator-model.js',base),'utf8'));
const lanesSource=(await readFile(new URL('resource-lanes.js',base),'utf8')).replace('./investigator-model.js',model);
const source=(await readFile(new URL('causal-graph.js',base),'utf8')).replace('./execution-timeline.js',url(await readFile(new URL('execution-timeline.js',base),'utf8'))).replace('./investigator-model.js',model).replace('./resource-lanes.js',url(lanesSource));
const {operationGraph,agentGraphs,graphSlice,graphLayout,displayGraph,visibleGraph,causalSummary,systemActivity,semanticLevel,zoomAt,panFrom,fitView,routeEdge,laneGeometry,laneAnchor,LANE,HOST,CausalGraph}=await import(url(source));
const {resourceModel,sampleAt,chartPaths,latestSample,formatValue,chartScale,AGENT_METRICS}=await import(url(lanesSource));
function graph(count=4) {
  const nodes=[],edges=[],operations=[],path=[];
  for(let i=0;i<count;i++) {
    const id=`op${i}`,start_node=`${id}:start`,end_node=`${id}:end`;
    nodes.push({id:start_node,operation:id,time:i*2,phase:'start',timing:'observed'},{id:end_node,operation:id,time:i*2+1,phase:'end',timing:'observed'});
    operations.push({id,action_id:id,label:id,agent:'a',kind:i%2?'tool':'model',band:i%2?'Tool':'LLM',start_node,end_node,duration:1,outcome:'success',metrics:{}});
    edges.push({id:`duration:${id}`,source:start_node,target:end_node,kind:'duration',confidence:'verified',blocking:true,weight:1});
    if(i)edges.push({id:`dep${i}`,source:`op${i-1}:end`,target:start_node,kind:'tool_result',confidence:'verified',blocking:true,gap:0,evidence:'captured ID'});
    path.push(start_node,end_node);
  }
  return {nodes,edges,operations,critical_path:{node_ids:path,edge_ids:edges.map(e=>e.id),slack:{}},coverage:{unknown:[],gaps:[]}};
}
function add(g,id,kind='tool',agent='a') {
  g.operations.push({id,action_id:id,label:id,agent,kind,band:kind==='tool'?'Tool':'System',start_node:`${id}:start`,end_node:`${id}:end`,duration:1,metrics:{pid:12}});
  g.nodes.push({id:`${id}:start`,operation:id,time:1,timing:'observed'},{id:`${id}:end`,operation:id,time:2,timing:'observed'});
}
function link(g,source,target,extra={}) {g.edges.push({id:`${source}->${target}`,source:`${source}:end`,target:`${target}:start`,kind:'tool_result',confidence:'verified',blocking:true,gap:0,...extra});}
const nav=()=>({scale:40,defaultScale:40});
test('wall-clock anchors derive from timestamps, independent of depth, duration and type',()=>{
 const g=graph(),p=operationGraph(g),l=graphLayout(graphSlice(p),nav());
 for(const n of p.nodes)assert.equal(l.positions.get(n.id).x,48+n.start*40);
 assert.equal(l.positions.get('op2').x-l.positions.get('op1').x,80);
 assert.equal(p.nodes[0].duration,1);assert.equal(p.nodes[0].kind,'llm');assert.equal(p.nodes[1].kind,'tool');
});
test('parallel and tightly adjacent operation cards never overlap at medium zoom',()=>{
 const g=graph(100);for(const n of g.nodes)n.time=n.phase==='end'?10:0;
 const p=operationGraph(g),l=graphLayout(graphSlice(p),nav());
 assert.equal(new Set([...l.positions.values()].map(p=>p.y)).size,100);
 const dense=graphLayout(graphSlice(operationGraph(graph(100))),{scale:24,defaultScale:40});
 const rows=new Map();for(const p of dense.positions.values()){if(!rows.has(p.y))rows.set(p.y,[]);rows.get(p.y).push(p.x);}
 for(const xs of rows.values()){xs.sort((a,b)=>a-b);for(let i=1;i<xs.length;i++)assert.ok(xs[i]-xs[i-1]>=216);}
});
test('syscall and scheduler filtering leaves exactly three primary categories and creates no edges',()=>{
 const g=graph(2);g.edges=g.edges.filter(e=>e.kind==='duration');add(g,'read','syscall');add(g,'queue','queue');link(g,'op0','read');link(g,'read','op1');
 const p=operationGraph(g);assert.equal(p.omitted,1);assert.equal(p.auxiliary,1);assert.equal(p.edges.length,0);assert.deepEqual(p.nodes.map(n=>n.kind),['llm','tool']);
});
test('unknown elapsed gaps retain the original edge evidence without creating a fourth operation category',()=>{
 const g=graph(2);g.edges.find(e=>e.id==='dep1').gap=1;const p=operationGraph(g);
 assert.equal(p.nodes.length,2);assert.equal(p.edges.length,1);assert.equal(p.edges[0].gap,1);assert.equal(p.edges[0].evidence_id,'dep1');
});
test('ancestry expansion is explicit, reversible, and never a blocking critical dependency',()=>{
 const g=graph(2);add(g,'process','process');link(g,'op1','process',{kind:'process_owner',confidence:'provenance',blocking:false});
 const p=operationGraph(g);assert.equal(graphSlice(p).nodes.length,2);
 const open=graphSlice(p,{processes:new Set(['op1'])});assert.equal(open.nodes.length,3);assert.equal(open.edges.at(-1).blocking,false);assert.equal(open.edges.at(-1).critical,false);
 assert.equal(graphSlice(p,{processes:new Set()}).nodes.length,2);
 g.edges.at(-1).confidence='inferred';assert.equal(graphSlice(operationGraph(g),{processes:new Set(['op1'])}).nodes.length,2);
});
test('unknown process ownership stays separate from all tools',()=>{
 const g=graph(2);add(g,'unknown-process','process');const p=operationGraph(g);
 assert.equal(graphSlice(p).nodes.length,2);const s=graphSlice(p,{mode:'processes'});assert.equal(s.nodes.length,1);assert.equal(s.edges.length,0);
});
test('critical-only disclosure shows real parallel dependencies and keeps inference opt-in',()=>{
 const g=graph(3);add(g,'parallel');link(g,'op0','parallel');link(g,'parallel','op2');
 link(g,'op0','op2',{id:'retry',confidence:'inferred',blocking:false});
 const p=operationGraph(g);assert.equal(graphSlice(p,{mode:'critical'}).nodes.length,3);
 assert.equal(graphSlice(p,{mode:'critical',branches:new Set(['op0'])}).nodes.length,4);
 assert.ok(!graphSlice(p).edges.some(e=>e.id==='retry'));assert.ok(graphSlice(p,{inferred:true}).edges.some(e=>e.id==='retry'));
});
test('cursor-centered zoom preserves the timestamp under the pointer across semantic levels',()=>{
 const n={scale:40,defaultScale:40,left:1200,top:100},cursor={x:360,y:200};
 const at=(n.left+cursor.x-48)/n.scale;
 for(const factor of [.4,.8,1.25,3]){const next=zoomAt(n,factor,cursor,1000,480,100);assert.ok(Math.abs((next.left+cursor.x-48)/next.scale-at)<1e-8);}
});
test('fit includes the timestamp extent and card clearance without altering recorded positions',()=>{
 const camera=fitView({start:40,end:100,minLane:0,maxLane:3},1000,480,40);
 assert.ok(48+100*camera.scale+190-camera.left<=1000);assert.ok(48+40*camera.scale-camera.left>=0);
 assert.ok(camera.scale>0);assert.equal(camera.top,0);
});
test('horizontal/vertical drag pan changes only the camera',()=>{
 assert.deepEqual(panFrom({x:300,y:200,left:800,top:500},{x:250,y:150}),{left:850,top:550});
 assert.deepEqual(panFrom({x:0,y:0,left:0,top:0},{x:200,y:200}),{left:0,top:0});
});
test('semantic zoom aggregates colliding labels while retaining original operation/edge identities',()=>{
 const p=operationGraph(graph(100)),s=graphSlice(p),n={scale:1,defaultScale:40},l=graphLayout(s,n),d=displayGraph(s,l);
 assert.equal(l.level,'overview');assert.ok(d.nodes.length<s.nodes.length);assert.ok(d.groups.size>0);
 const allMembers=d.nodes.flatMap(n=>n.members||[n.id]);assert.equal(new Set(allMembers).size,s.nodes.length);
 assert.ok(d.edges.every(e=>p.edges.some(raw=>raw.id===e.id)));assert.equal(semanticLevel(100,40),'detail');
});
test('edge routes connect actual boundary timestamps and run below unrelated cards',()=>{
 const p=operationGraph(graph(3)),s=graphSlice(p),l=graphLayout(s,nav());
 for(const e of s.edges){const route=routeEdge(e,l.positions,40);assert.equal(route.sx,48+e.sourceTime*40);assert.equal(route.tx,48+e.targetTime*40);assert.ok(route.tx>=route.sx);assert.ok(route.sy>l.positions.get(e.source).y+96);assert.match(route.path,/ V .* H /);}
});
test('stable live layout retains lanes and camera across appends and completion',()=>{
 const n={...nav(),left:500,top:80,selected:'op1'},a=graphLayout(graphSlice(operationGraph(graph(3))),n),b=graphLayout(graphSlice(operationGraph(graph(10))),n);
 for(const id of ['op0','op1','op2'])assert.deepEqual({...a.positions.get(id),node:null},{...b.positions.get(id),node:null});
 assert.equal(n.left,500);assert.equal(n.top,80);assert.equal(n.selected,'op1');
});
test('open execution uses the displayed clock with unknown measured duration; missing starts stay missing',()=>{
 const g=graph(1),op=g.operations[0];op.open_node=op.end_node;delete op.end_node;op.duration=null;op.outcome='running';
 const p=operationGraph(g,[],123);assert.equal(p.nodes[0].open,123);assert.equal(p.nodes[0].end,undefined);assert.equal(p.nodes[0].duration,null);
 delete op.start_node;assert.equal(operationGraph(g).nodes[0].start,undefined);
});
test('large traces are filtered, paged and viewport-culled before rendering',()=>{
 const g=graph(8);for(let i=8;i<4848;i++)add(g,`futex-${i}`,'syscall');const p=operationGraph(g),s=graphSlice(p),l=graphLayout(s,nav());
 assert.equal(p.rawCount,9696);assert.equal(p.omitted,4840);assert.equal(s.nodes.length,8);
 assert.ok(visibleGraph(s,l.positions,{left:0,right:1000,top:0,bottom:480}).nodes.length<=8);
 const big=operationGraph(graph(4848)),page=graphSlice(big);assert.equal(page.nodes.length,200);assert.equal(graphSlice(big,{offset:200}).nodes[0].id,'op200');
});
test('summary uses clipped interval unions, not sums of overlapping operations',()=>{
 const g=graph(2);g.summary={execution_duration:10};g.operations[0].duration=8;g.nodes[1].time=8;g.operations[1].duration=7;g.nodes[2].time=3;g.nodes[3].time=10;
 const html=causalSummary(g);assert.match(html,/LLM interval coverage/);assert.match(html,/8.00s/);assert.match(html,/7.00s/);assert.match(html,/Unattributed wall time<\/small><strong>0.000s/);assert.match(html,/must not be added/);assert.match(html,/Partial/);
});
test('system aggregation never attributes by PID/overlap or counts an incomplete syscall as runtime',()=>{
 const run={causal_graph:graph(2),actions:[{kind:'syscall',name:'write',duration:2,outcome:'success',metadata:{pid:12}},{kind:'syscall',name:'read',duration:99,outcome:'incomplete',metadata:{tool_action_id:'op0'}},{kind:'syscall',name:'read',duration:1,outcome:'success',metadata:{tool_action_id:'op0',process_action_id:'op1'}}]};
 assert.equal(systemActivity(run).unattributed,2);assert.equal(systemActivity(run,'op0').count,1);assert.equal(systemActivity(run,'op0').duration,0);
});
function mounted() {
 const elements=new Map();
 class Element {
  constructor(){this.handlers=new Map();this.classList={add(){},remove(){},toggle(){}};this.style={};this.dataset={};this.clientWidth=1000;this.clientHeight=480;this.scrollLeft=0;this.scrollTop=0;this.innerHTML='';this.firstElementChild=null;}
  querySelector(selector){if(!elements.has(selector))elements.set(selector,new Element());return elements.get(selector);}
  querySelectorAll(){return [];}
  insertBefore(){}setAttribute(){}setPointerCapture(){}
  getBoundingClientRect(){return {left:10,top:20};}
  addEventListener(name,fn){if(!this.handlers.has(name))this.handlers.set(name,[]);this.handlers.get(name).push(fn);}
  emit(name,e){for(const fn of this.handlers.get(name)||[])fn(e);}
 }
 globalThis.document={createElement:()=>new Element(),activeElement:null};globalThis.ResizeObserver=class {observe(){}disconnect(){}};globalThis.requestAnimationFrame=()=>1;globalThis.cancelAnimationFrame=()=>{};
 const host=new Element(),scroll=host.querySelector('.execution-timeline-scroll');scroll.firstElementChild=new Element();
 let history=0;const n={...nav(),left:500,top:30,lock:true},viewer=new CausalGraph(host,n,()=>{},()=>history++);viewer.update({causal_graph:graph(20),actions:[],duration:50});
 return {host,scroll,n,viewer,elements,history:()=>history};
}
test('real wheel handler consumes intentional zoom, preserves cursor time and disables live-follow',()=>{
 const {scroll,n}=mounted();const before=(scroll.scrollLeft+600-48)/n.scale;let prevented=false;
 scroll.emit('wheel',{ctrlKey:true,deltaY:50,clientX:610,clientY:220,preventDefault(){prevented=true;}});
 assert.ok(prevented);assert.equal(n.lock,false);assert.ok(Math.abs((scroll.scrollLeft+600-48)/n.scale-before)<1e-8);
});
test('pointer drag supports both axes and excludes node targets; touch pinch changes scale',()=>{
 const {scroll,n}=mounted(),empty={closest:()=>null};const before={left:scroll.scrollLeft,top:scroll.scrollTop};
 scroll.emit('pointerdown',{button:0,pointerId:1,clientX:300,clientY:200,target:empty});scroll.emit('pointermove',{pointerId:1,clientX:250,clientY:150});
 assert.equal(scroll.scrollLeft,before.left+50);assert.equal(scroll.scrollTop,before.top+50);assert.equal(n.lock,false);
 scroll.emit('pointerup',{pointerId:1});const oldLeft=scroll.scrollLeft;
 scroll.emit('pointerdown',{button:0,pointerId:1,clientX:300,clientY:200,target:{closest:()=>({})}});scroll.emit('pointermove',{pointerId:1,clientX:200,clientY:100});assert.equal(scroll.scrollLeft,oldLeft);
 for(const [id,x] of [[2,200],[3,400]])scroll.emit('pointerdown',{button:0,pointerId:id,pointerType:'touch',clientX:x,clientY:220,target:empty});
 scroll.emit('pointermove',{pointerId:3,clientX:410,clientY:220});const oldScale=n.scale;scroll.emit('pointermove',{pointerId:3,clientX:460,clientY:220});assert.ok(n.scale>oldScale);
});
test('highlighting can be disabled, reset/fit work, and Live alone follows appended events',()=>{
 const {viewer,host,n,elements,scroll}=mounted();assert.match(viewer.canvas.innerHTML,/causal-node llm critical/);
 const highlight=host.querySelector('[data-causal-highlight]');highlight.onchange({target:{checked:false}});assert.doesNotMatch(viewer.canvas.innerHTML,/causal-node llm critical/);
 viewer.manual();const left=scroll.scrollLeft;viewer.update({causal_graph:graph(30),actions:[],duration:80});assert.equal(scroll.scrollLeft,left);
 host.querySelector('[data-latest]').onclick();assert.equal(n.lock,true);viewer.update({causal_graph:graph(40),actions:[],duration:100});assert.ok(scroll.scrollLeft>left);
 viewer.fit();assert.equal(n.lock,false);assert.ok(n.scale<n.defaultScale);viewer.reset();assert.equal(n.scale,n.defaultScale);
});
if(process.env.AGENCY_CAUSAL_TRACE)test('actual large recording remains readable under zoom/pan without syscall DOM nodes',async()=>{
 const run=JSON.parse(await readFile(process.env.AGENCY_CAUSAL_TRACE,'utf8')),p=operationGraph(run.causal_graph,run.actions),s=graphSlice(p),n=nav();
 assert.ok(run.coverage.events>9000);assert.ok(p.omitted>1000);assert.ok(s.nodes.every(n=>['llm','tool','process'].includes(n.kind)));
 const initial=graphLayout(s,n);assert.ok(initial.positions.size<=320);
 for(const scale of [1,10,40,100]){n.scale=scale;const l=graphLayout(s,n),d=displayGraph(s,l),v=visibleGraph(d,l.positions,{left:48+50*scale,right:1048+50*scale,top:0,bottom:480,level:l.level});assert.ok(v.nodes.length<60);assert.ok(v.edges.length<200);}
 console.log(`${run.coverage.events} actual raw events; ${p.omitted} syscalls filtered; ${s.nodes.length} major operations; ${p.nodes.filter(n=>n.kind==='process').length} process observations; ${initial.bounds.maxLane+1} stable lanes`);
});

test('hour-long traces keep a bounded scroll surface at maximum zoom',()=>{
 const camera=zoomAt({scale:40,defaultScale:40,left:120000,top:0},1e6,{x:500,y:200},1000,480,36000);
 assert.ok(camera.scale*36000<=10000000);assert.ok(camera.scale>40);
});

// ---- Agent lanes + resource charts -------------------------------------
function multi(agents=['a','b'],count=4,{cross=false}={}) {
  const nodes=[],edges=[],operations=[],paths={};
  for(const [k,agent] of agents.entries()) {
    const path={node_ids:[],edge_ids:[],slack:{},duration:count*2-1,measured_work:count,unexplained_wait:count-1};
    for(let i=0;i<count;i++) {
      const id=`${agent}-op${i}`,start_node=`${id}:start`,end_node=`${id}:end`,t=k*.5+i*2;
      nodes.push({id:start_node,operation:id,agent,time:t,phase:'start',timing:'observed'},{id:end_node,operation:id,agent,time:t+1,phase:'end',timing:'observed'});
      operations.push({id,action_id:id,label:id,agent,kind:i%2?'tool':'model',band:i%2?'Tool':'LLM',start_node,end_node,duration:1,outcome:'success',metrics:{}});
      edges.push({id:`duration:${id}`,source:start_node,target:end_node,kind:'duration',confidence:'verified',blocking:true,weight:1});
      if(i)edges.push({id:`dep:${id}`,source:`${agent}-op${i-1}:end`,target:start_node,kind:'tool_result',confidence:'verified',blocking:true,gap:1,evidence:'captured ID'});
      path.node_ids.push(start_node,end_node);path.edge_ids.push(`duration:${id}`,...(i?[`dep:${id}`]:[]));
    }
    paths[agent]=path;
  }
  if(cross)edges.push({id:'cross',source:`${agents[0]}-op0:end`,target:`${agents[1]}-op1:start`,kind:'tool_result',confidence:'verified',blocking:true,gap:0});
  return {nodes,edges,operations,agent_critical_paths:paths,operation_critical_path:paths[agents[0]],system_critical_path:{available:false,reason:'not captured'},coverage:{unknown:[],gaps:[]}};
}
const ticks=(from,to,step,fn)=>{const out=[];for(let i=0;from+i*step<=to+1e-9;i++){const t=+(from+i*step).toFixed(6);out.push([t,fn(t)]);}return out;};
function laneRun(agents=['a','b'],count=4,extra={}) {
  const counters={};
  for(const id of agents){counters[`sandbox:sb-${id}:cpu_pct`]=ticks(0,10,.1,t=>t>=4&&t<=5?250:20);counters[`sandbox:sb-${id}:mem_mb`]=ticks(0,10,.1,()=>512);}
  counters['host:cpu_util_pct']=ticks(0,10,.1,()=>40);counters['host:mem_used_mb']=ticks(0,10,.1,()=>4096);counters['host:mem_total_mb']=[[0,16384]];
  for(const r of ['cpu','memory','io'])counters[`host:${r}_psi_some_pct`]=ticks(0,10,.1,t=>t>4?12:1);
  return {causal_graph:multi(agents,count,extra),actions:[],duration:10,agents:agents.map(id=>({id,label:id,sandbox:`sb-${id}`})),counters,clock:{events:'x',resources:'x',shared:true},...extra.run};
}
function lanes(run=laneRun(),{height=4000}={}) {
  const elements=new Map();
  class Element {
   constructor(){this.handlers=new Map();this.classList={add(){},remove(){},toggle(){}};this.style={};this.dataset={};this.clientWidth=1000;this.clientHeight=height;this.scrollLeft=0;this.scrollTop=0;this.innerHTML='';this.firstElementChild=null;this.hidden=false;}
   querySelector(selector){if(!elements.has(selector))elements.set(selector,new Element());return elements.get(selector);}
   querySelectorAll(){return [];}
   insertBefore(){}setAttribute(){}setPointerCapture(){}append(){}
   getBoundingClientRect(){return {left:10,top:20,width:1200};}
   addEventListener(name,fn){if(!this.handlers.has(name))this.handlers.set(name,[]);this.handlers.get(name).push(fn);}
   emit(name,e){for(const fn of this.handlers.get(name)||[])fn(e);}
  }
  globalThis.document={createElement:()=>new Element(),activeElement:null};globalThis.ResizeObserver=class {observe(){}disconnect(){}};globalThis.requestAnimationFrame=()=>1;globalThis.cancelAnimationFrame=()=>{};
  const host=new Element(),scroll=host.querySelector('.execution-timeline-scroll');scroll.firstElementChild=new Element();
  const n={scale:40,defaultScale:40,left:0,top:0,lock:false},viewer=new CausalGraph(host,n,()=>{},()=>{});viewer.update(run);
  return {host,scroll,n,viewer,run};
}
const laneMarkup=(html,id)=>html.split('data-lane-graph="').find(part=>part.startsWith(`${id}"`))||'';
test('Critical Path stays empty during a run and appears after completion',()=>{
  const pending={...laneRun(),causal_graph:{state:'pending',nodes:[],edges:[],operations:[]}};
  const {viewer,host}=lanes(pending);
  assert.equal(host.hidden,true);
  assert.equal(viewer.projection,undefined);
  assert.equal(viewer.resources,undefined);
  assert.match(causalSummary(pending.causal_graph,pending),/after the run completes/);
  viewer.paint(); // Resize/freeze callbacks must not project an ongoing graph.
  viewer.update(laneRun());
  assert.equal(host.hidden,false);
  assert.ok(viewer.projection.nodes.length);
  viewer.update(pending);
  assert.equal(host.hidden,true);
});
test('operations group into one lane per agent; a cross-agent edge is counted, never drawn',()=>{
  const p=operationGraph(multi(['a','b'],3,{cross:true})),{groups,crossing}=agentGraphs(p);
  assert.deepEqual([...groups.keys()],['a','b']);assert.equal(crossing,1);
  for(const [agent,g] of groups){assert.ok(g.nodes.every(n=>n.agent===agent));assert.ok(g.edges.every(e=>g.nodes.some(n=>n.id===e.source)&&g.nodes.some(n=>n.id===e.target)));}
  const {viewer}=lanes(laneRun(['a','b'],3,{cross:true}));assert.doesNotMatch(viewer.canvas.innerHTML,/data-causal-edge="cross"/);
});
test('causal edges render inside their own agent lane only',()=>{
  const {viewer}=lanes(),html=viewer.canvas.innerHTML;
  assert.match(laneMarkup(html,'a'),/data-causal-edge="dep:a-op1"/);assert.doesNotMatch(laneMarkup(html,'a'),/data-causal-node="b-/);
  assert.match(laneMarkup(html,'b'),/data-causal-node="b-op0"/);assert.doesNotMatch(laneMarkup(html,'b'),/dep:a-/);
});
test('agents collapse and expand independently; collapse-all includes the host section',()=>{
  const {viewer,host}=lanes();
  viewer.laneControl({dataset:{laneToggle:'a'}});let rows=viewer.geometry.byId;
  assert.equal(rows.get('a').graph,null);assert.equal(rows.get('a').charts.length,0);assert.ok(rows.get('b').graph);assert.equal(rows.get('b').charts.length,2);
  assert.doesNotMatch(viewer.canvas.innerHTML,/data-lane-graph="a"/);assert.match(viewer.overlay.innerHTML,/data-lane-toggle="a" aria-expanded="false"/);
  viewer.laneControl({dataset:{laneToggle:'a'}});assert.ok(viewer.geometry.byId.get('a').graph);
  host.querySelector('[data-lanes-collapse]').onclick();rows=viewer.geometry.byId;assert.ok([...rows.values()].every(r=>r.collapsed));
  host.querySelector('[data-lanes-expand]').onclick();assert.ok([...viewer.geometry.byId.values()].every(r=>!r.collapsed));
  const g=laneGeometry([{id:'x',graphRows:2,step:70,charts:['cpu']},{id:'y',collapsed:true,charts:['cpu']}],{charts:['cpu']});
  assert.ok(g.byId.get('y').bottom-g.byId.get('y').top===LANE.header);assert.ok(g.byId.get(HOST).top>g.byId.get('y').bottom);
});
test('every lane and chart shares one wall-clock coordinate: x = PAD + t * scale',()=>{
  const {viewer,n}=lanes(),html=viewer.canvas.innerHTML;
  for(const id of ['a-op1','b-op1']){const node=viewer.projection.lookup.get(id),m=html.match(new RegExp(`translate\\(([\\d.]+),[\\d.-]+\\)" class="causal-node[^"]*" data-causal-node="${id}"`));assert.ok(Math.abs(Number(m[1])-(48+node.start*n.scale))<1e-6);}
  const s={points:[[2.5,10],[3,10]],gap:1,tolerance:.1},paths=chartPaths(s,{scale:n.scale,pad:48,left:0,right:1000,top:0,height:30,max:10});
  assert.ok(paths.line.startsWith(`M${(48+2.5*n.scale).toFixed(1)} `));assert.equal(48+2.5*n.scale,48+viewer.projection.lookup.get('b-op1').start*n.scale);
});
test('zoom and pan move all lanes and charts together through one camera',()=>{
  const {viewer,scroll,n}=lanes();viewer.zoom(2,{x:500,y:100});
  const html=viewer.canvas.innerHTML;for(const id of ['a-op2','b-op2']){const node=viewer.projection.lookup.get(id);assert.match(html,new RegExp(`translate\\(${48+node.start*n.scale}[,)]`));}
  viewer.applyCamera({left:300,top:0});assert.equal(scroll.scrollLeft,300);assert.equal(viewer.overlay.style.transform,'translateX(300px)');
  assert.match(viewer.canvas.innerHTML,/<rect class="chart-bg" x="300"/);
});
test('cursor-centered zoom keeps both the timestamp and the hovered agent lane under the pointer',()=>{
  const {viewer,scroll,n}=lanes(laneRun(['a','b','c'],12),{height:480});const cursor={x:420,y:300};
  scroll.scrollTop=200;scroll.scrollLeft=900;viewer.paint();const contentY=scroll.scrollTop+cursor.y,anchor=laneAnchor(viewer.geometry,contentY),t=(scroll.scrollLeft+cursor.x-48)/n.scale;
  viewer.zoom(.7,cursor);const row=viewer.geometry.byId.get(anchor.id);
  assert.ok(Math.abs((scroll.scrollLeft+cursor.x-48)/n.scale-t)<1e-6);
  assert.ok(Math.abs((scroll.scrollTop+cursor.y-row.top)/(row.bottom-row.top)-anchor.fraction)<1e-6||scroll.scrollTop===0);
});
test('shared crosshair aligns to the time axis and reports each lane at that instant',()=>{
  const {viewer,scroll,n}=lanes();scroll.scrollLeft=0;viewer.paint();
  const t=4.5,clientX=10+48+t*n.scale,row=viewer.geometry.byId.get('a'),chart=row.charts[0];
  viewer.hover({clientX,clientY:20+chart.top+5});
  assert.equal(viewer.cursor.hidden,false);assert.equal(viewer.cursor.style.left,`${48+t*n.scale}px`);
  const tip=viewer.tip.innerHTML;assert.match(tip,/t = 4\.500s/);assert.match(tip,/<b>a<\/b> CPU <span title="sample at 4\.500s">250%/);assert.match(tip,/<b>b<\/b>/);assert.match(tip,/<b>Host<\/b> CPU <span[^>]*>40%/);
  assert.match(tip,/a · CPU: <b>250% \(2\.50 cores\)<\/b>/);
  viewer.hover(null);assert.equal(viewer.cursor.hidden,true);assert.equal(viewer.tip.hidden,true);
});
test('resource charts keep per-pixel peaks and break at gaps instead of interpolating',()=>{
  const pts=[...ticks(0,1,.01,t=>t===.5?100:1),...ticks(5,6,.01,()=>1)],s={points:pts,gap:.03,tolerance:.015};
  const paths=chartPaths(s,{scale:10,pad:0,left:0,right:100,top:0,height:100,max:100});
  assert.equal((paths.line.match(/M/g)||[]).length,2);assert.match(paths.line,/ 0\.0/);assert.ok(paths.count<=24);
  assert.equal(formatValue(120,'core%'),'120% (1.20 cores)');assert.equal(formatValue(1843,'MiB'),'1.80 GiB');
  assert.equal(chartScale({unit:'core%'},[250]),250);assert.equal(chartScale({unit:'host%'},[3]),100);
});
test('missing telemetry is shown as unavailable, never as zero or interpolated values',()=>{
  const s={points:[[1,5],[3,7]],tolerance:.15};assert.equal(sampleAt(s,2),null);assert.deepEqual(sampleAt(s,1.1),[1,5]);
  const run=laneRun(['a','b']);run.agents.push({id:'c',label:'c',status:'running'});delete run.counters['host:cpu_util_pct'];
  for(const key of Object.keys(run.counters))if(key.startsWith('host:'))delete run.counters[key];
  const m=resourceModel(run);assert.equal(m.agents.get('c').status,'unavailable');assert.equal(m.host.status,'unavailable');assert.match(m.host.reason,/not recorded/);
  const {viewer}=lanes(run);assert.match(viewer.overlay.innerHTML,/Resources unavailable/);assert.match(viewer.overlay.innerHTML,/Host-wide CPU, memory and pressure were not recorded/);
  assert.doesNotMatch(viewer.canvas.innerHTML,/data-key="chart:__host__/);
});
test('per-agent series come only from a sole-owned sandbox; host and shared measurements are never attributed',()=>{
  const run=laneRun(['a','b']);run.agents[1].sandbox='sb-a';run.counters['workload:cpu_pct']=ticks(0,10,1,()=>999);
  const m=resourceModel(run);assert.equal(m.agents.get('a').status,'shared');assert.equal(m.agents.get('b').status,'shared');assert.deepEqual(m.agents.get('a').series,{});
  assert.equal(m.host.series.cpu.points[0][1],40);
  const own=resourceModel(laneRun(['a','b']));assert.equal(own.agents.get('a').series.cpu.max,250);assert.equal(own.agents.get('a').series.cpu.scope,'sandbox sb-a');
  assert.ok(!Object.values(own.agents.get('a').series).some(s=>s.points.some(p=>p[1]===40&&s.key==='cpu')));
  assert.ok(Math.abs(own.agents.get('a').series.cpu_time.points.at(-1)[1]-4.53)<1e-6);
});
test('each agent highlights its own known critical path; a lane can turn its highlighting off',()=>{
  const {viewer}=lanes(),html=viewer.canvas.innerHTML;
  assert.match(laneMarkup(html,'a'),/causal-node llm critical/);assert.match(laneMarkup(html,'b'),/causal-node llm critical/);
  assert.match(viewer.overlay.innerHTML,/Known path 7\.0s/);
  viewer.laneControl({dataset:{laneHighlight:'b'}});const after=viewer.canvas.innerHTML;
  assert.match(laneMarkup(after,'a'),/critical/);assert.doesNotMatch(laneMarkup(after,'b'),/causal-node llm critical/);
  assert.match(causalSummary(multi()),/System-wide critical path<\/small><strong>Unavailable/);
  assert.match(causalSummary(multi()),/Longest known agent chain \(of 2\)/);
});
test('process subtrees expand inside their agent lane',()=>{
  const run=laneRun(['a','b'],2),g=run.causal_graph;
  g.operations.push({id:'proc',action_id:'proc',label:'gcc',agent:'a',kind:'process',band:'System',start_node:'proc:start',duration:null,metrics:{pid:7}});
  g.nodes.push({id:'proc:start',operation:'proc',agent:'a',time:1.5,timing:'execve_observation'});
  g.edges.push({id:'own',source:'a-op1:start',target:'proc:start',kind:'process_owner',confidence:'provenance',blocking:false});
  const {viewer}=lanes(run);assert.match(viewer.canvas.innerHTML,/data-expand="processes" data-operation="a-op1"/);assert.doesNotMatch(viewer.canvas.innerHTML,/data-causal-node="proc"/);
  viewer.nav.processes.add('a-op1');viewer.invalidate();viewer.paint();
  assert.match(laneMarkup(viewer.canvas.innerHTML,'a'),/data-causal-node="proc"/);assert.doesNotMatch(laneMarkup(viewer.canvas.innerHTML,'b'),/proc/);
});
test('live updates keep the camera and lane state unless following; following tracks the newest events',()=>{
  const {viewer,scroll,n,host}=lanes();viewer.laneControl({dataset:{laneToggle:'b'}});scroll.scrollLeft=40;viewer.paint();
  viewer.update(laneRun(['a','b'],8));assert.equal(scroll.scrollLeft,40);assert.ok(n.collapsed.has('b'));
  host.querySelector('[data-latest]').onclick();assert.equal(n.lock,true);const run=laneRun(['a','b'],20);run.duration=40;viewer.update(run);
  assert.ok(scroll.scrollLeft>40);assert.ok(viewer.geometry.byId.get('b').collapsed);
  viewer.scroll.emit('wheel',{deltaY:120,deltaX:0,preventDefault(){}});assert.equal(n.lock,true,'vertical scrolling keeps live-follow');
  viewer.scroll.emit('wheel',{deltaY:0,deltaX:120,preventDefault(){}});assert.equal(n.lock,false,'horizontal navigation releases it');
});
test('16 agents with thousands of operations stay paged, culled and bounded',()=>{
  const agents=Array.from({length:16},(_,i)=>`agent-${String(i).padStart(2,'0')}`),run=laneRun(agents,300);run.duration=700;
  const started=performance.now(),{viewer,n}=lanes(run,{height:700}),elapsedMs=performance.now()-started;
  assert.equal(viewer.order.length,16);assert.ok([...viewer.slices.values()].every(s=>s.nodes.length<=200));
  const cards=(viewer.canvas.innerHTML.match(/data-causal-node=/g)||[]).length,charts=(viewer.canvas.innerHTML.match(/class="lane-chart /g)||[]).length;
  assert.ok(cards<250,`${cards} cards rendered`);assert.ok(charts<40,`${charts} charts rendered`);assert.ok(elapsedMs<4000,`${elapsedMs}ms`);
  assert.match(viewer.overlay.innerHTML,/data-lane-page="agent-00:1"/);
  viewer.filterAgents(new Set(['agent-03']));assert.deepEqual(viewer.geometry.rows.map(r=>r.id),['agent-03',HOST]);
  assert.ok(Number.parseInt(viewer.scroll.style.height)<=Math.max(320,Math.round(900*.78)));
});
test('timestamp normalization: samples outside the displayed clock or on a foreign clock are withheld',()=>{
  const run=laneRun(['a']);run.duration=5;run.counters['sandbox:sb-a:cpu_pct'].push([-1,99]);
  const m=resourceModel(run);assert.ok(m.agents.get('a').series.cpu.points.every(p=>p[0]>=0&&p[0]<=5));assert.equal(latestSample(m.agents.get('a').series.cpu,5)[0],5);
  const foreign=laneRun(['a']);foreign.clock={events:'perf',resources:'unix',shared:false};const f=resourceModel(foreign);
  assert.equal(f.agents.get('a').status,'unavailable');assert.match(f.agents.get('a').reason,/different clock/);assert.equal(f.host.status,'unavailable');
  const live=laneRun(['a']);delete live.counters['sandbox:sb-a:cpu_pct'];live.counters['cg:sb-a:cpu_us']=ticks(0,10,1,t=>t*500000);
  assert.ok(Math.abs(resourceModel(live).agents.get('a').series.cpu.points[1][1]-50)<1e-6);
});
if(process.env.AGENCY_LANES_TRACE)test('actual multi-agent recording: one lane per agent with its own sandbox series',async()=>{
  const run=JSON.parse(await readFile(process.env.AGENCY_LANES_TRACE,'utf8')),{viewer}=lanes(run,{height:900}),m=viewer.resources;
  const agents=viewer.order.filter(id=>id!=='workflow');assert.ok(agents.length>=2);
  for(const id of agents){const r=m.agents.get(id);assert.equal(r.status,'available',`${id}: ${r.reason}`);assert.ok(r.series.cpu.points.length>100);assert.ok(r.sandboxes.every(s=>s.includes(id.replace(/^agent_/,''))));}
  for(const id of agents)assert.ok(viewer.projection.agentPaths.get(id)?.operations.length>0,`${id} has a known chain`);
  console.log(agents.map(id=>{const s=m.agents.get(id).series;return `${id}: ${s.cpu.points.length} CPU samples (~${s.cpu.interval.toFixed(3)}s), peak ${s.cpu.max.toFixed(0)}% of a core, ${viewer.projection.agentPaths.get(id).operations.length} path ops`;}).join('\n'),`\nhost: ${m.host.status}`);
});
