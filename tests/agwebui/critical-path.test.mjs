import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {loadModule} from './module-loader.mjs';
const {profileIndex, profileBounds, graphSelection, swimlaneLayout, visibleMarks, dependencyRoute, criticalContribution, historyURL, safeLink, processSelection, NODE_LIMIT} = await loadModule('critical-path-model.js');
const {renderOverview, renderInspector, renderTimeline, renderLaneLabels, renderProcessTable, graphNotice} = await loadModule('critical-path-ui.js');
const operation = (id, agent, start, end, band = 'Tool', critical = false) => ({id, action_id:`event-${id}`, agent, label:id, kind:band === 'LLM' ? 'model' : 'tool', band, start, end, duration:end - start, critical, status:'success', metrics:{}, evidence:['Source invocation identifiers'], limitations:[], process_ids:[]});
const edge = (id, source, target, source_time, target_time, critical = false) => ({id, source, target, source_time, target_time, kind:'wait_result', confidence:'recorded', blocking:true, critical, evidence:'Recorded result wait identifies producer'});
const fixture = () => ({run:{id:'recording', title:'Fan out / join', status:'completed', source:'recorded', duration:45}, bounds:{start:0, end:45}, summary:{wall_time:45, critical_path_duration:45, critical_path_measured:43, critical_path_coverage:43 / 45, aggregate_work:83, agent_count:4, process_count:1, unattributed_time:2, breakdown:{LLM:5, Tool:38, Wait:2}, dominant_operations:[{id:'B', label:'Worker B', duration:40}]},
 operations:[operation('spawn','orchestrator',0,1,'Tool',true), operation('A','a',1,16), operation('B','b',1,41,'Tool',true), operation('C','c',1,26), operation('join','orchestrator',41,42,'Wait',true), operation('final','orchestrator',42,45,'LLM',true)],
 edges:[edge('a-join','A','join',16,41), edge('b-join','B','join',41,41,true), edge('c-join','C','join',26,41), edge('join-final','join','final',42,42,true)],
 agents:['orchestrator','a','b','c'].map(id => ({id, label:`Agent ${id}`, operation_ids:id === 'orchestrator' ? ['spawn','join','final'] : [id.toUpperCase()], parent:id === 'orchestrator' ? null : 'orchestrator'})),
 processes:[{id:'gcc-image', identity:'sandbox:12:1024', pid:12, parent_id:'bash-image', agent:'b', owner_ids:['B'], label:'gcc', start:2, end:40, duration:38, attribution:'Explicit tool invocation ID and process identity', metrics:{cpu_time_s:37, rss_peak_mb:128, io_read_bytes:4096, io_write_bytes:0}, evidence:['Namespace PID with recorded process start'], limitations:[]}],
 critical_path:{duration:45, operation_ids:['spawn','B','join','final'], segments:[{operation_id:'spawn', start:0, end:1, duration:1}, {operation_id:'B', start:1, end:41, duration:40}, {operation_id:'join', start:41, end:42, duration:1}, {operation_id:'final', start:42, end:45, duration:3}], unexplained_wait:0}, coverage:{status:'partial', complete:false, ratio:43 / 45, limitations:['Missing recorded terminal workflow join'], issues:[], malformed_edges:[]}, links:{trace:'/api/recording/trace'}});
test('fan-in view keeps all recorded inputs and global critical identity', () => {
 const p = fixture(), index = profileIndex(p), graph = graphSelection(index);
 assert.deepEqual(graph.nodes.map(n => n.id), ['spawn','A','B','C','join','final']);
 assert.equal(graph.edges.length,4); assert.equal(graph.nodes.find(n => n.id === 'B').critical,true);
 const focused = graphSelection(index,{agent:'orchestrator'}); assert.ok(focused.nodes.some(n => n.agent === 'a')); assert.ok(focused.nodes.some(n => n.agent === 'b'));
 const layout = swimlaneLayout(index, graph, 12); assert.equal(layout.lanes.length,4);
 const html = renderTimeline(index,graph,layout); assert.match(html,/data-cp-edge="b-join"/); assert.match(html,/Agent|orchestrator/);
 const r = dependencyRoute(graph.edges.find(e => e.id === 'a-join'),layout); assert.equal(r.sx,16+16*12); assert.equal(r.tx,16+41*12);
});
test('independent parallel agents retain separate lanes without fabricated edges', () => {
 const p=fixture(); p.edges=[]; p.operations=p.operations.filter(n=>['A','B','C'].includes(n.id)); p.agents=p.agents.filter(a=>a.id!=='orchestrator'); p.critical_path.operation_ids=['B'];
 const index=profileIndex(p), graph=graphSelection(index), layout=swimlaneLayout(index,graph,10);
 assert.equal(graph.edges.length,0); assert.equal(layout.lanes.length,3); assert.deepEqual(graph.nodes.map(n=>n.id),['B']);
 const labels=renderLaneLabels(layout,index); assert.match(labels,/Show local activity/); assert.match(labels,/Agent a/);
});
test('process expansion uses recorded owner and stable identity without increasing critical contribution', () => {
 const p=fixture();p.operations.find(op=>op.id==='B').process_ids=['gcc-image'];
 p.operations.find(op=>op.id==='B').child_ids=['C'];
 const windows=[{sandbox_id:'sandbox-b',scope:'shared_sandbox',shared_with:['b','c'],start:1,end:41,sampled_window:{'I/O pressure (some)':{observed_min:0,observed_peak:12,samples:3}},basis:'Retained recorded samples; shared activity is not tool-owned'}];
 p.operations.find(op=>op.id==='B').metrics.sandbox_resources=windows;p.agents.find(a=>a.id==='b').metrics={sandbox_resources:windows};
 const index=profileIndex(p), graph=graphSelection(index,{expanded:new Set(['B'])});
 const gcc=graph.nodes.find(op=>op.id==='gcc-image'); assert.equal(gcc.overlay,true); assert.equal(gcc.critical,false); assert.equal(gcc.parent_id,'B');
 assert.equal(criticalContribution(index,'B'),40); assert.equal(criticalContribution(index,'gcc-image'),null);
 const html=renderInspector(index,{type:'operation',id:'gcc-image'}); assert.match(html,/sandbox:12:1024/); assert.match(html,/37s/); assert.match(html,/128 MiB/); assert.match(html,/Write bytes/); assert.match(html,/Explicit tool invocation ID/);
 assert.equal(graphSelection(index).nodes.some(op=>op.id==='gcc-image'),false);
 const toolHTML=renderInspector(index,{type:'operation',id:'B'});assert.match(toolHTML,/Recorded child operations · 1/);assert.match(toolHTML,/data-cp-select="C"/);assert.match(toolHTML,/Containment alone does not prove a blocking dependency/);assert.match(toolHTML,/Measurement scope/);assert.match(toolHTML,/shared_sandbox/);assert.match(toolHTML,/I\/O pressure \(some\)/);assert.match(toolHTML,/0–12 · 3 samples/);assert.match(toolHTML,/not establish a cause of latency/);
 assert.match(renderInspector(index,{type:'agent',id:'b'}),/Recorded sandbox resource windows/);
});
test('large compact profiles cap DOM nodes and edges and cull outside viewport', () => {
 const p=fixture(); p.operations=Array.from({length:10000},(_,i)=>operation(`op-${i}`,'a',i,i+.5,'Tool',true)); p.agents=[{id:'a',label:'A'}]; p.edges=[];p.critical_path.operation_ids=[];
 const index=profileIndex(p), graph=graphSelection(index);assert.equal(graph.nodes.length,NODE_LIMIT);assert.equal(graph.hidden,10000-NODE_LIMIT);
 const layout=swimlaneLayout(index,graph,20), visible=visibleMarks(graph,layout,{left:0,right:200,top:0,bottom:300}); assert.ok(visible.nodes.length<20);
 const html=renderTimeline(index,graph,layout,{left:0,right:200,top:0,bottom:300});assert.ok((html.match(/class="cp-node/g)||[]).length<20);
 const later=graphSelection(index,{offset:240}); assert.equal(later.nodes[0].id,'op-240');
 assert.equal(graphSelection(index,{selected:'op-9999'}).nodes.some(n=>n.id==='op-9999'),true);
});
test('incomplete starts are never placed at fabricated zero and missing ends remain inspectable', () => {
 const p=fixture();p.operations=[{...operation('missing-start','a',0,2),start:null,duration:null}, {...operation('missing-end','a',3,4),end:null,duration:null}];p.edges=[];p.critical_path.operation_ids=[];
 const index=profileIndex(p), graph=graphSelection(index,{mode:'all'}), layout=swimlaneLayout(index,graph,20);
 assert.equal(layout.positions.has('missing-start'),false); assert.equal(graph.untimed,1); assert.match(graphNotice(graph),/lack start times/);
 assert.match(renderTimeline(index,graph,layout),/Incomplete timing/);assert.match(renderInspector(index,{type:'operation',id:'missing-end'}),/Unavailable/);
 assert.deepEqual(profileBounds(p),{start:0,end:45});
});
test('coverage, truncation, and malformed graphs are explicit and content is escaped', () => {
 const p=fixture(); p.coverage.status='invalid';p.coverage.malformed_edges=['Clock inversion'];p.truncation={operations:9000,edges:10000};p.run.title='<script>bad</script>';p.operations[0].label='<img src=x onerror=alert(1)>';
 const html=renderOverview(p);assert.match(html,/Invalid dependency graph/);assert.match(html,/Clock inversion/);assert.match(html,/9,000 operations/); assert.doesNotMatch(html,/Critical path<\/small><strong>45s/);
 const inspector=renderInspector(profileIndex(p),{type:'operation',id:'spawn'});assert.match(inspector,/&lt;img/);assert.doesNotMatch(inspector,/<img src/);
 assert.equal(safeLink('javascript:alert(1)'),'');assert.equal(safeLink('//untrusted.example'),'');assert.equal(safeLink('/api/trace'),'/api/trace');
 const url=historyURL(p,p.operations[0]);assert.match(url,/run=recording/);assert.match(url,/event=event-spawn/);
});
test('unknown cross-agent relationships do not create dependency context', () => {
 const p=fixture();p.edges=p.edges.map(e=>({...e,confidence:'unknown',blocking:false}));
 const index=profileIndex(p), graph=graphSelection(index,{agent:'orchestrator'});
 assert.equal(graph.nodes.some(n=>n.agent==='a'),false);assert.equal(graph.nodes.some(n=>n.agent==='b'),false);
});
test('semantic zoom groups short spans without adding durations or losing source identities', () => {
 const p=fixture();p.operations=Array.from({length:100},(_,i)=>operation(`sync-${i}`,'orchestrator',i*.001,i*.001+.00004,'Wait'));p.edges=[];p.critical_path.operation_ids=[];
 const index=profileIndex(p), graph=graphSelection(index,{mode:'all'}), layout=swimlaneLayout(index,graph,1);
 assert.equal(layout.displayNodes.length,1);assert.equal(layout.displayNodes[0].aggregate,true);assert.equal(layout.displayNodes[0].duration,null);assert.equal(layout.displayNodes[0].members.length,100);
 assert.equal(layout.positions.size,101);assert.equal(layout.lanes[0].height,62);
 const detail=renderInspector(index,{type:'group',members:layout.displayNodes[0].members});assert.match(detail,/creates no execution duration or causal relationship/);assert.match(detail,/sync-99/);
 assert.ok(swimlaneLayout(index,graph,10000).displayNodes.length>1);
 assert.ok(swimlaneLayout(index,graph,1,'sync-10').displayNodes.some(n=>n.id==='sync-10'&&!n.aggregate));
});
test('unowned processes remain browsable with bounded paging and metric scope labels', () => {
 const p=fixture();p.processes=Array.from({length:500},(_,i)=>({id:`p-${i}`,identity:`ns:${i}:100`,label:'gcc',agent:i%2?'a':null,attribution:i%2?'sandbox_owner':'unattributed',metrics:{cpu_time_seconds:3,cpu_peak_percent:90,io_read_mb:12,io_write_mb:0,rss_peak_mb:50}}));
 const index=profileIndex(p), page=processSelection(index);assert.equal(page.rows.length,50);assert.equal(page.count,500);assert.equal(processSelection(index,{offset:50}).rows[0].id,'p-50');assert.equal(processSelection(index,{agent:'a'}).count,250);
 const html=renderProcessTable(page);assert.ok((html.match(/<tr>/g)||[]).length===51);assert.match(html,/unattributed/);assert.match(html,/12 \/ 0 MiB/);
 const inspector=renderInspector(index,{type:'operation',id:'p-0'});assert.match(inspector,/CPU time/);assert.match(inspector,/Peak sampled CPU/);assert.match(inspector,/90%/);
});
test('latest prerequisite gate is distinct from an incomplete candidate path', () => {
 const p=fixture();p.operations.find(op=>op.id==='B').gating=true;p.joins=[{operation_id:'join',latest_prerequisite_ids:['B'],latest_completion_time:41,candidate_path_matches_gate:false,prerequisites:[{operation_id:'A',observed_slack:25},{operation_id:'B',observed_slack:0}]}];
 const index=profileIndex(p), html=renderOverview(p);assert.match(html,/Latest recorded join prerequisites/);assert.match(html,/candidate path differs/);
 const inspector=renderInspector(index,{type:'operation',id:'B'});assert.match(inspector,/Observed join gate/);assert.match(inspector,/prevents a definitive global path/);assert.match(inspector,/Observed completion slack/);
});
test('span duration self-edges and older local-history edges do not overwhelm temporal view', () => {
 const p=fixture();p.operations=[operation('one','a',0,1,'LLM',true),operation('two','a',1,2,'Tool',true),operation('three','a',2,3,'LLM',true)];
 p.edges=[edge('duration','one','one',0,1,true),edge('older','one','three',1,2),edge('latest','two','three',2,2,true)];
 const index=profileIndex(p), graph=graphSelection(index,{mode:'all'});assert.equal(graph.edges.some(e=>e.source===e.target),false);assert.ok(graph.edges.some(e=>e.id==='latest'));
 p.operations[0].metrics={cpu_ms:'n/a',runqueue_ms:'n/a',blocked_ms:'n/a'};
 const html=renderInspector(profileIndex(p),{type:'operation',id:'one'});assert.doesNotMatch(html,/Unavailablems/);assert.doesNotMatch(html,/Run-queue delay/);
});
test('simplified path omits container fan-out while retaining original critical and join edges', () => {
 const p=fixture();p.agents=[{id:'a',label:'A'}];p.operations=[{...operation('request','a',0,100,'System',true),container:true},{...operation('engine','a',0,100,'System',true),container:true},...Array.from({length:20},(_,i)=>operation(`op-${i}`,'a',i*4,i*4+3,i%2?'Tool':'LLM',true))];p.edges=[];p.joins=[];
 for(let i=0;i<20;i++){
  p.edges.push({...edge(`request-${i}`,'request',`op-${i}`,0,i*4),kind:'request_start',source_phase:'start'});
  p.edges.push({...edge(`admission-${i}`,'engine',`op-${i}`,0,i*4),kind:'execution_admission',source_phase:'start'});
  p.edges.push({...edge(`complete-${i}`,`op-${i}`,'engine',i*4+3,100),kind:'execution_complete',target_phase:'end'});
  if(i)p.edges.push({...edge(`path-${i}`,`op-${i-1}`,`op-${i}`,(i-1)*4+3,i*4,true),kind:'tool_result'});
 }
 const index=profileIndex(p), compact=graphSelection(index);assert.equal(compact.edges.length,19);assert.ok(compact.edges.every(e=>e.id.startsWith('path-')));assert.equal(compact.edgeHidden,60);
 assert.ok(graphSelection(index,{mode:'all'}).edges.length>compact.edges.length);assert.ok(graphSelection(index,{relationships:true}).edges.length>compact.edges.length);assert.equal(index.incoming.get('engine').length,20);
 const fan=fixture();fan.operations.forEach(op=>op.agent='orchestrator');fan.joins=[{operation_id:'join',prerequisites:['A','B','C'].map(operation_id=>({operation_id})),latest_prerequisite_ids:['B']}];
 const joinGraph=graphSelection(profileIndex(fan));assert.equal(joinGraph.edges.length,4);assert.ok(joinGraph.nodes.some(op=>op.id==='A'));assert.ok(joinGraph.nodes.some(op=>op.id==='C'));
});
test('historical UI reads compact profiler endpoints without a raw event or replay subscription', async () => {
 const source=await readFile(new URL('../../agency/observability/agwebui/static/critical-path.js',import.meta.url),'utf8');
 assert.match(source,/\/api\/critical-path\/runs/);assert.match(source,/\/api\/critical-path\/\$\{encodeURIComponent/);
 assert.doesNotMatch(source,/WebSocket|TrajectoryStream|\/api\/events|\/api\/investigator\/runs/);
});
