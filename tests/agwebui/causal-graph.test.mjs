import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const base=new URL('../../agency/observability/agwebui/static/',import.meta.url);
const timeline=await readFile(new URL('execution-timeline.js',base),'utf8');
const source=(await readFile(new URL('causal-graph.js',base),'utf8')).replace('./execution-timeline.js',`data:text/javascript;base64,${Buffer.from(timeline).toString('base64')}`);
const {graphLayout,visibleGraph,causalSummary}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const graph=(count=2)=>{
  const nodes=[],edges=[],operations=[];
  for(let i=0;i<count;i++){
    const id=`op${i}`;
    nodes.push({id:`${id}:start`,operation:id,agent:'a',band:i%2?'Tool':'LLM',time:i,phase:'start'},{id:`${id}:end`,operation:id,agent:'a',band:i%2?'Tool':'LLM',time:i+1,phase:'end'});
    operations.push({id,agent:'a',band:i%2?'Tool':'LLM',start_node:`${id}:start`,end_node:`${id}:end`});
    edges.push({id,source:`${id}:start`,target:`${id}:end`,confidence:'verified'});
  }
  return {nodes,edges,operations};
};
test('nodes align with wall clock in separate LLM / Tool / System bands',()=>{
  const g=graph(),nav={scale:20},layout=graphLayout(g,nav);
  assert.equal(layout.positions.get('op1:start').x-layout.positions.get('op0:start').x,20);
  assert.equal(layout.positions.get('op0:end').x-layout.positions.get('op0:start').x,20);
  assert.ok(layout.positions.get('op1:start').y>layout.positions.get('op0:start').y);
});
test('live appends, completion and out-of-order evidence preserve existing node positions',()=>{
  const g=graph(),nav={scale:20};
  const first=graphLayout(g,nav).positions.get('op1:start');
  const next=graph(6);next.nodes.push({id:'other',operation:'other',agent:'b',band:'System',time:0});next.operations.push({id:'other',agent:'b',band:'System',start_node:'other'});
  const positions=graphLayout(next,nav).positions;
  assert.deepEqual(positions.get('op1:start'),first);
  assert.ok(positions.get('other').y>first.y);
});
test('viewport culls large traces while keeping crossing dependency arrows',()=>{
  const g=graph(32000),nav={scale:20},layout=graphLayout(g,nav);
  g.edges.push({id:'long',source:'op0:start',target:'op31999:end',confidence:'verified'});
  const visible=visibleGraph(g,layout.positions,{left:10000,right:10900,top:0,bottom:800});
  assert.ok(visible.nodes.length<120);
  assert.ok(visible.edges.length<100);
  assert.ok(visible.edges.some(e=>e.id==='long'));
});
test('inferred toggle and agent filters do not change verified graph data',()=>{
  const g=graph(),nav={scale:20},layout=graphLayout(g,nav),viewport={left:0,right:1000,top:0,bottom:800};
  g.edges.push({...g.edges[0],id:'inferred',confidence:'inferred'});
  assert.ok(!visibleGraph(g,layout.positions,viewport).edges.some(e=>e.id==='inferred'));
  assert.ok(visibleGraph(g,layout.positions,viewport,true).edges.some(e=>e.id==='inferred'));
  assert.deepEqual(visibleGraph(g,layout.positions,viewport,true,'missing'),{nodes:[],edges:[]});
  assert.equal(g.edges.length,3);
});
test('summary qualifies coverage and avoids claiming complete system critical path',()=>{
  const html=causalSummary({summary:{execution_duration:11,critical_path_duration:9,operation_time:{LLM:4,Tool:7,System:0}},coverage:{measured_operations:4,total_operations:5}});
  assert.ok(html.includes('Partial coverage'));
  assert.ok(html.includes('unexplained waits marked separately'));
  assert.ok(html.includes('concurrent operation times overlap'));
  assert.ok(html.includes('9.00s'));
});
test('historical provenance is optional and never obscures default blocking arrows',()=>{
  const g=graph(),layout=graphLayout(g,{scale:20}),viewport={left:0,right:1000,top:0,bottom:800};
  g.edges.push({...g.edges[0],id:'history',confidence:'provenance'});
  assert.ok(!visibleGraph(g,layout.positions,viewport).edges.some(e=>e.id==='history'));
  assert.ok(visibleGraph(g,layout.positions,viewport,false,'',true).edges.some(e=>e.id==='history'));
});
test('parallel calls beyond eight lanes remain distinct and append overflow without moving earlier bands',()=>{
  const g=graph(20);for(const n of g.nodes)n.time=n.phase==='start'?0:10;
  const nav={scale:20},first=graphLayout(graph(2),nav).positions.get('op1:start');
  const layout=graphLayout(g,nav);
  assert.equal(layout.positions.get('op1:start').y,first.y);
  const starts=g.nodes.filter(n=>n.phase==='start').map(n=>layout.positions.get(n.id).y);
  assert.equal(new Set(starts).size,20);
  assert.ok(layout.segments.some(s=>s.index>0));
});
test('agent filtering puts the selected bands in view while preserving unfiltered layout',()=>{
  const g=graph();g.nodes.push({id:'b',operation:'b',agent:'b',band:'Tool',time:0});g.operations.push({id:'b',agent:'b',band:'Tool',start_node:'b'});
  const nav={scale:20},before=graphLayout(g,nav).positions.get('b').y;
  nav.agent='b';assert.ok(graphLayout(g,nav).positions.get('b').y<400);
  nav.agent='';assert.equal(graphLayout(g,nav).positions.get('b').y,before);
});
