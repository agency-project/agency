import {systemActivity, overlapDuration} from './investigator-model.js';
export {systemActivity};
import {ExecutionTimeline} from './execution-timeline.js';
import {AGENT_METRICS, AGENT_DEFAULT, HOST_METRICS, HOST_DEFAULT, resourceModel, sampleAt, latestSample, chartPaths, chartScale, formatValue, shortValue} from './resource-lanes.js';
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const seconds = value => Number.isFinite(value)?`${value.toFixed(value<1?3:2)}s`:'unknown';
const elapsed = value => !Number.isFinite(value)?'—':value>=60?`${Math.floor(value/60)}m ${Math.round(value%60)}s`:`${value.toFixed(value<10?1:0)}s`;
const PAGE=200, LIMIT=320, WIDTH=190, HEIGHT=96, PAD=48;
// Vertical geometry of the lane stack. Every lane and chart shares one
// horizontal camera: x = PAD + seconds * scale, everywhere.
export const LANE={ruler:40,header:34,chart:30,chartGap:6,note:26,graphPad:8,gap:10};
export const HOST='__host__';
const blocking=e=>e.confidence==='verified'&&e.blocking===true;
const processRelationship=e=>['process_owner','process_parent','process_spawn'].includes(e.kind)&&['verified','provenance'].includes(e.confidence);
export const semanticLevel=(scale,reference=40)=>scale<reference*.6?'overview':scale>=reference*2?'detail':'medium';
export const rowStep=level=>level==='overview'?70:level==='detail'?224:168;
export function zoomAt(nav,factor,cursor,width,height,extent) {
  const old=nav.scale||nav.defaultScale||40;
  const min=Math.max(.00001,width/Math.max(1,extent)/8),scale=Math.max(min,Math.min(100000,10000000/Math.max(1,extent),old*factor));
  const at=((nav.left||0)+cursor.x-PAD)/old;
  const oldRow=rowStep(semanticLevel(old,nav.defaultScale)),nextRow=rowStep(semanticLevel(scale,nav.defaultScale));
  return {scale,left:Math.max(0,PAD+at*scale-cursor.x),top:Math.max(0,((nav.top||0)+cursor.y-52)*nextRow/oldRow+52-cursor.y),overviewRows:semanticLevel(scale,nav.defaultScale)==='overview'?nav.overviewRows:null};
}
export function fitView(bounds,width,height,reference=40) {
  const scale=Math.max(.00001,(width-PAD-WIDTH-32)/Math.max(.001,bounds.end-bounds.start));
  return {scale,left:Math.max(0,bounds.start*scale),top:0,overviewRows:semanticLevel(scale,reference)==='overview'?Math.max(1,Math.floor((height-80)/70)):null};
}
export function panFrom(origin,point) {return {left:Math.max(0,origin.left+origin.x-point.x),top:Math.max(0,origin.top+origin.y-point.y)};}

// Only three primary operation types. Syscalls and scheduler bookkeeping stay
// in the Systems profiler. Filtering never contracts a removed operation into
// a new edge. Unknown gaps annotate their original dependency, not fake work.
// Critical flags come from each agent's own chain when the backend provides
// them; no chain spans agents.
export function operationGraph(graph, actions=[], clock=null) {
  const raw=new Map((graph.nodes||[]).map(n=>[n.id,n])),actionMap=new Map(actions.map(a=>[a.id,a]));
  const analysis=graph.operation_critical_path||graph.critical_path||{},perAgent=graph.agent_critical_paths||null;
  const chains=perAgent?Object.values(perAgent):[analysis];
  const criticalNodes=new Set(chains.flatMap(c=>c.node_ids||[])),criticalEdges=new Set(chains.flatMap(c=>c.edge_ids||[]));
  const nodes=[],lookup=new Map(),boundary=new Map();let omitted=0,auxiliary=0;
  for(const op of graph.operations||[]) {
    const action=actionMap.get(op.action_id||op.id),kind=op.kind||action?.kind;
    if(kind==='syscall'){omitted++;continue;}
    const category=kind==='model'||op.band==='LLM'?'llm':kind==='process'?'process':op.band==='Tool'?'tool':null;
    if(!category){auxiliary++;continue;}
    const start=raw.get(op.start_node),end=raw.get(op.end_node),open=raw.get(op.open_node),chain=perAgent?perAgent[op.agent]||{}:analysis;
    const n={...op,kind:category,operation:op.id,start:start?.time,end:end?.time,open:open?(Number.isFinite(clock)?clock:open.time):undefined,
      model:action?.model||action?.metadata?.model||op.model||op.metrics?.model,
      preview:action?.command||action?.arguments||op.command_preview||'',timing:start?.timing,
      critical:[op.start_node,op.end_node,op.open_node].some(id=>criticalNodes.has(id)),slack:op.duration!=null&&start?.timing==='observed'?(chain.slack?.[op.start_node]??null):null};
    nodes.push(n);lookup.set(n.id,n);
    for(const id of [op.start_node,op.end_node,op.open_node])if(id)boundary.set(id,op.id);
  }
  const edges=[];
  for(const e of graph.edges||[]) {
    const source=boundary.get(e.source),target=boundary.get(e.target);
    if(!source||!target||source===target)continue;
    edges.push({...e,source,target,evidence_id:e.id,sourceTime:raw.get(e.source)?.time,targetTime:raw.get(e.target)?.time,critical:criticalEdges.has(e.id)});
  }
  const path=[...new Set(chains.flatMap(c=>c.node_ids||[]).map(id=>boundary.get(id)).filter(Boolean))];
  const agentPaths=new Map(Object.entries(perAgent||{}).map(([agent,c])=>[agent,{...c,operations:[...new Set((c.node_ids||[]).map(id=>boundary.get(id)).filter(Boolean))]}]));
  return {nodes,edges,lookup,path,agentPaths,omitted,auxiliary,analysis,system:graph.system_critical_path||null,rawCount:graph.nodes?.length||0};
}
// One subgraph per agent. Recorded dependencies are agent-scoped; an edge
// between agents is counted and left out of every lane rather than drawn as
// a dependency we cannot interpret.
export function agentGraphs(projection) {
  const groups=new Map(),owner=new Map();
  for(const n of projection.nodes){owner.set(n.id,n.agent);if(!groups.has(n.agent))groups.set(n.agent,{...projection,nodes:[],edges:[]});groups.get(n.agent).nodes.push(n);}
  let crossing=0;
  for(const e of projection.edges){const a=owner.get(e.source);if(a===owner.get(e.target))groups.get(a)?.edges.push(e);else crossing++;}
  return {groups,crossing};
}
export function graphSlice(graph,nav={}) {
  const all=graph.nodes.filter(n=>!nav.agent||n.agent===nav.agent),allIds=new Set(all.map(n=>n.id));
  const owned=new Set(graph.edges.filter(e=>e.kind==='process_owner'&&processRelationship(e)).map(e=>e.target));
  const mode=nav.mode||'critical';
  let sequence=(mode==='processes'?all.filter(n=>n.kind==='process'&&!owned.has(n.id)):mode==='critical'?all.filter(n=>n.critical&&n.kind!=='process'):all.filter(n=>n.kind!=='process'));
  const fallback=!sequence.length&&mode==='critical';if(fallback)sequence=all.filter(n=>n.kind!=='process');
  sequence.sort((a,b)=>(a.start??Infinity)-(b.start??Infinity)||a.id.localeCompare(b.id));
  const requested=nav.offset==='latest'?sequence.length-PAGE:nav.offset||0;
  const offset=Math.max(0,Math.min(requested,Math.max(0,sequence.length-1))),chosen=new Set(sequence.slice(offset,offset+PAGE).map(n=>n.id));
  if(nav.selected&&allIds.has(nav.selected))chosen.add(nav.selected);
  // Only expand captured outgoing process relationships, one disclosure at a
  // time. Unknown attribution is presented separately, never guessed.
  let progress=true;
  while(progress&&chosen.size<LIMIT){progress=false;for(const e of graph.edges)if(processRelationship(e)&&nav.processes?.has(e.source)&&chosen.has(e.source)&&allIds.has(e.target)&&!chosen.has(e.target)){chosen.add(e.target);progress=true;if(chosen.size>=LIMIT)break;}}
  for(const id of nav.branches||[])if(chosen.has(id))for(const e of graph.edges)if(blocking(e)&&(e.source===id||e.target===id)) {
    const other=e.source===id?e.target:e.source;if(allIds.has(other)&&chosen.size<LIMIT)chosen.add(other);
  }
  let nodes=all.filter(n=>chosen.has(n.id));
  const extras=new Map(nodes.map(n=>[n.id,{branches:new Set(),processes:new Set()}]));
  const edges=graph.edges.filter(e=>chosen.has(e.source)&&chosen.has(e.target)&&(blocking(e)||e.confidence==='unknown'||e.confidence==='inferred'&&nav.inferred||e.confidence==='provenance'&&(nav.provenance||processRelationship(e)))&&(mode!=='critical'||fallback||nav.branches?.size||nav.processes?.size||nav.inferred||nav.provenance||e.critical));
  for(const e of graph.edges) {
    if(extras.has(e.source)&&!chosen.has(e.target)&&allIds.has(e.target)) {
      if(processRelationship(e))extras.get(e.source).processes.add(e.target);
      else if(blocking(e))extras.get(e.source).branches.add(e.target);
    }
    if(extras.has(e.target)&&!chosen.has(e.source)&&allIds.has(e.source)&&blocking(e))extras.get(e.target).branches.add(e.source);
  }
  for(const value of extras.values()){value.branches=value.branches.size;value.processes=value.processes.size;}
  if(mode==='processes') {
    // Group only disconnected images of the same executable and actor. This
    // is disclosure of observations, never inferred ancestry or ownership.
    const related=new Set(graph.edges.filter(processRelationship).flatMap(e=>[e.source,e.target])),groups=new Map(),retained=[];
    for(const n of nodes){if(related.has(n.id)||n.id===nav.selected){retained.push(n);continue;}const key=`${n.agent}:${n.metrics?.path||n.label}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(n);}
    for(const members of groups.values())if(members.length===1)retained.push(members[0]);else {const first=members[0];retained.push({...first,id:`group:${first.id}`,label:`${first.label} · ${members.length} processes`,members:members.map(n=>n.id),metrics:{},duration:null,end:members.at(-1).start});}
    nodes=retained;
  }
  return {nodes,edges,extras,sequence,offset,fallback,capped:chosen.size>=LIMIT,hidden:all.length-nodes.length};
}

// Timestamps determine x, always. A deterministic interval pack chooses y
// near connected predecessors. Reserve card clearance at the minimum medium
// zoom. Live lane assignments survive appends/completions and zoom changes.
export function graphLayout(graph,nav={}) {
  nav.lanes ||= new Map();const scale=nav.scale||40,reference=nav.defaultScale||40;
  const incoming=new Map(graph.nodes.map(n=>[n.id,[]]));
  for(const e of graph.edges)if(blocking(e)||processRelationship(e))incoming.get(e.target)?.push(e.source);
  const lookup=new Map(graph.nodes.map(n=>[n.id,n])),occupied=new Map(),clearance=(WIDTH+26)/(reference*.6);
  const ordered=[...graph.nodes].sort((a,b)=>(a.start??a.end??Infinity)-(b.start??b.end??Infinity)||Number(b.critical)-Number(a.critical)||a.id.localeCompare(b.id));
  // Reserve retained lanes first: new evidence cannot move inspected cards.
  for(const n of ordered)if(nav.lanes.has(n.id)) {
    const lane=nav.lanes.get(n.id),at=n.start??n.end;
    if(!occupied.has(lane))occupied.set(lane,[]);occupied.get(lane).push([at??0,(at??0)+clearance]);
  }
  for(const n of ordered)if(!nav.lanes.has(n.id)) {
    const relatives=(incoming.get(n.id)||[]).map(id=>nav.lanes.get(id)).filter(Number.isFinite);
    const preferred=relatives.length?Math.round(relatives.reduce((a,b)=>a+b,0)/relatives.length)+(n.kind==='process'?1:0):0;
    const at=n.start??n.end??0;
    const free=lane=>!(occupied.get(lane)||[]).some(([from,to])=>at<to&&at+clearance>from);
    let lane=[...occupied.keys()].filter(free).sort((a,b)=>Math.abs(a-preferred)-Math.abs(b-preferred)||a-b)[0];
    if(lane==null){lane=0;while(occupied.has(lane))lane++;}
    nav.lanes.set(n.id,lane);if(!occupied.has(lane))occupied.set(lane,[]);occupied.get(lane).push([at,at+clearance]);
  }
  const level=semanticLevel(scale,reference),step=rowStep(level),positions=new Map();
  // Rows are compacted per lane: an agent's graph is as tall as its own
  // occupied rows, while each node keeps its stable row identity.
  const used=[...new Set(ordered.map(n=>nav.lanes.get(n.id)))].sort((a,b)=>a-b),rowOf=new Map(used.map((lane,i)=>[lane,i]));
  let end=0,maxLane=0,start=Infinity,minLane=Infinity,rows=0;
  for(const n of ordered) {
    const lane=nav.lanes.get(n.id),at=n.start??n.end,row=nav.compact?rowOf.get(lane):lane;
    const visualLane=level==='overview'&&nav.overviewRows?Math.min(row,nav.overviewRows-1):row;
    const p={x:PAD+(Number.isFinite(at)?at:0)*scale,y:52+visualLane*step,lane,node:n,unplaced:!Number.isFinite(at)};
    positions.set(n.id,p);end=Math.max(end,n.end??n.open??at??0);start=Math.min(start,at??Infinity);minLane=Math.min(minLane,lane);maxLane=Math.max(maxLane,lane);rows=Math.max(rows,visualLane+1);
  }
  return {positions,rows,step,width:Math.max(400,PAD+end*scale+(level==='detail'?300:WIDTH)+32),height:Math.max(280,52+(level==='overview'&&nav.overviewRows?Math.min(maxLane+1,nav.overviewRows):maxLane+1)*step+48),level,bounds:{start:Number.isFinite(start)?start:0,end,minLane:Number.isFinite(minLane)?minLane:0,maxLane},lookup};
}

// Semantic overview groups colliding labels on an existing lane. Members,
// original timestamp anchors and original edge identities stay available.
export function displayGraph(graph,layout) {
  if(layout.level!=='overview')return {...graph,groups:new Map(graph.nodes.filter(n=>n.members).map(n=>[n.id,n.members])),memberGroups:new Map(graph.nodes.map(n=>[n.id,n.id]))};
  const byLane=new Map(),groups=new Map(),memberGroups=new Map(),nodes=[];
  for(const n of graph.nodes){const lane=layout.positions.get(n.id).y;if(!byLane.has(lane))byLane.set(lane,[]);byLane.get(lane).push(n);}
  for(const list of byLane.values()) {
    list.sort((a,b)=>layout.positions.get(a.id).x-layout.positions.get(b.id).x);
    let members=[];
    const flush=()=>{if(!members.length)return;const first=members[0],last=members.at(-1);if(members.length===1){nodes.push(first);memberGroups.set(first.id,first.id);if(first.members)groups.set(first.id,first.members);}else {
      const actual=members.flatMap(n=>n.members||[n.id]),id=`group:${first.id}`,n={...first,id,label:`${actual.length} operations`,critical:members.some(n=>n.critical),members:actual,duration:null,end:last.end??last.start};
      groups.set(id,n.members);nodes.push(n);layout.positions.set(id,{...layout.positions.get(first.id),node:n});for(const member of members)memberGroups.set(member.id,id);
    }members=[];};
    for(const n of list){if(members.length&&layout.positions.get(n.id).x-layout.positions.get(members.at(-1).id).x>=122)flush();members.push(n);}flush();
  }
  const edges=[],seen=new Set();
  for(const e of graph.edges){const source=memberGroups.get(e.source),target=memberGroups.get(e.target);if(source===target)continue;const key=`${source}:${target}:${e.confidence}:${e.critical}`;if(seen.has(key))continue;seen.add(key);edges.push({...e,displaySource:source,displayTarget:target});}
  return {...graph,nodes,edges,groups,memberGroups};
}
export function visibleGraph(graph,positions,viewport) {
  const cardWidth=viewport.level==='overview'?110:viewport.level==='detail'?300:WIDTH,cardHeight=viewport.level==='overview'?44:viewport.level==='detail'?148:HEIGHT;
  const inside=p=>p&&p.x+cardWidth>=viewport.left&&p.x<=viewport.right&&p.y+cardHeight+30>=viewport.top&&p.y<=viewport.bottom;
  return {nodes:graph.nodes.filter(n=>inside(positions.get(n.id))),edges:graph.edges.filter(e=>{
    const a=positions.get(e.displaySource||e.source),b=positions.get(e.displayTarget||e.target);
    return a&&b&&Math.max(a.x,b.x)>=viewport.left-30&&Math.min(a.x,b.x)<=viewport.right+cardWidth&&Math.max(a.y,b.y)+cardHeight>=viewport.top&&Math.min(a.y,b.y)<=viewport.bottom;
  })};
}

// Route in the clear channel above/below cards. Arrow endpoints are the raw
// source and target event timestamps, not the edge of a readable label.
export function routeEdge(e,positions,scale,level='medium') {
  const a=positions.get(e.displaySource||e.source),b=positions.get(e.displayTarget||e.target);if(!a||!b)return null;
  const height=level==='overview'?44:level==='detail'?148:HEIGHT;
  const sx=PAD+e.sourceTime*scale,tx=PAD+e.targetTime*scale;
  if(!Number.isFinite(sx)||!Number.isFinite(tx))return null;
  const sy=a.y+height+10,ty=b.y+height+10;
  // Horizontal routes use a channel below the cards; the final vertical
  // segment approaches the destination's anchor without crossing its label.
  const channel=Math.max(sy,ty)+(level==='overview'?6:40);
  const obstacles=[...positions.values()].map(p=>({left:p.x-6,right:p.x+(level==='overview'?110:level==='detail'?300:WIDTH)+6,top:p.y-6,bottom:p.y+height+6}));
  const gutter=(x,y)=>{
    const clear=at=>!obstacles.some(o=>at>o.left&&at<o.right&&Math.max(y,channel)>o.top&&Math.min(y,channel)<o.bottom);
    if(clear(x))return x;
    return [x,...obstacles.flatMap(o=>[o.left-2,o.right+2])].filter(clear).sort((a,b)=>Math.abs(a-x)-Math.abs(b-x))[0]??x;
  };
  const gx=gutter(sx,sy),hx=gutter(tx,ty);
  return {sx,tx,sy,ty,path:`M ${sx} ${sy} H ${gx} V ${channel} H ${hx} V ${ty} H ${tx}`};
}

// Vertical stack: per agent a header, its causal graph (when expanded) and
// compact resource charts, then the Host Resources section. Pure, so lane
// placement, collapse and alignment are testable without a browser.
export function laneGeometry(lanes,host) {
  const rows=[];let y=LANE.ruler;
  const charts=(keys,top)=>keys.map((key,i)=>({key,top:top+i*(LANE.chart+LANE.chartGap),height:LANE.chart}));
  for(const lane of [...lanes,...(host?[{...host,id:HOST,kind:'host'}]:[])]) {
    const row={id:lane.id,kind:lane.kind||'agent',top:y,header:{top:y,height:LANE.header},graph:null,charts:[],notes:[],collapsed:Boolean(lane.collapsed)};
    y+=LANE.header;
    if(!lane.collapsed) {
      if(row.kind==='agent'){const height=lane.graphRows?LANE.graphPad+lane.graphRows*lane.step-10:LANE.note;row.graph={top:y,height,empty:!lane.graphRows};y+=height+4;}
      row.charts=charts(lane.charts||[],y);y+=row.charts.length*(LANE.chart+LANE.chartGap);
      for(const note of lane.notes||[]){row.notes.push({text:note,top:y,height:LANE.note});y+=LANE.note;}
    }
    row.bottom=y;y+=LANE.gap;rows.push(row);
  }
  return {rows,height:y+8,byId:new Map(rows.map(r=>[r.id,r]))};
}
// The lane, and relative position within it, under a content y coordinate:
// zoom keeps this point stationary even when card heights change.
export function laneAnchor(geometry,y) {
  const row=geometry.rows.find(r=>y>=r.top&&y<r.bottom+LANE.gap)||null;
  return row?{id:row.id,fraction:(y-row.top)/Math.max(1,row.bottom-row.top)}:null;
}

function shape(n,width,height) {
  if(n.kind==='process')return `<polygon class="causal-shape" points="18,0 ${width-18},0 ${width},${height/2} ${width-18},${height} 18,${height} 0,${height/2}"/>`;
  return `<rect class="causal-shape" width="${width}" height="${height}" rx="${n.kind==='llm'?14:2}"/>`;
}
const shorten=(s,limit=27)=>String(s??'').length>limit?String(s).slice(0,limit-1)+'…':String(s??'');

// Reconcile parsed SVG into the live tree. Keyed children (data-key) keep
// their DOM identity, focus and hit targets across clock and metric updates;
// only changed attributes and text are written.
function reconcile(current,next) {
  for(const attribute of [...current.attributes])if(!next.hasAttribute(attribute.name))current.removeAttribute(attribute.name);
  for(const attribute of next.attributes)if(current.getAttribute(attribute.name)!==attribute.value)current.setAttribute(attribute.name,attribute.value);
  const after=[...next.childNodes],keyed=after.length&&after.every(n=>n.nodeType===1&&n.hasAttribute('data-key'));
  if(keyed&&[...current.childNodes].every(n=>n.nodeType===1&&n.hasAttribute('data-key'))) {
    const old=new Map([...current.children].map(el=>[el.getAttribute('data-key'),el]));
    after.forEach((el,index)=>{const key=el.getAttribute('data-key'),kept=old.get(key);let target=el;if(kept){old.delete(key);reconcile(kept,el);target=kept;}if(current.children[index]!==target)current.insertBefore(target,current.children[index]||null);});
    for(const el of old.values())el.remove();return;
  }
  const before=[...current.childNodes];
  if(before.length!==after.length||before.some((n,i)=>n.nodeType!==after[i].nodeType||n.nodeName!==after[i].nodeName)){current.replaceChildren(...after);return;}
  for(let i=0;i<before.length;i++)if(before[i].nodeType===1)reconcile(before[i],after[i]);else if(before[i].nodeValue!==after[i].nodeValue)before[i].nodeValue=after[i].nodeValue;
}
const statusText={running:'Running',running_tool:'Running tool',running_skill:'Running skill',waiting_llm:'Waiting on LLM',executing:'Running',busy:'Running',paused:'Paused',done:'Finished',finished:'Finished',completed:'Finished',stopped:'Stopped',failed:'Failed',cancelled:'Cancelled',idle:'Idle',waiting:'Waiting'};
const chartTone=key=>key.includes('psi')?'pressure':key==='memory'?'memory':key.startsWith('disk')?'disk':key.startsWith('net')?'net':key==='cpu_time'?'time':'cpu';

export class CausalGraph extends ExecutionTimeline {
  constructor(host,navigation,select,onHistory=()=>{},onLive=null) {
    // Reuse the existing viewport lifecycle, freeze and scheduling machinery.
    super(host,navigation,()=>{},onHistory);this.selectGraph=select;host.classList.add('causal-graph');
    const nav=this.nav;nav.processes ||= new Set();nav.branches ||= new Set();nav.mode ||= 'critical';nav.lanes ||= new Map();
    nav.collapsed ||= new Set();nav.more ||= new Set();nav.offsets ||= new Map();nav.plain ||= new Set();
    if(nav.agent&&!nav.agents)nav.agents=new Set([nav.agent]);nav.agent='';
    nav.clockLayout=2;nav.defaultScale ||= Math.max(12,(this.scroll.clientWidth||900)/20);
    const extra=document.createElement('div');extra.className='causal-controls';
    extra.innerHTML='<label>Show <select data-causal-mode><option value="operations">LLM + tools</option><option value="critical">Critical path + expanded branches</option><option value="processes">Unattributed processes</option></select></label><details class="causal-agent-filter"><summary data-agent-summary>Agents</summary><div class="causal-agent-menu"><div><button data-agents-all>All</button><button data-agents-none>None</button></div><div data-agent-options></div></div></details><button data-lanes-collapse>Collapse all</button><button data-lanes-expand>Expand all</button><label><input data-causal-highlight type="checkbox" checked> Highlight critical paths</label><label><input data-causal-inferred type="checkbox"> Inferred</label><label><input data-causal-provenance type="checkbox"> History provenance</label><button data-collapse>Collapse processes</button><span class="causal-legend">Blue rounded: LLM · slate rectangle: tool · slate hexagon: Linux process · amber outline: known per-agent critical path · solid: blocking · dotted: provenance · dashed: inferred · ? unknown gap</span><span class="causal-legend" data-system-path></span><span data-system-activity></span>';
    host.insertBefore(extra,this.scroll);
    const mode=extra.querySelector('[data-causal-mode]');mode.value=nav.mode;mode.onchange=e=>{this.manual();nav.mode=e.target.value;nav.offsets.clear();this.resetLayout();this.reset();};
    for(const name of ['highlight','inferred','provenance']){const el=extra.querySelector(`[data-causal-${name}]`);el.checked=name==='highlight'?nav.highlight!==false:Boolean(nav[name]);el.onchange=e=>{nav[name]=e.target.checked;if(name==='highlight')nav.plain.clear();this.invalidate();this.paint();};}
    extra.querySelector('[data-collapse]').onclick=()=>{nav.processes.clear();nav.branches.clear();this.invalidate();this.paint();};
    extra.querySelector('[data-lanes-collapse]').onclick=()=>{for(const id of [...this.order||[],HOST])nav.collapsed.add(id);this.invalidate();this.paint();};
    extra.querySelector('[data-lanes-expand]').onclick=()=>{nav.collapsed.clear();this.invalidate();this.paint();};
    extra.querySelector('[data-agents-all]').onclick=()=>this.filterAgents(null);
    extra.querySelector('[data-agents-none]').onclick=()=>this.filterAgents(new Set());
    extra.querySelector('[data-agent-options]').onchange=e=>{const id=e.target?.dataset?.agentOption;if(id==null)return;const next=new Set(nav.agents||this.order||[]);if(e.target.checked)next.add(id);else next.delete(id);this.filterAgents(next.size===(this.order||[]).length?null:next);};
    const controls=host.querySelector('.timeline-controls');
    const reset=document.createElement('button');reset.textContent='Reset';reset.dataset.reset='';controls.insertBefore(reset,host.querySelector('[data-latest]'));reset.onclick=()=>this.reset();
    host.querySelector('[data-fit]').textContent='Fit';host.querySelector('[data-fit]').onclick=()=>this.fit();
    host.querySelector('[data-latest]').textContent='Live';host.querySelector('[data-latest]').onclick=()=>{nav.lock=true;host.querySelector('[data-lock]').checked=true;if(!onLive?.())this.latest();};
    host.querySelector('[data-zoom-in]').onclick=()=>this.zoom(1.25);host.querySelector('[data-zoom-out]').onclick=()=>this.zoom(.8);
    // Sticky-left lane headers and chart labels: one overlay translated by the
    // horizontal scroll, so panning never re-renders it.
    this.overlay=document.createElement('div');this.overlay.className='lane-overlay';this.scroll.append?.(this.overlay);
    this.cursor=document.createElement('div');this.cursor.className='time-cursor';this.cursor.hidden=true;this.scroll.append?.(this.cursor);
    this.tip=document.createElement('div');this.tip.className='time-cursor-tip';this.tip.hidden=true;this.tip.setAttribute('role','status');
    this.scroll.setAttribute('aria-label','Agent lanes on one wall-clock axis; Control plus wheel or pinch to zoom, drag empty space to pan, scroll vertically through agents');
    this.scroll.addEventListener('wheel',e=>{
      if(e.ctrlKey||e.metaKey){e.preventDefault();const rect=this.scroll.getBoundingClientRect();this.zoom(Math.exp(-Math.max(-100,Math.min(100,e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?this.scroll.clientHeight:1)))*.008),{x:e.clientX-rect.left,y:e.clientY-rect.top});}
      // Vertical scrolling moves through agents only; a horizontal gesture
      // navigates time and therefore releases live-follow.
      else if(e.shiftKey||Math.abs(e.deltaX||0)>Math.abs(e.deltaY||0))this.manual(false);
    },{passive:false});
    const interactive='[data-causal-node],[data-causal-edge],[data-expand],[data-lane-toggle],[data-lane-more],[data-lane-page],[data-lane-highlight]';
    this.scroll.addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest(interactive))return;this.manual();this.drag={x:e.clientX,y:e.clientY,left:this.scroll.scrollLeft,top:this.scroll.scrollTop};this.scroll.classList.add('dragging');this.scroll.setPointerCapture(e.pointerId);this.hover(null);});
    this.scroll.addEventListener('pointermove',e=>{if(!this.drag){this.hover(e);return;}this.applyCamera(panFrom(this.drag,{x:e.clientX,y:e.clientY}));});
    this.scroll.addEventListener('pointerleave',()=>this.hover(null));
    for(const name of ['pointerup','pointercancel'])this.scroll.addEventListener(name,()=>{this.drag=null;this.scroll.classList.remove('dragging');});
    this.touches=new Map();
    this.scroll.addEventListener('pointerdown',e=>{if(e.pointerType==='touch'){this.touches.set(e.pointerId,{x:e.clientX,y:e.clientY});this.pinchDistance=null;}});
    this.scroll.addEventListener('pointermove',e=>{if(!this.touches.has(e.pointerId))return;this.touches.set(e.pointerId,{x:e.clientX,y:e.clientY});if(this.touches.size===2){this.drag=null;const [a,b]=[...this.touches.values()],distance=Math.hypot(a.x-b.x,a.y-b.y),rect=this.scroll.getBoundingClientRect();if(this.pinchDistance)this.zoom(distance/this.pinchDistance,{x:(a.x+b.x)/2-rect.left,y:(a.y+b.y)/2-rect.top});this.pinchDistance=distance;}});
    for(const name of ['pointerup','pointercancel'])this.scroll.addEventListener(name,e=>{this.touches.delete(e.pointerId);this.pinchDistance=null;});
    this.canvas.addEventListener('click',e=>{
      const expand=e.target.closest('[data-expand]');if(expand){const id=expand.dataset.operation,set=nav[expand.dataset.expand];if(set.has(id))set.delete(id);else set.add(id);this.manual();this.invalidate();this.paint();return;}
      const item=e.target.closest('[data-causal-node],[data-causal-edge]');if(!item)return;
      if(item.dataset.causalEdge)this.selectGraph({type:'causal-edge',id:item.dataset.causalEdge});
      else {const id=item.dataset.causalNode,members=this.displays?.get(item.closest('[data-lane-graph]')?.dataset.laneGraph)?.groups.get(id);if(members)this.selectGraph({type:'causal-group',id,members});else this.selectGraph({type:'causal-operation',id});}
    });
    this.overlay.addEventListener?.('click',e=>this.laneControl(e.target.closest?.('[data-lane-toggle],[data-lane-more],[data-lane-page],[data-lane-highlight]')));
    host.querySelector('.notice:last-child').textContent='One lane per agent on one wall clock →. Cards keep recorded timestamps; anchor dots and duration bars mark the measured interval. Agent charts are that agent\'s own sandbox cgroup; Host Resources is the whole machine, including work outside Agency. Hover to read every lane at one instant (nearest recorded sample within its sampling interval; gaps are never interpolated). Ctrl/Cmd + wheel or pinch zooms around the pointer; drag empty space to pan; scroll vertically through agents. Resource correlation is not causation, and no system-wide critical path is claimed.';
    host.append?.(this.tip);
  }
  invalidate(){this.layoutCache=null;this.sliceCache=null;this.layouts=null;}
  resetLayout(){this.nav.lanes=new Map();this.invalidate();}
  manual(notify=true){this.nav.lock=false;this.host.querySelector('[data-lock]').checked=false;if(notify)this.onHistory();}
  applyCamera(camera){Object.assign(this.nav,camera);this.paint();this.scroll.scrollLeft=this.nav.left||0;this.scroll.scrollTop=this.nav.top||0;this.lastLeft=this.scroll.scrollLeft;this.lastTop=this.scroll.scrollTop;if(this.scroll.scrollLeft!==this.paintedLeft||this.scroll.scrollTop!==this.paintedTop)this.paint();}
  filterAgents(set){this.nav.agents=set;this.invalidate();this.paint();}
  laneControl(el){
    if(!el)return;const nav=this.nav;
    if(el.dataset.laneToggle){const id=el.dataset.laneToggle;if(nav.collapsed.has(id))nav.collapsed.delete(id);else nav.collapsed.add(id);}
    else if(el.dataset.laneMore){const id=el.dataset.laneMore;if(nav.more.has(id))nav.more.delete(id);else nav.more.add(id);}
    else if(el.dataset.laneHighlight){const id=el.dataset.laneHighlight;if(nav.plain.has(id))nav.plain.delete(id);else nav.plain.add(id);}
    else if(el.dataset.lanePage){const [id,direction]=[el.dataset.lanePage.slice(0,el.dataset.lanePage.lastIndexOf(':')),Number(el.dataset.lanePage.split(':').at(-1))];this.manual();const slice=this.slices?.get(id);nav.offsets.set(id,Math.max(0,(slice?.offset||0)+direction*PAGE));}
    this.invalidate();this.paint();
  }
  zoom(factor,cursor={x:this.scroll.clientWidth/2,y:this.scroll.clientHeight/2}){
    this.manual();const anchor=this.geometry?laneAnchor(this.geometry,this.scroll.scrollTop+cursor.y):null;
    const camera=zoomAt({...this.nav,left:this.scroll.scrollLeft,top:this.scroll.scrollTop},factor,cursor,this.scroll.clientWidth,this.scroll.clientHeight,this.run?.duration||1);
    // Horizontal: the timestamp under the pointer stays put. Vertical: the
    // same point of the same agent lane stays under the pointer.
    camera.top=this.scroll.scrollTop;camera.overviewRows=null;this.invalidate();Object.assign(this.nav,camera);this.paint();
    const row=anchor&&this.geometry?.byId.get(anchor.id);
    if(row)camera.top=Math.max(0,row.top+anchor.fraction*(row.bottom-row.top)-cursor.y);
    this.applyCamera(camera);
  }
  fit(){this.manual();this.paint();const camera=fitView(this.bounds||{start:0,end:this.run?.duration||1},this.scroll.clientWidth,this.scroll.clientHeight,this.nav.defaultScale);camera.top=this.scroll.scrollTop;camera.overviewRows=null;this.invalidate();this.applyCamera(camera);}
  firstStart(){let first=null;for(const slice of this.slices?.values()||[]){const n=slice.nodes.find(n=>n.critical&&Number.isFinite(n.start))||slice.nodes.find(n=>Number.isFinite(n.start));if(n&&(first==null||n.start<first))first=n.start;}return first||0;}
  reset(){this.manual();this.nav.scale=this.nav.defaultScale;this.nav.overviewRows=null;this.invalidate();this.paint();this.applyCamera({scale:this.nav.defaultScale,left:Math.max(0,this.firstStart()*this.nav.defaultScale-20),top:0});}
  latest(){if(!this.run)return;for(const id of this.order||[])this.nav.offsets.set(id,'latest');this.invalidate();this.paint();this.scroll.scrollLeft=Math.max(0,PAD+this.run.duration*this.nav.scale-this.scroll.clientWidth+WIDTH);this.lastLeft=this.nav.left=this.scroll.scrollLeft;this.paint();}
  setSelection(id){const raw=this.graph?.nodes.find(n=>n.id===id),op=this.graph?.operations.find(n=>n.action_id===id);this.nav.selected=raw?.operation||op?.id||id;this.canvas.querySelectorAll('[data-causal-node],[data-causal-edge]').forEach(el=>el.classList.toggle('selected',(el.dataset.causalNode||el.dataset.causalEdge)===this.nav.selected));}
  revealEvent(id){
    const node=this.projection?.nodes.find(n=>n.id===id||n.action_id===id);if(!node||!Number.isFinite(node.start))return {revealed:false,reason:'No timestamped primary operation is recorded for this event.'};
    const nav=this.nav;nav.selected=node.id;if(node.kind==='process')nav.mode='processes';
    if(nav.agents&&!nav.agents.has(node.agent))nav.agents.add(node.agent);nav.collapsed.delete(node.agent);
    this.invalidate();const span=Math.max(3,(node.end??node.start)-node.start),scale=(this.scroll.clientWidth-WIDTH-PAD)/span;nav.scale=scale;this.paint();
    const row=this.geometry?.byId.get(node.agent),p=this.layouts?.get(node.agent)?.positions.get(node.id);
    this.applyCamera({scale,left:Math.max(0,node.start*scale-20),top:Math.max(0,(row?.graph?.top??0)+(p?p.y-52:0)-20)});
    this.manual();this.selectGraph({type:'causal-operation',id:node.id});return {revealed:true};
  }
  update(run){
    this.run=run;this.graph=run.causal_graph||{nodes:[],edges:[],operations:[],coverage:{}};
    this.host.hidden=this.graph.state==='pending';
    if(this.host.hidden)return;
    this.projection=operationGraph(this.graph,run.actions,run.duration);this.split=agentGraphs(this.projection);this.resources=resourceModel(run);this.invalidate();
    // Lanes: every agent with operations or recorded sandbox resources, in
    // order of first observed operation.
    const first=new Map();for(const n of this.projection.nodes)if(Number.isFinite(n.start))first.set(n.agent,Math.min(first.get(n.agent)??Infinity,n.start));
    const ids=new Set([...this.split.groups.keys(),...(run.agents||[]).filter(a=>a.id!=='workflow'&&(this.resources.agents.get(a.id)?.status!=='unavailable'||a.status)).map(a=>a.id)]);
    this.order=[...ids].sort((a,b)=>(first.get(a)??Infinity)-(first.get(b)??Infinity)||a.localeCompare(b));
    const key=this.order.join('\0');if(key!==this.agentKey){this.agentKey=key;this.host.querySelector('[data-agent-options]').innerHTML=this.order.map(id=>`<label><input type="checkbox" data-agent-option="${esc(id)}"> ${esc(this.agentLabel(id))}</label>`).join('');}
    const activity=systemActivity(run);this.host.querySelector('[data-system-activity]').innerHTML=`${this.projection.omitted} syscall operations aggregated · ${activity.unattributed} retained events without recorded ownership · <button data-profiler-category="systems">Systems profiler →</button>`;
    const system=this.projection.system;this.host.querySelector('[data-system-path]').textContent=`System-wide critical path: unavailable — ${system?.reason||'cross-agent dependencies are not captured.'}${system?.cross_agent_edges?` ${system.cross_agent_edges} cross-agent edges recorded but not interpreted.`:''}`;
    if(!this.nav.scale){
      this.nav.scale=this.nav.defaultScale;this.paint();this.nav.left=Math.max(0,this.firstStart()*this.nav.scale-20);
      // Several agents open on the whole execution: who did what, when.
      if(this.visibleAgents().length>1){const camera=fitView(this.bounds,this.scroll.clientWidth,this.scroll.clientHeight,this.nav.defaultScale);this.nav.scale=camera.scale;this.nav.left=camera.left;this.invalidate();}
    }
    this.paint();if(this.restoreNavigation){this.scroll.scrollLeft=this.nav.left||0;this.scroll.scrollTop=this.nav.top||0;this.restoreNavigation=false;this.paint();}if(this.nav.lock)this.latest();
  }
  agentLabel(id){const a=this.run?.agents?.find(a=>a.id===id);return id==='workflow'?'Host workflow':a?.label&&a.label!==id?`${a.label}`:id;}
  visibleAgents(){return (this.order||[]).filter(id=>!this.nav.agents||this.nav.agents.has(id));}
  highlighting(agent){return this.nav.highlight!==false&&!this.nav.plain.has(agent);}
  // Lay out every visible agent's graph once per structural change. Layout
  // is cheap relative to rendering; rendering is culled to the viewport.
  prepare(){
    if(this.layouts)return;
    const nav=this.nav,agents=this.visibleAgents();this.slices=new Map();this.layouts=new Map();let start=Infinity,end=0;
    // At overview scale an agent keeps a few rows; colliding cards on a row
    // aggregate into "N operations" groups that retain their members.
    const overviewRows=agents.length>1?4:Math.max(4,Math.floor(((this.scroll.clientHeight||480)-160)/rowStep('overview')));
    for(const id of agents){
      const graph=this.split.groups.get(id);if(!graph)continue;
      const slice=graphSlice(graph,{...nav,agent:'',offset:nav.offsets.get(id)??0,overviewRows:null});
      const layout=graphLayout(slice,{...nav,lanes:nav.lanes,overviewRows,compact:true});
      this.slices.set(id,slice);this.layouts.set(id,layout);
      if(slice.nodes.length){start=Math.min(start,layout.bounds.start);end=Math.max(end,layout.bounds.end);}
    }
    this.bounds={start:Number.isFinite(start)?start:0,end:Math.max(end,Number.isFinite(start)?start+.001:this.run?.duration||1)};
  }
  laneSpecs(){
    const r=this.resources,nav=this.nav,step=rowStep(semanticLevel(nav.scale||40,nav.defaultScale));
    const lanes=this.visibleAgents().map(id=>{
      const res=r.agents.get(id),layout=this.layouts.get(id),more=nav.more.has(id),notes=[];
      let charts=[];
      if(!res||res.status!=='available')notes.push(`Resources: ${res?.reason||'no sandbox recorded for this actor.'}`);
      else {charts=(more?AGENT_METRICS.map(m=>m.key):AGENT_DEFAULT).filter(k=>res.series[k]);const missing=(more?AGENT_METRICS.map(m=>m.key):AGENT_DEFAULT).filter(k=>!res.series[k]);if(missing.length)notes.push(`Not recorded for this sandbox: ${missing.map(k=>AGENT_METRICS.find(m=>m.key===k).label).join(', ')}`);}
      return {id,kind:'agent',collapsed:nav.collapsed.has(id),graphRows:layout?.rows||0,step,charts,notes};
    });
    const more=nav.more.has(HOST),keys=more?HOST_METRICS.map(m=>m.key):HOST_DEFAULT,notes=[];
    if(r.host.status!=='available')notes.push(r.host.reason);else {const missing=keys.filter(k=>!r.host.series[k]);if(missing.length)notes.push(`Not recorded: ${missing.map(k=>HOST_METRICS.find(m=>m.key===k).label).join(', ')}`);}
    return {lanes,host:{collapsed:nav.collapsed.has(HOST),charts:keys.filter(k=>r.host.series[k]),notes}};
  }
  // Shared y scales: agent lanes are comparable at a glance.
  scales(){
    const r=this.resources,out=new Map();
    for(const metric of AGENT_METRICS)out.set(`agent:${metric.key}`,chartScale(metric,this.visibleAgents().map(id=>r.agents.get(id)?.series[metric.key]?.max||0)));
    for(const metric of HOST_METRICS){const s=r.host.series[metric.key];out.set(`host:${metric.key}`,chartScale(metric,[s?.max||0],{limit:s?.limit}));}
    return out;
  }
  paint(){
    if(this.graph?.state==='pending')return;
    if(!this.projection)return;
    this.prepare();
    const nav=this.nav,scale=nav.scale||40,level=semanticLevel(scale,nav.defaultScale),specs=this.laneSpecs(),geometry=this.geometry=laneGeometry(specs.lanes,specs.host),scales=this.scales();
    // A single agent uses the space it needs; many agents scroll vertically.
    const limit=Math.max(320,Math.round((globalThis.innerHeight||900)*.78)),wanted=Math.max(260,Math.min(limit,geometry.height+16));
    if(this.scroll.style.height!==`${wanted}px`)this.scroll.style.height=`${wanted}px`;
    const left=this.scroll.scrollLeft,top=this.scroll.scrollTop,clientWidth=this.scroll.clientWidth,clientHeight=this.scroll.clientHeight;
    this.paintedLeft=left;this.paintedTop=top;
    const contentWidth=Math.max(400,PAD+Math.max(this.bounds.end,this.run?.duration||0)*scale+(level==='detail'?300:WIDTH)+32);
    const width=Math.max(clientWidth,contentWidth),height=Math.max(clientHeight,geometry.height);
    const viewport={left:left-20,right:left+clientWidth+20,top:top-40,bottom:top+clientHeight+40,level};
    const focused=document.activeElement?.dataset?.causalNode;
    this.displays=new Map();let lanes='',charts='',cards=0,overlay='';
    for(const row of geometry.rows){
      const inView=row.bottom>=viewport.top&&row.top<=viewport.bottom;
      if(row.kind==='agent'&&row.graph&&!row.graph.empty&&inView&&row.graph.top+row.graph.height>=viewport.top&&row.graph.top<=viewport.bottom){
        const slice=this.slices.get(row.id),layout=this.layouts.get(row.id),dy=row.graph.top+LANE.graphPad-52;
        const display=displayGraph(slice,layout);this.displays.set(row.id,display);
        const local={...viewport,top:viewport.top-dy,bottom:viewport.bottom-dy},visible=visibleGraph(display,layout.positions,local);cards+=visible.nodes.length;
        lanes+=`<g data-key="lane:${esc(row.id)}" data-lane-graph="${esc(row.id)}" transform="translate(0,${dy})">${this.renderGraph(row.id,visible,slice,layout,scale)}</g>`;
      }
      if(inView)for(const chart of row.charts)charts+=this.renderChart(row,chart,scales,{left,clientWidth,scale});
      overlay+=this.renderHeader(row,clientWidth);
      if(!row.collapsed)for(const chart of row.charts)overlay+=this.chartLabel(row,chart,scales);
      if(row.graph?.empty)overlay+=`<div class="lane-note" style="top:${row.graph.top+4}px">No LLM, tool or process operations recorded for this agent${this.split.groups.has(row.id)?' in the current filter':''}.</div>`;
      for(const note of row.notes)overlay+=`<div class="lane-note" style="top:${note.top+4}px">${esc(note.text)}</div>`;
    }
    const stepBase=10**Math.floor(Math.log10(100/scale)),step=[1,2,5,10].map(v=>v*stepBase).find(v=>v*scale>=80)||10*stepBase;
    let ruler='',grid='',backdrop='';
    for(let at=Math.max(0,Math.floor((left-PAD)/scale/step)*step);PAD+at*scale<viewport.right;at+=step){const x=PAD+at*scale;ruler+=`<line x1="${x}" y1="26" x2="${x}" y2="34"/><text x="${x+4}" y="21">${at.toFixed(Math.min(6,Math.max(0,-Math.floor(Math.log10(step)))))}s</text>`;grid+=`<line x1="${x}" y1="${Math.max(LANE.ruler,top)}" x2="${x}" y2="${Math.min(height,top+clientHeight)}"/>`;}
    for(const row of geometry.rows)if(row.bottom>=viewport.top&&row.top<=viewport.bottom)backdrop+=`<rect class="lane-band ${row.kind}" x="${left}" y="${row.top}" width="${clientWidth}" height="${row.bottom-row.top+4}"/>`;
    this.canvas.style.width=`${width}px`;this.canvas.style.height=`${height}px`;
    const markup=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" aria-label="Agent lanes on one wall-clock axis"><defs data-key="defs"><marker id="causal-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-secondary)"/></marker><marker id="causal-arrow-critical" viewBox="0 0 10 10" refX="10" refY="5" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--amber)"/></marker></defs><g data-key="backdrop" class="lane-backdrop">${backdrop}</g><g data-key="grid" class="time-grid">${grid}</g><g data-key="charts" class="resource-layer">${charts}</g><g data-key="lanes" class="graph-layer">${lanes}</g><g data-key="ruler" class="causal-ruler" transform="translate(0,${top})"><rect x="${left}" width="${clientWidth}" height="36"/>${ruler}</g></svg>`;
    if(markup!==this.markup){
      this.markup=markup;const svg=this.canvas.firstElementChild;
      if(!svg||svg.tagName?.toLowerCase()!=='svg'||typeof DOMParser==='undefined')this.canvas.innerHTML=markup;
      else reconcile(svg,new DOMParser().parseFromString(markup,'image/svg+xml').documentElement);
    }
    this.overlay.style.transform=`translateX(${left}px)`;this.overlay.style.width=`${clientWidth}px`;this.overlay.style.height=`${height}px`;
    if(overlay!==this.overlayMarkup){const active=document.activeElement,key=active&&this.overlay.contains?.(active)?['laneToggle','laneMore','laneHighlight','lanePage'].map(k=>active.dataset?.[k]!=null?`[data-${k.replace(/[A-Z]/g,c=>'-'+c.toLowerCase())}="${CSS.escape(active.dataset[k])}"]`:null).find(Boolean):null;this.overlayMarkup=overlay;this.overlay.innerHTML=overlay;if(key)this.overlay.querySelector?.(key)?.focus({preventScroll:true});}
    this.canvas.querySelectorAll('[data-causal-node],[data-causal-edge],[data-expand]').forEach(el=>el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();el.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
    if(focused)this.canvas.querySelector(`[data-causal-node="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
    const agents=this.visibleAgents().length;
    this.host.querySelector('[data-range]').textContent=`${seconds(Math.max(0,(left-PAD)/scale))}–${seconds((left+clientWidth-PAD)/scale)} · ${level} · ${agents}/${(this.order||[]).length} agents · ${cards} cards / ${this.projection.nodes.length} operations`;
    const summary=this.host.querySelector('[data-agent-summary]');summary.textContent=`Agents: ${this.nav.agents?`${agents} of ${(this.order||[]).length}`:`all ${(this.order||[]).length}`}`;
    for(const box of this.host.querySelector('[data-agent-options]').querySelectorAll?.('[data-agent-option]')||[])box.checked=!this.nav.agents||this.nav.agents.has(box.dataset.agentOption);
    let hidden=0,capped=false,fallback=0;for(const s of this.slices.values()){hidden+=s.hidden;capped||=s.capped;fallback+=s.fallback?1:0;}
    const coverage=this.host.querySelector('[data-timing-coverage]');coverage.hidden=false;coverage.textContent=`${this.graph.coverage?.explanation||'Partial dependency coverage.'} ${hidden} operations collapsed. ${this.projection.auxiliary} scheduler/control operations remain in the profiler. ${capped?'Expansion limit reached in a lane; use paged process view.':''} ${fallback?`${fallback} lane${fallback===1?' has':'s have'} no verified operation chain in this page.`:''} Clock: ${this.resources.clock?`events and resources both on ${this.resources.clock.events}${this.resources.clock.shared===false?' — NOT shared; resources withheld':''}`:'event clock'}.`;
    if(this.hoverAt)this.hover(this.hoverAt);
  }
  renderGraph(agent,visible,slice,layout,scale){
    const positions=layout.positions,level=layout.level,highlighting=this.highlighting(agent);let lines='',cards='';
    for(const e of visible.edges){const route=routeEdge(e,positions,scale,level);if(!route)continue;const critical=e.critical&&highlighting;
      lines+=`<g data-key="edge:${esc(e.evidence_id)}" class="causal-edge ${esc(e.confidence)} ${critical?'critical':''} ${highlighting&&!e.critical?'dimmed':''} ${this.nav.selected===e.evidence_id?'selected':''}" data-causal-edge="${esc(e.evidence_id)}" tabindex="0" role="button" aria-label="${esc(e.kind+' · '+e.confidence+' · '+e.evidence)}"><title>${esc(e.kind+' · '+e.confidence+' · '+e.evidence)}${blocking(e)&&e.gap?' · unknown cause of '+seconds(e.gap)+' gap':''}</title><path class="causal-hit" d="${route.path}"/><path d="${route.path}" marker-end="url(#causal-arrow${critical?'-critical':''})"/>${blocking(e)&&e.gap>0?`<text class="causal-gap" x="${(route.sx+route.tx)/2}" y="${Math.max(route.sy,route.ty)+13}">?${level==='overview'||e.gap*scale<65?'':seconds(e.gap)}</text>`:''}</g>`;
    }
    for(const n of visible.nodes){const p=positions.get(n.id),extra=slice.extras.get(n.id)||{},overview=level==='overview',detail=level==='detail';const w=overview?110:detail?300:WIDTH,h=overview?44:detail?148:HEIGHT;
      const name=n.members?n.label:n.kind==='llm'?(n.model||'LLM request'):n.label||'Unnamed process';
      const title=`${name} · ${seconds(n.duration)} · ${seconds(n.start)} → ${seconds(n.end)} · ${n.outcome||'unknown'}${n.members?' · aggregated recorded operations':''}${n.critical?' · on this agent\'s known critical path':''}`;
      let body=`<text class="causal-label" x="${w/2}" y="21" text-anchor="middle">${esc(shorten(name,overview?16:27))}</text>`;
      if(overview)body+=`<text x="${w/2}" y="37" text-anchor="middle">${n.members?n.members.length+' recorded operations':esc(n.kind==='process'?'PID '+(n.metrics?.pid??'?'):seconds(n.duration))}</text>`;
      else {
        body+=`<text class="causal-duration" x="${w/2}" y="43" text-anchor="middle">${esc(n.duration==null?'Runtime unknown':seconds(n.duration))}</text>`;
        const preview=n.members?`${n.members.length} recorded observations`:n.kind==='process'?`PID ${n.metrics?.pid??'?'} · PPID ${n.metrics?.ppid??'?'}`:n.kind==='llm'?n.outcome||'Status unknown':String(n.preview||'Arguments unavailable').split('\n')[0];
        body+=`<text x="${w/2}" y="62" text-anchor="middle">${esc(shorten(preview,27))}</text><text class="causal-time" x="${w/2}" y="80" text-anchor="middle">${esc(p.unplaced?'Timestamp unknown':seconds(n.start))} → ${esc(seconds(n.end))}</text>`;
      }
      if(detail){
        const fullName=String(name),preview=n.kind==='process'?`PID ${n.metrics?.pid??'?'} · PPID ${n.metrics?.ppid??'?'}`:n.kind==='llm'?n.outcome||'Status unknown':String(n.preview||'Arguments unavailable');
        const exact=value=>Number.isFinite(value)?value.toFixed(6)+'s':'unknown';
        body=`<text class="causal-label" x="150" y="21" text-anchor="middle">${esc(fullName.slice(0,44))}</text><text x="150" y="38" text-anchor="middle">${esc(fullName.slice(44,88))}</text><text class="causal-duration" x="150" y="58" text-anchor="middle">${esc(n.duration==null?'Runtime unknown':seconds(n.duration))}</text><text x="150" y="78" text-anchor="middle">${esc(preview.slice(0,45))}</text><text x="150" y="95" text-anchor="middle">${esc(preview.slice(45,90))}</text><text class="causal-time" x="150" y="114" text-anchor="middle">${esc(exact(n.start))} → ${esc(exact(n.end))}</text><text x="150" y="135" text-anchor="middle">${esc(n.slack!=null?'Slack '+seconds(n.slack):'')}</text>`;
      }
      let buttons='';if(!overview&&!n.members)for(const [key,label] of [['processes','Processes'],['branches','Branches']]){const expanded=this.nav[key].has(n.id);if(extra[key]||expanded)buttons+=`<g data-expand="${key}" data-operation="${esc(n.id)}" tabindex="0" role="button" aria-label="${expanded?'Collapse':'Expand'} ${label.toLowerCase()} for ${esc(n.label||name)}" transform="translate(${key==='processes'?0:99},${h+34})"><rect x="0" y="-12" width="91" height="24" rx="4"/><text x="45" text-anchor="middle">${expanded?'−':'+'} ${label}${extra[key]?' '+extra[key]:''}</text></g>`;}
      const intervalEnd=Number.isFinite(n.end)?PAD+n.end*scale:Number.isFinite(n.open)?PAD+n.open*scale:null;
      const interval=Number.isFinite(n.start)&&intervalEnd!=null?`<line class="causal-interval ${n.members?'causal-group-span':n.end==null?'ongoing':n.duration==null?'bounds':''}" x1="0" y1="${h+10}" x2="${intervalEnd-p.x}" y2="${h+10}"/><circle class="causal-end" cx="${intervalEnd-p.x}" cy="${h+10}" r="3"/>`:'';
      cards+=`<g data-key="node:${esc(n.id)}" transform="translate(${p.x},${p.y})" class="causal-node ${n.kind} ${n.critical&&highlighting?'critical':''} ${highlighting&&!n.critical?'dimmed':''} ${this.nav.selected===n.id?'selected':''}" data-causal-node="${esc(n.id)}" tabindex="0" role="button" aria-label="${esc(title)}"><title>${esc(title)}</title>${shape(n,w,h)}${body}${interval}${p.unplaced?'':`<circle class="causal-anchor" cx="0" cy="${h+10}" r="4"/>`}${buttons}</g>`;
    }
    return `<g data-key="edges">${lines}</g><g data-key="nodes">${cards}</g>`;
  }
  metricFor(row,key){return row.kind==='host'?{metric:HOST_METRICS.find(m=>m.key===key),series:this.resources.host.series[key],scaleKey:`host:${key}`}:{metric:AGENT_METRICS.find(m=>m.key===key),series:this.resources.agents.get(row.id)?.series[key],scaleKey:`agent:${key}`};}
  renderChart(row,chart,scales,{left,clientWidth,scale}){
    const {series,scaleKey}=this.metricFor(row,chart.key),max=scales.get(scaleKey);
    const paths=chartPaths(series,{scale,pad:PAD,left:left-4,right:left+clientWidth+4,top:chart.top+2,height:chart.height-4,max});
    return `<g data-key="chart:${esc(row.id)}:${chart.key}" class="lane-chart ${chartTone(chart.key)}"><rect class="chart-bg" x="${left}" y="${chart.top}" width="${clientWidth}" height="${chart.height}"/><path class="chart-area" d="${paths.area}"/><path class="chart-line" d="${paths.line}"/></g>`;
  }
  chartLabel(row,chart,scales){
    const {metric,series}=this.metricFor(row,chart.key),max=scales.get(this.metricFor(row,chart.key).scaleKey);
    const unit=metric.unit==='core%'?'% of 1 core':metric.unit==='host%'?'% of all CPUs':metric.unit==='psi%'?'% time stalled':metric.unit;
    return `<div class="lane-chart-label" style="top:${chart.top}px;height:${chart.height}px" title="${esc(`${metric.label} · ${metric.note} · ${series?.scope||''} · sampled every ~${series?.interval?seconds(series.interval):'unknown'}`)}"><span>${esc(metric.label)}</span><small>${esc(unit)} · 0–${esc(shortValue(max,metric.unit==='core%'?'%':metric.unit))}</small></div>`;
  }
  renderHeader(row,clientWidth){
    const nav=this.nav,collapsed=nav.collapsed.has(row.id),caret=`<button class="lane-caret" data-lane-toggle="${esc(row.id)}" aria-expanded="${!collapsed}" aria-label="${collapsed?'Expand':'Collapse'} ${esc(row.kind==='host'?'host resources':row.id)}">${collapsed?'▶':'▼'}</button>`;
    const clock=this.run?.duration;
    if(row.kind==='host'){
      const h=this.resources.host,cpu=latestSample(h.series.cpu,clock),mem=latestSample(h.series.memory,clock);
      const facts=[h.cpus?`${h.cpus} CPUs`:null,cpu?`CPU ${formatValue(cpu[1],'%')}`:null,mem?`Memory ${formatValue(mem[1],'MiB')}${h.series.memory?.limit?` / ${formatValue(h.series.memory.limit,'MiB')}`:''}`:null].filter(Boolean);
      return `<div class="lane-header host" style="top:${row.header.top}px;width:${clientWidth}px">${caret}<strong>Host Resources</strong><span class="lane-scope">whole machine · includes processes outside Agency · not attributed to any agent</span>${facts.map(f=>`<span>${esc(f)}</span>`).join('')}${h.status==='available'?`<button data-lane-more="${HOST}" aria-pressed="${nav.more.has(HOST)}">${nav.more.has(HOST)?'Fewer metrics':'More metrics'}</button>`:''}</div>`;
    }
    const nodes=this.split.groups.get(row.id)?.nodes||[],res=this.resources.agents.get(row.id),agent=this.run?.agents?.find(a=>a.id===row.id);
    let begin=Infinity,end=-Infinity,open=false;for(const n of nodes){if(Number.isFinite(n.start))begin=Math.min(begin,n.start);const stop=n.end??n.open;if(Number.isFinite(stop))end=Math.max(end,stop);if(n.end==null&&Number.isFinite(n.open))open=true;}
    const status=statusText[agent?.status]||(open?'Running':this.run?.mode==='live'&&this.run?.status==='running'?'Idle':nodes.length?'Finished':'No operations');
    const cpu=latestSample(res?.series.cpu,clock),mem=latestSample(res?.series.memory,clock);
    const resource=res?.status==='available'?[cpu?`<span title="Sandbox cgroup CPU at ${seconds(cpu[0])}; 100% = one core">CPU ${esc(formatValue(cpu[1],'core%'))}</span>`:'<span>CPU —</span>',mem?`<span title="Sandbox cgroup memory at ${seconds(mem[0])}">Memory ${esc(formatValue(mem[1],'MiB'))}</span>`:'<span>Memory —</span>']:[`<span class="lane-muted" title="${esc(res?.reason||'')}">${res?.status==='shared'?'Resources: shared sandbox':'Resources unavailable'}</span>`];
    const chain=this.projection.agentPaths.get(row.id),slice=this.slices?.get(row.id),total=slice?.sequence.length||0;
    const path=chain&&Number.isFinite(chain.duration)?`<span class="lane-path" title="Known chain: ${seconds(chain.measured_work)} measured work + ${seconds(chain.unexplained_wait)} unexplained wait. Missing dependencies may hide constraints.">Known path ${esc(elapsed(chain.duration))}</span>`:chain?.error?'<span class="lane-muted">Path unavailable (cyclic evidence)</span>':'';
    const paging=total>PAGE?`<button data-lane-page="${esc(row.id)}:-1" ${slice.offset===0?'disabled':''} aria-label="Earlier operations">◀</button><span>${slice.offset+1}–${Math.min(total,slice.offset+PAGE)} of ${total}</span><button data-lane-page="${esc(row.id)}:1" ${slice.offset+PAGE>=total?'disabled':''} aria-label="Later operations">▶</button>`:'';
    const label=this.agentLabel(row.id);
    return `<div class="lane-header" style="top:${row.header.top}px;width:${clientWidth}px">${caret}<strong title="${esc(row.id)}">${esc(label)}</strong><span class="lane-status ${esc(status.toLowerCase().replace(/\s+/g,'-'))}">${esc(status)}</span><span title="First recorded operation start to latest end">${esc(elapsed(end-begin))}</span>${resource.join('')}${path}<span class="lane-muted">${nodes.length} ops</span>${paging}${chain?`<button data-lane-highlight="${esc(row.id)}" aria-pressed="${this.highlighting(row.id)}" ${this.nav.highlight===false?'disabled':''}>Path ${this.highlighting(row.id)?'on':'off'}</button>`:''}${res?.status==='available'?`<button data-lane-more="${esc(row.id)}" aria-pressed="${nav.more.has(row.id)}">${nav.more.has(row.id)?'Fewer metrics':'More metrics'}</button>`:''}</div>`;
  }
  // Shared crosshair: one timestamp, every visible lane. Values are recorded
  // samples within each series' own tolerance; otherwise "no sample".
  report(t,y){
    const r=this.resources,geometry=this.geometry,top=this.scroll.scrollTop,bottom=top+this.scroll.clientHeight;
    const hovered=geometry?.rows.find(row=>y>=row.top&&y<=row.bottom),chart=hovered?.charts.find(c=>y>=c.top&&y<=c.top+c.height);
    const cell=(s,unit)=>{const p=sampleAt(s,t);return p?`<span title="sample at ${p[0].toFixed(3)}s">${esc(shortValue(p[1],unit))}</span>`:'<span class="lane-muted">no sample</span>';};
    const lines=[];let shown=0,more=0;
    for(const row of geometry?.rows||[]){
      if(row.kind!=='agent'||row.bottom<top||row.top>bottom)continue;
      if(shown>=8){more++;continue;}shown++;
      const res=r.agents.get(row.id),active=row===hovered;
      const values=res?.status==='available'?`CPU ${cell(res.series.cpu,'core%')} · Mem ${cell(res.series.memory,'MiB')}${res.series.cpu_psi?` · PSI ${cell(res.series.cpu_psi,'psi%')}`:''}`:`<span class="lane-muted">${res?.status==='shared'?'shared sandbox':'not recorded'}</span>`;
      lines.push(`<div class="${active?'active':''}"><b>${esc(this.agentLabel(row.id))}</b> ${values}</div>`);
    }
    if(more)lines.push(`<div class="lane-muted">+${more} more agents in view</div>`);
    const h=r.host.series;
    const host=r.host.status==='available'?`<div class="${hovered?.kind==='host'?'active':''}"><b>Host</b> CPU ${cell(h.cpu,'%')} · Mem ${cell(h.memory,'MiB')} · PSI cpu ${cell(h.cpu_psi,'psi%')} mem ${cell(h.memory_psi,'psi%')} io ${cell(h.io_psi,'psi%')}</div>`:'<div class="lane-muted">Host: not recorded</div>';
    let focus='';
    if(chart){const {metric,series}=this.metricFor(hovered,chart.key),p=sampleAt(series,t);focus=`<div class="tip-focus">${esc(hovered.kind==='host'?'Host':this.agentLabel(hovered.id))} · ${esc(metric.label)}: ${p?`<b>${esc(formatValue(p[1],metric.unit==='host%'?'%':metric.unit))}</b> at ${p[0].toFixed(3)}s (±${seconds(series.tolerance)})`:'no sample within the sampling interval'}</div>`;}
    return `<div class="tip-time">t = ${t.toFixed(t<10?3:2)}s</div>${focus}${lines.join('')}${host}`;
  }
  hover(e){
    this.hoverAt=e;
    if(!e||!this.run||!this.geometry){this.cursor.hidden=true;this.tip.hidden=true;return;}
    const rect=this.scroll.getBoundingClientRect(),x=e.clientX-rect.left+this.scroll.scrollLeft,y=e.clientY-rect.top+this.scroll.scrollTop,t=(x-PAD)/(this.nav.scale||40);
    if(!(t>=0)||e.clientY-rect.top<36){this.cursor.hidden=true;this.tip.hidden=true;return;}
    this.cursorTime=t;this.cursor.hidden=false;this.cursor.style.left=`${x}px`;this.cursor.style.height=`${Math.max(this.scroll.clientHeight,this.geometry.height)}px`;
    const html=this.report(t,y);if(html!==this.tipMarkup){this.tipMarkup=html;this.tip.innerHTML=html;}
    this.tip.hidden=false;const host=this.host.getBoundingClientRect?.()||{left:0,top:0,width:1e6},px=e.clientX-host.left,py=e.clientY-host.top;
    this.tip.style.left=`${px+300>host.width?Math.max(4,px-300):px+16}px`;this.tip.style.top=`${py+16}px`;
  }
}
export function causalSummary(graph,run={}) {
  if(graph?.state==='pending')return '<p class="notice">Critical Path will be available after the run completes.</p>';
  if(!graph)return '<p class="notice">No causal graph available.</p>';
  const s=graph.summary||{},raw=new Map((graph.nodes||[]).map(n=>[n.id,n]));
  const chains=Object.values(graph.agent_critical_paths||{}).filter(c=>Number.isFinite(c.duration));
  const longest=chains.length?Math.max(...chains.map(c=>c.duration)):(graph.operation_critical_path||graph.critical_path||{}).duration??s.critical_path_duration;
  const from=Number.isFinite(run.execution_started_at)&&Number.isFinite(run.time_origin)?run.execution_started_at-run.time_origin:0,end=from+(s.execution_duration||0);
  const intervals={LLM:[],Tool:[]};let timed=0,total=0;
  for(const op of graph.operations||[])if(['LLM','Tool'].includes(op.band)&&op.kind!=='syscall') {
    total++;const a=raw.get(op.start_node),b=raw.get(op.end_node);if(a?.timing==='observed'&&b?.timing==='observed'&&Number.isFinite(op.duration)){timed++;intervals[op.band].push({start:a.time,duration:op.duration});}
  }
  const llm=overlapDuration(intervals.LLM,from,end),tool=overlapDuration(intervals.Tool,from,end),attributed=overlapDuration([...intervals.LLM,...intervals.Tool],from,end);
  const links=(graph.edges||[]).filter(e=>blocking(e)&&e.kind!=='duration').length;
  const values=[['Execution wall time',seconds(s.execution_duration)],[chains.length>1?`Longest known agent chain (of ${chains.length})`:'Known critical-path contribution',seconds(longest)],['System-wide critical path','Unavailable'],['LLM interval coverage',seconds(llm)],['Tool interval coverage',seconds(tool)],['Unattributed wall time',seconds(Math.max(0,(s.execution_duration||0)-attributed))],['Dependency coverage',`Partial · ${links} verified blocking links`]];
  const path=graph.operation_critical_path||graph.critical_path||{};
  return `<div class="causal-summary panel">${values.map(([label,value])=>`<div><small>${label}</small><strong>${value}</strong></div>`).join('')}<p>${timed}/${total} LLM/tool operations timed. ${chains.length?`Each agent's known chain is computed over its own recorded dependencies; cross-agent dependencies are not captured, so no chain is the system-wide critical path.`:`Recorded operation chain: ${seconds(path.measured_work||0)} measured work + ${seconds(path.unexplained_wait||0)} unexplained wait.`} LLM and tool coverage use interval unions; they can overlap and must not be added. Unattributed time lies outside their combined measured intervals, including other work or missing observations. Syscalls and process ancestry never establish blocking dependencies.</p></div>`;
}
