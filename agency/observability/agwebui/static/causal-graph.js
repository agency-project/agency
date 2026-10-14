import {ExecutionTimeline} from './execution-timeline.js';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const seconds = value => `${Number(value).toFixed(2)}s`;
const bands=['LLM','Tool','System'];
// Append overflow band segments instead of moving existing nodes or hiding
// simultaneous calls. Each segment reserves eight lanes; capacity is unbounded.
export function graphLayout(graph, nav) {
  nav.agents ||= [];nav.lanes ||= new Map();nav.bands ||= new Map();nav.segments ||= [];nav.nextY ||= 60;
  const operations=new Map(graph.operations.map(o=>[o.id,o])),nodes=new Map(graph.nodes.map(n=>[n.id,n]));
  const allocate=(agent,band,index)=>{const segment={id:`${agent}:${band}:${index}`,agent,band,index,y:nav.nextY};nav.nextY+=150;nav.segments.push(segment);return segment;};
  const activeAgents=new Set(graph.operations.map(o=>o.agent));
  for(const agent of activeAgents)if(!nav.agents.includes(agent)){
    nav.agents.push(agent);
    for(const band of bands)nav.bands.set(`${agent}:${band}`,[allocate(agent,band,0)]);
    nav.nextY+=30;
  }
  const endsByBand=new Map();
  for(const o of graph.operations){
    const slot=`${o.agent}:${o.band}`,ends=endsByBand.get(slot)||[];
    const start=nodes.get(o.start_node)?.time??Infinity,end=nodes.get(o.end_node)?.time??Infinity;
    let lane=nav.lanes.get(o.id);
    if(lane==null){lane=ends.findIndex(at=>at<=start);if(lane<0)lane=ends.length;nav.lanes.set(o.id,lane);}
    ends[lane]=Math.max(ends[lane]??-Infinity,end);endsByBand.set(slot,ends);
    const segments=nav.bands.get(slot);
    while(segments.length<=Math.floor(lane/8))segments.push(allocate(o.agent,o.band,segments.length));
  }
  const segments=nav.segments.filter(s=>activeAgents.has(s.agent)&&(!nav.agent||s.agent===nav.agent))
    .map((s,i)=>({...s,y:nav.agent?60+i*150:s.y}));
  const segmentMap=new Map(segments.map(s=>[s.id,s]));
  const positions=new Map(graph.nodes.map(n=>{
    const lane=nav.lanes.get(n.operation)||0,slot=nav.bands.get(`${n.agent}:${n.band}`)?.[Math.floor(lane/8)];
    const segment=slot&&segmentMap.get(slot.id);
    return [n.id,{x:180+n.time*nav.scale,y:(segment?.y??slot?.y??60)+32+(lane%8)*16,node:n,operation:operations.get(n.operation)}];
  }));
  return {positions,height:Math.max(320,nav.agent?segments.length*150+100:nav.nextY+40),agents:nav.agents,segments};
}
export function visibleGraph(graph, positions, viewport, inferred=false, agent='', provenance=false) {
  const intersects=(a,b)=>Math.max(a.x,b.x)>=viewport.left-40&&Math.min(a.x,b.x)<=viewport.right+40&&Math.max(a.y,b.y)>=viewport.top-40&&Math.min(a.y,b.y)<=viewport.bottom+40;
  const nodes=graph.nodes.filter(n=>{const p=positions.get(n.id);return (!agent||n.agent===agent)&&p.x>=viewport.left-100&&p.x<=viewport.right+100&&p.y>=viewport.top-40&&p.y<=viewport.bottom+40;});
  const edges=graph.edges.filter(e=>{const a=positions.get(e.source),b=positions.get(e.target);return a&&b&&(!agent||a.node.agent===agent||b.node.agent===agent)&&(inferred||e.confidence!=='inferred')&&(provenance||e.confidence!=='provenance')&&intersects(a,b);});
  return {nodes,edges};
}

export class CausalGraph extends ExecutionTimeline {
  constructor(host,navigation,select,onHistory=()=>{}) {
    super(host,navigation,()=>{},onHistory);
    this.selectGraph=select;this.host.classList.add('causal-graph');
    const extra=document.createElement('div');extra.className='causal-controls';
    extra.innerHTML='<label>Agent <select data-causal-agent aria-label="Critical path agent filter"><option value="">All agents</option></select></label><label><input type="checkbox" data-causal-critical checked> Highlight critical path</label><label><input type="checkbox" data-causal-inferred> Show inferred relationships</label><label><input type="checkbox" data-causal-provenance> Show provenance</label><span>● start ◇ end · solid: verified · dotted: provenance · dashed: inferred / unknown</span>';
    host.insertBefore(extra,this.scroll);
    extra.querySelector('[data-causal-agent]').onchange=e=>{this.nav.agent=e.target.value;this.scroll.scrollTop=0;this.nav.top=0;this.paint();};
    extra.querySelector('[data-causal-critical]').checked=navigation.highlight!==false;
    extra.querySelector('[data-causal-critical]').onchange=e=>{this.nav.highlight=e.target.checked;this.paint();};
    extra.querySelector('[data-causal-inferred]').checked=Boolean(navigation.inferred);
    extra.querySelector('[data-causal-inferred]').onchange=e=>{this.nav.inferred=e.target.checked;this.paint();};
    extra.querySelector('[data-causal-provenance]').checked=Boolean(navigation.provenance);
    extra.querySelector('[data-causal-provenance]').onchange=e=>{this.nav.provenance=e.target.checked;this.paint();};
    this.scroll.setAttribute('aria-label','Causal dependency graph; scroll to pan, Control plus wheel to zoom');
    this.scroll.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();this.zoom(Math.exp(-e.deltaY*.002));}},{passive:false});
    this.canvas.addEventListener('click',e=>{const item=e.target.closest('[data-causal-node],[data-causal-edge]');if(item)this.selectGraph({type:item.dataset.causalNode?'causal-node':'causal-edge',id:item.dataset.causalNode||item.dataset.causalEdge});});
    this.canvas.addEventListener('dblclick',e=>{const item=e.target.closest('[data-causal-node]');if(item){this.nav.expanded ||= new Set();const op=this.graph.nodes.find(n=>n.id===item.dataset.causalNode)?.operation;if(this.nav.expanded.has(op))this.nav.expanded.delete(op);else this.nav.expanded.add(op);this.paint();}});
    const notice=host.querySelector('.notice:last-child');if(notice)notice.textContent='Arrows carry dependencies. Connected start/end nodes show measured work. Double-click a node to expand an operation. Unknown tool bounds contribute no measured work; unexplained waits are marked separately; dotted provenance never blocks execution.';
  }
  setSelection(id){
    const node=this.graph?.nodes.find(n=>n.id===id||(n.action_id===id&&n.phase==='start'));
    this.nav.selected=node?.id||id;
    this.canvas.querySelectorAll('[data-causal-node],[data-causal-edge]').forEach(el=>el.classList.toggle('selected',(el.dataset.causalNode||el.dataset.causalEdge)===this.nav.selected));
  }
  revealEvent(){return {revealed:false,reason:"Use graph node selection to inspect dependency evidence."};}
  update(run) {
    this.graph=run.causal_graph||{nodes:[],edges:[],operations:[],summary:{},coverage:{unknown:[],gaps:[]}};
    for(const n of this.graph.nodes)if(n.phase==='open')n.time=run.duration;
    this.graph.summary.execution_duration=run.execution_started_at!=null?run.execution_wall_time:run.duration;
    const filter=this.host.querySelector('[data-causal-agent]');
    const agents=[...new Set(this.graph.nodes.map(n=>n.agent))];
    if(agents.join('\0')!==this.agentKey){filter.innerHTML='<option value="">All agents</option>'+agents.map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join('');this.agentKey=agents.join('\0');}
    filter.value=this.nav.agent||'';
    super.update(run,run.actions,'',run.duration);
  }
  paint() {
    if(!this.graph||!this.run)return;
    const scale=this.nav.scale||1;
    const layout=graphLayout(this.graph,this.nav),positions=layout.positions;
    const width=Math.max(this.scroll.clientWidth,220+(this.run.duration||1)*scale);
    const left=this.scroll.scrollLeft||0,top=this.scroll.scrollTop||0;
    const viewport={left,right:left+this.scroll.clientWidth,top,bottom:top+this.scroll.clientHeight};
    const visible=visibleGraph(this.graph,positions,viewport,this.nav.inferred,this.nav.agent,this.nav.provenance);
    const focused=document.activeElement?.dataset?.causalNode;
    let lines='',dots='',labels='';
    for(const segment of layout.segments){
      const y=segment.y;
      labels+=`<rect class="causal-band" x="${left}" y="${y}" width="${this.scroll.clientWidth}" height="148"/><text x="${left+12}" y="${y+22}">${esc(segment.band)}${segment.index?' · continued':''}</text>`;
      if(segment.band==='LLM'&&!segment.index)labels+=`<text x="${left+12}" y="${y-12}" class="causal-agent">${esc(segment.agent)}</text>`;
      if(segment.index)labels+=`<text x="${left+12}" y="${y+42}">${esc(segment.agent)}</text>`;
    }
    for(const e of visible.edges){
      const a=positions.get(e.source),b=positions.get(e.target),critical=e.critical&&this.nav.highlight!==false;
      const bend=Math.min(60,Math.max(20,(b.x-a.x)/3));
      const path=a.y===b.y?`M ${a.x} ${a.y} L ${b.x} ${b.y}`:`M ${a.x} ${a.y} C ${a.x+bend} ${a.y}, ${b.x-bend} ${b.y}, ${b.x} ${b.y}`;
      if(e.gap>.01&&e.blocking&&e.kind!=='duration')lines+=`<text x="${(a.x+b.x)/2}" y="${(a.y+b.y)/2+15}">? ${seconds(e.gap)} unexplained</text>`;
      lines+=`<g class="causal-edge ${esc(e.confidence)} ${critical?'critical':''} ${e.gap?'has-gap':''} ${this.nav.selected===e.id?'selected':''}" data-causal-edge="${esc(e.id)}" tabindex="0" role="button" aria-label="${esc(e.kind+' · '+e.evidence)}"><title>${esc(e.kind+' · '+e.confidence+' · '+e.evidence)}${e.gap?' · unexplained gap '+seconds(e.gap):''}</title><rect x="${Math.min(a.x,b.x)-6}" y="${Math.min(a.y,b.y)-6}" width="${Math.abs(b.x-a.x)+12}" height="${Math.abs(b.y-a.y)+12}" fill="transparent" pointer-events="none"/><path class="causal-hit" d="${path}"/><path d="${path}" marker-end="url(#causal-arrow${critical?'-critical':''})"/></g>`;
    }
    for(const n of visible.nodes){
      const p=positions.get(n.id),critical=n.critical&&this.nav.highlight!==false;
      const label=`${n.label} · ${n.phase} · ${seconds(n.time)} · ${n.timing}`;
      const shape=n.phase==='end'?`<path d="M ${p.x} ${p.y-6} l 6 6 l -6 6 l -6 -6 Z"/>`:`<circle cx="${p.x}" cy="${p.y}" r="5"/>`;
      dots+=`<g class="causal-node ${n.band.toLowerCase()} ${critical?'critical':''} ${n.timing!=='observed'?'unknown':''} ${this.nav.selected===n.id?'selected':''}" data-causal-node="${esc(n.id)}" tabindex="0" role="button" aria-label="${esc(label)}"><title>${esc(label)} · slack ${n.slack==null?'unknown':seconds(n.slack)} in observed chain</title>${n.phase==='start'?`<rect x="${p.x-8}" y="${p.y-19}" width="${Math.max(28,Math.min(230,n.label.length*6+20))}" height="26" fill="transparent"/>`:''}${shape}${n.phase==='start'?`<text x="${p.x+9}" y="${p.y-7}">${esc(n.label.slice(0,35))}</text>`:''}${this.nav.expanded?.has(n.operation)?`<text x="${p.x+9}" y="${p.y+15}">${esc(n.phase)} ${seconds(n.time)} · ${p.operation.duration==null?'duration unmeasured':seconds(p.operation.duration)} · ${esc(p.operation.outcome)}</text>`:''}</g>`;
    }
    const base=Math.max(.001,10**Math.floor(Math.log10(90/scale))),tick=base*([1,2,5,10].find(v=>v*base>=90/scale)||10);
    for(let at=Math.max(0,Math.floor((left-180)/scale/tick)*tick);at<=(viewport.right-180)/scale;at+=tick)labels+=`<text x="${180+at*scale}" y="${top+20}" class="causal-tick">${seconds(at)}</text>`;
    this.canvas.style.width=`${width}px`;this.canvas.style.height=`${layout.height}px`;
    this.canvas.innerHTML=`<svg width="${width}" height="${layout.height}" aria-label="Execution dependency graph"><defs><marker id="causal-arrow" viewBox="0 0 10 10" refX="12" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#879aad"/></marker><marker id="causal-arrow-critical" viewBox="0 0 10 10" refX="12" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#edba72"/></marker></defs>${labels}${lines}${dots}</svg>`;
    this.canvas.querySelectorAll('[data-causal-node],[data-causal-edge]').forEach(el=>el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();el.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
    if(focused)this.canvas.querySelector(`[data-causal-node="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
    this.host.querySelector('[data-range]').textContent=`${visible.nodes.length} visible / ${this.graph.nodes.length} nodes · ${seconds(left/scale)}–${seconds(viewport.right/scale)}`;
    const coverage=this.host.querySelector('[data-timing-coverage]');coverage.hidden=false;
    coverage.textContent=this.graph.nodes.length?`${this.graph.coverage.explanation} ${this.graph.coverage.unknown.length} missing observations · ${this.graph.coverage.gaps.length} unexplained dependency gaps.`:'No causal evidence recorded for this prefix.';
  }
}

export function causalSummary(graph) {
  if(!graph)return '<p class="notice">No causal graph is available in this recording.</p>';
  const s=graph.summary;
  return `<div class="causal-summary panel"><div><small>Execution wall time</small><strong>${seconds(s.execution_duration)}</strong></div><div><small>Critical path · observed chain</small><strong>${s.critical_path_duration==null?'Unavailable':seconds(s.critical_path_duration)}</strong></div>${bands.map(b=>`<div><small>${b} measured operation time</small><strong>${seconds(s.operation_time[b])}</strong></div>`).join('')}<p>Path contains ${seconds(s.critical_path_measured_work??0)} measured work + ${seconds(s.critical_path_unexplained_wait??0)} unexplained wait. Partial coverage · ${graph.coverage.measured_operations}/${graph.coverage.total_operations} operations timed · concurrent operation times overlap · unexplained waits marked separately</p></div>`;
}
