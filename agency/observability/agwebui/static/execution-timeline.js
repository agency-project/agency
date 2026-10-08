// Renders canonical actions and counters on one wall-clock axis. The viewport
// owns navigation; live ingestion never changes its scale or scroll when unlocked.
const escape = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const seconds = value => `${Number(value).toFixed(1)}s`;
export class ExecutionTimeline {
  constructor(host, navigation, select) {
    this.host=host; this.nav=navigation; this.select=select;
    host.innerHTML=`<div class="timeline-controls"><button data-zoom-out aria-label="Zoom out">−</button><button data-zoom-in aria-label="Zoom in">+</button><button data-fit>Fit trace</button><button data-latest>Jump to latest</button><label><input type="checkbox" data-lock> Lock to Present</label><span data-range></span></div><div class="execution-timeline-scroll" tabindex="0" aria-label="Execution timeline; scroll horizontally to pan"><div class="execution-timeline-canvas"></div></div><p class="notice">Shared wall-clock axis · drag the scrollbar to pan · overlapping calls occupy subrows · dots are measured samples; gaps are not interpolated. Scrolling backward releases Lock to Present.</p>`;
    this.scroll=host.querySelector('.execution-timeline-scroll');this.canvas=this.scroll.firstElementChild;
    this.scroll.scrollLeft=navigation.left||0;
    host.querySelector('[data-lock]').checked=navigation.lock;
    host.querySelector('[data-lock]').onchange=e=>{navigation.lock=e.target.checked;if(navigation.lock)this.latest();};
    host.querySelector('[data-zoom-in]').onclick=()=>this.zoom(2);
    host.querySelector('[data-zoom-out]').onclick=()=>this.zoom(.5);
    host.querySelector('[data-fit]').onclick=()=>{navigation.scale=null;navigation.left=0;this.scroll.scrollLeft=0;this.update(this.run,this.actions,this.agentFilter,this.extent);};
    host.querySelector('[data-latest]').onclick=()=>this.latest();
    this.scroll.addEventListener('scroll',()=>{
      if(this.nav.lock&&this.scroll.scrollLeft<(this.lastLeft||0)-2){this.nav.lock=false;host.querySelector('[data-lock]').checked=false;}
      this.nav.left=this.lastLeft=this.scroll.scrollLeft;
      this.schedule();
    });
    this.canvas.addEventListener('click',event=>{const block=event.target.closest('[data-timeline-action]');if(block)this.select(block.dataset.timelineAction);});
    this.resize=new ResizeObserver(()=>this.schedule());this.resize.observe(this.scroll);
  }
  destroy(){this.resize.disconnect();cancelAnimationFrame(this.frame);}
  schedule(){if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=null;this.paint();});}
  zoom(factor){const center=(this.scroll.scrollLeft+this.scroll.clientWidth/2-180)/(this.nav.scale||1);this.nav.scale=Math.max(.01,Math.min(10000,this.nav.scale*factor));this.paint();this.scroll.scrollLeft=Math.max(0,center*this.nav.scale-this.scroll.clientWidth/2+180);}
  latest(){this.paint();this.scroll.scrollLeft=this.scroll.scrollWidth;this.lastLeft=this.nav.left=this.scroll.scrollLeft;}
  update(run, actions=run.actions, agentFilter="",extent=run.duration){this.run=run;this.extent=extent;this.actions=actions;this.agentFilter=agentFilter;if(!this.nav.scale)this.nav.scale=Math.max(.01,(this.scroll.clientWidth-210)/Math.max(1,this.extent));this.paint();if(this.nav.lock)this.latest();}
  paint(){
    if(!this.run)return;
    const run=this.run,scale=this.nav.scale,left=180;
    const width=Math.max(this.scroll.clientWidth,left+Math.max(1,this.extent)*scale+30);
    this.canvas.style.width=`${width}px`;
    const from=Math.min(this.extent,Math.max(0,(this.scroll.scrollLeft-left)/scale)),to=(this.scroll.scrollLeft+this.scroll.clientWidth)/scale;
    this.host.querySelector('[data-range]').textContent=`${seconds(from)} – ${seconds(Math.min(this.extent,to))} · ${run.time_origin?new Date(run.time_origin*1000).toISOString():'run-relative time'}`;
    let axis=this.canvas.querySelector('.execution-time-axis');
    if(!axis){axis=document.createElement('div');axis.className='execution-time-axis';this.canvas.append(axis);}
    const step=10**Math.floor(Math.log10(100/scale)),tickStep=[1,2,5,10].map(n=>n*step).find(n=>n*scale>=80)||step*10;
    let ticks='<span class="timeline-track-label">WALL-CLOCK TIME</span>';
    for(let t=Math.ceil(from/tickStep)*tickStep;t<=Math.min(to,this.extent);t+=tickStep)ticks+=`<span class="timeline-tick" style="left:${left+t*scale}px">${seconds(t)}</span>`;
    if(axis.innerHTML!==ticks)axis.innerHTML=ticks;
    const agents=new Map();for(const a of this.actions||[]){if(!agents.has(a.agent))agents.set(a.agent,[]);agents.get(a.agent).push(a);}
    for(const agent of run.agents)if((!this.agentFilter||this.agentFilter===agent.id)&&!agents.has(agent.id)&&agent.id!=='workflow')agents.set(agent.id,[]);
    const rows=[];
    for(const [id,actions] of agents){
      const ends=[];let blocks='';
      for(const a of [...actions].sort((a,b)=>a.start-b.start)){
        const duration=a.outcome==='running'?Math.max(a.duration,run.duration-a.start):a.duration;
        let lane=ends.findIndex(end=>end<=a.start);if(lane<0)lane=ends.length;ends[lane]=a.start+Math.max(duration,0.000001);
        if(a.start>to||a.start+Math.max(duration,5/scale)<from)continue;
        const tip=`${a.name} · ${a.outcome} · ${seconds(a.start)} → ${seconds(a.start+duration)}${a.missing_start?' · start missing':''}`;
        blocks+=`<button class="timeline-block ${a.outcome==='running'?'ongoing':''} ${a.outcome==='failed'?'failed':''} ${a.kind==='model'?'model':''}" data-timeline-action="${escape(a.id)}" title="${escape(tip)}" aria-label="${escape(tip)}" style="left:${left+a.start*scale}px;top:${8+lane*30}px;width:${Math.max(5,duration*scale)}px">${escape(a.name)}</button>`;
      }
      rows.push([`agent:${id}`,Math.max(52,ends.length*30+16),`<span class="timeline-track-label">${escape(run.agents.find(a=>a.id===id)?.label||id)}<small>${actions.length} calls</small></span>${blocks}`]);
    }
    for(const [name,samples] of Object.entries(run.counters||{})){
      const valid=samples.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
      const high=valid.reduce((m,p)=>Math.max(m,p[1]),1);
      // Keep only visible samples and at most one observation per horizontal pixel.
      const pixels=new Map();for(const p of valid)if(p[0]>=from&&p[0]<=to)pixels.set(Math.floor(p[0]*scale),p);
      const points=[...pixels.values()].map(([t,v])=>`<circle cx="${left+t*scale}" cy="${57-v/high*43}" r="3" tabindex="0"><title>${escape(name)}: ${v.toFixed(2)} at ${seconds(t)}</title></circle>`).join('');
      rows.push([`metric:${name}`,76,`<span class="timeline-track-label" title="${escape(name)}">${escape(name)}<small>0 – ${high.toFixed(1)} · sampled</small></span><svg width="${width}" height="76" aria-label="${escape(name)}">${points}</svg>`]);
    }
    if(!Object.keys(run.counters||{}).length)rows.push(['no-metrics',48,'<span class="timeline-no-metrics">System metrics unavailable · no samples recorded</span>']);
    if(!agents.size)rows.unshift(['empty',55,'<span class="timeline-no-metrics">No agent events recorded yet</span>']);
    const old=new Map([...this.canvas.querySelectorAll('[data-track]')].map(e=>[e.dataset.track,e]));
    for(const [index,[id,height,html]] of rows.entries()){let row=old.get(id);if(!row){row=document.createElement('div');row.className='execution-track';row.dataset.track=id;this.canvas.append(row);}old.delete(id);if(this.canvas.children[index+1]!==row)this.canvas.insertBefore(row,this.canvas.children[index+1]||null);row.style.height=`${height}px`;if(row.innerHTML!==html)row.innerHTML=html;}
    for(const row of old.values())row.remove();
  }
}
