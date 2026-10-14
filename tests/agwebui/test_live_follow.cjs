// Run with: node --test tests/agwebui/test_live_follow.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(replay=false,view=null) {
  const listeners={}, nodes=new Map();
  const timers=new Map(),delays=[];let timerId=0;
  const flush=()=>{const pending=[...timers.values()];timers.clear();pending.forEach(fn=>fn());};
  class Element {
    constructor(){this.innerHTML='';this.value='';this.options=[];this.children=[];
      this.handlers={};this.scrollTop=0;this.dataset={};this.parts=new Map();this.classList={toggle:()=>{}};this.animations=[];this.style={setProperty:()=>{}};}
    setAttribute(name,value){this[name]=value;}
    focus(){document.activeElement=this;}
    contains(el){return this===el||this.children.some(c=>c.contains(el))||[...this.parts.values()].some(c=>c.contains(el));}
    get parentElement(){return this.parent;}
    addEventListener(type,fn){this.handlers[type]=fn;}
    emit(type){this.handlers[type]?.();}
    closest(selector){let el=this;while(el){if(selector==='[data-call-card]'&&el.dataset.callCard)return el;el=el.parent||el.host;}return null;}
    getAttribute(name){return name==='data-action'?this.dataset.action:this[name];}
    scrollIntoView(){throw Error('Unexpected automatic scrolling');}
    get firstElementChild(){return this.children[0];}
    querySelector(selector){
      if(this.dataset.agentColumn&&['.agent-current-calls','.agent-history-calls'].includes(selector))return this.querySelector('.agent-call-stack').querySelector(selector);
      if(this.dataset.agentColumn&&selector==='[data-agent-controls]')return this.querySelector('.agent-column-heading').querySelector(selector);
      if(selector==='.empty-state')return null;
      if(selector==='.first-experience')return this.children.find(child=>child.className==='first-experience')||null;
      if(!this.parts.has(selector)){const part=new Element();part.host=this;if(selector==='[data-action]')part.dataset.action=this.dataset.callCard;this.parts.set(selector,part);}
      return this.parts.get(selector);
    }
    querySelectorAll(selector){
      const key={'[data-agent-column]':'agentColumn','[data-call-group]':'callGroup','[data-call-card]':'callCard','[data-episode-card]':'episodeCard'}[selector];
      const descendants=[...this.children,...this.parts.values()];
      return descendants.flatMap(child=>[...(child.dataset[key]?[child]:[]),...child.querySelectorAll(selector)]);
    }
    append(child){this.insertBefore(child,null);}
    insertBefore(child,before){
      child.remove();
      const index=before?this.children.indexOf(before):this.children.length;
      this.children.splice(index,0,child);child.parent=this;
    }
    remove(){if(this.parent){this.parent.children.splice(this.parent.children.indexOf(this),1);this.parent=null;}}
    getBoundingClientRect(){
      let top=0,current=this;
      while(current){if(current.parent)top+=Math.max(0,current.parent.children.indexOf(current))*100-(current.parent.scrollTop||0);current=current.parent||current.host;}
      return {left:0,top,bottom:top+100};
    }
    animate(frames,options){this.animations.push({frames,options});}
  }
  const node=id=>{
    if(!nodes.has(id))nodes.set(id,new Element());
    return nodes.get(id);
  };
  const document={hidden:false, activeElement:null, getElementById:node,
    createElement:tag=>{
      const el=new Element();
      if(tag==='template')Object.defineProperty(el,'content',{get:()=>{
        const button=new Element();
        button.dataset.action=el.innerHTML.match(/data-action="([^"]+)"/)[1];
        button.innerHTML=el.innerHTML;return {firstElementChild:button};
      }});
      return el;
    },
    querySelector:()=>null, querySelectorAll:selector=>[...nodes.values()].filter(el=>selector==='[data-agent-controls]'?el.dataset.agentControls:el.controlMenu), addEventListener:(name,fn)=>{
      (listeners[name]??=[]).push(fn);
    }};
  let subscription;const sent=[],confirmations=[];let autoConfirm=true;
  let metricScope=null,summaryRenders=0,clipboardFails=false;const copied=[];
  const context=vm.createContext({document, URLSearchParams, URL, Date, structuredClone,
    setTimeout:(fn,delay)=>{const id=++timerId;timers.set(id,fn);delays.push(delay);return id;},
    clearTimeout:id=>timers.delete(id),
    location:{search:view?`?view=${view}`:replay?'?mode=replay':''},
    window:{scrollY:0, confirm:()=>true, addEventListener:()=>{}},
    TrajectoryStream:class {
      constructor(options){subscription=options;}
      close(){}
      sendExecution(command,extra){sent.push({type:'execution_command',command,...extra});return true;}
      sendReplay(type,extra){sent.push({type,...extra});return true;}
    },
    matchesAction:()=>true,
    // The summary module is not loaded here; its scope contract is this small.
    metricScope:()=>metricScope,setMetricScope:ids=>{metricScope=ids==null?null:new Set(ids);},renderMetricsSummary:()=>{summaryRenders++;},
    navigator:{clipboard:{writeText:async text=>{if(clipboardFails)throw Error('denied');copied.push(text);}}},
    requestControlConfirmation:options=>{confirmations.push(options);if(autoConfirm&&!options.validate())options.onConfirm();},refreshControlConfirmation:()=>{},
  });
  const source=fs.readFileSync(path.join(__dirname,
    '../../agency/observability/agwebui/static/investigator.js'),'utf8')
    .replace(/^import .*;\n/gm,'').replace(/refresh\(\);\s*$/,'');
  vm.runInContext(fs.readFileSync(path.join(__dirname,
    '../../agency/observability/agwebui/static/investigator-model.js'),'utf8')
    .replace(/export function/g,'function'),context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,
    '../../agency/observability/agwebui/static/trajectory-stream.js'),'utf8')
    .split('export class TrajectoryStream')[0].replace(/export function/g,'function'),context);
  for(const file of ['agent-roster.js','event-details.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../../agency/observability/agwebui/static/'+file),'utf8').replace(/^export /gm,''),context);
  vm.runInContext(source+`
    // Isolate follow behavior from chart/inspector markup, retaining the
    // real stream callback, selection handlers and updateTrajectory path.
    var realRenderSession=renderSession,realRenderInspector=renderInspector;
    render=()=>{};renderRunChrome=()=>renderSystemMetrics();renderSession=()=>{};
    renderInspector=()=>{};catalogOptions=()=>{};updateUrl=()=>{};
    var realReconcile=reconcileActivities;
    reconcileActivities=()=>{lastWindowEnd=state.windowEnd;};
    var paints=0,realUpdate=updateTrajectory;
    updateTrajectory=(...args)=>{paints++;realUpdate(...args);};
    var lastWindowEnd;
    globalThis.ui={state,renderMetricsDashboard,readingState,holdColumn,liveColumn,followLatest,olderCalls,setView,setSelection,syncInspector,toggleInspector,loadRun,selectAction,handleClick,agentExecutionControls,holdHistory,pauseDisplay,resumeDisplay,pauseResumeExecution,stopExecution,sendExecutionCommand,receiveCommandResult,openInteraction,submitInteraction,closeInteraction,renderComposer,renderInteractionStatus,renderInteractionOperation,revealEvent,renderRoster,pauseReplay,wallClockRange,followClock,projectClock,renderSession:realRenderSession,renderInspector:realRenderInspector,actionButton,originalPayload,readingPins,
      windowEnd:()=>lastWindowEnd,reconcile:realReconcile,groupCallStack,renderCallStack,renderSystemMetrics,
      paints:()=>paints,setTimeline:viewer=>{timelineViewer=viewer;},setCausal:viewer=>{causalViewer=viewer;},visibleAgentColumns,addAgentColumn,removeAgentColumn,selectAgentColumn,setContent,trajectoryView,payloadSection,renderFullPayload,pagePayload,fullPayloads,handlePayloadClick,views,renderAgentColumnControls};
  `,context);
  const agentButton=(kind,id='a')=>{
    const attribute=`data-execution-${kind}`,html=context.ui.agentExecutionControls(id);
    const match=html.match(new RegExp(`<button[^>]*${attribute}="${id}"[^>]*>([^<]*)</button>`));
    return {hidden:!match,disabled:!match||/disabled/.test(match[0]),textContent:match?.[1]||'',
      onclick:()=>context.ui.handleClick({target:{closest:selector=>{
        if(!selector.includes(`[${attribute}]`))return null;
        return {dataset:{[`execution${kind[0].toUpperCase()+kind.slice(1)}`]:id},hasAttribute:name=>name===attribute};
      }}})};
  };
  const run={id:'live',duration:1,actions:[{id:'old',agent:'a',episode:'one',start:0,duration:1,kind:'tool',outcome:'success',files:[]}],
    episodes:[{id:'one',actions:['old'],status:'completed'}],agents:[],edges:[],coverage:{}};
  context.ui.loadRun('live',null,replay);
  subscription.onMessage({type:'snapshot',run});
  return {ui:context.ui, copied, failClipboard:value=>clipboardFails=value, metricScope:()=>metricScope&&[...metricScope],summaryRenders:()=>summaryRenders, confirmations,setAutoConfirm:value=>autoConfirm=value, document, node,agentButton,flush,delays,sent,setFetch:fn=>context.fetch=fn,receive:message=>subscription.onMessage(message.execution_controls?{...message,execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},...message.execution_controls}}:message),transport:value=>subscription.onTransport(value),window:context.window,
    feed:agent=>{const columns=node('call-stack').querySelectorAll('[data-agent-column]');const body=(agent?columns.find(c=>c.dataset.agentColumn===agent):columns[0]).querySelector('.agent-call-stack');const column=body.host;return {get children(){return [...column.querySelector('.agent-current-calls').children,...column.querySelector('.agent-history-calls').children];},get firstElementChild(){return this.children[0];}};},
    columnFeed:agent=>node('call-stack').children.find(c=>c.dataset.agentColumn===agent).querySelector('.agent-call-stack'),
    visibility:hidden=>{document.hidden=hidden;listeners.visibilitychange.forEach(fn=>fn());},
    append:(paint=true)=>{subscription.onMessage({type:'patch', patches:[{
      actions:[{id:'new-call',agent:'a',episode:'two',start:1,duration:1,kind:'tool',outcome:'success',files:[]}],
      episodes:[{id:'two',actions:['new-call'],status:'running'}],meta:{duration:2},
    }]});if(paint)flush();},
  };
}

test('trajectory is the default execution view and the inspector starts collapsed',()=>{
  const p=page();p.ui.renderInspector();
  assert.equal(p.ui.state.view,'trajectory');
  assert.equal(p.node('inspector').hidden,true);
  assert.equal(p.node('inspector-toggle').disabled,true);
});
test('top system dashboard follows resource-only patches and freezes with the held display',()=>{
  const p=page(),run=p.ui.state.run;
  run.duration=1;run.status='running';run.counters={'proc:42-100:rss_mb':[[1,999]],'workload:memory_mb':[[1,100]],'workload:cpu_pct':[[1,200]]};
  p.transport('connected');p.ui.renderSystemMetrics();
  const panel=p.node('system-metrics');
  assert.match(panel.innerHTML,/Live samples/);assert.match(panel.innerHTML,/200 %/);
  assert.match(panel.innerHTML,/Workload/);assert.doesNotMatch(panel.innerHTML,/999 MB/);
  p.receive({type:'patch',patches:[{counter_samples:{'workload:memory_mb':[[2,150]]},meta:{duration:2}}]});p.flush();
  assert.match(panel.innerHTML,/150 MB/);
  p.ui.pauseDisplay();p.ui.renderSystemMetrics();
  const held=panel.innerHTML;assert.match(held,/Frozen view/);
  p.receive({type:'patch',patches:[{counter_samples:{'workload:memory_mb':[[3,250]]},meta:{duration:3}}]});p.flush();
  p.ui.renderSystemMetrics();assert.equal(panel.innerHTML,held);
  p.ui.resumeDisplay();assert.match(panel.innerHTML,/250 MB/);
  p.transport('disconnected');p.ui.renderSystemMetrics();assert.match(panel.innerHTML,/Disconnected/);
  run.counters={};p.ui.renderSystemMetrics();assert.match(panel.innerHTML,/No usable system samples/);
});
test('first-use prompt is removed when agent columns become available',()=>{
  const p=page(),stack=p.node('call-stack'),run=structuredClone(p.ui.state.run);
  p.ui.state.run.actions=[];p.ui.renderCallStack();
  assert.match(stack.innerHTML,/class="first-experience"/);
  // Represent the section created by this markup in the small DOM fixture.
  const prompt=p.document.createElement('section');prompt.className='first-experience';stack.append(prompt);
  p.ui.state.run=run;p.ui.renderCallStack();
  assert.equal(stack.children.includes(prompt),false);
  assert.equal(stack.querySelectorAll('[data-agent-column]').length,1);
});
test('explicit alternative execution views remain selected on entry',()=>{
  assert.equal(page(false,'timeline').ui.state.view,'timeline');
  assert.equal(page(false,'profiler').ui.state.view,'profiler');
  // Links to the removed views open the trajectory instead of failing.
  for(const removed of ['context','resources','agents','compare'])assert.equal(page(false,removed).ui.state.view,'trajectory');
});
test('inspector opens on selection, closes without clearing identity and stays closed on updates',()=>{
  const p=page();p.ui.selectAction('old');p.ui.renderInspector();
  const selected=p.ui.state.selected;
  assert.equal(selected.execution,'live');assert.equal(selected.id,'old');
  assert.equal(p.node('inspector').hidden,false);
  p.ui.toggleInspector(false);p.append();p.ui.renderInspector(true);
  assert.equal(p.node('inspector').hidden,true);assert.equal(p.ui.state.selected,selected);
  assert.equal(p.document.activeElement,p.node('inspector-toggle'));
  p.ui.selectAction('new-call');p.ui.renderInspector();
  assert.equal(p.node('inspector').hidden,false);assert.equal(p.ui.state.selected.id,'new-call');
});
test('reloading the same selected recording rebuilds its cleared inspector',()=>{
  const p=page();p.ui.selectAction('old');p.ui.renderInspector();
  const run=structuredClone(p.ui.state.run);
  p.ui.loadRun('live',{type:'action',id:'old'});p.receive({type:'snapshot',run});p.ui.renderInspector();
  assert.match(p.node('inspector').innerHTML,/SELECTION INSPECTOR/);
  assert.equal(p.ui.state.selected.id,'old');
});
test('trajectory and timeline switches retain the same execution and event selection',()=>{
  const p=page();p.ui.selectAction('old');p.ui.renderInspector();p.ui.toggleInspector(false);
  const run=p.ui.state.run,selected=p.ui.state.selected;
  p.ui.setView('timeline');p.ui.renderInspector();p.ui.setView('trajectory');
  assert.equal(p.ui.state.run,run);assert.equal(p.ui.state.selected,selected);
  assert.equal(p.ui.state.inspectorOpen,false);
});
test('live calls keep following while the profiler is hidden and restored',()=>{
  const p=page();p.visibility(true);p.append();p.visibility(false);
  assert.equal(p.ui.state.follow,true);
  assert.equal(p.ui.state.cursor,2);
  assert.equal(p.ui.state.unread.size,0);
  assert.equal(p.ui.windowEnd(),null);
});

test('concurrent completions migrate reused selected cards into start-ordered history',()=>{
  const p=page();p.ui.state.run.actions=[
    {id:'slow',agent:'a',kind:'tool',start:0,duration:20,outcome:'running',files:[]},
    {id:'fast',agent:'a',kind:'tool',start:1,duration:2,outcome:'running',files:[]},
  ];
  p.ui.renderCallStack();p.ui.selectAction('slow');
  const original=p.feed('a').firstElementChild.querySelector('.call-group-grid').children.find(c=>c.dataset.callCard==='slow');
  const button=original.querySelector('[data-action]');button.focus();
  p.receive({type:'updates',patches:[{actions:[{id:'fast',outcome:'success',duration:2},{id:'slow',outcome:'success',duration:20}]}]});
  p.flush();p.ui.renderCallStack();p.ui.renderInspector();
  assert.deepEqual(p.feed('a').children.map(g=>g.dataset.callGroup),['fast','slow']);
  assert.equal(p.feed('a').children[1].querySelector('.call-group-grid').firstElementChild,original);
  assert.equal(p.document.activeElement,button);assert.equal(p.ui.state.selected.id,'slow');
  assert.match(p.node('inspector').innerHTML,/success/);
  assert.equal(p.ui.state.run.actions.filter(a=>a.id==='slow').length,1);
});
test('scrolling holds just that column and preserves expanded reading content during high-volume updates',()=>{
  const p=page();p.ui.state.run.agents=[{id:'a'},{id:'b'}];p.ui.renderCallStack();
  const feed=p.columnFeed('a');feed.scrollTop=100;feed.emit('scroll');
  const column=p.node('call-stack').children[0];
  const card=p.feed('a').firstElementChild.querySelector('.call-group-grid').firstElementChild;
  const details=card.querySelector('.tool-details');details.open=true;details.innerHTML='Expanded immutable evidence';
  p.ui.selectAction('old');
  for(let batch=0;batch<10;batch++)p.receive({type:'updates',patches:[{actions:Array.from({length:500},(_,i)=>({id:`new-${batch*500+i}`,agent:'a',kind:'tool',start:batch*500+i+1,duration:1,outcome:'success',files:[],event_ids:[`event-${batch*500+i}`]}))}]});
  p.flush();p.ui.renderCallStack();
  assert.equal(p.ui.readingState('a').follow,false);assert.equal(p.ui.readingState('b').follow,true);
  assert.equal(p.node('call-stack').children[0],column);assert.equal(feed.scrollTop,100);
  assert.equal(card.querySelector('.tool-details'),details);assert.equal(details.open,true);
  assert.equal(details.innerHTML,'Expanded immutable evidence');
  assert.equal(p.ui.state.selected.id,'old');assert(p.ui.state.run.actions.length<=2001);
  assert.equal(p.ui.state.unread.size,256);assert.equal(p.ui.state.unread.overflow,true);
  assert(p.feed('a').children.length<=102);
  p.ui.followLatest();p.ui.renderCallStack();
  assert.equal(feed.scrollTop,0);assert.equal(p.ui.readingState('a').follow,true);
  assert.equal(p.ui.state.unread.size,0);
});
test('visible agents retain their live history page when another agent dominates the buffer',()=>{
  const p=page();p.ui.state.run.agents=[{id:'a'},{id:'b'}];
  p.receive({type:'updates',patches:[{actions:Array.from({length:60},(_,i)=>({id:`b-${i}`,agent:'b',kind:'tool',start:i,duration:1,outcome:'success',files:[]}))}]});
  p.flush();p.ui.renderCallStack();
  p.ui.readingState('b').follow=false;p.ui.readingState('b').ids=['b-0'];
  p.receive({type:'updates',patches:[{actions:Array.from({length:3000},(_,i)=>({id:`a-${i}`,agent:'a',kind:'tool',start:i+100,duration:1,outcome:'success',files:[]}))}]});
  p.flush();p.ui.followLatest();p.ui.renderCallStack();
  const ids=p.ui.readingState('b').ids;
  assert.equal(ids.length,50);assert.equal(ids[0],'b-59');assert.equal(ids.at(-1),'b-10');
  assert(p.ui.state.run.actions.length<=2500);
});
test('unfreezing preserves held column position; only Jump to live resets all columns',()=>{
  const p=page();p.ui.renderCallStack();const feed=p.columnFeed('a');feed.scrollTop=100;feed.emit('scroll');
  p.ui.pauseDisplay();const held=p.ui.state.run;p.append();
  p.ui.setView('timeline');p.ui.setView('trajectory');
  assert.equal(p.ui.state.run,held);assert.equal(p.ui.state.displayPaused,true);
  assert.equal(held.actions.length,1);assert.equal(p.sent.length,0);
  p.ui.resumeDisplay();p.ui.renderCallStack();
  assert.equal(p.ui.readingState('a').follow,false);assert.equal(feed.scrollTop,100);
  assert.equal(p.ui.state.selected,null);assert.equal(p.ui.state.run.actions.length,2);
  p.ui.followLatest();assert.equal(feed.scrollTop,0);assert.equal(p.sent.length,0);
});
test('detached columns restore their scroll offset before reading-anchor adjustment',()=>{
  const p=page();p.ui.renderCallStack();const feed=p.columnFeed('a');
  feed.scrollTop=100;feed.emit('scroll');
  p.node('call-stack').children[0].remove();feed.scrollTop=0;
  p.ui.renderCallStack();assert.equal(feed.scrollTop,100);
});
test('expanded call evidence is retained and explicitly refreshed after completion',()=>{
  const p=page(),a=p.ui.state.run.actions[0];a.outcome='running';p.ui.renderCallStack();
  const card=p.feed('a').firstElementChild.querySelector('.call-group-grid').firstElementChild;
  const details=card.querySelector('.tool-details'),body=details.querySelector('.tool-detail-body');
  details.open=true;body.innerHTML='Evidence being read';body.focus();
  p.receive({type:'updates',patches:[{actions:[{id:'old',outcome:'success',result:'Completed output',event_ids:['end']}]}]});
  p.flush();p.ui.renderCallStack();
  assert.equal(body.innerHTML,'Evidence being read');assert.equal(p.document.activeElement,body);
  assert.equal(details.dataset.stale,'true');assert.match(card.querySelector('.call-card-status').innerHTML,/New evidence · Refresh details/);
  const button={dataset:{refreshDetails:'old'},closest:()=>card,remove:()=>{}};
  p.ui.handleClick({target:{closest:()=>button}});
  assert.match(body.innerHTML,/Completed output/);assert.equal(details.dataset.stale,undefined);
  assert.equal(p.document.activeElement,details.querySelector('summary'));
});
test('unchanged selected inspector preserves loaded payload across view switches',()=>{
  const p=page();p.ui.selectAction('old');p.ui.renderInspector();
  p.node('inspector').innerHTML+='Loaded full immutable payload';
  p.ui.setView('timeline');p.ui.renderInspector();p.ui.setView('trajectory');p.ui.renderInspector();
  assert.match(p.node('inspector').innerHTML,/Loaded full immutable payload/);
});
test('an agent outside the recent live window can load its first historical page',async()=>{
  const p=page();p.ui.state.run.actions=[];p.ui.state.run.agents=[{id:'a'}];
  p.ui.state.run.coverage.client_window='Windowed';p.ui.renderCallStack();
  const urls=[];p.setFetch(async url=>{urls.push(url);return {ok:true,json:async()=>({execution:{id:'live'},actions:[{id:'older',agent:'a',kind:'tool',start:0,duration:1,outcome:'success',files:[]}],has_more:false})};});
  await p.ui.olderCalls('a');
  assert.equal(urls.length,1);assert(!urls[0].includes('before_start'));
  assert.equal(p.ui.state.run.actions[0].id,'older');assert.equal(p.ui.readingState('a').follow,false);
  assert.equal(p.feed('a').firstElementChild.dataset.callGroup,'older');
});
test('history pages from another execution cannot enter the displayed model',async()=>{
  const p=page();p.ui.state.run.coverage.client_window='Windowed';p.ui.renderCallStack();
  p.setFetch(async()=>({ok:true,json:async()=>({execution:{id:'live',time_origin:999},actions:[{id:'foreign'}],has_more:false})}));
  await p.ui.olderCalls('a');
  assert(!p.ui.state.run.actions.some(a=>a.id==='foreign'));
  assert.match(p.ui.readingState('a').error,/different or unidentified execution/);
});
test('reconnect snapshots merge duplicate delivery without replacing selected action identity',()=>{
  const p=page();p.ui.selectAction('old');const action=p.ui.state.run.actions[0],selection=p.ui.state.selected;
  const snapshot=structuredClone(p.ui.state.run);snapshot.actions[0].outcome='running';snapshot.actions.push({...snapshot.actions[0]});
  p.receive({type:'snapshot',run:snapshot,resynced:true});p.receive({type:'snapshot',run:snapshot,resynced:true});
  assert.equal(p.ui.state.run.actions.length,1);assert.equal(p.ui.state.run.actions[0],action);
  assert.equal(action.outcome,'success');assert.equal(p.ui.state.selected,selection);
});
test('live control updates never repaint frozen event columns',()=>{
  const p=debuggerPage();p.ui.renderCallStack();p.ui.pauseDisplay();
  const card=p.feed('a').firstElementChild.querySelector('.call-group-grid').firstElementChild;
  p.receive({type:'updates',patches:[{agents:[{id:'a',status:'paused',control_ts:20}],actions:[{id:'old',result:'New evidence'}]}],execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']}});
  p.ui.renderSession();
  assert.equal(p.ui.state.run.actions[0].result,undefined);assert.equal(p.ui.state.liveRun.actions[0].result,'New evidence');
  assert.equal(p.feed('a').firstElementChild.querySelector('.call-group-grid').firstElementChild,card);
  assert.match(p.node('display-state').textContent,/agents may still be running/);
  assert.match(p.node('execution-state').textContent,/1 agent with pause intent recorded/);
  assert.equal(p.node('follow-live').hidden,false);
  assert.match(p.node('call-stack').children[0].querySelector('.agent-column-heading').querySelector('.column-follow').querySelector('span').textContent,/View frozen/);
});

test('inspecting a live call keeps the incoming activity window current',()=>{
  const p=page();p.ui.selectAction('old');p.append();
  assert.equal(p.ui.state.selected.id,'old');
  assert.equal(p.ui.state.follow,true);
  assert.equal(p.ui.state.cursor,2);
  assert.equal(p.ui.windowEnd(),null);
});

test('explicit history browsing stays held across visibility changes',()=>{
  const p=page();p.ui.state.follow=false;p.ui.state.cursor=0.5;p.ui.state.windowEnd=1;
  p.visibility(true);p.append();p.visibility(false);
  assert.equal(p.ui.state.follow,false);
  assert.equal(p.ui.state.cursor,0.5);
  assert.equal(p.ui.windowEnd(),1);
  assert.equal(p.ui.state.unread.size,1);
  p.node('follow-live').onclick();
  assert.equal(p.ui.state.cursor,2);
  assert.equal(p.ui.windowEnd(),null);
  assert.equal(p.ui.state.unread.size,0);
});

test('replay selections still hold history',()=>{
  const p=page(true);p.ui.selectAction('old');p.append();
  assert.equal(p.ui.state.follow,false);
  assert.equal(p.ui.windowEnd(),1);
});

test('live overlay updates preserve both chart scroll positions',()=>{
  const p=page();p.ui.state.trajectoryLayout='overlay';
  p.ui.state.run.agents=[{id:'a',label:'Agent A'}];
  p.ui.state.run.actions[0].name='Tool';
  const region=p.node('content').querySelector('.trajectory-scroll');
  region.scrollLeft=140;region.scrollTop=220;
  // Replacing the chart resets its scrolling element before it is restored.
  Object.defineProperty(p.node('content'),'innerHTML',{set(){region.scrollLeft=0;region.scrollTop=0;}});
  p.append();
  assert.equal(region.scrollLeft,140);
  assert.equal(region.scrollTop,220);
});

test('new activities and calls prepend without replacing expanded cards',()=>{
  const p=page();p.ui.reconcile();
  const list=p.node('activity-list'),old=list.firstElementChild;
  old.open=true;
  p.append();p.ui.reconcile();
  assert.deepEqual(list.children.map(c=>c.dataset.episodeCard),['two','one']);
  assert.equal(list.children[1],old);
  assert.equal(old.open,true);
  p.ui.state.run.actions.push({id:'third',episode:'two',start:2,duration:1,kind:'tool',outcome:'success'});
  p.ui.state.run.episodes[1].actions.push('third');
  p.ui.reconcile();
  assert.deepEqual(list.firstElementChild.querySelector('.episode-actions').children.map(c=>c.dataset.action),['third','new-call']);
});

test('running overlapping calls remain pinned even after newer calls finish',()=>{
  const p=page();
  const calls=[
    {id:'slow',agent:'a',kind:'tool',start:0,duration:10,outcome:'running'},
    {id:'peer',agent:'a',kind:'tool',start:1,duration:1,outcome:'success'},
    {id:'newer',agent:'b',kind:'tool',start:5,duration:1,outcome:'success'},
  ];
  const groups=p.ui.groupCallStack(calls);
  assert.deepEqual(Array.from(groups,g=>g.id),['running:a:tool','newer','peer']);
  assert.deepEqual(Array.from(groups[0].actions,a=>a.id),['slow']);
  p.ui.state.run.actions=calls;p.ui.renderCallStack();
  const stack=p.feed();
  assert.equal(stack.firstElementChild.dataset.callGroup,'running:a:tool');
  assert.equal(stack.firstElementChild.querySelector('.call-group-grid').children.length,1);
  calls[0].outcome='success';calls[0].duration=2;p.ui.renderCallStack();
  assert.equal(p.feed('b').firstElementChild.dataset.callGroup,'newer');
  assert.ok(stack.children.every(group=>!group.className.includes('running-group')));
});

test('sequential calls and separate actors are not labelled concurrent',()=>{
  const p=page();
  const calls=[
    {id:'first',agent:'a',kind:'tool',start:0,duration:1,outcome:'success'},
    {id:'second',agent:'a',kind:'tool',start:1,duration:1,outcome:'success'},
    {id:'other',agent:'b',kind:'tool',start:0,duration:2,outcome:'success'},
    {id:'model',agent:'a',kind:'model',start:0,duration:2,outcome:'success'},
  ];
  assert.equal(p.ui.groupCallStack(calls).length,4);
});

test('old running calls survive the completed history limit',()=>{
  const p=page();
  p.ui.state.run.actions=[{id:'old-running',agent:'a',kind:'tool',start:0,duration:60,outcome:'running'},
    ...Array.from({length:55},(_,i)=>({id:`finished-${i}`,agent:'b',kind:'tool',
      start:i+1,duration:0.5,outcome:'success'}))];
  p.ui.renderCallStack();
  const stack=p.feed();
  assert.equal(stack.children.length,1);
  assert.equal(p.feed('b').children.length,50);
  assert.equal(stack.firstElementChild.dataset.callGroup,'running:a:tool');
  assert.equal(p.feed('b').firstElementChild.dataset.callGroup,'finished-54');
});

test('a finished call leaves its running peers and reuses its card in serial history',()=>{
  const p=page();
  const calls=[
    {id:'a',agent:'agent',kind:'tool',start:0,duration:5,outcome:'running'},
    {id:'b',agent:'agent',kind:'tool',start:1,duration:2,outcome:'running'},
  ];
  p.ui.state.run.actions=calls;p.ui.renderCallStack();
  const stack=p.feed();
  const grid=stack.firstElementChild.querySelector('.call-group-grid');
  const card=grid.children.find(c=>c.dataset.callCard==='b');
  assert.equal(grid.children.length,2);
  calls[1].outcome='success';p.ui.renderCallStack();
  assert.equal(grid.children.length,1);
  assert.equal(grid.firstElementChild.dataset.callCard,'a');
  assert.equal(stack.children[1].dataset.callGroup,'b');
  assert.equal(stack.children[1].querySelector('.call-group-grid').firstElementChild,card);
  assert.equal(card.animations.length,0);
});

test('rapid stream messages render once per bounded UI batch',()=>{
  const p=page();p.append(false);p.append(false);
  assert.equal(p.ui.paints(),0);
  assert.deepEqual(p.delays,[120]);
  p.flush();assert.equal(p.ui.paints(),1);
  p.append(false);p.flush();assert.equal(p.ui.paints(),2);
});

test('switching runs cancels a pending UI batch',()=>{
  const p=page();p.append(false);p.ui.loadRun('live');p.flush();
  assert.equal(p.ui.paints(),0);
});

test('reduced motion skips card animations',()=>{
  const p=page();p.window.matchMedia=()=>({matches:true});
  p.ui.renderCallStack();
  const card=p.feed().firstElementChild.querySelector('.call-group-grid').firstElementChild;
  assert.equal(card.animations.length,0);
});

test('defaults to three stable columns and supports more than three',()=>{
  const p=page();
  p.ui.state.run.agents=['workflow','a','b','c','d'].map(id=>({id}));
  p.ui.renderCallStack();
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','b','c']);
  const original=p.node('call-stack').firstElementChild;
  p.ui.state.run.agents.unshift({id:'new'});p.ui.renderCallStack();
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','b','c']);
  assert.equal(p.node('call-stack').firstElementChild,original);
  p.ui.addAgentColumn('c');p.ui.addAgentColumn('d');
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','b','c','d']);
  assert.equal(p.node('add-agent-column').disabled,false);
  p.ui.removeAgentColumn('b');p.ui.renderCallStack();
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','c','d']);
  p.ui.addAgentColumn('d');
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','c','d']);
});

test('hidden agents show activity counts and each column contains its own calls',()=>{
  const p=page();p.ui.state.run.agents=['a','b','c'].map(id=>({id}));
  p.ui.state.run.actions=[
    {id:'one',agent:'a',kind:'tool',start:0,duration:1,outcome:'running'},
    {id:'two',agent:'b',kind:'tool',start:1,duration:1,outcome:'success'},
    {id:'hidden-running',agent:'c',kind:'tool',start:2,duration:1,outcome:'running'},
    {id:'hidden-failed',agent:'c',kind:'tool',start:3,duration:1,outcome:'failed'},
  ];
  p.ui.renderCallStack();p.ui.removeAgentColumn('c');p.ui.renderCallStack();
  assert.equal(p.feed('a').firstElementChild.dataset.callGroup,'running:a:tool');
  assert.equal(p.feed('b').firstElementChild.dataset.callGroup,'two');
  assert.match(p.node('hidden-agents').innerHTML,/c · 1 retained call failure/);
  p.ui.selectAgentColumn('b');
  assert.equal(p.ui.state.activeColumn,'b');
  const columns=p.node('call-stack').querySelectorAll('[data-agent-column]');
  assert.equal(columns.filter(column=>column.className.includes('active-agent-column')).length,1);
  assert.equal(columns[1].className,'agent-column active-agent-column');
  p.ui.addAgentColumn('c');
  assert.equal(p.ui.state.activeColumn,'c');
  assert.match(p.node('agent-tabs').innerHTML,/1 running · 1 failed/);
});

test('agent filtering temporarily focuses one column without losing chosen columns',()=>{
  const p=page();p.ui.state.run.agents=['a','b'].map(id=>({id}));
  p.ui.renderCallStack();p.ui.state.agent='b';p.ui.renderCallStack();
  assert.deepEqual(p.node('call-stack').children.map(column=>column.dataset.agentColumn),['b']);
  p.ui.state.agent='';p.ui.renderCallStack();
  assert.deepEqual(p.node('call-stack').children.map(column=>column.dataset.agentColumn),['a','b']);
});






function debuggerPage() {
  const p=page();
  p.ui.state.run.agents=[{id:'a',label:'A',status:'waiting_llm'},{id:'b',label:'B',status:'running'}];
  p.ui.state.run.status='running';
  p.ui.state.transport='connected';
  p.ui.state.executionControls={relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']};
  return p;
}

test('display pause freezes mutable evidence while live updates continue without commands',()=>{
  const p=debuggerPage();
  const run=p.ui.state.run;
  Object.assign(run.actions[0],{outcome:'running',result:'Waiting',event_ids:['start'],files:[],name:'Read'});
  run.counters={cpu:[[0,10]]};
  p.ui.renderCallStack();
  const column=p.node('call-stack').firstElementChild;
  p.node('display-toggle').onclick();
  const frozen=p.ui.state.run;
  assert.notEqual(frozen,run);
  assert.equal(p.ui.state.liveRun,run);
  p.receive({type:'updates',patches:[{
    actions:[{id:'old',outcome:'success',result:'Finished',event_ids:['start','end']}],
    counter_samples:{cpu:[[2,50]]},meta:{duration:3},
  }],execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']}});
  p.append();p.visibility(true);p.visibility(false);
  assert.equal(p.sent.length,0);
  assert.equal(p.ui.state.displayPaused,true);
  assert.equal(frozen.actions.length,1);
  assert.equal(frozen.actions[0].result,'Waiting');
  assert.equal(frozen.counters.cpu.length,1);
  assert.equal(frozen.duration,1);
  assert.equal(p.ui.state.liveRun.actions.length,2);
  assert.equal(p.ui.state.liveRun.actions[0].result,'Finished');
  assert.equal(p.ui.state.liveRun.counters.cpu.length,2);
  assert.equal(p.node('call-stack').firstElementChild,column);
  assert.equal(p.ui.paints(),0);
  p.ui.selectAction('old');p.ui.renderInspector();p.ui.renderSession();
  assert.match(p.node('inspector').innerHTML,/Waiting/);
  assert.doesNotMatch(p.node('inspector').innerHTML,/Finished/);
  assert.equal(p.node('display-toggle').textContent,'Unfreeze view');
  assert.match(p.node('display-state').textContent,/2 calls changed/);
  p.node('display-toggle').onclick();
  assert.equal(p.ui.state.run,run);
  assert.equal(p.ui.state.liveRun,null);
  assert.equal(p.ui.state.selected.id,'old');
  assert.equal(p.ui.state.displayPaused,false);
  assert.equal(p.ui.state.cursor,2);
  assert.equal(p.ui.paints(),1);
  assert.equal(p.sent.length,0);
});

test('execution controls use live agent state while the display is frozen',()=>{
  const p=debuggerPage();p.ui.pauseDisplay();
  p.node('execution-all').onclick();
  assert.equal(p.sent[0].command,'pause_all');
  const run=structuredClone(p.ui.state.liveRun);
  for(const agent of run.agents){agent.status='paused';agent.control_ts=10;}
  p.receive({type:'snapshot',run,execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']}});
  assert.equal(p.ui.state.executionPending,null);
  assert.equal(p.ui.state.run.agents[0].status,'waiting_llm');
  assert.equal(p.ui.state.displayPaused,true);
  p.ui.renderSession();
  assert.equal(p.node('execution-all').textContent,'▶ Resume all active agents');
  p.node('execution-all').onclick();
  assert.equal(p.sent[1].command,'resume_all');
  assert.equal(p.ui.state.displayPaused,true);
  p.ui.state.selected={type:'agent',id:'a'};p.ui.renderSession();
  assert.equal(p.agentButton('agent').textContent,'▶ Resume');
});

test('reconnect snapshots are retained behind the paused display and run switching resets it',()=>{
  const p=page();p.ui.pauseDisplay();
  const frozen=p.ui.state.run;
  const live=p.ui.state.liveRun;
  const recovered=structuredClone(live);
  recovered.duration=9;recovered.actions.push({id:'recovered',outcome:'success'});
  p.transport('disconnected');p.transport('connected');
  p.receive({type:'snapshot',run:recovered,resynced:true});
  assert.equal(p.ui.state.run,frozen);
  assert.equal(p.ui.state.liveRun,live);assert.equal(live.actions.filter(a=>a.id==='recovered').length,1);
  p.ui.resumeDisplay();
  assert.equal(p.ui.state.run,live);
  assert.equal(p.ui.state.cursor,9);
  p.ui.pauseDisplay();p.ui.loadRun('live');
  assert.equal(p.ui.state.displayPaused,false);
  assert.equal(p.ui.state.liveRun,null);
  assert.equal(p.ui.state.displayChanges.size,0);
});

test('pausing drains the pending UI batch and duration ticks do not inflate update counts',()=>{
  const p=page();p.append(false);p.ui.pauseDisplay();
  assert.equal(p.ui.paints(),1);
  p.receive({type:'updates',patches:[{meta:{duration:20}}]});p.flush();
  assert.equal(p.ui.state.run.duration,2);
  assert.equal(p.ui.state.liveRun.duration,20);
  assert.equal(p.ui.state.displayChanges.size,0);
  assert.equal(p.ui.paints(),1);
});

test('display pause during replay sends no playback or execution commands',()=>{
  const p=page(true);p.ui.pauseDisplay();p.append();
  assert.equal(p.ui.state.run.actions.length,1);
  assert.equal(p.sent.length,0);
  p.ui.resumeDisplay();
  assert.equal(p.ui.state.run.actions.length,2);
  assert.equal(p.sent.length,0);
});

test('timeline scroll and resize paints keep reading the frozen snapshot',()=>{
  const p=page();p.ui.state.view='timeline';
  const viewer={update(run,actions){this.run=run;this.actions=actions;}};
  p.ui.setTimeline(viewer);p.ui.pauseDisplay();p.append();
  assert.equal(viewer.run,p.ui.state.run);
  assert.equal(viewer.run.duration,1);
  assert.equal(viewer.actions.length,1);
  p.ui.resumeDisplay();
  assert.equal(viewer.run.duration,2);
  assert.equal(viewer.actions.length,2);
});

test('debugger controls follow action selection and send named-agent commands',()=>{
  const p=debuggerPage();p.ui.selectAction('old');p.ui.renderSession();
  assert.equal(p.agentButton('agent').hidden,false);
  assert.equal(p.agentButton('agent').disabled,false);
  p.agentButton('agent').onclick();
  assert.equal(p.sent[0].command,'pause');
  assert.equal(p.sent[0].agname,'a');
  assert.equal(p.ui.state.run.agents[0].status,'waiting_llm');
  assert(p.ui.state.executionPending);
  p.receive({type:'execution_command_result',id:p.sent[0].id,queued:true});
  assert(p.ui.state.executionPending); // Queuing is not authoritative state.
  const run=p.ui.state.run;
  run.agents[0].status='paused';run.agents[0].control_ts=10;
  p.receive({type:'snapshot',run,execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']}});
  p.ui.renderSession();
  assert.equal(p.ui.state.executionPending,null);
  assert.equal(p.agentButton('agent').textContent,'▶ Resume');
  assert.match(p.node('execution-state').textContent,/1 agent with pause intent recorded/);
  p.agentButton('agent').onclick();
  assert.equal(p.sent[1].command,'resume');
  assert.equal(p.sent[1].agname,'a');
});

test('global debugger controls target all agents and expose resume for mixed paused states',()=>{
  const p=debuggerPage();p.ui.renderSession();
  assert.equal(p.agentButton('agent').hidden,false);
  p.node('execution-all').onclick();
  assert.equal(p.sent[0].command,'pause_all');
  assert.equal(p.sent[0].agname,undefined);
  const run=p.ui.state.run;
  for(const agent of run.agents){agent.status='paused';agent.control_ts=10;}
  p.receive({type:'snapshot',run,execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']}});
  p.ui.renderSession();
  p.node('execution-all').onclick();
  assert.equal(p.sent[1].command,'resume_all');
});

test('replay and saved review hide execution controls and replay pause uses only playback',()=>{
  const p=debuggerPage();p.ui.state.replay=true;p.ui.renderSession();
  assert.equal(p.node('execution-all').hidden,true);
  p.ui.pauseResumeExecution(true);
  assert.equal(p.sent.length,0);
  p.ui.state.replayState={playing:true};p.node('display-toggle').onclick();
  assert.equal(p.sent[0].type,'pause');assert.equal(p.sent[0].command,undefined);
  p.ui.state.replay=false;p.ui.state.run.id='saved';p.ui.renderSession();
  assert.equal(p.node('execution-all').hidden,true);
});

test('failed, unconfirmed and disconnected commands retain recorded status and recover controls',()=>{
  const p=debuggerPage();p.ui.pauseResumeExecution(true);
  p.receive({type:'execution_command_result',id:p.sent[0].id,error:'Delivery failed'});
  assert.equal(p.ui.state.executionPending,null);
  assert.equal(p.ui.state.executionMessage,'Delivery failed');
  assert.equal(p.ui.state.run.agents[0].status,'waiting_llm');
  p.ui.pauseResumeExecution(true);p.flush();
  assert.equal(p.ui.state.executionPending,null);
  assert.match(p.ui.state.executionMessage,/No execution confirmation/);
  p.ui.pauseResumeExecution(true);p.transport('disconnected');
  assert.equal(p.ui.state.executionPending,null);
  p.ui.renderSession();assert.equal(p.node('execution-all').disabled,true);
  p.transport('connected');p.ui.renderSession();assert.equal(p.node('execution-all').disabled,false);
  p.ui.state.run.status='completed';p.ui.renderSession();assert.equal(p.node('execution-all').disabled,true);
  p.ui.state.run.status='running';p.ui.state.executionControls.available=false;
  p.ui.renderSession();assert.equal(p.node('execution-all').disabled,true);
});

test('paused agent inspection keeps wall interval distinct from pause-excluded execution time',()=>{
  const p=debuggerPage();
  p.ui.state.run.agents[0].status='paused';
  p.ui.state.selected={type:'agent',id:'a'};p.ui.renderInspector();
  assert.match(p.node('inspector').innerHTML,/pill paused/);
  assert.match(p.ui.agentExecutionControls('a'),/Resume/);
  assert.match(p.ui.actionButton({...p.ui.state.run.actions[0],duration:25,execution_duration:5}),/>25.0s</);
});


test('completed or missing selected agents disable controls and retain an inspectable UI',()=>{
  const p=debuggerPage();
  p.ui.state.selected={type:'agent',id:'a'};
  p.ui.state.executionControls.agents=['b'];
  p.ui.state.run.agents[0].status='completed';p.ui.renderSession();
  assert.equal(p.agentButton('agent').disabled,true);
  p.ui.state.run.agents=p.ui.state.run.agents.filter(a=>a.id!=='a');
  p.ui.renderInspector();p.ui.renderSession();
  assert.match(p.node('inspector').innerHTML,/no longer available/);
  assert.equal(p.agentButton('agent').hidden,true);
});


test('stop targets selected agent and SIGKILL can escalate while stop awaits confirmation',()=>{
  const p=debuggerPage();p.ui.selectAction('old');p.ui.renderSession();
  p.agentButton('stop').onclick();p.ui.renderSession();
  assert.equal(p.sent[0].command,'stop');assert.equal(p.sent[0].agname,'a');
  assert(p.ui.state.executionPending);
  assert.equal(p.agentButton('stop').disabled,true);
  assert.equal(p.agentButton('kill').disabled,false);
  p.receive({type:'execution_command_result',id:p.sent[0].id,queued:true});
  assert(p.ui.state.executionPending);
  assert.equal(p.ui.state.run.agents[0].status,'waiting_llm');
  p.agentButton('kill').onclick();
  assert.equal(p.sent[1].command,'kill');assert.equal(p.sent[1].agname,'a');
  const run=p.ui.state.run;run.agents[0].stop_ts=10;run.agents[0].stop_force=true;
  p.receive({type:'snapshot',run,execution_controls:{relay_id:'fixture-relay',execution:{id:'live'},available:true,agents:['a','b']}});
  assert.equal(p.ui.state.executionPending,null);
  assert.match(p.ui.state.executionMessage,/SIGKILL request recorded; process exit unconfirmed/);
  assert.equal(run.agents[0].status,'waiting_llm');
});

test('stop all uses live execution while display is paused and SIGKILL all keeps all scope',()=>{
  const p=debuggerPage();p.ui.pauseDisplay();p.ui.renderSession();
  p.node('execution-stop-all').onclick();
  assert.equal(p.sent[0].command,'stop_all');assert.equal(p.sent[0].agname,undefined);
  p.node('execution-kill-all').onclick();
  assert.equal(p.sent[1].command,'kill_all');assert.equal(p.sent[1].agname,undefined);
  const run=p.ui.state.liveRun;
  run.agents.forEach(agent=>{agent.status='cancelled';});run.status='cancelled';
  p.receive({type:'snapshot',run,execution_controls:{available:false,agents:[]}});
  assert.equal(p.ui.state.executionPending,null);
  assert.match(p.ui.state.executionMessage,/terminal state recorded/);
});

test('stop controls hide in saved/replay views and disable on disconnect or completion',()=>{
  const p=debuggerPage();p.ui.selectAction('old');
  const ids=['execution-stop-all','execution-kill-all'];
  p.transport('disconnected');p.ui.renderSession();
  ids.forEach(id=>assert.equal(p.node(id).disabled,true));
  p.transport('connected');p.ui.renderSession();
  ids.forEach(id=>assert.equal(p.node(id).disabled,false));
  p.ui.state.replay=true;p.ui.renderSession();
  assert.equal(p.node('all-agent-controls').hidden,true);
  assert.equal(p.ui.agentExecutionControls('a'),'');
  p.ui.state.replay=false;p.ui.state.run.id='saved';p.ui.renderSession();
  assert.equal(p.node('all-agent-controls').hidden,true);
  assert.equal(p.ui.agentExecutionControls('a'),'');
  p.ui.state.run.id='live';p.ui.state.run.status='completed';p.ui.renderSession();
  ids.forEach(id=>assert.equal(p.node(id).disabled,true));
});


test('column controls target their own agent even when another agent is selected',()=>{
  const p=debuggerPage();p.ui.state.selected={type:'agent',id:'a'};
  for(const [attribute,command] of [['data-execution-stop','stop'],['data-execution-kill','kill']]) {
    const kind=attribute.split('-').at(-1);
    const element={dataset:{[`execution${kind[0].toUpperCase()+kind.slice(1)}`]:'b'},hasAttribute:name=>name===attribute};
    p.ui.handleClick({target:{closest:selector=>selector.includes(`[${attribute}]`)?element:null}});
    assert.equal(p.sent.at(-1).command,command);
    assert.equal(p.sent.at(-1).agname,'b');
  }
});


test('column headings contain agent controls while global controls sit under the trajectory heading',()=>{
  const p=debuggerPage();p.ui.renderCallStack();
  for(const column of p.node('call-stack').children) {
    const html=column.querySelector('[data-agent-controls]').innerHTML;
    assert.match(html,new RegExp(`data-execution-agent="${column.dataset.agentColumn}"`));
    assert.match(html,/Agent controls/);assert.match(html,/SIGKILL/);
    assert.doesNotMatch(html,/Stop all agents/);
  }
  const pageHtml=fs.readFileSync(path.join(__dirname,'../../agency/observability/agwebui/static/investigator.html'),'utf8');
  // One mounted control group holds the bulk agent controls and the session actions; the old middle bar is gone.
  const strip=pageHtml.slice(pageHtml.indexOf('<section id="trajectory-controls"'),pageHtml.indexOf('</section>',pageHtml.indexOf('<section id="trajectory-controls"')));
  for(const id of ['all-agent-controls','execution-all','execution-stop-all','execution-kill-all','open-live','start-replay','replay-restart','reconnect-updates','execution-message','control-scope'])assert.match(strip,new RegExp(`id="${id}"`));
  assert.doesNotMatch(pageHtml,/trajectory-session|session-bar|session-actions/);
  const view=p.ui.trajectoryView();
  assert.match(view,/<h2>Execution Trajectory<\/h2>[\s\S]*?<div id="trajectory-controls-slot"><\/div>[\s\S]*id="call-stack"/);
  p.ui.setContent(view);
  assert.equal(p.node('trajectory-controls').parentElement,p.node('trajectory-controls-slot'));
  // Every replacement of the view markup goes through setContent, so the mounted controls are never destroyed with it.
  const script=fs.readFileSync(path.join(__dirname,'../../agency/observability/agwebui/static/investigator.js'),'utf8');
  const writes=[...script.matchAll(/\$\('content'\)\.innerHTML\s*=\s*([^;]{0,24})/g)].map(match=>match[1]);
  assert.deepEqual([...new Set(writes)],['html','detailedProfilerView()']);
  assert.doesNotMatch(pageHtml,/id="execution-agent"|id="execution-stop-agent"|id="execution-kill-agent"/);
});

test('column controls update from live state while the displayed evidence stays frozen',()=>{
  const p=debuggerPage();p.ui.pauseDisplay();
  const group=p.node('column-controls');group.dataset.agentControls='b';
  p.ui.state.liveRun.agents[1].status='paused';p.ui.renderSession();
  assert.match(group.innerHTML,/Resume/);
  assert.equal(p.ui.state.run.agents[1].status,'running');
  p.agentButton('agent','b').onclick();
  assert.equal(p.sent[0].command,'resume');assert.equal(p.sent[0].agname,'b');
  p.ui.renderSession();assert.equal(group.querySelector('[data-execution-stop]').disabled,true);
  assert.equal(group.querySelector('[data-execution-kill]').disabled,false);
});

test('pause closes control menus and updates controls without replacing their focus targets',()=>{
  const p=debuggerPage(),group=p.node('column-controls');group.dataset.agentControls='a';
  p.ui.renderSession();
  const button=group.querySelector('[data-execution-agent]');button.focus();
  const menu=p.node('open-control-menu');menu.controlMenu=true;menu.open=true;
  p.ui.pauseResumeExecution(false,'a');p.ui.renderSession();
  assert.equal(menu.open,false);assert.equal(button.disabled,true);
  assert.equal(group.querySelector('[data-execution-agent]'),button);assert.equal(p.document.activeElement,button);
  p.receive({type:'execution_command_result',id:p.sent[0].id,queued:false,error:'Fixture rejection'});
  p.ui.state.run.agents[0].status='paused';p.ui.renderSession();
  assert.equal(button.textContent,'▶ Resume');assert.equal(button.disabled,false);
  assert.equal(group.querySelector('[data-execution-agent]'),button);
  assert.equal(p.window.scrollY,0);
});


test('pending pause blocks duplicate submissions and remains requested until recorded state',()=>{
  const p=debuggerPage();p.ui.pauseResumeExecution(false,'a');p.ui.pauseResumeExecution(false,'a');
  assert.equal(p.sent.length,1);assert.equal(p.sent[0].relay_id,'fixture-relay');assert.equal(p.sent[0].execution.id,'live');
  assert.match(p.ui.agentExecutionControls('a'),/Pausing/);
  p.receive({type:'execution_command_result',id:p.sent[0].id,queued:true,targets:['a']});
  assert(p.ui.state.executionPending);assert.equal(p.ui.state.run.agents[0].status,'waiting_llm');
});

test('partial bulk receipts retain specific failed targets and recorded accepted targets',()=>{
  const p=debuggerPage();p.ui.pauseResumeExecution(true);
  p.receive({type:'execution_command_result',id:p.sent[0].id,state:'partial',targets:[{agname:'a',state:'accepted'},{agname:'b',state:'failed',detail:'daemon unavailable'}]});
  assert(p.ui.state.executionPending);assert.match(p.ui.state.executionMessage,/Some targets failed/);
  const run=p.ui.state.run;run.agents[0].status='paused';run.agents[0].control_ts=10;
  p.receive({type:'snapshot',run,execution_controls:{available:true,agents:['a','b']}});
  assert.equal(p.ui.state.executionPending,null);assert.match(p.ui.state.executionMessage,/daemon unavailable/);assert.match(p.ui.state.executionMessage,/paused state recorded/);
});

test('destructive confirmations name the execution, targets and agent count; cancel sends nothing',()=>{
  const p=debuggerPage();p.setAutoConfirm(false);
  p.ui.stopExecution(true,true);assert.equal(p.sent.length,0);assert.equal(p.confirmations[0].force,true);assert.equal(p.confirmations[0].targets.length,2);assert.match(p.confirmations[0].targets[0],/A.*a/);
  p.ui.pauseResumeExecution(false,'a');assert.equal(p.confirmations.length,1);assert.equal(p.sent.length,1);
});

test('a replacement relay cannot confirm an older control request',()=>{
  const p=debuggerPage();p.ui.pauseResumeExecution(false,'a');
  p.receive({type:'snapshot',run:p.ui.state.run,execution_controls:{available:true,agents:['a','b'],relay_id:'replacement'}});
  assert.equal(p.ui.state.executionPending,null);assert.match(p.ui.state.executionMessage,/execution changed/);assert.equal(p.sent.length,1);
});

function interactionPage(){
  const p=debuggerPage();p.ui.state.executionControls.capabilities={a:{queue_message:true,redirect:['run-active'],run:[{id:'echo',label:'Echo',input_schema:'{"label":"string"}'}]}};
  p.ui.openInteraction('a');p.node('interaction-operation').value='queue_message';p.node('interaction-message').value='first line\nsecond line';return p;
}

test('interaction queues exact multiline input once with an execution fence',()=>{
  const p=interactionPage();p.ui.submitInteraction();p.ui.submitInteraction();
  assert.equal(p.sent.length,1);assert.equal(p.sent[0].command,'queue_message');assert.equal(p.sent[0].message,'first line\nsecond line');assert.equal(p.sent[0].relay_id,'fixture-relay');
  p.receive({type:'execution_command_result',id:p.sent[0].id,queued:true,targets:['a']});
  assert.equal(p.ui.state.interaction.pending,true);assert.match(p.ui.state.interaction.status,/dispatch not confirmed/);
  p.receive({type:'snapshot',run:p.ui.state.run,execution_controls:{...p.ui.state.executionControls,receipts:[{id:p.sent[0].id,state:'accepted',targets:[{agname:'a',state:'queued',scope:'future_retained_context'}]}]}});
  assert.equal(p.ui.state.interaction.pending,false);assert.match(p.ui.state.interaction.status,/queued/);assert.doesNotMatch(p.ui.state.interaction.status,/followed|success/);
});

test('recorded and replay interaction panels remain read-only',()=>{
  const p=interactionPage();p.ui.state.replay=true;p.ui.openInteraction('a');p.ui.submitInteraction();
  assert.equal(p.sent.length,0);assert.match(p.node('inspector').innerHTML,/read-only/);
  p.ui.state.replay=false;p.ui.state.run.id='saved';p.ui.openInteraction('a');p.ui.submitInteraction();assert.equal(p.sent.length,0);
});

test('interaction target changes and disconnect after submission never cause a retry',()=>{
  const p=interactionPage();p.ui.submitInteraction();p.transport('disconnected');
  assert.match(p.ui.state.interaction.status,/outcome unknown/);p.transport('connected');p.ui.submitInteraction();assert.equal(p.sent.length,1);
  p.ui.state.executionControls.relay_id='new-runtime';p.ui.renderInteractionStatus();p.ui.submitInteraction();
  assert.match(p.ui.state.interaction.invalid,/execution changed/);assert.equal(p.sent.length,1);
});

test('task selection does not submit; invalid JSON and unregistered tasks cannot run',()=>{
  const p=interactionPage();p.node('interaction-operation').value='run';p.node('interaction-task').value='echo';p.ui.renderInteractionOperation();assert.equal(p.sent.length,0);
  p.node('interaction-input').value='[1,2]';p.ui.submitInteraction();assert.equal(p.sent.length,0);assert.match(p.ui.state.interaction.status,/JSON object/);
  p.node('interaction-input').value='{"label":"hello"}';p.node('interaction-task').value='not-registered';p.ui.submitInteraction();assert.equal(p.sent.length,0);
  p.node('interaction-task').value='echo';p.ui.submitInteraction();assert.equal(p.sent[0].command,'run');assert.deepEqual(JSON.parse(JSON.stringify(p.sent[0].input)),{label:'hello'});
});

test('redirect uses only an active explicit invocation and becomes unavailable when it ends',()=>{
  const p=interactionPage();p.node('interaction-operation').value='redirect';p.node('interaction-invocation').value='run-active';p.ui.submitInteraction();
  assert.equal(p.sent[0].command,'redirect');assert.equal(p.sent[0].invocation,'run-active');
  p.ui.closeInteraction();p.ui.openInteraction('a');p.node('interaction-operation').value='redirect';p.node('interaction-invocation').value='run-ended';p.ui.submitInteraction();assert.equal(p.sent.length,1);
  assert.match(p.ui.state.interaction.status,/no longer redirectable/);
});

test('frozen composer targets the current live execution and retains frozen inspection',()=>{
  const p=interactionPage();p.ui.pauseDisplay();const frozen=p.ui.state.run;p.ui.renderInteractionStatus();
  assert.match(p.node('interaction-target-status').textContent,/View frozen.*CURRENT LIVE/);p.ui.submitInteraction();assert.equal(p.sent.length,1);assert.equal(p.ui.state.run,frozen);assert.equal(p.ui.state.displayPaused,true);
});

test('filtered roster return and unpin preserve actual pinned order',()=>{
  const p=page();p.ui.state.run.agents=[{id:'a'},{id:'b'},{id:'c'}];p.ui.visibleAgentColumns();p.ui.state.agent='a';
  p.ui.selectAgentColumn('b');assert.equal(p.ui.state.agent,'');assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','b','c']);
  p.ui.state.agent='b';p.ui.removeAgentColumn('b');assert.equal(p.ui.state.agent,'');assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','c']);
});


test('native disclosures do not bubble into the content view bookkeeping attribute',()=>{
  const p=page();p.ui.state.view='trajectory';
  // A native summary's nearest data-view is the workspace bookkeeping node,
  // not a view switch. Clicking it must leave the view and selected event alone.
  p.ui.state.selected={execution:'a',id:'selected',type:'action',run:'a'};
  p.ui.handleClick({target:{closest:selector=>selector.split(',').includes('[data-view]')?{dataset:{view:'timeline'}}:null}});
  assert.equal(p.ui.state.view,'trajectory');assert.equal(p.ui.state.selected.id,'selected');
});

test('full captured model input skips start events that have no prompt payload',async()=>{
  const p=page(),urls=[];
  p.setFetch(async url=>{urls.push(url);return {ok:true,json:async()=>({payload:url.endsWith('start')?{}:{context:{blocks:[{source:'Original task',text:'complete captured prompt'}]}}})};});
  const text=await p.ui.originalPayload({kind:'model',input_truncated:true,event_ids:['start','complete']},'input',{id:'live'});
  assert.match(text,/complete captured prompt/);assert.equal(urls.length,2);
});

test('full captured model output keeps explicit tool requests alongside emitted text',async()=>{
  const p=page();p.setFetch(async()=>({ok:true,json:async()=>({payload:{result:'emitted preview',model_response:[{type:'text',text:'emitted preview'},{type:'tool_use',id:'real-tool-id',name:'Bash',input:{command:'meaningful command'}}]}})}));
  const text=await p.ui.originalPayload({kind:'model',output_truncated:true,event_ids:['complete']},'output',{id:'live'});
  assert.match(text,/emitted preview/);assert.match(text,/meaningful command/);assert.match(text,/real-tool-id/);
});

test('later context commitment and failure update composer after publication acknowledgement',()=>{
  const p=interactionPage();p.node('interaction-message').value='retained';p.ui.submitInteraction();const id=p.sent[0].id;
  const target={agname:'a',state:'queued',scope:'future_retained_context',detail:'Publication recorded'};
  p.ui.receiveCommandResult({id,targets:[target]});assert.match(p.ui.state.interaction.status,/queued/);
  p.ui.receiveCommandResult({id,targets:[{...target,context_state:'committed'}]});assert.match(p.ui.state.interaction.status,/Context committed/);assert.match(p.ui.state.interaction.status,/consumption unconfirmed/);
  p.ui.receiveCommandResult({id,targets:[{...target,context_state:'failed',detail:'predecessor cancelled'}]});assert.match(p.ui.state.interaction.status,/Context failed/);assert.match(p.ui.state.interaction.status,/predecessor cancelled/);
  assert.equal(p.sent.length,1);
});

test('composer keeps multiline draft, operation, form node and keyboard focus through rerenders',()=>{
  const p=interactionPage(),input=p.node('interaction-message');input.value='first line\nsecond line';input.focus();p.node('interaction-operation').value='run';p.node('inspector').innerHTML='preserved composer';
  p.ui.renderComposer();p.ui.setView('timeline');p.ui.renderInspector();
  assert.equal(p.node('inspector').innerHTML,'preserved composer');assert.equal(input.value,'first line\nsecond line');assert.equal(p.node('interaction-operation').value,'run');assert.equal(p.document.activeElement,input);
});

test('six held columns retain their reading pages through high-volume updates',()=>{
  const p=page();p.ui.state.run={id:'live',actions:[],agents:[],episodes:[],intervals:[]};p.ui.state.agentColumns=[];
  for(let n=0;n<6;n++){
    const agent=`agent-${n}`;p.ui.state.agentColumns.push(agent);p.ui.state.run.agents.push({id:agent});
    const r=p.ui.readingState(agent);r.follow=false;
    r.ids=Array.from({length:102},(_,i)=>`${agent}-held-${i}`);r.activeIds=Array.from({length:102},(_,i)=>`${agent}-active-${i}`);
    for(const id of [...r.ids,...r.activeIds])p.ui.state.run.actions.push({id,agent,start:0,outcome:id.includes('active')?'running':'success'});
    for(let i=0;i<500;i++)p.ui.state.run.actions.push({id:`${agent}-new-${i}`,agent,start:1+i,outcome:'success'});
  }
  p.receive({type:'snapshot',run:p.ui.state.run});
  const held=p.ui.readingPins();assert(held.length>=1500);
  const ids=new Set(p.ui.state.run.actions.map(a=>a.id));for(const id of held)assert(ids.has(id),id);
  assert(p.ui.state.run.actions.length<=4200);
});

test('observation-capable relay waits for applied facts despite a recorded pause intent',()=>{
  const p=debuggerPage();p.ui.state.executionControls.control_observations=true;p.ui.pauseResumeExecution(false,'a');
  const run=p.ui.state.run;run.agents[0].status='paused';run.agents[0].control_ts=10;
  p.receive({type:'snapshot',run,execution_controls:{...p.ui.state.executionControls}});
  assert(p.ui.state.executionPending);
  const receipt={type:'execution_command_result',id:p.sent[0].id,state:'accepted',targets:[{agname:'a',state:'accepted',application:{state:'pending'}}]};
  p.receive(receipt);p.receive({type:'snapshot',run,execution_controls:{...p.ui.state.executionControls}});
  assert(p.ui.state.executionPending);assert.match(p.ui.agentExecutionControls('a'),/Pausing/);
  p.receive({...receipt,targets:[{agname:'a',state:'accepted',application:{state:'confirmed',scope:'local_harness_tasks',observations:[{state:'confirmed',local_state:'suspended',request_id:'A',observed_at:100}]}}]});
  p.receive({type:'snapshot',run,execution_controls:{...p.ui.state.executionControls}});
  assert.equal(p.ui.state.executionPending,null);assert.match(p.ui.state.executionMessage,/Local suspension: confirmed/);assert.match(p.ui.state.executionMessage,/invocation A/);
  assert.match(p.ui.state.executionMessage,/Remote operations and cleanup are not established/);
});

test('confirmation cannot submit after disconnect, replacement relay, changed bulk targets or replay',()=>{
  for(const change of [p=>p.transport('disconnected'),p=>p.ui.state.executionControls.relay_id='new',p=>p.ui.state.executionControls.agents=['a'],p=>p.ui.state.replay=true]){
    const p=debuggerPage();p.setAutoConfirm(false);p.ui.stopExecution(true,true);assert.equal(p.sent.length,0);
    change(p);assert(p.confirmations[0].validate());p.confirmations[0].onConfirm();assert.equal(p.sent.length,0);
  }
});

test('frozen confirmation states the live target and partial observed outcomes stay explicit',()=>{
  const p=debuggerPage();p.ui.pauseDisplay();p.setAutoConfirm(false);p.ui.stopExecution(true,true);
  assert.equal(p.confirmations[0].frozen,true);p.confirmations[0].onConfirm();
  const receipt={type:'execution_command_result',id:p.sent[0].id,state:'accepted',targets:[{agname:'a',state:'accepted',application:{state:'confirmed'}},{agname:'b',state:'accepted',application:{state:'unknown',detail:'connection lost'}}]};
  p.receive(receipt);p.receive({type:'snapshot',run:p.ui.state.liveRun,execution_controls:{...p.ui.state.executionControls}});
  assert.equal(p.ui.state.executionPending,null);assert.match(p.ui.state.executionMessage,/application confirmed/);assert.match(p.ui.state.executionMessage,/connection lost/);
  assert.equal(p.ui.state.displayPaused,true);assert.equal(p.sent.length,1);
});


test('Profiler hides the summary dashboard and trajectory restores it',()=>{
  const p=page(false,'profiler');
  p.ui.renderMetricsDashboard();
  assert.equal(p.node('summary-strip').hidden,true);
  assert.equal(p.node('metrics').hidden,true);
  p.ui.state.run=null;p.ui.setView('trajectory');p.ui.renderMetricsDashboard();
  assert.equal(p.node('summary-strip').hidden,false);
  assert.equal(p.node('metrics').hidden,false);
});

test('time cursor starts with the wall clock, not with setup before the first agent execution',()=>{
  const p=page(),run=p.ui.state.run;
  Object.assign(run,{duration:30,time_origin:100,execution_started_at:null});
  assert.deepEqual({...p.ui.wallClockRange()},{start:0,end:0,pending:true});
  run.execution_started_at=120;
  const started=p.ui.wallClockRange();
  assert.equal(started.start,20);assert.equal(started.end,30);assert.equal(started.pending,undefined);
  delete run.execution_started_at;
  assert.equal(p.ui.wallClockRange().start,0);assert.equal(p.ui.wallClockRange().end,30);
});

test('projected clock never retreats when reports arrive with uneven delay',()=>{
  const p=page(),{followClock,projectClock}=p.ui;
  // The server clock runs 1:1; each report is delivered 0–600 ms late, in order.
  const delays=[50,600,20,400,0,550,30,300,10,500];
  let clock=null,last=-Infinity,now=0;
  for(const [index,delay] of delays.entries()){
    const sent=1000+index*300;now=Math.max(now+1,sent+delay);
    for(const at of [now-1,now]){
      if(at===now)clock=followClock(clock,sent/1000,1,now);
      if(!clock)continue;
      const shown=projectClock(clock,at);
      assert.ok(shown>=last-1e-9,`retreated at report ${index}`);last=shown;
    }
  }
  // It settles on the least-delayed report rather than drifting ahead of the server.
  assert.ok(projectClock(clock,now)<=now/1000+1e-9&&projectClock(clock,now)>now/1000-.05);
  // A real jump (seek or pause) is taken as reported, and silence holds the value.
  assert.equal(projectClock(followClock(clock,1,1,now),now),1);
  assert.equal(projectClock(followClock(clock,2,0,now),now+9000),2);
  assert.equal(projectClock(clock,now+60000),projectClock(clock,now+3000));
});

// Agent-scoped metrics: compact column status and column-to-dashboard selection.
const usage=(p,agent)=>p.node('call-stack').children.find(c=>c.dataset.agentColumn===agent).querySelector('.agent-column-heading').querySelector('.column-usage');
test('column status shows live per-agent sandbox CPU and compact tokens beside the existing in-flight count',()=>{
  const p=page(),run=p.ui.state.run;
  p.receive({type:'patch',patches:[{agents:[{id:'a',label:'Agent A',status:'running',sandbox:'sandbox_a_0000'},{id:'b',label:'Agent B',status:'running'}],
    actions:[{id:'m',agent:'a',episode:'one',start:1,duration:1,kind:'model',outcome:'success',tokens:8000,output_tokens:200,files:[]},{id:'bt',agent:'b',episode:'b1',start:1,duration:1,kind:'tool',outcome:'running',files:[]}],
    episodes:[{id:'b1',agent:'b',actions:['bt'],status:'running'}],
    counter_samples:{'workload:cpu_pct':[[2,640]],'host:cpu_pct':[[2,900]],'cg:sandbox_a_0000:cpu_pct':[[2,12.4]]},meta:{duration:2}}]});
  p.flush();p.ui.reconcile();
  assert.equal(usage(p,'a').textContent,'CPU 12% · 8.2K tok');
  assert.match(usage(p,'a').title,/sandbox container.*100% = one core.*8,000 input \+ 200 output/);
  // Missing measurements are dashes; the workload or host value is never shown as an agent's CPU.
  assert.equal(usage(p,'b').textContent,'CPU — · — tok');assert.match(usage(p,'b').title,/no sandbox measurement recorded.*no usage reported/);
  const heading=p.node('call-stack').children.find(c=>c.dataset.agentColumn==='b').querySelector('.agent-column-heading');
  assert.equal(heading.querySelector('.column-call-count').textContent,'1 in flight');
  // Values follow the existing stream; execution-wide counters win over the retained window.
  p.receive({type:'patch',patches:[{counter_samples:{'cg:sandbox_a_0000:cpu_pct':[[3,3.44]]},meta:{duration:3,metrics_by_agent:{a:{models:90,input:1390000,output:10000,input_samples:90,output_samples:90}}}}]});
  p.flush();p.ui.reconcile();
  assert.equal(usage(p,'a').textContent,'CPU 3.4% · 1.4M tok');
  // A container that stopped reporting has no current utilization.
  p.receive({type:'patch',patches:[{meta:{duration:60}}]});p.flush();p.ui.reconcile();
  assert.equal(usage(p,'a').textContent,'CPU — · 1.4M tok');assert.match(usage(p,'a').title,/no current container sample/);
  assert.equal(run.agents.length>=2,true);
});
test('selecting a column header scopes the dashboard to that agent without changing columns or controls',()=>{
  const p=page();
  p.receive({type:'patch',patches:[{agents:[{id:'a',label:'Agent A',status:'running'},{id:'b',label:'Agent B',status:'running'}],
    actions:[{id:'bt',agent:'b',episode:'b1',start:1,duration:1,kind:'tool',outcome:'running',files:[]}],episodes:[{id:'b1',agent:'b',actions:['bt'],status:'running'}],meta:{duration:2}}]});
  p.flush();p.ui.reconcile();
  const columns=()=>p.node('call-stack').querySelectorAll('[data-agent-column]').map(c=>c.dataset.agentColumn);
  const before=columns(),follow=p.ui.state.follow,renders=p.summaryRenders();
  const header=agent=>({target:{closest:selector=>selector==='[data-scope-column]'?{dataset:{scopeColumn:agent}}:null}});
  p.ui.handleClick(header('b'));
  assert.deepEqual(p.metricScope(),['b']);assert.equal(p.summaryRenders(),renders+1);
  assert.deepEqual(columns(),before);assert.equal(p.ui.state.follow,follow);assert.equal(p.ui.state.agent,'');
  p.ui.renderCallStack();
  assert.match(p.node('call-stack').children.find(c=>c.dataset.agentColumn==='b').className,/metric-scoped-column/);
  assert.doesNotMatch(p.node('call-stack').children.find(c=>c.dataset.agentColumn==='a').className,/metric-scoped-column/);
  // Another header replaces the single selection; the same header again returns to all agents.
  p.ui.handleClick(header('a'));assert.deepEqual(p.metricScope(),['a']);
  p.ui.handleClick(header('a'));assert.equal(p.metricScope(),null);
  // Controls inside the header keep their own action and never change the scope.
  const live=p.ui.readingState('a');live.follow=false;
  p.ui.handleClick({target:{closest:selector=>selector==='[data-scope-column]'?{dataset:{scopeColumn:'a'}}:selector.startsWith('button,')||selector.includes('[data-column-live]')?{dataset:{columnLive:'a'},hasAttribute:()=>false}:null}});
  assert.equal(p.metricScope(),null);assert.equal(live.follow,true);assert.deepEqual(columns(),before);
  // Live updates keep flowing into both columns while one agent is selected.
  p.ui.handleClick(header('b'));p.append();p.ui.reconcile();
  assert.deepEqual(p.metricScope(),['b']);assert.equal(columns().length,2);
  assert.equal(p.feed('a').children.some(card=>card.dataset.callCard==='new-call'||card.dataset.callGroup==='new-call'),true);
});

// UI reorganisation: three views, relocated controls, readable payloads, typed cards.
const pageSource=()=>fs.readFileSync(path.join(__dirname,'../../agency/observability/agwebui/static/investigator.html'),'utf8');
test('navigation is Execution Trajectory, Timeline, Profiler and the removed views are gone',()=>{
  const p=page();
  assert.equal(JSON.stringify(p.ui.views),JSON.stringify([['trajectory','Execution Trajectory'],['timeline','Timeline'],['profiler','Profiler']]));
  assert.equal(p.ui.state.view,'trajectory');
  const selection=(p.ui.selectAction('old'),p.ui.state.selected),run=p.ui.state.run;
  for(const view of ['timeline','profiler','trajectory']){p.ui.setView(view);assert.equal(p.ui.state.view,view);assert.equal(p.ui.state.run,run);assert.equal(p.ui.state.selected,selection);}
  // Profiler categories still deep-link into the Profiler view.
  p.ui.setView('llm');assert.equal(p.ui.state.view,'profiler');assert.equal(p.ui.state.profilerCategory,'llm');
  for(const removed of ['agents','compare','context','resources']){p.ui.setView(removed);assert.equal(p.ui.state.view,'profiler');}
  const script=fs.readFileSync(path.join(__dirname,'../../agency/observability/agwebui/static/investigator.js'),'utf8');
  assert.doesNotMatch(script,/Tandem|Compare runs|Context & information|Resources & waits|compareView|contextView|resourcesView|agentsView/);
  assert.doesNotMatch(pageSource(),/id="execution-views"/);
});
test('the global agent dropdown is gone while roster focus stays reversible',()=>{
  const p=page(),html=pageSource();
  assert.doesNotMatch(html,/agent-select/);
  p.ui.state.run.agents=[{id:'a',label:'Agent A'},{id:'b',label:'Agent B'}];
  // Focusing an agent from the roster still filters the trajectory to it.
  p.ui.handleClick({target:{closest:selector=>selector.includes('[data-agent]')?{dataset:{agent:'b'},hasAttribute:()=>false}:null}});
  assert.equal(p.ui.state.agent,'b');assert.deepEqual(Array.from(p.ui.visibleAgentColumns()),['b']);
  p.ui.renderAgentColumnControls(p.ui.visibleAgentColumns());
  assert.equal(p.node('agent-focus-clear').hidden,false);assert.match(p.node('agent-focus-clear').textContent,/Showing only Agent B/);
  p.ui.handleClick({target:{closest:selector=>selector.includes('[data-clear-agent]')?{dataset:{},hasAttribute:name=>name==='data-clear-agent'}:null}});
  assert.equal(p.ui.state.agent,'');
  p.ui.renderAgentColumnControls(p.ui.visibleAgentColumns());assert.equal(p.node('agent-focus-clear').hidden,true);
});
test('Jump to live sits beside Freeze view and column headings have no reorder buttons',()=>{
  const html=pageSource();
  assert.match(html,/<div class="display-controls"[^>]*><button id="display-toggle" class="display-control"[^>]*>Freeze view<\/button><button id="follow-live" class="display-control"[^>]*>Jump to live<\/button><\/div>/);
  const p=page();p.ui.renderCallStack();
  const column=p.node('call-stack').children[0];
  assert.doesNotMatch(column.innerHTML,/data-column-order|Move .* column/);
  assert.match(column.innerHTML,/data-remove-column="a"/);assert.match(column.innerHTML,/data-agent-controls="a"/);
  // The two display controls keep their separate behaviour.
  p.ui.pauseDisplay();assert.equal(p.ui.state.displayPaused,true);
  p.ui.followLatest();assert.equal(p.ui.state.displayPaused,false);assert.equal(p.ui.state.follow,true);
});
test('tool calls and model requests carry a type class and a text label',()=>{
  const p=page();
  const tool=p.ui.actionButton({id:'t',agent:'a',kind:'tool',name:'bash',command:'ls',start:0,duration:1,outcome:'success'});
  const model=p.ui.actionButton({id:'m',agent:'a',kind:'model',name:'chat',model:'model-x',start:0,duration:1,outcome:'success'});
  assert.match(tool,/class="action-row type-tool /);assert.match(tool,/<span class="call-type">Tool<\/span>bash/);
  assert.match(model,/class="action-row type-model /);assert.match(model,/<span class="call-type">Model request<\/span>/);
  // Status indicators keep their own meaning beside the type.
  assert.match(tool,/class="status-icon success"/);
  p.ui.state.run.actions.push({id:'m1',agent:'a',episode:'one',kind:'model',name:'chat',start:1,duration:1,outcome:'success',files:[]});p.ui.renderCallStack();
  const cards=p.node('call-stack').querySelectorAll('[data-call-card]');
  assert.equal(cards.find(card=>card.dataset.callCard==='old').className,'call-card type-tool');
  assert.equal(cards.find(card=>card.dataset.callCard==='m1').className,'call-card type-model');
});
test('JSON payloads are pretty by default with a per-payload raw toggle; other text is untouched',()=>{
  const p=page(),recorded='{"path":"a.py","big":12345678901234567890,"nested":{"ok":true,"items":[1,2.50]}}';
  const html=p.ui.payloadSection({id:'call',kind:'tool',arguments:recorded,result:'plain <b>text</b> output'},'input');
  assert.match(html,/data-payload-view="pretty"/);
  assert.match(html,/<span class="payload-format" role="group" aria-label="Input format" ><button data-payload-format="pretty" aria-pressed="true">Pretty<\/button><button data-payload-format="raw" aria-pressed="false">Raw<\/button>/);
  // Pretty: indented and highlighted, values spelled exactly as recorded.
  assert.match(html,/payload-pretty">\{\n  <span class="json-key">&quot;path&quot;:<\/span> <span class="json-string">&quot;a\.py&quot;<\/span>/);
  assert.match(html,/<span class="json-number">12345678901234567890<\/span>/);assert.match(html,/<span class="json-number">2\.50<\/span>/);
  // Raw: the recorded text itself.
  assert.ok(html.includes(`<pre class="payload-text payload-raw">${recorded.replace(/"/g,'&quot;')}</pre>`));
  assert.match(html,/data-copy-payload="call" data-payload-kind="input"/);
  const plain=p.ui.payloadSection({id:'call',kind:'tool',arguments:recorded,result:'plain <b>text</b> output'},'output');
  assert.match(plain,/role="group" aria-label="Output format" hidden>/);assert.doesNotMatch(plain,/payload-pretty/);
  assert.match(plain,/<pre class="payload-text">plain &lt;b&gt;text&lt;\/b&gt; output<\/pre>/);assert.match(plain,/data-copy-payload="call" data-payload-kind="output"/);
  // Invalid or clipped JSON is never reformatted.
  assert.doesNotMatch(p.ui.payloadSection({id:'x',kind:'tool',arguments:'{"cut": "off'},'input'),/payload-pretty/);
});
function payloadFixture(view='pretty'){
  const pre={focus(){}},format={hidden:true},options=['pretty','raw'].map(payloadFormat=>({dataset:{payloadFormat},setAttribute(name,value){this[name]=value;}}));
  const section={dataset:{payloadView:view,payloadKey:'old:input'},classList:{add(name){section.loaded=name;}},querySelector:selector=>selector==='.payload-format'?format:viewer,querySelectorAll:()=>options};
  const viewer={closest:()=>section,querySelector:()=>pre,set innerHTML(html){this.html=html;}};
  return {pre,format,options,section,viewer};
}
test('the full payload viewer switches between pretty and raw without changing the text and pages on line breaks',()=>{
  const p=page(),f=payloadFixture(),recorded='{"a":1,"b":{"c":"<tag>"}}';
  p.ui.renderFullPayload(f.viewer,recorded);
  assert.equal(f.format.hidden,false);assert.match(f.viewer.html,/Formatted JSON/);
  assert.match(f.pre.innerHTML,/<span class="json-key">&quot;a&quot;:<\/span> <span class="json-number">1<\/span>/);assert.match(f.pre.innerHTML,/&lt;tag&gt;/);
  // Toggling to Raw re-renders the same stored text, unformatted.
  const raw={target:{closest:selector=>selector==='[data-payload-format]'?Object.assign(f.options[1],{closest:()=>f.section}):null}};
  p.ui.handlePayloadClick(raw);
  assert.equal(f.section.dataset.payloadView,'raw');assert.equal(f.options[1]['aria-pressed'],'true');assert.equal(f.options[0]['aria-pressed'],'false');
  assert.equal(f.pre.textContent,recorded);assert.match(f.viewer.html,/Captured text · characters 1–25 of 25/);
  assert.equal(p.ui.fullPayloads.get(f.viewer).text,recorded);
  // The choice is per payload and survives the section being rebuilt.
  assert.match(p.ui.payloadSection({id:'old',kind:'tool',arguments:recorded},'input'),/data-payload-view="raw"/);
  assert.match(p.ui.payloadSection({id:'old',kind:'tool',result:recorded},'output'),/data-payload-view="pretty"/);
  // Non-JSON text never offers the toggle.
  const text=payloadFixture();p.ui.renderFullPayload(text.viewer,'just output');assert.equal(text.format.hidden,true);assert.equal(text.pre.textContent,'just output');
  // Long pretty JSON pages end on a line break and can go back.
  const long=payloadFixture(),big=JSON.stringify({rows:Array.from({length:3000},(_,i)=>({index:i,label:'row '+i}))});
  p.ui.renderFullPayload(long.viewer,big);
  const first=p.ui.fullPayloads.get(long.viewer);assert.ok(first.next<=20000);assert.match(long.viewer.html,/data-payload-page="next" >/);
  const pageButton=direction=>({dataset:{payloadPage:direction},closest:()=>long.viewer});
  p.ui.pagePayload(pageButton('next'));
  assert.deepEqual(Array.from(p.ui.fullPayloads.get(long.viewer).starts),[0,first.next]);assert.doesNotMatch(long.pre.innerHTML,/^[^\s<]*"/);
  p.ui.pagePayload(pageButton('previous'));assert.deepEqual(Array.from(p.ui.fullPayloads.get(long.viewer).starts),[0]);
});
test('copy takes the complete recorded payload in either view and reports the outcome on the button',async()=>{
  const p=page(),recorded='{"command":"'+'x'.repeat(9000)+'","n":1.0}';
  p.ui.state.run.actions[0].arguments=recorded;
  for(const view of ['pretty','raw']){
    const f=payloadFixture(view),button={dataset:{copyPayload:'old',payloadKind:'input'},textContent:'Copy',closest:selector=>selector==='.payload-section'?f.section:null};
    await p.ui.handlePayloadClick({target:{closest:selector=>selector==='[data-full-payload],[data-copy-payload]'?button:null}});
    // Longer than the 4,000-character preview, and never the reformatted text.
    assert.equal(p.copied.at(-1),recorded);assert.equal(button.textContent,'Copied ✓');assert.equal(button.disabled,false);
    p.flush();assert.equal(button.textContent,'Copy');
  }
  assert.equal(p.copied.length,2);
});


test('Critical Path uses the frozen snapshot for scrolling and resize while ingestion continues',()=>{
  const p=page();let displayed,held;
  const viewer={update:run=>{displayed=run;},setFrozen:value=>{held=value;}};
  p.ui.setCausal(viewer);p.ui.pauseDisplay();
  assert.equal(displayed,p.ui.state.run);assert.equal(held,true);
  const clock=displayed.duration;p.append();
  assert.equal(displayed.duration,clock);
  assert.equal(p.ui.state.liveRun.duration,clock+1);
});
