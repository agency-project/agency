// Run with: node --test tests/agwebui/test_live_follow.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(replay=false) {
  const listeners={}, nodes=new Map();
  const timers=new Map(),delays=[];let timerId=0;
  const flush=()=>{const pending=[...timers.values()];timers.clear();pending.forEach(fn=>fn());};
  class Element {
    constructor(){this.innerHTML='';this.value='';this.options=[];this.children=[];
      this.dataset={};this.parts=new Map();this.classList={toggle:()=>{}};this.animations=[];this.style={setProperty:()=>{}};}
    setAttribute(name,value){this[name]=value;}
    contains(){return false;}
    scrollIntoView(){throw Error('Unexpected automatic scrolling');}
    get firstElementChild(){return this.children[0];}
    querySelector(selector){
      if(selector==='.empty-state')return null;
      if(!this.parts.has(selector)){const part=new Element();part.host=this;this.parts.set(selector,part);}
      return this.parts.get(selector);
    }
    querySelectorAll(selector){
      const key={'[data-agent-column]':'agentColumn','[data-call-group]':'callGroup','[data-call-card]':'callCard','[data-episode-card]':'episodeCard'}[selector];
      const descendants=[...this.children,...this.parts.values()];
      return descendants.flatMap(child=>[...(child.dataset[key]?[child]:[]),...child.querySelectorAll(selector)]);
    }
    insertBefore(child,before){
      child.remove();
      const index=before?this.children.indexOf(before):this.children.length;
      this.children.splice(index,0,child);child.parent=this;
    }
    remove(){if(this.parent){this.parent.children.splice(this.parent.children.indexOf(this),1);this.parent=null;}}
    getBoundingClientRect(){
      let top=0,current=this;
      while(current){if(current.parent)top+=Math.max(0,current.parent.children.indexOf(current))*100;current=current.parent||current.host;}
      return {left:0,top};
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
    querySelector:()=>null, querySelectorAll:()=>[...nodes.values()].filter(el=>el.dataset.agentControls), addEventListener:(name,fn)=>{
      (listeners[name]??=[]).push(fn);
    }};
  let subscription;const sent=[];
  const context=vm.createContext({document, URLSearchParams, URL, Date, structuredClone,
    setTimeout:(fn,delay)=>{const id=++timerId;timers.set(id,fn);delays.push(delay);return id;},
    clearTimeout:id=>timers.delete(id),
    location:{search:replay?'?mode=replay':''},
    window:{scrollY:0, addEventListener:()=>{}},
    TrajectoryStream:class {
      constructor(options){subscription=options;}
      close(){}
      sendExecution(command,extra){sent.push({type:'execution_command',command,...extra});return true;}
      sendReplay(type,extra){sent.push({type,...extra});return true;}
    },
    matchesAction:()=>true,
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
  vm.runInContext(source+`
    // Isolate follow behavior from chart/inspector markup, retaining the
    // real stream callback, selection handlers and updateTrajectory path.
    var realRenderSession=renderSession,realRenderInspector=renderInspector;
    render=()=>{};renderRunChrome=()=>{};renderSession=()=>{};
    renderInspector=()=>{};catalogOptions=()=>{};updateUrl=()=>{};
    var realReconcile=reconcileActivities;
    reconcileActivities=()=>{lastWindowEnd=state.windowEnd;};
    var paints=0,realUpdate=updateTrajectory;
    updateTrajectory=(...args)=>{paints++;realUpdate(...args);};
    var lastWindowEnd;
    globalThis.ui={state,loadRun,selectAction,handleClick,agentExecutionControls,holdHistory,pauseDisplay,resumeDisplay,pauseResumeExecution,pauseReplay,renderSession:realRenderSession,renderInspector:realRenderInspector,actionButton,
      windowEnd:()=>lastWindowEnd,reconcile:realReconcile,groupCallStack,renderCallStack,
      paints:()=>paints,setTimeline:viewer=>{timelineViewer=viewer;},visibleAgentColumns,addAgentColumn,removeAgentColumn,selectAgentColumn,renderTandemBoard,systemMetricTimelines,tandemConcurrentWork,tandemMetricTable,agentsView};
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
  const run={id:'live',duration:1,actions:[{id:'old',agent:'a',episode:'one',start:0,duration:1,kind:'tool',outcome:'success'}],
    episodes:[{id:'one',actions:['old'],status:'completed'}],agents:[],edges:[],coverage:{}};
  context.ui.loadRun('live',null,replay);
  subscription.onMessage({type:'snapshot',run});
  return {ui:context.ui, document, node,agentButton,flush,delays,sent,receive:message=>subscription.onMessage(message),transport:value=>subscription.onTransport(value),window:context.window,
    feed:agent=>{const columns=node('call-stack').querySelectorAll('[data-agent-column]');return (agent?columns.find(c=>c.dataset.agentColumn===agent):columns[0]).querySelector('.agent-call-stack');},
    visibility:hidden=>{document.hidden=hidden;listeners.visibilitychange.forEach(fn=>fn());},
    append:(paint=true)=>{subscription.onMessage({type:'patch', patches:[{
      actions:[{id:'new-call',agent:'a',episode:'two',start:1,duration:1,kind:'tool',outcome:'success'}],
      episodes:[{id:'two',actions:['new-call'],status:'running'}],meta:{duration:2},
    }]});if(paint)flush();},
  };
}

test('live calls keep following while the profiler is hidden and restored',()=>{
  const p=page();p.visibility(true);p.append();p.visibility(false);
  assert.equal(p.ui.state.follow,true);
  assert.equal(p.ui.state.cursor,2);
  assert.equal(p.ui.state.unread.size,0);
  assert.equal(p.ui.windowEnd(),null);
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
  assert.ok(card.animations.some(animation=>animation.options.duration===220));
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

test('defaults to two stable columns and supports more than three',()=>{
  const p=page();
  p.ui.state.run.agents=['workflow','a','b','c','d'].map(id=>({id}));
  p.ui.renderCallStack();
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','b']);
  const original=p.node('call-stack').firstElementChild;
  p.ui.state.run.agents.unshift({id:'new'});p.ui.renderCallStack();
  assert.deepEqual(Array.from(p.ui.state.agentColumns),['a','b']);
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
  p.ui.renderCallStack();
  assert.equal(p.feed('a').firstElementChild.dataset.callGroup,'running:a:tool');
  assert.equal(p.feed('b').firstElementChild.dataset.callGroup,'two');
  assert.match(p.node('hidden-agents').innerHTML,/c · 1 running · 1 failed/);
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


test('tandem live updates preserve expanded episodes and isolate agent calls',()=>{
  const p=page(),run=p.ui.state.run;
  run.agents=[{id:'a',label:'A'},{id:'b',label:'B',parent:'a'}];
  run.episodes[0].agent='a';run.episodes[0].title='First';
  run.counters={};run.intervals=[];
  p.ui.state.view='agents';p.ui.renderTandemBoard();
  const column=p.node('tandem-columns').children[0],body=column.querySelector('.col-body'),episode=body.children[0];
  episode.open=true;
  p.append();p.flush();
  assert.equal(p.node('tandem-columns').children[0],column);
  assert.equal(body.children[0],episode);
  assert.equal(episode.open,true);
  assert.match(p.node('tandem-system-metrics').innerHTML,/System metrics unavailable/);
  assert.match(p.node('tandem-concurrent').innerHTML,/data-action="new-call"/);
  p.ui.removeAgentColumn('b');
  assert.equal(p.node('tandem-columns').children.length,1);
});

test('metric tracks share interval coordinates, filter invalid samples and never fabricate data',()=>{
  const p=page(),run=p.ui.state.run;
  run.duration=10;run.counters={'cpu %':[[0,5],[5,25],[10,10],[20,100],[2,null]]};
  const chart=p.ui.systemMetricTimelines();
  assert.match(chart,/3 samples · peak 25/);
  assert.match(chart,/cx="439"/); // 130 + half the 618px plot
  assert.doesNotMatch(chart,/NaN|undefined/);
  run.counters={};assert.match(p.ui.systemMetricTimelines(),/no recorded resource samples/);
});

test('concurrent work separates overlapping calls and supports live actions without profiler spans',()=>{
  const p=page(),run=p.ui.state.run;
  run.duration=5;run.actions=[{id:'one',agent:'a',start:0,duration:4,kind:'tool',name:'read'},
    {id:'two',agent:'a',start:1,duration:3,kind:'tool',name:'test'}];
  const html=p.ui.tandemConcurrentWork([{id:'a',label:'A'}]);
  assert.match(html,/A \/ 2/);
  assert.match(html,/data-action="one"/);assert.match(html,/data-action="two"/);
});


function debuggerPage() {
  const p=page();
  p.ui.state.run.agents=[{id:'a',label:'A',status:'waiting_llm'},{id:'b',label:'B',status:'running'}];
  p.ui.state.run.status='running';
  p.ui.state.transport='connected';
  p.ui.state.executionControls={available:true,agents:['a','b']};
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
  }],execution_controls:{available:true,agents:['a','b']}});
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
  assert.equal(p.node('display-toggle').textContent,'▶ Resume Display');
  assert.match(p.node('display-state').textContent,/2 calls updated/);
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
  p.receive({type:'snapshot',run,execution_controls:{available:true,agents:['a','b']}});
  assert.equal(p.ui.state.executionPending,null);
  assert.equal(p.ui.state.run.agents[0].status,'waiting_llm');
  assert.equal(p.ui.state.displayPaused,true);
  p.ui.renderSession();
  assert.equal(p.node('execution-all').textContent,'▶ Resume Agents');
  p.node('execution-all').onclick();
  assert.equal(p.sent[1].command,'resume_all');
  assert.equal(p.ui.state.displayPaused,true);
  p.ui.state.selected={type:'agent',id:'a'};p.ui.renderSession();
  assert.equal(p.agentButton('agent').textContent,'▶ Resume');
});

test('reconnect snapshots are retained behind the paused display and run switching resets it',()=>{
  const p=page();p.ui.pauseDisplay();
  const frozen=p.ui.state.run;
  const recovered=structuredClone(p.ui.state.liveRun);
  recovered.duration=9;recovered.actions.push({id:'recovered',outcome:'success'});
  p.transport('disconnected');p.transport('connected');
  p.receive({type:'snapshot',run:recovered,resynced:true});
  assert.equal(p.ui.state.run,frozen);
  assert.equal(p.ui.state.liveRun,recovered);
  p.ui.resumeDisplay();
  assert.equal(p.ui.state.run,recovered);
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
  p.receive({type:'snapshot',run,execution_controls:{available:true,agents:['a','b']}});
  p.ui.renderSession();
  assert.equal(p.ui.state.executionPending,null);
  assert.equal(p.agentButton('agent').textContent,'▶ Resume');
  assert.match(p.node('execution-state').textContent,/1 agent paused/);
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
  p.receive({type:'snapshot',run,execution_controls:{available:true,agents:['a','b']}});
  p.ui.renderSession();
  p.node('execution-all').onclick();
  assert.equal(p.sent[1].command,'resume_all');
});

test('replay and saved review hide execution controls and replay pause uses only playback',()=>{
  const p=debuggerPage();p.ui.state.replay=true;p.ui.renderSession();
  assert.equal(p.node('execution-all').hidden,true);
  p.ui.pauseResumeExecution(true);
  assert.equal(p.sent.length,0);
  p.ui.state.replayState={playing:true};p.node('replay-play').onclick();
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

test('paused agent inspection and duration labels use the recorded execution duration',()=>{
  const p=debuggerPage();
  p.ui.state.run.agents[0].status='paused';
  p.ui.state.selected={type:'agent',id:'a'};p.ui.renderInspector();
  assert.match(p.node('inspector').innerHTML,/pill paused/);
  assert.match(p.ui.agentExecutionControls('a'),/Resume/);
  assert.match(p.ui.actionButton({...p.ui.state.run.actions[0],duration:25,execution_duration:5}),/>5.0s</);
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
  p.receive({type:'snapshot',run,execution_controls:{available:true,agents:['a','b']}});
  assert.equal(p.ui.state.executionPending,null);
  assert.match(p.ui.state.executionMessage,/SIGKILL delivered/);
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
  assert.equal(p.ui.state.executionMessage,'Execution stopped.');
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


test('trajectory and tandem column headings contain agent controls while global controls stay separate',()=>{
  const p=debuggerPage();p.ui.renderCallStack();
  for(const column of p.node('call-stack').children) {
    const html=column.querySelector('.agent-column-heading').innerHTML;
    assert.match(html,new RegExp(`data-agent-controls="${column.dataset.agentColumn}"`));
    assert.match(html,/Agent controls/);assert.match(html,/SIGKILL/);
    assert.doesNotMatch(html,/Stop all agents/);
  }
  p.ui.renderTandemBoard();
  for(const column of p.node('tandem-columns').children) {
    assert.match(column.querySelector('.col-head').innerHTML,/data-agent-controls=/);
  }
  const pageHtml=fs.readFileSync(path.join(__dirname,'../../agency/observability/agwebui/static/investigator.html'),'utf8');
  assert.match(pageHtml,/id="all-agent-controls"/);
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
  p.ui.renderSession();assert.match(group.innerHTML,/data-execution-stop="b"[^>]*disabled/);
  assert.doesNotMatch(group.innerHTML,/data-execution-kill="b"[^>]*disabled/);
});
