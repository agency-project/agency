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
      this.dataset={};this.parts=new Map();this.classList={toggle:()=>{}};this.animations=[];}
    contains(){return false;}
    scrollIntoView(){throw Error('Unexpected automatic scrolling');}
    get firstElementChild(){return this.children[0];}
    querySelector(selector){
      if(selector==='.empty-state')return null;
      if(!this.parts.has(selector)){const part=new Element();part.host=this;this.parts.set(selector,part);}
      return this.parts.get(selector);
    }
    querySelectorAll(selector){
      const key={'[data-call-group]':'callGroup','[data-call-card]':'callCard','[data-episode-card]':'episodeCard'}[selector];
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
    querySelector:()=>null, addEventListener:(name,fn)=>{
      (listeners[name]??=[]).push(fn);
    }};
  let subscription;
  const context=vm.createContext({document, URLSearchParams, URL, Date,
    setTimeout:(fn,delay)=>{const id=++timerId;timers.set(id,fn);delays.push(delay);return id;},
    clearTimeout:id=>timers.delete(id),
    location:{search:replay?'?mode=replay':''},
    window:{scrollY:0, addEventListener:()=>{}},
    TrajectoryStream:class {
      constructor(options){subscription=options;}
      close(){}
    },
    matchesAction:()=>true,
    changedTrajectoryActions:()=>['new-call'],
    mergeTrajectory:(run,patch)=>{
      run.actions.push(patch.action);run.episodes.push(patch.episode);
      run.duration=patch.duration;return [patch.action.id];
    },
  });
  const source=fs.readFileSync(path.join(__dirname,
    '../../agency/observability/agwebui/static/investigator.js'),'utf8')
    .replace(/^import .*;\n/gm,'').replace(/refresh\(\);\s*$/,'');
  vm.runInContext(fs.readFileSync(path.join(__dirname,
    '../../agency/observability/agwebui/static/investigator-model.js'),'utf8')
    .replace(/export function/g,'function'),context);
  vm.runInContext(source+`
    // Isolate follow behavior from chart/inspector markup, retaining the
    // real stream callback, selection handlers and updateTrajectory path.
    render=()=>{};renderRunChrome=()=>{};renderSession=()=>{};
    renderInspector=()=>{};catalogOptions=()=>{};updateUrl=()=>{};
    var realReconcile=reconcileActivities;
    reconcileActivities=()=>{lastWindowEnd=state.windowEnd;};
    var paints=0,realUpdate=updateTrajectory;
    updateTrajectory=(...args)=>{paints++;realUpdate(...args);};
    var lastWindowEnd;
    globalThis.ui={state,loadRun,selectAction,holdHistory,
      windowEnd:()=>lastWindowEnd,reconcile:realReconcile,groupCallStack,renderCallStack,
      paints:()=>paints};
  `,context);
  const run={id:'live',duration:1,actions:[{id:'old',episode:'one',start:0,duration:1,kind:'tool',outcome:'success'}],
    episodes:[{id:'one',actions:['old'],status:'completed'}],agents:[],edges:[],coverage:{}};
  context.ui.loadRun('live',null,replay);
  subscription.onMessage({type:'snapshot',run});
  return {ui:context.ui, document, node,flush,delays,window:context.window,
    visibility:hidden=>{document.hidden=hidden;listeners.visibilitychange.forEach(fn=>fn());},
    append:(paint=true)=>{subscription.onMessage({type:'patch', patches:[{
      action:{id:'new-call',episode:'two',start:1,duration:1,kind:'tool',outcome:'success'},
      episode:{id:'two',actions:['new-call'],status:'running'},duration:2,
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
  const p=page();p.node('cursor').oninput({target:{value:'0.5'}});
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
  const stack=p.node('call-stack');
  assert.equal(stack.firstElementChild.dataset.callGroup,'running:a:tool');
  assert.equal(stack.firstElementChild.querySelector('.call-group-grid').children.length,1);
  calls[0].outcome='success';calls[0].duration=2;p.ui.renderCallStack();
  assert.equal(stack.firstElementChild.dataset.callGroup,'newer');
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
  const stack=p.node('call-stack');
  assert.equal(stack.children.length,51);
  assert.equal(stack.firstElementChild.dataset.callGroup,'running:a:tool');
  assert.equal(stack.children[1].dataset.callGroup,'finished-54');
});

test('a finished call leaves its running peers and reuses its card in serial history',()=>{
  const p=page();
  const calls=[
    {id:'a',agent:'agent',kind:'tool',start:0,duration:5,outcome:'running'},
    {id:'b',agent:'agent',kind:'tool',start:1,duration:2,outcome:'running'},
  ];
  p.ui.state.run.actions=calls;p.ui.renderCallStack();
  const stack=p.node('call-stack');
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
  const card=p.node('call-stack').firstElementChild.querySelector('.call-group-grid').firstElementChild;
  assert.equal(card.animations.length,0);
});
