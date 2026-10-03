// Run with: node --test tests/agwebui/test_live_follow.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(replay=false) {
  const listeners={}, nodes=new Map();
  class Element {
    constructor(){this.innerHTML='';this.value='';this.options=[];this.children=[];
      this.dataset={};this.parts=new Map();this.classList={toggle:()=>{}};}
    contains(){return false;}
    scrollIntoView(){throw Error('Unexpected automatic scrolling');}
    get firstElementChild(){return this.children[0];}
    querySelector(selector){
      if(selector==='.empty-state')return null;
      if(!this.parts.has(selector))this.parts.set(selector,new Element());
      return this.parts.get(selector);
    }
    querySelectorAll(){return this.children;}
    insertBefore(child,before){
      child.remove();
      const index=before?this.children.indexOf(before):this.children.length;
      this.children.splice(index,0,child);child.parent=this;
    }
    remove(){if(this.parent){this.parent.children.splice(this.parent.children.indexOf(this),1);this.parent=null;}}
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
  vm.runInContext(source+`
    // Isolate follow behavior from chart/inspector markup, retaining the
    // real stream callback, selection handlers and updateTrajectory path.
    render=()=>{};renderRunChrome=()=>{};renderSession=()=>{};
    renderInspector=()=>{};catalogOptions=()=>{};updateUrl=()=>{};
    var realReconcile=reconcileActivities;
    reconcileActivities=()=>{lastWindowEnd=state.windowEnd;};
    var lastWindowEnd;
    globalThis.ui={state,loadRun,selectAction,holdHistory,
      windowEnd:()=>lastWindowEnd,reconcile:realReconcile};
  `,context);
  const run={id:'live',duration:1,actions:[{id:'old',episode:'one',start:0,duration:1,kind:'tool'}],
    episodes:[{id:'one',actions:['old'],status:'completed'}],agents:[],edges:[],coverage:{}};
  context.ui.loadRun('live',null,replay);
  subscription.onMessage({type:'snapshot',run});
  return {ui:context.ui, document, node,
    visibility:hidden=>{document.hidden=hidden;listeners.visibilitychange.forEach(fn=>fn());},
    append:()=>subscription.onMessage({type:'patch', patches:[{
      action:{id:'new-call',episode:'two',start:1,duration:1,kind:'tool'},
      episode:{id:'two',actions:['new-call'],status:'running'},duration:2,
    }]}),
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
  p.ui.state.run.actions.push({id:'third',episode:'two',start:2,duration:1,kind:'tool'});
  p.ui.state.run.episodes[1].actions.push('third');
  p.ui.reconcile();
  assert.deepEqual(list.firstElementChild.querySelector('.episode-actions').children.map(c=>c.dataset.action),['third','new-call']);
});
