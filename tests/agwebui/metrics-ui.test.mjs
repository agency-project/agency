import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const root=new URL('../../agency/observability/agwebui/static/',import.meta.url);
const asURL=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const model=asURL(await readFile(new URL('metrics-model.js',root),'utf8'));
const existing=asURL(await readFile(new URL('investigator-model.js',root),'utf8'));
let stored=null;
globalThis.localStorage={getItem:()=>stored,setItem:(k,v)=>{stored=v;}};
const source=(await readFile(new URL('metrics-ui.js',root),'utf8')).replace('./metrics-model.js',model).replace('./investigator-model.js',existing);
const {metricsSummary,renderMetricsSummary,profilerView,handleMetricEvent}=await import(asURL(source));
const {measurementView}=await import(model);
const run={id:'test',title:'Fixture',duration:5,status:'running',actions:[{id:'a',agent:'worker',name:'bash',kind:'tool',start:0,duration:5,outcome:'running'}],agents:[{id:'worker',label:'Worker',status:'running'}],counters:{},coverage:{}};
function click(dataset){let target;const el={dataset,hasAttribute:key=>key==='data-metric-toggle'&&dataset.toggle};handleMetricEvent({target:{closest:()=>el}},{render:()=>{},navigate:id=>{target=id;},reveal:id=>{target=id;}});return target;}
test('six defaults, expansion, category selection and persistence',()=>{assert.match(metricsSummary(run),/Metrics \+/);click({toggle:true});assert.match(metricsSummary(run),/View detailed Activity profiler/);click({metricCategory:'LLM'});assert.match(metricsSummary(run),/View detailed LLM profiler/);assert.equal(JSON.parse(stored).category,'LLM');click({metricPin:'tools'});assert.equal(JSON.parse(stored).pins.length,5);click({metricPin:'oldest'});assert.equal(JSON.parse(stored).pins.length,6);});
test('category and event navigation use exact existing identifiers',()=>{assert.equal(click({metricView:'tools',metricKey:'errors'}),'tools');assert.equal(click({metricEvent:'a'}),'a');});
test('all dedicated views render with missing telemetry and no invented charts',()=>{for(const category of ['Activity','LLM','Tools','Systems'])assert.match(profilerView(run,category),new RegExp(category+' profiler'));assert.doesNotMatch(profilerView(run,'Systems'),/<svg/);assert.match(profilerView(run,'LLM'),/First-token boundary is not projected/);});
test('agent elapsed clock takes precedence over program duration',()=>{
  const html=metricsSummary({...run,duration:120,execution_started_at:100,execution_wall_time:20});
  assert.match(html,/20s · running/);assert.doesNotMatch(html,/120s · running/);
  assert.match(metricsSummary({...run,execution_started_at:null,execution_wall_time:0}),/0s · not started/);
});
test('latency has both units, charts expose axes and a useful explanation',()=>{
  const observed={...run,actions:[{...run.actions[0],kind:'model',outcome:'success',duration:1.25,tokens:12,output_tokens:8}]};
  const html=profilerView(observed,'LLM');
  assert.match(html,/1,250 ms \/ 1.25 s/);
  assert.match(html,/Time since recording start \(s\)/);
  assert.match(html,/Latency \(ms \/ s\)/);
  assert.match(html,/Sorted observation index/);
  assert.match(html,/Spikes show when request latency increased/);
});
test('systems memory cards identify each process and distinguish RAM',()=>{
  const html=profilerView({...run,counters:{'workload:memory_mb':[[1,100]],'proc:12:rss_mb':[[1,20]],'proc:34:rss_mb':[[1,30]]}},'Systems');
  assert.match(html,/RAM · Workload/);assert.match(html,/RSS · Process 12/);assert.match(html,/RSS · Process 34/);
});
test('compact Systems selects curated measurements and only one process RSS',()=>{
  click({metricCategory:'Systems'});
  if(!metricsSummary(run).includes('View detailed Systems'))click({toggle:true});
  const html=metricsSummary({...run,counters:{'proc:12:rss_mb':[[1,20]],'proc:34:rss_mb':[[1,30]],'proc:12:vms_mb':[[1,99]],'workload:memory_mb':[[1,100]],'workload:cpu_pct':[[1,25]]}});
  assert.match(html,/CPU · Workload/);assert.match(html,/RAM · Workload/);assert.match(html,/RSS · Process 12/);
  assert.doesNotMatch(html,/RSS · Process 34/);assert.doesNotMatch(html,/Virtual memory/);
});


test('live metric updates preserve pressed toggle and category controls',()=>{
  const makeNode=()=>({dataset:{},parts:new Map(),cards:[],
    set innerHTML(html){this.cards=[...html.matchAll(/data-metric-card="([^"]+)"/g)].map(match=>({...makeNode(),dataset:{metricCard:match[1]}}));},
    querySelectorAll(){return this.cards;},querySelector(key){if(!this.parts.has(key))this.parts.set(key,makeNode());return this.parts.get(key);},
    setAttribute(name,value){this[name]=value;},removeAttribute(name){delete this.dataset[name.replace('data-','').replace(/-([a-z])/g,(_,letter)=>letter.toUpperCase())];}});
  let mounts=0,nodes,tabs;
  const root={set innerHTML(value){mounts++;nodes=new Map(['[data-metric-toggle]','.prof-summary-slots','.prof-expanded','.prof-cards','.prof-detail-link','.prof-summary-note'].map(k=>[k,makeNode()]));tabs=['Activity','LLM','Tools','Systems'].map(metricCategory=>({...makeNode(),dataset:{metricCategory}}));nodes.get('.prof-expanded').querySelector=k=>nodes.get(k);},querySelector:k=>nodes?.get(k),querySelectorAll:()=>tabs};
  click({metricCategory:'Activity'});
  renderMetricsSummary(root,run);
  const pressed=root.querySelector('[data-metric-toggle]'),pressedTab=tabs[0],metric=root.querySelector('.prof-cards').cards[0],pressedPin=metric.querySelector('.prof-pin');
  for(let i=0;i<100;i++)renderMetricsSummary(root,{...run,duration:5+i});
  assert.equal(root.querySelector('[data-metric-toggle]'),pressed);
  assert.equal(tabs[0],pressedTab);assert.equal(mounts,1);
  assert.equal(root.querySelector('.prof-cards').cards[0],metric);assert.equal(metric.querySelector('.prof-pin'),pressedPin);assert.equal(metric.querySelector('.prof-value').querySelector('strong').textContent,'104s');
  if(!JSON.parse(stored).expanded)click({toggle:true});
  renderMetricsSummary(root,run,{preferencesOnly:true});
  assert.equal(pressed['aria-expanded'],'true');assert.equal(root.querySelector('.prof-expanded').hidden,false);
  click({toggle:true});
  const unreadable={get actions(){throw Error('Collapse must not project telemetry');}};
  renderMetricsSummary(root,unreadable,{preferencesOnly:true});
  assert.equal(pressed['aria-expanded'],'false');assert.equal(root.querySelector('.prof-expanded').hidden,true);
});
test('non-system summaries never traverse resource history',()=>{
  click({metricCategory:'Activity'});
  const unreadable={...run,get counters(){throw Error('Unrequested resource projection');}};
  assert.match(metricsSummary(unreadable),/Metrics/);
  assert.match(profilerView(unreadable,'LLM'),/LLM profiler/);
});

function pinDialogFixture(){
  const dialogs=[],origin={dataset:{metricPin:'tool95'},isConnected:true,hasAttribute:()=>false,focus(options){this.focusOptions=options;}};
  const document={body:{append(dialog){dialogs.push(dialog);}},createElement(){
    const parts=new Map(),events={};
    const button=()=>({dataset:{},events:{},addEventListener(type,fn){this.events[type]=fn;},focus(options){this.focusOptions=options;}});
    return {events,slots:[],setAttribute(name,value){this[name]=value;},set innerHTML(html){this.html=html;this.slots=[...html.matchAll(/data-pin-slot="(\d+)"/g)].map(match=>({...button(),dataset:{pinSlot:match[1]}}));},querySelectorAll: function(){return this.slots;},querySelector(selector){if(selector==='[data-pin-slot]')return this.slots[0];if(!parts.has(selector))parts.set(selector,button());return parts.get(selector);},addEventListener(type,fn){events[type]=fn;},showModal(){this.open=true;},close(){this.open=false;},remove(){dialogs.splice(dialogs.indexOf(this),1);}};
  }};
  globalThis.document=document;
  let renders=0;
  return {dialogs,origin,open(){handleMetricEvent({target:{closest:()=>origin}},{render:()=>renders++,navigate:()=>{},reveal:()=>{}});return dialogs[0];},get renders(){return renders;}};
}
test('a full summary opens six actual replacement buttons and saves the selected slot',()=>{
  click({metricPin:'oldest'});click({metricPin:'tools'}); // Restore six slots with tool95 absent.
  const before=JSON.parse(stored).pins,fixture=pinDialogFixture();metricsSummary(run);
  const dialog=fixture.open();assert.equal(dialog.open,true);assert.equal(dialog.slots.length,6);
  assert.deepEqual(JSON.parse(stored).pins,before);assert.equal(fixture.renders,0);
  dialog.slots[2].events.click();
  const after=JSON.parse(stored).pins;assert.equal(after[2],'tool95');assert.equal(after.length,6);
  assert.equal(fixture.renders,1);assert.equal(fixture.dialogs.length,0);assert.equal(fixture.origin.focusOptions.preventScroll,true);
});
test('Escape cancels the slot picker without changing preferences',()=>{
  const fixture=pinDialogFixture();fixture.origin.dataset.metricPin='tool50';
  const before=stored,dialog=fixture.open();let prevented=false;
  dialog.events.cancel({preventDefault(){prevented=true;}});
  assert.equal(prevented,true);assert.equal(stored,before);assert.equal(fixture.renders,0);assert.equal(fixture.dialogs.length,0);
});

const measured=(measurement,extra={})=>({...run,execution_started_at:100,execution_wall_time:20,measurement:{state:'running',policy:'automatic',held:false,automatic_end_at:null,automatic_end_source:null,effective_end_at:null,end_source:null,...measurement},...extra});
test('wall clock card shows the backend duration, its state and whether the boundary is automatic or manual',()=>{
  // Deliberately inconsistent timestamps: the card must show the reported duration, not derive one.
  const ended=metricsSummary(measured({state:'ended',automatic_end_at:900,effective_end_at:900,automatic_end_source:'workload_completed',end_source:'workload_done'},{duration:500,status:'running'}));
  assert.match(ended,/Wall clock · automatic/);assert.match(ended,/20s · ended/);assert.doesNotMatch(ended,/800s|500s/);
  assert.match(ended,/profiler finalization excluded/);
  assert.match(metricsSummary(measured({})),/20s · running/);
  const manual=metricsSummary(measured({state:'ended',policy:'manual',automatic_end_at:130,effective_end_at:120,end_source:'user'}));
  assert.match(manual,/Wall clock · manual/);assert.match(manual,/20s · ended/);assert.match(manual,/automatic boundary is kept/);
  const held=metricsSummary(measured({policy:'manual',held:true,automatic_end_at:130}));
  assert.match(held,/Wall clock · manual/);assert.match(held,/20s · running/);assert.match(held,/counts continuously/);
  assert.match(metricsSummary(measured({state:'pending'},{execution_started_at:null,execution_wall_time:0})),/0s · not started/);
  // Saved traces without measurement metadata keep the previous wording.
  assert.match(metricsSummary({...run,duration:12,status:'completed'}),/Wall clock<\/small><strong>12s · completed/);
});
test('only measurement actions that fit the current state are offered',()=>{
  const actions=(measurement,extra)=>measurementView(measured(measurement,extra)).actions;
  assert.deepEqual(actions({}),{end:true,hold:true,reset:false,endAt:true});
  assert.deepEqual(actions({state:'ended',automatic_end_at:130,effective_end_at:130}),{end:false,hold:true,reset:false,endAt:true});
  assert.deepEqual(actions({state:'ended',policy:'manual',effective_end_at:120,end_source:'user'}),{end:false,hold:true,reset:true,endAt:true});
  assert.deepEqual(actions({policy:'manual',held:true}),{end:true,hold:false,reset:true,endAt:true});
  assert.deepEqual(actions({state:'pending'},{execution_started_at:null,execution_wall_time:0}),{end:false,hold:true,reset:false,endAt:false});
  assert.equal(measurementView({...run,execution_wall_time:20}),null);
});
test('the measurement menu is live-only, stays mounted and reflects the current state',()=>{
  const makeNode=()=>({dataset:{},parts:new Map(),cards:[],open:true,
    set innerHTML(html){this.cards=[...html.matchAll(/data-metric-card="([^"]+)"/g)].map(match=>({...makeNode(),dataset:{metricCard:match[1]}}));},
    querySelectorAll(){return this.cards;},querySelector(key){if(!this.parts.has(key))this.parts.set(key,makeNode());return this.parts.get(key);},
    setAttribute(name,value){this[name]=value;},removeAttribute(){}});
  let nodes;
  const root={set innerHTML(value){nodes=new Map(['[data-metric-toggle]','.prof-summary-slots','.prof-expanded','.prof-cards','.prof-detail-link','.prof-summary-note'].map(k=>[k,makeNode()]));nodes.get('.prof-expanded').querySelector=k=>nodes.get(k);},querySelector:k=>nodes?.get(k),querySelectorAll:()=>[]};
  const reset={dataset:{},hasAttribute:key=>key==='data-metric-reset'};handleMetricEvent({target:{closest:()=>reset}},{render:()=>{},navigate:()=>{},reveal:()=>{}});
  renderMetricsSummary(root,measured({}));
  const wall=root.querySelector('.prof-summary-slots').cards.find(card=>card.dataset.metricCard==='wall'),menu=wall.querySelector('.measure-menu');
  assert.equal(menu.hidden,true);assert.equal(menu.open,false); // Replay, saved review, frozen or disconnected.
  renderMetricsSummary(root,measured({}),{measurement:{enabled:true,message:''}});
  const hidden=name=>menu.querySelector(name).hidden;
  assert.equal(menu.hidden,false);assert.equal(hidden('[data-measure="end"]'),false);assert.equal(hidden('[data-measure="hold"]'),false);assert.equal(hidden('[data-measure="reset"]'),true);
  menu.open=true;
  renderMetricsSummary(root,measured({state:'ended',policy:'manual',effective_end_at:120,end_source:'user'},{execution_wall_time:20.04}),{measurement:{enabled:true,message:'The measurement cannot end in the future.'}});
  assert.equal(wall.querySelector('.measure-menu'),menu);assert.equal(menu.open,true);
  assert.equal(hidden('[data-measure="end"]'),true);assert.equal(hidden('[data-measure="reset"]'),false);assert.equal(hidden('.measure-end-at'),false);
  assert.equal(menu.querySelector('.measure-status').textContent,'Ended by you');
  assert.equal(menu.querySelector('.measure-message').textContent,'The measurement cannot end in the future.');
  assert.equal(menu.querySelector('input').placeholder,'20.0');
  assert.equal(wall.querySelector('.prof-value').querySelector('strong').textContent,'20s · ended');
  // No non-wall card ever receives measurement controls.
  assert.equal(root.querySelector('.prof-summary-slots').cards.find(card=>card.dataset.metricCard==='tools').parts.has('.measure-menu'),false);
});
test('measurement clicks send only the command and a user-chosen elapsed time',()=>{
  const sent=[];let renders=0,navigated=false;
  const press=(measure,value)=>handleMetricEvent({target:{closest:()=>({dataset:{measure},hasAttribute:()=>false,closest:()=>({querySelector:()=>({value})})})}},{render:()=>renders++,navigate:()=>{navigated=true;},reveal:()=>{navigated=true;},measure:(command,extra)=>sent.push([command,extra])});
  for(const command of ['end','hold','reset'])assert.equal(press(command),true);
  assert.deepEqual(sent,[['end',undefined],['hold',undefined],['reset',undefined]]);
  press('end-at','25.5');assert.deepEqual(sent.at(-1),['end',{elapsed:25.5}]);
  for(const invalid of ['',' ','-1','abc','Infinity'])press('end-at',invalid);
  assert.equal(sent.length,9);for(const [,extra] of sent.slice(4))assert.match(extra.error,/seconds since the measurement started/);
  // No timestamp is invented client-side and no agent control or navigation fires.
  assert.equal(JSON.stringify(sent).includes('at"'),false);assert.equal(navigated,false);assert.equal(renders,0);
});

// Agent-scoped summary: one dashboard, filtered through the multi-select scope.
const {setMetricScope,metricScope}=await import(asURL(source));
const press=(attributes,dataset={},counts={render:0,rescope:0})=>{handleMetricEvent({target:{closest:()=>({dataset,hasAttribute:name=>attributes.includes(name)})}},{render:()=>{counts.render++;},rescope:()=>{counts.rescope++;},navigate:()=>{},reveal:()=>{}});return counts;};
const tool=(id,agent,duration,outcome='success')=>({id,agent,name:'bash',kind:'tool',start:0,duration,outcome});
const llm=(id,agent,tokens,output_tokens)=>({id,agent,name:'chat',kind:'model',start:1,duration:2,outcome:'success',tokens,output_tokens});
const team=()=>({id:'team',title:'Team',duration:10,status:'running',coverage:{},
  agents:[{id:'workflow',label:'workflow'},{id:'a',label:'Agent A',status:'running',sandbox:'sandbox_a_0000'},{id:'b',label:'Agent B',status:'running',sandbox:'sandbox_b_0000'},{id:'c',label:'Agent C',status:'running'}],
  actions:[tool('a1','a',1),tool('a2','a',9),tool('b1','b',2,'failed'),tool('c1','c',100),llm('am','a',1000,200),llm('bm','b',50,5),llm('cm','c')],
  counters:{'workload:cpu_pct':[[10,300]],'workload:memory_mb':[[10,9000]],'cg:sandbox_a_0000:cpu_pct':[[10,12.5]],'sandbox:sandbox_a_0000:mem_mb':[[10,256]],'cg:sandbox_b_0000:cpu_pct':[[10,150]],'sandbox:sandbox_b_0000:mem_mb':[[10,100]]}});
function scopeFixture(){
  const makeNode=()=>({dataset:{},parts:new Map(),cards:[],boxes:[],rebuilt:0,
    set innerHTML(html){this.rebuilt++;this.html=html;this.cards=[...html.matchAll(/data-metric-card="([^"]+)"/g)].map(match=>({...makeNode(),dataset:{metricCard:match[1]}}));
      this.boxes=[...html.matchAll(/data-scope-agent="([^"]+)"/g)].map(match=>({dataset:{scopeAgent:match[1]},checked:false,parentElement:makeNode()}));},
    querySelectorAll(selector){return selector==='[data-scope-agent]'?this.boxes:this.cards;},querySelector(key){if(!this.parts.has(key))this.parts.set(key,makeNode());return this.parts.get(key);},
    setAttribute(name,value){this[name]=value;},removeAttribute(){}});
  let nodes,mounts=0;
  const root={set innerHTML(value){mounts++;nodes=new Map(['[data-metric-toggle]','.prof-summary-slots','.prof-expanded','.prof-cards','.prof-detail-link','.prof-summary-note','.metric-scope'].map(key=>[key,makeNode()]));nodes.get('.prof-expanded').querySelector=key=>nodes.get(key);},querySelector:key=>nodes?.get(key),querySelectorAll:()=>[]};
  const draw=run=>{renderMetricsSummary(root,run);renderMetricsSummary(root,run);};
  const menu=()=>nodes.get('.metric-scope'),list=()=>menu().querySelector('.metric-scope-agents');
  return {draw,list,mounts:()=>mounts,label:()=>menu().querySelector('.metric-scope-label').textContent,all:()=>menu().querySelector('[data-scope-all]').checked,
    checked:()=>list().boxes.filter(box=>box.checked).map(box=>box.dataset.scopeAgent),
    status:id=>list().boxes.find(box=>box.dataset.scopeAgent===id).parentElement.querySelector('small').textContent,
    value:key=>nodes.get('.prof-summary-slots').cards.find(card=>card.dataset.metricCard===key).querySelector('.prof-value').querySelector('strong').textContent,
    card:key=>nodes.get('.prof-summary-slots').cards.find(card=>card.dataset.metricCard===key),
    slotsHtml:()=>nodes.get('.prof-summary-slots').html,slotsBuilt:()=>nodes.get('.prof-summary-slots').rebuilt,note:()=>nodes.get('.prof-summary-note').textContent};
}
test('scope defaults to all agents and the same cards filter to one agent or any subset',()=>{
  press(['data-metric-reset']);setMetricScope(null);
  const run=team(),view=scopeFixture();view.draw(run);
  assert.equal(view.label(),'All agents (3)');assert.equal(view.all(),true);assert.deepEqual(view.checked(),['a','b','c']);
  assert.equal(view.value('tools'),'4');assert.equal(view.value('agents'),'3');assert.equal(view.value('tokens'),'1,050 / 205');assert.equal(view.card('wall').dataset.unscoped,'false');
  // Unticking one agent from "All agents" leaves the rest selected.
  const counts=press([],{scopeAgent:'b'});assert.deepEqual(counts,{render:0,rescope:1});
  view.draw(run);
  assert.equal(view.label(),'2 of 3 agents');assert.equal(view.all(),false);assert.deepEqual(view.checked(),['a','c']);
  assert.equal(view.value('tools'),'3');assert.equal(view.value('agents'),'2 of 3');assert.equal(view.value('tokens'),'1,000 / 200');assert.equal(view.value('errors'),'0 · 0%');
  press([],{scopeAgent:'a'});view.draw(run);
  assert.equal(view.label(),'Agent C');assert.equal(view.value('tools'),'1');assert.equal(view.value('tokens'),'Unavailable / Unavailable');
  // Wall clock is not attributable to agents; it stays execution-wide and says so.
  assert.equal(view.card('wall').dataset.unscoped,'true');assert.match(view.card('wall').querySelector('.prof-value').querySelector('small').textContent,/entire execution/);
  assert.equal(view.mounts(),1);
});
test('any subset can be selected, including none, and every agent can be restored',()=>{
  const run=team(),view=scopeFixture();
  // Clearing the last selected agent leaves a valid empty scope.
  setMetricScope(['c']);view.draw(run);
  press([],{scopeAgent:'c'});view.draw(run);
  assert.equal(metricScope().size,0);assert.equal(view.label(),'No agents selected');assert.equal(view.all(),false);assert.deepEqual(view.checked(),[]);
  // Agents can be reselected one at a time from nothing.
  press([],{scopeAgent:'b'});view.draw(run);
  assert.deepEqual([...metricScope()],['b']);assert.equal(view.label(),'Agent B');assert.equal(view.value('tools'),'1');
  press([],{scopeAgent:'a'});view.draw(run);assert.equal(view.label(),'2 of 3 agents');
  // Completing the set is all agents again, so later agents are included.
  press([],{scopeAgent:'c'});assert.equal(metricScope(),null);
  // "All agents" switches between every agent and none.
  view.draw(run);assert.deepEqual(press(['data-scope-all']),{render:0,rescope:1});view.draw(run);
  assert.equal(metricScope().size,0);assert.deepEqual(view.checked(),[]);
  press(['data-scope-all']);view.draw(run);
  assert.equal(metricScope(),null);assert.equal(view.all(),true);assert.deepEqual(view.checked(),['a','b','c']);
  // From a partial selection it selects all.
  setMetricScope(['b']);view.draw(run);press(['data-scope-all']);assert.equal(metricScope(),null);
  setMetricScope([]);assert.equal(metricScope().size,0);setMetricScope(null);
});
test('zero selected agents shows a deliberate empty state instead of stale or computed cards',()=>{
  press(['data-metric-reset']);
  const run=team(),view=scopeFixture();
  setMetricScope(null);view.draw(run);assert.equal(view.value('tools'),'4');
  const slots=view.card('tools');
  setMetricScope([]);
  // Nothing may be projected for an empty scope; reading calls or samples would throw here.
  const guard=source=>Object.defineProperties({...source},{actions:{get(){throw Error('No projection for an empty scope');}},counters:{get(){throw Error('No samples for an empty scope');}}});
  const guarded=guard(run);
  view.draw(guarded);
  assert.equal(view.label(),'No agents selected');assert.equal(view.card('tools'),undefined);
  const html=view.slotsHtml();
  assert.match(html,/class="prof-scope-empty" role="status"/);assert.match(html,/<strong>No agents selected<\/strong>/);
  assert.match(html,/Choose agents in the Metrics scope menu/);assert.match(html,/<button type="button" data-scope-all>Select all agents<\/button>/);
  assert.match(view.note(),/no agents selected/);
  // The selector stays mounted and usable, and the empty state is not rebuilt on every live update.
  const built=view.slotsBuilt();view.draw(guard({...run,duration:99}));assert.equal(view.slotsBuilt(),built);assert.equal(view.mounts(),1);
  assert.deepEqual(view.list().boxes.map(box=>box.dataset.scopeAgent),['a','b','c']);
  // The string form used for static rendering agrees.
  assert.match(metricsSummary(guarded),/No agents selected/);assert.doesNotMatch(metricsSummary(guarded),/data-metric-card/);
  // Restoring from the empty state brings the cards back with current values.
  assert.deepEqual(press(['data-scope-all']),{render:0,rescope:1});
  view.draw({...run,duration:12});
  assert.equal(metricScope(),null);assert.equal(view.label(),'All agents (3)');assert.equal(view.value('tools'),'4');assert.notEqual(view.card('tools'),slots);
  // A column header (or any single agent) also leaves the empty state.
  setMetricScope([]);view.draw(run);setMetricScope(['a']);view.draw(run);assert.equal(view.value('tools'),'2');
  setMetricScope(null);
});
test('a selection is kept while agents spawn, terminate and report live updates',()=>{
  const run=team(),view=scopeFixture();
  setMetricScope(['a']);view.draw(run);
  const boxes=view.list().boxes,built=view.list().rebuilt;
  // Live additive update for the selected agent, with execution-wide counters present.
  const live={...run,duration:11,metrics_totals:{tools:900,models:90,completed_tools:900,tool_errors:1,input:5e5,output:1e4,input_samples:90,output_samples:90},
    metrics_by_agent:{a:{tools:400,models:30,completed_tools:400,tool_errors:0,input:12000,output:300,input_samples:30,output_samples:30},b:{tools:500,models:60,completed_tools:500,tool_errors:1,input:488000,output:9700,input_samples:60,output_samples:60}}};
  view.draw(live);
  assert.equal(view.value('tools'),'400');assert.equal(view.value('tokens'),'12K / 300');assert.equal(view.list().boxes,boxes);
  // Termination changes the status in place; the mounted checkboxes are not replaced.
  view.draw({...live,agents:live.agents.map(agent=>agent.id==='a'?{...agent,status:'completed'}:agent)});
  assert.equal(view.status('a'),'completed');assert.equal(view.list().rebuilt,built);assert.equal(view.label(),'Agent A');assert.equal(view.value('tools'),'400');
  // A spawned agent is listed but does not join an explicit selection.
  view.draw({...live,agents:[...live.agents,{id:'d',label:'Agent D',status:'running'}]});
  assert.deepEqual(view.list().boxes.map(box=>box.dataset.scopeAgent),['a','b','c','d']);assert.deepEqual(view.checked(),['a']);assert.equal(view.label(),'Agent A');
  // With all agents selected the new agent is included automatically.
  setMetricScope(null);view.draw({...live,agents:[...live.agents,{id:'d',label:'Agent D',status:'running'}]});
  assert.equal(view.label(),'All agents (4)');assert.equal(view.value('tools'),'900');assert.deepEqual(view.checked(),['a','b','c','d']);
  // A spawn while nothing is selected stays empty: the choice was explicit.
  setMetricScope([]);view.draw({...live,agents:[...live.agents,{id:'d',label:'Agent D',status:'running'}]});
  assert.equal(view.label(),'No agents selected');assert.deepEqual(view.checked(),[]);assert.equal(view.list().boxes.length,4);
  setMetricScope(null);
});
test('system cards keep entire-system meaning while agent sandbox cards follow the scope',()=>{
  press([],{metricCategory:'Systems'});if(!metricsSummary(team()).includes('View detailed Systems'))press(['data-metric-toggle']);
  const run=team();
  const all=metricsSummary(run);
  // "All agents" is the agents' own containers; the workload series is the entire system.
  assert.match(all,/CPU · agent sandboxes<\/small><strong>162\.5 %/);assert.match(all,/CPU · Workload<\/small><strong>300 %/);
  assert.match(all,/Sum of 2 of 2 sandbox containers of all agents/);assert.match(all,/excludes orchestrator, host services, runtime daemons and model inference/);
  setMetricScope(['a']);const one=metricsSummary(run);
  assert.match(one,/CPU · agent sandboxes<\/small><strong>12\.5 %/);assert.match(one,/RAM · agent sandboxes<\/small><strong>256 MiB/);
  assert.match(one,/data-metric-card="system:workload:cpu_pct" data-unscoped="true"/);assert.match(one,/Not filtered by the agent selection · Workload/);
  assert.match(one,/CPU · Workload<\/small><strong>300 %/);assert.match(one,/Scope: Agent A/);
  // No sandbox recorded for the selected agent: marked unavailable, never borrowed from the system.
  setMetricScope(['c']);const none=metricsSummary(run);
  assert.match(none,/CPU · agent sandboxes<\/small><strong>Unavailable/);assert.match(none,/No sandbox recorded for these agents/);
  setMetricScope(null);press([],{metricCategory:'Activity'});
});
