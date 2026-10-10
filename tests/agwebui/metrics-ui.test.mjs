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
