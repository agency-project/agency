// Presentation helpers for captured evidence. These never change execution or
// derive agent health / model understanding from an individual call result.
export const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const text=value=>typeof value==='string'?value:value==null?'':JSON.stringify(value,null,2);
const parsed=value=>{if(typeof value!=='string'||value.length>200000)return value;try{return JSON.parse(value);}catch{return value;}};
const meaningful=value=>{const valueText=text(value).trim();return valueText&&!['{}','[]','null'].includes(valueText)?valueText:'';};
const clipped=(value,limit=220)=>{const valueText=text(value);return valueText.length>limit?valueText.slice(0,limit-1)+'…':valueText;};
const modelCall=action=>['model','model_request'].includes(action?.kind);
const controlCall=action=>action?.kind==='control'||action?.kind==='intervention';
const time=value=>!finite(value)?'Unavailable':value<1?`${Math.round(value*1000)}ms`:value<60?`${value.toFixed(1)}s`:`${Math.floor(Math.round(value)/60)}m ${Math.round(value)%60}s`;

export function payloadValue(action,kind) {
  if(kind==='input'){
    if(modelCall(action)&&Array.isArray(action.context?.blocks)&&action.context.blocks.length)
      return action.context.blocks.map(block=>block.preview??text(block)).join('\n\n');
    return meaningful(action.arguments)||meaningful(action.command)||meaningful(action.input)||'';
  }
  const error=meaningful(action.error),result=(modelCall(action)&&meaningful(action.model_response))||meaningful(action.result)||meaningful(action.output);
  return error&&result&&error!==result?`Recorded error\n${error}\n\nCaptured output\n${result}`:error||result||'';
}

export function payloadWindow(value,{offset=0,limit=4000}={}) {
  const raw=text(value);
  const size=Math.min(32000,Math.max(1,Number.isFinite(limit)?Math.floor(limit):4000));
  const start=Math.min(raw.length,Math.max(0,Number.isFinite(offset)?Math.floor(offset):0));
  const end=Math.min(raw.length,start+size);
  return {text:raw.slice(start,end),offset:start,end,total:raw.length,previous:start>0,next:end<raw.length};
}

export function renderPayloadWindow(value,options={}) {
  const window=payloadWindow(value,options);
  return `<pre class="payload-text" tabindex="0" aria-label="Captured text">${escapeHTML(window.text)}</pre>${window.previous||window.next?`<p class="muted">Characters ${window.offset+1}–${window.end} of ${window.total}. Additional captured text is available.</p>`:''}`;
}

// JSON is re-indented token by token, never parsed and re-serialised: numbers,
// escapes, key order and duplicate keys keep their recorded spelling. Only a
// complete, valid JSON object or array qualifies; anything else returns null.
export function prettyJson(value) {
  if(typeof value!=='string'||value.length>200000)return null;
  const source=value.trim();
  if(source[0]!=='{'&&source[0]!=='[')return null;
  try{JSON.parse(source);}catch{return null;}
  let out='',depth=0;
  const line=()=>'\n'+'  '.repeat(depth),blank=char=>char===' '||char==='\n'||char==='\t'||char==='\r';
  for(let i=0;i<source.length;i++){
    const char=source[i];
    if(char==='"'){let end=i+1;while(source[end]!=='"')end+=source[end]==='\\'?2:1;out+=source.slice(i,end+1);i=end;}
    else if(char==='{'||char==='['){
      let next=i+1;while(blank(source[next]))next++;
      if(source[next]===(char==='{'?'}':']')){out+=char+source[next];i=next;}else{depth++;out+=char+line();}
    }
    else if(char==='}'||char===']'){depth--;out+=line()+char;}
    else if(char===',')out+=','+line();
    else if(char===':')out+=': ';
    else if(!blank(char))out+=char;
  }
  return out;
}

// Every character is escaped, including text outside a recognised token: a
// clipped preview can end inside a string.
export function highlightJson(value) {
  const source=String(value??'');let html='',last=0;
  for(const match of source.matchAll(/"(?:\\.|[^"\\])*"(\s*:)?|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)){
    const token=match[0],kind=token[0]==='"'?(match[1]?'key':'string'):/^[tfn]/i.test(token)?'literal':'number';
    html+=escapeHTML(source.slice(last,match.index))+`<span class="json-${kind}">${escapeHTML(token)}</span>`;last=match.index+token.length;
  }
  return html+escapeHTML(source.slice(last));
}

export function actionState(action) {
  const status=action.request_state||action.outcome||'unknown';
  const type=modelCall(action)?'Request':controlCall(action)?'Control':['queue','dependency','setup'].includes(action.kind)?'Operation':'Tool';
  const states={running:['◷',`${type} running`,'running'],success:['✓',`${type} completed`,'success'],completed:['✓',`${type} completed`,'success'],failed:['!',`${type} failed`,'failed'],failure:['!',`${type} failed`,'failed'],cancelled:['■',`${type} cancelled`,'cancelled'],incomplete:['?',`${type} incomplete`,'unknown'],queued:['◷',controlCall(action)?'Context queued':'Request queued','queue'],context_committed:['↳','Retained context committed · model consumption unknown','success'],submitted:['↳','Invocation submitted · completion unconfirmed','queue'],blocked:['◷','Request blocked','queue'],requested:['◷','Requested · not acknowledged','queue'],accepted:['↳','Accepted · application unconfirmed','queue'],observed:['✓','Local outcome observed','success'],applied:['✓','Application recorded','success'],delivered:['↳','Input delivered · response unconfirmed','success'],partial:['!','Partial outcome','failed'],unknown:['?','Outcome unknown','unknown']};
  const [icon,label,semantic]=states[status]||['?',`${type} state: ${status}`,'unknown'];
  return {status,icon,label,semantic};
}

function toolCommand(action) {
  const args=parsed(action.arguments);
  if(args&&typeof args==='object'&&!Array.isArray(args)){
    for(const key of ['command','cmd','script','message','query','path','file_path','url'])if(meaningful(args[key]))return text(args[key]);
  }
  return meaningful(action.command)||meaningful(action.arguments)||meaningful(action.intent)||'Arguments not captured';
}

function resultPreview(action) {
  if(meaningful(action.error))return clipped(action.error);
  const result=parsed(action.result??action.output);
  if(result&&typeof result==='object'&&!Array.isArray(result)){
    const code=result.exit_code??result.returncode;
    const recordedError=result.error||(result.isError?result.message:'');
    const lines=text(recordedError||result.stderr||result.stdout||result.output).split('\n').map(line=>line.trim()).filter(Boolean);
    const specific=lines.filter(line=>/AssertionError|^FAILED\b|\bError:|^FAIL\b/.test(line));
    const output=(specific.length?specific:lines.slice(-2)).join(' · ');
    if(code!=null)return clipped(`Exit ${code}${output?' · '+output:''}`);
    if(output)return clipped(output);
  }
  return clipped(meaningful(action.result_preview)||meaningful(action.result)||meaningful(action.output));
}

export function eventCard(action,run={}) {
  const state=actionState(action),agent=(run.agents||[]).find(item=>item.id===action.agent);
  const identity=[
    ...[action.model,action.metadata?.model,action.metadata?.['llm.model']].map(value=>({value,source:'event'})),
    {value:agent?.model,source:'agent metadata'},{value:run.model,source:'execution metadata'},
  ].find(item=>meaningful(item.value)&&!['Unavailable','Recorded in event metadata','Unknown','unknown','n/a'].includes(item.value));
  const model=identity?.value,modelSource=identity?.source||'unavailable';
  const usage=[];
  if(modelCall(action)){
    if(finite(action.tokens)&&action.tokens>=0)usage.push(`${action.tokens.toLocaleString('en')} input tokens`);
    if(finite(action.output_tokens)&&action.output_tokens>=0)usage.push(`${action.output_tokens.toLocaleString('en')} output tokens`);
  }
  const elapsed=action.missing_start||String(action.source||'').includes('timestamp_missing')?null:finite(action.duration)&&action.duration>=0?action.duration:null;
  let primary,secondary,preview=resultPreview(action),previewLabel='Captured result';
  if(modelCall(action)){
    primary=model?modelSource==='event'?String(model):`Configured ${clipped(model,220)} · request identity unrecorded`:'Model identity unavailable';secondary=modelSource==='event'||!model?'Model request':`Model request · ${modelSource}`;previewLabel='Emitted output';
    if(!preview)preview=Array.isArray(action.tool_call_ids)&&action.tool_call_ids.length?`${action.tool_call_ids.length} recorded tool request${action.tool_call_ids.length===1?'':'s'}`:'No emitted text captured';
  }else if(controlCall(action)){
    const requested=action.requested_action||action.control_action||action.name||'Recorded intervention';
    const target=action.target_agent||action.target||action.agent||'Target not captured';
    primary=meaningful(action.input)||meaningful(action.command)||String(requested);secondary=`${requested} → ${target}`;previewLabel='Recorded outcome';
    const args=parsed(action.arguments);
    if(requested==='run'&&args&&typeof args==='object')primary=`Run task ${args.task||'identity unavailable'}${meaningful(args.input)?' · '+text(args.input):''}`;
    if(!preview)preview=state.label;
  }else{
    primary=toolCommand(action);secondary=action.name?.replace(/^tool:/,'')||'Tool name unavailable';
    if(action.error||state.semantic==='failed')previewLabel='Recorded call failure';
    if(!preview)preview=state.status==='running'?'Result not captured yet':'No result captured';
  }
  // A command or argument set recorded as JSON is shown formatted, like every other payload.
  const formatted=prettyJson(primary);
  const type=modelCall(action)?'model':controlCall(action)?'control':['queue','dependency','setup'].includes(action.kind)?'operation':'tool';
  return {primary:clipped(formatted??primary,360),json:formatted!=null,type,typeLabel:{model:'Model',control:'Control',operation:'Operation',tool:'Tool'}[type],secondary,state,elapsed,elapsedText:elapsed==null?'Elapsed unavailable':`${state.status==='running'?'Elapsed':'Wall interval'} ${time(elapsed)}`,preview,previewLabel,usage,modelSource,scope:modelCall(action)?'model request':controlCall(action)?'human intervention':'tool call'};
}

export function renderEventCard(action,run={}) {
  const card=eventCard(action,run);
  return `<span class="call-kind">${escapeHTML(card.secondary)}</span><strong class="call-meaning${card.json?' json-label':''}">${card.json?highlightJson(card.primary):escapeHTML(card.primary)}</strong><span class="call-state ${escapeHTML(card.state.semantic)}">${escapeHTML(card.state.icon)} ${escapeHTML(card.state.label)} · ${escapeHTML(card.elapsedText)}</span><span class="call-result"><span class="muted">${escapeHTML(card.previewLabel)}: </span>${escapeHTML(card.preview)}</span>${card.usage.length?`<span class="call-usage">${escapeHTML(card.usage.join(' · '))}</span>`:''}`;
}

export function eventRelationships(action,run={}) {
  const actions=run.actions||[],links=new Map(),unresolved=[];
  const link=(candidate,relation,evidence)=>{
    if(candidate&&candidate.id!==action.id&&candidate.agent===action.agent)links.set(candidate.id,{id:candidate.id,agent:candidate.agent,label:eventCard(candidate,run).primary,relation,evidence});
  };
  if(modelCall(action)){
    const ids=new Set(Array.isArray(action.tool_call_ids)?action.tool_call_ids:[]);
    for(const candidate of actions)if(candidate.agent===action.agent&&!modelCall(candidate)&&!controlCall(candidate)){
      if(candidate.model_id===action.id)link(candidate,'Requested tool execution','Recorded originating model action ID');
      else if(candidate.model_tool_call_id&&ids.has(candidate.model_tool_call_id))link(candidate,'Requested tool execution','Matching recorded tool-call ID');
    }
    for(const id of ids)if(!actions.some(candidate=>candidate.agent===action.agent&&candidate.model_tool_call_id===id))unresolved.push(id);
  }else if(!controlCall(action)){
    if(action.model_id){
      const origin=actions.find(candidate=>candidate.id===action.model_id&&candidate.agent===action.agent&&modelCall(candidate));
      if(origin)link(origin,'Originating model response','Recorded originating model action ID');else unresolved.push(action.model_id);
    }else if(action.model_tool_call_id){
      const origins=actions.filter(candidate=>candidate.agent===action.agent&&modelCall(candidate)&&candidate.tool_call_ids?.includes(action.model_tool_call_id));
      if(origins.length===1)link(origins[0],'Originating model response','Matching recorded tool-call ID');else unresolved.push(action.model_tool_call_id);
    }
  }
  return {links:[...links.values()],unresolved,unavailable:links.size?'':unresolved.length?'Explicit correlation recorded; linked evidence is unavailable or ambiguous in this window.':'No explicit model/tool correlation recorded. Timing and adjacency do not establish a relationship.'};
}

function sampleScope(name) {
  if(/^(workload|workload_total):/.test(name)||name.startsWith('workload_total '))return 'Workload-wide; not an agent measurement';
  if(name.startsWith('sandbox:'))return `Sandbox ${name.split(':')[1]}; not an operation measurement`;
  if(name.startsWith('process:'))return `Process ${name.split(':')[1]}; agent attribution not recorded`;
  if(/^(host|system):/.test(name))return 'Host-wide; not an agent measurement';
  if(/^gpu\d+:/.test(name))return `Device ${name.split(':')[0]}; agent attribution not recorded`;
  return 'Recorded series; scope not recorded';
}

function sampleUnit(name) {
  if(/(?:_pct|percent|cpu %)\b/.test(name))return '%';
  if(/(?:_mb_s|MB\/s)\b/.test(name))return 'MB/s';
  if(/(?:_mb|memory_mb|MB)\b/.test(name))return 'MB';
  if(/(?:_ms|milliseconds)\b/.test(name))return 'ms';
  if(/(?:_w|_w_est)\b/.test(name))return 'W';
  return 'unit not recorded';
}

export function eventMeasurements(action,run={}) {
  const startUnavailable=action.missing_start||String(action.source||'').includes('timestamp_missing');
  const rows=[],start=!startUnavailable&&finite(action.start)?action.start:null;
  const duration=!startUnavailable&&finite(action.duration)&&action.duration>=0?action.duration:null;
  const end=start!=null&&duration!=null?start+duration:start;
  const provenance=run.source==='synthetic'?'synthetic example':'recorded';
  if(duration!=null)rows.push({key:'wall',label:'Wall interval',value:duration,unit:'s',scope:'This call',provenance:action.source==='reconstructed_trace_boundary'?'reconstructed from recorded span':action.timing==='host_event_boundary'?`calculated from ${provenance} boundaries`:provenance,interval:start==null?null:[start,end],note:'This interval is not a sum of concurrent call durations.'});
  if(!startUnavailable&&finite(action.execution_duration)&&action.execution_duration>=0)rows.push({key:'active',label:'Interval excluding recorded pauses',value:action.execution_duration,unit:'s',scope:'This call',provenance:`calculated from ${provenance} boundaries and pause intervals`,interval:start==null?null:[start,end],note:'Local pause intervals do not establish remote operation suspension.'});
  const intervalMetrics={cpu_ms:'Thread CPU',runqueue_ms:'Runqueue',blocked_ms:'Blocked'};
  for(const [key,label] of Object.entries(intervalMetrics))if(finite(action.metadata?.[key]))rows.push({key,label,value:action.metadata[key],unit:'ms',scope:'Recorded operation interval',provenance,interval:start==null?null:[start,end],note:'Recorded interval metric; overlapping spans are not exclusive work.'});
  if(modelCall(action))for(const [key,label] of [['tokens','Input tokens'],['output_tokens','Output tokens']])if(finite(action[key])&&action[key]>=0)rows.push({key,label,value:action[key],unit:'tokens',scope:'This model request',provenance,interval:start==null?null:[start,end],note:'Reported usage; absent usage is unavailable.'});
  const samples=[];
  if(start!=null)for(const [name,points] of Object.entries(run.counters||{})){
    let nearest=null;
    // Never substitute a later, current host sample for a completed call.
    for(const point of Array.isArray(points)?points:[])if(Array.isArray(point)&&finite(point[0])&&finite(point[1])&&point[0]<=end&&(!nearest||Math.abs(point[0]-start)<Math.abs(nearest[0]-start)||(Math.abs(point[0]-start)===Math.abs(nearest[0]-start)&&point[0]<nearest[0])))nearest=point;
    if(nearest)samples.push({key:name,label:name,value:nearest[1],unit:sampleUnit(name),scope:sampleScope(name),provenance,at:nearest[0],delta:Math.abs(nearest[0]-start),note:'Nearest recorded sample to call start, no later than this call interval; scope and units follow the recorded series name, proximity is not operation attribution.'});
  }
  return {rows,samples,coverage:[startUnavailable||start==null?'Start measurement unavailable; resource samples are not attributed to a call-start time.':'',!rows.some(row=>intervalMetrics[row.key])?'No thread CPU / runqueue / blocked measurements recorded.':'',!samples.length?'No recorded resource samples available at or before this call interval.':''].filter(Boolean).join(' ')};
}
