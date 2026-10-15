import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('../../agency/observability/agwebui/static/control-confirmation.js',import.meta.url),'utf8').replace(/^export /gm,'');
function fixture(){
  const nodes=[],origin={isConnected:true,focus(){document.activeElement=this;}};
  const document={activeElement:origin,body:{append(el){nodes.push(el);}},createElement(tag){
    const el={tag,textContent:'',children:[],parts:new Map(),events:{},setAttribute(k,v){this[k]=v;},append(child){this.children.push(child);},querySelector(selector){if(!this.parts.has(selector))this.parts.set(selector,document.createElement(selector));return this.parts.get(selector);},addEventListener(event,fn){this.events[event]=fn;},focus(){document.activeElement=this;},showModal(){this.open=true;},close(){this.open=false;},remove(){nodes.splice(nodes.indexOf(this),1);}};
    return el;
  },getElementById:()=>null,querySelectorAll:()=>[]};
  const context=vm.createContext({document});vm.runInContext(source,context);
  let confirmations=0,reason='';
  const options={force:true,execution:'Execution <script>',targets:['Agent <b>','Agent B'],frozen:true,validate:()=>reason,onConfirm:()=>confirmations++};
  return {document,nodes,origin,context,options,get confirmations(){return confirmations;},setReason:value=>reason=value,open(){context.requestControlConfirmation(options);return nodes[0];}};
}

test('dialog is labelled, bounded by targets, safe text, and defaults focus to Cancel',()=>{
  const f=fixture(),dialog=f.open();
  assert.equal(dialog['aria-labelledby'],'control-confirmation-title');
  assert.equal(dialog.querySelector('h2').textContent,'Force stop 2 agents?');
  assert.match(dialog.querySelector('.confirmation-execution').textContent,/current live/i);
  assert.equal(dialog.querySelector('ul').children[0].textContent,'Agent <b>');
  assert.match(dialog.querySelector('#control-confirmation-consequence').textContent,/SIGKILL.*cleanup is not guaranteed/);
  assert.equal(f.document.activeElement,dialog.querySelector('[data-confirm-cancel]'));
  f.context.requestControlConfirmation(f.options);assert.equal(f.nodes.length,1);
  let prevented=false;dialog.events.keydown({key:'Enter',target:dialog.querySelector('ul'),preventDefault:()=>prevented=true});assert.equal(prevented,true);assert.equal(f.confirmations,0);
});

test('Escape cancels without mutation and restores focus',()=>{
  const f=fixture(),dialog=f.open();let prevented=false;
  dialog.events.cancel({preventDefault:()=>prevented=true});
  assert.equal(prevented,true);assert.equal(f.confirmations,0);assert.equal(f.nodes.length,0);assert.equal(f.document.activeElement,f.origin);
});

test('a changed target disables confirmation and validation is repeated at submission',()=>{
  const f=fixture(),dialog=f.open();f.setReason('Execution changed');f.context.refreshControlConfirmation();
  const button=dialog.querySelector('[data-confirm-submit]');assert.equal(button.disabled,true);assert.equal(dialog.querySelector('.confirmation-status').textContent,'Execution changed');
  button.events.click();assert.equal(f.confirmations,0);assert.equal(f.nodes.length,1);
  f.setReason('');button.events.click();button.events.click();assert.equal(f.confirmations,1);assert.equal(f.nodes.length,0);
});

test('ordinary stop confirms SIGTERM consequences separately from force stop',()=>{
  const f=fixture();f.options.force=false;const dialog=f.open();
  assert.match(dialog.querySelector('#control-confirmation-consequence').textContent,/SIGTERM.*may ignore/);
  assert.equal(dialog.querySelector('[data-confirm-submit]').textContent,'Stop · SIGTERM');
});
