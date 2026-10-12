// An inspectable, keyboard-accessible destructive confirmation. No mutation
// happens until the captured target passes validation again at confirmation.
let active=null;

export function refreshControlConfirmation() {
  if(!active)return;
  const reason=active.options.validate?.()||'';
  active.submit.disabled=Boolean(reason);
  active.status.textContent=reason||'Review the targets before confirming.';
}

export function requestControlConfirmation(options) {
  if(active){active.cancel.focus();return;}
  const origin=document.activeElement,dialog=document.createElement('dialog');
  const originId=origin?.id,focusAttribute=['data-execution-stop','data-execution-kill'].find(name=>origin?.hasAttribute?.(name)),focusTarget=focusAttribute?origin.getAttribute(focusAttribute):null;
  dialog.className='control-confirmation';
  dialog.setAttribute('aria-labelledby','control-confirmation-title');
  dialog.setAttribute('aria-describedby','control-confirmation-consequence');
  dialog.innerHTML='<h2 id="control-confirmation-title"></h2><p class="confirmation-execution"></p><ul class="confirmation-targets" tabindex="0"></ul><p id="control-confirmation-consequence"></p><p class="confirmation-status" role="status" aria-live="polite"></p><div class="confirmation-actions"><button type="button" data-confirm-cancel>Cancel</button><button type="button" data-confirm-submit></button></div>';
  dialog.querySelector('h2').textContent=`${options.force?'Force stop':'Stop'} ${options.targets.length} agent${options.targets.length===1?'':'s'}?`;
  dialog.querySelector('.confirmation-execution').textContent=`Execution: ${options.execution}${options.frozen?' · view frozen; targets are CURRENT LIVE agents':''}`;
  for(const target of options.targets){const item=document.createElement('li');item.textContent=target;dialog.querySelector('ul').append(item);}
  dialog.querySelector('#control-confirmation-consequence').textContent=options.force?'Cancel outstanding work and send SIGKILL. Process exit must be observed separately; cleanup is not guaranteed.':'Cancel outstanding work and send SIGTERM. A process may ignore it; exit and cleanup are not guaranteed.';
  const cancel=dialog.querySelector('[data-confirm-cancel]'),submit=dialog.querySelector('[data-confirm-submit]');
  submit.textContent=options.force?'Force stop · SIGKILL':'Stop · SIGTERM';
  let resolved=false;
  const finish=confirmed=>{
    if(resolved)return;
    if(confirmed){refreshControlConfirmation();if(submit.disabled)return;}
    resolved=true;active=null;dialog.close();dialog.remove();
    if(origin?.isConnected)origin.focus({preventScroll:true});
    else {
      const replacement=originId?document.getElementById(originId):focusAttribute?[...document.querySelectorAll(`[${focusAttribute}]`)].find(el=>el.getAttribute(focusAttribute)===focusTarget):null;
      replacement?.focus({preventScroll:true});
    }
    if(confirmed)options.onConfirm();
  };
  active={options,cancel,submit,status:dialog.querySelector('.confirmation-status')};
  cancel.addEventListener('click',()=>finish(false));
  submit.addEventListener('click',()=>finish(true));
  dialog.addEventListener('cancel',event=>{event.preventDefault();finish(false);});
  // Enter while focus is on the target list must not become an implicit submit.
  dialog.addEventListener('keydown',event=>{if(event.key==='Enter'&&event.target!==submit&&event.target!==cancel)event.preventDefault();});
  document.body.append(dialog);dialog.showModal();refreshControlConfirmation();cancel.focus();
}
