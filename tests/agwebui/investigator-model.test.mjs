import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
const source = await readFile(new URL('../../agency/observability/agwebui/static/investigator-model.js', import.meta.url),'utf8');
const {alignSequences, overlapDuration, matchesAction, packActionTracks} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('insertion identifies divergence and the next matched reconvergence',()=>{
  const left=['search','read','edit','test'].map(kind=>({kind}));
  const right=['search','read','read','edit','test'].map(kind=>({kind}));
  const rows=alignSequences(left,right);
  assert.equal(rows.filter(r=>!r.match).length,1);
  assert.equal(rows[2].left,null);
  assert.equal(rows[3].reconverged,true);
});
test('deletion and empty sequences retain all unmatched actions',()=>{
  assert.equal(alignSequences([{kind:'edit'}],[])[0].right,null);
  assert.equal(alignSequences([],[{kind:'test'}])[0].left,null);
  assert.deepEqual(alignSequences([],[]),[]);
});
test('parallel nested model spans do not double count wall time',()=>{
  const intervals=[{start:1,duration:5},{start:2,duration:2},{start:5,duration:4}];
  assert.equal(overlapDuration(intervals,0,10),8);
  assert.equal(overlapDuration(intervals,3,7),4);
});
test('search and agent filters use the same event projection',()=>{
  const action={agent:'worker',name:'Bash',intent:'Locate scheduler',command:'rg dispatch',result:'found',files:['orchestrator.py']};
  assert(matchesAction(action,'worker','ORCHESTRATOR'));
  assert(!matchesAction(action,'supervisor','scheduler'));
  assert(matchesAction(action,'','found'));
});
test('nested actions stay visible and adjacent actions reuse a track',()=>{
  const actions=[{id:'next',start:10,duration:2},{id:'nested',start:2,duration:3},{id:'long',start:0,duration:10},{id:'parallel',start:3,duration:1}];
  const tracks=packActionTracks(actions);
  assert.deepEqual(tracks.map(track=>track.map(a=>a.id)),[['long','next'],['nested'],['parallel']]);
  assert.equal(actions[0].id,'next');
});
test('instant events reserve their visible width without losing actions',()=>{
  const actions=[{id:'one',start:0,duration:0},{id:'two',start:0,duration:0},{id:'three',start:1,duration:0}];
  assert.deepEqual(packActionTracks(actions,1).map(track=>track.map(a=>a.id)),[['one','three'],['two']]);
  assert.deepEqual(packActionTracks([]),[]);
});
