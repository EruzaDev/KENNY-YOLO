// Isolated browser-logic tests. Landmark fixtures test flow, not model accuracy.
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
async function moduleFrom(path) {
  return import(`data:text/javascript;base64,${Buffer.from(await readFile(path)).toString('base64')}`);
}
const {HandGate, handBounds, noHandResult} = await moduleFrom('frontend/js/hand_gate.js');
const {Camera} = await moduleFrom('frontend/js/camera.js');

test('missing or multiple hands do not create classifier input', async () => {
  const gate = new HandGate();
  gate.detector = {detect: () => ({landmarks: []})};
  assert.equal((await gate.crop({width:640,height:480})).image, null);
  gate.detector = {detect: () => ({landmarks:[[],[]]})};
  assert.equal((await gate.crop({width:640,height:480})).reason, 'Show one hand at a time');
  assert.deepEqual(noHandResult().predictions, []);
  assert.equal(noHandResult().stable_gesture, null);
});

test('hand crop includes all landmarks with bounded padding', () => {
  const points = Array.from({length:21}, (_,i) => ({x:.2 + i*.02, y:.3 + i*.02}));
  const box = handBounds(points,640,480);
  assert.ok(box.x <= .2*640 && box.y <= .3*480);
  assert.ok(box.x + box.width >= .6*640 && box.y + box.height >= .7*480);
  assert.ok(box.x >= 0 && box.y >= 0 && box.x+box.width <= 640 && box.y+box.height <= 480);
  assert.equal(handBounds([],640,480),null);
  assert.equal(handBounds(points.map(() => ({x:NaN,y:0})),640,480),null);
});

test('no-hand camera frame sends reset instead of arbitrary scene pixels', async () => {
  globalThis.WebSocket = {OPEN:1};
  const messages=[];
  const video={videoWidth:640,videoHeight:480,srcObject:null};
  const camera=new Camera(video,()=>({confidence:.8,smoothing:3,frequency:5}),()=>{},()=>{},()=>{},
    {crop:async()=>({image:null,reason:'No hand detected'})});
  camera.socket={readyState:1,send:message=>messages.push(JSON.parse(message)),close:()=>{}};
  camera.stream={getTracks:()=>[]};
  await camera.send();
  camera.stop();
  assert.equal(messages.length,1);
  assert.equal(messages[0].type,'no_hand');
  assert.equal(messages[0].image,undefined);
});

test('camera stop discards a hand crop that completes late', async () => {
  globalThis.WebSocket = {OPEN:1};
  let resolve, sent=0;
  const pending=new Promise(r=>{resolve=r;});
  const camera=new Camera({videoWidth:640,videoHeight:480},()=>({}),()=>{},()=>{},()=>{},
    {crop:()=>pending});
  camera.socket={readyState:1,send:()=>{sent++;},close:()=>{}};
  camera.stream={getTracks:()=>[]};
  const operation=camera.send();
  camera.stop();
  resolve({image:{toDataURL:()=>{throw new Error('Should not read a stale crop');}}});
  await operation;
  assert.equal(sent,0);
});

test('unavailable hand detector fails closed without creating a prediction', async () => {
  const gate = new HandGate();
  // Node intentionally cannot load this browser HTTPS module; exercise the error path.
  await assert.rejects(gate.crop({width:640,height:480}), /Recognition is paused/);
  assert.equal(gate.detector,null);
});

const {thumbEvidence, okPinchEvidence, checkOkEvidence} = await moduleFrom('frontend/js/hand_gate.js');
function foldedHand(extendedThumb) {
  const points=Array.from({length:21},()=>({x:0,y:0,z:0}));
  for (const [j,x] of [-.04,-.015,.015,.04].entries()) {
    const i=5+j*4;
    [[.05,0],[.09,0],[.07,.02],[.04,.03]].forEach(([y,z],k)=>{points[i+k]={x,y,z};});
  }
  points[1]={x:-.04,y:.02,z:0};points[2]={x:-.06,y:.04,z:0};
  points[3]=extendedThumb ? {x:-.08,y:.08,z:0} : {x:-.05,y:.07,z:0};
  points[4]=extendedThumb ? {x:-.10,y:.12,z:0} : {x:-.005,y:.045,z:0};
  return points;
}
test('thumb evidence distinguishes clear folded/extended geometry, independent of mirror and scale',()=>{
  for (const [extended,label] of [[true,'thumb_extended'],[false,'thumb_folded']]) {
    const points=foldedHand(extended);
    assert.equal(thumbEvidence(points),label);
    assert.equal(thumbEvidence(points.map(p=>({x:-p.x*20,y:p.y*20,z:p.z*20}))),label);
  }
  assert.equal(thumbEvidence(undefined),'unknown');
  assert.equal(thumbEvidence(Array(21).fill({x:0,y:0,z:0})),'unknown');
  const open=foldedHand(true);
  for (const i of [5,9,13,17]) for(let k=0;k<4;k++) open[i+k]={x:.02,y:.05+k*.04,z:0};
  assert.equal(thumbEvidence(open),'unknown');
});
test('camera thumb check stays optional and sends evidence only when enabled',async()=>{
  globalThis.WebSocket={OPEN:1};
  for(const enabled of [true,false]) {
    const messages=[];
    const camera=new Camera({videoWidth:640,videoHeight:480},()=>({confidence:.8,smoothing:3,thumbGuard:enabled}),()=>{},()=>{},()=>{},
      {crop:async()=>({image:{toDataURL:()=> 'data:image/jpeg;base64,eA=='},thumbEvidence:'thumb_folded'})});
    camera.stream={getTracks:()=>[]};
    camera.socket={readyState:1,send:x=>messages.push(JSON.parse(x)),close:()=>{}};
    await camera.send();camera.stop();
    assert.equal(messages[0].pose_hint,enabled?'thumb_folded':'unknown');
  }
});

test('separated fingertips veto an OK score without assigning another class', () => {
  const points = foldedHand(true);
  assert.equal(okPinchEvidence(points), 'separate');
  assert.equal(okPinchEvidence(points.map(p=>({x:-p.x*10,y:p.y*10,z:p.z*10}))), 'separate');
  points[4] = {...points[8]};
  assert.equal(okPinchEvidence(points), 'pinched');
  assert.equal(okPinchEvidence(undefined), 'unknown');
  assert.equal(okPinchEvidence(Array(21).fill({x:0,y:0,z:0})), 'unknown');
  const ambiguous = Array.from({length:21},()=>({x:0,y:0,z:0}));
  ambiguous[9]={x:0,y:.1,z:0}; ambiguous[4]={x:-.1,y:.1,z:0}; ambiguous[8]={x:.1,y:.1,z:0};
  assert.equal(okPinchEvidence(ambiguous), 'unknown'); // Distance alone must not veto OK.
  const result = {predictions:[{class_name:'ok',confidence:.999}], scores:{ok:.999}, stable_gesture:'ok'};
  const rejected = checkOkEvidence(result, 'separate');
  assert.deepEqual(rejected.predictions, []);
  assert.equal(rejected.stable_gesture, null);
  assert.deepEqual(rejected.scores, result.scores);
  assert.equal(checkOkEvidence(result, 'pinched'), result);
  assert.equal(checkOkEvidence(result, 'unknown'), result);
  const palm = {predictions:[{class_name:'open_palm'}]};
  assert.equal(checkOkEvidence(palm, 'separate'), palm);
});

test('saved camera example matches the classified request and clears on stop',async()=>{
  const originalNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator');
  const originalWebSocket=globalThis.WebSocket, originalLocation=globalThis.location;
  class Socket {
    static OPEN=1;
    constructor(){this.readyState=1;this.messages=[];}
    send(message){this.messages.push(JSON.parse(message));}
    close(){}
  }
  try {
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{mediaDevices:{getUserMedia:async()=>({
      getVideoTracks:()=>[{}], getTracks:()=>[]})}}});
    globalThis.WebSocket=Socket;globalThis.location={protocol:'http:',host:'testserver'};
    let shown;
    const camera=new Camera({videoWidth:640,videoHeight:480,play:async()=>{}},
      ()=>({confidence:.8,smoothing:3,frequency:5}),result=>{shown=result;},()=>{},()=>{},
      {load:async()=>{},filterResult:checkOkEvidence,crop:async()=>({image:{toDataURL:()=> 'data:image/jpeg;base64,ZXhhY3Q='},
        box:{x:40,y:50,width:100,height:120}})});
    await camera.start();
    await camera.send();
    const result={predictions:[{class_name:'thumbs_up',confidence:.99}],scores:{thumbs_up:.99},latency_ms:1};
    camera.socket.onmessage({data:JSON.stringify(result)});
    assert.equal(camera.lastFrame.image,camera.socket.messages[0].image);
    assert.equal(camera.lastFrame.image,'ZXhhY3Q=');
    assert.deepEqual(camera.lastFrame.result,shown);
    assert.deepEqual(shown.hand_box,{x:40,y:50,width:100,height:120});
    assert.equal(shown.source_width,640);
    assert.equal(shown.source_height,480);
    assert.equal(camera.socket.messages[0].hand_box,undefined);
    assert.equal(camera.lastFrame.session,camera.sessionId);
    camera.pendingFrame = {...camera.lastFrame, okEvidence:'separate'};
    camera.socket.onmessage({data:JSON.stringify({predictions:[{class_name:'ok',confidence:.999}],
      scores:{ok:.999},stable_gesture:'ok'})});
    assert.deepEqual(shown.predictions, []);
    assert.equal(shown.stable_gesture, null);
    assert.equal(camera.lastFrame.result, shown);
    camera.stop();
    assert.equal(camera.lastFrame,null);
    assert.equal(camera.pendingFrame,null);
  } finally {
    if(originalNavigator) Object.defineProperty(globalThis,'navigator',originalNavigator);
    else delete globalThis.navigator;
    globalThis.WebSocket=originalWebSocket;globalThis.location=originalLocation;
  }
});
