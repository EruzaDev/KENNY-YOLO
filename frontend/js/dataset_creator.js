import {gestureHint} from './hand_gate.js?v=2';
const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay');
let landmarker, stream, running = false, saved = 0, skipped = 0, lastCapture = 0;
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
async function loadDetector() {
  if (landmarker) return;
  $('status').textContent = 'Loading MediaPipe hand model…';
  const base = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
  const {FilesetResolver, HandLandmarker} = await import(`${base}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${base}/wasm`);
  landmarker = await HandLandmarker.createFromOptions(files, {baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5});
}
function paint(points, width, height) {
  const ctx=overlay.getContext('2d'); overlay.width=width; overlay.height=height;
  ctx.clearRect(0,0,width,height); const xs=points.map(p=>p.x*width),ys=points.map(p=>p.y*height);
  const pad=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))*.25;
  const x=Math.max(0,Math.min(...xs)-pad),y=Math.max(0,Math.min(...ys)-pad),r=Math.min(width,Math.max(...xs)+pad),b=Math.min(height,Math.max(...ys)+pad);
  ctx.strokeStyle='#b9f567';ctx.lineWidth=Math.max(2,width/500);ctx.strokeRect(x,y,r-x,b-y);
  ctx.strokeStyle='#72e0c0';ctx.lineWidth=Math.max(2,width/600);
  for(const [a,c] of HAND_EDGES){ctx.beginPath();ctx.moveTo(points[a].x*width,points[a].y*height);ctx.lineTo(points[c].x*width,points[c].y*height);ctx.stroke();}
  points.forEach(p=>{ctx.beginPath();ctx.arc(p.x*width,p.y*height,Math.max(3,width/180),0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='#153a2e';ctx.lineWidth=2;ctx.stroke();});
  return {x:x/width,y:y/height,width:(r-x)/width,height:(b-y)/height};
}
async function persist(points, box, label) {
  const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  const ctx=canvas.getContext('2d');ctx.drawImage(video,0,0);
  // Keep the training image clean; live joints/box remain visible in the preview,
  // and this same geometry is retained in the paired JSON annotation.
  const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.9));
  const form=new FormData();form.append('image',blob,'capture.jpg');form.append('label',label);
  form.append('annotation',JSON.stringify({label,bbox:box,landmarks:points.map(p=>({x:p.x,y:p.y,z:p.z})),captured_at:new Date().toISOString(),width:canvas.width,height:canvas.height,annotator:'MediaPipe Hand Landmarker'}));
  const response=await fetch('/api/dataset/capture',{method:'POST',body:form});if(!response.ok)throw Error((await response.json()).detail||'Could not save capture');
}
async function loop() {
  while(running) {
    const started=performance.now(), now=started;
    if(video.readyState>=2) {
      const result=landmarker.detectForVideo(video,now), points=result.landmarks?.[0];
      if(points) {
        const box=paint(points,video.videoWidth,video.videoHeight), pose=gestureHint(points,video.videoWidth,video.videoHeight), label=$('label').value;
        $('status').textContent=pose!=='unknown'?`Detected ${pose.replace('_',' ')}${pose===label?' · saving…':' · hold '+label.replace('_',' ')}`:'Hand seen · pose unclear; spread fingers and keep the full hand in view';
        if(pose===label && now-lastCapture>=333 && saved<+$('count').value) {
          lastCapture=now;
          try { await persist(points,box,label); saved++; }
          catch(e){$('status').textContent=e.message; running=false; break;}
          $('bar').style.width=`${Math.min(100,saved/(+$('count').value)*100)}%`;
          $('saved').textContent=`${saved} saved · ${skipped} skipped`;
        }
      } else {overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);$('status').textContent='No hand detected · keep one hand in view';skipped++;}
    }
    if(saved>=+$('count').value){running=false;break;}
    await new Promise(resolve=>setTimeout(resolve,Math.max(0,100-(performance.now()-started))));
  }
  $('stop').disabled=true;$('start').disabled=false;
  if(saved>=+$('count').value)$('status').textContent=`Complete · saved ${saved} ${$('label').value.replace('_',' ')} images.`;
}
$('start').onclick=async()=>{
  try {
    $('start').disabled=true;
    if(!stream){await loadDetector();stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:15,max:24}},audio:false});video.srcObject=stream;await video.play();}
    saved=0;skipped=0;lastCapture=0;$('bar').style.width='0%';$('saved').textContent='0 saved · 0 skipped';running=true;$('stop').disabled=false;$('status').textContent='Show the selected gesture';loop();
  } catch(e){$('status').textContent=e.message;$('start').disabled=false;}
};
$('stop').onclick=()=>{running=false;$('status').textContent=`Stopped · saved ${saved} of ${$('count').value}.`};
window.addEventListener('pagehide',()=>stream?.getTracks().forEach(t=>t.stop()));
