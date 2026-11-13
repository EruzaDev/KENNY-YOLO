import {request} from './api.js';
import {Camera} from './camera.js';
import {HandGate, noHandResult} from './hand_gate.js';
const $ = id => document.getElementById(id);
let ready = false, mode = 'camera', objectURL = null, uploadVersion = 0, pending = null;
const settings = () => ({confidence: +$('confidence').value / 100, smoothing: +$('smoothing').value, frequency: +$('frequency').value, thumbGuard: $('thumb-guard').checked});
const error = message => { $('error').textContent = message; };
const handGate = new HandGate();
let gestureFunctions = {};
let gestureLabels = {};

function clearResult() {
  $('gesture').textContent = 'No gesture detected'; $('score').textContent = 'Waiting for an image';
  $('class-scores').textContent = '';
  $('function').textContent = 'Function: —';
  $('overlay').hidden = true; $('timing').textContent = '— ms · — FPS';
}
function display(result, fps = null) {
  error('');
  const name = 'stable_gesture' in result ? result.stable_gesture : result.predictions[0]?.class_name;
  const prediction = name && result.predictions.find(p => p.class_name === name);
  const label = name ? prediction?.display_name || gestureLabels[name] || name.replaceAll('_', ' ') : 'No gesture detected';
  $('gesture').textContent = label;
  const gestureFunction = prediction?.function || gestureFunctions[name];
  $('function').textContent = name && gestureFunction ? `Function: ${gestureFunction} (display only)` : 'Function: —';
  $('score').textContent = prediction ? `${(prediction.confidence * 100).toFixed(1)}% confidence · hand crop` : result.rejection_reason || 'Waiting for a stable prediction above threshold';
  $('overlay').hidden = false; $('overlay').textContent = `${label}${name && gestureFunction ? ` · ${gestureFunction}` : ''}${prediction ? ` · ${(prediction.confidence * 100).toFixed(1)}%` : ''}`;
  $('class-scores').textContent = Object.entries(result.scores || {})
    .filter(([,value]) => Number.isFinite(value))
    .map(([label,value]) => `${label.replaceAll('_', ' ')}: ${(value * 100).toFixed(1)}%`).join(' · ');
  $('timing').textContent = `${result.latency_ms} ms · ${fps ? fps.toFixed(1) : '—'} FPS`;
}
const camera = new Camera($('video'), settings, display, error, () => {
  $('start').disabled = !ready; $('stop').disabled = true; $('save-example').disabled = true;
  $('video').hidden = true; $('empty').hidden = false; clearResult();
}, handGate);
function switchMode(next) {
  camera.stop(); pending?.abort(); uploadVersion++; mode = next; error(''); clearResult();
  $('camera-tab').classList.toggle('active', next === 'camera'); $('upload-tab').classList.toggle('active', next === 'upload');
  $('camera-actions').hidden = next !== 'camera'; $('upload-actions').hidden = next !== 'upload';
  $('example-actions').hidden = next !== 'camera';
  $('preview').hidden = next !== 'upload' || !objectURL; $('empty').hidden = !$('preview').hidden;
}
$('camera-tab').onclick = () => switchMode('camera'); $('upload-tab').onclick = () => switchMode('upload');
$('start').onclick = async () => {
  if (!ready) return;
  if (!navigator.mediaDevices?.getUserMedia) { error('Camera access requires HTTPS or localhost and a supported browser.'); return; }
  error('Loading hand detector…'); clearResult();
  const startup = camera.start();
  $('start').disabled = true;
  $('video').hidden = false; $('empty').hidden = true; $('stop').disabled = false;
  await startup;
  if (camera.stream) { error(''); $('save-example').disabled = false; }
};
$('stop').onclick = () => camera.stop();
async function analyzeUpload() {
  const file = $('upload').files[0]; if (!file || mode !== 'upload') return;
  pending?.abort(); pending = new AbortController(); const version = ++uploadVersion;
  if (!['image/jpeg', 'image/png'].includes(file.type) || !file.size || file.size > 5 * 1024 * 1024) { error('Choose a nonempty JPEG or PNG image up to 5 MB.'); return; }
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = URL.createObjectURL(file); $('preview').src = objectURL; $('preview').hidden = false; $('empty').hidden = true;
  clearResult();
  if (!ready) { error('A trained checkpoint must be installed before inference.'); return; }
  error('Checking for a hand…');
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
    const hand = await handGate.crop(bitmap);
    if (version !== uploadVersion) return;
    if (!hand.image) { display(noHandResult(hand.reason)); return; }
    const blob = await new Promise(resolve => hand.image.toBlob(resolve, 'image/jpeg', .95));
    if (!blob) throw new Error('Could not prepare the hand image.');
    if (version !== uploadVersion) return;
    const form = new FormData(); form.append('file', blob, 'hand.jpg');
    error('Classifying hand…');
    const result = await request(`predict?confidence=${settings().confidence}&pose_hint=${encodeURIComponent(settings().thumbGuard ? hand.thumbEvidence || 'unknown' : 'unknown')}`, {method: 'POST', body: form, signal: pending.signal});
    if (version === uploadVersion) display(result);
  }
  catch (e) { if (e.name !== 'AbortError' && version === uploadVersion) error(e.message); }
  finally { bitmap?.close(); }
}
$('upload').onchange = analyzeUpload;
$('reset').onclick = () => { pending?.abort(); uploadVersion++; $('upload').value = ''; if (objectURL) URL.revokeObjectURL(objectURL); objectURL = null; $('preview').hidden = true; $('empty').hidden = false; error(''); clearResult(); };
for (const [id, suffix] of [['confidence','%'], ['smoothing',' frames'], ['frequency',' FPS']]) {
  $(id).oninput = () => { $(`${id}-value`).textContent = $(id).value + suffix; };
}
$('confidence').onchange = analyzeUpload;
$('thumb-guard').onchange = analyzeUpload;
window.addEventListener('pagehide', () => { camera.stop(); pending?.abort(); if (objectURL) URL.revokeObjectURL(objectURL); });
document.addEventListener('visibilitychange', () => { if (document.hidden) camera.stop(); });
try {
  const model = await request('model'); ready = model.ready;
  gestureFunctions = model.gesture_functions || {};
  gestureLabels = model.gesture_labels || {};
  $('connection').textContent = ready ? '● Model ready' : '○ Model unavailable';
  $('notice').textContent = ready ? model.limitations : model.error || 'Train the model before running recognition.';
  $('architecture').textContent = model.architecture || 'YOLOv8n-cls (planned)';
  $('checkpoint').textContent = `${model.checkpoint} · ${(model.checkpoint_sha256 || '').slice(0, 12)}`;
  $('device').textContent = model.device;
  const warnings = [];
  if (model.missing_classes?.includes('closed_fist')) warnings.push('The running model cannot recognize Closed Fist. Stop the old server and restart with the current three-class checkpoint.');
  if (model.restart_required) warnings.push('Model files changed after this server loaded. Stop and restart the server to load the current weights.');
  $('model-status').textContent = !model.ready ? 'No trained model is loaded.' : warnings.length ? warnings.join(' ') : `Loaded gestures: ${(model.classes || []).map(name => name.replaceAll('_', ' ')).join(', ')}.`;
  if (warnings.length) { $('notice').textContent = warnings.join(' '); $('connection').textContent = '○ Model restart needed'; }
  if (model.metrics) $('accuracy').textContent = `${(model.metrics.accuracy * 100).toFixed(1)}% · n=${model.metrics.count}`;
} catch (e) { $('connection').textContent = 'Backend offline'; $('notice').textContent = e.message; }
$('start').disabled = !ready;

$('save-example').onclick = async () => {
  const label = $('example-label').value;
  if (!['closed_fist','thumbs_up','open_palm'].includes(label)) {
    error('Choose the actual gesture before saving an example.'); return;
  }
  const video = $('video');
  if (!camera.stream || !video.videoWidth || !video.videoHeight) return;
  // Match the saved example to the actual frame/decision most recently displayed.
  const classified = camera.lastFrame;
  if (classified?.image) {
    const link = document.createElement('a');
    link.download = `${label}__session_${classified.session}__${classified.capturedAt}__hand.jpg`;
    link.href = `data:image/jpeg;base64,${classified.image}`;
    link.click();
    error('Saved the latest classified crop with your chosen label. Attach this JPEG when reporting a wrong prediction.');
    return;
  }
  const generation = camera.generation, session = camera.sessionId;
  const snapshot = document.createElement('canvas');
  snapshot.width = video.videoWidth; snapshot.height = video.videoHeight;
  snapshot.getContext('2d').drawImage(video,0,0,snapshot.width,snapshot.height);
  $('save-example').disabled = true;
  try {
    // Separate drawing buffer prevents capture from overwriting an in-flight camera crop.
    const captureGate = new HandGate();
    captureGate.detector = handGate.detector;
    const hand = await captureGate.crop(snapshot);
    if (generation !== camera.generation || !camera.stream) return;
    const canvas = hand.image || snapshot;
    const link = document.createElement('a');
    link.download = `${label}__session_${session}__${Date.now()}__${hand.image ? 'hand' : 'full'}.jpg`;
    link.href = canvas.toDataURL('image/jpeg',.95);
    link.click();
    error('Example downloaded locally. Keep the session ID with the image when sharing it for retraining.');
  } catch (e) { error(e.message); }
  finally { $('save-example').disabled = !camera.stream; }
};
