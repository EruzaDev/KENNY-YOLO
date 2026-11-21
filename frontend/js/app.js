import {request} from './api.js';
import {Camera, videoInputs} from './camera.js?v=12';
import {HandGate, noHandResult} from './hand_gate.js?v=12';
import {projectHandBox} from './hand_overlay.js?v=8';
const $ = id => document.getElementById(id);
let ready = false, mode = 'camera', objectURL = null, uploadVersion = 0, pending = null;
const settings = () => ({confidence: +$('confidence').value / 100, smoothing: +$('smoothing').value, frequency: +$('frequency').value, thumbGuard: $('thumb-guard').checked, deviceId: $('camera-select').value});
const error = message => { $('error').textContent = message; };
const handGate = new HandGate();
let gestureFunctions = {};
let gestureLabels = {};
let loadedCheckpoint = null;
let handOverlay = null;
const jointEdges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function renderJoints(points, width, height) {
  const canvas = $('joint-overlay');
  const ctx = canvas.getContext('2d');
  if (!points || points.length !== 21 || !width || !height) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); canvas.hidden = true; return;
  }
  canvas.width = width; canvas.height = height; canvas.hidden = false;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#72e0c0'; ctx.lineWidth = Math.max(2, width / 600);
  for (const [a, b] of jointEdges) {
    ctx.beginPath(); ctx.moveTo(points[a].x * width, points[a].y * height);
    ctx.lineTo(points[b].x * width, points[b].y * height); ctx.stroke();
  }
  for (const point of points) {
    ctx.beginPath(); ctx.arc(point.x * width, point.y * height, Math.max(3, width / 180), 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#153a2e'; ctx.lineWidth = 2; ctx.stroke();
  }
}

function renderHandOverlay() {
  const element = $('hand-box');
  const box = handOverlay && projectHandBox(handOverlay.box, handOverlay.width, handOverlay.height,
    $('stage').clientWidth, $('stage').clientHeight, mode === 'camera');
  element.hidden = !box;
  if (!box) return;
  Object.assign(element.style, {left: `${box.left}px`, top: `${box.top}px`,
    width: `${box.width}px`, height: `${box.height}px`});
  $('hand-box-label').textContent = handOverlay.label;
}
if (globalThis.ResizeObserver) new ResizeObserver(renderHandOverlay).observe($('stage'));
else window.addEventListener('resize', renderHandOverlay);
$('preview').addEventListener('load', renderHandOverlay);

function clearResult() {
  $('gesture').textContent = 'No gesture detected'; $('score').textContent = 'Waiting for an image';
  $('class-scores').textContent = '';
  $('function').textContent = 'Function: —';
  $('prediction-icon').textContent = '↗';
  $('confidence-bar').style.width = '0%';
  $('overlay').hidden = true; $('timing').textContent = '— ms · — FPS';
  handOverlay = null; renderHandOverlay();
  renderJoints(null);
}
function display(result, fps = null) {
  error('');
  renderJoints(result.landmarks, result.source_width, result.source_height);
  const name = 'stable_gesture' in result ? result.stable_gesture : result.predictions[0]?.class_name;
  const prediction = name && result.predictions.find(p => p.class_name === name);
  const label = name ? prediction?.display_name || gestureLabels[name] || name.replaceAll('_', ' ') : 'No gesture detected';
  $('gesture').textContent = label;
  $('prediction-icon').textContent = ({open_palm: '✋', thumbs_up: '👍', ok: '👌'})[name] || '↗';
  $('confidence-bar').style.width = prediction ? `${prediction.confidence * 100}%` : '0%';
  const gestureFunction = prediction?.function || gestureFunctions[name];
  $('function').textContent = name && gestureFunction ? `Function: ${gestureFunction} (display only)` : 'Function: —';
  $('score').textContent = result.joint_verified
    ? `${result.joint_corrected ? 'Joint check corrected the image model' : 'Joint check confirms the image model'} · ${gestureLabels[result.joint_gesture] || result.joint_gesture}; image model favored ${gestureLabels[result.model_top_class] || result.model_top_class} (${(result.model_top_confidence * 100).toFixed(1)}%)`
    : prediction ? `${(prediction.confidence * 100).toFixed(1)}% confidence · hand crop` : result.rejection_reason || 'Waiting for a stable prediction above threshold';
  handOverlay = result.hand_box ? {
    box: result.hand_box, width: result.source_width, height: result.source_height,
    label: result.joint_verified ? `${label} · joint checked` : prediction ? `${label} · ${(prediction.confidence * 100).toFixed(1)}% confidence` : 'Hand · checking pose joints',
  } : null;
  renderHandOverlay();
  $('overlay').hidden = false; $('overlay').textContent = `${label}${name && gestureFunction ? ` · ${gestureFunction}` : ''}${result.joint_verified ? ' · joint checked' : prediction ? ` · ${(prediction.confidence * 100).toFixed(1)}%` : ''}`;
  $('class-scores').textContent = Object.entries(result.scores || {})
    .filter(([,value]) => Number.isFinite(value))
    .map(([label,value]) => `${label.replaceAll('_', ' ')}: ${(value * 100).toFixed(1)}%`).join(' · ');
  $('timing').textContent = `${result.latency_ms} ms · ${fps ? fps.toFixed(1) : '—'} FPS`;
}
const camera = new Camera($('video'), settings, display, error, () => {
  $('start').disabled = !ready; $('stop').disabled = true; $('save-example').disabled = true;
  $('camera-select').disabled = false;
  $('video').hidden = true; $('empty').hidden = false; clearResult();
}, handGate);
function switchMode(next) {
  camera.stop(); pending?.abort(); uploadVersion++; mode = next; error(''); clearResult();
  $('camera-tab').classList.toggle('active', next === 'camera'); $('upload-tab').classList.toggle('active', next === 'upload');
  $('camera-tab').setAttribute('aria-pressed', String(next === 'camera'));
  $('upload-tab').setAttribute('aria-pressed', String(next === 'upload'));
  $('camera-actions').hidden = next !== 'camera'; $('upload-actions').hidden = next !== 'upload';
  $('camera-source').hidden = next !== 'camera';
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
  $('camera-select').disabled = true;
  $('video').hidden = false; $('empty').hidden = true; $('stop').disabled = false;
  await startup;
  $('camera-select').disabled = false;
  await refreshCameras();
  if (camera.stream) { error(''); $('save-example').disabled = false; }
};
$('stop').onclick = () => camera.stop();
async function refreshCameras(requestAccess = false) {
  const devices = navigator.mediaDevices;
  if (!devices?.enumerateDevices) {
    $('camera-help').textContent = 'Camera selection requires localhost or HTTPS and a supported browser.';
    return;
  }
  const button = $('refresh-cameras');
  button.disabled = true;
  let permissionStream;
  try {
    // Ask only after an explicit click, never on page load. Release this temporary
    // stream immediately; do not replace or stop an already active camera.
    if (requestAccess && !camera.stream) {
      permissionStream = await devices.getUserMedia({video: true, audio: false});
      permissionStream.getTracks().forEach(track => track.stop());
    }
    const cameras = videoInputs(await devices.enumerateDevices());
    const selected = $('camera-select').value;
    $('camera-select').replaceChildren(new Option('Default camera', ''));
    cameras.forEach((device, index) => {
      $('camera-select').add(new Option(device.label || `Camera ${index + 1} (allow access to see name)`, device.deviceId));
    });
    if (selected && !cameras.some(device => device.deviceId === selected)) {
      const unavailable = new Option('Selected camera disconnected — choose another', selected);
      unavailable.disabled = true;
      $('camera-select').add(unavailable);
      if (camera.stream) { camera.stop(); error('Selected camera disconnected. Choose another camera.'); }
    }
    $('camera-select').value = selected;
    $('camera-help').textContent = cameras.length
      ? `${cameras.length} camera${cameras.length === 1 ? '' : 's'} available. For Orbbec Astra, choose the RGB/color camera, not depth or IR.`
      : 'No cameras listed. Connect your camera, click Refresh cameras, and allow camera access.';
  } catch (e) {
    $('camera-help').textContent = e.name === 'NotAllowedError'
      ? 'Allow camera access in your browser, then refresh to reveal camera names.'
      : `Could not list cameras: ${e.message}`;
  } finally {
    permissionStream?.getTracks().forEach(track => track.stop());
    button.disabled = false;
  }
}
$('refresh-cameras').onclick = () => refreshCameras(true);
$('camera-select').onchange = () => { if (camera.stream) $('start').onclick(); };
const cameraDeviceChange = () => refreshCameras();
navigator.mediaDevices?.addEventListener?.('devicechange', cameraDeviceChange);
window.addEventListener('pagehide', () => navigator.mediaDevices?.removeEventListener?.('devicechange', cameraDeviceChange));
refreshCameras();
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
    if (version === uploadVersion) display({...handGate.filterResult(result, hand.okEvidence), hand_box: hand.box,
      source_width: bitmap.width, source_height: bitmap.height});
  }
  catch (e) { if (e.name !== 'AbortError' && version === uploadVersion) error(e.message); }
  finally { bitmap?.close(); }
}
$('upload').onchange = analyzeUpload;
$('upload').closest('label').onkeydown = event => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('upload').click(); }
};
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
  loadedCheckpoint = model.checkpoint_sha256;
  gestureFunctions = model.gesture_functions || {};
  gestureLabels = model.gesture_labels || {};
  $('connection').textContent = ready ? '● Model ready' : '○ Model unavailable';
  $('notice').textContent = ready ? model.limitations : model.error || 'Train the model before running recognition.';
  $('architecture').textContent = model.architecture || 'YOLOv8n-cls (planned)';
  $('checkpoint').textContent = `${model.checkpoint} · ${(model.checkpoint_sha256 || '').slice(0, 12)}`;
  $('device').textContent = model.device;
  const warnings = [];
  if (model.missing_classes?.includes('ok')) warnings.push('The running model cannot recognize OK. Stop the old server and restart with the current three-class checkpoint.');
  if (model.restart_required) warnings.push('Model files changed after this server loaded. Stop and restart the server to load the current weights.');
  $('model-status').textContent = !model.ready ? 'No trained model is loaded.' : warnings.length ? warnings.join(' ') : `Loaded gestures: ${(model.classes || []).map(name => name.replaceAll('_', ' ')).join(', ')}.`;
  if (warnings.length) { $('notice').textContent = warnings.join(' '); $('connection').textContent = '○ Model restart needed'; }
  if (model.metrics) $('accuracy').textContent = `${(model.metrics.accuracy * 100).toFixed(1)}% · n=${model.metrics.count}`;
} catch (e) { $('connection').textContent = 'Backend offline'; $('notice').textContent = e.message; }
$('start').disabled = !ready;

async function loadResults() {
  try {
    const results = await request('results');
    if (!ready || !results.available || results.checkpoint_sha256 !== loadedCheckpoint) {
      $('results-status').textContent = 'No verified results are available for the loaded model.';
      return;
    }
    const {training, metrics, training_fit: fit} = results;
    const correct = report => Math.round(report.accuracy * report.count);
    $('results-status').textContent = `${results.title} · checkpoint ${results.checkpoint_sha256.slice(0, 12)} · confidence threshold 80%`;
    $('result-epochs').textContent = `${training.epochs} epochs`;
    $('result-training-count').textContent = `${training.image_count} training files before class balancing`;
    $('result-validation').textContent = `${correct(metrics.validation)} / ${metrics.validation.count}`;
    $('result-regression').textContent = `${correct(metrics.regression)} / ${metrics.regression.count}`;
    $('result-training-fit').textContent = fit ? `${fit.accepted_correct} / ${fit.count}` : 'Not measured';
    $('result-summary').textContent = results.summary || `Original validation views: ${results.original_validation.correct}/${results.original_validation.count}. Expanded validation: ${(metrics.validation.accuracy * 100).toFixed(1)}% accepted correctly; ${(metrics.validation.raw_top1_accuracy * 100).toFixed(1)}% raw top-1. Webcam training: ${training.webcam_frames} crops from ${training.webcam_sessions} sessions, kept in training only.`;
    $('checkpoint').textContent = `${results.checkpoint} · ${results.checkpoint_sha256.slice(0, 12)}`;
    const addRow = (id, values) => {
      const row = document.createElement('tr');
      values.forEach((value, index) => {
        const cell = document.createElement(index ? 'td' : 'th');
        if (!index) cell.scope = 'row';
        cell.textContent = String(value);
        row.append(cell);
      });
      $(id).append(row);
    };
    $('dataset-results').replaceChildren(); $('class-results').replaceChildren();
    for (const label of results.classes || ['ok', 'open_palm', 'thumbs_up']) {
      const name = gestureLabels[label] || label.replaceAll('_', ' ');
      addRow('dataset-results', [name, ...['train', 'val', 'test'].map(split => training.splits[split][label])]);
      const scores = metrics.regression.classification_report[label];
      addRow('class-results', [name, ...['precision', 'recall', 'f1-score'].map(key => `${(scores[key] * 100).toFixed(1)}%`), scores.support]);
    }
    for (const [image, link, asset] of [
      ['result-curves', 'curves-link', 'training-curves'],
      ['result-confusion', 'confusion-link', 'validation-confusion'],
      ['result-crops', null, 'training-crops'],
    ]) {
      if (!results.assets[asset]) { $(image).closest('details, figure').hidden = true; continue; }
      $(image).src = `${results.assets[asset]}?checkpoint=${results.checkpoint_sha256.slice(0, 12)}`;
      if (link) $(link).href = $(image).src;
    }
    $('result-limitations').replaceChildren(...results.limitations.map(text => {
      const item = document.createElement('li'); item.textContent = text; return item;
    }));
    $('results-content').hidden = false;
  } catch (e) { $('results-status').textContent = `Results unavailable: ${e.message}`; }
}
loadResults();

$('save-example').onclick = async () => {
  const label = $('example-label').value;
  if (!['ok','thumbs_up','open_palm'].includes(label)) {
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
