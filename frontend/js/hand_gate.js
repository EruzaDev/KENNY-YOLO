// A real, independent hand detector gates the trained YOLO gesture classifier.
// No detector / no hand means no gesture inference, never a whole-frame fallback.
export class HandGate {
  constructor() {
    this.detector = null;
    this.loading = null;
    this.canvas = null;
  }
  async load() {
    if (this.detector) return;
    if (!this.loading) {
      this.loading = (async () => {
        const base = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
        const {FilesetResolver, HandLandmarker} = await import(`${base}/vision_bundle.mjs`);
        const fileset = await FilesetResolver.forVisionTasks(`${base}/wasm`);
        this.detector = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'CPU',
          },
          runningMode: 'IMAGE', numHands: 2,
          minHandDetectionConfidence: .6, minHandPresenceConfidence: .6,
        });
      })().catch(() => {
        this.loading = null;
        throw new Error('Hand detector could not load. Allow access to cdn.jsdelivr.net and storage.googleapis.com, then retry. Recognition is paused.');
      });
    }
    await this.loading;
  }
  async crop(source) {
    await this.load();
    // IMAGE mode rechecks presence every sample instead of retaining a tracked hand.
    const result = this.detector.detect(source);
    if (!result.landmarks?.length) return {image: null, reason: 'No hand detected'};
    if (result.landmarks.length !== 1) return {image: null, reason: 'Show one hand at a time'};
    const width = source.videoWidth || source.naturalWidth || source.width;
    const height = source.videoHeight || source.naturalHeight || source.height;
    const box = handBounds(result.landmarks[0], width, height);
    if (!box) return {image: null, reason: 'Move your hand closer and keep it fully visible'};
    this.canvas ||= document.createElement('canvas');
    const scale = Math.min(1, 640 / Math.max(box.width, box.height));
    this.canvas.width = Math.max(1, Math.round(box.width * scale));
    this.canvas.height = Math.max(1, Math.round(box.height * scale));
    this.canvas.getContext('2d').drawImage(source, box.x, box.y, box.width, box.height,
      0, 0, this.canvas.width, this.canvas.height);
    return {image: this.canvas, box, reason: null,
      thumbEvidence: thumbEvidence(result.worldLandmarks?.[0]),
      okEvidence: okPinchEvidence(result.worldLandmarks?.[0])};
  }
  filterResult(result, evidence) { return checkOkEvidence(result, evidence); }
}

// Veto clearly conflicting anatomy; distance alone is unreliable for oblique OK poses.
export function okPinchEvidence(points) {
  if (!Array.isArray(points) || points.length !== 21 ||
      points.some(p => ![p.x,p.y,p.z].every(Number.isFinite))) return 'unknown';
  const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);
  const palm = Math.max(distance(points[0],points[9]), distance(points[5],points[17]));
  if (palm < 1e-6) return 'unknown';
  const ratio = distance(points[4],points[8]) / palm;
  if (ratio < .35) return 'pinched';
  // All four fingers folded with an extended thumb clearly contradicts an OK ring.
  // An open-hand veto was rejected because oblique OK poses produced false vetoes.
  return ratio > .7 && thumbEvidence(points) === 'thumb_extended' ? 'separate' : 'unknown';
}

export function checkOkEvidence(result, evidence) {
  if (evidence !== 'separate' || !result.predictions?.some(p => p.class_name === 'ok')) return result;
  return {...result, predictions: [], stable_gesture: null, message: 'No gesture detected',
    rejected_predictions: result.predictions,
    rejection_reason: 'OK score conflicts with a clear thumbs-up pose. Hold the pose clearly.'};
}

export function handBounds(points, width, height) {
  if (!Array.isArray(points) || points.length !== 21 || width <= 0 || height <= 0 ||
      points.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) return null;
  const xs = points.map(p => p.x * width), ys = points.map(p => p.y * height);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  if (maxX - minX < 24 || maxY - minY < 24) return null;
  const padding = Math.max(maxX - minX, maxY - minY) * .25;
  const x = Math.max(0, Math.floor(minX - padding));
  const y = Math.max(0, Math.floor(minY - padding));
  const right = Math.min(width, Math.ceil(maxX + padding));
  const bottom = Math.min(height, Math.ceil(maxY + padding));
  return right > x && bottom > y ? {x, y, width: right - x, height: bottom - y} : null;
}

export function noHandResult(reason = 'No hand detected') {
  return {task: 'classify', predictions: [], scores: {}, stable_gesture: null,
    message: 'No gesture detected', rejection_reason: reason, latency_ms: 0};
}

// Conservative evidence only: never manufacture a gesture class from landmarks.
// Ambiguous or missing world landmarks leave the YOLO decision unchanged.
export function thumbEvidence(points) {
  if (!Array.isArray(points) || points.length !== 21 ||
      points.some(p => ![p.x,p.y,p.z].every(Number.isFinite))) return 'unknown';
  const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);
  const angle = (a,b,c) => {
    const u=[a.x-b.x,a.y-b.y,a.z-b.z],v=[c.x-b.x,c.y-b.y,c.z-b.z];
    const magnitude=Math.hypot(...u)*Math.hypot(...v);
    return magnitude > 1e-10 ? Math.acos(Math.max(-1,Math.min(1,u.reduce((sum,x,i)=>sum+x*v[i],0)/magnitude)))*180/Math.PI : NaN;
  };
  const folded = [5,9,13,17].every(i => angle(points[i],points[i+1],points[i+2]) < 130 &&
    distance(points[i+3],points[0]) < 1.1*distance(points[i+1],points[0]));
  if (!folded) return 'unknown';
  const thumbAngle=angle(points[2],points[3],points[4]);
  const ipRadius=distance(points[3],points[0]), tipRadius=distance(points[4],points[0]);
  if (ipRadius < 1e-6 || !Number.isFinite(thumbAngle)) return 'unknown';
  if (thumbAngle < 125 || tipRadius < .95*ipRadius) return 'thumb_folded';
  if (thumbAngle > 160 && tipRadius > 1.18*ipRadius &&
      distance(points[4],points[5]) > 1.05*distance(points[3],points[5])) return 'thumb_extended';
  return 'unknown';
}
