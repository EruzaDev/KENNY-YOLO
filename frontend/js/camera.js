export function cameraConstraints(deviceId = '') {
  return {video: {width: {ideal: 640}, height: {ideal: 480},
    ...(deviceId ? {deviceId: {exact: deviceId}} : {})}, audio: false};
}

export function videoInputs(devices) {
  return devices.filter(device => device.kind === 'videoinput' && device.deviceId);
}

export class Camera {
  constructor(video, settings, onResult, onError, onStop, handGate) {
    Object.assign(this, {video, settings, onResult, onError, onStop, handGate});
    this.generation = 0;
  }
  async start() {
    this.stop();
    const generation = this.generation;
    try {
      await this.handGate.load();
      if (generation !== this.generation) return;
      const stream = await navigator.mediaDevices.getUserMedia(cameraConstraints(this.settings().deviceId));
      if (generation !== this.generation) { stream.getTracks().forEach(t => t.stop()); return; }
      this.stream = stream;
      this.sessionId = globalThis.crypto?.randomUUID?.() || String(Date.now());
      stream.getVideoTracks()[0].onended = () => { this.onError('Camera disconnected. Start the camera again.'); this.stop(); };
      this.video.srcObject = stream;
      await this.video.play();
      if (generation !== this.generation) return;
      const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/webcam`);
      this.socket = socket;
      socket.onopen = () => { if (generation === this.generation) this.send(); };
      socket.onmessage = event => {
        if (generation !== this.generation) return;
        clearTimeout(this.timeout);
        try {
          const classified = JSON.parse(event.data);
          const checked = this.handGate.filterResult
            ? this.handGate.filterResult(classified, this.pendingFrame?.okEvidence) : classified;
          const result = {...checked,
            hand_box: this.pendingFrame?.hand_box || null,
            source_width: this.pendingFrame?.source_width,
            source_height: this.pendingFrame?.source_height};
          if (result.error) throw new Error(result.error);
          const now = performance.now();
          this.lastFrame = this.pendingFrame?.image ? {...this.pendingFrame, result} : null;
          this.pendingFrame = null;
          this.onResult(result, this.lastResponse ? 1000 / (now - this.lastResponse) : null);
          this.lastResponse = now;
          const delay = Math.max(0, 1000 / this.settings().frequency - (now - this.sent));
          this.timer = setTimeout(() => this.send(), delay);
        } catch (error) { this.onError(error.message); this.stop(); }
      };
      socket.onerror = () => {
        if (generation !== this.generation) return;
        this.onError('Backend connection failed. Check that the server is running.'); this.stop();
      };
      socket.onclose = () => {
        if (generation === this.generation && this.stream) {
          this.onError('Backend connection closed. Restart your camera.'); this.stop();
        }
      };
    } catch (error) {
      if (generation !== this.generation) return;
      const messages = {
        NotAllowedError: 'Camera permission denied. Allow camera access in your browser.',
        NotFoundError: 'Selected camera was not found. Reconnect it and refresh the camera list.',
        OverconstrainedError: 'Selected camera is unavailable or does not provide a supported video stream. Choose its RGB camera and refresh the list.',
        NotReadableError: 'Camera could not start. Close other camera apps and check the camera driver.',
      };
      this.onError(messages[error.name] || error.message);
      this.stop();
    }
  }
  async send() {
    if (this.socket?.readyState !== WebSocket.OPEN || !this.stream) return;
    if (!this.video.videoWidth || !this.video.videoHeight) {
      this.timer = setTimeout(() => this.send(), 100); return;
    }
    const generation = this.generation;
    this.sent = performance.now();
    try {
      const frame = await this.handGate.crop(this.video);
      if (generation !== this.generation || this.socket?.readyState !== WebSocket.OPEN) return;
      const {confidence, smoothing, thumbGuard} = this.settings();
      const message = frame.image
        ? {image: frame.image.toDataURL('image/jpeg', .9).split(',')[1], confidence, smoothing, pose_hint: thumbGuard ? frame.thumbEvidence || 'unknown' : 'unknown'}
        : {type: 'no_hand', confidence, smoothing, reason: frame.reason};
      this.pendingFrame = {...message, session: this.sessionId, capturedAt: Date.now(),
        okEvidence: frame.okEvidence || 'unknown',
        hand_box: frame.image ? frame.box : null,
        source_width: this.video.videoWidth, source_height: this.video.videoHeight};
      this.socket.send(JSON.stringify(message));
      this.timeout = setTimeout(() => { this.onError('Inference timed out. Check the backend.'); this.stop(); }, 30000);
    } catch (error) {
      if (generation !== this.generation) return;
      this.onError(error.message); this.stop();
    }
  }
  stop() {
    this.generation++;
    clearTimeout(this.timer); clearTimeout(this.timeout);
    const socket = this.socket; this.socket = null;
    if (socket) {
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      socket.close();
    }
    this.stream?.getTracks().forEach(track => { track.onended = null; track.stop(); });
    this.stream = null; this.video.srcObject = null; this.lastResponse = null; this.sessionId = null; this.pendingFrame = null; this.lastFrame = null;
    this.onStop();
  }
}
