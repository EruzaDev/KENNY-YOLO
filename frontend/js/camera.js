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
      const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
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
          const result = JSON.parse(event.data);
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
      this.onError(error.name === 'NotAllowedError' ? 'Camera permission denied. Allow camera access in your browser.' : error.message);
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
      this.pendingFrame = {...message, session: this.sessionId, capturedAt: Date.now()};
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
