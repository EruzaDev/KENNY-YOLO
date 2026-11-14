import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const code = await readFile('frontend/js/camera.js');
const {Camera, cameraConstraints, videoInputs} = await import(`data:text/javascript;base64,${code.toString('base64')}`);

test('selected camera uses exact device ID and optional resolution', () => {
  assert.deepEqual(cameraConstraints('astra-rgb'), {
    video: {width: {ideal: 640}, height: {ideal: 480}, deviceId: {exact: 'astra-rgb'}},
    audio: false,
  });
  assert.equal('deviceId' in cameraConstraints().video, false);
});

test('camera list excludes microphones, speakers and unlabeled IDs', () => {
  assert.deepEqual(videoInputs([
    {kind: 'audioinput', deviceId: 'mic'}, {kind: 'videoinput', deviceId: 'astra'},
    {kind: 'videoinput', deviceId: ''}, {kind: 'audiooutput', deviceId: 'speaker'},
  ]), [{kind: 'videoinput', deviceId: 'astra'}]);
});

test('switching cameras releases the old stream and websocket', async () => {
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const oldSocket = globalThis.WebSocket, oldLocation = globalThis.location;
  const constraints = [], tracks = [], sockets = [];
  let selected = 'built-in';
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {mediaDevices: {
    getUserMedia: async value => {
      constraints.push(value);
      const track = {stopped: false, stop() { this.stopped = true; }};
      tracks.push(track);
      return {getTracks: () => [track], getVideoTracks: () => [track]};
    },
  }}});
  globalThis.location = {protocol: 'http:', host: 'localhost'};
  globalThis.WebSocket = class {
    constructor() { sockets.push(this); }
    close() { this.closed = true; }
  };
  const camera = new Camera({play: async () => {}}, () => ({deviceId: selected}),
    () => {}, error => assert.fail(error), () => {}, {load: async () => {}});
  try {
    await camera.start();
    selected = 'astra-rgb';
    await camera.start();
    assert.equal(tracks[0].stopped, true);
    assert.equal(sockets[0].closed, true);
    assert.equal(constraints[1].video.deviceId.exact, 'astra-rgb');
    assert.equal(tracks[1].stopped, false);
    camera.stop();
    assert.equal(tracks[1].stopped, true);
  } finally {
    camera.stop();
    if (oldNavigator) Object.defineProperty(globalThis, 'navigator', oldNavigator);
    else delete globalThis.navigator;
    globalThis.WebSocket = oldSocket;
    globalThis.location = oldLocation;
  }
});
