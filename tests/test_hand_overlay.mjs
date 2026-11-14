import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const code = await readFile('frontend/js/hand_overlay.js');
const {projectHandBox} = await import(`data:text/javascript;base64,${code.toString('base64')}`);
const box = {x: 40, y: 60, width: 120, height: 160};

test('hand box follows the mirrored camera rather than the raw source', () => {
  assert.deepEqual(projectHandBox(box, 640, 480, 640, 480, true),
    {left: 480, top: 60, width: 120, height: 160});
  assert.deepEqual(projectHandBox(box, 640, 480, 640, 480),
    {left: 40, top: 60, width: 120, height: 160});
});
test('letterbox offsets and resizing keep the box aligned', () => {
  assert.deepEqual(projectHandBox(box, 640, 480, 640, 640),
    {left: 40, top: 140, width: 120, height: 160});
  assert.deepEqual(projectHandBox(box, 640, 480, 320, 240),
    {left: 20, top: 30, width: 60, height: 80});
});
test('portrait image gets a horizontal contain offset', () => {
  assert.deepEqual(projectHandBox(box, 480, 640, 640, 640),
    {left: 120, top: 60, width: 120, height: 160});
});
test('missing, malformed and out-of-frame boxes are not rendered', () => {
  for (const invalid of [null, {...box, x: NaN}, {...box, width: -1}, {...box, x: 700}]) {
    assert.equal(projectHandBox(invalid, 640, 480, 640, 480), null);
  }
  assert.equal(projectHandBox(box, 0, 480, 640, 480), null);
});
