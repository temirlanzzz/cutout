import test from 'node:test';
import assert from 'node:assert/strict';
import { dimensions, paintMask, MaskHistory, exportName } from '../editor.js';
test('dimensions preserve aspect, never upscale and avoid zero dimensions', () => {
  assert.deepEqual(dimensions(4000, 2000), [2048, 1024]);
  assert.deepEqual(dimensions(100, 300), [100, 300]);
  assert.deepEqual(dimensions(1, 100000), [1, 2048]);
});
test('erase and restore keep pixels outside the brush untouched', () => {
  const m = new Uint8ClampedArray(100).fill(255);
  paintMask(m, 10, 10, {x:5,y:5}, {x:5,y:5}, 3, false, 0);
  assert.equal(m[55], 0); assert.equal(m[0], 255);
  paintMask(m, 10, 10, {x:5,y:5}, {x:5,y:5}, 3, true, 0);
  assert.equal(m[55], 255);
});
test('soft edge has partial alpha and repeated stamps are idempotent', () => {
  const m = new Uint8ClampedArray(100).fill(255);
  paintMask(m, 10, 10, {x:5,y:5}, {x:5,y:5}, 4, false, .8);
  assert(m.some(v => v > 0 && v < 255));
  const first = m.slice(); paintMask(m, 10, 10, {x:5,y:5}, {x:5,y:5}, 4, false, .8);
  assert.deepEqual(m, first);
});
test('fast pointer movement paints a continuous stroke, clipped at edges', () => {
  const m = new Uint8ClampedArray(100).fill(255);
  paintMask(m, 10, 10, {x:-2,y:5}, {x:12,y:5}, 2, false, 0);
  assert.deepEqual([...m.slice(50,60)], Array(10).fill(0));
  assert.equal(m.length, 100);
});
test('undo snapshots are independent, bounded, and a new stroke clears redo', () => {
  const h = new MaskHistory(2); let m = new Uint8ClampedArray([1]);
  h.save(m); m[0]=2; h.save(m); m[0]=3; h.save(m); m[0]=4;
  m=h.undo(m); assert.equal(m[0],3); m=h.undo(m); assert.equal(m[0],2);
  m=h.undo(m); assert.equal(m[0],2); m=h.redo(m); assert.equal(m[0],3);
  h.save(m); m[0]=9; assert.equal(h.redo(m)[0],9);
  h.clear(); assert.equal(h.past.length+h.future.length,0);
});
test('export names are sanitized and format matches extension', () => {
  assert.equal(exportName('my photo.jpg', 'transparent', 'png'), 'cutout-my-photo.png');
  assert.equal(exportName('watch', 'color', 'webp'), 'cutout-watch-color.webp');
  assert.equal(exportName('', 'blur', 'bad'), 'cutout-photo-blur.png');
});
