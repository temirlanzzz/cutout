// Inference worker: loads the ISNet/ORMBG model with transformers.js and returns a soft alpha mask.
// Runs off the main thread so the UI keeps painting while ONNX Runtime is busy.
import { AutoModel, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const MODEL_ID = 'onnx-community/ormbg-ONNX';

// The web build of transformers.js only looks on the hub when local models are off;
// leaving this ambiguous made the original build fail silently.
env.allowLocalModels = false;
env.useBrowserCache = true;

let model = null;
let processor = null;
let info = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function pickDevice(pref) {
  // Auto means WASM: in my tests ORT's WebGPU provider rejects this ISNet export at run time
  // ("using ceil() in shape computation is not yet supported for MaxPool"), so WebGPU is opt-in
  // and the page restarts this worker on WASM when it fails.
  if (pref !== 'webgpu') return 'wasm';
  if (!('gpu' in navigator)) return 'wasm';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 'wasm';
    // fp16 weights keep the WebGPU download at ~88 MB instead of ~176 MB for fp32.
    if (adapter.features.has('shader-f16')) return 'webgpu';
  } catch (_) { /* fall through */ }
  return 'wasm';
}

async function load(pref) {
  const isolated = self.crossOriginIsolated === true;
  const cores = navigator.hardwareConcurrency || 4;
  const threads = isolated ? Math.max(1, Math.min(cores, 8)) : 1;
  env.backends.onnx.wasm.numThreads = threads;

  let device = await pickDevice(pref);
  const t0 = performance.now();
  const opts = (dev) => ({
    device: dev,
    dtype: dev === 'webgpu' ? 'fp16' : 'q8',
    config: { model_type: 'custom' },
    progress_callback: (p) => post({ type: 'progress', ...p }),
  });
  try {
    model = await AutoModel.from_pretrained(MODEL_ID, opts(device));
  } catch (err) {
    if (device !== 'webgpu') throw err;
    post({ type: 'note', text: 'webgpu failed (' + (err && err.message || err) + '), falling back to wasm' });
    device = 'wasm';
    model = await AutoModel.from_pretrained(MODEL_ID, opts(device));
  }
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  const loadMs = performance.now() - t0;

  const session = model.sessions && (model.sessions.model || Object.values(model.sessions)[0]);
  const inputName = (session && session.inputNames && session.inputNames[0]) || 'input';
  const outputName = (session && session.outputNames && session.outputNames[0]) || null;

  info = {
    device,
    dtype: device === 'webgpu' ? 'fp16' : 'int8 (q8)',
    threads: device === 'wasm' ? threads : null,
    isolated,
    cores,
    inputName,
    outputName,
    loadMs,
    runs: 0,
  };
  post({ type: 'ready', info });
}

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

async function run(id, pixels, width, height) {
  // The page sends already-decoded RGBA at <=1024 px so EXIF orientation and
  // decoding match exactly what the user sees.
  const image = new RawImage(new Uint8ClampedArray(pixels), width, height, 4).rgb();
  const t0 = performance.now();
  const { pixel_values } = await processor(image);
  const t1 = performance.now();
  const outputs = await model({ [info.inputName]: pixel_values });
  const t2 = performance.now();

  const out = (info.outputName && outputs[info.outputName]) || Object.values(outputs)[0];
  const dims = out.dims; // [1, 1, H, W]
  const h = dims[dims.length - 2];
  const w = dims[dims.length - 1];
  const n = w * h;
  const src = out.data;
  const mask = new Uint8ClampedArray(n);
  if (out.type === 'float16' && !(typeof Float16Array !== 'undefined' && src instanceof Float16Array)) {
    for (let i = 0; i < n; i++) mask[i] = halfToFloat(src[i]) * 255 + 0.5;
  } else {
    for (let i = 0; i < n; i++) mask[i] = src[i] * 255 + 0.5;
  }
  const t3 = performance.now();
  info.runs++;
  post({
    type: 'mask', id, width: w, height: h, mask,
    timings: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0, run: info.runs },
    image: { width: image.width, height: image.height },
  }, [mask.buffer]);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') await load(msg.pref || 'auto');
    else if (msg.type === 'run') await run(msg.id, msg.pixels, msg.width, msg.height);
  } catch (err) {
    post({ type: 'error', where: msg.type, message: (err && (err.message || String(err))) || 'unknown error' });
  }
};
