import { dimensions, paintMask, MaskHistory, exportName } from './editor.js';
// Cutout — page logic: input, worker orchestration, compositing, before/after divider.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const MAX_SIDE = 2048;      // compositing resolution cap
const MODEL_SIDE = 1024;    // what we hand the worker (model input is 1024x1024)
const params = new URLSearchParams(location.search);

const el = {
  status: $('#status'), prog: $('#prog'), drop: $('#drop'), stage: $('#stage'), stageWrap: $('#stageWrap'),
  before: $('#before'), after: $('#after'), checker: $('#checker'), busyText: $('#busyText'),
  file: $('#file'), download: $('#download'), reset: $('#reset'),
  sBackend: $('#sBackend'), sIso: $('#sIso'), sLoad: $('#sLoad'), sInfer: $('#sInfer'), sPre: $('#sPre'), sImg: $('#sImg'),
  blurAmt: $('#blurAmt'), blurOut: $('#blurOut'), custom: $('#custom'), prefNote: $('#prefNote'),
};

// Public hook for automated measurement (headless Chrome) and curious visitors.
const cutout = window.cutout = { results: [], errors: [], info: null, busy: false };

const state = {
  pref: ['auto', 'webgpu', 'wasm'].includes(params.get('backend')) ? params.get('backend') : 'auto',
  worker: null, ready: null, info: null, runId: 0,
  name: 'photo', W: 0, H: 0,
  src: document.createElement('canvas'),
  cut: document.createElement('canvas'),
  maskA: document.createElement('canvas'),   // white with alpha = mask (model resolution)
  maskG: document.createElement('canvas'),   // grayscale mask for the "mask" view
  hasMask: false,
  mode: 'transparent', color: '#ffffff', grad: 'studio', blur: 14,
  view: 'split', x: 50, loadId: 0, tool: 'compare', alpha: null, history: new MaskHistory(), brush: 60, softness: .35, cursor: { x: 0, y: 0 },
};

const fmtS = (ms) => ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';
const mb = (b) => (b / 1048576).toFixed(1);

function setStatus(text, err) {
  el.status.textContent = text;
  el.status.classList.toggle('err', !!err);
}
let progressTimer;
function setProgress(frac) {
  clearTimeout(progressTimer);
  if (frac == null) { el.prog.style.opacity = 0; progressTimer = setTimeout(() => { el.prog.style.width = '0' }, 400); return; }
  el.prog.style.opacity = 1;
  el.prog.style.width = (Math.max(0, Math.min(1, frac)) * 100).toFixed(1) + '%';
}

/* ---------------- worker ---------------- */
function ensureWorker() {
  if (state.ready) return state.ready;
  const files = new Map();
  let t0 = performance.now();
  state.worker = new Worker(new URL('worker.js', import.meta.url), { type: 'module' });
  state.ready = new Promise((resolve, reject) => {
    state.worker.onmessage = (e) => {
      const m = e.data;
      if (m.id != null && m.id !== state.runId) return;
      if (m.type === 'progress') {
        if (m.status === 'progress' && m.total) {
          files.set(m.file, { loaded: m.loaded, total: m.total });
          let loaded = 0, total = 0;
          for (const f of files.values()) { if (f.total > 1e5) { loaded += f.loaded; total += f.total } }
          if (total) {
            setProgress(loaded / total);
            setStatus('downloading model · ' + mb(loaded) + ' / ' + mb(total) + ' mb');
          }
        } else if (m.status === 'initiate' && /\.onnx$/.test(m.file || '')) {
          setStatus('fetching ' + m.file.split('/').pop());
        } else if (m.status === 'done' && /\.onnx$/.test(m.file || '')) {
          setProgress(1);
          setStatus('compiling session…');
        }
      } else if (m.type === 'ready') {
        state.info = cutout.info = m.info;
        setProgress(null);
        showInfo();
        resolve(m.info);
      } else if (m.type === 'mask') {
        onMask(m);
      } else if (m.type === 'error') {
        if (m.where === 'run' && state.info && state.info.device === 'webgpu') {
          // ORT's WebGPU provider can't run this ISNet export (MaxPool with ceil_mode).
          // A fresh worker on WASM is the only reliable recovery; reusing the module kept failing.
          cutout.fallback = 'webgpu failed: ' + m.message;
          cutout.notes = (cutout.notes || []).concat(cutout.fallback);
          state.worker.terminate();
          state.worker = null; state.ready = null; state.info = null; state.pref = 'wasm';
          $$('#backendPref button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.pref === 'wasm')));
          el.prefNote.textContent = 'WebGPU was tried and failed here (' + m.message + '), so this run fell back to WASM.';
          setStatus('webgpu failed · retrying on wasm…');
          segment();
          return;
        }
        cutout.errors.push(m.where + ': ' + m.message);
        setProgress(null);
        setStatus((m.where === 'load' ? 'model failed to load: ' : 'inference failed: ') + m.message, true);
        el.stage.classList.remove('working');
        cutout.busy = false;
        if (m.where === 'load' && state.worker) { state.ready = null; state.worker.terminate(); reject(new Error(m.message)); }
      }
    };
    state.worker.onerror = (e) => {
      const msg = e.message || 'worker failed to start';
      cutout.busy = false;
      el.stage.classList.remove('working');
      setProgress(null);
      cutout.errors.push('worker: ' + msg);
      setStatus('worker error: ' + msg, true);
      state.ready = null;
      reject(new Error(msg));
    };
  });
  state.ready.catch(() => {});
  setStatus('starting worker…');
  state.worker.postMessage({ type: 'load', pref: state.pref });
  return state.ready;
}

function showInfo() {
  const i = state.info;
  if (!i) return;
  el.sBackend.textContent = i.device === 'webgpu' ? 'webgpu · fp16' : 'wasm int8 ×' + i.threads + (cutout.fallback ? ' (fallback)' : '');
  el.sIso.textContent = i.isolated ? 'yes' : 'no (single-threaded)';
  el.sLoad.textContent = fmtS(i.loadMs);
}

/* ---------------- input ---------------- */
async function loadBlob(blob, name, token = ++state.loadId) {
  if (token !== state.loadId) return;
  if (!blob || !/^image\//.test(blob.type || 'image/')) { setStatus('that file is not an image', true); return; }
  let bmp;
  try { bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' }); }
  catch (_) { if (token === state.loadId) setStatus('Could not decode that image. Try JPG, PNG or WebP.', true); return; }
  if (token !== state.loadId) { bmp.close(); return; }
  state.runId++;
  clearEdits();
  state.name = (name || 'photo').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '-').slice(0, 40) || 'photo';
  const [W, H] = dimensions(bmp.width, bmp.height, MAX_SIDE);
  state.W = W; state.H = H;
  for (const c of [state.src, state.cut, el.before, el.after]) { c.width = W; c.height = H; }
  const sctx = state.src.getContext('2d');
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(bmp, 0, 0, W, H);
  bmp.close && bmp.close();
  el.before.getContext('2d').drawImage(state.src, 0, 0);
  state.hasMask = false;

  el.drop.hidden = true;
  el.stage.hidden = false;
  el.stage.style.aspectRatio = W + ' / ' + H;
  el.stage.style.width = 'min(100%, calc(var(--stage-h) * ' + (W / H).toFixed(4) + '))';
  el.stage.classList.add('working');
  el.download.disabled = true;
  el.reset.disabled = false;
  el.sImg.textContent = W + '×' + H;
  $('#manual').disabled = false;
  updateExportInfo();
  if ($('#useAI').checked) segment(); else startManual();
}

async function segment() {
  clearEdits();
  el.download.disabled = true;
  const id = ++state.runId;
  cutout.busy = true;
  const s = Math.min(1, MODEL_SIDE / Math.max(state.W, state.H));
  const w = Math.max(1, Math.round(state.W * s)), h = Math.max(1, Math.round(state.H * s));
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d'); ctx.imageSmoothingQuality = 'high'; ctx.drawImage(state.src, 0, 0, w, h);
  const pixels = ctx.getImageData(0, 0, w, h).data.buffer;
  el.busyText.textContent = state.info ? 'segmenting' : 'loading model';
  try { await ensureWorker(); } catch (err) {
    if (id === state.runId) { cutout.busy = false; el.stage.classList.remove('working'); setProgress(null); setStatus('AI unavailable. Start manually below, or reopen the photo to retry.', true); }
    return;
  }
  if (id !== state.runId) return;
  el.busyText.textContent = 'segmenting';
  setStatus('segmenting on ' + (state.info.device === 'webgpu' ? 'webgpu' : 'wasm ×' + state.info.threads) + '…');
  state.t0 = performance.now();
  state.worker.postMessage({ type: 'run', id, pixels, width: w, height: h }, [pixels]);
}

function onMask(m) {
  if (m.id !== state.runId) return;
  const { width: w, height: h, mask } = m;
  for (const c of [state.maskA, state.maskG]) { c.width = w; c.height = h; }
  const a = new ImageData(w, h), g = new ImageData(w, h);
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    const v = mask[i];
    a.data[j] = a.data[j + 1] = a.data[j + 2] = 255; a.data[j + 3] = v;
    g.data[j] = g.data[j + 1] = g.data[j + 2] = v; g.data[j + 3] = 255;
  }
  state.maskA.getContext('2d').putImageData(a, 0, 0);
  state.maskG.getContext('2d').putImageData(g, 0, 0);

  const cctx = state.cut.getContext('2d');
  cctx.globalCompositeOperation = 'source-over';
  cctx.clearRect(0, 0, state.W, state.H);
  cctx.drawImage(state.src, 0, 0);
  cctx.globalCompositeOperation = 'destination-in';
  cctx.imageSmoothingEnabled = true; cctx.imageSmoothingQuality = 'high';
  cctx.drawImage(state.maskA, 0, 0, state.W, state.H);
  cctx.globalCompositeOperation = 'source-over';
  state.hasMask = true;
  $('#manual').disabled = true;
  const scaled = document.createElement('canvas'); scaled.width = state.W; scaled.height = state.H;
  scaled.getContext('2d').drawImage(state.maskA, 0, 0, state.W, state.H);
  const rgba = scaled.getContext('2d').getImageData(0, 0, state.W, state.H).data;
  state.alpha = Uint8ClampedArray.from({ length: state.W * state.H }, (_, i) => rgba[i * 4 + 3]);
  updateEditControls();

  const t = m.timings;
  const wall = performance.now() - state.t0;
  el.sInfer.textContent = fmtS(t.infer) + (t.run === 1 ? ' (first run)' : '');
  el.sPre.textContent = fmtS(t.pre) + ' / ' + fmtS(t.post);
  const rec = { name: state.name, run: t.run, device: state.info.device, threads: state.info.threads, isolated: state.info.isolated,
    loadMs: state.info.loadMs, pre: t.pre, infer: t.infer, post: t.post, wall, image: [state.W, state.H] };
  cutout.results.push(rec);
  setStatus('done · ' + fmtS(t.infer) + ' inference');
  el.stage.classList.remove('working');
  el.download.disabled = false;
  cutout.busy = false;
  render();
}

/* ---------------- compositing ---------------- */
function paintBackground(ctx, W, H) {
  if (state.mode === 'color') {
    ctx.fillStyle = state.color; ctx.fillRect(0, 0, W, H);
  } else if (state.mode === 'gradient') {
    let g;
    if (state.grad === 'studio') {
      g = ctx.createRadialGradient(W / 2, H * 0.4, 0, W / 2, H * 0.4, Math.max(W, H) * 0.75);
      g.addColorStop(0, '#f2f1ed'); g.addColorStop(1, '#b9b5ab');
    } else if (state.grad === 'ember') {
      g = ctx.createLinearGradient(0, 0, W * 0.35, H);
      g.addColorStop(0, '#ffd9c4'); g.addColorStop(1, '#ff7a45');
    } else {
      g = ctx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, '#3a3833'); g.addColorStop(1, '#121211');
    }
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  } else if (state.mode === 'blur') {
    // Slider value is in on-screen pixels; convert to image pixels so the look matches the preview.
    const shown = el.stage.getBoundingClientRect().width || W;
    const r = state.blur * (W / shown);
    const pad = r * 2;
    if ('filter' in ctx) {
      ctx.filter = 'blur(' + r.toFixed(1) + 'px)';
      ctx.drawImage(state.src, -pad, -pad, W + pad * 2, H + pad * 2);
      ctx.filter = 'none';
    } else {
      const k = Math.max(1, Math.round(r / 2));
      const t = document.createElement('canvas'); t.width = Math.max(1, W / k | 0); t.height = Math.max(1, H / k | 0);
      t.getContext('2d').drawImage(state.src, 0, 0, t.width, t.height);
      ctx.imageSmoothingQuality = 'high'; ctx.drawImage(t, 0, 0, W, H);
    }
  }
}

function composite(ctx, W, H) {
  ctx.clearRect(0, 0, W, H);
  paintBackground(ctx, W, H);
  ctx.drawImage(state.cut, 0, 0, W, H);
}

function render() {
  if (!state.W) return;
  const ctx = el.after.getContext('2d');
  const W = state.W, H = state.H;
  if (!state.hasMask) { ctx.clearRect(0, 0, W, H); return; }
  if (state.view === 'mask') {
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(state.maskG, 0, 0, W, H);
  } else {
    composite(ctx, W, H);
  }
  el.checker.hidden = !(state.mode === 'transparent' && state.view !== 'mask');
  el.stage.classList.toggle('nosplit', state.view !== 'split');
}

/* ---------------- divider ---------------- */
function setX(x) {
  state.x = Math.max(0, Math.min(100, x));
  el.stage.style.setProperty('--x', state.x + '%');
  el.stage.setAttribute('aria-valuenow', Math.round(state.x));
}
let dragging = false;
const fromEvent = (e) => { const r = el.stage.getBoundingClientRect(); setX((e.clientX - r.left) / r.width * 100); };
el.stage.addEventListener('pointerdown', (e) => { if (state.tool !== 'compare' || state.view !== 'split' || cutout.busy) return; dragging = true; el.stage.setPointerCapture(e.pointerId); fromEvent(e); });
el.stage.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
el.stage.addEventListener('pointerup', () => { dragging = false; });
el.stage.addEventListener('pointercancel', () => { dragging = false; });
el.stage.addEventListener('keydown', (e) => {
  if (state.tool !== 'compare') return;
  const step = e.shiftKey ? 10 : 2;
  if (e.key === 'ArrowLeft') { setX(state.x - step); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { setX(state.x + step); e.preventDefault(); }
  else if (e.key === 'Home') { setX(0); e.preventDefault(); }
  else if (e.key === 'End') { setX(100); e.preventDefault(); }
});
setX(50);

/* ---------------- controls ---------------- */
function pressGroup(container, btn) { $$('button', container).forEach((b) => b.setAttribute('aria-pressed', String(b === btn))); }

$('#bgMode').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  pressGroup($('#bgMode'), b);
  state.mode = b.dataset.mode;
  $$('.opts').forEach((o) => o.classList.toggle('on', o.dataset.for === state.mode));
  render();
});
$('#swatches').addEventListener('click', (e) => {
  const b = e.target.closest('button.sw'); if (!b) return;
  $$('.sw', $('#swatches')).forEach((s) => s.setAttribute('aria-pressed', String(s === b)));
  state.color = b.dataset.c; render();
});
el.custom.addEventListener('input', () => {
  $$('.sw', $('#swatches')).forEach((s) => s.setAttribute('aria-pressed', 'false'));
  el.custom.parentElement.style.background = el.custom.value;
  state.color = el.custom.value; render();
});
$('#grads').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  pressGroup($('#grads'), b); state.grad = b.dataset.g; render();
});
let blurRaf = 0;
el.blurAmt.addEventListener('input', () => {
  state.blur = +el.blurAmt.value; el.blurOut.textContent = state.blur + 'px';
  cancelAnimationFrame(blurRaf); blurRaf = requestAnimationFrame(render);
});
$('#viewMode').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  setTool('compare');
  pressGroup($('#viewMode'), b); state.view = b.dataset.view; render();
});
$('#backendPref').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b || b.dataset.pref === state.pref) return;
  pressGroup($('#backendPref'), b);
  state.pref = b.dataset.pref;
  if (state.worker) state.worker.terminate();
  state.worker = null; state.ready = null; state.info = null;
  for (const k of ['sBackend', 'sIso', 'sLoad', 'sInfer', 'sPre']) el[k].textContent = '—';
  el.prefNote.textContent = state.pref === 'webgpu'
    ? 'Trying WebGPU (fp16, ~88 MB). On my machine ONNX Runtime rejects one MaxPool op on WebGPU, so expect a fallback to WASM.'
    : 'WASM with int8 weights (~44 MB) and one thread per core, up to 8, when the page is cross-origin isolated.';
  if (state.W && $('#useAI').checked) { el.stage.classList.add('working'); segment(); } else setStatus('backend set to ' + state.pref);
});
$$('#backendPref button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.pref === state.pref)));

function updateExportInfo() {
  const scale = +$('#exportScale').value;
  $('#exportInfo').textContent = state.W ? `${Math.max(1, Math.round(state.W * scale))} × ${Math.max(1, Math.round(state.H * scale))} px · ${$('#exportFormat').value.toUpperCase()}` : 'Up to 2048 px · transparency preserved';
}
$('#exportScale').onchange = updateExportInfo;
$('#exportFormat').onchange = updateExportInfo;
el.download.addEventListener('click', () => {
  if (!state.hasMask || cutout.busy) return;
  cancelAnimationFrame(editFrame); applyAlpha();
  const c = document.createElement('canvas');
  const scale = +$('#exportScale').value;
  c.width = Math.max(1, Math.round(state.W * scale)); c.height = Math.max(1, Math.round(state.H * scale));
  // Composite at working resolution, then downsample to keep blur identical at all export sizes.
  const full = document.createElement('canvas'); full.width = state.W; full.height = state.H;
  composite(full.getContext('2d'), state.W, state.H);
  const ctx = c.getContext('2d'); ctx.imageSmoothingQuality = 'high'; ctx.drawImage(full, 0, 0, c.width, c.height);
  const format = $('#exportFormat').value;
  const name = exportName(state.name, state.mode, format);
  c.toBlob((blob) => {
    if (!blob) { setStatus('Export failed. Try a smaller size or PNG.', true); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = blob.type === 'image/png' ? name.replace(/\.webp$/, '.png') : name;
    document.body.appendChild(a); a.click(); a.remove();
    setStatus(`Exported ${c.width} × ${c.height} · ${mb(blob.size)} MB`);
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/' + format, .92);
});
el.reset.addEventListener('click', () => {
  if (cutout.busy && state.worker) { state.worker.terminate(); state.worker = null; state.ready = null; state.info = null; setProgress(null); }
  state.loadId++; clearEdits(); $('#manual').disabled = true;
  state.runId++; state.W = 0; state.hasMask = false; cutout.busy = false;
  el.stage.hidden = true; el.drop.hidden = false; el.stage.classList.remove('working');
  el.download.disabled = true; el.reset.disabled = true; el.sImg.textContent = '—';
  $$('.thumb').forEach((t) => t.setAttribute('aria-pressed', 'false'));
  updateExportInfo();
  setStatus(state.info ? 'model ready · drop another photo' : 'model not loaded · ~44 mb on first run');
});

// file picker, drag & drop, paste, samples
const pick = () => el.file.click();
$('#pick').addEventListener('click', pick);
$('#pick2').addEventListener('click', pick);
el.file.addEventListener('change', () => { const f = el.file.files[0]; if (f) { $$('.thumb').forEach((t) => t.setAttribute('aria-pressed', 'false')); loadBlob(f, f.name); } el.file.value = ''; });

let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; el.drop.classList.add('over'); el.stageWrap.classList.add('over'); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; el.drop.classList.remove('over'); el.stageWrap.classList.remove('over'); } });
window.addEventListener('drop', (e) => {
  e.preventDefault(); dragDepth = 0; el.drop.classList.remove('over'); el.stageWrap.classList.remove('over');
  const f = [...(e.dataTransfer && e.dataTransfer.files || [])].find((x) => /^image\//.test(x.type));
  if (f) { $$('.thumb').forEach((t) => t.setAttribute('aria-pressed', 'false')); loadBlob(f, f.name); }
  else setStatus('drop an image file (jpg, png, webp)', true);
});
window.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData && e.clipboardData.items || [])].find((i) => /^image\//.test(i.type));
  if (item) loadBlob(item.getAsFile(), 'pasted');
});

async function runSample(name) {
  if (!['portrait', 'dog', 'watch'].includes(name)) { setStatus('Unknown sample. Choose one of the three photos.', true); return; }
  const token = ++state.loadId;
  $$('.thumb').forEach((t) => t.setAttribute('aria-pressed', String(t.dataset.sample === name)));
  try {
    const res = await fetch('photos/' + name + '.jpg');
    if (!res.ok) throw new Error('Photo unavailable');
    await loadBlob(await res.blob(), name, token);
  } catch (_) { if (token === state.loadId) setStatus('Could not open sample. Try your own photo.', true); }
}
$$('.thumb').forEach((t) => t.addEventListener('click', () => runSample(t.dataset.sample)));
cutout.runSample = runSample;

// theme toggle (shared with the portfolio)
const root = document.documentElement;
$('#theme').onclick = () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme:dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
};

el.sIso.textContent = window.crossOriginIsolated ? 'yes' : 'no (single-threaded)';
if (params.get('sample')) runSample(params.get('sample'));

/* ---------------- refinement workbench ---------------- */
let stroke = null;
let editFrame = 0;
function clearEdits() {
  stroke = null; cancelAnimationFrame(editFrame);
  state.alpha = null; state.history.clear(); state.hasMask = false;
  setTool('compare'); updateEditControls();
}
function updateEditControls() {
  $('#undo').disabled = !state.history.past.length;
  $('#redo').disabled = !state.history.future.length;
  $$('#toolMode button').forEach(b => { b.disabled = b.dataset.tool !== 'compare' && !state.alpha; });
}
function setTool(tool) {
  state.tool = tool;
  $$('#toolMode button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tool === tool)));
  $('#brushControls').hidden = tool === 'compare';
  $('#brushCursor').hidden = true;
  el.stage.classList.toggle('editing', tool !== 'compare');
  el.stage.setAttribute('role', tool === 'compare' ? 'slider' : 'group');
  el.stage.setAttribute('aria-label', tool === 'compare' ? 'Before / after divider' : `${tool} brush canvas. Arrow keys move brush, Enter paints.`);
  for (const attr of ['aria-valuemin', 'aria-valuemax', 'aria-valuenow']) el.stage.removeAttribute(attr);
  if (tool === 'compare') {
    state.view = 'split';
    $$('#viewMode button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === 'split')));
    el.stage.setAttribute('aria-valuemin', '0'); el.stage.setAttribute('aria-valuemax', '100'); setX(state.x);
  } else {
    state.view = 'result';
    $$('#viewMode button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === 'result')));
    state.cursor = { x: state.W / 2, y: state.H / 2 };
  }
  render();
}
function applyAlpha() {
  if (!state.alpha) return;
  for (const c of [state.maskA, state.maskG]) { c.width = state.W; c.height = state.H; }
  const a = new ImageData(state.W, state.H), g = new ImageData(state.W, state.H);
  for (let i = 0; i < state.alpha.length; i++) {
    const j = i * 4, v = state.alpha[i];
    a.data[j] = a.data[j + 1] = a.data[j + 2] = 255; a.data[j + 3] = v;
    g.data[j] = g.data[j + 1] = g.data[j + 2] = v; g.data[j + 3] = 255;
  }
  state.maskA.getContext('2d').putImageData(a, 0, 0);
  state.maskG.getContext('2d').putImageData(g, 0, 0);
  const ctx = state.cut.getContext('2d');
  ctx.clearRect(0, 0, state.W, state.H); ctx.drawImage(state.src, 0, 0);
  ctx.globalCompositeOperation = 'destination-in'; ctx.drawImage(state.maskA, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  render(); updateEditControls();
}
function startManual() {
  if (!state.W) return;
  state.runId++;
  if (state.worker && cutout.busy) { state.worker.terminate(); state.worker = null; state.ready = null; state.info = null; }
  clearEdits();
  state.alpha = new Uint8ClampedArray(state.W * state.H).fill(255);
  state.hasMask = true; cutout.busy = false;
  el.stage.classList.remove('working'); setProgress(null);
  el.download.disabled = false; $('#manual').disabled = true;
  applyAlpha(); setTool('erase');
  setStatus('Manual mode · erase the background, restore any detail');
}
$('#manual').onclick = startManual;
$('#toolMode').onclick = e => { const b = e.target.closest('button'); if (b && !b.disabled) setTool(b.dataset.tool); };
$('#brushSize').oninput = e => { state.brush = +e.target.value; $('#brushOut').textContent = state.brush + ' px'; };
$('#brushSoftness').oninput = e => { state.softness = +e.target.value / 100; $('#softOut').textContent = e.target.value + '%'; };
function historyStep(direction) {
  if (!state.alpha) return;
  state.alpha = state.history[direction](state.alpha); applyAlpha();
  setStatus(direction === 'undo' ? 'Mask edit undone' : 'Mask edit restored');
}
$('#undo').onclick = () => historyStep('undo');
$('#redo').onclick = () => historyStep('redo');
function point(e) {
  const r = el.stage.getBoundingClientRect();
  return { x: Math.max(0, Math.min(state.W, (e.clientX - r.left) / r.width * state.W)), y: Math.max(0, Math.min(state.H, (e.clientY - r.top) / r.height * state.H)) };
}
function showCursor(p) {
  const c = $('#brushCursor'); c.hidden = false;
  c.style.left = p.x / state.W * 100 + '%'; c.style.top = p.y / state.H * 100 + '%';
  c.style.width = state.brush / state.W * 100 + '%'; c.style.height = state.brush / state.H * 100 + '%';
}
function paint(from, to) {
  paintMask(state.alpha, state.W, state.H, from, to, state.brush / 2, state.tool === 'restore', state.softness);
  cancelAnimationFrame(editFrame); editFrame = requestAnimationFrame(applyAlpha);
}
el.stage.addEventListener('pointerdown', e => {
  if (state.tool === 'compare' || !state.alpha || e.button !== 0) return;
  e.preventDefault(); el.stage.focus(); el.stage.setPointerCapture(e.pointerId);
  state.history.save(state.alpha); stroke = state.cursor = point(e); paint(stroke, stroke); showCursor(stroke);
});
el.stage.addEventListener('pointermove', e => {
  if (state.tool === 'compare' || !state.alpha) return;
  const p = state.cursor = point(e); showCursor(p);
  if (stroke) { paint(stroke, p); stroke = p; }
});
function finishStroke() { if (stroke) { stroke = null; cancelAnimationFrame(editFrame); applyAlpha(); setStatus('Mask refined · undo is available'); } }
el.stage.addEventListener('pointerup', finishStroke);
el.stage.addEventListener('pointercancel', finishStroke);
el.stage.addEventListener('lostpointercapture', finishStroke);
el.stage.addEventListener('pointerleave', () => { if (!stroke) $('#brushCursor').hidden = true; });
el.stage.addEventListener('keydown', e => {
  if (state.tool === 'compare' || !state.alpha) return;
  const step = e.shiftKey ? 40 : 5, p = state.cursor;
  if (e.key === 'ArrowLeft') p.x = Math.max(0, p.x - step);
  else if (e.key === 'ArrowRight') p.x = Math.min(state.W, p.x + step);
  else if (e.key === 'ArrowUp') p.y = Math.max(0, p.y - step);
  else if (e.key === 'ArrowDown') p.y = Math.min(state.H, p.y + step);
  else if (e.key === 'Enter' || e.key === ' ') { state.history.save(state.alpha); paint(p, p); }
  else return;
  showCursor(p); e.preventDefault();
});
window.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && state.alpha) { e.preventDefault(); historyStep(e.shiftKey ? 'redo' : 'undo'); }
});
