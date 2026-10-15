// Optional real-browser regression suite. Run: node tests/browser.mjs
// Requires Playwright and a Chromium installation; no model/network is needed.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';
import assert from 'node:assert/strict';
const { chromium } = createRequire(import.meta.url)('playwright');
const root = resolve(import.meta.dirname, '..');
const server = createServer(async (req, res) => {
  try {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    const file = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(root + '/')) throw new Error('Invalid path');
    const body = await readFile(file);
    res.setHeader('Content-Type', ({'.html':'text/html','.js':'text/javascript','.css':'text/css','.jpg':'image/jpeg'})[extname(file)] || 'application/octet-stream');
    res.end(body);
  } catch { res.statusCode = 404; res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
let browser;
try {
  browser = await chromium.launch({headless:true, ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
  const page = await browser.newPage({viewport:{width:1440,height:1000}, reducedMotion:'reduce'});
  const errors=[]; page.on('pageerror', e => errors.push(e.message));
  const external=[];
  await page.route('**/*', route => {
    if (!route.request().url().startsWith('http://127.0.0.1')) { external.push(route.request().url()); return route.abort(); }
    return route.continue();
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/?noiso`);
  await page.locator('#useAI').uncheck();
  await page.locator('[data-sample="watch"]').click();
  await page.waitForFunction(() => !document.querySelector('#download').disabled);
  assert.equal(await page.locator('[data-tool="erase"]').getAttribute('aria-pressed'), 'true');
  // Compare actual alpha-plane pixels, independent of button/UI state or PNG encoding.
  const alphaSnapshot = () => page.locator('#after').evaluate(async c => {
    const rgba = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const alpha = Uint8Array.from({length:c.width*c.height}, (_, i) => rgba[i*4+3]);
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', alpha))].map(v => v.toString(16).padStart(2,'0')).join('');
    return {width:c.width, height:c.height, digest, center:alpha[Math.floor(c.height*.5)*c.width+Math.floor(c.width*.6)], corner:alpha[0]};
  });
  const before = await alphaSnapshot();
  assert.equal(before.center, 255); assert.equal(before.corner, 255);
  const box = await page.locator('#stage').boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
  await page.mouse.down(); await page.mouse.move(box.x+box.width*.7,box.y+box.height/2,{steps:5}); await page.mouse.up();
  const edited = await alphaSnapshot();
  assert.notEqual(before.digest, edited.digest);
  assert.equal(edited.center, 0, 'The continuous brush stroke must fully erase its center');
  assert.equal(edited.corner, 255, 'The brush must not change distant pixels');
  await page.locator('#undo').click();
  assert.deepEqual(await alphaSnapshot(),before, 'Undo restores every alpha pixel');
  await page.locator('#redo').click();
  assert.deepEqual(await alphaSnapshot(),edited, 'Redo restores every edited alpha pixel');
  await page.locator('[data-tool="compare"]').click();
  assert.equal(await page.locator('[data-view="split"]').getAttribute('aria-pressed'),'true');
  await page.locator('[data-view="mask"]').click();
  for (const scale of [1, .5]) {
    await page.locator('#exportScale').selectOption(String(scale));
    for (const format of ['png','webp']) {
      await page.locator('#exportFormat').selectOption(format);
      const downloadPromise=page.waitForEvent('download'); await page.locator('#download').click();
      const download=await downloadPromise;
      assert(download.suggestedFilename().endsWith('.'+format));
      const bytes = await readFile(await download.path());
      // Check the downloaded file signature, then decode its actual bytes in Chromium.
      if (format === 'png') assert.equal(bytes.subarray(0,8).toString('hex'), '89504e470d0a1a0a');
      else { assert.equal(bytes.subarray(0,4).toString(), 'RIFF'); assert.equal(bytes.subarray(8,12).toString(), 'WEBP'); }
      const decoded = await page.evaluate(async ({base64, format}) => {
        const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], {type:'image/'+format}));
        const canvas = document.createElement('canvas'); canvas.width=bitmap.width; canvas.height=bitmap.height;
        const ctx=canvas.getContext('2d'); ctx.drawImage(bitmap,0,0); bitmap.close();
        const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
        let transparent=0, partial=0;
        for(let i=3;i<pixels.length;i+=4) { if(pixels[i]===0) transparent++; else if(pixels[i]<255) partial++; }
        const center=(Math.floor(canvas.height*.5)*canvas.width+Math.floor(canvas.width*.6))*4;
        return {width:canvas.width,height:canvas.height,center:pixels[center+3],corner:pixels[3],transparent,partial};
      }, {base64:bytes.toString('base64'), format});
      assert.equal(decoded.width, Math.max(1,Math.round(edited.width*scale)), `${format}: exported width at ${scale} scale`);
      assert.equal(decoded.height, Math.max(1,Math.round(edited.height*scale)), `${format}: exported height at ${scale} scale`);
      assert.equal(decoded.center, 0, `${format}: edited hole stays transparent, even when exporting from mask view`);
      assert.equal(decoded.corner, 255, `${format}: untouched source stays opaque`);
      assert(decoded.transparent > 0, `${format}: file contains erased pixels`);
      assert(decoded.partial > 0, `${format}: file preserves the soft brush edge`);
    }
  }
  await page.locator('[data-view="result"]').click();
  await page.screenshot({path:resolve(root,'docs/studio-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:resolve(root,'docs/studio-mobile.png'),fullPage:true});
  const overflow = await page.evaluate(() => ({
    viewport:innerWidth, document:document.documentElement.scrollWidth,
    elements:[...document.querySelectorAll('body *')].filter(el => {
      const r=el.getBoundingClientRect(); return r.width>0 && (r.right>innerWidth+1 || r.left< -1);
    }).map(el=>({tag:el.tagName,id:el.id,class:el.className,width:el.getBoundingClientRect().width})).slice(0,20),
  }));
  assert(overflow.document <= overflow.viewport, JSON.stringify(overflow));
  let release;
  const gate = new Promise(r => { release = r; });
  await page.route('**/photos/portrait.jpg', async route => { await gate; await route.continue(); });
  await page.locator('[data-sample="portrait"]').click();
  await page.locator('#reset').click();
  const pendingResponse = page.waitForResponse(r => r.url().endsWith('/photos/portrait.jpg'));
  release(); await pendingResponse;
  await page.waitForTimeout(300);
  assert(await page.locator('#drop').isVisible()); assert(await page.locator('#undo').isDisabled());
  assert.deepEqual(errors,[]);
  assert(!external.some(u => /huggingface|jsdelivr/.test(u)), 'Manual mode must not request model assets');
  console.log('PASS: manual edit, alpha-pixel undo/redo, compare, decoded PNG/WebP dimensions and transparency at full/half scale, mobile overflow, clear and pending-load cancellation');
} finally { await browser?.close(); server.close(); }
