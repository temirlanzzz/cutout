// Deterministic mask operations, independent of DOM, model and network.
export function dimensions(width, height, max = 2048) {
  const scale = Math.min(1, max / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

export function paintMask(mask, width, height, from, to, radius, restore = false, softness = .35) {
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * .3)));
  for (let step = 0; step <= steps; step++) {
    const x = from.x + (to.x - from.x) * step / steps;
    const y = from.y + (to.y - from.y) * step / steps;
    for (let py = Math.max(0, Math.floor(y - radius)); py < Math.min(height, Math.ceil(y + radius)); py++) {
      for (let px = Math.max(0, Math.floor(x - radius)); px < Math.min(width, Math.ceil(x + radius)); px++) {
        const d = Math.hypot(px + .5 - x, py + .5 - y) / radius;
        if (d >= 1) continue;
        const strength = Math.min(1, (1 - d) / Math.max(.001, softness));
        const i = py * width + px;
        mask[i] = restore ? Math.max(mask[i], Math.round(255 * strength)) : Math.min(mask[i], Math.round(255 * (1 - strength)));
      }
    }
  }
  return mask;
}

export class MaskHistory {
  constructor(limit = 12) { this.limit = limit; this.clear(); }
  clear() { this.past = []; this.future = []; }
  save(mask) { this.past.push(mask.slice()); if (this.past.length > this.limit) this.past.shift(); this.future = []; }
  undo(mask) { if (!this.past.length) return mask; this.future.push(mask.slice()); return this.past.pop(); }
  redo(mask) { if (!this.future.length) return mask; this.past.push(mask.slice()); return this.future.pop(); }
}

export function exportName(name, mode, format) {
  const safe = name.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '-').slice(0, 40) || 'photo';
  return `cutout-${safe}${mode === 'transparent' ? '' : '-' + mode}.${format === 'webp' ? 'webp' : 'png'}`;
}
