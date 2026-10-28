// ---------------------------------------------------------------------------
// Trail: a soft "tyre mark" memory of where the car has been, fading with age.
//
// Segments live in a ring buffer and are drawn oldest -> newest. Age is measured in
// simulation steps, so the fade does not depend on the speed slider. Segments are grouped
// into a few alpha buckets and each bucket is stroked as ONE path, which keeps it cheap
// (a handful of canvas strokes per frame) and stops overlapping segments from stacking
// within a bucket.
// ---------------------------------------------------------------------------
class Trail {
  constructor({ capacity = 3000, life = 3000, maxAlpha = 0.22, width = 9, rgb = '40, 90, 200' } = {}) {
    this.cap = capacity;
    this.life = life;            // steps until a mark has fully faded
    this.maxAlpha = maxAlpha;    // opacity of a brand-new mark (kept low on purpose)
    this.width = width;
    this.rgb = rgb;
    this.xy = new Float32Array(capacity * 4);
    this.tick = new Float64Array(capacity);
    this.head = 0;
    this.count = 0;
  }

  clear() { this.head = 0; this.count = 0; }

  add(x1, y1, x2, y2, tick) {
    const o = this.head * 4;
    this.xy[o] = x1; this.xy[o + 1] = y1; this.xy[o + 2] = x2; this.xy[o + 3] = y2;
    this.tick[this.head] = tick;
    this.head = (this.head + 1) % this.cap;
    if (this.count < this.cap) this.count++;
  }

  draw(nowTick) {
    if (!this.count) return;
    const ctx = drawingContext, B = 24;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = this.width;

    let cur = -1;
    const start = (this.head - this.count + this.cap) % this.cap;
    for (let k = 0; k < this.count; k++) {
      const i = (start + k) % this.cap;
      const age = nowTick - this.tick[i];
      if (age >= this.life) continue;
      const b = Math.min(B - 1, Math.floor((age / this.life) * B));
      if (b !== cur) {
        if (cur >= 0) ctx.stroke();
        const t = (b + 0.5) / B;                       // 0 = fresh, 1 = gone
        ctx.strokeStyle = `rgba(${this.rgb}, ${(this.maxAlpha * Math.pow(1 - t, 1.6)).toFixed(3)})`;
        ctx.beginPath();
        cur = b;
      }
      const o = i * 4;
      ctx.moveTo(this.xy[o], this.xy[o + 1]);
      ctx.lineTo(this.xy[o + 2], this.xy[o + 3]);
    }
    if (cur >= 0) ctx.stroke();
    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// LiveChart: scrolling line chart with one or more series, drawn with p5.
//   min / max given  -> fixed y range (e.g. throttle 0..1)
//   min / max absent -> auto range with a little padding
// Newest value is at the right edge.
// ---------------------------------------------------------------------------
class LiveChart {
  constructor({ title, len = 300, min, max, series = [{ color: '#1f6feb' }], digits = 2, robust = false }) {
    this.robust = robust;   // auto-range from the 3rd..97th percentile so rare spikes don't flatten the chart
    this.title = title;
    this.len = len;
    this.min = min;
    this.max = max;
    this.digits = digits;
    this.series = series.map(s => ({ color: s.color, name: s.name || '', data: [] }));
  }

  push(v, si = 0) {
    if (!Number.isFinite(v)) return;
    const d = this.series[si].data;
    d.push(v);
    if (d.length > this.len) d.shift();
  }

  last(si = 0) {
    const d = this.series[si].data;
    return d.length ? d[d.length - 1] : null;
  }

  toJSON() {
    const r = v => Math.round(v * 1e5) / 1e5;
    return { title: this.title, series: this.series.map(s => s.data.map(r)) };
  }
  load(o) {
    if (!o || !o.series) return;
    o.series.forEach((arr, i) => { if (this.series[i]) this.series[i].data = arr.slice(-this.len); });
  }

  draw(x, y, w, h) {
    push();
    translate(x, y);
    const top = 18, pad = 4;
    const pw = w - 2 * pad, ph = h - top - pad;

    // frame
    fill(255); stroke(190); strokeWeight(1);
    rect(0, 0, w, h, 3);

    // y range
    let lo = this.min, hi = this.max;
    if (lo === undefined || hi === undefined) {
      let a = Infinity, b = -Infinity;
      if (this.robust) {
        const all = this.series.flatMap(s => s.data).sort((p, q) => p - q);
        if (all.length) { a = all[Math.floor(0.03 * (all.length - 1))]; b = all[Math.ceil(0.97 * (all.length - 1))]; }
      } else {
        for (const s of this.series) for (const v of s.data) { if (v < a) a = v; if (v > b) b = v; }
      }
      if (a === Infinity) { a = 0; b = 1; }
      if (a === b) { a -= 0.5; b += 0.5; }
      const m = (b - a) * 0.08;
      lo = lo ?? a - m; hi = hi ?? b + m;
    }
    const yOf = v => pad + top + ph - ((clamp(v, lo, hi) - lo) / (hi - lo)) * ph;

    // zero line (if in range)
    if (lo < 0 && hi > 0) {
      stroke(225); line(pad, yOf(0), w - pad, yOf(0));
    }

    // series
    const dx = pw / (this.len - 1);
    noFill(); strokeWeight(1.4);
    for (const s of this.series) {
      if (s.data.length < 2) continue;
      stroke(s.color);
      beginShape();
      const n = s.data.length;
      for (let i = 0; i < n; i++) vertex(pad + pw - (n - 1 - i) * dx, yOf(s.data[i]));
      endShape();
    }

    // labels
    noStroke(); textSize(11);
    fill(60); textAlign(LEFT, TOP); text(this.title, 6, 3);
    const lv = this.last(0);
    fill(this.series[0].color); textAlign(RIGHT, TOP);
    text(lv === null ? '-' : lv.toFixed(this.digits), w - 6, 3);
    fill(150); textSize(9);
    textAlign(LEFT, TOP);    text(hi.toFixed(this.digits), 6, top + 1);
    textAlign(LEFT, BOTTOM); text(lo.toFixed(this.digits), 6, h - 1);
    pop();
  }
}
