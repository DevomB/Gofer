/* Gofer landing page. No framework: canvas boards, inline SVG charts and the
   real engine (engine.js) for the play section. Data comes from web/data/,
   written by `go run -tags onnx ./cmd/sitedata`. */
(function () {
  'use strict';

  const EMPTY = 0, BLACK = 1, WHITE = 2, PASS = -1, SIZE = 9, N = SIZE * SIZE, KOMI = 7.5;
  const COLS = 'ABCDEFGHJ';
  const RED = '#d63c1b';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = id => document.getElementById(id);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const pct = x => Math.round(100 * x) + '%';
  const vertex = p => p < 0 ? 'pass' : COLS[p % SIZE] + (SIZE - ((p / SIZE) | 0));
  const fetchJSON = url => fetch(url).then(r => { if (!r.ok) throw new Error(url + ': HTTP ' + r.status); return r.json(); });

  function onVisible(el, fn, opts) {
    const io = new IntersectionObserver(es => es.forEach(e => fn(e.isIntersecting, e)), opts || { threshold: 0.25 });
    io.observe(el);
    return io;
  }

  /* ============================================================
     rules, enough to replay recorded games
     ============================================================ */
  const NB = [];
  for (let p = 0; p < N; p++) {
    const x = p % SIZE, y = (p / SIZE) | 0, l = [];
    if (x > 0) l.push(p - 1);
    if (x < SIZE - 1) l.push(p + 1);
    if (y > 0) l.push(p - SIZE);
    if (y < SIZE - 1) l.push(p + SIZE);
    NB.push(l);
  }
  function chain(cells, p) {
    const c = cells[p], seen = new Set([p]), stack = [p];
    let libs = 0;
    const libSeen = new Set();
    while (stack.length) {
      const q = stack.pop();
      for (const r of NB[q]) {
        if (cells[r] === EMPTY) { if (!libSeen.has(r)) { libSeen.add(r); libs++; } }
        else if (cells[r] === c && !seen.has(r)) { seen.add(r); stack.push(r); }
      }
    }
    return { stones: seen, libs };
  }
  // play returns a new position and the number of stones it captured.
  function play(cells, p, color) {
    const next = Int8Array.from(cells);
    if (p < 0) return { cells: next, captured: 0 };
    next[p] = color;
    let captured = 0;
    for (const r of NB[p]) {
      if (next[r] === 3 - color) {
        const ch = chain(next, r);
        if (ch.libs === 0) for (const q of ch.stones) { next[q] = EMPTY; captured++; }
      }
    }
    const own = chain(next, p);
    if (own.libs === 0) for (const q of own.stones) next[q] = EMPTY;
    return { cells: next, captured };
  }
  // replay turns a move list into the positions after every move.
  function replay(moves, first) {
    let cells = new Int8Array(N), color = first || BLACK;
    const out = [{ cells, last: -2, capB: 0, capW: 0 }];
    let capB = 0, capW = 0;
    for (const m of moves) {
      const r = play(cells, m, color);
      if (color === BLACK) capB += r.captured; else capW += r.captured;
      cells = r.cells;
      out.push({ cells, last: m, capB, capW });
      color = 3 - color;
    }
    return out;
  }

  /* ============================================================
     the board renderer
     ============================================================ */
  function rng(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  const woodCache = new Map();
  function wood(w, seed) {
    const key = w + ':' + seed;
    if (woodCache.has(key)) return woodCache.get(key);
    const c = document.createElement('canvas');
    c.width = c.height = w;
    const g = c.getContext('2d'), r = rng(seed), k = w / 600;
    const base = g.createLinearGradient(0, 0, w, w * .8);
    base.addColorStop(0, '#ebc684'); base.addColorStop(.55, '#e2b670'); base.addColorStop(1, '#d6a65c');
    g.fillStyle = base; g.fillRect(0, 0, w, w);
    for (let i = 0; i < 110; i++) {
      const y0 = r() * w * 1.1 - w * .05, amp = (2 + r() * 7) * k, f = (1 + r() * 2.2) * Math.PI * 2 / w, ph = r() * 6.3;
      const dark = r() < .62;
      g.strokeStyle = dark ? `rgba(150, 92, 30, ${.035 + r() * .075})` : `rgba(255, 238, 196, ${.05 + r() * .08})`;
      g.lineWidth = (.5 + r() * 2.4) * k;
      g.beginPath();
      for (let x = -4; x <= w + 4; x += w / 48) {
        const y = y0 + Math.sin(x * f + ph) * amp + Math.sin(x * f * 2.9 + ph * 1.7) * amp * .3 + x * .05;
        if (x < 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
    }
    const v = g.createRadialGradient(w * .42, w * .38, w * .05, w * .5, w * .5, w * .78);
    v.addColorStop(0, 'rgba(255, 244, 214, .22)'); v.addColorStop(1, 'rgba(110, 62, 14, .2)');
    g.fillStyle = v; g.fillRect(0, 0, w, w);
    woodCache.set(key, c);
    return c;
  }

  const spriteCache = new Map();
  function stoneSprite(r, color) {
    const key = Math.round(r * 4) + ':' + color;
    if (spriteCache.has(key)) return spriteCache.get(key);
    const pad = Math.ceil(r * .6), s = Math.ceil(2 * r + 2 * pad);
    const c = document.createElement('canvas');
    c.width = c.height = s;
    const g = c.getContext('2d'), cx = s / 2, cy = s / 2;
    g.save();
    g.shadowColor = 'rgba(48, 24, 0, .5)'; g.shadowBlur = r * .38; g.shadowOffsetX = r * .07; g.shadowOffsetY = r * .17;
    g.beginPath(); g.arc(cx, cy, r * .97, 0, Math.PI * 2);
    g.fillStyle = color === BLACK ? '#111' : '#d8d1c1'; g.fill();
    g.restore();
    const gr = g.createRadialGradient(cx - r * .36, cy - r * .4, r * .04, cx, cy, r * 1.02);
    if (color === BLACK) {
      gr.addColorStop(0, '#6d6c68'); gr.addColorStop(.22, '#33322e'); gr.addColorStop(.7, '#121110'); gr.addColorStop(1, '#040404');
    } else {
      gr.addColorStop(0, '#ffffff'); gr.addColorStop(.45, '#f7f4ed'); gr.addColorStop(.86, '#ddd6c6'); gr.addColorStop(1, '#c2b8a3');
    }
    g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fillStyle = gr; g.fill();
    if (color === WHITE) {
      // clamshell grain
      g.save(); g.clip();
      g.strokeStyle = 'rgba(150, 138, 112, .09)'; g.lineWidth = Math.max(.6, r * .045);
      for (let i = 0; i < 7; i++) { g.beginPath(); g.arc(cx + r * 1.15, cy - r * .55, r * (.55 + i * .26), 0, Math.PI * 2); g.stroke(); }
      g.restore();
    } else {
      g.beginPath(); g.ellipse(cx - r * .32, cy - r * .42, r * .3, r * .16, -.6, 0, Math.PI * 2);
      g.fillStyle = 'rgba(255, 255, 255, .1)'; g.fill();
    }
    const out = { c, s };
    spriteCache.set(key, out);
    return out;
  }

  const easeOut = t => 1 - Math.pow(1 - t, 3);
  const easeBack = t => { const c1 = 1.5, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };

  class Goban {
    constructor(canvas, opts) {
      opts = opts || {};
      this.cv = canvas;
      this.ctx = canvas.getContext('2d');
      this.coords = opts.coords !== false;
      this.lite = !!opts.lite;
      this.seed = opts.seed || 7;
      this.cells = new Int8Array(N);
      this.anim = new Map();   // point -> start time of its placement
      this.fading = [];        // {p, color, t0}
      this.last = -2;
      this.marks = [];         // candidate rings {p, label, w (0..1), t0}
      this.heat = null; this.heatFrom = null; this.heatT0 = 0;
      this.ghost = null;       // {p, color}
      this.raf = 0;
      this.w = 0;
      new ResizeObserver(() => this.resize()).observe(canvas);
      this.resize();
    }
    resize() {
      const rect = this.cv.getBoundingClientRect();
      if (!rect.width) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(rect.width * dpr);
      if (w === this.w) return;
      this.w = w;
      this.cv.width = this.cv.height = w;
      const mf = this.coords ? 1.02 : .62;
      this.cell = w / (SIZE - 1 + 2 * mf);
      this.m = this.cell * mf;
      this.r = this.cell * .485;
      this.draw(performance.now());
    }
    xy(p) { return [this.m + (p % SIZE) * this.cell, this.m + ((p / SIZE) | 0) * this.cell]; }
    pointAt(ev) {
      const rect = this.cv.getBoundingClientRect(), k = this.w / rect.width;
      const x = (ev.clientX - rect.left) * k, y = (ev.clientY - rect.top) * k;
      const cx = Math.round((x - this.m) / this.cell), cy = Math.round((y - this.m) / this.cell);
      if (cx < 0 || cy < 0 || cx >= SIZE || cy >= SIZE) return -1;
      return cy * SIZE + cx;
    }
    // set shows a position. animate drops new stones in and fades captured ones
    // out; opts.stagger spreads the drops over that many ms, opts.fade=false
    // removes stones at once (used when switching between unrelated positions).
    set(cells, last, animate, opts) {
      const now = performance.now(), stagger = (opts && opts.stagger) || 0, fade = !opts || opts.fade !== false;
      if (!fade) this.fading = [];
      for (let p = 0; p < N; p++) {
        const a = this.cells[p], b = cells[p];
        if (a === b) continue;
        if (a !== EMPTY && animate && fade) this.fading.push({ p, color: a, t0: now });
        if (b !== EMPTY && animate) this.anim.set(p, now + Math.random() * stagger); else this.anim.delete(p);
      }
      this.cells = Int8Array.from(cells);
      this.last = last === undefined ? -2 : last;
      this.kick();
    }
    setMarks(marks) {
      const now = performance.now(), old = new Map(this.marks.map(m => [m.p, m]));
      this.marks = marks.map((m, i) => Object.assign({}, m, { t0: old.has(m.p) ? old.get(m.p).t0 : now + i * 70 }));
      this.kick();
    }
    setHeat(values) {
      this.heatFrom = this.heatNow(performance.now()) || new Array(N).fill(0);
      this.heat = values; this.heatT0 = performance.now();
      this.kick();
    }
    heatNow(now) {
      if (!this.heat) return null;
      const t = reduceMotion ? 1 : clamp((now - this.heatT0) / 450, 0, 1), e = easeOut(t), from = this.heatFrom;
      if (!from || t >= 1) return this.heat;
      return this.heat.map((v, i) => from[i] + (v - from[i]) * e);
    }
    setGhost(g) {
      const same = (g && this.ghost && g.p === this.ghost.p && g.color === this.ghost.color) || (!g && !this.ghost);
      if (same) return;
      this.ghost = g; this.kick();
    }
    kick() {
      if (this.raf) return;
      this.raf = requestAnimationFrame(t => { this.raf = 0; if (this.draw(t)) this.kick(); });
    }
    draw(now) {
      if (!this.w) return false;
      const g = this.ctx, w = this.w, cell = this.cell, m = this.m;
      let busy = false;
      g.clearRect(0, 0, w, w);
      g.drawImage(wood(w, this.seed), 0, 0);

      // grid
      const lw = Math.max(1, w / (this.lite ? 220 : 560));
      g.strokeStyle = 'rgba(52, 32, 8, .78)'; g.lineWidth = lw;
      g.beginPath();
      for (let i = 0; i < SIZE; i++) {
        const o = Math.round(m + i * cell) + (lw % 2 ? .5 : 0);
        g.moveTo(m, o); g.lineTo(m + (SIZE - 1) * cell, o);
        g.moveTo(o, m); g.lineTo(o, m + (SIZE - 1) * cell);
      }
      g.stroke();
      g.lineWidth = lw * 1.8;
      g.strokeRect(m, m, (SIZE - 1) * cell, (SIZE - 1) * cell);
      g.fillStyle = 'rgba(52, 32, 8, .85)';
      for (const p of [20, 24, 40, 56, 60]) { const [x, y] = this.xy(p); g.beginPath(); g.arc(x, y, Math.max(1.5, cell * .075), 0, Math.PI * 2); g.fill(); }
      if (this.coords) {
        g.fillStyle = 'rgba(70, 44, 12, .62)';
        g.font = `500 ${Math.round(cell * .22)}px "JetBrains Mono", ui-monospace, monospace`;
        g.textAlign = 'center'; g.textBaseline = 'middle';
        for (let i = 0; i < SIZE; i++) {
          g.fillText(COLS[i], m + i * cell, w - m * .3);
          g.fillText(String(SIZE - i), m * .3, m + i * cell);
        }
      }

      // heat (raw policy bubbles)
      const heat = this.heatNow(now);
      if (heat) {
        if (heat !== this.heat) busy = true;
        for (let p = 0; p < N; p++) {
          const v = heat[p];
          if (v <= .005 || this.cells[p]) continue;
          const [x, y] = this.xy(p);
          g.beginPath(); g.arc(x, y, this.r * .96 * v, 0, Math.PI * 2);
          g.fillStyle = `rgba(214, 60, 27, ${.55 + .4 * v})`; g.fill();
        }
      }

      // fading (captured) stones
      this.fading = this.fading.filter(f => {
        const t = (now - f.t0) / 320;
        if (t >= 1 || reduceMotion) return false;
        const [x, y] = this.xy(f.p);
        g.globalAlpha = 1 - easeOut(t);
        this.stone(x, y, f.color, 1 - .15 * t);
        g.globalAlpha = 1;
        busy = true;
        return true;
      });

      // stones
      for (let p = 0; p < N; p++) {
        const c = this.cells[p];
        if (!c) continue;
        const [x, y] = this.xy(p);
        const t0 = this.anim.get(p);
        if (t0 !== undefined && !reduceMotion) {
          const t = clamp((now - t0) / 280, 0, 1);
          if (t >= 1) this.anim.delete(p); else busy = true;
          g.globalAlpha = clamp(t * 3, 0, 1);
          this.stone(x, y, c, 1 + .22 * (1 - easeBack(t)));
          g.globalAlpha = 1;
        } else this.stone(x, y, c, 1);
      }

      // ghost
      if (this.ghost && this.ghost.p >= 0 && !this.cells[this.ghost.p]) {
        const [x, y] = this.xy(this.ghost.p);
        g.globalAlpha = .45; this.stone(x, y, this.ghost.color, 1); g.globalAlpha = 1;
      }

      // candidate rings
      for (const mk of this.marks) {
        if (mk.p < 0) continue;
        const t = reduceMotion ? 1 : clamp((now - mk.t0) / 260, 0, 1);
        if (t < 1) busy = true;
        if (t <= 0) continue;
        const [x, y] = this.xy(mk.p), s = easeBack(t), rr = this.r * .78 * s;
        g.beginPath(); g.arc(x, y, rr, 0, Math.PI * 2);
        g.fillStyle = mk.best ? 'rgba(214, 60, 27, .92)' : `rgba(255, 250, 240, ${.55 + .3 * (mk.w || 0)})`;
        g.fill();
        g.lineWidth = Math.max(1.5, cell * .045); g.strokeStyle = RED; g.stroke();
        if (mk.label && cell > 28) {
          g.fillStyle = mk.best ? '#fff' : '#7a2410';
          g.font = `700 ${Math.round(cell * .25)}px "JetBrains Mono", ui-monospace, monospace`;
          g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(mk.label, x, y + 1);
        }
      }

      // last move
      if (this.last >= 0 && this.cells[this.last]) {
        const [x, y] = this.xy(this.last);
        g.beginPath(); g.arc(x, y, this.r * .3, 0, Math.PI * 2);
        g.fillStyle = RED; g.fill();
        if (this.cells[this.last] === BLACK) { g.lineWidth = Math.max(1, cell * .02); g.strokeStyle = 'rgba(0,0,0,.35)'; g.stroke(); }
      }
      return busy;
    }
    stone(x, y, color, scale) {
      const sp = stoneSprite(this.r, color), s = sp.s * scale;
      this.ctx.drawImage(sp.c, x - s / 2, y - s / 2, s, s);
    }
  }

  /* ============================================================
     win bar
     ============================================================ */
  function winbar(fill, bLabel, wLabel) {
    return function (pb) {
      pb = clamp(pb, 0, 1);
      fill.style.width = (100 * pb).toFixed(1) + '%';
      bLabel.textContent = pct(pb);
      wLabel.textContent = pct(1 - pb);
    };
  }

  /* ============================================================
     hero: Gofer's own analysis of a game it won against GNU Go
     ============================================================ */
  async function hero() {
    const data = await fetchJSON('data/hero.json');
    const board = new Goban($('hero-board'), { coords: true, seed: 3 });
    const cap = $('hero-cap'), pb = $('hero-pb'), pw = $('hero-pw');
    const setWin = winbar($('hero-wb'), $('hero-wb-b'), $('hero-wb-w'));
    const moves = data.moves.map(f => f.p);
    const pos = replay(moves);
    const result = data.result || '';
    let k = 0, playing = false, visible = false, timer = 0, token = 0;

    const who = c => c === 'B' ? 'GNU Go' : 'Gofer';
    function cands(f) {
      const total = f.top.reduce((a, c) => a + c.n, 0) || 1;
      return f.top.filter(c => c.p >= 0).slice(0, 4).map((c, i) => ({ p: c.p, label: String(Math.round(100 * c.n / total)), w: c.n / total, best: i === 0 }));
    }
    function think(i) {
      const f = data.moves[i];
      board.set(pos[i].cells, pos[i].last, false);
      board.setMarks(cands(f));
      setWin(f.bw);
      pb.classList.toggle('thinking', f.c === 'B'); pw.classList.toggle('thinking', f.c === 'W');
      const total = f.top.reduce((a, c) => a + c.n, 0) || 1;
      const list = f.top.slice(0, 3).map(c => `<span class="mv">${vertex(c.p)}</span> ${Math.round(100 * c.n / total)}%`).join(' · ');
      cap.innerHTML = f.c === 'W'
        ? `<span class="muted">Move ${i + 1} · Gofer (White) is weighing</span> ${list}`
        : `<span class="muted">Move ${i + 1} · GNU Go to play. Gofer expects</span> ${list}`;
    }
    function place(i) {
      const f = data.moves[i];
      board.setMarks([]);
      board.set(pos[i + 1].cells, f.p, true);
      pb.classList.remove('thinking'); pw.classList.remove('thinking');
      const rank = f.top.findIndex(c => c.p === f.p);
      let note;
      if (f.c === 'W') note = f.p < 0 ? 'Gofer passes' : `Gofer plays <span class="mv">${vertex(f.p)}</span>`;
      else if (f.p < 0) note = 'GNU Go passes';
      else note = `GNU Go plays <span class="mv">${vertex(f.p)}</span>` + (rank === 0 ? ', exactly as Gofer expected' : rank > 0 ? `, Gofer's #${rank + 1} guess` : ", outside Gofer's top five");
      const after = data.moves[i + 1];
      if (after) setWin(after.bw);
      cap.innerHTML = `<span class="muted">Move ${i + 1} ·</span> ${note}`;
    }
    function finale() {
      pb.classList.remove('thinking'); pw.classList.remove('thinking');
      board.setMarks([]);
      board.set(pos[moves.length].cells, -2, false);
      setWin(result.startsWith('W') ? 0 : 1);
      cap.innerHTML = `<span class="mv">Gofer wins by ${result.replace(/^[BW]\+/, '')}</span> <span class="muted">(${result}). It won 66 of 100 games like this one against GNU Go's default level.</span>`;
    }
    async function run() {
      const my = ++token;
      while (playing && my === token) {
        if (k >= moves.length) {
          finale();
          await sleep(5200);
          if (!playing || my !== token) return;
          k = 0;
          board.set(new Int8Array(N), -2, false);
          await sleep(500);
          continue;
        }
        think(k);
        await sleep(data.moves[k].c === 'W' ? 1150 : 800);
        if (!playing || my !== token) return;
        place(k);
        k++;
        await sleep(560);
      }
    }
    function setPlaying(on) {
      playing = on;
      $('hero-toggle').setAttribute('aria-label', on ? 'Pause' : 'Play');
      $('hero-toggle').innerHTML = on
        ? '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" rx="1"/><rect x="9.5" y="2" width="3.5" height="12" rx="1"/></svg>'
        : '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5z"/></svg>';
      if (on) run(); else token++;
    }
    let userPaused = reduceMotion;
    $('hero-toggle').addEventListener('click', () => {
      if (playing) { userPaused = true; setPlaying(false); } else { userPaused = false; setPlaying(true); }
    });
    $('hero-restart').addEventListener('click', () => {
      k = 0; board.set(new Int8Array(N), -2, false); board.setMarks([]); setWin(.5);
      userPaused = false; token++; setPlaying(true);
    });
    onVisible($('hero-goban'), v => {
      visible = v && !document.hidden;
      if (visible && !userPaused && !playing) setPlaying(true);
      if (!visible && playing) setPlaying(false);
    }, { threshold: .3 });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && playing) setPlaying(false);
      else if (!document.hidden && visible && !userPaused && !playing) setPlaying(true);
    });
    if (reduceMotion) { finale(); setPlaying(false); }
    else { think(0); }
  }

  /* ============================================================
     count-up numbers
     ============================================================ */
  function countUps() {
    document.querySelectorAll('[data-count]').forEach(el => {
      const target = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0);
      const fmt = x => x.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
      if (reduceMotion) { el.textContent = fmt(target); return; }
      let done = false;
      const io = onVisible(el, v => {
        if (!v || done) return;
        done = true; io.disconnect();
        const t0 = performance.now(), dur = 1300;
        el.textContent = fmt(0);
        (function tick(t) {
          const e = easeOut(clamp((t - t0) / dur, 0, 1));
          el.textContent = fmt(dec ? Math.round(target * e * 10 ** dec) / 10 ** dec : Math.round(target * e));
          if (e < 1) requestAnimationFrame(tick);
        })(t0);
      }, { threshold: .6 });
    });
  }

  /* ============================================================
     01 the loop
     ============================================================ */
  function loopSection(gens) {
    // a self-play game of the final network, played on a loop
    const mini = new Goban($('loop-board'), { coords: false, lite: true, seed: 11 });
    const game = gens[gens.length - 1].game;
    const pos = replay(game.moves);
    let i = 0, on = false, tok = 0;
    async function spin() {
      const my = ++tok;
      while (on && my === tok) {
        if (i >= pos.length) { await sleep(1600); i = 0; continue; }
        mini.set(pos[i].cells, pos[i].last, i > 0);
        i++;
        await sleep(i === 1 ? 500 : 150);
      }
    }
    if (reduceMotion) mini.set(pos[pos.length - 1].cells, -2, false);
    else onVisible($('loop-board'), v => { on = v; if (v) spin(); else tok++; });

    // policy loss by generation
    const svg = $('loss-spark');
    const pts = gens.filter(g => g.policyLoss).map(g => [g.gen, g.policyLoss]);
    const W = 300, H = 150, px = 18, py = 22;
    const lo = 2.2, hi = 4.6;
    const X = gx => px + (gx / 16) * (W - 2 * px), Y = v => py + (hi - v) / (hi - lo) * (H - 2 * py);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ');
    const first = pts[0], lastp = pts[pts.length - 1];
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML =
      `<path class="area" d="${d} L ${X(lastp[0])} ${H - 8} L ${X(first[0])} ${H - 8} Z"/>` +
      `<path class="ln" d="${d}" pathLength="1" style="stroke-dasharray:1;stroke-dashoffset:${reduceMotion ? 0 : 1};transition:stroke-dashoffset 1.6s cubic-bezier(.2,.7,.2,1)"/>` +
      `<circle class="pt" cx="${X(first[0])}" cy="${Y(first[1])}" r="3.5"/><circle class="pt" cx="${X(lastp[0])}" cy="${Y(lastp[1])}" r="3.5"/>` +
      `<text x="${X(first[0]) + 8}" y="${Y(first[1]) + 4}" class="hi">${first[1].toFixed(2)}</text>` +
      `<text x="${X(first[0]) + 8}" y="${Y(first[1]) + 17}">gen 0: random</text>` +
      `<text x="${X(lastp[0])}" y="${Y(lastp[1]) - 10}" text-anchor="end" class="hi">${lastp[1].toFixed(2)}</text>` +
      `<text x="${X(lastp[0])}" y="${H - 12}" text-anchor="end">gen 16</text>`;
    onVisible(svg, v => { if (v) svg.querySelector('.ln').style.strokeDashoffset = '0'; });

    // generation 16's report card
    const g16 = gens[16];
    const rows = (g16.results || []).filter(r => r.opp === 'gen-0015' || r.opp === 'gnugo-l6');
    const tally = $('tally');
    tally.innerHTML = rows.map(r => {
      const dots = [];
      for (let i = 0; i < r.w + r.l; i++) dots.push(`<i class="${i < r.w ? '' : 'l'}" style="transition-delay:${(i * 18)}ms"></i>`);
      return `<div class="tally-row"><div class="tally-head"><span>vs ${oppName(r.opp)}</span><b>${r.w}–${r.l}</b></div><div class="tally-dots">${dots.join('')}</div></div>`;
    }).join('');
  }

  function oppName(o) {
    if (o.startsWith('gen-')) return 'generation ' + (+o.slice(4));
    if (o === 'rollout') return 'Monte-Carlo';
    const m = /^gnugo-l(\d+)$/.exec(o);
    return m ? 'GNU Go L' + m[1] : o;
  }

  /* ============================================================
     02 watch it learn
     ============================================================ */
  const NOTES = [
    'Random weights: it knows the rules and nothing else. Every point looks equally good, the edge included, and its games end in a tangle of stones.',
    'One lap in, it already leans away from the edge. It wins all 40 games against generation 0 and loses all 80 against GNU Go.',
    'Stronger again, but GNU Go still wins every game.',
    'A big jump to +1670, rated above a plain Monte-Carlo player, with a sudden taste for the second line.',
    'Its first wins against GNU Go: 4 of 40 at level 1 and 3 of 40 at level 6.',
    'Policy loss keeps falling: it is getting better at predicting its own search.',
    'Beats generation 5 in 31 of 40 games, and wins 6 of 40 against GNU Go level 6.',
    'Elo +1975. The first line has almost vanished from its opening instinct.',
    '8 wins out of 40 against both GNU Go levels.',
    '<em>A step back.</em> It loses to generation 8, 16–24. Self-play is noisy; the loop keeps going anyway.',
    'Back on track: level with GNU Go level 1 at 19–21, and 40–0 against the Monte-Carlo player.',
    'Beats generation 10, 22–18.',
    'Overtakes GNU Go level 1, 26–14, and splits 20–20 with level 6.',
    'Elo +2297, the best rating so far.',
    '<em>Another dip:</em> it loses to generation 13, 17–23, though it still beats GNU Go level 1.',
    'Beats generation 14, 23–17.',
    'Now it beats GNU Go level 6 as well, 23–17.',
    'The released network, <code>models/gofer-9x9.onnx</code>. With 400 visits a move it beats GNU Go’s default level 66–34. Its favourite openings: the centre and the 4-4 points.',
  ];

  function gensSection(data) {
    const gens = data.gens, last = gens.length - 1;
    const board = new Goban($('gen-board'), { coords: true, seed: 5 });
    const games = gens.map(g => replay(g.game ? g.game.moves : []));
    const range = $('gen-range'), strip = $('gen-strip');
    let cur = -1, view = 'game', replayTok = 0, introTok = 0, userTouched = false;

    // thumbnails
    gens.forEach((g, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'thumb'; b.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-label', 'Generation ' + i);
      b.innerHTML = '<canvas></canvas><span>' + i + '</span>';
      strip.appendChild(b);
      const gb = new Goban(b.querySelector('canvas'), { coords: false, lite: true, seed: 20 + i });
      const fin = games[i][games[i].length - 1];
      gb.set(fin.cells, -2, false);
      b.addEventListener('click', () => {
        touch(); select(i, true);
        const top = $('gen-board').getBoundingClientRect().top;
        if (top < 0 || top > window.innerHeight * .6) $('gen-board').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
      });
    });

    // Elo chart
    const host = $('elo-chart');
    let chart = null;
    function drawChart() {
      const W = Math.max(280, Math.round(host.getBoundingClientRect().width)), H = Math.round(clamp(W * .5, 200, 290));
      const pl = 44, pr = 12, pt = 14, pb = 28, top = 2600;
      const X = g => pl + g / last * (W - pl - pr), Y = e => pt + (1 - e / top) * (H - pt - pb);
      const line = gens.map((g, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(g.elo).toFixed(1)).join(' ');
      const band = gens.map((g, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(g.elo + 1.96 * g.se).toFixed(1)).join(' ') + ' ' +
        gens.slice().reverse().map((g, j) => 'L' + X(last - j).toFixed(1) + ' ' + Y(Math.max(0, g.elo - 1.96 * g.se)).toFixed(1)).join(' ') + ' Z';
      let grid = '', axis = '';
      for (let e = 0; e <= 2500; e += 500) {
        grid += `<line x1="${pl}" x2="${W - pr}" y1="${Y(e)}" y2="${Y(e)}"/>`;
        axis += `<text x="${pl - 8}" y="${Y(e) + 4}" text-anchor="end">${e}</text>`;
      }
      for (let g = 0; g <= last; g += (W < 480 ? 4 : 2)) axis += `<text x="${X(g)}" y="${H - 8}" text-anchor="middle">${g}</text>`;
      const A = data.anchors || {};
      const anchor = (key, label, dy, right) => A[key] === undefined ? '' :
        `<g class="anchor"><line x1="${pl}" x2="${W - pr}" y1="${Y(A[key])}" y2="${Y(A[key])}"/><text x="${right ? W - pr - 4 : pl + 6}" y="${Y(A[key]) + dy}" text-anchor="${right ? 'end' : 'start'}">${label} · ${Math.round(A[key])}</text></g>`;
      const pts = gens.map((g, i) => `<circle class="elo-pt" data-g="${i}" cx="${X(i)}" cy="${Y(g.elo)}" r="${W < 480 ? 3 : 4}"/>`).join('');
      host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Elo rating by generation">
        <defs><linearGradient id="eloFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${RED}" stop-opacity=".22"/><stop offset="1" stop-color="${RED}" stop-opacity="0"/></linearGradient></defs>
        <g class="grid">${grid}</g><g class="axis">${axis}</g>
        ${anchor('gnugo-l6', W < 520 ? 'GNU Go L6' : 'GNU Go level 6', -6, false)}${anchor('gnugo-l1', W < 520 ? 'GNU Go L1' : 'GNU Go level 1', 14, false)}${anchor('rollout', W < 520 ? 'rollouts' : 'Monte-Carlo rollouts', -6, true)}
        <path class="ci" d="${band}"/>
        <path class="elo-area" d="${line} L ${X(last)} ${Y(0)} L ${X(0)} ${Y(0)} Z"/>
        <path class="elo-line" d="${line}"/>
        <line class="cursor" id="chart-cursor" y1="${pt}" y2="${H - pb}"/>
        ${pts}
        <circle class="cursor-dot" id="chart-dot" r="7"/>
        <rect id="chart-hit" x="${pl}" y="0" width="${W - pl - pr}" height="${H}" fill="transparent"/>
      </svg>`;
      chart = { X, Y };
      const hit = host.querySelector('#chart-hit');
      const pick = ev => {
        const r = hit.getBoundingClientRect();
        const g = Math.round(clamp((ev.clientX - r.left) / r.width, 0, 1) * last);
        if (g !== cur) select(g, true);
      };
      hit.addEventListener('pointerdown', ev => { touch(); pick(ev); hit.setPointerCapture(ev.pointerId); });
      hit.addEventListener('pointermove', ev => { if (ev.buttons) pick(ev); });
      placeCursor();
    }
    function placeCursor() {
      if (!chart || cur < 0) return;
      const c = host.querySelector('#chart-cursor'), d = host.querySelector('#chart-dot');
      const x = chart.X(cur), y = chart.Y(gens[cur].elo);
      c.setAttribute('x1', x); c.setAttribute('x2', x);
      d.setAttribute('cx', x); d.setAttribute('cy', y);
      host.querySelectorAll('.elo-pt').forEach(p => p.classList.toggle('on', +p.dataset.g <= cur));
    }
    new ResizeObserver(() => { const w = Math.round(host.getBoundingClientRect().width); if (w && w !== host._w) { host._w = w; drawChart(); } }).observe(host);

    function heatOf(g) {
      // bubble size ~ sqrt(probability), on a fixed scale so generations compare honestly
      return g.opening.slice(0, N).map(p => Math.sqrt(clamp(p / .03, 0, 1)));
    }
    function caption() {
      const g = gens[cur];
      if (view === 'instinct') {
        const op = g.opening.slice(0, N), best = op.indexOf(Math.max(...op));
        return `Raw network policy on an empty board, before any search. Top: ${vertex(best)} ${(100 * op[best]).toFixed(1)}% (uniform = 1.2%).`;
      }
      if (!g.game) return 'No self-play game recorded.';
      return `Generation ${cur} vs itself · ${g.game.moves.length} moves · ${g.game.result}`;
    }
    function showBoard(animate) {
      replayTok++;
      const g = gens[cur];
      $('gen-replay').hidden = view !== 'game';
      if (view === 'instinct') {
        board.set(new Int8Array(N), -2, animate, { fade: false });
        board.setHeat(heatOf(g));
      } else {
        board.heat = null; board.heatFrom = null;
        const fin = games[cur][games[cur].length - 1];
        board.set(fin.cells, -2, animate, { stagger: 240, fade: false });
      }
      $('gen-cap').textContent = caption();
    }
    function select(i, animate) {
      i = clamp(i, 0, last);
      if (i === cur) return;
      cur = i;
      const g = gens[i];
      $('gen-num').textContent = String(i);
      $('gen-elo').textContent = (g.elo > 0 ? '+' : '') + Math.round(g.elo);
      $('gen-se').textContent = i === 0 ? 'the baseline: random play' : '± ' + Math.round(1.96 * g.se) + ' (95%) · gen 0 = 0';
      $('gen-note').innerHTML = NOTES[i] || '';
      const chips = (g.results || []).map(r => `<li class="chip ${r.w > r.l ? 'win' : r.w < r.l ? 'loss' : ''}">vs ${oppName(r.opp)} <b>${r.w}–${r.l}</b></li>`);
      if (i === last) chips.unshift('<li class="chip big">vs GNU Go level 10 <b>66–34</b> · 100 games</li>');
      if (!chips.length) chips.push('<li class="chip">no matches: every later generation is measured against it</li>');
      $('gen-results').innerHTML = chips.join('');
      range.value = String(i);
      range.style.setProperty('--p', (100 * i / last) + '%');
      strip.querySelectorAll('.thumb').forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
      placeCursor();
      showBoard(animate);
    }
    function touch() { userTouched = true; introTok++; }

    range.addEventListener('input', () => { touch(); select(+range.value, true); });
    document.querySelectorAll('#generations .seg button').forEach(b => b.addEventListener('click', () => {
      document.querySelectorAll('#generations .seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      view = b.dataset.view; touch(); showBoard(true);
    }));
    $('gen-replay').addEventListener('click', async () => {
      touch();
      const my = ++replayTok, pos = games[cur];
      board.set(new Int8Array(N), -2, false);
      for (let j = 1; j < pos.length; j++) {
        if (my !== replayTok) return;
        board.set(pos[j].cells, pos[j].last, true);
        await sleep(reduceMotion ? 0 : 90);
      }
      board.set(pos[pos.length - 1].cells, -2, false);
    });

    // the intro: on first sight, sweep from generation 0 to the final network
    if (reduceMotion) select(last, false);
    else {
      select(0, false);
      let started = false;
      const io = onVisible($('gen-board'), v => {
        if (!v || started || userTouched) return;
        started = true; io.disconnect();
        const my = ++introTok;
        (async () => {
          await sleep(500);
          for (let i = 1; i <= last; i++) {
            if (my !== introTok) return;
            select(i, true);
            await sleep(i < 4 ? 1000 : 700);
          }
        })();
      }, { threshold: .45 });
    }
  }

  /* ============================================================
     03 play the real engine
     ============================================================ */
  function playSection() {
    const board = new Goban($('play-board'), { coords: true, seed: 9 });
    const overlay = $('play-overlay'), overlayIn = $('play-overlay-in');
    const setWin = winbar($('pp-wb'), $('pp-wb-b'), $('pp-wb-w'));
    const turn = $('pp-turn'), prog = $('pp-progress'), pbar = $('pp-pbar'), cands = $('pp-cands'), result = $('pp-result');
    let engine = null, phase = 'idle', human = BLACK, visits = 400, state = null, token = 0;

    function radio(group, attr, onPick) {
      const btns = document.querySelectorAll(group);
      btns.forEach(b => b.addEventListener('click', () => {
        btns.forEach(x => x.setAttribute('aria-checked', String(x === b)));
        onPick(+b.dataset[attr]);
      }));
    }
    const NAMES = { 100: 'Casual', 400: 'Club', 1600: 'Full' };
    const hint = () => { const h = $('ov-hint'); if (h) h.textContent = `You play ${human === BLACK ? 'Black' : 'White'} · ${NAMES[visits]} strength`; };
    radio('.side', 'color', c => { human = c; hint(); });
    radio('.seg-dark button', 'visits', v => { visits = v; hint(); });
    const idle = '<button type="button" class="btn btn-red btn-lg" id="ov-start">Start a game <span class="arrow" aria-hidden="true">→</span></button><p class="msg" id="ov-hint"></p>';
    overlay.addEventListener('click', ev => { if (ev.target.closest('#ov-start')) start(); });

    function setTurn(text, color, thinking) {
      turn.innerHTML = `<i class="stone ${color === BLACK ? 'b' : 'w'}"></i><span>${text}</span>`;
      turn.classList.toggle('thinking', !!thinking);
    }
    function show(s, animate) {
      state = s;
      board.set(Int8Array.from(s.cells), s.last, animate);
      $('pp-move').textContent = s.moveNum;
      $('pp-caps').textContent = s.capturesB + ' · ' + s.capturesW;
    }
    function showCands(top, done) {
      if (!top || !top.length) { cands.innerHTML = '<li class="cands-empty">Gofer\'s candidate moves appear here while it thinks.</li>'; return; }
      const total = top.reduce((a, c) => a + c.visits, 0) || 1;
      cands.innerHTML = top.slice(0, 5).map(c =>
        `<li class="cand"><span class="mv">${vertex(c.move)}</span><span class="bar"><i style="width:${(100 * c.visits / total).toFixed(1)}%"></i></span><span class="wr">${pct((c.q + 1) / 2)}</span></li>`).join('');
      board.setMarks(done ? [] : top.filter(c => c.move >= 0).slice(0, 4).map((c, i) => ({ p: c.move, label: String(Math.round(100 * c.visits / total)), w: c.visits / total, best: i === 0 })));
    }
    function goferColor() { return human === BLACK ? WHITE : BLACK; }
    function yourTurn(msg) {
      phase = 'human';
      setTurn(msg || 'Your move', human, false);
      prog.textContent = ''; pbar.style.width = '0';
    }
    function finish() {
      phase = 'over';
      const m = state.margin, winner = m > 0 ? BLACK : WHITE, res = (m > 0 ? 'B+' : 'W+') + Math.abs(m);
      setWin(winner === BLACK ? 1 : 0);
      setTurn(winner === human ? 'You win!' : 'Gofer wins', winner, false);
      result.hidden = false;
      result.innerHTML = (winner === human ? 'You beat Gofer by ' : 'Gofer wins by ') + Math.abs(m) + ' points' +
        `<small>${res} under area scoring: stones on the board plus surrounded empty points, komi 7.5.</small>`;
      board.setGhost(null);
      prog.textContent = ''; pbar.style.width = '0';
    }
    async function engineMove() {
      const my = token, gc = goferColor();
      phase = 'thinking';
      setTurn('Gofer is thinking', gc, true);
      board.setGhost(null);
      try {
        const res = await engine.genmove(visits, p => {
          if (my !== token) return;
          showCands(p.top, false);
          prog.textContent = p.done + ' / ' + p.total;
          pbar.style.width = (100 * p.done / p.total) + '%';
          const pg = (p.value + 1) / 2;
          setWin(gc === BLACK ? pg : 1 - pg);
        });
        if (my !== token) return;
        board.setMarks([]);
        show(res.state, true);
        const pg = (res.value + 1) / 2;
        setWin(gc === BLACK ? pg : 1 - pg);
        if (res.state.gameOver) return finish();
        yourTurn(res.move === PASS ? 'Gofer passes. Your move' : 'Your move');
      } catch (err) {
        if (my !== token) return;
        yourTurn('Engine error: ' + err.message);
      }
    }
    async function start() {
      token++;
      overlay.hidden = false;
      if (!engine) {
        phase = 'loading';
        $('play-start').disabled = true;
        overlayIn.innerHTML = '<div class="spinner"></div><p class="msg" id="load-msg">Starting…</p>';
        try {
          engine = await window.loadGofer(msg => { const el = $('load-msg'); if (el && msg) el.textContent = msg; });
        } catch (err) {
          overlayIn.textContent = '';
          overlayIn.insertAdjacentHTML('beforeend', '<p>Could not start the engine.</p><p class="msg"></p>');
          overlayIn.querySelector('.msg').textContent = String(err.message || err);
          phase = 'idle'; $('play-start').disabled = false;
          return;
        }
        $('play-start').disabled = false;
      }
      overlay.hidden = true;
      $('pp-setup').hidden = true; $('pp-game').hidden = false; result.hidden = true;
      showCands(null);
      show(engine.newGame(KOMI), false);
      setWin(.5);
      if (human === WHITE) engineMove(); else yourTurn();
    }
    function humanMove(p) {
      if (phase !== 'human') return;
      const r = engine.play(p);
      if (!r.ok) {
        setTurn(/ko|superko|repeat/i.test(r.error) ? 'Illegal: that repeats a position (ko)' : 'Illegal move', human, false);
        return;
      }
      show(r.state, true);
      board.setGhost(null);
      if (r.state.gameOver) return finish();
      engineMove();
    }

    $('play-start').addEventListener('click', start);
    const canvas = $('play-board');
    canvas.addEventListener('pointermove', ev => {
      if (phase !== 'human' || ev.pointerType === 'touch') return;
      const p = board.pointAt(ev);
      const ok = p >= 0 && state && state.legal.includes(p);
      board.setGhost(ok ? { p, color: human } : null);
      canvas.style.cursor = ok ? 'pointer' : 'default';
    });
    canvas.addEventListener('pointerleave', () => board.setGhost(null));
    canvas.addEventListener('click', ev => { const p = board.pointAt(ev); if (p >= 0) humanMove(p); });
    $('pp-pass').addEventListener('click', () => humanMove(PASS));
    $('pp-undo').addEventListener('click', () => {
      if (!engine || !state || phase === 'thinking' || phase === 'loading' || state.moveNum === 0) return;
      token++;
      const n = state.toPlay === human ? 2 : 1;
      result.hidden = true;
      board.setMarks([]);
      show(engine.undo(Math.min(n, state.moveNum)), false);
      if (state.toPlay !== human) engineMove(); else yourTurn();
    });
    $('pp-new').addEventListener('click', () => {
      if (phase === 'loading') return;
      token++;
      phase = 'idle';
      board.setMarks([]); board.setGhost(null);
      $('pp-setup').hidden = false; $('pp-game').hidden = true;
      overlay.hidden = false;
      overlayIn.innerHTML = idle; hint();
    });
  }

  /* ============================================================
     04 terminal + quick start
     ============================================================ */
  async function terminal() {
    const sessions = await fetchJSON('data/sessions.json');
    const body = $('term-body'), title = $('term-title');
    let tok = 0, started = false;
    async function type(key) {
      const my = ++tok, s = sessions[key];
      title.textContent = s.title;
      body.textContent = '';
      for (const l of s.lines) {
        if (my !== tok) return;
        const span = document.createElement('span');
        span.className = 't-' + l.t;
        body.appendChild(span);
        if ((l.t === 'cmd' || l.t === 'in') && !reduceMotion) {
          span.classList.add('caret');
          for (const ch of l.s) { if (my !== tok) return; span.textContent += ch; await sleep(15); }
          span.classList.remove('caret');
          await sleep(220);
        } else {
          span.textContent = l.s;
          if (!reduceMotion) await sleep(28);
        }
        body.appendChild(document.createTextNode('\n'));
        body.scrollTop = body.scrollHeight;
      }
    }
    const tabs = document.querySelectorAll('.term-tabs button');
    tabs.forEach(b => b.addEventListener('click', () => {
      tabs.forEach(x => x.setAttribute('aria-selected', String(x === b)));
      started = true; type(b.dataset.session);
    }));
    const io = onVisible(body, v => { if (v && !started) { started = true; io.disconnect(); type('play'); } }, { threshold: .3 });
  }

  function copyButton() {
    const btn = $('quick-copy');
    btn.addEventListener('click', async () => {
      const text = $('quick-code').innerText.split('\n').map(l => l.replace(/\s+#.*$/, '')).join('\n');
      try { await navigator.clipboard.writeText(text); }
      catch (e) {
        const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch (e2) { /* nothing else to try */ }
        ta.remove();
      }
      btn.classList.add('done');
      btn.setAttribute('aria-label', 'Copied');
      setTimeout(() => { btn.classList.remove('done'); btn.setAttribute('aria-label', 'Copy commands'); }, 1600);
    });
  }

  /* ============================================================
     chrome: nav border, reveals
     ============================================================ */
  function chrome() {
    const nav = $('nav');
    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    const els = document.querySelectorAll('.rv');
    if (reduceMotion || !('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('in')); return; }
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: .12, rootMargin: '0px 0px -40px 0px' });
    els.forEach(e => io.observe(e));
  }

  /* ============================================================
     boot
     ============================================================ */
  chrome();
  countUps();
  copyButton();
  playSection();
  hero().catch(err => { $('hero-cap').textContent = 'Could not load the game: ' + err.message; });
  fetchJSON('data/generations.json').then(data => {
    loopSection(data.gens);
    gensSection(data);
  }).catch(err => { $('gen-cap').textContent = 'Could not load the generations: ' + err.message; });
  terminal().catch(() => { $('term-body').textContent = 'Could not load the captured sessions.'; });
})();
