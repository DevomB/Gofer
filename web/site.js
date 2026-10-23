/* Gofer site — navigation, the hero board (watch a real game or play the real
   engine in WebAssembly, see engine.js), the Elo chart and terminal sessions. */
(function () {
  'use strict';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = id => document.getElementById(id);

  /* ---------- mobile nav ---------- */
  const navToggle = document.getElementById('nav-toggle');
  const mobileNav = document.getElementById('mobile-nav');
  const navIconOpen = document.getElementById('nav-icon-open');
  const navIconClose = document.getElementById('nav-icon-close');
  function setNavOpen(open) {
    mobileNav.classList.toggle('open', open);
    navToggle.setAttribute('aria-expanded', String(open));
    navIconOpen.classList.toggle('hidden', open);
    navIconClose.classList.toggle('hidden', !open);
  }
  navToggle.addEventListener('click', () => setNavOpen(!mobileNav.classList.contains('open')));
  mobileNav.querySelectorAll('.nav-link').forEach(a => a.addEventListener('click', () => setNavOpen(false)));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') setNavOpen(false); });
  document.addEventListener('click', e => {
    if (mobileNav.classList.contains('open') && !mobileNav.contains(e.target) && !navToggle.contains(e.target)) setNavOpen(false);
  });

  /* ---------- scroll reveal ---------- */
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('on'); io.unobserve(e.target); }
  }), { threshold: 0.12 });
  document.querySelectorAll('.rv').forEach(el => io.observe(el));

  /* ---------- copy buttons ---------- */
  function legacyCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.focus(); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* no-op: clipboard unavailable */ }
    document.body.removeChild(ta);
  }
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const flash = () => {
        const icon = btn.querySelector('.copy-icon');
        if (icon) { icon.classList.add('copy-ok'); setTimeout(() => icon.classList.remove('copy-ok'), 1200); }
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(btn.dataset.copy).then(flash, () => { legacyCopy(btn.dataset.copy); flash(); });
      } else {
        legacyCopy(btn.dataset.copy); flash();
      }
    });
  });


  /* ============================================================
     HERO BOARD — watch a real game, or play the real engine
     ============================================================ */
  const EMPTY = 0, BLACK = 1, WHITE = 2, SIZE = 9, N = SIZE * SIZE, PASS = -1;
  const COLS = 'ABCDEFGHJ';
  const KOMI = 7.5;
  function vertexName(i) { return i < 0 ? 'pass' : COLS[i % SIZE] + (SIZE - ((i / SIZE) | 0)); }
  function neighbors(i) {
    const x = i % SIZE, y = (i / SIZE) | 0, out = [];
    if (x > 0) out.push(i - 1);
    if (x < SIZE - 1) out.push(i + 1);
    if (y > 0) out.push(i - SIZE);
    if (y < SIZE - 1) out.push(i + SIZE);
    return out;
  }
  function chainAt(grid, i) {
    const color = grid[i], seen = new Set([i]), stack = [i], stones = [];
    let libs = 0;
    while (stack.length) {
      const p = stack.pop(); stones.push(p);
      for (const n of neighbors(p)) {
        if (grid[n] === EMPTY) libs++;
        else if (grid[n] === color && !seen.has(n)) { seen.add(n); stack.push(n); }
      }
    }
    return { stones, libs };
  }
  // Applies a recorded (legal) move and returns the captured points.
  function applyMove(grid, i, color) {
    if (i === PASS) return [];
    grid[i] = color;
    const opp = color === BLACK ? WHITE : BLACK, captured = [];
    for (const n of neighbors(i)) {
      if (grid[n] === opp) {
        const c = chainAt(grid, n);
        if (c.libs === 0) for (const s of c.stones) { grid[s] = EMPTY; captured.push(s); }
      }
    }
    return captured;
  }

  /* A game from the evaluation: GNU Go 3.8 level 10 (Black) vs Gofer at
     400 visits (White) after a two-move random opening. Result W+12.5. */
  const REPLAY_SGF = ';B[bd];W[gg];B[ff];W[ge];B[gf];W[hf];B[fc];W[fe];B[fg];W[ef];B[hg];W[eg];B[gh];W[fh];B[gg];W[ed];B[cg];W[dc];B[hd];W[he];B[eh];W[dh];B[fi];W[dg];B[cb];W[eb];B[fb];W[cf];B[db];W[ec];B[bg];W[bf];B[gd];W[ea];B[cc];W[fa];B[dd];W[de];B[gb];W[ga];B[di];W[ci];B[ei];W[ch];B[cd];W[hb];B[be];W[fd];B[af];W[bh];B[ag];W[ah];B[ae];W[hc];B[ce];W[df];B[if];W[ie];B[ig];W[da];B[bb];W[ca];B[ba];W[ab];B[ac];W[gi];B[hi];W[gc];B[];W[]';
  const replayMoves = [];
  REPLAY_SGF.replace(/;([BW])\[([a-z]{0,2})\]/g, (_, c, p) => {
    replayMoves.push({ color: c === 'B' ? BLACK : WHITE, i: p ? (p.charCodeAt(1) - 97) * SIZE + (p.charCodeAt(0) - 97) : PASS });
  });
  // Precompute every position so scrubbing is instant.
  const frames = [{ grid: new Int8Array(N), capB: 0, capW: 0, captured: [] }];
  for (const mv of replayMoves) {
    const prev = frames[frames.length - 1], grid = prev.grid.slice();
    const captured = applyMove(grid, mv.i, mv.color);
    frames.push({
      grid, captured,
      capB: prev.capB + (mv.color === BLACK ? captured.length : 0),
      capW: prev.capW + (mv.color === WHITE ? captured.length : 0),
    });
  }

  /* ---------- canvas ---------- */
  const canvas = $('goban'), ctx = canvas.getContext('2d');
  let px = 0, cell = 0, pad = 0;
  const anims = []; // {i, color, type: 'place'|'capture', t0}
  let hoverIdx = -1;

  function layout() {
    const w = canvas.parentElement.clientWidth, dpr = window.devicePixelRatio || 1;
    px = w;
    canvas.style.height = w + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(w * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    pad = px * 0.075;
    cell = (px - pad * 2) / (SIZE - 1);
    draw();
  }
  function stoneXY(i) { return [pad + (i % SIZE) * cell, pad + ((i / SIZE) | 0) * cell]; }

  function drawWood() {
    const g1 = ctx.createLinearGradient(0, 0, px, px);
    g1.addColorStop(0, '#C89E5E'); g1.addColorStop(0.5, '#BC9152'); g1.addColorStop(1, '#A87E42');
    ctx.fillStyle = g1; ctx.fillRect(0, 0, px, px);
    ctx.strokeStyle = 'rgba(90,62,30,0.09)'; ctx.lineWidth = 1;
    for (let k = 0; k < 22; k++) {
      const y0 = (k * 97 % 613) / 613 * px;
      ctx.beginPath();
      for (let x = 0; x <= px; x += 8) {
        const y = y0 + Math.sin(x * 0.011 + k * 2.1) * 3.2 + Math.sin(x * 0.037 + k) * 1.1;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    const vg = ctx.createRadialGradient(px / 2, px / 2, px * 0.35, px / 2, px / 2, px * 0.75);
    vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(40,20,0,0.18)');
    ctx.fillStyle = vg; ctx.fillRect(0, 0, px, px);
  }
  function drawGrid() {
    ctx.strokeStyle = 'rgba(52,36,18,0.85)';
    ctx.lineWidth = Math.max(1, px / 480);
    for (let k = 0; k < SIZE; k++) {
      const p = pad + k * cell;
      ctx.beginPath(); ctx.moveTo(pad, p); ctx.lineTo(px - pad, p); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p, pad); ctx.lineTo(p, px - pad); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(52,36,18,0.9)';
    for (const [hx, hy] of [[2, 2], [6, 2], [4, 4], [2, 6], [6, 6]]) {
      const [x, y] = stoneXY(hy * SIZE + hx);
      ctx.beginPath(); ctx.arc(x, y, Math.max(1.5, cell * 0.08), 0, 7); ctx.fill();
    }
    ctx.fillStyle = 'rgba(52,36,18,0.6)';
    ctx.font = `${Math.max(8, px / 44)}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let k = 0; k < SIZE; k++) {
      ctx.fillText(COLS[k], pad + k * cell, px - pad * 0.4);
      ctx.fillText(String(SIZE - k), pad * 0.4, pad + k * cell);
    }
  }
  function drawStone(x, y, color, r, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.shadowColor = 'rgba(30,15,0,0.45)'; ctx.shadowBlur = r * 0.5; ctx.shadowOffsetY = r * 0.18;
    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.15, x, y, r * 1.05);
    if (color === BLACK) { g.addColorStop(0, '#4a4f55'); g.addColorStop(0.4, '#22252a'); g.addColorStop(1, '#0b0c0e'); }
    else { g.addColorStop(0, '#FFFEF9'); g.addColorStop(0.55, '#F2EDE0'); g.addColorStop(1, '#C9C2B0'); }
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
    ctx.restore();
  }

  // What the board shows right now (set by the active mode).
  let view = { grid: frames[0].grid, last: -2, ghost: EMPTY, candidates: [] };

  function draw() {
    if (!px) return;
    drawWood(); drawGrid();
    const r = cell * 0.47, now = performance.now(), g = view.grid;
    for (const a of anims) {
      if (a.type !== 'capture') continue;
      const t = (now - a.t0) / 320;
      if (t >= 1) continue;
      const [x, y] = stoneXY(a.i);
      drawStone(x, y, a.color, r * (1 - t * 0.4), 1 - t);
    }
    for (let i = 0; i < N; i++) {
      if (g[i] === EMPTY) continue;
      let rr = r, aa = 1;
      const a = anims.find(a2 => a2.i === i && a2.type === 'place');
      if (a) { const t = Math.min(1, (now - a.t0) / 180); rr = r * (0.6 + 0.4 * t); aa = t; }
      const [x, y] = stoneXY(i);
      drawStone(x, y, g[i], rr, aa);
    }
    // candidate moves while the engine thinks: violet rings sized by visit share
    for (const c of view.candidates) {
      if (g[c.i] !== EMPTY) continue;
      const [x, y] = stoneXY(c.i);
      ctx.strokeStyle = `rgba(169,156,255,${0.35 + 0.65 * c.share})`;
      ctx.lineWidth = Math.max(1.5, px / 260);
      ctx.beginPath(); ctx.arc(x, y, r * (0.3 + 0.55 * c.share), 0, 7); ctx.stroke();
    }
    if (view.last >= 0 && g[view.last] !== EMPTY) {
      const [x, y] = stoneXY(view.last);
      ctx.strokeStyle = g[view.last] === BLACK ? 'rgba(232,194,133,0.95)' : 'rgba(90,60,20,0.8)';
      ctx.lineWidth = Math.max(1.4, px / 300);
      ctx.beginPath(); ctx.arc(x, y, r * 0.45, 0, 7); ctx.stroke();
    }
    if (view.ghost !== EMPTY && hoverIdx >= 0 && g[hoverIdx] === EMPTY) {
      const [x, y] = stoneXY(hoverIdx);
      drawStone(x, y, view.ghost, r, 0.45);
    }
  }
  let rafOn = false;
  function animate() {
    if (reduceMotion) { anims.length = 0; draw(); return; }
    if (rafOn) return;
    rafOn = true;
    (function tick() {
      draw();
      const now = performance.now();
      for (let k = anims.length - 1; k >= 0; k--) if (now - anims[k].t0 > 400) anims.splice(k, 1);
      if (anims.length) requestAnimationFrame(tick); else { rafOn = false; draw(); }
    })();
  }
  // Queue place/capture animations for the difference between two positions.
  function animateDiff(before, after) {
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      if (before[i] === EMPTY && after[i] !== EMPTY) anims.push({ i, color: after[i], type: 'place', t0 });
      else if (before[i] !== EMPTY && after[i] === EMPTY) anims.push({ i, color: before[i], type: 'capture', t0 });
    }
    animate();
  }

  const statusEl = $('board-status'), capsEl = $('board-caps'), captionEl = $('board-caption'), labelEl = $('board-label');
  function setCaps(b, w) { capsEl.textContent = 'captures — B ' + b + ' · W ' + w; }

  /* ---------- watch mode ---------- */
  let mode = 'watch', replayN = 0, playing = false, timer = null, speedIdx = 0;
  const speeds = [700, 350, 170];
  const scrub = $('rp-scrub'), countEl = $('rp-count'), playBtn = $('rp-play');
  scrub.max = replayMoves.length;

  function replaySet(n, withAnim) {
    n = Math.max(0, Math.min(replayMoves.length, n));
    const before = frames[replayN].grid;
    replayN = n;
    const f = frames[n], mv = n > 0 ? replayMoves[n - 1] : null;
    view = { grid: f.grid, last: mv ? mv.i : -2, ghost: EMPTY, candidates: [] };
    scrub.value = n;
    countEl.textContent = n + ' / ' + replayMoves.length;
    setCaps(f.capB, f.capW);
    if (!mv) statusEl.textContent = '● GNU Go vs ○ Gofer';
    else statusEl.textContent = 'move ' + n + ' — ' + (mv.color === BLACK ? '● GNU Go ' : '○ Gofer ') + (mv.i < 0 ? 'passes' : vertexName(mv.i)) + (n <= 2 ? ' · random opening' : '');
    if (n === replayMoves.length) statusEl.textContent = 'both pass — Gofer wins by 12.5 points';
    if (withAnim) animateDiff(before, f.grid); else draw();
  }
  function setPlayIcon() {
    playBtn.innerHTML = playing
      ? '<svg width="13" height="13" viewBox="0 0 14 14" fill="#8B7CF6" aria-hidden="true"><path d="M3 2h3v10H3zM8 2h3v10H8z"/></svg>'
      : '<svg width="13" height="13" viewBox="0 0 14 14" fill="#8B7CF6" aria-hidden="true"><path d="M3.5 2 12 7l-8.5 5z"/></svg>';
    playBtn.setAttribute('aria-label', playing ? 'Pause replay' : 'Play replay');
  }
  function stop() { playing = false; clearInterval(timer); setPlayIcon(); }
  function start() {
    if (replayN >= replayMoves.length) replaySet(0, false);
    playing = true; setPlayIcon();
    clearInterval(timer);
    timer = setInterval(() => {
      if (mode !== 'watch') return;
      if (replayN >= replayMoves.length) {
        stop();
        setTimeout(() => { if (mode === 'watch' && !playing && replayN === replayMoves.length) { replaySet(0, false); start(); } }, 4000);
        return;
      }
      replaySet(replayN + 1, true);
    }, speeds[speedIdx]);
  }
  setPlayIcon();
  playBtn.addEventListener('click', () => (playing ? stop() : start()));
  $('rp-start').addEventListener('click', () => { stop(); replaySet(0, false); });
  $('rp-back').addEventListener('click', () => { stop(); replaySet(replayN - 1, false); });
  $('rp-fwd').addEventListener('click', () => { stop(); replaySet(replayN + 1, true); });
  $('rp-speed').addEventListener('click', e => {
    speedIdx = (speedIdx + 1) % speeds.length;
    e.currentTarget.textContent = '×' + [1, 2, 4][speedIdx];
    if (playing) start();
  });
  scrub.addEventListener('input', () => { stop(); replaySet(+scrub.value, false); });

  /* ---------- play mode: the real engine, in WebAssembly ---------- */
  const overlay = $('play-overlay'), loadStatus = $('load-status'), turnEl = $('turn-label'), turnDot = $('turn-dot');
  const playMsg = $('fp-msg'), strengthEl = $('fp-strength');
  const wbFill = $('wb-fill'), wbB = $('wb-b'), wbW = $('wb-w');
  let engine = null, phase = 'idle', human = BLACK, state = null, gameToken = 0;

  function setWinbar(blackProb) {
    const p = Math.max(0, Math.min(1, blackProb));
    wbFill.style.width = (100 * p).toFixed(1) + '%';
    wbB.textContent = Math.round(100 * p) + '%';
    wbW.textContent = Math.round(100 * (1 - p)) + '%';
  }
  function colorName(c) { return c === BLACK ? 'Black' : 'White'; }
  function showState(s, withAnim) {
    const before = state ? state.cells : new Array(N).fill(EMPTY);
    state = s;
    const grid = Int8Array.from(s.cells);
    view = { grid, last: s.last, ghost: phase === 'human' ? human : EMPTY, candidates: [] };
    setCaps(s.capturesB, s.capturesW);
    statusEl.textContent = s.moveNum === 0 ? 'komi ' + KOMI + ' · area scoring' : 'move ' + s.moveNum + (s.last >= -1 ? ' — ' + (s.toPlay === BLACK ? '○ ' : '● ') + vertexName(s.last) : '');
    if (withAnim) animateDiff(before, grid); else draw();
  }
  function setTurn() {
    const me = phase === 'thinking' ? (human === BLACK ? WHITE : BLACK) : human;
    turnDot.className = 'inline-block w-3 h-3 rounded-full border ' + (me === BLACK ? 'bg-black border-white/30' : 'bg-[#F2EDE0] border-black/30');
    turnEl.textContent = phase === 'thinking' ? 'Gofer is thinking…' : phase === 'over' ? 'Game over' : 'Your move (' + colorName(human) + ')';
    view.ghost = phase === 'human' ? human : EMPTY;
    draw();
  }
  function finish() {
    phase = 'over';
    const m = state.margin, winner = m > 0 ? BLACK : WHITE;
    const result = (m > 0 ? 'B+' : 'W+') + Math.abs(m);
    playMsg.textContent = (winner === human ? 'You win ' : 'Gofer wins ') + result + ' — area scoring, every stone on the board counts';
    setWinbar(winner === BLACK ? 1 : 0);
    setTurn();
  }
  async function engineMove() {
    const token = gameToken;
    phase = 'thinking'; setTurn();
    const visits = +strengthEl.value;
    playMsg.textContent = '';
    const gofer = human === BLACK ? WHITE : BLACK;
    try {
      const res = await engine.genmove(visits, p => {
        if (token !== gameToken) return;
        const total = p.top.reduce((a, c) => a + c.visits, 0) || 1;
        view.candidates = p.top.map(c => ({ i: c.move, share: c.visits / total }));
        turnEl.textContent = 'Gofer is thinking… ' + Math.round(100 * p.done / p.total) + '%';
        const pg = (p.value + 1) / 2;
        setWinbar(gofer === BLACK ? pg : 1 - pg);
        draw();
      });
      if (token !== gameToken) return;
      phase = 'human';
      showState(res.state, true);
      const pg = (res.value + 1) / 2;
      setWinbar(gofer === BLACK ? pg : 1 - pg);
      playMsg.textContent = res.move === PASS ? 'Gofer passes' + (res.state.gameOver ? '' : ' — capture any dead stones, then pass to end the game') : '';
      if (res.state.gameOver) finish(); else setTurn();
    } catch (err) {
      phase = 'human'; setTurn();
      playMsg.textContent = 'engine error: ' + err.message;
    }
  }
  async function newGame(color) {
    gameToken++;
    human = color;
    overlay.classList.add('hidden'); overlay.classList.remove('flex');
    if (!engine) {
      overlay.classList.remove('hidden'); overlay.classList.add('flex');
      phase = 'loading';
      try {
        engine = await window.loadGofer(msg => { loadStatus.textContent = msg; });
      } catch (err) {
        loadStatus.textContent = 'Could not start the engine: ' + err.message;
        phase = 'idle';
        return;
      }
      overlay.classList.add('hidden'); overlay.classList.remove('flex');
    }
    state = null;
    phase = 'human';
    showState(engine.newGame(KOMI), false);
    setWinbar(0.5);
    playMsg.textContent = '';
    setTurn();
    if (human === WHITE) engineMove();
  }
  function boardIdxFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const cx = Math.round((x - pad) / cell), cy = Math.round((y - pad) / cell);
    if (cx < 0 || cx >= SIZE || cy < 0 || cy >= SIZE) return -1;
    const [sx, sy] = stoneXY(cy * SIZE + cx);
    return Math.hypot(sx - x, sy - y) > cell * 0.55 ? -1 : cy * SIZE + cx;
  }
  function humanMove(i) {
    if (mode !== 'play' || phase !== 'human') return;
    const r = engine.play(i);
    if (!r.ok) { playMsg.textContent = /ko|superko|repeat/i.test(r.error) ? 'illegal — that would repeat a position (ko)' : 'illegal move'; return; }
    playMsg.textContent = i === PASS ? 'You pass' : '';
    phase = 'thinking';
    showState(r.state, true);
    if (r.state.gameOver) finish(); else engineMove();
  }
  canvas.addEventListener('mousemove', e => {
    if (mode !== 'play') return;
    const i = boardIdxFromEvent(e);
    if (i !== hoverIdx) { hoverIdx = i; draw(); }
    canvas.style.cursor = phase === 'human' && i >= 0 && state && state.legal.includes(i) ? 'pointer' : 'default';
  });
  canvas.addEventListener('mouseleave', () => { hoverIdx = -1; draw(); });
  canvas.addEventListener('click', e => { const i = boardIdxFromEvent(e); if (i >= 0) humanMove(i); });
  $('fp-pass').addEventListener('click', () => humanMove(PASS));
  $('fp-undo').addEventListener('click', () => {
    if (mode !== 'play' || !engine || phase === 'thinking' || phase === 'loading' || !state || state.moveNum === 0) return;
    // take back the engine's reply and your move
    const n = state.toPlay === human ? 2 : 1;
    phase = 'human';
    showState(engine.undo(Math.min(n, state.moveNum)), false);
    if (state.toPlay !== human) engineMove(); else { playMsg.textContent = ''; setTurn(); }
  });
  $('fp-new').addEventListener('click', () => {
    if (phase === 'loading') return;
    gameToken++;
    phase = 'idle';
    overlay.classList.remove('hidden'); overlay.classList.add('flex');
    loadStatus.textContent = '';
  });
  document.querySelectorAll('.start-btn').forEach(b => b.addEventListener('click', () => newGame(+b.dataset.color)));

  /* ---------- mode toggle ---------- */
  const modeWatch = $('mode-watch'), modePlay = $('mode-play');
  const ON = 'mode-btn px-2.5 py-1 rounded-md bg-surface2 text-violet border border-violet/40';
  const OFF = 'mode-btn px-2.5 py-1 rounded-md text-ink-mute border border-transparent hover:text-ink-dim';
  function setMode(m) {
    mode = m;
    modeWatch.className = m === 'watch' ? ON : OFF;
    modePlay.className = m === 'play' ? ON : OFF;
    modeWatch.setAttribute('aria-selected', String(m === 'watch'));
    modePlay.setAttribute('aria-selected', String(m === 'play'));
    $('watch-ctl').classList.toggle('hidden', m !== 'watch');
    $('play-ctl').classList.toggle('hidden', m !== 'play');
    hoverIdx = -1;
    if (m === 'play') {
      stop();
      labelEl.textContent = 'you vs Gofer · in your browser';
      captionEl.innerHTML = 'The same search and network as <span class="text-ink-dim">bin/gofer</span>, compiled to WebAssembly. Area scoring: capture dead stones before you pass.';
      if (state) { showState(state, false); setTurn(); }
      else { overlay.classList.remove('hidden'); overlay.classList.add('flex'); view = { grid: new Int8Array(N), last: -2, ghost: EMPTY, candidates: [] }; draw(); }
    } else {
      overlay.classList.add('hidden'); overlay.classList.remove('flex');
      labelEl.textContent = 'vs GNU Go level 10';
      captionEl.innerHTML = 'A game from the final evaluation — Gofer (White, 400 visits a move) won 66–34 over 100 games against <span class="text-ink-dim">GNU Go 3.8 at level 10</span>.';
      replaySet(replayN, false);
    }
  }
  modeWatch.addEventListener('click', () => setMode('watch'));
  modePlay.addEventListener('click', () => setMode('play'));
  document.querySelectorAll('[data-play-gofer]').forEach(a => a.addEventListener('click', e => {
    e.preventDefault();
    setMode('play');
    $('board-card').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
  }));

  /* ---------- boot ---------- */
  layout();
  new ResizeObserver(layout).observe(canvas.parentElement);
  setMode('watch');
  if (reduceMotion) replaySet(replayMoves.length, false);
  else {
    const heroIO = new IntersectionObserver(es => {
      for (const e of es) {
        if (mode !== 'watch') continue;
        if (e.isIntersecting && !playing) start();
        else if (!e.isIntersecting && playing) stop();
      }
    }, { threshold: 0.3 });
    heroIO.observe(canvas);
  }

  /* ============================================================
     ELO CHART — strength by generation (docs/results)
     ============================================================ */
  (function eloChart() {
    const host = $('elo-chart');
    if (!host) return;
    const gens = [[0, 0, 0], [1, 763, 90], [2, 1176, 74], [3, 1670, 50], [4, 1743, 35], [5, 1768, 39], [6, 1882, 33], [7, 1975, 39], [8, 2015, 29], [9, 1975, 40], [10, 2131, 28], [11, 2184, 39], [12, 2238, 27], [13, 2297, 39], [14, 2219, 27], [15, 2243, 39], [16, 2301, 28], [17, 2318, 54]];
    const anchors = [['rollout MCTS', 1448], ['GNU Go level 1', 2180], ['GNU Go level 6', 2241]];
    const mono = 'font-family="JetBrains Mono, monospace" font-size="10.5"';
    // Drawn at the container's pixel width so text stays the same size on every screen.
    function render() {
      const W = Math.max(280, host.clientWidth), H = W < 500 ? 230 : 290;
      const L = 44, R = 10, T = 14, B = 34, ymax = 2500, gmax = 17;
      const x = g => L + (W - L - R) * g / gmax, y = e => T + (H - T - B) * (1 - e / ymax);
      const step = W < 500 ? 4 : 2;
      let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Elo by generation: from 0 (random play) to 2318 at generation 17, passing rollout MCTS at generation 3 and GNU Go levels 1 and 6 around generation 12">`;
      for (let e = 0; e <= ymax; e += 500) {
        svg += `<line x1="${L}" x2="${W - R}" y1="${y(e)}" y2="${y(e)}" stroke="#1C1F33"/>`;
        svg += `<text x="${L - 8}" y="${y(e) + 3.5}" text-anchor="end" fill="#636880" ${mono}>${e}</text>`;
      }
      for (let g = 0; g <= gmax; g += step) svg += `<text x="${x(g)}" y="${H - B + 17}" text-anchor="middle" fill="#636880" ${mono}>${g}</text>`;
      svg += `<text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle" fill="#636880" ${mono}>generation</text>`;
      anchors.forEach(([name, e], k) => {
        svg += `<line x1="${L}" x2="${W - R}" y1="${y(e)}" y2="${y(e)}" stroke="#D6A85C" stroke-opacity=".5" stroke-dasharray="4 5"/>`;
        // GNU Go labels sit at the left, where the curve is still far below them;
        // the rollout label sits at the right, where the curve is far above it
        const left = k > 0;
        svg += `<text x="${left ? L + 8 : W - R - 6}" y="${y(e) + (k === 1 ? 13 : -6)}" text-anchor="${left ? 'start' : 'end'}" fill="#E8C285" ${mono}>${name}</text>`;
      });
      svg += `<path d="${gens.map(([g, e], k) => (k ? 'L' : 'M') + x(g).toFixed(1) + ' ' + y(e).toFixed(1)).join(' ')}" fill="none" stroke="#8B7CF6" stroke-width="2"/>`;
      for (const [g, e, s] of gens) {
        if (s) svg += `<line x1="${x(g)}" x2="${x(g)}" y1="${y(e - s)}" y2="${y(e + s)}" stroke="#8B7CF6" stroke-opacity=".45"/>`;
        svg += `<circle cx="${x(g)}" cy="${y(e)}" r="3.2" fill="#07070D" stroke="#A99CFF" stroke-width="1.6"><title>generation ${g}: ${e} ± ${s} Elo</title></circle>`;
      }
      host.innerHTML = svg + '</svg>';
    }
    let lastW = 0;
    new ResizeObserver(() => { if (host.clientWidth !== lastW) { lastW = host.clientWidth; render(); } }).observe(host);
  })();


  /* ============================================================
     TERMINAL — real captured sessions
     ============================================================ */
  const B9 = [
    '  A B C D E F G H J',
    ' 9 . . . . . . . . .', ' 8 . . . . . . . . .', ' 7 . . . . . . . . .',
    ' 6 . . . . . . . . .', ' 5 . . . . . . . . .', ' 4 . . . . . . . . .',
    ' 3 . . . . . . . . .', ' 2 . . . . . . . . .', ' 1 . . . . . . . . .',
  ];
  const SESSIONS = {
    "play": {
      "title": "bin/gofer play -model models/gofer-9x9.onnx",
      "lines": [
        {
          "t": "cmd",
          "s": "bin/gofer play -model models/gofer-9x9.onnx"
        },
        {
          "t": "out",
          "s": "Moves: D4, pass, resign, hint, quit"
        },
        {
          "t": "in",
          "s": "E5"
        },
        {
          "t": "gold",
          "s": "Gofer plays D4 (win 54%, pv D4 D6 F4 E4 E3)"
        },
        {
          "t": "dim",
          "s": "   A B C D E F G H J"
        },
        {
          "t": "dim",
          "s": " 9 . . . . . . . . .  9"
        },
        {
          "t": "dim",
          "s": " 8 . . . . . . . . .  8"
        },
        {
          "t": "dim",
          "s": " 7 . . . . . . . . .  7"
        },
        {
          "t": "dim",
          "s": " 6 . . . . . . . . .  6"
        },
        {
          "t": "dim",
          "s": " 5 . . . . X . . . .  5"
        },
        {
          "t": "dim",
          "s": " 4 . . . O). . . . .  4"
        },
        {
          "t": "dim",
          "s": " 3 . . . . . . . . .  3"
        },
        {
          "t": "dim",
          "s": " 2 . . . . . . . . .  2"
        },
        {
          "t": "dim",
          "s": " 1 . . . . . . . . .  1"
        },
        {
          "t": "in",
          "s": "hint"
        },
        {
          "t": "out",
          "s": "hint: D6 (win 45%)"
        },
        {
          "t": "in",
          "s": "C6"
        },
        {
          "t": "dim",
          "s": "   A B C D E F G H J"
        },
        {
          "t": "dim",
          "s": " 9 . . . . . . . . .  9"
        },
        {
          "t": "dim",
          "s": " 8 . . . . . . . . .  8"
        },
        {
          "t": "dim",
          "s": " 7 . . . . . . . . .  7"
        },
        {
          "t": "dim",
          "s": " 6 . . X). . . . . .  6"
        },
        {
          "t": "dim",
          "s": " 5 . . . . X . . . .  5"
        },
        {
          "t": "dim",
          "s": " 4 . . . O . . . . .  4"
        },
        {
          "t": "dim",
          "s": " 3 . . . . . . . . .  3"
        },
        {
          "t": "dim",
          "s": " 2 . . . . . . . . .  2"
        },
        {
          "t": "dim",
          "s": " 1 . . . . . . . . .  1"
        },
        {
          "t": "gold",
          "s": "Gofer plays F4 (win 54%, pv F4 F5 E3 G4 G3 H4)"
        }
      ]
    },
    "gtp": {
      "title": "bin/gofer gtp -model models/gofer-9x9.onnx",
      "lines": [
        {
          "t": "cmd",
          "s": "bin/gofer gtp -model models/gofer-9x9.onnx"
        },
        {
          "t": "in",
          "s": "boardsize 9"
        },
        {
          "t": "out",
          "s": "= "
        },
        {
          "t": "in",
          "s": "clear_board"
        },
        {
          "t": "out",
          "s": "= "
        },
        {
          "t": "in",
          "s": "komi 7.5"
        },
        {
          "t": "out",
          "s": "= "
        },
        {
          "t": "in",
          "s": "play B E5"
        },
        {
          "t": "out",
          "s": "= "
        },
        {
          "t": "in",
          "s": "genmove W"
        },
        {
          "t": "gold",
          "s": "= D4"
        },
        {
          "t": "in",
          "s": "play B C4"
        },
        {
          "t": "out",
          "s": "= "
        },
        {
          "t": "in",
          "s": "gofer-analyze 800"
        },
        {
          "t": "gold",
          "s": "= value +0.125 winrate 56.2% visits 800"
        },
        {
          "t": "gold",
          "s": "D5    visits   459 winrate  58.4% prior  13.4%"
        },
        {
          "t": "out",
          "s": "C3    visits    62 winrate  55.4% prior   8.5%"
        },
        {
          "t": "out",
          "s": "C5    visits    58 winrate  54.1% prior  11.3%"
        },
        {
          "t": "out",
          "s": "E3    visits    24 winrate  53.1% prior   6.4%"
        },
        {
          "t": "out",
          "s": "E4    visits    20 winrate  55.4% prior   2.9%"
        },
        {
          "t": "out",
          "s": "F4    visits    17 winrate  52.1% prior   5.0%"
        },
        {
          "t": "out",
          "s": "D3    visits    14 winrate  52.5% prior   3.8%"
        },
        {
          "t": "out",
          "s": "F5    visits     8 winrate  54.6% prior   1.4%"
        },
        {
          "t": "out",
          "s": "pv D5 D6 C5 E4 E3 D3 C3 D2"
        }
      ]
    },
    "selfplay": {
      "title": "bin/gofer selfplay",
      "lines": [
        {
          "t": "cmd",
          "s": "bin/gofer selfplay -model runs/9x9/models/gen-0016.onnx -games 600 \\"
        },
        {
          "t": "cmd2",
          "s": "    -full-visits 160 -fast-visits 32 -full-prob 0.25 -out runs/9x9/data/gen-0016"
        },
        {
          "t": "out",
          "s": "selfplay: 50/600 games, 1.0 games/s, 841 rows, black wins 38.0%"
        },
        {
          "t": "out",
          "s": "selfplay: 150/600 games, 1.2 games/s, 2717 rows, black wins 44.0%"
        },
        {
          "t": "out",
          "s": "selfplay: 250/600 games, 1.2 games/s, 4527 rows, black wins 46.8%"
        },
        {
          "t": "out",
          "s": "selfplay: 350/600 games, 1.2 games/s, 6336 rows, black wins 44.9%"
        },
        {
          "t": "out",
          "s": "selfplay: 450/600 games, 1.2 games/s, 8168 rows, black wins 45.3%"
        },
        {
          "t": "dim",
          "s": "…"
        },
        {
          "t": "gold",
          "s": "selfplay: 600 games, 10992 rows in 470s (1.28 games/s), black won 44.5%, resigned 440, false resigns 1/53"
        },
        {
          "t": "com",
          "s": "# 25% of moves get a full 160-visit search and become training rows;"
        },
        {
          "t": "com",
          "s": "# the rest get 32 visits and are only played (playout-cap randomization)."
        }
      ]
    },
    "loop": {
      "title": "bin/gofer arena",
      "lines": [
        {
          "t": "cmd",
          "s": "bin/gofer arena -a nn:models/gofer-9x9.onnx,visits=400 -b gnugo,level=10 -games 100"
        },
        {
          "t": "gold",
          "s": "gen-0017@400 vs gnugo-l10: 66-34-0  score 0.660 [0.563, 0.745]  elo +115 [+44, +187]"
        },
        {
          "t": "cmd",
          "s": "bin/gofer arena -a nn:models/gofer-9x9.onnx,visits=400 -b gnugo,level=1 -games 100"
        },
        {
          "t": "gold",
          "s": "gen-0017@400 vs gnugo-l1: 78-22-0  score 0.780 [0.689, 0.850]  elo +220 [+138, +301]"
        },
        {
          "t": "com",
          "s": "# paired random openings, colors swapped, GNU Go judges dead stones."
        },
        {
          "t": "cmd",
          "s": "grep eval: runs/9x9/log.txt | tail -4"
        },
        {
          "t": "out",
          "s": "eval: gen-0016 vs gen-0015: 25-15-0 elo +89 [-21, +198]"
        },
        {
          "t": "out",
          "s": "eval: gen-0016 vs rollout: 40-0-0 score 1.00"
        },
        {
          "t": "out",
          "s": "eval: gen-0016 vs gnugo-l1: 26-14-0 score 0.65"
        },
        {
          "t": "out",
          "s": "eval: gen-0016 vs gnugo-l6: 23-17-0 score 0.57"
        }
      ]
    }
  };

  const termBody = $('term-body'), termTitle = $('term-title');
  let termToken = 0;
  function lineSpan(l) {
    const span = document.createElement('span');
    if (l.t === 'cmd') { span.innerHTML = '<span style="color:#8B7CF6">$ </span>'; span.appendChild(document.createTextNode(l.s)); span.style.color = '#EAEAF4'; }
    else if (l.t === 'cmd2') { span.textContent = l.s; span.style.color = '#EAEAF4'; }
    else if (l.t === 'in') { span.innerHTML = '<span style="color:#8B7CF6">&gt; </span>'; span.appendChild(document.createTextNode(l.s)); span.style.color = '#EAEAF4'; }
    else if (l.t === 'gold') { span.textContent = l.s; span.style.color = '#D6A85C'; }
    else if (l.t === 'com') { span.textContent = l.s; span.style.color = '#636880'; }
    else if (l.t === 'dim') { span.textContent = l.s; span.style.color = '#8F93AC'; }
    else { span.textContent = l.s; }
    return span;
  }
  async function typeSession(key) {
    const my = ++termToken;
    const ses = SESSIONS[key];
    termTitle.textContent = ses.title;
    termBody.textContent = '';
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    for (const l of ses.lines) {
      if (my !== termToken) return;
      const isTyped = (l.t === 'cmd' || l.t === 'in') && !reduceMotion;
      if (isTyped) {
        const prefix = document.createElement('span');
        prefix.innerHTML = l.t === 'cmd' ? '<span style="color:#8B7CF6">$ </span>' : '<span style="color:#8B7CF6">&gt; </span>';
        const body = document.createElement('span');
        body.style.color = '#EAEAF4';
        termBody.appendChild(prefix); termBody.appendChild(body);
        body.classList.add('caret');
        for (const ch of l.s) {
          if (my !== termToken) return;
          body.textContent += ch;
          await sleep(16);
        }
        body.classList.remove('caret');
        termBody.appendChild(document.createTextNode('\n'));
        await sleep(140);
      } else {
        termBody.appendChild(lineSpan(l));
        termBody.appendChild(document.createTextNode('\n'));
        if (!reduceMotion) await sleep(26);
      }
      termBody.scrollTop = termBody.scrollHeight;
    }
  }
  document.querySelectorAll('.term-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.term-tab').forEach(b => b.className = 'term-tab px-3 py-1.5 rounded-lg border border-line text-ink-dim hover:text-ink');
      btn.className = 'term-tab px-3 py-1.5 rounded-lg border border-violet/50 bg-surface2 text-violet';
      typeSession(btn.dataset.tab);
    });
  });
  const termIO = new IntersectionObserver(es => {
    if (es.some(e => e.isIntersecting)) { termIO.disconnect(); typeSession('play'); }
  }, { threshold: 0.25 });
  termIO.observe(termBody);
})();
