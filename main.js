'use strict';

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

// ── PAPER DATA (Tables 1–3) ───────────────────────────────────
// Layout scores: [T1, T2, T3, Overall], 0–5, macro-averaged over subcategories.
const MODELS = [
  { n: 'GLM-4 32B',         f: 'glm',   s: 0, j1: [3.93, 3.14, 3.64, 3.57], j2: [4.55, 4.13, 4.31, 4.33], piou: 0.993 },
  { n: 'Qwen2.5-Coder 32B', f: 'qwen',  s: 0, j1: [3.79, 3.07, 3.46, 3.44], j2: [4.53, 3.96, 4.03, 4.17], piou: 0.994 },
  { n: 'Qwen2.5-Coder 14B', f: 'qwen',  s: 1, j1: [3.66, 3.02, 3.22, 3.30], j2: [4.35, 3.93, 4.11, 4.13], piou: 0.995 },
  { n: 'Gemma 2 27B',       f: 'gemma', s: 0, j1: [3.31, 2.28, 3.18, 2.92], j2: [4.15, 3.43, 3.91, 3.83], piou: 0.992 },
  { n: 'GLM-4 9B',          f: 'glm',   s: 1, j1: [3.27, 2.59, 2.76, 2.87], j2: [4.13, 3.44, 3.65, 3.74], piou: 0.991 },
  { n: 'Gemma 2 9B',        f: 'gemma', s: 1, j1: [3.13, 2.09, 2.00, 2.41], j2: [4.09, 3.18, 2.70, 3.32], piou: 0.980 },
  { n: 'Llama 3.1 8B',      f: 'llama', s: 1, j1: [2.41, 1.91, 1.79, 2.03], j2: [3.65, 2.81, 2.90, 3.12], piou: 0.929 },
  { n: 'CodeLlama 34B',     f: 'llama', s: 0, j1: [2.31, 1.48, 1.47, 1.75], j2: [3.54, 2.55, 2.53, 2.87], piou: 0.984 },
];
const POOLED = { j1: [3.23, 2.45, 2.69, 2.79], j2: [4.12, 3.43, 3.52, 3.69] };
const TIER_IDX = { t1: 0, t2: 1, t3: 2, all: 3 };

// Table 3: nested-CV pooled R² (prompt TF-IDF vs residual-stream activations)
const DECOMP = [
  { n: 'GLM-4 32B', rows: [
    { c: 'Own',      p: 0.121,  a: 0.318, sig: true },
    { c: 'Shared',   p: 0.289,  a: 0.550, sig: true },
    { c: 'Specific', p: -0.014, a: 0.005, sig: false },
  ]},
  { n: 'Gemma 2 27B', rows: [
    { c: 'Own',      p: 0.085,  a: 0.306, sig: true },
    { c: 'Shared',   p: 0.293,  a: 0.573, sig: true },
    { c: 'Specific', p: -0.029, a: -0.005, sig: false },
  ]},
];

// Exp 3 intervention: continuation rate, null_asis → moved_unused
const CONT = [
  { n: 'GLM-4 32B',   a: 43.8, b: 84.6 },
  { n: 'Gemma 2 27B', a: 2.5,  b: 41.2 },
];

// ── LAYOUT-SCORE CHART ────────────────────────────────────────
const chartState = { judge: 'j1', tier: 'all' };
let chartRows = null;

function buildChart() {
  const host = $('#chart');
  if (!host) return;
  host.innerHTML = '';
  const axis = el('div', 'c-axis');
  for (let v = 0; v <= 5; v++) {
    const g = el('div', 'c-grid'); g.style.left = (v / 5 * 100) + '%';
    g.appendChild(el('span', null, String(v)));
    axis.appendChild(g);
  }
  const thr = el('div', 'c-thr'); thr.style.left = '60%'; thr.title = 'Pass threshold 3.0';
  axis.appendChild(thr);
  host.appendChild(axis);

  const head = el('div', 'cr cr-head');
  head.append(el('div', 'cr-l', 'Model'), el('div', 'cr-track'), el('div', 'cr-piou', 'PIoU'));
  host.appendChild(head);

  chartRows = MODELS.map(m => {
    const row = el('div', 'cr');
    row.style.position = 'relative';
    const lbl = el('div', 'cr-l', m.n); lbl.title = m.n;
    const track = el('div', 'cr-track');
    const bar = el('div', `cr-bar f-${m.f}${m.s ? ' s' : ''}`);
    const val = el('div', 'cr-val');
    track.append(bar, val);
    row.append(lbl, track, el('div', 'cr-piou', m.piou.toFixed(3)));
    host.appendChild(row);
    return { m, row, bar, val };
  });
  host.style.position = 'relative';
  updateChart();
}

function updateChart() {
  if (!chartRows) return;
  const k = TIER_IDX[chartState.tier];
  const sorted = [...chartRows].sort((a, b) => b.m[chartState.judge][k] - a.m[chartState.judge][k]);
  const host = $('#chart');
  sorted.forEach(r => host.appendChild(r.row)); // re-order DOM
  sorted.forEach(r => {
    const v = r.m[chartState.judge][k];
    const pct = v / 5 * 100;
    r.bar.style.width = pct + '%';
    r.val.style.left = pct + '%';
    r.val.textContent = v.toFixed(2);
    r.row.setAttribute('aria-label', `${r.m.n}: ${v.toFixed(2)} of 5; translation PIoU ${r.m.piou.toFixed(3)}`);
  });
}

function bindSeg(attr, key) {
  $$(`.seg-b[data-${attr}]`).forEach(b => b.addEventListener('click', () => {
    $$(`.seg-b[data-${attr}]`).forEach(x => { x.classList.remove('act'); x.setAttribute('aria-selected', 'false'); });
    b.classList.add('act'); b.setAttribute('aria-selected', 'true');
    chartState[key] = b.dataset[attr];
    updateChart();
  }));
}

// ── FULL TABLE ────────────────────────────────────────────────
function buildTable() {
  const tb = $('#tbody');
  if (!tb) return;
  const best = { j1: [0, 1, 2, 3].map(i => Math.max(...MODELS.map(m => m.j1[i]))), j2: [0, 1, 2, 3].map(i => Math.max(...MODELS.map(m => m.j2[i]))) };
  const bestP = Math.max(...MODELS.map(m => m.piou));
  MODELS.forEach(m => {
    const tr = el('tr');
    tr.appendChild(el('td', null, m.n));
    ['j1', 'j2'].forEach(j => m[j].forEach((v, i) => tr.appendChild(el('td', v === best[j][i] ? 'best' : null, v.toFixed(2)))));
    tr.appendChild(el('td', m.piou === bestP ? 'best' : null, m.piou.toFixed(3)));
    tb.appendChild(tr);
  });
  const tr = el('tr', 'pooled');
  tr.appendChild(el('td', null, 'Pooled'));
  ['j1', 'j2'].forEach(j => POOLED[j].forEach(v => tr.appendChild(el('td', null, v.toFixed(2)))));
  tr.appendChild(el('td', null, '—'));
  tb.appendChild(tr);
}

// ── DECOMPOSITION + CONTINUATION ─────────────────────────────
function buildDecomp() {
  const host = $('#decomp');
  if (!host) return;
  const MAX = 0.6;
  const w = v => Math.max(0, v) / MAX * 100;
  DECOMP.forEach(m => {
    host.appendChild(el('div', 'dm', m.n));
    m.rows.forEach(r => {
      const row = el('div', 'dr');
      row.appendChild(el('div', 'dr-l', r.c));
      const bars = el('div', 'dr-bars');
      [['p', r.p, ''], ['a', r.a, r.sig ? ' ***' : '']].forEach(([cls, v, star]) => {
        const b = el('div', 'db ' + cls);
        const i = el('i'); i.style.width = w(v) + '%';
        const s = el('span', null, (v < 0 ? '−' : '') + Math.abs(v).toFixed(3) + star); s.style.left = w(v) + '%';
        b.append(i, s); bars.appendChild(b);
      });
      row.appendChild(bars);
      host.appendChild(row);
    });
  });
}

function buildCont() {
  const host = $('#cont');
  if (!host) return;
  CONT.forEach(c => {
    const card = el('div', 'cm');
    const h = el('div', 'cm-h'); h.append(el('b', null, c.n), el('span', null, `${c.a}% → ${c.b}%`));
    card.appendChild(h);
    [['n', 'Unchanged', c.a], ['m', 'Moved', c.b]].forEach(([cls, lab, v]) => {
      const r = el('div', 'cb ' + cls);
      const t = el('div', 'cb-t'); const i = el('i'); i.dataset.w = v + '%'; i.style.width = REDUCED ? v + '%' : '0%';
      t.appendChild(i);
      r.append(el('div', 'cb-l', lab), t, el('div', 'cb-v', v.toFixed(1) + '%'));
      card.appendChild(r);
    });
    host.appendChild(card);
  });
}

// ── PIPELINE OCCUPANCY ILLUSTRATION ──────────────────────────
function buildOcc() {
  const host = $('.occ');
  if (!host) return;
  // Illustrative 6×6 occupancy: dense centre, sparse rays.
  const g = [
    .25, .05, .35, .35, .05, .25,
    .05, .55, .85, .85, .55, .05,
    .35, .85, 1, 1, .85, .35,
    .35, .85, 1, 1, .85, .35,
    .05, .55, .85, .85, .55, .05,
    .25, .05, .35, .35, .05, .25,
  ];
  g.forEach(v => { const i = el('i'); i.style.opacity = (0.12 + 0.88 * v).toFixed(2); host.appendChild(i); });
}

// ── AM-BENCH SAMPLES ──────────────────────────────────────────
const SAMPLES = [
  'a yellow diamond on grey', 'a yellow sun with rays on blue', 'a red and blue checkerboard', 'a white shape inside a teal frame',
  'concentric navy and white rings', 'a yellow fish on black', 'a red rounded shape', 'a striped red, white and blue flag',
  'a grey mouse on beige', 'red, yellow and green lights on a pole', 'diagonal runs of colored squares', 'a black and white checkerboard',
  'a house under a brown roofline on grass', 'six orange figures on black', 'a beach sunset with two palm trees',
];
function buildSamples() {
  const host = $('#samples');
  if (!host) return;
  SAMPLES.forEach((alt, i) => {
    const img = el('img');
    img.src = `assets/am-bench/sample-${String(i + 1).padStart(2, '0')}.png`;
    img.alt = `AM-Bench layout output: ${alt}`;
    img.decoding = 'async'; img.width = 154; img.height = 154;
    host.appendChild(img);
  });
}

// ── PROTOTYPE GALLERY ─────────────────────────────────────────
const gdata = [
  // Code Canvas
  {p: 'a red heart',                  t: 'code',  src: 'gallery_images/code-canvas/a-red-heart-1771859003512.png'},
  {p: 'pyramids of giza',            t: 'code',  src: 'gallery_images/code-canvas/symbolic-pyramids-of-giza-1772107669287.png'},
  {p: 'a sunset in malibu',          t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-sunset-in-malibu-1772108429304.png'},
  {p: 'flag of the united states',   t: 'code',  src: 'gallery_images/code-canvas/symbolic-flag-of-the-united-s-1772109529064.png'},
  {p: 'a penguin',                   t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-penguin-1772110080534.png'},
  {p: 'a car',                       t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-car-1772114891174.png'},
  {p: 'a house by the mountains',    t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-house-by-the-mount-1772115857558.png'},
  {p: 'oranges',                     t: 'code',  src: 'gallery_images/code-canvas/symbolic-oranges-1772116021564.png'},
  {p: 'a red strawberry',            t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-red-strawberry-1772116167946.png'},
  {p: 'a red strawberry',            t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-red-strawberry-1772116413723.png'},
  {p: 'starry night by van gogh',    t: 'code',  src: 'gallery_images/code-canvas/symbolic-starry-night-by-van--1772533732490.png'},
  {p: 'a sunset in malibu',          t: 'code',  src: 'gallery_images/code-canvas/symbolic-a-sunset-in-malibu-1772629040402.png'},
  {p: "gaudí's sagrada família",     t: 'code',  src: 'gallery_images/code-canvas/symbolic-gaudi-s-sagrada-fami-1772629582165.png'},
  {p: 'an alien planet',             t: 'code',  src: 'gallery_images/code-canvas/symbolic-an-alien-planet-1772629723598.png'},
  {p: 'mars',                        t: 'code',  src: 'gallery_images/code-canvas/symbolic-mars-1772630556722.png'},
  {p: 'guitar',                      t: 'code',  src: 'gallery_images/code-canvas/symbolic-guitar--1772660859483.png'},
  {p: 'pyramids of giza',            t: 'code',  src: 'gallery_images/code-canvas/symbolic-pyramids-of-giza-1772661467477.png'},
  {p: "king tut's tomb",             t: 'code',  src: 'gallery_images/code-canvas/symbolic-king-tut-s-tomb-1772661642734.png'},
  {p: 'pizza',                       t: 'code',  src: 'gallery_images/code-canvas/symbolic-pizza-1772662270005.png'},
  {p: 'elephant',                    t: 'code',  src: 'gallery_images/code-canvas/symbolic-elephant-1772662649045.png'},
  // ASCII Canvas
  {p: 'a house',                     t: 'ascii', src: 'gallery_images/ascii-canvas/a-house-1772020117175.png'},
  {p: 'an astronaut',                t: 'ascii', src: 'gallery_images/ascii-canvas/an-astronaut-1772020335977.png'},
  {p: 'an astronaut',                t: 'ascii', src: 'gallery_images/ascii-canvas/an-astronaut-1772020369859.png'},
  {p: 'a house',                     t: 'ascii', src: 'gallery_images/ascii-canvas/a-house-1772020447989.png'},
  {p: 'ancient rome',                t: 'ascii', src: 'gallery_images/ascii-canvas/ancient-rome-1772022050599.png'},
  {p: 'ancient rome',                t: 'ascii', src: 'gallery_images/ascii-canvas/ancient-rome-1772022097006.png'},
  {p: 'ancient greece',              t: 'ascii', src: 'gallery_images/ascii-canvas/ancient-greece-1772022272121.png'},
  {p: 'ancient egypt',               t: 'ascii', src: 'gallery_images/ascii-canvas/ancient-egypt-1772022327409.png'},
  {p: 'a mummy from egypt',          t: 'ascii', src: 'gallery_images/ascii-canvas/a-mummy-from-egypt-1772022669684.png'},
  {p: 'stop signs',                  t: 'ascii', src: 'gallery_images/ascii-canvas/stop-signs-1772023617592.png'},
  {p: 'eggs sunny side up',          t: 'ascii', src: 'gallery_images/ascii-canvas/eggs-sunny-side-1772024307740.png'},
  {p: 'egg sunny side up',           t: 'ascii', src: 'gallery_images/ascii-canvas/egg-sunny-side-up-1772024620222.png'},
  {p: 'stop sign',                   t: 'ascii', src: 'gallery_images/ascii-canvas/stop-sign-1772024722662.png'},
  {p: 'traffic sign',                t: 'ascii', src: 'gallery_images/ascii-canvas/traffic-sign-1772024765113.png'},
  {p: 'a red heart',                 t: 'ascii', src: 'gallery_images/ascii-canvas/a-red-heart-1772024836144.png'},
  {p: 'cvpr conference',             t: 'ascii', src: 'gallery_images/ascii-canvas/cvpr-conference-1772025285082.png'},
  {p: 'ducks',                       t: 'ascii', src: 'gallery_images/ascii-canvas/ducks-1772099854837.png'},
  {p: 'butterfly',                   t: 'ascii', src: 'gallery_images/ascii-canvas/butterfly-1772099898256.png'},
  {p: 'a sunset in malibu',          t: 'ascii', src: 'gallery_images/ascii-canvas/mosaic-a-sunset-in-malibu-1772101308458.png'},
];

let cf = 'all';
function renderG() {
  const gg = $('#gg');
  if (!gg) return;
  gg.innerHTML = '';
  (cf === 'all' ? gdata : gdata.filter(d => d.t === cf)).forEach(item => {
    const label = item.t === 'ascii' ? 'ASCII canvas' : 'Code canvas';
    const btn = el('button', 'gi'); btn.type = 'button';
    btn.setAttribute('aria-label', `Enlarge: “${item.p}”, ${label}`);
    const img = el('img'); img.src = item.src; img.alt = `${label} output for “${item.p}”`; img.loading = 'lazy';
    const tag = el('div', 'ttag ' + item.t, item.t === 'ascii' ? 'ASCII' : 'Code');
    btn.append(img, el('div', 'lbl', `“${item.p}”`), tag);
    btn.addEventListener('click', () => openLb(item.src, `“${item.p}” · ${label}`, btn));
    gg.appendChild(btn);
  });
}
$$('.fb').forEach(b => b.addEventListener('click', () => {
  cf = b.dataset.f;
  $$('.fb').forEach(x => { x.classList.remove('act'); x.setAttribute('aria-pressed', 'false'); });
  b.classList.add('act'); b.setAttribute('aria-pressed', 'true');
  renderG();
}));

// ── LIGHTBOX ─────────────────────────────────────────────────
let lbReturn = null;
function openLb(src, cap, from) {
  const lb = $('#lb');
  $('#lbimg').src = src; $('#lbimg').alt = cap; $('#lbcap').textContent = cap;
  lb.hidden = false; lbReturn = from; $('#lbx').focus();
  document.body.style.overflow = 'hidden';
}
function closeLb() {
  $('#lb').hidden = true; document.body.style.overflow = '';
  if (lbReturn) lbReturn.focus();
}
$('#lbx').addEventListener('click', closeLb);
$('#lb').addEventListener('click', e => { if (e.target.id === 'lb') closeLb(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('#lb').hidden) closeLb(); });

// ── CITATION ─────────────────────────────────────────────────
let curBib = 'wacv';
$$('.bib-t').forEach(b => b.addEventListener('click', () => {
  curBib = b.dataset.bib;
  $$('.bib-t').forEach(x => { x.classList.remove('act'); x.setAttribute('aria-selected', 'false'); });
  b.classList.add('act'); b.setAttribute('aria-selected', 'true');
  $$('.bib-p').forEach(p => { p.hidden = p.id !== 'bib-' + curBib; });
}));
$('#cpbtn').addEventListener('click', () => {
  const txt = $('#bib-' + curBib).textContent;
  const done = () => { const b = $('#cpbtn'); b.textContent = 'Copied!'; setTimeout(() => (b.textContent = 'Copy'), 2000); };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(txt).then(done).catch(() => fallbackCopy(txt, done));
  } else fallbackCopy(txt, done);
});
function fallbackCopy(txt, done) {
  const ta = el('textarea'); ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); done(); } catch (_) { /* ignore */ }
  ta.remove();
}

// ── HERO "BOUNCE" ANIMATION ─────────────────────────────────
// A wave of colour noise sweeps in left → right, the mosaics lock in right → left,
// then the noise sweeps back left → right and the tiles return right → left, in a loop.
// Uses drawImage only (no getImageData), so it works on file:// and GitHub Pages alike.
const PANEL_SRCS = [
  'symbolic-a-red-apple--1772103332350.png',
  'symbolic-king-tut-s-tomb-1772661687365.png',
  'symbolic-oranges-1772115891729.png',
  'symbolic-a-bird-1772110130942.png',
];
const QUARTET_FB = ['#E63946', '#FCD34D', '#2A9D8F', '#2B52C8'];
const CHAOS = [
  '#E63946','#FF4D6D','#FF8500','#FCD34D','#CFFF04',
  '#2A9D8F','#06D6A0','#00B4D8','#3D66DC','#7B2FBE',
  '#F4A261','#E76F51','#FFBE0B','#8338EC','#3A86FF',
  '#FF006E','#FB5607','#43AA8B','#277DA1','#F72585'
];

// Timeline (ms). One cycle = tiles sweep in (R→L) · hold · noise sweeps back (L→R) · hold.
const HERO = {
  CELL: 28,
  INTRO: 1100,      // initial noise fade-in sweep, left → right
  TILE_SWEEP: 4200, // mosaics lock in, right → left
  TILE_HOLD: 2600,  // all tiles visible
  NOISE_SWEEP: 2600,// tiles dissolve back to noise, left → right
  NOISE_HOLD: 700,  // all noise before the next tile sweep
  FADE: 180,        // cross-fade per cell at a state change
  FLASH: 0.22,      // brightness of the wave front (0 to disable)
  SLANT: 0.08,      // how much rows tilt the wave front
  J_LOCK: 400,      // random per-cell spread of each wave front
  J_FREE: 300,
};
HERO.PERIOD = HERO.TILE_SWEEP + HERO.J_LOCK + HERO.TILE_HOLD + HERO.NOISE_SWEEP + HERO.J_FREE + HERO.NOISE_HOLD;
HERO.T0 = HERO.INTRO * 0.55; // first tile sweep starts while the intro wave is still travelling

const hc = $('#heroCvs');
const hctx = hc.getContext('2d');
let _panelImgs = [], _target = null, _aw, _ah, _cells = [], _start, _rafId, _resizeT, _lastW = 0, _heroVisible = true;

// Deterministic per-cell, per-cycle jitter in [0, 1).
function _hash(i, k) {
  let h = (i * 374761393 + k * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
const _clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);

function _preloadPanels() {
  return Promise.all(PANEL_SRCS.map(src => new Promise(res => {
    const img = new Image();
    img.onload = () => res(img); img.onerror = () => res(null); img.src = src;
  })));
}

function _buildTarget(w, h) {
  _target = document.createElement('canvas');
  _target.width = w; _target.height = h;
  const ctx = _target.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  const pw = w / 4;
  _panelImgs.forEach((img, i) => {
    if (img) ctx.drawImage(img, i * pw, 0, pw, h);
    else { ctx.fillStyle = QUARTET_FB[i]; ctx.fillRect(i * pw, 0, pw, h); }
  });
}

function _buildCells() {
  _aw = hc.offsetWidth; _ah = hc.offsetHeight;
  if (!_aw || !_ah) return;
  hc.width = _aw; hc.height = _ah;
  hctx.imageSmoothingEnabled = false;
  _buildTarget(_aw, _ah);
  const cols = Math.ceil(_aw / HERO.CELL), rows = Math.ceil(_ah / HERO.CELL);
  _start = performance.now();
  _cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const nx = c / cols, ny = r / rows;
      const pos = (nx + HERO.SLANT * ny) / (1 + HERO.SLANT); // 0 = left edge, 1 = right edge
      const flipInterval = 80 + nx * 340 + Math.random() * 140;
      _cells.push({ i: _cells.length, c, r, nx, pos, flipInterval,
        enter: pos * HERO.INTRO + Math.random() * 80,
        nextFlip: Math.random() * flipInterval,
        chaosColor: CHAOS[(Math.random() * CHAOS.length) | 0] });
    }
  }
}

// Switch times for cell in cycle k: [lock (noise → tile), free (tile → noise)].
function _times(cell, k) {
  const ck = HERO.T0 + k * HERO.PERIOD;
  const tLock = ck + (1 - cell.pos) * HERO.TILE_SWEEP + _hash(cell.i, 2 * k) * HERO.J_LOCK;
  const tFree = ck + HERO.TILE_SWEEP + HERO.J_LOCK + HERO.TILE_HOLD
              + cell.pos * HERO.NOISE_SWEEP + _hash(cell.i, 2 * k + 1) * HERO.J_FREE;
  return [tLock, tFree];
}

// Tile mix for a cell at time t (0 = noise, 1 = mosaic tile) and the time of its last switch.
function _mix(cell, t) {
  const k = Math.floor((t - HERO.T0) / HERO.PERIOD);
  if (k < 0) return [0, -Infinity];
  const [tLock, tFree] = _times(cell, k);
  if (t >= tFree) return [1 - _clamp01((t - tFree) / HERO.FADE), tFree];
  if (t >= tLock) return [_clamp01((t - tLock) / HERO.FADE), tLock];
  if (k === 0) return [0, -Infinity];
  const prevFree = _times(cell, k - 1)[1];
  return [1 - _clamp01((t - prevFree) / HERO.FADE), prevFree];
}

function _paint(t) {
  hctx.clearRect(0, 0, _aw, _ah);
  const cs = HERO.CELL - 1;
  for (const cell of _cells) {
    const enterAlpha = _clamp01((t - cell.enter) / 150);
    if (enterAlpha <= 0) continue;
    const [m, since] = _mix(cell, t);
    const cx = cell.c * HERO.CELL, cy = cell.r * HERO.CELL;

    if (m < 1) {
      if (t >= cell.nextFlip) {
        cell.chaosColor = CHAOS[(Math.random() * CHAOS.length) | 0];
        cell.nextFlip = t + cell.flipInterval * (0.5 + Math.random());
      }
      hctx.globalAlpha = (0.26 + (1 - cell.nx) * 0.20) * enterAlpha * (1 - m);
      hctx.fillStyle = cell.chaosColor;
      hctx.fillRect(cx, cy, cs, cs);
    }
    if (m > 0 && _target) {
      hctx.globalAlpha = (0.42 + cell.nx * 0.20) * enterAlpha * m;
      hctx.drawImage(_target, cx, cy, cs, cs, cx, cy, cs, cs);
    }
    // Bright flash where the wave front passes.
    const age = t - since;
    if (HERO.FLASH > 0 && age >= 0 && age < 400) {
      hctx.globalAlpha = HERO.FLASH * Math.exp(-age / 120) * enterAlpha;
      hctx.fillStyle = '#F4F1EA';
      hctx.fillRect(cx, cy, cs, cs);
    }
  }
  hctx.globalAlpha = 1;
}

function _drawFrame(now) {
  if (!_heroVisible) { _rafId = null; return; }
  _paint(now - _start);
  _rafId = requestAnimationFrame(_drawFrame);
}

async function initHero() {
  if (_rafId) cancelAnimationFrame(_rafId);
  _rafId = null;
  if (_panelImgs.length === 0) _panelImgs = await _preloadPanels();
  _buildCells();
  _lastW = _aw;
  if (REDUCED) { // static frame: every tile locked in, no flashes
    _cells.forEach(c => { c.enter = -1e9; });
    const flash = HERO.FLASH; HERO.FLASH = 0;
    _paint(HERO.T0 + HERO.TILE_SWEEP + 800);
    HERO.FLASH = flash;
    return;
  }
  _rafId = requestAnimationFrame(_drawFrame);
}

setTimeout(initHero, 60);
window.addEventListener('resize', () => {
  clearTimeout(_resizeT);
  _resizeT = setTimeout(() => { if (hc.offsetWidth !== _lastW) initHero(); }, 150); // ignore mobile URL-bar height jitter
});
// Pause the animation while the hero is off-screen (the timeline resumes where it left off).
let _pausedAt = 0;
new IntersectionObserver(es => es.forEach(e => {
  _heroVisible = e.isIntersecting;
  if (!_heroVisible) { _pausedAt = performance.now(); return; }
  if (!_rafId && !REDUCED && _cells.length) {
    if (_pausedAt) _start += performance.now() - _pausedAt;
    _rafId = requestAnimationFrame(_drawFrame);
  }
})).observe(hc);

// ── SCROLL REVEAL + NAV HIGHLIGHT ────────────────────────────
const revealObs = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return;
  e.target.classList.add('vis');
  $$('.cb-t i', e.target).forEach(i => (i.style.width = i.dataset.w));
  revealObs.unobserve(e.target);
}), { threshold: 0.04 });

const navLinks = $$('.nav-links a');
const navObs = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return;
  navLinks.forEach(a => a.classList.toggle('cur', a.getAttribute('href') === '#' + e.target.id));
}), { rootMargin: '-45% 0px -50% 0px' });

// ── INIT ─────────────────────────────────────────────────────
buildOcc();
buildSamples();
buildChart();
bindSeg('judge', 'judge');
bindSeg('tier', 'tier');
buildTable();
buildDecomp();
buildCont();
renderG();
$$('.rv').forEach(s => revealObs.observe(s));
['abstract', 'benchmark', 'results', 'art', 'cite'].forEach(id => { const s = document.getElementById(id); if (s) navObs.observe(s); });

// ── DAY / NIGHT TOGGLE ───────────────────────────────────────
(function () {
  const btn = document.getElementById('themeBtn');
  if (!btn) return;
  const root = document.documentElement;
  const meta = document.querySelector('meta[name="theme-color"]');
  const sync = () => {
    const light = root.dataset.theme === 'light';
    const label = light ? 'Switch to dark theme' : 'Switch to light theme';
    btn.setAttribute('aria-label', label); btn.title = label;
    btn.setAttribute('aria-pressed', String(light));
    if (meta) meta.content = light ? '#F4EFE4' : '#0B0B0C';
  };
  btn.addEventListener('click', () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    if (!REDUCED) { root.classList.add('theming'); setTimeout(() => root.classList.remove('theming'), 400); }
    if (next === 'light') root.dataset.theme = 'light'; else delete root.dataset.theme;
    try { localStorage.setItem('am-theme', next); } catch (_) { /* storage unavailable */ }
    sync();
  });
  sync();
})();
