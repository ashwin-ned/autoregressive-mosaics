// ── GALLERY ──────────────────────────────────────────────────
const gdata = [
  // Code Canvas
  {p: 'hierarchical mosaic',         t: 'code',  src: 'gallery_images/code-canvas/hierarchical-mosaic-1771859003512.png'},
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

function setF(t, btn) {
  cf = t;
  document.querySelectorAll('.fb').forEach(b => b.classList.remove('act'));
  btn.classList.add('act');
  renderG();
}

function renderG() {
  const gg = document.getElementById('gg');
  gg.innerHTML = '';
  (cf === 'all' ? gdata : gdata.filter(d => d.t === cf)).forEach(item => {
    const div = document.createElement('div'); div.className = 'gi';
    const img = document.createElement('img');
    img.src = item.src;
    img.alt = item.p;
    img.loading = 'lazy';
    const lbl = document.createElement('div'); lbl.className = 'lbl'; lbl.textContent = `"${item.p}"`;
    const tag = document.createElement('div');
    tag.className = 'ttag ' + (item.t === 'ascii' ? 'ascii' : 'code');
    tag.textContent = item.t === 'ascii' ? 'ASCII' : 'Code';
    div.append(img, lbl, tag);
    gg.appendChild(div);
  });
}

renderG();

// ── HERO CONVERGENCE ANIMATION ──────────────────────────────
// Left: vibrant colorful chaos → Right: 4 mosaic images tiled side-by-side.
// Uses drawImage (not getImageData) so it works on file:// and GitHub Pages alike.
// Each image occupies exactly 25% of the canvas width.

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

const hc = document.getElementById('heroCvs');
const hctx = hc.getContext('2d');
const CELL = 28;

let _panelImgs = [];   // loaded HTMLImageElements
let _target = null;    // offscreen canvas: 4 images tiled at hero dimensions
let _aw, _ah, _cells = [], _start, _rafId, _resizeT, _loopAt;

async function _preloadPanels() {
  _panelImgs = await Promise.all(
    PANEL_SRCS.map(src => new Promise(res => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = () => res(null);
      img.src = src;
    }))
  );
}

// Pre-render all 4 images into a single offscreen canvas at hero dimensions.
// drawImage has no CORS restriction — no getImageData needed.
function _buildTarget(w, h) {
  _target = document.createElement('canvas');
  _target.width = w; _target.height = h;
  const ctx = _target.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  const pw = w / 4;
  _panelImgs.forEach((img, i) => {
    if (img) {
      ctx.drawImage(img, i * pw, 0, pw, h);
    } else {
      ctx.fillStyle = QUARTET_FB[i];
      ctx.fillRect(i * pw, 0, pw, h);
    }
  });
}

function _buildCells() {
  _aw = hc.offsetWidth; _ah = hc.offsetHeight;
  if (!_aw || !_ah) return;
  hc.width = _aw; hc.height = _ah;
  hctx.imageSmoothingEnabled = false;
  _buildTarget(_aw, _ah);

  const cols = Math.ceil(_aw / CELL), rows = Math.ceil(_ah / CELL);
  _start = performance.now();
  _cells = [];
  let maxLock = 0;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const nx = c / cols;
      const enterDelay = c * 10 + r * 3 + Math.random() * 30;
      const lockAt = enterDelay + 200 + (1 - nx) * 4500 + Math.random() * 500;
      maxLock = Math.max(maxLock, lockAt);
      const flipInterval = 80 + nx * 340 + Math.random() * 140;

      _cells.push({
        c, r, nx,
        chaosColor: CHAOS[Math.floor(Math.random() * CHAOS.length)],
        nextFlip: enterDelay + Math.random() * flipInterval,
        flipInterval, enterDelay, lockAt,
        locked: false,
      });
    }
  }
  _loopAt = maxLock + 2500;
}

function _drawFrame(now) {
  const age = now - _start;

  if (age > _loopAt) {
    _buildCells();
    _rafId = requestAnimationFrame(_drawFrame);
    return;
  }

  hctx.clearRect(0, 0, _aw, _ah);
  for (const cell of _cells) {
    const entered = age - cell.enterDelay;
    if (entered < 0) continue;
    const enterAlpha = Math.min(1, entered / 150);

    if (!cell.locked && age >= cell.lockAt) cell.locked = true;
    if (!cell.locked && age >= cell.nextFlip) {
      cell.chaosColor = CHAOS[Math.floor(Math.random() * CHAOS.length)];
      cell.nextFlip = age + cell.flipInterval * (0.5 + Math.random());
    }

    const cx = cell.c * CELL, cy = cell.r * CELL, cs = CELL - 1;

    if (cell.locked && _target) {
      hctx.globalAlpha = (0.42 + cell.nx * 0.20) * enterAlpha;
      hctx.drawImage(_target, cx, cy, cs, cs, cx, cy, cs, cs);
    } else {
      hctx.globalAlpha = (0.26 + (1 - cell.nx) * 0.20) * enterAlpha;
      hctx.fillStyle = cell.chaosColor;
      hctx.fillRect(cx, cy, cs, cs);
    }
  }
  hctx.globalAlpha = 1;
  _rafId = requestAnimationFrame(_drawFrame);
}

async function initHero() {
  if (_rafId) cancelAnimationFrame(_rafId);
  if (_panelImgs.length === 0) await _preloadPanels();
  _buildCells();
  _rafId = requestAnimationFrame(_drawFrame);
}

setTimeout(initHero, 80);
window.addEventListener('resize', () => { clearTimeout(_resizeT); _resizeT = setTimeout(initHero, 100); });

// ── COPY BIBTEX ──────────────────────────────────────────────
function doCopy() {
  const txt = `@misc{ned2026autoregressivemosaics,\n  author       = {Nedungadi, Ashwin},\n  title        = {Autoregressive Mosaics},\n  year         = {2026},\n  publisher    = {GitHub},\n  booktitle    = {CVPR AI Art Gallery},\n  howpublished = {\\url{https://github.com/ashwin-ned/autoregressive-mosaics}}\n}`;
  navigator.clipboard.writeText(txt).then(() => {
    const b = document.getElementById('cpbtn');
    b.textContent = 'Copied!';
    setTimeout(() => b.textContent = 'Copy', 2000);
  });
}

// ── SCROLL REVEAL ─────────────────────────────────────────────
const obs = new IntersectionObserver(
  es => es.forEach(e => { if (e.isIntersecting) e.target.classList.add('vis'); }),
  { threshold: 0.06 }
);
document.querySelectorAll('.rv').forEach(el => obs.observe(el));
