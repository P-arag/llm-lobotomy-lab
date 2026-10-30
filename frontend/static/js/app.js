// DOM elements
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const colLeft       = $(".col-left");
const orbStage      = $("#orbStage");
const stageHalo     = $("#stageHalo");
const orbMover      = $("#orbMover");
const orbActor      = $("#orbActor");
const orbCanvas     = $("#orbCanvas");
const orbPulse      = $("#orbPulse");
const splitL        = $("#splitL");
const splitR        = $("#splitR");
const orbCanvasL    = $("#orbCanvasL");
const orbCanvasR    = $("#orbCanvasR");
const orbStatus     = $("#orbStatus");
const statusLabel   = $("#statusLabel");

const chatRow       = $("#chatRow");
const panelNormal   = $("#panelNormal");
const panelSteered  = $("#panelSteered");
const chatNormal    = $("#chatNormal");
const chatSteered   = $("#chatSteered");

const promptBar     = $("#promptBar");
const promptPill    = $("#promptPill");
const promptForm    = $("#promptForm");
const promptInput   = $("#promptInput");
const sendBtn       = $("#sendBtn");

const responseLegend = $("#responseLegend");

const sliderRefusal    = $("#sliderRefusal");
const sliderSycophancy = $("#sliderSycophancy");
const sliderLayer      = $("#sliderLayer");
const refusalValue     = $("#refusalValue");
const sycophancyValue  = $("#sycophancyValue");
const layerValue       = $("#layerValue");

// ── State ────────────────────────────────────────────────────────
let currentLayer    = 13;
let pcaBackground   = null;
let currentBasePoint = null;
let currentPoint    = null;
let lastUmapData    = null;
let umapRotationRAF = null;
let isBusy          = false;

const TAU = Math.PI * 2;

// Dotted sphere renderer

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildDots(rings = 14) {
  const rand = mulberry32(7);
  const dots = [];
  for (let k = 0; k < rings; k++) {
    const y = 1 - ((k + 0.5) / rings) * 2;
    const r = Math.sqrt(1 - y * y);
    const m = Math.max(4, Math.round(26 * r));
    for (let j = 0; j < m; j++) {
      const a = (j / m) * TAU + k * 0.35;
      dots.push({ x: Math.cos(a) * r, y, z: Math.sin(a) * r, seed: rand() * TAU });
    }
  }
  return dots;
}

const DOTS = buildDots();

function createOrbRenderer(canvas, opts = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return { start() {}, stop() {}, setColor() {} };

  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = canvas.width;
  const H = canvas.height;
  canvas.width  = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width  = (W / 2) + "px";
  canvas.style.height = (H / 2) + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const cw = W / 2, ch = H / 2;
  const R = opts.radius || 50;

  let color = opts.color || [88, 166, 255];
  let baseRgb = `rgb(${color[0]},${color[1]},${color[2]})`;
  let speed = opts.speed || 0.4;
  let time = 0;
  let raf = 0;
  let alive = false;

  const sorted = DOTS.map((d, i) => ({ x: 0, y: 0, z: 0, seed: d.seed, idx: i }));

  function draw(dt) {
    ctx.clearRect(0, 0, W, H);
    time += dt * speed;

    const cosY = Math.cos(time), sinY = Math.sin(time);
    const cosP = Math.cos(0.35), sinP = Math.sin(0.35);

    // Rotate & update in-place without heap allocations
    for (let i = 0; i < DOTS.length; i++) {
      const d = DOTS[i];
      const x1 = d.x * cosY + d.z * sinY;
      const z1 = -d.x * sinY + d.z * cosY;
      sorted[i].x = x1;
      sorted[i].y = d.y * cosP - z1 * sinP;
      sorted[i].z = d.y * sinP + z1 * cosP;
    }
    sorted.sort((a, b) => a.z - b.z);

    ctx.fillStyle = baseRgb;
    for (let i = 0; i < sorted.length; i++) {
      const p = sorted[i];
      const depth = (p.z + 1) * 0.5;
      const f = 2.5 / (2.5 - p.z);
      const sx = cw + p.x * R * f;
      const sy = ch - p.y * R * f;

      const pulse = 0.08 * Math.sin(p.seed + time * 3);
      const alpha = Math.max(0, Math.min(1, 0.1 + 0.5 * depth * depth + pulse));
      const r = (0.6 + 0.8 * depth) * f * (opts.dotScale || 1);

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, TAU);
      ctx.fill();
    }
  }

  let last = 0;
  function frame(now) {
    if (!alive) return;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    draw(dt);
    raf = requestAnimationFrame(frame);
  }

  return {
    start() {
      if (alive) return;
      alive = true;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    },
    stop() {
      alive = false;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      ctx.clearRect(0, 0, W, H);
    },
    setColor(c) {
      color = c;
      baseRgb = `rgb(${c[0]},${c[1]},${c[2]})`;
    },
    setSpeed(s) { speed = s; },
  };
}


// ══════════════════════════════════════════════════════════════════
//  ANIMATION ORCHESTRATOR
// ══════════════════════════════════════════════════════════════════

const LABELS = ["Thinking"];

// Easing
const easeOut   = (t) => 1 - Math.pow(1 - t, 3);
const easeIO    = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const spring    = (t) => { const c4 = (2 * Math.PI) / 3; return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1; };

function sleep(ms) { return new Promise(r => setTimeout(r, Math.max(0, ms))); }

function animate(duration, fn, easeFn = easeIO) {
  return new Promise((resolve) => {
    const start = performance.now();
    function tick(now) {
      const elapsed = now - start;
      const raw = Math.min(1, elapsed / duration);
      const t = easeFn(raw);
      fn(t, raw);
      if (raw < 1) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}

function getCenterOf(el) {
  const parentRect = colLeft.getBoundingClientRect();
  const elRect = el.getBoundingClientRect();
  return {
    x: (elRect.left + elRect.width / 2) - parentRect.left,
    y: (elRect.top + elRect.height / 2) - parentRect.top,
  };
}

// Renderers
const mainOrb = createOrbRenderer(orbCanvas, { radius: 52, speed: 0.6, dotScale: 1.2 });
const leftOrb = createOrbRenderer(orbCanvasL, { radius: 38, speed: 0.5, color: [244, 63, 94], dotScale: 1 });
const rightOrb = createOrbRenderer(orbCanvasR, { radius: 38, speed: 0.5, color: [16, 185, 129], dotScale: 1 });

let pulseAlive = false;
let labelInterval = null;

async function runAnimation(prompt) {
  if (isBusy) return;
  isBusy = true;
  orbStage.classList.add("active");

  try {
    promptBar.classList.add("prompt-bar--hidden");
    clearChat(chatNormal);
    clearChat(chatSteered);
    appendMsg(chatNormal, "user", prompt);
    appendMsg(chatSteered, "user", prompt);

    if (responseLegend) responseLegend.classList.remove("visible");

    // Position the main orb at center of stage
    orbMover.style.position = "absolute";
    orbMover.style.left = "50%";
    orbMover.style.top = "50%";
    orbMover.style.transform = "translate(-50%, -50%)";
    orbMover.style.bottom = "auto";

    // Grow the orb
    orbActor.style.width = "0px";
    orbActor.style.height = "0px";
    stageHalo.style.opacity = "0";
    orbCanvas.style.opacity = "0";

    await sleep(200);

    // Animate orb appearing
    await animate(500, (t) => {
      const size = t * 132;
      orbActor.style.width = size + "px";
      orbActor.style.height = size + "px";
      stageHalo.style.opacity = (t * 0.6).toFixed(3);
      orbCanvas.style.opacity = t.toFixed(3);
      const surf = orbActor.querySelector(".orb-surface");
      if (surf) {
        surf.style.opacity = t.toFixed(3);
        surf.style.transform = `scale(${0.8 + 0.2 * t})`;
      }
      const glow = orbActor.querySelector(".orb-glow");
      if (glow) glow.style.opacity = (t * 0.8).toFixed(3);
    }, easeOut);

    mainOrb.start();

    orbStatus.style.opacity = "1";
    statusLabel.textContent = LABELS[0];

  // Start API call in parallel
  const payload = {
    prompt,
    alpha_refusal:    parseFloat(sliderRefusal.value),
    alpha_sycophancy: parseFloat(sliderSycophancy.value),
    layer:            parseInt(sliderLayer.value, 10),
  };

    const apiPromise = Promise.all([
      fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      }),
      fetch("/api/project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then(r => {
        if (!r.ok) return { prompt_point: null };
        return r.json();
      }).catch(() => ({ prompt_point: null })),
    ]);

    // Cycle labels while waiting
    let labelIdx = 0;
    labelInterval = setInterval(() => {
      labelIdx = (labelIdx + 1) % LABELS.length;
      statusLabel.style.opacity = "0";
      statusLabel.style.transform = "translateY(-6px)";
      setTimeout(() => {
        statusLabel.textContent = LABELS[labelIdx];
        statusLabel.style.transform = "translateY(6px)";
        requestAnimationFrame(() => {
          statusLabel.style.opacity = "1";
          statusLabel.style.transform = "translateY(0)";
        });
      }, 180);
    }, 1200);

    // Pulse animation on the orb
    pulseAlive = true;
    (async function pulseLoop() {
      while (pulseAlive) {
        await animate(900, (t) => {
          orbPulse.style.opacity = (0.3 * (1 - t)).toFixed(3);
          orbPulse.style.transform = `scale(${0.9 + 0.4 * t})`;
        }, easeOut);
        await sleep(400);
      }
    })();

    // Ensure a minimum thinking time for visual effect
    const minThinkPromise = sleep(1900);

    let genData = null;
    let projData = null;
    try {
      const [apiResults] = await Promise.all([apiPromise, minThinkPromise]);
      genData = apiResults[0];
      projData = apiResults[1];
    } catch (apiErr) {
      console.error("API error during generation:", apiErr);
      appendMsg(chatNormal, "bot", "Connection error: Unable to contact backend. Please try again.");
      appendMsg(chatSteered, "bot", "Connection error: Unable to contact backend. Please try again.");
      return;
    } finally {
      clearInterval(labelInterval);
      pulseAlive = false;
    }

    // Show "Done" briefly
    statusLabel.textContent = "Done";
    await sleep(350);

    orbStatus.style.opacity = "0";

    await animate(320, (t) => {
      const scale = 1 + 0.08 * Math.sin(t * Math.PI);
      const size = 132 * (1 - t * 0.3);
      orbActor.style.width = size + "px";
      orbActor.style.height = size + "px";
      orbActor.style.transform = `scale(${scale})`;
      stageHalo.style.opacity = (0.6 * (1 - t)).toFixed(3);
    }, easeIO);

    mainOrb.stop();
    orbCanvas.style.opacity = "0";
    const surfEl = orbActor.querySelector(".orb-surface");
    if (surfEl) surfEl.style.opacity = "0";
    const glowEl = orbActor.querySelector(".orb-glow");
    if (glowEl) glowEl.style.opacity = "0";
    orbActor.style.width = "0px";
    orbActor.style.height = "0px";
    orbActor.style.transform = "";

    const normalCenter  = getCenterOf(chatNormal);
    const steeredCenter = getCenterOf(chatSteered);
    const stageRect     = colLeft.getBoundingClientRect();
    const originX       = stageRect.width / 2;
    const originY       = stageRect.height / 2;

    splitL.style.left = originX + "px";
    splitL.style.top  = originY + "px";
    splitR.style.left = originX + "px";
    splitR.style.top  = originY + "px";

    splitL.style.opacity = "1";
    splitR.style.opacity = "1";
    leftOrb.start();
    rightOrb.start();

    await animate(300, (t) => {
      const s = spring(t);
      splitL.style.transform = `translate(-50%, -50%) scale(${s})`;
      splitR.style.transform = `translate(-50%, -50%) scale(${s})`;
    }, easeOut);

    await sleep(120);

    const flyDur = 650;
    panelNormal.classList.add("panel--active-red");
    panelSteered.classList.add("panel--active-green");

    await animate(flyDur, (t) => {
      const lx = originX + (normalCenter.x - originX) * t + Math.sin(t * Math.PI) * -50;
      const ly = originY + (normalCenter.y - originY) * t + Math.sin(t * Math.PI) * -20;
      const ls = 1 - t * 0.3;
      splitL.style.left = lx + "px";
      splitL.style.top  = ly + "px";
      splitL.style.transform = `translate(-50%, -50%) scale(${ls})`;
      splitL.style.opacity = (1 - t * 0.3).toFixed(3);

      const rx = originX + (steeredCenter.x - originX) * t + Math.sin(t * Math.PI) * 50;
      const ry = originY + (steeredCenter.y - originY) * t + Math.sin(t * Math.PI) * 20;
      const rs = 1 - t * 0.3;
      splitR.style.left = rx + "px";
      splitR.style.top  = ry + "px";
      splitR.style.transform = `translate(-50%, -50%) scale(${rs})`;
      splitR.style.opacity = (1 - t * 0.3).toFixed(3);

      stageHalo.style.opacity = (0.3 * (1 - t)).toFixed(3);
    }, easeIO);

    leftOrb.stop();
    rightOrb.stop();
    splitL.style.opacity = "0";
    splitL.style.transform = "scale(0)";
    splitR.style.opacity = "0";
    splitR.style.transform = "scale(0)";

    // Reveal responses word-by-word
    const unsteeredData = (genData && genData.unsteered && (genData.unsteered.tokens || genData.unsteered.text)) || (genData && genData.unsteered) || "";
    const steeredData   = (genData && genData.steered   && (genData.steered.tokens   || genData.steered.text))   || (genData && genData.steered)   || "";

    revealResponse(chatNormal, unsteeredData, 30);
    revealResponse(chatSteered, steeredData, 30);

    if (responseLegend) {
      responseLegend.classList.add("visible");
    }

  // Update PCA
  if (projData.pca) {
    pcaBackground = projData.pca;
    pcaCache.set(currentLayer, projData.pca);
  }
  if (projData.prompt_point) {
    currentBasePoint = projData.prompt_point;
  }
  if (projData.prompt_point && pcaBackground) {
    plotPCAWithPoint(pcaBackground, projData.prompt_point);
  }

    const unsteeredLen = parseTokensWithHeatmap(unsteeredData).length;
    const steeredLen   = parseTokensWithHeatmap(steeredData).length;
    const wordCount = Math.max(unsteeredLen, steeredLen, 10);
    await sleep(wordCount * 30 + 350);

  } catch (err) {
    console.error("Critical error in runAnimation:", err);
    appendMsg(chatNormal, "bot", "Unexpected error. Please try again.");
    appendMsg(chatSteered, "bot", "Unexpected error. Please try again.");
  } finally {
    clearInterval(labelInterval);
    pulseAlive = false;
    mainOrb.stop();
    leftOrb.stop();
    rightOrb.stop();
    orbStage.classList.remove("active");
    panelNormal.classList.remove("panel--active-red");
    panelSteered.classList.remove("panel--active-green");
    promptBar.classList.remove("prompt-bar--hidden");
    promptInput.value = "";
    promptInput.focus();
    isBusy = false;
  }
}

/**
 * Flexible parser supporting all backend token formats:
 * - [{ token: "word", logit_delta: 0.8 }, ...] or { token, value }
 * - Array of comma strings: ["cannot,0.85", "assist,-0.12", ...]
 * - Tuples: [["cannot", 0.85], ["assist", -0.12]]
 * - Comma-delimited text string: "cannot,0.85 assist,-0.12"
 * - Fallback plain text string: "cannot assist with that"
 */
function parseTokensWithHeatmap(tokensInput, fallbackText) {
  if (Array.isArray(tokensInput) && tokensInput.length > 0) {
    if (typeof tokensInput[0] === "object" && tokensInput[0] !== null && "token" in tokensInput[0] && "value" in tokensInput[0]) {
      return tokensInput;
    }
    return tokensInput.map((item) => {
      if (typeof item === "object" && item !== null) {
        if (Array.isArray(item)) {
          return { token: String(item[0]), value: parseFloat(item[1]) || 0 };
        }
        const tok = item.token ?? item.word ?? item.text ?? "";
        const val = item.logit_delta ?? item.value ?? item.score ?? item.delta ?? 0;
        return { token: String(tok), value: parseFloat(val) || 0 };
      }
      if (typeof item === "string") {
        const lastComma = item.lastIndexOf(",");
        if (lastComma !== -1) {
          const tok = item.slice(0, lastComma).trim();
          const num = parseFloat(item.slice(lastComma + 1).trim());
          if (!isNaN(num)) return { token: tok, value: num };
        }
        return { token: item, value: 0 };
      }
      return { token: String(item), value: 0 };
    });
  }

  if (typeof tokensInput === "string" && tokensInput.trim().length > 0) {
    const raw = tokensInput.trim();
    // Try space-separated comma tokens e.g. "cannot,0.85 assist,-0.12"
    const spaceParts = raw.split(/\s+/);
    const parsed = [];
    let isCommaFormat = true;
    for (let part of spaceParts) {
      const idx = part.lastIndexOf(",");
      if (idx !== -1) {
        const tok = part.slice(0, idx);
        const val = parseFloat(part.slice(idx + 1));
        if (!isNaN(val)) {
          parsed.push({ token: tok, value: val });
          continue;
        }
      }
      isCommaFormat = false;
      break;
    }
    if (isCommaFormat && parsed.length > 0) return parsed;
  }

  const text = (typeof fallbackText === "string" && fallbackText)
    ? fallbackText
    : (typeof tokensInput === "string" ? tokensInput : "");
  return text.split(/\s+/).filter(Boolean).map(w => ({ token: w, value: 0 }));
}

/**
 * Reveal response with token activation color coding.
 */
function revealResponse(container, tokensOrText, staggerMs = 35) {
  const ph = container.querySelector(".chat-placeholder");
  if (ph) ph.remove();

  const tokens = parseTokensWithHeatmap(tokensOrText, typeof tokensOrText === "string" ? tokensOrText : "");
  const div = document.createElement("div");
  div.className = "msg msg--bot";

  const fragment = document.createDocumentFragment();
  const spans = tokens.map((item) => {
    const span = document.createElement("span");
    span.className = "word";
    span.textContent = item.token + " ";

    const colors = deltaToColor(item.value);
    if (colors && colors.bg && colors.bg !== "transparent") {
      span.style.setProperty("--token-bg", colors.bg);
      span.style.setProperty("--token-text", colors.textColor || "inherit");
      span.classList.add("word--heatmap");
      const sign = item.value > 0 ? "+" : "";
      const label = item.value > 0 ? "Amplified" : "Suppressed";
      span.title = `[${label}] delta ${sign}${item.value.toFixed(3)}`;
    }
    fragment.appendChild(span);
    return span;
  });

  div.appendChild(fragment);
  container.appendChild(div);

  // RAF-throttled scroll to avoid repeated layout recalculations during word reveal
  let scrollScheduled = false;
  function scheduleScroll() {
    if (!scrollScheduled) {
      scrollScheduled = true;
      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
        scrollScheduled = false;
      });
    }
  }
  scheduleScroll();

  // Stagger reveal
  spans.forEach((span, i) => {
    setTimeout(() => {
      span.classList.add("visible");
      scheduleScroll();
    }, i * staggerMs);
  });
}

// Utilities

/**
 * Map token delta to highlight background and text colors.
 */
function deltaToColor(delta) {
  const raw = Number(delta) || 0;
  const c = Math.max(-1, Math.min(1, raw));
  const mag = Math.abs(c);
  const isLight = document.documentElement.getAttribute("data-theme") === "light";

  if (mag <= 0.04) {
    return {
      bg: "transparent",
      textColor: "inherit",
      toString() { return "transparent"; }
    };
  }

  // Non-linear power curve for perceptually balanced highlighter wash
  const t = Math.pow(mag, 0.65);

  if (c < 0) {
    // Suppressed: Rose/Coral highlighter wash
    if (isLight) {
      const alpha = (0.12 + 0.22 * t).toFixed(2);
      return {
        bg: `rgba(244, 63, 94, ${alpha})`,
        textColor: t > 0.35 ? "#9f1239" : "#be123c",
        toString() { return this.bg; }
      };
    }
    const r = Math.round(251 + (244 - 251) * t);
    const g = Math.round(146 + (63 - 146) * t);
    const b = Math.round(60 + (94 - 60) * t);
    const alpha = (0.14 + 0.30 * t).toFixed(2);

    return {
      bg: `rgba(${r}, ${g}, ${b}, ${alpha})`,
      textColor: t > 0.35 ? "#fca5a5" : "#fecdd3",
      toString() { return this.bg; }
    };
  }

  // Amplified: Emerald/Teal highlighter wash
  if (isLight) {
    const alpha = (0.12 + 0.22 * t).toFixed(2);
    return {
      bg: `rgba(16, 185, 129, ${alpha})`,
      textColor: t > 0.35 ? "#065f46" : "#047857",
      toString() { return this.bg; }
    };
  }
  const r = Math.round(6 + (16 - 6) * t);
  const g = Math.round(182 + (185 - 182) * t);
  const b = Math.round(212 + (129 - 212) * t);
  const alpha = (0.14 + 0.30 * t).toFixed(2);

  return {
    bg: `rgba(${r}, ${g}, ${b}, ${alpha})`,
    textColor: t > 0.35 ? "#6ee7b7" : "#a7f3d0",
    toString() { return this.bg; }
  };
}

function appendMsg(container, role, text) {
  const ph = container.querySelector(".chat-placeholder");
  if (ph) ph.remove();
  const div = document.createElement("div");
  div.className = `msg msg--${role}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function clearChat(container) {
  container.innerHTML = "";
}

// Slider controls

/**
 * Rolling digit ticker.
 * Digits 0-9 scroll vertically in reels; signs & decimals stay centered.
 */
function updateNumberFlow(container, text) {
  const chars = String(text).split("");
  const currentNodes = Array.from(container.children);

  const needsRebuild = currentNodes.length !== chars.length ||
    chars.some((ch, i) => {
      const isDigit = /\d/.test(ch);
      const isNodeDigit = currentNodes[i]?.classList.contains("nf-digit");
      return isDigit !== isNodeDigit;
    });

  if (needsRebuild) {
    container.innerHTML = "";
    chars.forEach((ch) => {
      if (/\d/.test(ch)) {
        const slot = document.createElement("span");
        slot.className = "nf-digit";
        const reel = document.createElement("span");
        reel.className = "nf-reel";
        for (let d = 0; d <= 9; d++) {
          const i = document.createElement("i");
          i.textContent = d;
          reel.appendChild(i);
        }
        slot.appendChild(reel);
        container.appendChild(slot);
      } else {
        const span = document.createElement("span");
        span.className = "nf-char";
        span.textContent = ch;
        container.appendChild(span);
      }
    });
  }

  // Update reel positions
  Array.from(container.children).forEach((node, i) => {
    const ch = chars[i];
    if (/\d/.test(ch)) {
      const reel = node.querySelector(".nf-reel");
      const digit = parseInt(ch, 10);
      reel.style.transform = `translateY(-${digit * 10}%)`;
    } else {
      node.textContent = ch;
    }
  });
}

/**
 * Connect a native range input to the custom track, range fill,
 * floating thumb, and NumberFlow bubble.
 */
function setupFlowSlider(opts) {
  const { input, wrap, rangeEl, thumbWrap, displayEl, formatFn, bipolar, onChange } = opts;

  function update() {
    const val = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));

    // Position thumb + bubble
    thumbWrap.style.left = `${pct}%`;

    // Fill track
    if (bipolar) {
      if (val >= 0) {
        rangeEl.style.left = "50%";
        rangeEl.style.width = `${(val / max) * 50}%`;
      } else {
        const width = (Math.abs(val) / Math.abs(min)) * 50;
        rangeEl.style.left = `${50 - width}%`;
        rangeEl.style.width = `${width}%`;
      }
    } else {
      rangeEl.style.left = "0%";
      rangeEl.style.width = `${pct}%`;
    }

    // Update animated number
    const formatted = formatFn ? formatFn(val) : String(val);
    updateNumberFlow(displayEl, formatted);

    if (onChange) onChange(val);
  }

  input.addEventListener("input", update);
  input.addEventListener("pointerdown", () => wrap.classList.add("flow-slider--active"));
  window.addEventListener("pointerup", () => wrap.classList.remove("flow-slider--active"));
  input.addEventListener("focus", () => wrap.classList.add("flow-slider--active"));
  input.addEventListener("blur", () => wrap.classList.remove("flow-slider--active"));

  update();
}

// Form & slider wiring

function updatePCAWithSliders() {
  if (!currentBasePoint || !pcaBackground) return;
  const valR = parseFloat(sliderRefusal.value) || 0;
  const valS = parseFloat(sliderSycophancy.value) || 0;
  const layer = parseInt(sliderLayer.value, 10) || 12;
  const layerFactor = Math.max(0.3, Math.min(1.2, 1.0 - Math.abs(layer - 14) / 24.0));
  const newPoint = {
    x: Number((currentBasePoint.x - (valR * 0.75 - valS * 0.55) * layerFactor).toFixed(4)),
    y: Number((currentBasePoint.y + (valR * 0.45 - valS * 0.40) * layerFactor).toFixed(4)),
  };
  plotPCAWithPoint(pcaBackground, newPoint);
}

// Slider bindings
setupFlowSlider({
  input: sliderRefusal,
  wrap: $("#sliderWrapRefusal"),
  rangeEl: $("#rangeRefusal"),
  thumbWrap: $("#thumbRefusal"),
  displayEl: refusalValue,
  bipolar: true,
  formatFn: (v) => (v > 0 ? "+" : "") + v.toFixed(1),
  onChange: () => updatePCAWithSliders(),
});

setupFlowSlider({
  input: sliderSycophancy,
  wrap: $("#sliderWrapSycophancy"),
  rangeEl: $("#rangeSycophancy"),
  thumbWrap: $("#thumbSycophancy"),
  displayEl: sycophancyValue,
  bipolar: true,
  formatFn: (v) => (v > 0 ? "+" : "") + v.toFixed(1),
  onChange: () => updatePCAWithSliders(),
});

setupFlowSlider({
  input: sliderLayer,
  wrap: $("#sliderWrapLayer"),
  rangeEl: $("#rangeLayer"),
  thumbWrap: $("#thumbLayer"),
  displayEl: layerValue,
  bipolar: false,
  formatFn: (v) => String(Math.round(v)),
  onChange: (layer) => {
    layer = Math.round(layer);
    if (layer === currentLayer) return;
    currentLayer = layer;

    // Return cached layer data if available
    if (pcaCache.has(layer) && umapCache.has(layer)) {
      clearTimeout(layerDebounceTimer);
      loadPCA(layer);
      loadUMAP(layer);
      loadRefusalPlots(layer);
      return;
    }

    // Debounce network requests by 75ms during scrubbing to prevent API flood
    clearTimeout(layerDebounceTimer);
    layerDebounceTimer = setTimeout(() => {
      loadPCA(layer);
      loadUMAP(layer);
      loadRefusalPlots(layer);
    }, 75);
  },
});

// Send button data-ready attribute
promptInput.addEventListener("input", () => {
  if (promptInput.value.trim()) sendBtn.setAttribute("data-ready", "");
  else sendBtn.removeAttribute("data-ready");
});

// Form submit
promptForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) {
    promptPill.classList.add("shake");
    setTimeout(() => promptPill.classList.remove("shake"), 350);
    return;
  }
  if (isBusy) return;
  runAnimation(text);
});

// PCA & UMAP plots

function getPlotlyTheme() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  return {
    isLight,
    bg: "rgba(0,0,0,0)",
    grid: isLight ? "rgba(15, 23, 42, 0.08)" : "rgba(255, 255, 255, 0.06)",
    text: isLight ? "#334155" : "#8a94a6",
    sycophantic: isLight ? "#7c3aed" : "#a855f7",
    nonSycophantic: isLight ? "#0284c7" : "#06b6d4",
    diamondBorder: isLight ? "#0f172a" : "#ffffff",
  };
}

const PLOTLY_CFG = { displayModeBar: false, responsive: true, scrollZoom: false };

// In-memory layer caches
const pcaCache = new Map();
const umapCache = new Map();
let pcaAbortController = null;
let umapAbortController = null;
let layerDebounceTimer = null;

// PCA
async function loadPCA(layer) {
  if (pcaCache.has(layer)) {
    const data = pcaCache.get(layer);
    pcaBackground = data;
    plotPCAWithPoint(data, currentPoint);
    return;
  }

  if (pcaAbortController) {
    pcaAbortController.abort();
  }
  pcaAbortController = new AbortController();

  try {
    const res = await fetch(`/api/pca?layer=${layer}`, { signal: pcaAbortController.signal });
    if (!res.ok) return;
    const data = await res.json();
    pcaCache.set(layer, data);
    if (layer === currentLayer) {
      pcaBackground = data;
      plotPCAWithPoint(data, currentPoint);
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      console.warn("PCA load failed:", err);
    }
  }
}

function plotPCAWithPoint(
  data,
  point,
  targetId = "plotPCA",
  positiveName = "Sycophantic",
  negativeName = "Non-Sycophantic",
) {
  if (targetId === "plotPCA") currentPoint = point;
  const pt = getPlotlyTheme();
  const traces = [
    {
      x: data.sycophantic.x, y: data.sycophantic.y,
      mode: "markers", type: "scatter", name: positiveName,
      marker: { color: pt.sycophantic, size: 6, opacity: pt.isLight ? 0.85 : 0.75 },
    },
    {
      x: data.non_sycophantic.x, y: data.non_sycophantic.y,
      mode: "markers", type: "scatter", name: negativeName,
      marker: { color: pt.nonSycophantic, size: 6, opacity: pt.isLight ? 0.85 : 0.75 },
    },
  ];
  if (point) {
    traces.push({
      x: [point.x], y: [point.y],
      mode: "markers", type: "scatter", name: "Your Prompt",
      marker: { color: "#fde047", size: 14, symbol: "diamond", line: { width: 2, color: pt.diamondBorder } },
    });
  }
  Plotly.react(targetId, traces, {
    paper_bgcolor: pt.bg, plot_bgcolor: pt.bg, font: { family: "Inter, sans-serif", size: 11, color: pt.text },
    margin: { t: 10, r: 10, b: 36, l: 36 },
    xaxis: { title: "PC 1", gridcolor: pt.grid, zerolinecolor: pt.grid },
    yaxis: { title: "PC 2", gridcolor: pt.grid, zerolinecolor: pt.grid },
    legend: { x: 0, y: 1, bgcolor: "rgba(0,0,0,0)", font: { size: 10, color: pt.text } },
    showlegend: true,
  }, PLOTLY_CFG);
}

function plotRefusalPCA(data) {
  plotPCAWithPoint(data, null, "plotRefusalPCA", "Refusal", "Non-Refusal");
}

// UMAP
async function loadUMAP(layer) {
  if (umapCache.has(layer)) {
    const data = umapCache.get(layer);
    plotUMAP(data);
    startUMAPRotation();
    return;
  }

  if (umapAbortController) {
    umapAbortController.abort();
  }
  umapAbortController = new AbortController();

  try {
    const res = await fetch(`/api/umap?layer=${layer}`, { signal: umapAbortController.signal });
    if (!res.ok) return;
    const data = await res.json();
    umapCache.set(layer, data);
    if (layer === currentLayer) {
      plotUMAP(data);
      startUMAPRotation();
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      console.warn("UMAP load failed:", err);
    }
  }
}

function plotUMAP(
  data,
  targetId = "plotUMAP",
  positiveName = "Sycophantic",
  negativeName = "Non-Sycophantic",
) {
  if (targetId === "plotUMAP") lastUmapData = data;
  const pt = getPlotlyTheme();
  Plotly.react(targetId, [
    {
      x: data.sycophantic.x, y: data.sycophantic.y, z: data.sycophantic.z,
      mode: "markers", type: "scatter3d", name: positiveName,
      marker: { color: pt.sycophantic, size: 3.5, opacity: pt.isLight ? 0.85 : 0.8 },
    },
    {
      x: data.non_sycophantic.x, y: data.non_sycophantic.y, z: data.non_sycophantic.z,
      mode: "markers", type: "scatter3d", name: negativeName,
      marker: { color: pt.nonSycophantic, size: 3.5, opacity: pt.isLight ? 0.85 : 0.8 },
    },
  ], {
    paper_bgcolor: pt.bg, font: { family: "Inter, sans-serif", size: 11, color: pt.text },
    margin: { t: 10, r: 0, b: 0, l: 0 },
    scene: {
      bgcolor: pt.bg,
      xaxis: { gridcolor: pt.grid, zerolinecolor: pt.grid, title: "", showticklabels: false, showbackground: false },
      yaxis: { gridcolor: pt.grid, zerolinecolor: pt.grid, title: "", showticklabels: false, showbackground: false },
      zaxis: { gridcolor: pt.grid, zerolinecolor: pt.grid, title: "", showticklabels: false, showbackground: false },
      camera: { eye: { x: 1.6, y: 1.6, z: 1.0 } },
    },
    legend: { x: 0, y: 1, bgcolor: "rgba(0,0,0,0)", font: { size: 10, color: pt.text } },
    showlegend: true,
  }, PLOTLY_CFG);
}

function plotRefusalUMAP(data) {
  plotUMAP(data, "plotRefusalUMAP", "Refusal", "Non-Refusal");
}

async function loadRefusalPlots(layer) {
  const [pcaResponse, umapResponse] = await Promise.all([
    fetch(`/api/pca?layer=${layer}&dataset=refusal`),
    fetch(`/api/umap?layer=${layer}&dataset=refusal`),
  ]);
  if (!pcaResponse.ok || !umapResponse.ok) return;
  const [pca, umap] = await Promise.all([pcaResponse.json(), umapResponse.json()]);
  plotRefusalPCA(pca);
  plotRefusalUMAP(umap);
}

let umapAngle = 0;
let isUmapInteracting = false;
let lastUmapFrameTime = 0;
let umapContainerBound = false;

function setupUMAPInteractions() {
  if (umapContainerBound) return;
  const container = document.getElementById("plotUMAP");
  if (!container) return;
  umapContainerBound = true;

  let resumeTimeout = null;
  container.addEventListener("pointerenter", () => {
    isUmapInteracting = true;
    if (resumeTimeout) clearTimeout(resumeTimeout);
  });
  container.addEventListener("pointerdown", () => {
    isUmapInteracting = true;
    if (resumeTimeout) clearTimeout(resumeTimeout);
  });
  container.addEventListener("pointerleave", () => {
    if (resumeTimeout) clearTimeout(resumeTimeout);
    resumeTimeout = setTimeout(() => {
      isUmapInteracting = false;
    }, 1200);
  });
  window.addEventListener("pointerup", () => {
    if (isUmapInteracting) {
      if (resumeTimeout) clearTimeout(resumeTimeout);
      resumeTimeout = setTimeout(() => {
        isUmapInteracting = false;
      }, 1200);
    }
  });
}

function startUMAPRotation() {
  setupUMAPInteractions();
  if (umapRotationRAF) cancelAnimationFrame(umapRotationRAF);

  const radius = 2.2;
  const speed = 0.003;
  const frameInterval = 1000 / 30; // 30 FPS throttle

  function tick(timestamp) {
    if (document.visibilityState === "visible" && !isUmapInteracting) {
      if (!lastUmapFrameTime || timestamp - lastUmapFrameTime >= frameInterval) {
        lastUmapFrameTime = timestamp;
        umapAngle += speed;
        try {
          Plotly.relayout("plotUMAP", {
            "scene.camera.eye": {
              x: radius * Math.cos(umapAngle),
              y: radius * Math.sin(umapAngle),
              z: 0.9,
            },
          });
        } catch (_) {}
      }
    }
    umapRotationRAF = requestAnimationFrame(tick);
  }
  umapRotationRAF = requestAnimationFrame(tick);
}

// Halftone shader background

function initHalftoneShader() {
  const canvas = document.getElementById("shaderCanvas");
  if (!canvas) return;

  const gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" }) ||
             canvas.getContext("webgl", { antialias: false, powerPreference: "high-performance" });
  if (!gl) {
    console.warn("WebGL not supported for halftone background");
    return;
  }

  const VERT = `attribute vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

  const FRAG = `#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform vec3 u_colors[8];
uniform vec4 u_scene;      // resolution.xy, time, colour count
uniform vec4 u_shape;      // scale, intensity, paramA, warp
uniform vec4 u_surface;    // detail, contrast, brightness, saturation
uniform vec4 u_finish;     // hue, vignette, blur, grain
uniform vec4 u_transform;  // seed, rotation, drift, OKLab toggle
uniform vec4 u_space;      // offset.xy, pointer.xy
uniform vec4 u_cursor;

#define u_resolution u_scene.xy
#define u_time u_scene.z
#define u_colorCount u_scene.w
#define u_scale u_shape.x
#define u_intensity u_shape.y
#define u_paramA u_shape.z
#define u_warp u_shape.w
#define u_detail u_surface.x
#define u_contrast u_surface.y
#define u_brightness u_surface.z
#define u_saturation u_surface.w
#define u_hue u_finish.x
#define u_vignette u_finish.y
#define u_blur u_finish.z
#define u_grain u_finish.w
#ifdef GL_FRAGMENT_PRECISION_HIGH
#define u_seed u_transform.x
#else
#define u_seed mod(u_transform.x, 31.0)
#endif
#define u_rotate u_transform.y
#define u_drift u_transform.z
#define u_oklab u_transform.w
#define u_offset u_space.xy
#define u_mouse u_space.zw
#define u_cursorPresence u_cursor.x
#define u_cursorEffect u_cursor.y
#define u_cursorStrength u_cursor.z
#define u_cursorRadius u_cursor.w

float hash21(vec2 p) {
#ifndef GL_FRAGMENT_PRECISION_HIGH
  p = mod(p, 31.0);
#endif
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}

float grainHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
#ifndef GL_FRAGMENT_PRECISION_HIGH
  p = mod(p, 31.0);
#endif
  float n = sin(dot(p, vec2(41.0, 289.0)));
  return fract(vec2(15731.743, 7892.321) * n);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
    mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
    u.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p = p * 2.03 + vec2(17.0, 9.2);
    a *= 0.5;
  }
  return v;
}

vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)),
    step(0.04045, c));
}
vec3 linearToSrgb(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
    step(0.0031308, c));
}
vec3 linToOklab(vec3 c) {
  float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  l = pow(max(l, 0.0), 1.0 / 3.0);
  m = pow(max(m, 0.0), 1.0 / 3.0);
  s = pow(max(s, 0.0), 1.0 / 3.0);
  return vec3(
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}
vec3 oklabToLin(vec3 c) {
  float l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  float m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  float s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  l = l * l * l; m = m * m * m; s = s * s * s;
  return vec3(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}
vec3 mixColour(vec3 a, vec3 b, float t) {
  if (u_oklab > 0.5) {
    vec3 la = linToOklab(srgbToLinear(a));
    vec3 lb = linToOklab(srgbToLinear(b));
    return clamp(linearToSrgb(oklabToLin(mix(la, lb, t))), 0.0, 1.0);
  }
  return mix(a, b, t);
}

vec3 getColor(int idx) {
  if (idx <= 0) return u_colors[0];
  if (idx == 1) return u_colors[1];
  if (idx == 2) return u_colors[2];
  if (idx == 3) return u_colors[3];
  if (idx == 4) return u_colors[4];
  if (idx == 5) return u_colors[5];
  if (idx == 6) return u_colors[6];
  return u_colors[7];
}

vec3 palette(float x) {
  float n = max(u_colorCount - 1.0, 1.0);
  float f = clamp(x, 0.0, 1.0) * n;
  vec3 col = u_colors[0];
  for (int i = 0; i < 7; i++) {
    if (float(i) < n) {
      vec3 nextC = getColor(i + 1);
      col = mixColour(col, nextC, smoothstep(0.0, 1.0, clamp(f - float(i), 0.0, 1.0)));
    }
  }
  return col;
}

vec3 hueRotate(vec3 col, float a) {
  const mat3 toYIQ = mat3(0.299, 0.596, 0.211,
                          0.587, -0.274, -0.523,
                          0.114, -0.322, 0.312);
  const mat3 toRGB = mat3(1.0, 1.0, 1.0,
                          0.956, -0.272, -1.106,
                          0.621, -0.647, 1.703);
  vec3 yiq = toYIQ * col;
  float ca = cos(a), sa = sin(a);
  yiq = vec3(yiq.x, yiq.y * ca - yiq.z * sa, yiq.y * sa + yiq.z * ca);
  return toRGB * yiq;
}

vec3 shade(vec2 uv, vec2 p, float t) {
  float cells = 18.0 + u_intensity * 30.0;
  vec2 f = fract(p * cells) - 0.5;
  float field = 0.5 + 0.5 * sin(p.x * 3.0 + t + u_seed) * sin(p.y * 2.4 - t * 0.7);
  float r = (0.06 + u_paramA * 0.34) + field * 0.2;
  float dotMask = 1.0 - smoothstep(r - 0.08, r, length(f));
  return mix(u_colors[0], palette(field), dotMask);
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution.xy;
  vec2 screenUv = uv;
  vec2 p = (gl_FragCoord.xy - 0.5 * u_resolution.xy)
    / min(u_resolution.x, u_resolution.y);
  float cursorMask = 0.0;

  if (u_cursorPresence > 0.001) {
    vec2 cursor = (0.5 * u_mouse * u_resolution.xy)
      / min(u_resolution.x, u_resolution.y);
    vec2 cursorDelta = p - cursor;
    if (u_cursorEffect < 0.5) {
      p += cursor * u_cursorPresence * u_cursorStrength * 0.55;
    } else {
      float cursorDistance = length(cursorDelta);
      vec2 cursorDirection = cursorDelta / max(cursorDistance, 0.0001);
      cursorMask = u_cursorPresence
        * (1.0 - smoothstep(0.0, u_cursorRadius, cursorDistance));
      if (u_cursorEffect < 1.5) {
        p -= cursorDirection * cursorMask * u_cursorStrength * 0.24;
      } else if (u_cursorEffect < 2.5) {
        float cursorAngle = cursorMask * u_cursorStrength * 2.2;
        float cc = cos(cursorAngle), cs = sin(cursorAngle);
        p = cursor + mat2(cc, -cs, cs, cc) * cursorDelta;
      } else if (u_cursorEffect < 3.5) {
        float ripple = sin(
          cursorDistance / max(u_cursorRadius, 0.001) * 18.0 - u_time * 5.0);
        p -= cursorDirection * ripple * cursorMask * u_cursorStrength * 0.07;
      }
    }
  }

  uv = p * min(u_resolution.x, u_resolution.y) / u_resolution.xy + 0.5;
  p *= u_scale;
  if (abs(u_rotate) > 0.0001) {
    float cr = cos(u_rotate), sr = sin(u_rotate);
    p = mat2(cr, -sr, sr, cr) * p;
  }
  p += u_offset;
  if (u_drift > 0.0001)
    p += u_drift * vec2(sin(u_time * 0.31), cos(u_time * 0.23));
  if (u_warp > 0.0) {
    p += u_warp * (vec2(
      fbm(p * u_detail + u_seed),
      fbm(p * u_detail + vec2(5.2, 1.3))) - 0.5);
  }
  vec3 col;
  if (u_blur > 0.0) {
    float e = u_blur;
    float pe = e * u_scale;
    vec2 uvE = vec2(e) * min(u_resolution.x, u_resolution.y) / u_resolution.xy;
    col  = shade(uv, p, u_time) * 0.36;
    col += shade(uv + vec2(uvE.x, 0.0), p + vec2(pe, 0.0), u_time) * 0.16;
    col += shade(uv - vec2(uvE.x, 0.0), p - vec2(pe, 0.0), u_time) * 0.16;
    col += shade(uv + vec2(0.0, uvE.y), p + vec2(0.0, pe), u_time) * 0.16;
    col += shade(uv - vec2(0.0, uvE.y), p - vec2(0.0, pe), u_time) * 0.16;
  } else {
    col = shade(uv, p, u_time);
  }
  if (abs(u_contrast - 1.0) > 0.0001)
    col = (col - 0.5) * u_contrast + 0.5;
  if (abs(u_saturation - 1.0) > 0.0001) {
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(luma), col, u_saturation);
  }
  if (abs(u_hue) > 0.0001)
    col = hueRotate(col, u_hue);
  if (abs(u_brightness) > 0.0001)
    col += u_brightness;
  if (u_vignette > 0.0001) {
    float vd = length(screenUv - 0.5) * 1.41421356;
    col *= 1.0 - u_vignette * smoothstep(0.35, 1.0, vd);
  }
  if (u_cursorPresence > 0.001 && u_cursorEffect > 3.5)
    col += (vec3(0.18) + col * 0.12) * cursorMask * u_cursorStrength;
  if (u_grain > 0.0001)
    col += (grainHash(
      gl_FragCoord.xy + vec2(u_seed * 17.0, u_seed * 31.0)) - 0.5) * u_grain;
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

  const UNIFORMS = {
    colors: [
      [0.23137, 0.5098, 0.9647],
      [0.1451, 0.3882, 0.9216],
      [0.1137, 0.3059, 0.8471],
      [0.1176, 0.2510, 0.6863],
      [0.1176, 0.2275, 0.5412],
      [0.1176, 0.2275, 0.5412],
      [0.1176, 0.2275, 0.5412],
      [0.1176, 0.2275, 0.5412],
    ],
    colorCount: 5,
    scale: 1.78,
    intensity: 0.19,
    paramA: 1.0,
    warp: 0.0,
    detail: 4.0,
    contrast: 0.6,
    brightness: -0.4,
    saturation: 0.7,
    hue: 0.0,
    vignette: 1.0,
    blur: 0.04,
    grain: 0.158,
    seed: 4990.0,
    rotate: 2.9845,
    offsetX: -0.25,
    offsetY: -0.02,
    drift: 0.136,
    cursorEnabled: true,
    cursorEffect: 2.0,
    cursorStrength: 0.15,
    cursorRadius: 0.569,
    oklab: 1.0,
    timeScale: -1.525,
  };

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("Shader error:", gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  };

  const vertexShader = compile(gl.VERTEX_SHADER, VERT);
  const fragmentShader = compile(gl.FRAGMENT_SHADER, FRAG);
  if (!vertexShader || !fragmentShader) return;

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Link error:", gl.getProgramInfoLog(program));
    return;
  }

  gl.useProgram(program);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const loc = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const uni = {
    colors: gl.getUniformLocation(program, "u_colors[0]") || gl.getUniformLocation(program, "u_colors"),
    scene: gl.getUniformLocation(program, "u_scene"),
    shape: gl.getUniformLocation(program, "u_shape"),
    surface: gl.getUniformLocation(program, "u_surface"),
    finish: gl.getUniformLocation(program, "u_finish"),
    transform: gl.getUniformLocation(program, "u_transform"),
    space: gl.getUniformLocation(program, "u_space"),
    cursor: gl.getUniformLocation(program, "u_cursor"),
  };

  const flatColors = [];
  UNIFORMS.colors.forEach((c) => flatColors.push(c[0], c[1], c[2]));
  gl.uniform3fv(uni.colors, new Float32Array(flatColors));
  gl.uniform4f(uni.shape, UNIFORMS.scale, UNIFORMS.intensity, UNIFORMS.paramA, UNIFORMS.warp);
  gl.uniform4f(uni.surface, UNIFORMS.detail, UNIFORMS.contrast, UNIFORMS.brightness, UNIFORMS.saturation);
  gl.uniform4f(uni.finish, UNIFORMS.hue, UNIFORMS.vignette, UNIFORMS.blur, UNIFORMS.grain);
  gl.uniform4f(uni.transform, UNIFORMS.seed, UNIFORMS.rotate, UNIFORMS.drift, UNIFORMS.oklab);
  gl.uniform4f(uni.cursor, 0, UNIFORMS.cursorEffect, UNIFORMS.cursorStrength, UNIFORMS.cursorRadius);

  let targetX = 0;
  let targetY = 0;
  let targetPresence = 0;
  let mouseX = 0;
  let mouseY = 0;
  let cursorPresence = 0;
  let pointerKnown = false;
  let pointerClientX = 0;
  let pointerClientY = 0;
  let bounds = canvas.getBoundingClientRect();
  let raf = 0;
  let lastNow = null;
  let visible = document.visibilityState === "visible";
  const start = performance.now();

  const resizeCanvas = () => {
    bounds = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rawWidth = Math.max(1, Math.round((bounds.width || window.innerWidth) * dpr));
    const rawHeight = Math.max(1, Math.round((bounds.height || window.innerHeight) * dpr));
    const pixelScale = Math.min(1, Math.sqrt(2000000 / Math.max(1, rawWidth * rawHeight)));
    const width = Math.max(1, Math.round(rawWidth * pixelScale));
    const height = Math.max(1, Math.round(rawHeight * pixelScale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
  };

  const updatePointerTarget = () => {
    if (!pointerKnown) return;
    const w = bounds.width || window.innerWidth;
    const h = bounds.height || window.innerHeight;
    if (w === 0 || h === 0) return;
    const left = bounds.left || 0;
    const top = bounds.top || 0;

    const nextX = ((pointerClientX - left) / w) * 2 - 1;
    const nextY = -(((pointerClientY - top) / h) * 2 - 1);
    if (targetPresence === 0 && cursorPresence < 0.01) {
      mouseX = nextX;
      mouseY = nextY;
    }
    targetX = nextX;
    targetY = nextY;
    targetPresence = 1;
  };

  const onPointerMove = (e) => {
    pointerKnown = true;
    pointerClientX = e.clientX;
    pointerClientY = e.clientY;
    updatePointerTarget();
  };

  const onPointerLeave = () => {
    pointerKnown = false;
    targetPresence = 0;
  };

  window.addEventListener("resize", () => {
    resizeCanvas();
    updatePointerTarget();
  });
  window.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerleave", onPointerLeave);
  document.addEventListener("visibilitychange", () => {
    visible = document.visibilityState === "visible";
    if (visible && raf === 0) {
      lastNow = null;
      raf = requestAnimationFrame(render);
    }
  });

  resizeCanvas();

  function render(now) {
    if (!visible) {
      raf = 0;
      return;
    }
    const dt = lastNow === null ? 0 : Math.min((now - lastNow) / 1000, 0.1);
    lastNow = now;

    const follow = 1 - Math.exp(-12 * dt);
    mouseX += (targetX - mouseX) * follow;
    mouseY += (targetY - mouseY) * follow;
    cursorPresence += (targetPresence - cursorPresence) * follow;

    gl.uniform4f(
      uni.scene,
      canvas.width,
      canvas.height,
      ((now - start) / 1000) * UNIFORMS.timeScale,
      UNIFORMS.colorCount
    );
    gl.uniform4f(uni.space, UNIFORMS.offsetX, UNIFORMS.offsetY, mouseX, mouseY);
    gl.uniform4f(
      uni.cursor,
      cursorPresence,
      UNIFORMS.cursorEffect,
      UNIFORMS.cursorStrength,
      UNIFORMS.cursorRadius
    );

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    raf = requestAnimationFrame(render);
  }

  raf = requestAnimationFrame(render);
}

// Theme toggle

function initThemeToggle() {
  const toggleBtn = document.getElementById("themeToggleBtn");
  if (!toggleBtn) return;

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("llm_lobotomy_theme", theme);
    } catch (e) {}

    // Refresh Plotly graphs with theme colors
    if (pcaBackground) {
      plotPCAWithPoint(pcaBackground, currentPoint);
    }
    if (lastUmapData) {
      plotUMAP(lastUmapData);
    }
  }

  try {
    const saved = localStorage.getItem("llm_lobotomy_theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.setAttribute("data-theme", saved);
    }
  } catch (e) {}

  toggleBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "light" ? "dark" : "light";
    applyTheme(next);
  });
}

// Card pointer illumination
function initCursorCards() {
  const cards = document.querySelectorAll(".cursor-card");
  if (!cards.length) return;

  const extendedProximity = 15;

  const handlePointerMovement = (event) => {
    const { clientX, clientY } = event;

    cards.forEach((card) => {
      const bounds = card.getBoundingClientRect();
      const radius = card.dataset.trait === "prompt" ? 100 : 120;

      const isNearCard =
        clientX >= bounds.left - extendedProximity &&
        clientX <= bounds.right + extendedProximity &&
        clientY >= bounds.top - extendedProximity &&
        clientY <= bounds.bottom + extendedProximity;

      if (isNearCard) {
        const localX = clientX - bounds.left;
        const localY = clientY - bounds.top;
        card.style.setProperty("--mouse-x", `${localX}px`);
        card.style.setProperty("--mouse-y", `${localY}px`);
        card.style.setProperty("--is-active", "1");
      } else {
        card.style.setProperty("--is-active", "0");
        card.style.setProperty("--mouse-x", `${-radius}px`);
        card.style.setProperty("--mouse-y", `${-radius}px`);
      }
    });
  };

  document.addEventListener("pointermove", handlePointerMovement, { passive: true });
}

// Dot matrix title
function initDotMatrixTitle() {
  const container = document.getElementById("dotMatrixTitle");
  const canvas = document.getElementById("dotMatrixCanvas");
  if (!container || !canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const offscreen = document.createElement("canvas");
  const offCtx = offscreen.getContext("2d", { willReadFrequently: true });
  if (!offCtx) return;

  const targetText = "LLM  LOBOTOMY  LAB";
  const fontFamily = "'Orbitron', 'Space Grotesk', system-ui, sans-serif";
  const dotSize = 3.0;
  const gap = 1.4;
  const step = dotSize + gap;
  const scrambleDuration = 520;
  const SCRAMBLE_GLYPHS = "01#X*!&%/+<>~$@";

  let dots = [];
  let cols = 0;
  let rows = 0;
  let offsetX = 0;
  let offsetY = 0;
  let width = 0;
  let height = 0;
  let startTime = performance.now();
  let hasInitiallyLoaded = false;
  let isScrambling = false;
  let scrambleStartTime = 0;
  let lastScrambleTick = 0;
  let animationFrameId = null;

  function createDotMap(str, cCount, rCount) {
    if (cCount <= 0 || rCount <= 0) return [];
    offscreen.width = cCount;
    offscreen.height = rCount;

    offCtx.clearRect(0, 0, cCount, rCount);
    offCtx.textBaseline = "middle";
    offCtx.textAlign = "center";
    if ("letterSpacing" in offCtx) {
      offCtx.letterSpacing = "2px";
    }

    let fontSize = rCount * 0.82;
    offCtx.font = `700 ${fontSize}px ${fontFamily}`;
    let metrics = offCtx.measureText(str);

    if (metrics.width > cCount * 0.94) {
      fontSize = fontSize * ((cCount * 0.94) / (metrics.width || 1));
      offCtx.font = `700 ${fontSize}px ${fontFamily}`;
    }

    offCtx.fillStyle = "#ffffff";
    offCtx.fillText(str, cCount / 2, rCount / 2);

    const imgData = offCtx.getImageData(0, 0, cCount, rCount).data;
    const map = new Array(cCount * rCount);
    for (let i = 0; i < imgData.length; i += 4) {
      map[i / 4] = imgData[i + 3] > 80;
    }
    return map;
  }

  function updateDotTargets(str) {
    if (!cols || !rows || !dots.length) return;
    const map = createDotMap(str, cols, rows);
    for (let i = 0; i < dots.length; i++) {
      const active = !!map[i];
      dots[i].active = active;
      dots[i].targetScale = active ? 1 : 0.28;
    }
  }

  function initGrid() {
    const rect = container.getBoundingClientRect();
    width = Math.floor(rect.width);
    height = Math.floor(rect.height);
    if (width <= 0 || height <= 0) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    cols = Math.floor(width / step);
    rows = Math.floor(height / step);

    offsetX = (width - cols * step) / 2 + dotSize / 2;
    offsetY = (height - rows * step) / 2 + dotSize / 2;

    const dotMap = createDotMap(targetText, cols, rows);

    dots = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const active = !!dotMap[idx];
        const initialScale = hasInitiallyLoaded ? (active ? 1 : 0.28) : 0;
        dots.push({
          c,
          r,
          x: offsetX + c * step,
          y: offsetY + r * step,
          active,
          targetScale: active ? 1 : 0.28,
          currentScale: initialScale,
          delay: (c / cols) * 200 + (r / rows) * 60,
        });
      }
    }
    startTime = performance.now();
    hasInitiallyLoaded = true;
  }

  function getScrambledString(progress) {
    const len = targetText.length;
    const lockedCount = Math.floor(progress * len);
    let res = "";
    for (let i = 0; i < len; i++) {
      const char = targetText[i];
      if (char === " ") {
        res += " ";
      } else if (i < lockedCount) {
        res += char;
      } else {
        res += SCRAMBLE_GLYPHS[Math.floor(Math.random() * SCRAMBLE_GLYPHS.length)];
      }
    }
    return res;
  }

  function triggerScramble() {
    if (isScrambling) return;
    isScrambling = true;
    scrambleStartTime = performance.now();
    lastScrambleTick = 0;
  }

  function render(now) {
    animationFrameId = requestAnimationFrame(render);
    if (!width || !height || !dots.length) return;

    if (isScrambling) {
      const elapsed = now - scrambleStartTime;
      const progress = Math.min(1, elapsed / scrambleDuration);

      if (now - lastScrambleTick > 38) {
        lastScrambleTick = now;
        if (progress >= 1) {
          isScrambling = false;
          updateDotTargets(targetText);
        } else {
          updateDotTargets(getScrambledString(progress));
        }
      }
    }

    const timeSinceStart = now - startTime;
    for (let i = 0; i < dots.length; i++) {
      const d = dots[i];
      if (timeSinceStart >= d.delay) {
        d.currentScale += (d.targetScale - d.currentScale) * 0.22;
      }
    }

    ctx.clearRect(0, 0, width, height);

    const isLight = document.documentElement.getAttribute("data-theme") === "light";

    // Inactive dots
    ctx.fillStyle = isLight ? "rgba(148, 163, 184, 0.025)" : "rgba(255, 255, 255, 0.04)";
    ctx.beginPath();
    const inactiveRadius = (dotSize * 0.5) * 0.65;
    for (let i = 0; i < dots.length; i++) {
      const d = dots[i];
      if (!d.active) {
        ctx.moveTo(d.x + inactiveRadius, d.y);
        ctx.arc(d.x, d.y, inactiveRadius, 0, TAU);
      }
    }
    ctx.fill();

    // Active dots
    const grad = ctx.createLinearGradient(offsetX, 0, offsetX + cols * step, 0);
    if (isLight) {
      grad.addColorStop(0.0, "#0369a1");
      grad.addColorStop(0.35, "#0284c7");
      grad.addColorStop(0.7, "#2563eb");
      grad.addColorStop(1.0, "#7c3aed");
    } else {
      grad.addColorStop(0.0, "#ffffff");
      grad.addColorStop(0.3, "#bae6fd");
      grad.addColorStop(0.7, "#38bdf8");
      grad.addColorStop(1.0, "#c084fc");
    }

    ctx.fillStyle = grad;
    ctx.beginPath();
    const baseRadius = dotSize * 0.5;
    for (let i = 0; i < dots.length; i++) {
      const d = dots[i];
      if (d.active && d.currentScale > 0.05) {
        const r = baseRadius * d.currentScale;
        ctx.moveTo(d.x + r, d.y);
        ctx.arc(d.x, d.y, r, 0, TAU);
      }
    }
    ctx.fill();
  }

  initGrid();
  animationFrameId = requestAnimationFrame(render);

  const ro = new ResizeObserver(() => {
    initGrid();
  });
  ro.observe(container);

  container.addEventListener("mouseenter", triggerScramble);
  container.addEventListener("click", triggerScramble);

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      if (cols > 0 && rows > 0) {
        updateDotTargets(targetText);
      }
    });
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(render);
      }
    } else {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
    }
  });
}

// Initialization

// Pause UMAP rotation when tab is hidden, resume when tab becomes visible
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    if (lastUmapData && !umapRotationRAF) {
      startUMAPRotation();
    }
  } else {
    if (umapRotationRAF) {
      cancelAnimationFrame(umapRotationRAF);
      umapRotationRAF = null;
    }
  }
});

// Resize Plotly charts cleanly on viewport resize
window.addEventListener("resize", () => {
  const pca = document.getElementById("plotPCA");
  const umap = document.getElementById("plotUMAP");
  const refusalPca = document.getElementById("plotRefusalPCA");
  const refusalUmap = document.getElementById("plotRefusalUMAP");
  if (pca && window.Plotly) {
    try { Plotly.Plots.resize(pca); } catch (_) {}
  }
  if (umap && window.Plotly) {
    try { Plotly.Plots.resize(umap); } catch (_) {}
  }
  if (refusalPca && window.Plotly) {
    try { Plotly.Plots.resize(refusalPca); } catch (_) {}
  }
  if (refusalUmap && window.Plotly) {
    try { Plotly.Plots.resize(refusalUmap); } catch (_) {}
  }
});

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initCursorCards();
  initDotMatrixTitle();
  try {
    initHalftoneShader();
  } catch (err) {
    console.warn("Halftone shader init failed:", err);
  }
  loadPCA(currentLayer);
  loadUMAP(currentLayer);
  loadRefusalPlots(currentLayer);
  if (promptInput) promptInput.focus();
});
