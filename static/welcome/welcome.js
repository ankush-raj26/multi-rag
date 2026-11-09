/* ============================================================================
   Routing assembly · landing page behaviour
   ----------------------------------------------------------------------------
   1. Your question  -> POST /api/route -> the REAL router decision (route,
      layer, reason) plus a preview of what retrieval would find.
   2. A three.js drawing of the router: three gate plates (regex, semantic,
      LLM picker) above six outlet bins, drawn as line work.
   3. Scroll drives it: sheet R-1 orbits the assembly, sheet R-3 drops your
      question through the gates. The packet stops at the gate that REALLY
      caught it and rides a bypass rail to its bin.
   4. The decision also stamps the matching detail card (R-4), redlines your
      path through the LangGraph (R-5), and pre-fills the live bot (R-6).

   The scroll engine (scrollcraft.js) pins the sheets and runs the rail; this
   file only reads where each sheet is and draws the scene for that moment.
   ========================================================================== */

import * as THREE from "three";

// ---------------------------------------------------------------- constants
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const FINE_POINTER = matchMedia("(hover: hover) and (pointer: fine)").matches;
const ROUTES = ["none", "dense", "bm25", "hybrid", "page", "agentic"];
const ROUTE_LABEL = {
  none: "no retrieval", dense: "dense RAG", bm25: "BM25 RAG",
  hybrid: "hybrid RAG", page: "page-index RAG", agentic: "deep agent",
};
const LAYER_GATE = { regex: 0, semantic: 1, llm: 2 };   // fallback = no gate caught it
const GATE_Y = [5, 0.5, -4];          // heights of the three plates
const TOP_Y = 9.2;                    // where the packet hovers before the drop
const BIN_Y = -8;                     // outlet bins
const RAIL_Y = -6.4;                  // bottom of the bypass rails
const BIN_X = [-6.25, -3.75, -1.25, 1.25, 3.75, 6.25];
const SIDE_X = 5.3;                   // bypass rails run outside the plates
const GATE_P = [0.14, 0.32, 0.5];     // drop-sheet progress when the packet reaches each gate

const css = getComputedStyle(document.documentElement);
const INK = new THREE.Color(css.getPropertyValue("--sc-ink").trim() || "#EAF0F6");
const RED = new THREE.Color(css.getPropertyValue("--sc-accent").trim() || "#FF6B45");

// ---------------------------------------------------------------- helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;
const deg = (d) => (d * Math.PI) / 180;

/** Progress 0..1 through a pinned sheet (same formula the engine uses). */
function sheetProgress(el) {
  const r = el.getBoundingClientRect();
  const travel = Math.max(el.offsetHeight - innerHeight, 1);
  return { p: clamp(-r.top / travel), visible: r.bottom > 0 && r.top < innerHeight, top: r.top };
}

// ============================================================ 1. DECISIONS
const state = {
  question: "",
  decision: null,      // the router's answer for the current question
  frames: [],          // packet keyframes [p, x, y] built from the decision
  askTouched: false,   // stop pre-filling the ask box once the visitor edits it
};
const cache = new Map();

/** Offline stand-in: only router layer 1 (the regex rules), clearly labelled. */
function offlineDecision(q) {
  const greet = /^\s*(hi+|hello+|hey+|hiya|yo|namaste|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|thx|ok(ay)?|bye|goodbye|how\s+are\s+you|who\s+are\s+you)[\s!.?,]*$/i;
  const page = /\bpage\s*(?:no\.?|number)?\s*(\d{1,4})\b/i;
  const exact = /("[^"]+"|\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b[A-Z]{2,}-?\d+\b|\bclause\s+\d)/i;
  let d = { route: "hybrid", layer: "fallback", reason: "no layer was confident — safe default", page: null };
  if (greet.test(q)) d = { route: "none", layer: "regex", reason: "greeting / small talk — no retrieval needed", page: null };
  else if (page.test(q)) { const n = +q.match(page)[1]; d = { route: "page", layer: "regex", reason: `mentions page ${n}`, page: n }; }
  else if (exact.test(q)) d = { route: "bm25", layer: "regex", reason: "exact term (quote / email / code) — keyword search", page: null };
  return { ...d, picker_enabled: false, semantic_enabled: false, hits: [], hits_mode: "", index_chunks: 0, live: false };
}

async function fetchDecision(q) {
  if (cache.has(q)) return cache.get(q);
  let d;
  try {
    const res = await fetch("/api/route", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: q }),
    });
    if (!res.ok) throw new Error(String(res.status));
    d = { ...(await res.json()), live: true };
  } catch {
    d = offlineDecision(q);   // server unreachable: say so on screen, never pretend
  }
  cache.set(q, d);
  return d;
}

/** Packet path through the assembly for a decision, as [progress, x, y] keyframes. */
function buildFrames(d) {
  const gate = d.layer in LAYER_GATE ? LAYER_GATE[d.layer] : -1;
  const binX = BIN_X[Math.max(0, ROUTES.indexOf(d.route))];
  const f = [[0, 0, TOP_Y], [0.06, 0, TOP_Y]];
  if (gate >= 0) {
    for (let j = 0; j < gate; j++) f.push([GATE_P[j], 0, GATE_Y[j]]);   // falls through earlier apertures
    const pk = GATE_P[gate];
    const side = binX < 0 ? -SIDE_X : SIDE_X;
    const rest = GATE_Y[gate] + 0.62;                                    // sits ON the plate that caught it
    f.push([pk, 0, rest], [pk + 0.06, 0, rest], [pk + 0.14, side, rest],
      [pk + 0.26, side, RAIL_Y], [pk + 0.32, binX, RAIL_Y], [pk + 0.38, binX, BIN_Y]);
  } else {
    for (let j = 0; j < 3; j++) f.push([GATE_P[j], 0, GATE_Y[j]]);      // nothing caught it
    f.push([0.62, 0, RAIL_Y], [0.7, binX, RAIL_Y], [0.78, binX, BIN_Y]);
  }
  f.push([1, binX, BIN_Y]);
  return f;
}

function packetAt(p) {
  const f = state.frames;
  if (!f.length) return { x: 0, y: TOP_Y };
  for (let i = 1; i < f.length; i++) {
    if (p <= f[i][0]) {
      const [p0, x0, y0] = f[i - 1];
      const [p1, x1, y1] = f[i];
      const t = smooth(clamp((p - p0) / Math.max(p1 - p0, 1e-6)));
      return { x: lerp(x0, x1, t), y: lerp(y0, y1, t) };
    }
  }
  const last = f[f.length - 1];
  return { x: last[1], y: last[2] };
}

/** Where each gate stands at drop progress p: waiting | passed | caught | skipped. */
function gateStatus(p) {
  const d = state.decision;
  const gate = d && d.layer in LAYER_GATE ? LAYER_GATE[d.layer] : -1;
  return [0, 1, 2].map((j) => {
    if (gate === j) return p >= GATE_P[j] ? "caught" : "waiting";
    if (gate >= 0 && j > gate) return p >= GATE_P[gate] + 0.06 ? "skipped" : "waiting";
    return p >= GATE_P[j] ? "passed" : "waiting";
  });
}
const landedAt = () => (state.frames.length ? state.frames[state.frames.length - 2][0] : 1);

// ============================================================ 2. THE RIG
const canvas = $("#rig");
const flat = $("#rig-flat");
let gl = null;

const rig = (() => {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch {
    return null;    // no WebGL: the flat elevation drawing takes over
  }
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);

  const mat = (opacity, color = INK) => new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const seg = (pts, m) => {
    const g = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
    return new THREE.LineSegments(g, m);
  };
  const loop = (r, y, m, n = 72) => {
    const pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r)); }
    return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), m);
  };
  const boxEdges = (w, h, d, m) => new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)), m);

  // --- the three gates. Each owns its materials so it can be lit or dimmed alone.
  const gates = GATE_Y.map((y, i) => {
    const group = new THREE.Group();
    group.position.y = y;
    const body = mat(0.85), detail = mat(0.42), hot = mat(0.95);
    const fill = new THREE.Mesh(
      i === 1 ? new THREE.CylinderGeometry(3.6, 3.6, 0.3, 64) : new THREE.BoxGeometry(7, 0.3, 7),
      new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0.035, depthWrite: false }),
    );
    group.add(fill);
    if (i === 1) {
      group.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.CylinderGeometry(3.6, 3.6, 0.3, 64), 30), body));
      group.add(loop(1.3, 0.16, detail), loop(2.5, 0.16, detail));
    } else {
      group.add(boxEdges(7, 0.3, 7, body));
    }
    const aperture = loop(0.55, 0.17, hot, 4);        // the hole a passing question drops through
    aperture.rotation.y = Math.PI / 4;
    group.add(aperture);

    let extra = {};
    if (i === 0) {
      // L1 regex sieve: rows of slots, the "patterns" a message is tested against
      const pts = [];
      for (let r = -2.4; r <= 2.41; r += 0.8) for (const s of [-1, 1]) pts.push([s * 0.9, 0.16, r], [s * 3.0, 0.16, r]);
      group.add(seg(pts, detail));
    }
    if (i === 1) {
      // L2 semantic field: five clusters of example questions, one per route it can pick
      const clusters = {};
      // same five routes as the prototypes in app/chat/router.py
      ["none", "dense", "bm25", "hybrid", "agentic"].forEach((route, c) => {
        const a0 = deg(30 + c * 72), pos = [];
        for (let k = 0; k < 9; k++) {
          const a = a0 + deg(((k * 37) % 25) - 12), r = 2.75 + ((k * 13) % 7) * 0.08;
          pos.push(Math.cos(a) * r, 0.18, Math.sin(a) * r);
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
        const m = new THREE.PointsMaterial({ color: INK, size: 0.13, transparent: true, opacity: 0.8 });
        group.add(new THREE.Points(g, m));
        clusters[route] = { m, at: new THREE.Vector3(Math.cos(a0) * 2.85, y + 0.18, Math.sin(a0) * 2.85) };
      });
      extra.clusters = clusters;
      extra.link = seg([[0, 0, 0], [0, 0, 0]], mat(0.9, RED));
      extra.link.visible = false;
      scene.add(extra.link);
    }
    if (i === 2) {
      // L3 LLM picker: a dial with one tick per route and a needle
      group.add(loop(2.6, 0.16, detail));
      const ticks = [];
      ROUTES.forEach((_, k) => { const a = deg(-150 + k * 60); ticks.push([Math.cos(a) * 2.3, 0.16, Math.sin(a) * 2.3], [Math.cos(a) * 2.9, 0.16, Math.sin(a) * 2.9]); });
      group.add(seg(ticks, detail));
      const needle = seg([[0, 0.2, 0], [2.1, 0.2, 0]], mat(0.95));
      group.add(needle);
      extra.needle = needle;
    }
    scene.add(group);
    return { group, body, detail, hot, aperture, fill, ...extra };
  });

  // --- structure: corner columns, the drop axis, bypass rails, manifold, bins
  const faint = mat(0.18), rails = mat(0.3);
  const cols = [];
  for (const sx of [-3.5, 3.5]) for (const sz of [-3.5, 3.5]) cols.push([sx, 5, sz], [sx, -4, sz]);
  scene.add(seg(cols, faint));
  const axis = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, TOP_Y + 0.8, 0), new THREE.Vector3(0, RAIL_Y, 0)]),
    new THREE.LineDashedMaterial({ color: INK, transparent: true, opacity: 0.35, dashSize: 0.3, gapSize: 0.25 }),
  );
  axis.computeLineDistances();
  scene.add(axis);
  scene.add(seg([[-SIDE_X, 5.6, 0], [-SIDE_X, RAIL_Y, 0], [SIDE_X, 5.6, 0], [SIDE_X, RAIL_Y, 0], [-SIDE_X, RAIL_Y, 0], [SIDE_X, RAIL_Y, 0]], rails));
  const bins = BIN_X.map((x) => {
    const m = mat(0.6);
    const b = boxEdges(1.7, 1.4, 1.7, m);
    b.position.set(x, BIN_Y - 0.1, 0);
    scene.add(b);
    return m;
  });

  // --- the packet: your question, in redline
  const packet = new THREE.Group();
  packet.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.OctahedronGeometry(0.48)), mat(1, RED)));
  packet.add(new THREE.Mesh(new THREE.OctahedronGeometry(0.47), new THREE.MeshBasicMaterial({ color: RED, transparent: true, opacity: 0.28, depthWrite: false })));
  scene.add(packet);

  return { renderer, scene, camera, gates, bins, packet };
})();

if (rig) gl = rig; else { canvas.hidden = true; buildFlat(); }

// --- No-WebGL fallback: the same assembly as a flat elevation (x, -y) in SVG.
function buildFlat() {
  flat.hidden = false;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent = flat) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); parent.appendChild(n); return n; };
  GATE_Y.forEach((y, i) => { el("rect", { x: -3.5, y: -y - 0.15, width: 7, height: 0.3, class: "fl", "data-gate": i }); });
  el("path", { d: `M0 ${-TOP_Y - 0.8} V${-RAIL_Y} M${-SIDE_X} -5.6 V${-RAIL_Y} M${SIDE_X} -5.6 V${-RAIL_Y} M${-SIDE_X} ${-RAIL_Y} H${SIDE_X}`, class: "fl-faint" });
  BIN_X.forEach((x, i) => { el("rect", { x: x - 0.85, y: -BIN_Y - 0.6, width: 1.7, height: 1.4, class: "fl", "data-bin": i }); });
  el("circle", { r: 0.42, class: "fl-red", id: "flat-packet" });
}

// ============================================================ 3. CAMERA
const cam = { az: deg(38), el: deg(24), dist: 40, ty: 0.5, ox: 0, oy: 0 };   // smoothed live values
const pointer = { x: 0, y: 0 };
if (FINE_POINTER && !REDUCE) {
  addEventListener("pointermove", (e) => { pointer.x = e.clientX / innerWidth - 0.5; pointer.y = e.clientY / innerHeight - 0.5; }, { passive: true });
}

/** Distance that fits a half-height/half-width box in view at the current aspect. */
function fitDist(halfH, halfW) {
  const t = Math.tan(deg(30 / 2));
  return Math.max(halfH / t, halfW / (t * (innerWidth / innerHeight)));
}
const isMobile = () => innerWidth <= 860;

/** Target camera for the current scroll position. */
function cameraTarget(hero, drop, pkt) {
  const m = isMobile();
  // R-1: orbit from an isometric view down toward an elevation, whole assembly in frame.
  const hp = REDUCE ? 0 : hero.p;
  const heroT = {
    az: deg(lerp(38, 24, hp)), el: deg(lerp(24, 13, hp)),
    dist: fitDist(10.6, 8.4) * (m ? (innerHeight < 700 ? 1.5 : 1.3) : 1.14), ty: m ? 1.2 : 0.9,
    ox: m ? 0 : -0.2, oy: m ? (innerHeight < 700 ? -0.4 : -0.3) : 0,   // push the drawing right (desktop) / down (phone) of the copy
  };
  if (drop.p <= 0 && drop.top > 0) return heroT;
  // R-3: follow the packet down, then pull back to show where it landed.
  const dp = REDUCE ? 1 : drop.p;
  const landed = smooth(clamp((dp - landedAt()) / 0.08));
  const follow = {
    az: deg(lerp(24, 8, dp)), el: deg(lerp(13, 7, dp)),
    dist: lerp(fitDist(5.8, 8.2), fitDist(7.2, 8.6), landed),
    ty: lerp(clamp(pkt.y, -5.5, 7.5), -5.6, landed),
    ox: m ? 0 : lerp(-0.19, -0.15, landed), oy: m ? 0.15 : 0,   // recentre the bin row on landing
  };
  const blend = smooth(clamp(dp / 0.07));
  const out = {};
  for (const k in heroT) out[k] = lerp(heroT[k], follow[k], blend);
  return out;
}

function applyCamera() {
  const c = gl.camera;
  const az = cam.az + pointer.x * deg(3), el = cam.el - pointer.y * deg(2);
  c.position.set(Math.sin(az) * Math.cos(el) * cam.dist, cam.ty + Math.sin(el) * cam.dist, Math.cos(az) * Math.cos(el) * cam.dist);
  c.lookAt(0, cam.ty, 0);
  const w = innerWidth, h = innerHeight;
  c.setViewOffset(w, h, cam.ox * w, cam.oy * h, w, h);   // shifts the drawing without moving the copy
  c.updateProjectionMatrix();
}

function resize() {
  if (!gl) return;
  gl.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, isMobile() ? 1.5 : 2));
  gl.renderer.setSize(innerWidth, innerHeight, false);
  gl.camera.aspect = innerWidth / innerHeight;
}
addEventListener("resize", resize);
resize();

// ============================================================ 4. LABELS
const tagEl = Object.fromEntries($$("[data-tag]").map((n) => [n.dataset.tag, n]));
const binTags = ROUTES.map((r) => {
  const n = document.createElement("span");
  n.className = "bintag";
  n.textContent = ROUTE_LABEL[r];
  $(".tags").appendChild(n);
  return n;
});
const v3 = new THREE.Vector3();

/** Screen position of a scene point (3D projection, or the flat SVG's transform). */
function project(x, y, z = 0) {
  if (gl) {
    v3.set(x, y, z).project(gl.camera);
    return { x: (v3.x * 0.5 + 0.5) * innerWidth, y: (-v3.y * 0.5 + 0.5) * innerHeight, ok: v3.z < 1 };
  }
  const m = flat.getScreenCTM();
  if (!m) return { x: 0, y: 0, ok: false };
  const pt = new DOMPoint(x, -y).matrixTransform(m);
  return { x: pt.x, y: pt.y, ok: true };
}
function place(node, pos, opacity, dx = 0, dy = 0) {
  if (!node) return;
  node.style.transform = `translate3d(${Math.round(pos.x + dx)}px, ${Math.round(pos.y + dy)}px, 0) translateY(-50%)`;
  node.style.opacity = pos.ok ? String(opacity) : "0";
}

// ============================================================ 5. THE PANELS
const logLines = $$(".log__lines li");
const logQ = $("#log-q"), logSrc = $("#log-src"), qStatus = $("#qstatus");
let lastLogKey = "";

function updateLog(dp) {
  const d = state.decision;
  if (!d) return;
  const st = gateStatus(dp);
  const done = dp >= landedAt() - 0.01;
  const key = st.join() + done + d.route + d.layer + d.reason;
  if (key === lastLogKey) return;   // only touch the DOM when something changed
  lastLogKey = key;
  const gate = d.layer in LAYER_GATE ? LAYER_GATE[d.layer] : -1;
  st.forEach((s, j) => {
    const li = logLines[j], v = li.querySelector(".log__v");
    li.className = s === "waiting" ? "" : `is-${s}`;
    if (s === "waiting") v.textContent = "waiting";
    else if (s === "caught") v.textContent = `caught: ${d.reason}`;
    else if (s === "skipped") v.textContent = "not reached";
    else if (j === 1 && !d.semantic_enabled) v.textContent = "no embedder loaded, skipped";
    else if (j === 2 && !d.picker_enabled) v.textContent = "switched off, passed through";
    else v.textContent = "no confident match, passed through";
  });
  const out = logLines[3], ov = out.querySelector(".log__v");
  out.className = `log__out${done ? " is-done" : ""}`;
  ov.textContent = done ? `${ROUTE_LABEL[d.route]} · via ${d.layer}${gate < 0 ? ` (${d.reason})` : ""}` : "pending";
}

function updatePanels() {
  const d = state.decision;
  const live = d.live ? "" : " · offline preview: regex layer only";
  qStatus.textContent = `Router: ${ROUTE_LABEL[d.route]} via ${d.layer}${live}`;
  logQ.textContent = `“${state.question}”`;
  tagEl.packet.textContent = state.question;
  logSrc.textContent = !d.live
    ? "Server not reachable: showing the regex layer only."
    : d.route === "none"
      ? "No search: the model answers directly."
      : `Retrieval preview: ${d.hits.length} passage${d.hits.length === 1 ? "" : "s"} (${d.hits_mode}) from ${d.index_chunks} indexed chunks.`;
  tagEl.g2.textContent = d.picker_enabled || !d.live ? "L3 · LLM picker" : "L3 · LLM picker (off)";
  tagEl.g1.textContent = d.semantic_enabled || !d.live ? "L2 · Semantic field" : "L2 · Semantic field (no embedder)";

  // R-4: stamp the detail card for this route
  $$(".detail").forEach((c) => c.classList.toggle("is-selected", c.dataset.route === d.route));
  const list = $("#found-list"), empty = $("#found-empty");
  list.replaceChildren();
  if (d.route === "none") empty.textContent = "Nothing. It was small talk, so nothing was fetched.";
  else if (!d.live) empty.textContent = "Start the server to see the real passages.";
  else if (!d.index_chunks) empty.textContent = "No passages yet. Index documents with: uv run python -m app.ingest";
  else if (!d.hits.length) empty.textContent = "Nothing in the documents is similar enough to this question, so the bot answers without them.";
  else empty.textContent = "";
  for (const h of d.hits) {
    const li = document.createElement("li");
    const meta = document.createElement("span");
    meta.className = "found__meta";
    if (h.page && /\.pdf$/i.test(h.source)) {
      const a = document.createElement("a");
      a.href = `/files/${encodeURIComponent(h.source)}#page=${h.page}`;
      a.target = "_blank"; a.rel = "noopener";
      a.textContent = `${h.source} · p.${h.page}`;
      meta.append(a, ` · score ${h.score.toFixed(3)}`);
    } else {
      meta.textContent = `${h.source} · score ${h.score.toFixed(3)}`;
    }
    const snip = document.createElement("span");
    snip.className = "found__snip";
    snip.textContent = h.snippet;
    li.append(meta, snip);
    list.append(li);
  }

  // R-5: redline your path through the graph
  const path = d.route === "none"
    ? ["start-route", "route-generate", "generate-end"]
    : d.route === "agentic"
      ? ["start-route", "route-agent", "agent-end"]
      : ["start-route", "route-retrieve", "retrieve-generate", "generate-end"];
  $$(".graph [data-edge]").forEach((e) => e.classList.toggle("is-yours", path.includes(e.dataset.edge)));
  const nodes = { none: ["START", "route", "generate", "END"], agentic: ["START", "route", "deep_agent", "END"] }[d.route]
    || ["START", "route", "retrieve", "generate", "END"];
  $$(".graph .nodes g").forEach((g) => g.classList.toggle("is-yours", nodes.includes(g.textContent.trim())));

  // chips show which sample is active
  $$(".chip").forEach((c) => c.classList.toggle("is-on", c.dataset.q === state.question));
  lastLogKey = "";
}

async function setQuestion(q) {
  q = q.trim().slice(0, 500);
  if (!q) return;
  state.question = q;
  $("#q").value = q;
  if (!state.askTouched) $("#ask-q").value = q;
  qStatus.textContent = "Routing…";
  const d = await fetchDecision(q);
  if (state.question !== q) return;   // a newer question arrived while this one was in flight
  state.decision = d;
  state.frames = buildFrames(d);
  updatePanels();
}

// --- inputs
$("#qform").addEventListener("submit", (e) => { e.preventDefault(); setQuestion($("#q").value); });
let typeTimer;
$("#q").addEventListener("input", () => { clearTimeout(typeTimer); typeTimer = setTimeout(() => setQuestion($("#q").value), 650); });
$$(".chip").forEach((c) => c.addEventListener("click", () => setQuestion(c.dataset.q)));
$("#ask-q").addEventListener("input", () => { state.askTouched = true; });

// ============================================================ 6. ASK THE BOT
const askForm = $("#askform");
askForm.addEventListener("submit", async (e) => {
  e.preventDefault();   // without JS the form posts to /ui/start and opens the chat instead
  const content = $("#ask-q").value.trim();
  if (!content) return;
  const btn = askForm.querySelector(".askform__go");
  const box = $("#answer"), routeEl = $("#answer-route"), text = $("#answer-text"), srcs = $("#answer-sources"), more = $("#answer-more");
  btn.disabled = true; btn.textContent = "Asking…";
  box.hidden = false; routeEl.textContent = ""; text.textContent = "Thinking…"; srcs.replaceChildren();
  try {
    const s = await fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (!s.ok) throw new Error(`could not start a chat (${s.status})`);
    const session = await s.json();
    const r = await fetch(`/api/sessions/${session.id}/messages`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }),
    });
    if (!r.ok) throw new Error(`the bot did not answer (${r.status})`);
    const turn = await r.json();
    const msg = turn.assistant_message;
    if (msg.route) routeEl.textContent = `${ROUTE_LABEL[msg.route.route] || msg.route.route} · via ${msg.route.layer}`;
    text.textContent = msg.content;
    for (const src of msg.sources || []) {
      const li = document.createElement("li");
      if (src.page && /\.pdf$/i.test(src.source)) {
        const a = document.createElement("a");
        a.href = `/files/${encodeURIComponent(src.source)}#page=${src.page}`;
        a.target = "_blank"; a.rel = "noopener";
        a.textContent = `${src.source} · p.${src.page}`;
        li.append(a);
      } else li.textContent = src.source;
      srcs.append(li);
    }
    more.href = `/sessions/${session.id}`;
  } catch (err) {
    text.textContent = `Couldn't reach the bot: ${err.message}. Is the server running?`;
  } finally {
    btn.disabled = false; btn.textContent = "Ask the bot";
  }
});

// ============================================================ 7. NAV + PAPER
const sheetLinks = Object.fromEntries($$(".tb-sheets a").map((a) => [a.dataset.sheet, a]));
const io = new IntersectionObserver((entries) => {
  for (const en of entries) {
    if (!en.isIntersecting) continue;
    Object.values(sheetLinks).forEach((a) => { a.classList.remove("is-current"); a.removeAttribute("aria-current"); });
    const a = sheetLinks[en.target.id];
    if (a) { a.classList.add("is-current"); a.setAttribute("aria-current", "location"); }
  }
}, { rootMargin: "-45% 0px -50% 0px" });
$$("main > section").forEach((s) => io.observe(s));

const paper = $(".paper");
const dim = $(".dimline"), dimLabel = $(".dimline__label");

// ============================================================ 8. THE FRAME LOOP
const heroEl = $("#assembly"), dropEl = $("#drop");

function frame(t) {
  const hero = sheetProgress(heroEl);
  const drop = sheetProgress(dropEl);
  const show = hero.visible || drop.visible;

  // far plane: the paper drifts at 12% of scroll (mod one major grid cell, so it never jumps)
  if (!REDUCE) paper.style.transform = `translate3d(0, ${-((scrollY * 0.12) % 140)}px, 0)`;
  // near plane: the dimension line overtakes the assembly during R-1
  if (!REDUCE && dim) { const y = -hero.p * 160; dim.style.transform = `translate3d(0, ${y}px, 0)`; dimLabel.style.transform = `translate3d(0, ${y}px, 0)`; }

  (gl ? canvas : flat).classList.toggle("is-off", !show);
  $(".tags").style.visibility = show ? "visible" : "hidden";

  if (show && state.decision) {
    const dp = drop.p <= 0 && drop.top > 0 ? 0 : (REDUCE ? 1 : drop.p);
    const pk = packetAt(dp);
    const bob = REDUCE ? 0 : Math.sin(t / 600) * 0.08;
    const st = gateStatus(dp);
    const d = state.decision;
    const gate = d.layer in LAYER_GATE ? LAYER_GATE[d.layer] : -1;
    const landed = dp >= landedAt() - 0.01;
    const routeIdx = Math.max(0, ROUTES.indexOf(d.route));

    if (gl) {
      // camera: ease toward this moment's target (snaps under reduced motion)
      const target = cameraTarget(hero, drop, pk);
      const k = REDUCE ? 1 : 0.1;
      for (const key in target) cam[key] = lerp(cam[key], target[key], k);
      applyCamera();

      gl.packet.position.set(pk.x, pk.y + (dp < 0.06 ? bob : 0), 0);
      if (!REDUCE) gl.packet.rotation.y = t / 1400;

      gl.gates.forEach((g, j) => {
        const s = st[j];
        g.body.color.copy(s === "caught" ? RED : INK);
        g.body.opacity = s === "skipped" ? 0.28 : 0.85;
        g.detail.opacity = s === "skipped" ? 0.15 : 0.42;
        const near = Math.abs(dp - GATE_P[j]) < 0.035 && s === "passed";
        g.hot.color.copy(near ? RED : INK);
        g.hot.opacity = s === "skipped" ? 0.2 : near ? 1 : 0.6;
        if (j === 2) {
          const picked = s === "caught" ? deg(-150 + routeIdx * 60) : deg(-90);
          g.needle.rotation.y = -lerp(-g.needle.rotation.y, picked, REDUCE ? 1 : 0.12);
          g.group.visible = true;
          g.body.opacity *= d.picker_enabled || !d.live ? 1 : 0.5;
        }
        if (j === 1) {
          for (const r in g.clusters) g.clusters[r].m.color.copy(s === "caught" && r === d.route ? RED : INK);
          const c = g.clusters[d.route];
          g.link.visible = s === "caught" && !!c && !landed;
          if (g.link.visible) {
            g.link.geometry.setFromPoints([new THREE.Vector3(pk.x, pk.y, 0), c.at]);
          }
        }
      });
      gl.bins.forEach((m, i) => { m.color.copy(landed && i === routeIdx ? RED : INK); m.opacity = landed && i !== routeIdx ? 0.3 : 0.65; });
      if (!document.hidden) gl.renderer.render(gl.scene, gl.camera);
    } else {
      const fp = $("#flat-packet");
      fp.setAttribute("cx", pk.x); fp.setAttribute("cy", -pk.y);
      $$("[data-gate]", flat).forEach((r, j) => r.setAttribute("class", st[j] === "caught" ? "fl-red" : "fl"));
      $$("[data-bin]", flat).forEach((r, i) => r.setAttribute("class", landed && i === routeIdx ? "fl-red" : "fl"));
    }

    // labels that ride the drawing
    const heroOn = drop.p <= 0 && drop.top > innerHeight * 0.35;
    const gateTagOpacity = heroOn ? 1 : 0.75;
    GATE_Y.forEach((y, j) => {
      const pos = project(j === 1 ? 3.6 : 3.5, y, j === 1 ? 0 : 3.5);
      place(tagEl[`g${j}`], pos, isMobile() ? 0 : gateTagOpacity, 8);   // phones: the log carries this
      tagEl[`g${j}`].classList.toggle("is-dim", st[j] === "skipped");
    });
    place(tagEl.bins, project(7.2, BIN_Y, 0), 0, 8);
    binTags.forEach((n, i) => {
      const pos = project(BIN_X[i], BIN_Y - 1.4, 0);
      n.style.transform = `translate3d(${Math.round(pos.x)}px, ${Math.round(pos.y)}px, 0) translate(-50%, 0)`;
      n.style.opacity = !heroOn && dp > landedAt() - 0.12 && !isMobile() ? (i === routeIdx && landed ? "1" : "0.6") : "0";
      n.style.color = landed && i === routeIdx ? "var(--sc-accent)" : "";
    });
    place(tagEl.packet, project(pk.x + 0.6, pk.y, 0), isMobile() ? 0 : 1, 6);

    // the stamp: where the router made its call
    const stamp = tagEl.stamp;
    const caughtNow = gate >= 0 ? st[gate] === "caught" : landed;
    if (caughtNow && !heroOn) {
      stamp.hidden = false;
      stamp.textContent = gate >= 0 ? `Caught · L${gate + 1} ${d.layer}` : `Fallback · ${ROUTE_LABEL[d.route]}`;
      // centred above the point where the call was made, clear of the log and the packet label
      const at = gate >= 0 ? project(0, GATE_Y[gate] + 1.9, 0) : project(BIN_X[routeIdx], BIN_Y + 2.1, 0);
      place(stamp, at, 1, -stamp.offsetWidth / 2);
    } else stamp.hidden = true;

    updateLog(dp);
  }
  requestAnimationFrame(frame);
}

// ============================================================ go
setQuestion($("#q").value).then(() => requestAnimationFrame(frame));
