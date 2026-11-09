/* ==========================================================================
   Router panel · the 3-layer router drawn as line work (three.js)
   ----------------------------------------------------------------------------
   Same assembly as /welcome, sized for a side panel: three gate plates
   (regex, semantic, LLM picker) over six outlet bins. chat.js sends events:
     router:thinking  the packet waits above the gates while the bot works
     router:play      {route, layer, reason} the packet drops through the gate
                      that REALLY decided and rides a rail into its bin
     router:idle      back to waiting
   Drag the sheet to turn the drawing. Renders only while something moves.
   ========================================================================== */

import * as THREE from "three";

const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const ROUTES = ["none", "dense", "bm25", "hybrid", "page", "agentic"];
const ROUTE_LABEL = {
  none: "No retrieval", dense: "Dense RAG", bm25: "BM25 RAG",
  hybrid: "Hybrid RAG", page: "Page-index RAG", agentic: "Deep agent",
};
const BIN_TAG = ["none", "dense", "bm25", "hybrid", "page", "agent"];
const LAYER_TEXT = {
  regex: "Decided by layer 1, regex rules",
  semantic: "Decided by layer 2, semantic analyzer",
  llm: "Decided by layer 3, LLM picker",
  fallback: "No layer was sure, safe default",
};
const LAYER_GATE = { regex: 0, semantic: 1, llm: 2 };
const GATE_Y = [5, 0.5, -4];
const GATE_TAG = ["L1 regex", "L2 semantic", "L3 LLM"];
const TOP_Y = 8.6;
const BIN_Y = -8;
const RAIL_Y = -6.4;
const BIN_X = [-6.25, -3.75, -1.25, 1.25, 3.75, 6.25];
const SIDE_X = 5.3;
const GATE_P = [0.14, 0.32, 0.5];
const DURATION = 2800;

const css = getComputedStyle(document.documentElement);
const INK = new THREE.Color(css.getPropertyValue("--sheet-ink").trim() || "#EAF0F6");
const RED = new THREE.Color(css.getPropertyValue("--sheet-red").trim() || "#FF6B45");

const $ = (s) => document.querySelector(s);
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;
const deg = (d) => (d * Math.PI) / 180;

const sheet = $("#sheet");
const canvas = $("#router-canvas");
const tagsEl = $("#sheet-tags");
const statusEl = $("#router-status");

// ---------------------------------------------------------------- the decision card (works without WebGL)
const emptyResult = (d) => d && d.route !== "none" && d.retrieved === 0;   // searched, nothing relevant
function showDecision(d) {
  $("#decision").hidden = false;
  const label = ROUTE_LABEL[d.route] || d.route;
  $("#decision-route").textContent = emptyResult(d) ? `${label}: nothing relevant` : label;
  $("#decision").classList.toggle("is-empty", emptyResult(d));
  $("#decision-layer").textContent = LAYER_TEXT[d.layer] || d.layer;
  $("#decision-reason").textContent = emptyResult(d)
    ? `${d.reason ? d.reason + ". " : ""}It searched the documents, but no passage was similar enough to the question, so it answered without them.`
    : d.reason || "";
}
const statusText = (d) => `${ROUTE_LABEL[d.route] || d.route} · ${emptyResult(d) ? "no match" : d.layer}`;
function setStatus(text, live = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("is-live", live);
}

// ---------------------------------------------------------------- packet path
function buildFrames(d) {
  const gate = d.layer in LAYER_GATE ? LAYER_GATE[d.layer] : -1;
  const binX = BIN_X[Math.max(0, ROUTES.indexOf(d.route))];
  const f = [[0, 0, TOP_Y], [0.06, 0, TOP_Y]];
  if (gate >= 0) {
    for (let j = 0; j < gate; j++) f.push([GATE_P[j], 0, GATE_Y[j]]);
    const pk = GATE_P[gate];
    const side = binX < 0 ? -SIDE_X : SIDE_X;
    const rest = GATE_Y[gate] + 0.62;
    f.push([pk, 0, rest], [pk + 0.06, 0, rest], [pk + 0.14, side, rest],
      [pk + 0.26, side, RAIL_Y], [pk + 0.32, binX, RAIL_Y], [pk + 0.38, binX, BIN_Y]);
  } else {
    for (let j = 0; j < 3; j++) f.push([GATE_P[j], 0, GATE_Y[j]]);
    f.push([0.62, 0, RAIL_Y], [0.7, binX, RAIL_Y], [0.78, binX, BIN_Y]);
  }
  f.push([1, binX, BIN_Y]);
  return f;
}
function packetAt(f, p) {
  for (let i = 1; i < f.length; i++) {
    if (p <= f[i][0]) {
      const [p0, x0, y0] = f[i - 1], [p1, x1, y1] = f[i];
      const t = smooth(clamp((p - p0) / Math.max(p1 - p0, 1e-6)));
      return { x: lerp(x0, x1, t), y: lerp(y0, y1, t) };
    }
  }
  const last = f[f.length - 1];
  return { x: last[1], y: last[2] };
}

// ---------------------------------------------------------------- scene
function buildRig() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch {
    return null;
  }
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);

  const mat = (opacity, color = INK) => new THREE.LineBasicMaterial({ color: color.clone(), transparent: true, opacity });
  const seg = (pts, m) => new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p))), m);
  const loop = (r, y, m, n = 72) => {
    const pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r)); }
    return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), m);
  };
  const boxEdges = (w, h, d, m) => new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)), m);

  const gates = GATE_Y.map((y, i) => {
    const group = new THREE.Group();
    group.position.y = y;
    const body = mat(0.85), detail = mat(0.4), hot = mat(0.9);
    const shape = i === 1 ? new THREE.CylinderGeometry(3.6, 3.6, 0.3, 64) : new THREE.BoxGeometry(7, 0.3, 7);
    const fill = new THREE.Mesh(shape, new THREE.MeshBasicMaterial({ color: INK.clone(), transparent: true, opacity: 0.035, depthWrite: false }));
    group.add(fill);
    group.add(i === 1 ? new THREE.LineSegments(new THREE.EdgesGeometry(shape, 30), body) : boxEdges(7, 0.3, 7, body));
    if (i === 1) group.add(loop(1.3, 0.16, detail), loop(2.5, 0.16, detail));
    const aperture = loop(0.55, 0.17, hot, 4);
    aperture.rotation.y = Math.PI / 4;
    group.add(aperture);
    const extra = {};
    if (i === 0) {
      const pts = [];
      for (let r = -2.4; r <= 2.41; r += 0.8) for (const s of [-1, 1]) pts.push([s * 0.9, 0.16, r], [s * 3.0, 0.16, r]);
      group.add(seg(pts, detail));
    }
    if (i === 1) {
      // example-question clusters, one per route the semantic layer can pick (app/chat/router.py)
      extra.clusters = {};
      ["none", "dense", "bm25", "hybrid", "agentic"].forEach((route, c) => {
        const a0 = deg(30 + c * 72), pos = [];
        for (let k = 0; k < 9; k++) {
          const a = a0 + deg(((k * 37) % 25) - 12), r = 2.75 + ((k * 13) % 7) * 0.08;
          pos.push(Math.cos(a) * r, 0.18, Math.sin(a) * r);
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
        const m = new THREE.PointsMaterial({ color: INK.clone(), size: 0.14, transparent: true, opacity: 0.75 });
        group.add(new THREE.Points(g, m));
        extra.clusters[route] = { m, at: new THREE.Vector3(Math.cos(a0) * 2.85, y + 0.18, Math.sin(a0) * 2.85) };
      });
      extra.link = seg([[0, 0, 0], [0, 0, 0]], mat(0.9, RED));
      extra.link.visible = false;
      scene.add(extra.link);
    }
    if (i === 2) {
      group.add(loop(2.6, 0.16, detail));
      const ticks = [];
      ROUTES.forEach((_, k) => { const a = deg(-150 + k * 60); ticks.push([Math.cos(a) * 2.3, 0.16, Math.sin(a) * 2.3], [Math.cos(a) * 2.9, 0.16, Math.sin(a) * 2.9]); });
      group.add(seg(ticks, detail));
      extra.needleMat = mat(0.9);
      extra.needle = seg([[0, 0.2, 0], [2.1, 0.2, 0]], extra.needleMat);
      group.add(extra.needle);
    }
    scene.add(group);
    return { group, body, detail, hot, aperture, fill, ...extra };
  });

  const faint = mat(0.18), rails = mat(0.3);
  const cols = [];
  for (const sx of [-3.5, 3.5]) for (const sz of [-3.5, 3.5]) cols.push([sx, 5, sz], [sx, -4, sz]);
  scene.add(seg(cols, faint));
  const axis = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, TOP_Y + 0.8, 0), new THREE.Vector3(0, RAIL_Y, 0)]),
    new THREE.LineDashedMaterial({ color: INK, transparent: true, opacity: 0.3, dashSize: 0.3, gapSize: 0.25 }),
  );
  axis.computeLineDistances();
  scene.add(axis);
  scene.add(seg([[-SIDE_X, 5.6, 0], [-SIDE_X, RAIL_Y, 0], [SIDE_X, 5.6, 0], [SIDE_X, RAIL_Y, 0], [-SIDE_X, RAIL_Y, 0], [SIDE_X, RAIL_Y, 0]], rails));

  const bins = BIN_X.map((x) => {
    const m = mat(0.6);
    const fillM = new THREE.MeshBasicMaterial({ color: RED.clone(), transparent: true, opacity: 0, depthWrite: false });
    const b = boxEdges(1.7, 1.4, 1.7, m);
    const f = new THREE.Mesh(new THREE.BoxGeometry(1.68, 1.38, 1.68), fillM);
    b.position.set(x, BIN_Y - 0.1, 0);
    f.position.copy(b.position);
    scene.add(b, f);
    return { m, fillM, pos: b.position };
  });

  const packetEdge = mat(1, RED);
  const packetFill = new THREE.MeshBasicMaterial({ color: RED.clone(), transparent: true, opacity: 0.28, depthWrite: false });
  const packet = new THREE.Group();
  packet.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.OctahedronGeometry(0.5)), packetEdge));
  packet.add(new THREE.Mesh(new THREE.OctahedronGeometry(0.49), packetFill));
  scene.add(packet);

  return { renderer, scene, camera, gates, bins, packet };
}

const rig = buildRig();
if (!rig) sheet.classList.add("no-gl");

// ---------------------------------------------------------------- labels riding the scene
const tags = [];
if (rig) {
  GATE_TAG.forEach((text, i) => tags.push({ el: tag(text, "gate"), at: new THREE.Vector3(3.6, GATE_Y[i], 3.6), kind: "gate", i }));
  BIN_TAG.forEach((text, i) => tags.push({ el: tag(text, "bin"), at: new THREE.Vector3(BIN_X[i], BIN_Y - 1.25, 0.85), kind: "bin", i }));
}
function tag(text, kind) {
  const el = document.createElement("span");
  el.className = "sheet-tag " + kind;
  el.textContent = text;
  tagsEl.appendChild(el);
  return el;
}

// ---------------------------------------------------------------- state
const state = {
  mode: "idle",          // idle | thinking | playing | done
  decision: null,
  frames: [],
  t0: 0,
  az: deg(34), el: deg(18),   // camera orbit (drag changes az/el)
  dragging: null,
};

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

// ---------------------------------------------------------------- drawing one frame
function draw(now) {
  const { gates, bins, packet, camera, renderer } = rig;
  let p = 0, pos = { x: 0, y: TOP_Y };
  if (state.mode === "playing" || state.mode === "done") {
    p = state.mode === "done" ? 1 : clamp((now - state.t0) / DURATION);
    pos = packetAt(state.frames, p);
    if (p >= 1 && state.mode === "playing") {
      state.mode = "done";
      setStatus(statusText(state.decision));
    }
  }
  const thinking = state.mode === "thinking";
  const statuses = state.decision && !thinking && state.mode !== "idle" ? gateStatus(p) : ["waiting", "waiting", "waiting"];
  // an empty search lands in the bin but the bin stays unlit: nothing came out of it
  const landed = state.decision && !thinking && state.mode !== "idle" && p >= landedAt() && !emptyResult(state.decision);
  const binIdx = state.decision ? ROUTES.indexOf(state.decision.route) : -1;

  // packet: waits and turns while thinking, then follows its path
  packet.position.set(pos.x, pos.y, 0);
  packet.rotation.y = thinking ? now * 0.0025 : lerp(0, Math.PI * 2, p);
  packet.scale.setScalar(thinking ? 1 + Math.sin(now * 0.006) * 0.08 : 1);

  gates.forEach((g, i) => {
    const s = statuses[i];
    const caught = s === "caught";
    g.body.color.copy(caught ? RED : INK);
    g.body.opacity = caught ? 1 : s === "passed" ? 0.45 : s === "skipped" ? 0.16 : 0.85;
    g.detail.opacity = s === "skipped" ? 0.1 : 0.4;
    g.hot.color.copy(caught ? RED : INK);
    g.hot.opacity = thinking && i === 0 ? 0.5 + Math.sin(now * 0.008) * 0.4 : s === "skipped" ? 0.15 : 0.9;
    g.fill.material.color.copy(caught ? RED : INK);
    g.fill.material.opacity = caught ? 0.09 : 0.035;
    tags[i] && tags[i].el.classList.toggle("is-hot", caught);
  });

  // semantic layer: a redline to the cluster of examples that matched
  const sem = gates[1];
  const semCaught = statuses[1] === "caught";
  Object.entries(sem.clusters).forEach(([route, c]) => {
    const hit = semCaught && state.decision.route === route;
    c.m.color.copy(hit ? RED : INK);
    c.m.opacity = semCaught && !hit ? 0.25 : 0.75;
  });
  const target = semCaught && sem.clusters[state.decision.route];
  sem.link.visible = !!target && p < landedAt();
  if (sem.link.visible) {
    sem.link.geometry.setFromPoints([new THREE.Vector3(pos.x, pos.y, 0), target.at]);
  }

  // LLM picker: the needle swings to the chosen route
  const llm = gates[2];
  const llmCaught = statuses[2] === "caught";
  const needleTo = binIdx >= 0 ? deg(-(-150 + binIdx * 60)) : 0;
  llm.needle.rotation.y = llmCaught ? needleTo : thinking ? now * 0.002 : 0;
  llm.needleMat.color.copy(llmCaught ? RED : INK);

  bins.forEach((b, i) => {
    const hit = landed && i === binIdx;
    b.m.color.copy(hit ? RED : INK);
    b.m.opacity = hit ? 1 : landed ? 0.3 : 0.6;
    b.fillM.opacity = hit ? 0.22 : 0;
    tags[3 + i] && tags[3 + i].el.classList.toggle("is-hot", hit);
  });

  // camera: fixed orbit, nudged by drag; leans toward the action while playing
  const w = sheet.clientWidth, h = sheet.clientHeight;
  const tanH = Math.tan(deg(15));
  const dist = Math.max(10.4 / tanH, 9 / (tanH * (w / h))) * 0.98;
  const ty = -0.3;
  const az = state.az, el = state.el;
  camera.position.set(Math.sin(az) * Math.cos(el) * dist, ty + Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist);
  camera.lookAt(0, ty, 0);
  camera.updateMatrixWorld();
  renderer.render(rig.scene, camera);

  // labels follow their anchors
  const v = new THREE.Vector3();
  tags.forEach((t) => {
    v.copy(t.at).project(camera);
    t.el.style.translate = `${((v.x + 1) / 2) * w}px ${((1 - v.y) / 2) * h}px`;
  });

  return state.mode === "thinking" || state.mode === "playing" || !!state.dragging;
}

// render loop that sleeps when nothing moves
let raf = 0;
function tick(now) {
  raf = 0;
  if (draw(now)) raf = requestAnimationFrame(tick);
}
function wake() {
  if (rig && !raf) raf = requestAnimationFrame(tick);
}

function resize() {
  if (!rig) return;
  const w = sheet.clientWidth, h = sheet.clientHeight;
  if (!w || !h) return;
  rig.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  rig.renderer.setSize(w, h, false);
  rig.camera.aspect = w / h;
  rig.camera.updateProjectionMatrix();
  wake();
}
new ResizeObserver(resize).observe(sheet);

// ---------------------------------------------------------------- drag to turn
if (rig) {
  sheet.addEventListener("pointerdown", (e) => {
    state.dragging = { x: e.clientX, y: e.clientY, az: state.az, el: state.el };
    sheet.setPointerCapture(e.pointerId);
    sheet.classList.add("is-dragging");
    wake();
  });
  sheet.addEventListener("pointermove", (e) => {
    const d = state.dragging;
    if (!d) return;
    state.az = clamp(d.az - (e.clientX - d.x) * 0.008, deg(-70), deg(80));
    state.el = clamp(d.el + (e.clientY - d.y) * 0.005, deg(2), deg(48));
  });
  const end = () => { state.dragging = null; sheet.classList.remove("is-dragging"); };
  sheet.addEventListener("pointerup", end);
  sheet.addEventListener("pointercancel", end);
}

// ---------------------------------------------------------------- events from chat.js
document.addEventListener("router:thinking", () => {
  state.mode = "thinking";
  setStatus("Routing your question", true);
  wake();
});
document.addEventListener("router:idle", () => {
  state.mode = "idle";
  state.decision = null;
  setStatus("Waiting for a question");
  wake();
});
document.addEventListener("router:play", (e) => {
  const d = e.detail;
  state.decision = d;
  state.frames = buildFrames(d);
  showDecision(d);
  if (REDUCE || d.instant || !rig) {
    state.mode = "done";
    setStatus(statusText(d));
  } else {
    state.mode = "playing";
    state.t0 = performance.now();
    setStatus(`Dropping through the gates`, true);
  }
  wake();
});
// the answer finished: now we know whether the search found anything relevant
document.addEventListener("router:outcome", (e) => {
  const d = e.detail;
  if (!state.decision || state.decision.route !== d.route) return;
  state.decision = { ...state.decision, retrieved: d.retrieved };
  showDecision(state.decision);
  if (state.mode === "done") setStatus(statusText(state.decision));
  wake();
});
document.addEventListener("router:resize", resize);

resize();
document.dispatchEvent(new CustomEvent("router:ready"));
