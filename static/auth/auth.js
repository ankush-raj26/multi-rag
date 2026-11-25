/* The desk and the plate. Motion (window.Motion, vendored) drives: the entrance, the tab ink,
   the panel crossfade, the lock pins settling as a password is typed, and the error shake.
   Everything degrades: without Motion the page still works, it just does not move. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const body = document.body;
  const M = window.Motion || null;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const animate = (el, keys, opts) => (M && !reduced && el ? M.animate(el, keys, opts) : null);
  body.classList.add("js");

  const errorBox = $("#desk-error"), noteBox = $("#desk-note");
  const hide = (el) => { if (el) el.hidden = true; };
  const say = (el, text) => { if (!el) return; el.textContent = text; el.hidden = false; animate(el, { opacity: [0, 1], y: [-4, 0] }, { duration: 0.28 }); };
  const shake = () => animate($("#desk-card"), { x: [0, -6, 6, -4, 4, 0] }, { duration: 0.4 });

  // ---- entrance -----------------------------------------------------------
  const ins = $$("[data-in]");
  if (M && !reduced) {
    M.animate(ins, { opacity: [0, 1], y: [10, 0] }, { duration: 0.6, delay: M.stagger(0.07, { startDelay: 0.05 }), easing: [0.22, 1, 0.36, 1] });
    animate($("#desk-card"), { opacity: [0, 1], y: [12, 0] }, { duration: 0.6, delay: 0.25, easing: [0.22, 1, 0.36, 1] });
  } else ins.forEach((el) => (el.style.opacity = 1));

  // ---- panels + the sliding tab ink ---------------------------------------
  const PANELS = ["signin", "signup", "forgot", "verify", "reset"];
  const ink = $("#desk-ink");
  function moveInk(mode) {
    const tab = $(`.desk-tab[data-go="${mode}"]`);
    if (!ink) return;
    if (!tab) { ink.style.opacity = 0; return; }
    ink.style.opacity = 1;
    const x = tab.offsetLeft, w = tab.offsetWidth;
    if (!animate(ink, { x, width: w }, { duration: 0.35, easing: [0.22, 1, 0.36, 1] })) { ink.style.transform = `translateX(${x}px)`; ink.style.width = `${w}px`; }
  }
  let current = null;
  function show(mode, push = true) {
    if (!PANELS.includes(mode)) mode = "signin";
    const from = current && $(`.panel[data-panel="${current}"]`);
    const to = $(`.panel[data-panel="${mode}"]`);
    body.dataset.mode = mode;
    current = mode;
    $$(".desk-tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.go === mode)));
    moveInk(mode === "forgot" ? "signin" : mode);
    const swap = () => {
      $$(".panel").forEach((p) => p.classList.toggle("is-on", p === to));
      animate(to, { opacity: [0, 1], x: [10, 0] }, { duration: 0.32, easing: [0.22, 1, 0.36, 1] });
      const first = $("input:not([type=hidden])", to);
      if (first && push) first.focus({ preventScroll: true });
    };
    if (from && from !== to && M && !reduced) animate(from, { opacity: [1, 0], x: [0, -10] }, { duration: 0.18 }).finished.then(swap);
    else swap();
    if (push) { const url = new URL(location.href); url.searchParams.set("mode", mode); history.replaceState(null, "", url); }
  }
  $$("[data-go]").forEach((el) => el.addEventListener("click", () => { hide(errorBox); hide(noteBox); show(el.dataset.go); }));
  const initial = body.dataset.mode;
  show(initial === "sent" ? "signin" : initial, false);
  addEventListener("resize", () => moveInk(current === "forgot" ? "signin" : current));

  // ---- the lock: pins settle on the shear line as the key is cut ----------
  const PINS = $$(".pin-bar"), note = $("#lk-note"), state = $("#lock-state");
  const SHEAR = 100, TOP = 56, N = PINS.length, MIN = 8;
  function cut(n) {
    const settled = Math.min(n, N);
    PINS.forEach((pin, i) => {
      // settled pins sit on the shear line; the rest float above it at their own height
      const h = i < settled ? SHEAR - TOP - 2 : 28 + ((i * 37) % 5) * 3;
      const y = i < settled ? TOP + 2 : TOP;
      if (!animate(pin, { height: h, y }, { type: "spring", stiffness: 260, damping: 18, mass: 0.6 })) { pin.setAttribute("height", h); pin.setAttribute("y", y); }
    });
    if (note) note.textContent = `${settled} of ${N} pins set${n >= MIN ? " · it turns" : ""}`;
    if (state) state.textContent = n >= MIN ? "open" : "locked";
  }
  $$("input[data-cuts]").forEach((inp) => {
    inp.addEventListener("input", () => cut(inp.value.length));
    inp.addEventListener("focus", () => cut(inp.value.length));
  });
  cut(0);

  // ---- submits: fetch, inline errors, follow `next` --------------------------
  async function post(url, data) {
    const res = await fetch(url, { method: "POST", body: data, headers: { Accept: "application/json" }, credentials: "same-origin" });
    try { return await res.json(); } catch { return { ok: false, error: `Request failed (${res.status})` }; }
  }
  $$("form.panel").forEach((form) => {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      hide(errorBox); hide(noteBox);
      const button = form.querySelector("button[type=submit]"), label = button.querySelector("span");
      const text = label.textContent;
      button.disabled = true; label.textContent = "Working…";
      try {
        const out = await post(form.action, new FormData(form));
        if (!out.ok) { say(errorBox, out.error || "Something went wrong."); shake(); return; }
        if (form.dataset.panel === "forgot") { say(noteBox, out.message || "Check your inbox."); return; }
        if (out.next) {
          const url = new URL(out.next, location.origin);
          if (url.pathname === "/auth/verify") {
            const u = url.searchParams.get("u") || "";
            $$("input[name=u]").forEach((i) => (i.value = u));
            const resend = $("#resend"); if (resend) resend.dataset.u = u;
            say(noteBox, out.message || "Code sent.");
            show("verify");
            return;
          }
          location.href = out.next;
        }
      } catch { say(errorBox, "Network problem. Try again."); shake(); }
      finally { button.disabled = false; label.textContent = text; }
    });
  });
  $("#resend")?.addEventListener("click", async (ev) => {
    const u = ev.currentTarget.dataset.u; if (!u) return;
    const fd = new FormData(); fd.set("u", u);
    const out = await post("/auth/resend", fd);
    out.ok ? say(noteBox, out.message || "Sent.") : (say(errorBox, out.error || "Could not resend."), shake());
  });
})();
