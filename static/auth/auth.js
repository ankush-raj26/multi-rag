/* The desk: panel switching, fetch-based submits with inline errors, and the signature move:
   the key blade is re-cut from the number of characters typed, never from the characters. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const body = document.body;
  const errorBox = $("#desk-error");
  const noteBox = $("#desk-note");

  // ---- panels --------------------------------------------------------------
  const PANELS = ["signin", "signup", "forgot", "verify", "reset"];
  function show(mode, push = true) {
    if (!PANELS.includes(mode)) mode = "signin";
    body.dataset.mode = mode;
    $$(".panel").forEach((p) => p.classList.toggle("is-on", p.dataset.panel === mode));
    $$(".desk-tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.go === mode)));
    if (push) {
      const url = new URL(location.href);
      url.searchParams.set("mode", mode);
      history.replaceState(null, "", url);
    }
    const first = $(`.panel[data-panel="${mode}"] input:not([type=hidden])`);
    if (first && push) first.focus({ preventScroll: true });
  }
  $$("[data-go]").forEach((el) => el.addEventListener("click", () => { hide(errorBox); hide(noteBox); show(el.dataset.go); }));
  const initial = body.dataset.mode;
  show(initial === "sent" ? "signin" : initial, false);

  // ---- the signature move: cut the key from the password length ------------
  const blade = $("#lk-blade");
  const note = $("#lk-note");
  const MAX_CUTS = 12;
  function bladePath(cuts) {
    const x0 = 70, x1 = 300, top = 48, bottom = 72, depthMax = 14;
    if (cuts <= 0) return `M${x0} ${top} H${x1} v${bottom - top} H${x0} z`;
    const n = Math.min(cuts, MAX_CUTS);
    const step = (x1 - x0 - 20) / n;
    let d = `M${x0} ${top} H${x1} V${bottom} `;
    for (let i = n - 1; i >= 0; i--) {
      // deterministic heights from the position only (never from the character)
      const depth = 4 + ((i * 7919) % 11) / 10 * depthMax;
      const xr = x0 + 20 + (i + 1) * step, xl = x0 + 20 + i * step;
      d += `H${xr.toFixed(1)} v${-depth.toFixed(1)} H${(xl + step * 0.25).toFixed(1)} v${depth.toFixed(1)} `;
    }
    return d + `H${x0} z`;
  }
  function cut(n) {
    if (!blade) return;
    blade.setAttribute("d", bladePath(n));
    if (note) note.textContent = n === 0 ? "no cuts yet" : n < 8 ? `${n} cut${n > 1 ? "s" : ""} · too few` : `${Math.min(n, MAX_CUTS)} cuts${n > MAX_CUTS ? " · deep enough" : ""}`;
  }
  $$("input[data-cuts]").forEach((inp) => {
    inp.addEventListener("input", () => cut(inp.value.length));
    inp.addEventListener("focus", () => cut(inp.value.length));
  });
  cut(0);

  // ---- submits: fetch, show the error inline, follow `next` -----------------
  function hide(el) { if (el) el.hidden = true; }
  function say(el, text) { if (!el) return; el.textContent = text; el.hidden = false; }

  async function post(url, data) {
    const res = await fetch(url, { method: "POST", body: data, headers: { Accept: "application/json" }, credentials: "same-origin" });
    let payload = {};
    try { payload = await res.json(); } catch { payload = { ok: false, error: `Request failed (${res.status})` }; }
    return payload;
  }

  $$("form.panel").forEach((form) => {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      hide(errorBox); hide(noteBox);
      const button = form.querySelector("button[type=submit]");
      const label = button.textContent;
      button.disabled = true; button.textContent = "Working…";
      try {
        const out = await post(form.action, new FormData(form));
        if (!out.ok) { say(errorBox, out.error || "Something went wrong."); return; }
        if (form.dataset.panel === "forgot") { say(noteBox, out.message || "Check your inbox."); return; }
        if (out.next) {
          const url = new URL(out.next, location.origin);
          if (url.pathname === "/auth/verify") { // stay on this page, just switch panels
            const u = url.searchParams.get("u") || "";
            $$("input[name=u]").forEach((i) => (i.value = u));
            const resend = $("#resend"); if (resend) resend.dataset.u = u;
            say(noteBox, out.message || "Code sent.");
            show("verify");
            return;
          }
          location.href = out.next;
        }
      } catch (e) {
        say(errorBox, "Network problem. Try again.");
      } finally {
        button.disabled = false; button.textContent = label;
      }
    });
  });

  $("#resend")?.addEventListener("click", async (ev) => {
    const u = ev.currentTarget.dataset.u;
    if (!u) return;
    const fd = new FormData(); fd.set("u", u);
    const out = await post("/auth/resend", fd);
    out.ok ? say(noteBox, out.message || "Sent.") : say(errorBox, out.error || "Could not resend.");
  });
})();
