/* ==========================================================================
   Chat page behaviour (plain script, runs after htmx)
   - composer: auto-grow, Enter sends, your question shows at once
   - the answer streams in (Server-Sent Events from /ui/sessions/{id}/stream);
     the router panel animates the route as soon as the router decides
   - markdown for answers, inline rename, drag-and-drop upload, PDF overlay
   The 3D drawing lives in router3d.js and listens for "router:*" events.
   ========================================================================== */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));
  const scroller = $("#scroll");
  const toBottom = () => { if (scroller) scroller.scrollTop = scroller.scrollHeight; };

  // ---------------------------------------------------------------- markdown
  function renderMarkdown(root = document) {
    if (!window.marked || !window.DOMPurify) return;   // CDN blocked: plain text stays readable
    $$(".content[data-md]:not(.md)", root).forEach((el) => {
      el.innerHTML = DOMPurify.sanitize(marked.parse(el.textContent, { breaks: true, gfm: true }));
      el.classList.add("md");
    });
  }

  // ---------------------------------------------------------------- router events
  function decisionOf(article) {
    if (!article || !article.dataset.route) return null;
    const { route, layer, reason } = article.dataset;
    return { route, layer, reason };
  }
  function replay(article, opts = {}) {
    const d = decisionOf(article);
    if (!d) return;
    $$(".stamp.is-playing").forEach((s) => s.classList.remove("is-playing"));
    article.querySelector(".stamp")?.classList.add("is-playing");
    emit("router:play", { ...d, instant: !!opts.instant });
  }
  const lastAnswer = () => $$("#messages .message.assistant[data-route]").pop();

  // ---------------------------------------------------------------- composer
  const composer = $("#composer");
  const textarea = composer?.querySelector("textarea");
  const send = composer?.querySelector(".send:not(.stop)");
  const stop = composer?.querySelector(".stop");
  const ROUTES = ["none", "dense", "bm25", "hybrid", "page", "agentic"];
  const ROUTE_LABEL = { none: "No retrieval", dense: "Dense RAG", bm25: "BM25 RAG", hybrid: "Hybrid RAG", page: "Page-index RAG", agentic: "Deep agent" };
  const WORKING = { none: "Writing", agentic: "The agent is researching", page: "Reading the page" };
  let pending = null;

  function grow() {
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
    send.disabled = !textarea.value.trim() || !!pending;
  }

  function ask(text) {
    if (!composer || pending) return;
    textarea.value = text;
    grow();
    composer.requestSubmit();
  }

  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const thinkingRow = (label) => '<div class="thinking"><span class="slots" aria-hidden="true">' + "<i></i>".repeat(6) +
    "</span><span>" + esc(label) + "</span></div>";
  const stampHtml = (d) => '<button type="button" class="stamp is-playing" title="' + esc(d.reason || "") + '"><span class="slots" aria-hidden="true">' +
    ROUTES.map((r) => (r === d.route ? '<i class="on"></i>' : "<i></i>")).join("") + '</span><span class="stamp-route">' +
    esc(ROUTE_LABEL[d.route] || d.route) + '</span><span class="stamp-layer">via ' + esc(d.layer) + "</span></button>";

  // follow the answer down while it grows, but stop once your question reaches the top,
  // so a long answer reads from its start (and never fight a reader who scrolled up)
  function follow() {
    if (!pending || pending.detached) return;
    const top = pending.q.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 16;
    const room = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
    scroller.scrollTop += Math.max(0, Math.min(room, top));
  }

  // render the streamed text as markdown at most once per frame
  function paint() {
    if (!pending || pending.painted) return;
    pending.painted = true;
    requestAnimationFrame(() => {
      if (!pending) return;
      pending.painted = false;
      const md = window.marked && window.DOMPurify;
      pending.body.classList.toggle("md", !!md);
      if (md) pending.body.innerHTML = DOMPurify.sanitize(marked.parse(pending.text, { breaks: true, gfm: true }));
      else pending.body.textContent = pending.text;
      follow();
    });
  }

  function onEvent(kind, data) {
    const p = pending;
    if (kind === "route") {
      p.route = data;
      p.a.insertAdjacentHTML("afterbegin", stampHtml(data));
      p.a.querySelector(".thinking span:last-child").textContent = WORKING[data.route] || "Searching with " + ROUTE_LABEL[data.route];
      $$(".stamp.is-playing").forEach((s) => s !== p.a.querySelector(".stamp") && s.classList.remove("is-playing"));
      emit("router:play", data);   // the packet drops while the answer is being written
    } else if (kind === "token") {
      if (!p.text) { p.a.querySelector(".thinking")?.remove(); p.a.classList.add("streaming"); }
      p.text += data.text;
      paint();
    } else if (kind === "done") {
      p.done = true;
      const tpl = document.createElement("template");
      tpl.innerHTML = data.html;
      const fresh = [...tpl.content.querySelectorAll("article")];
      fresh.forEach((n) => (n.style.animation = "none"));   // already on screen: no second entrance
      p.q.replaceWith(fresh[0]);
      p.a.replaceWith(fresh[1]);
      renderMarkdown($("#messages"));
      const a = fresh[1];
      // the deep agent can fall back to hybrid: replay only if the final route changed
      if (a.dataset.route && (!p.route || p.route.route !== a.dataset.route)) replay(a);
      else a.querySelector(".stamp")?.classList.add("is-playing");
      if (!a.dataset.route) emit("router:idle");
    }
  }

  async function submitTurn() {
    const text = textarea.value.trim();
    if (!text || pending) return;
    $("#intro")?.remove();
    const q = document.createElement("article");
    q.className = "message user";
    q.innerHTML = '<div class="content"></div>';
    q.querySelector(".content").textContent = text;
    const a = document.createElement("article");
    a.className = "message assistant";
    a.innerHTML = thinkingRow("Routing your question") + '<div class="content"></div>';
    $("#messages").append(q, a);
    const ctrl = new AbortController();
    pending = { text: "", q, a, body: a.querySelector(".content"), ctrl, route: null, done: false, detached: false };
    textarea.value = "";
    grow();
    send.hidden = true; stop.hidden = false;
    scroller.scrollTop = scroller.scrollHeight;
    emit("router:thinking");

    let failed = false;
    try {
      const res = await fetch(composer.dataset.stream, {
        method: "POST", body: new URLSearchParams({ content: text }), signal: ctrl.signal,
        headers: { Accept: "text/event-stream" },
      });
      if (!res.ok || !res.body) throw new Error(String(res.status));
      // Server-Sent Events over fetch: blocks separated by a blank line, "event:" + "data:" lines
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += value;
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, cut);
          buf = buf.slice(cut + 2);
          const kind = /^event: (.*)$/m.exec(block)?.[1];
          const data = /^data: (.*)$/m.exec(block)?.[1];
          if (kind && data) onEvent(kind, JSON.parse(data));
        }
      }
      if (!pending.done) throw new Error("stream ended early");
    } catch (err) {
      const stopped = ctrl.signal.aborted;
      failed = !stopped;
      a.classList.remove("streaming");
      a.querySelector(".thinking")?.remove();
      if (stopped) {
        a.insertAdjacentHTML("beforeend", '<p class="turn-note">Stopped. What was written so far is saved</p>');
      } else {
        a.insertAdjacentHTML("beforeend", '<p class="turn-error">The answer did not come through. Your question is back in the box, try sending it again</p>');
        textarea.value = text;
      }
      if (!pending.route) emit("router:idle");
    }
    pending = null;
    send.hidden = false; stop.hidden = true;
    grow();
    updateSessionCount();
    if (failed) toBottom();
    textarea.focus();
  }

  if (composer) {
    textarea.addEventListener("input", grow);
    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        composer.requestSubmit();
      }
    });
    composer.addEventListener("submit", (e) => { e.preventDefault(); submitTurn(); });
    stop.addEventListener("click", () => pending?.ctrl.abort());
    // a reader who scrolls up while the answer streams keeps their place
    scroller.addEventListener("wheel", (e) => { if (pending && e.deltaY < 0) pending.detached = true; }, { passive: true });
    scroller.addEventListener("touchmove", () => { if (pending) pending.detached = true; }, { passive: true });
    grow();
  }

  function updateSessionCount() {
    const meta = $(".session-link.active .session-meta");
    if (meta) meta.textContent = $$("#messages .message").length;
  }

  // ---------------------------------------------------------------- clicks
  document.addEventListener("click", (e) => {
    const t = e.target;
    const stamp = t.closest("[data-replay]");
    if (stamp) {
      replay(stamp.closest(".message"));
      if (matchMedia("(max-width: 1180px)").matches) togglePanel(true);
      return;
    }
    const starter = t.closest("[data-ask]");
    if (starter) return ask(starter.dataset.ask);
    const pdf = t.closest("[data-pdf]");
    if (pdf) return viewPdf(pdf.dataset.pdf, parseInt(pdf.dataset.page, 10) || 1);
    if (t.closest("[data-close-pdf]") || t.id === "pdf-overlay") return closePdf();
    const tg = t.closest("[data-toggle]");
    if (tg) return tg.dataset.toggle === "panel" ? togglePanel() : toggleSidebar();
    if (t.closest("#replay")) { const s = $(".stamp.is-playing"); replay(s ? s.closest(".message") : lastAnswer()); }
  });

  function togglePanel(force) {
    const p = $("#panel");
    const open = force ?? !p.classList.contains("is-open");
    p.classList.toggle("is-open", open);
    if (open) emit("router:resize");
  }
  function toggleSidebar(force) {
    const s = $("#sidebar"), scrim = $(".scrim");
    const open = force ?? !s.classList.contains("is-open");
    s.classList.toggle("is-open", open);
    scrim.hidden = !open;
  }

  // ---------------------------------------------------------------- rename
  const rename = $(".rename");
  if (rename) {
    const input = rename.querySelector("input");
    let saved = input.value;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { input.value = saved; input.blur(); }
    });
    input.addEventListener("blur", () => {
      const v = input.value.trim();
      if (!v) { input.value = saved; return; }
      if (v !== saved) { saved = v; rename.submit(); }
    });
  }

  // ---------------------------------------------------------------- upload
  const drop = $("#dropzone");
  if (drop) {
    const file = drop.querySelector("input[type=file]");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("is-over")));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      if (!e.dataTransfer.files.length) return;
      file.files = e.dataTransfer.files;
      file.dispatchEvent(new Event("change", { bubbles: true }));
    });
    drop.addEventListener("htmx:afterRequest", () => drop.reset());
  }

  // ---------------------------------------------------------------- PDF overlay
  function viewPdf(name, page) {
    page = page || 1;
    const url = "/files/" + encodeURIComponent(name) + "#page=" + page;
    $("#pdf-frame").src = url;
    $("#pdf-overlay-newtab").href = url;
    $("#pdf-overlay-title").textContent = name + " · page " + page;
    const sel = $("#pdf-select"); if (sel) sel.value = name;
    const pg = $("#pdf-page"); if (pg) pg.value = page;
    $("#pdf-overlay").hidden = false;
  }
  function closePdf() {
    $("#pdf-overlay").hidden = true;
    $("#pdf-frame").src = "about:blank";   // stop rendering
  }
  window.viewPdf = viewPdf;
  $("#viewer-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    viewPdf($("#pdf-select").value, parseInt($("#pdf-page").value, 10) || 1);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#pdf-overlay").hidden) return closePdf();
    togglePanel(false); toggleSidebar(false);
  });

  // ---------------------------------------------------------------- start
  renderMarkdown();
  toBottom();
  // the router module may load after us; it asks for the latest answer when ready
  document.addEventListener("router:ready", () => {
    const a = lastAnswer();
    if (a) replay(a);
  });
  window.__chatReady = true;
})();
