(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));
  const scroller = $("#scroll");
  const toBottom = () => { if (scroller) scroller.scrollTop = scroller.scrollHeight; };

  function renderMarkdown(root = document) {
    if (!window.marked || !window.DOMPurify) return;
    $$(".content[data-md]:not(.md)", root).forEach((el) => {
      el.innerHTML = DOMPurify.sanitize(marked.parse(el.textContent, { breaks: true, gfm: true }));
      el.classList.add("md");
    });
  }

  function decisionOf(article) {
    if (!article || !article.dataset.route) return null;
    const { route, layer, reason } = article.dataset;
    const retrieved = article.dataset.retrieved === undefined ? null : Number(article.dataset.retrieved);
    return { route, layer, reason, retrieved };
  }
  function replay(article, opts = {}) {
    const d = decisionOf(article);
    if (!d) return;
    $$(".stamp.is-playing").forEach((s) => s.classList.remove("is-playing"));
    article.querySelector(".stamp")?.classList.add("is-playing");
    emit("router:play", { ...d, instant: !!opts.instant });
  }
  const lastAnswer = () => $$("#messages .message.assistant[data-route]").pop();

  const picker = $("#doc-picker");
  const docMenu = $("#doc-menu");
  const docTrigger = $("#doc-trigger");
  const docBoxes = () => (picker ? $$('input[name="docs"]', picker) : []);
  const pickedDocs = () => docBoxes().filter((b) => b.checked).map((b) => b.value);

  function syncPicker() {
    if (!picker) return;
    const picked = pickedDocs();
    $("[data-doc-all]", picker).checked = picked.length === 0;
    $(".doc-trigger-text", picker).textContent =
      picked.length === 0 ? "All documents" : picked.length === 1 ? picked[0] : picked.length + " documents";
    picker.classList.toggle("is-scoped", picked.length > 0);
    docTrigger.title = picked.length ? picked.join(", ") : "All documents";
  }
  function setPicked(names) {
    docBoxes().forEach((b) => (b.checked = names.includes(b.value)));
    syncPicker();
  }
  function toggleDocMenu(force) {
    if (!picker) return;
    const open = force ?? docMenu.hidden;
    docMenu.hidden = !open;
    docTrigger.setAttribute("aria-expanded", String(open));
  }
  if (picker) {
    docTrigger.addEventListener("click", () => toggleDocMenu());
    picker.addEventListener("change", (e) => {
      if (e.target.matches("[data-doc-all]")) setPicked([]);
      else syncPicker();
    });
    document.addEventListener("click", (e) => { if (!docMenu.hidden && !picker.contains(e.target)) toggleDocMenu(false); });
    syncPicker();
  }

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

  function ask(text, docs) {
    if (!composer || pending) return;
    if (docs) setPicked(docs);
    textarea.value = text;
    grow();
    composer.requestSubmit();
  }

  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const docTagsHtml = (names) => names.length ? '<div class="doc-tags">' +
    names.map((n) => '<span class="doc-tag">' + esc(n) + "</span>").join("") + "</div>" : "";
  const thinkingRow = (label) => '<div class="thinking"><span class="slots" aria-hidden="true">' + "<i></i>".repeat(6) +
    "</span><span>" + esc(label) + "</span></div>";
  const stampHtml = (d) => '<button type="button" class="stamp is-playing" title="' + esc(d.reason || "") + '"><span class="slots" aria-hidden="true">' +
    ROUTES.map((r) => (r === d.route ? '<i class="on"></i>' : "<i></i>")).join("") + '</span><span class="stamp-route">' +
    esc(ROUTE_LABEL[d.route] || d.route) + '</span><span class="stamp-layer">via ' + esc(d.layer) + "</span></button>";

  function follow() {
    if (!pending || pending.detached) return;
    const top = pending.q.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 16;
    const room = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
    scroller.scrollTop += Math.max(0, Math.min(room, top));
  }

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
      emit("router:play", data);
    } else if (kind === "token") {
      if (!p.text) { p.a.querySelector(".thinking")?.remove(); p.a.classList.add("streaming"); }
      p.text += data.text;
      paint();
    } else if (kind === "done") {
      p.done = true;
      const tpl = document.createElement("template");
      tpl.innerHTML = data.html;
      const fresh = [...tpl.content.querySelectorAll("article")];
      fresh.forEach((n) => (n.style.animation = "none"));
      p.q.replaceWith(fresh[0]);
      p.a.replaceWith(fresh[1]);
      renderMarkdown($("#messages"));
      const a = fresh[1];
      if (a.dataset.route && (!p.route || p.route.route !== a.dataset.route)) replay(a);
      else a.querySelector(".stamp")?.classList.add("is-playing");
      const d = decisionOf(a);
      if (d) emit("router:outcome", d);
      if (!a.dataset.route) emit("router:idle");
    }
  }

  async function submitTurn() {
    const text = textarea.value.trim();
    if (!text || pending) return;
    const docs = pickedDocs();
    toggleDocMenu(false);
    $("#intro")?.remove();
    const q = document.createElement("article");
    q.className = "message user";
    q.innerHTML = '<div class="content"></div>' + docTagsHtml(docs);
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
      const body = new URLSearchParams({ content: text });
      docs.forEach((d) => body.append("docs", d));
      const res = await fetch(composer.dataset.stream, {
        method: "POST", body, signal: ctrl.signal,
        headers: { Accept: "text/event-stream" },
      });
      if (!res.ok || !res.body) throw new Error(String(res.status));
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
    scroller.addEventListener("wheel", (e) => { if (pending && e.deltaY < 0) pending.detached = true; }, { passive: true });
    scroller.addEventListener("touchmove", () => { if (pending) pending.detached = true; }, { passive: true });
    grow();
  }

  function updateSessionCount() {
    const meta = $(".session-link.active .session-meta");
    if (meta) meta.textContent = $$("#messages .message").length;
  }

  document.addEventListener("click", (e) => {
    const t = e.target;
    const stamp = t.closest("[data-replay]");
    if (stamp) {
      replay(stamp.closest(".message"));
      if (matchMedia("(max-width: 1180px)").matches) togglePanel(true);
      return;
    }
    const starter = t.closest("[data-ask]");
    if (starter) return ask(starter.dataset.ask, starter.dataset.docs ? starter.dataset.docs.split(",") : undefined);
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
    $("#pdf-frame").src = "about:blank";
  }
  window.viewPdf = viewPdf;
  $("#viewer-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    viewPdf($("#pdf-select").value, parseInt($("#pdf-page").value, 10) || 1);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#pdf-overlay").hidden) return closePdf();
    if (docMenu && !docMenu.hidden) { toggleDocMenu(false); return docTrigger.focus(); }
    togglePanel(false); toggleSidebar(false);
  });

  renderMarkdown();
  toBottom();
  document.addEventListener("router:ready", () => {
    const a = lastAnswer();
    if (a) replay(a);
  });
  window.__chatReady = true;
})();
