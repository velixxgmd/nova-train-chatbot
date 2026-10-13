/* Nova Assistant frontend.
 *
 * All rendering goes through escapeHtml / mdToHtml (whitelist-based) — model
 * output is NEVER inserted as raw HTML. Conversations persist in localStorage
 * under a versioned schema; corrupt storage is discarded safely.
 */
"use strict";

const STORAGE_KEY = "nova.conversations.v1";
const PREFS_KEY = "nova.prefs.v1";

/* ---------------------------------------------------------------- utils */
const $ = (id) => document.getElementById(id);
const log = $("log");

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function safeUrl(url) {
  try {
    const u = new URL(url, "http://x.invalid"); // base just for scheme parsing
    return ["http:", "https:", "mailto:"].includes(u.protocol) ? url : "#";
  } catch { return "#"; }
}

/* Minimal, safe markdown: headings, paragraphs, lists, links, inline code,
 * fenced code blocks. No raw HTML ever passes through. */
function mdToHtml(src) {
  const blocks = [];
  let text = src.replace(/\r\n/g, "\n");

  // fenced code blocks first
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const i = blocks.push(
      `<div class="codeblock"><pre><code>${escapeHtml(code)}</code></pre>` +
      `<button class="copycode" type="button">copy</button></div>`) - 1;
    return `\u0000CODE${i}\u0000`;
  });
  // inline code
  text = text.replace(/`([^`\n]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
  // links [text](url)
  text = text.replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g,
    (_, label, url) => `<a href="${escapeHtml(safeUrl(url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`);

  const lines = text.split("\n");
  let html = "", inList = false, para = [];
  const flushPara = () => {
    if (para.length) { html += `<p>${para.join("<br>")}</p>`; para = []; }
  };
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const line of lines) {
    const m = line.match(/^\u0000CODE(\d+)\u0000$/);
    if (m) { flushPara(); closeList(); html += blocks[+m[1]]; continue; }
    if (/^\s*$/.test(line)) { flushPara(); closeList(); continue; }
    const h = line.match(/^(#{1,3})\s+(.*)/);
    if (h) { flushPara(); closeList(); html += `<h${h[1].length}>${escapeHtml(h[2])}</h${h[1].length}>`; continue; }
    const li = line.match(/^\s*[-*]\s+(.*)/);
    if (li) { flushPara(); if (!inList) { html += "<ul>"; inList = true; } html += `<li>${escapeHtml(li[1])}</li>`; continue; }
    closeList(); para.push(escapeHtml(line));
  }
  flushPara(); closeList();
  return html;
}

function relTime(ts) {
  const d = new Date(ts) - Date.now();
  if (d < 60e3) return "just now";
  if (d < 3600e3) return Math.floor(d / 60e3) + "m ago";
  if (d < 86400e3) return Math.floor(d / 3600e3) + "h ago";
  return new Date(ts).toLocaleDateString();
}

/* ------------------------------------------------------- state + storage */
let store = loadStore();
let activeId = store.conversations.length ? store.conversations[0].id : null;
let prefs = loadPrefs();

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { version: 1, conversations: [] };
    const data = JSON.parse(raw);
    if (data && data.version === 1 && Array.isArray(data.conversations)) return data;
    console.warn("discarding conversation storage with unknown schema");
  } catch (e) { console.warn("discarding corrupt conversation storage:", e); }
  return { version: 1, conversations: [] };
}
function saveStore() {
  try { store.version = 1; localStorage.setItem(STORAGE_KEY, JSON.stringify(store)); }
  catch (e) { console.warn("could not persist conversations:", e); }
}
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") || {}; }
  catch { return {}; }
}
function savePrefs() {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch {}
}

function newConversation() {
  const c = { id: "c" + Date.now() + Math.random().toString(36).slice(2, 6),
              title: "New conversation", messages: [], created: new Date().toISOString() };
  store.conversations.unshift(c);
  activeId = c.id;
  saveStore(); renderSidebar(); renderConversation();
}

function current() { return store.conversations.find(c => c.id === activeId); }

function renameConversation(id, title) {
  const c = store.conversations.find(x => x.id === id);
  if (c) { c.title = title.slice(0, 60); saveStore(); renderSidebar(); renderTopbar(); }
}
function deleteConversation(id) {
  const c = store.conversations.find(x => x.id === id);
  if (!c) return;
  if (!confirm(`Delete "${c.title}"? This cannot be undone.`)) return;
  store.conversations = store.conversations.filter(x => x.id !== id);
  if (activeId === id) activeId = store.conversations.length ? store.conversations[0].id : null;
  saveStore(); renderSidebar(); renderConversation();
}

/* ---------------------------------------------------------- server status */
let serverReady = false;
async function pingServer() {
  const pill = $("serverStatus"), mp = $("modelPill"), mt = $("modelText");
  try {
    const r = await fetch("/api/health", { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error(r.status);
    serverReady = true;
    pill.textContent = "server online";
    pill.parentElement.classList.add("ok");
    pill.parentElement.classList.remove("err");
    mp.classList.remove("err");
    const m = await fetch("/api/model").then(r => r.json());
    mt.textContent = `${m.name || "model"} · ${(m.context_tokens || 0)} ctx · ${m.device || "cpu"}`;
    mp.title = m.capabilities || "";
    $("serverDefaultsNote").textContent =
      `Server defaults: temperature ${m.settings?.temperature ?? 0.8}, ` +
      `max new tokens ${m.settings?.max_new_tokens ?? 96}. Nova is a small experimental model — ` +
      `answers may be wrong or incoherent.`;
  } catch (e) {
    serverReady = false;
    pill.textContent = "server offline";
    pill.parentElement.classList.remove("ok");
    mp.classList.add("err");
    mt.textContent = "model unreachable";
  }
}

/* -------------------------------------------------------- conversation UI */
const TEMPLATE = document.getElementById("welcomeTpl");

function renderSidebar() {
  const list = $("convlist");
  list.innerHTML = "";
  if (!store.conversations.length) {
    const d = document.createElement("div");
    d.className = "empty-hint";
    d.style.cssText = "padding:12px;color:var(--text-faint);font-size:11px;text-align:center";
    d.textContent = "No conversations yet. Start a new chat!";
    list.appendChild(d);
    return;
  }
  for (const c of store.conversations) {
    const item = document.createElement("div");
    item.className = "conv-item" + (c.id === activeId ? " active" : "");
    item.setAttribute("role", "listitem");
    const t = document.createElement("span");
    t.className = "title"; t.textContent = c.title;
    item.appendChild(t);
    const acts = document.createElement("span");
    acts.className = "actions";
    const rb = document.createElement("button");
    rb.className = "icon-btn"; rb.textContent = "✎"; rb.title = "Rename";
    rb.setAttribute("aria-label", "Rename conversation");
    rb.onclick = (e) => { e.stopPropagation();
      const name = prompt("Rename conversation:", c.title);
      if (name && name.trim()) renameConversation(c.id, name.trim()); };
    const db = document.createElement("button");
    db.className = "icon-btn"; db.textContent = "🗑"; db.title = "Delete";
    db.setAttribute("aria-label", "Delete conversation");
    db.onclick = (e) => { e.stopPropagation(); deleteConversation(c.id); };
    acts.append(rb, db);
    item.appendChild(acts);
    item.onclick = () => { activeId = c.id; saveStore(); renderSidebar(); renderConversation(); closeDrawer(); };
    list.appendChild(item);
  }
}

function renderTopbar() {
  const c = current();
  $("convTitle").textContent = c ? c.title : "New conversation";
}

function clearWelcome() {
  const w = $("welcome");
  if (w) w.remove();
}

function addMessageEl(role, html, meta) {
  const wrap = document.createElement("div");
  wrap.className = "message " + role;
  const av = document.createElement("div");
  av.className = "avatar"; av.textContent = role === "user" ? "You" : "✦";
  const body = document.createElement("div");
  body.className = "msg-body";
  const content = document.createElement("div");
  content.className = "msg-content";
  content.innerHTML = html;   // already sanitized by callers
  body.appendChild(content);
  if (meta) {
    const m = document.createElement("div");
    m.className = "msg-meta"; m.textContent = meta;
    body.appendChild(m);
  }
  const cp = document.createElement("button");
  cp.className = "msg-action"; cp.type = "button"; cp.textContent = "copy";
  cp.onclick = () => navigator.clipboard.writeText(content.innerText);
  body.appendChild(cp);
  wrap.append(av, body);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return { wrap, content };
}

function showWelcome() {
  log.innerHTML = "";
  log.appendChild(TEMPLATE.content.cloneNode(true));
  document.querySelectorAll(".example-card").forEach(b =>
    b.addEventListener("click", () => { input.value = b.dataset.prompt; autoGrow(); input.focus(); }));
}

function renderConversation() {
  const c = current();
  renderTopbar();
  if (!c || !c.messages.length) { showWelcome(); return; }
  log.innerHTML = "";
  for (const m of c.messages) {
    addMessageEl(m.role === "user" ? "user" : "assistant",
      m.role === "user" ? escapeHtml(m.text) : mdToHtml(m.text),
      relTime(m.ts));
  }
}

/* ------------------------------------------------------------- streaming */
let controller = null;

async function sendMessage(text) {
  if (!text.trim()) return;
  if (!serverReady) { await pingServer(); if (!serverReady) { showOfflineError(); return; } }
  const c = current() || (newConversation(), current());
  clearWelcome();
  const userMsg = { role: "user", text, ts: new Date().toISOString() };
  c.messages.push(userMsg);
  if (c.messages.length === 1) { c.title = text.slice(0, 40) || "New conversation"; renderSidebar(); renderTopbar(); }
  saveStore();
  addMessageEl("user", escapeHtml(text), "just now");
  $("input").value = ""; autoGrow();

  // assistant placeholder
  const asst = { role: "assistant", text: "", ts: new Date().toISOString() };
  const { content } = addMessageEl("assistant", "", "");
  content.innerHTML = '<div class="thinking">Nova is thinking<span class="dots"></span></div>';
  setBusy(true);

  controller = new AbortController();
  let got = "", done = false;
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: text, temperature: prefs.temperature ?? 0.8,
                             max_new_tokens: prefs.maxTokens ?? 96 }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error("server error " + res.status);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done: rd, value } = await reader.read();
      if (rd) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 2);
        if (!line.startsWith("data: ")) continue;
        let payload;
        try { payload = JSON.parse(line.slice(6)); }
        catch { continue; }
        if (payload.piece) {
          if (!got) content.innerHTML = "";
          got += payload.piece;
          content.innerHTML = mdToHtml(got) + '<span class="cursor">▍</span>';
          log.scrollTop = log.scrollHeight;
        }
        if (payload.refusal) { got = payload.refusal; done = true; }
        if (payload.done) { done = true; }
        if (payload.error) { got += "\n\n*generation error: " + payload.error + "*"; done = true; }
      }
    }
    if (!done && !got) throw new Error("stream ended without data");
    content.innerHTML = mdToHtml(got);
  } catch (e) {
    content.innerHTML = mdToHtml(got);
    if (e.name === "AbortError") {
      if (!got) content.innerHTML = '<div class="thinking">stopped before any output</div>';
    } else {
      content.innerHTML =
        `<div class="notice err"><div>Generation failed: ${escapeHtml(e.message)}</div>` +
        `<button class="retry" type="button">Retry</button></div>`;
      content.querySelector(".retry").onclick = () => { resendLast(); };
    }
  } finally {
    controller = null;
    setBusy(false);
    asst.text = got;
    if (got) { c.messages.push(asst); saveStore(); }
    log.scrollTop = log.scrollHeight;
  }
}

function showOfflineError() {
  const { content } = addMessageEl("assistant",
    `<div class="notice err"><div>Server is offline — launch it with:
    "python scripts/chat_server.py" from the nova_assistant folder.</div>
    <button class="retry" type="button">Retry connection</button></div>`);
  content.querySelector(".retry").onclick = () => pingServer();
}

function resendLast() {
  const c = current();
  if (!c || !c.messages.length) return;
  const lastUser = [...c.messages].reverse().find(m => m.role === "user");
  if (lastUser) sendMessage(lastUser.text);
}

function setBusy(b) {
  $("sendBtn").disabled = b;
  $("sendBtn").style.display = b ? "none" : "";
  $("stopBtn").style.display = b ? "" : "none";
  $("input").focus();
}

/* ------------------------------------------------------------ composer */
const input = $("input");
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (controller) return;
  const v = input.value.trim();
  if (v) sendMessage(v);
});
$("stopBtn").addEventListener("click", () => { if (controller) controller.abort(); });
document.querySelectorAll(".example-card").forEach(b =>
  b.addEventListener("click", () => { input.value = b.dataset.prompt; autoGrow(); input.focus(); }));

/* ------------------------------------------------------------- settings */
$("settingsBtn").onclick = () => {
  $("tempSlider").value = prefs.temperature ?? 0.8;
  $("tempVal").textContent = prefs.temperature ?? 0.8;
  $("maxTokens").value = prefs.maxTokens ?? 96;
  $("settingsModal").hidden = false;
};
$("tempSlider").oninput = (e) => { $("tempVal").textContent = e.target.value; };
$("settingsModal").addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined && e.target.hasAttribute("data-close")) {
    prefs.temperature = parseFloat($("tempSlider").value);
    prefs.maxTokens = parseInt($("maxTokens").value, 10);
    savePrefs();
    $("settingsModal").hidden = true;
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("settingsModal").hidden) $("settingsModal").hidden = true;
});

/* ------------------------------------------------------------- theme */
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("themeBtn").textContent = t === "light" ? "☾" : "◐";
}
$("themeBtn").onclick = () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  prefs.theme = next; savePrefs(); applyTheme(next);
};

/* ------------------------------------------------------------- drawer */
function closeDrawer() { $("sidebar").classList.remove("open"); }
$("menuBtn").onclick = () => $("sidebar").classList.toggle("open");
document.addEventListener("click", (e) => {
  if ($("sidebar").classList.contains("open") &&
      !$("sidebar").contains(e.target) && e.target !== $("menuBtn")) closeDrawer();
});
$("newchatBtn").onclick = () => { newConversation(); closeDrawer(); };

/* --------------------------------------------------------------- boot */
applyTheme(prefs.theme || "dark");
renderSidebar();
if (activeId) renderConversation(); else showWelcome();
autoGrow();
pingServer().then(() => { if (!serverReady) setTimeout(pingServer, 3000); });
