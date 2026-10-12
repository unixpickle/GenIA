const $ = (selector, root = document) => root.querySelector(selector);
const history = new Map();
const cards = new Map();
const deletedIds = new Set();
const queueCards = new Map();
const pendingLogs = new WeakSet();
let activeIds = new Set();
let next = null;
let viewer = null;
let previewURL = null;
let noticeTimer;
let polling = false;
const labels = {succeeded: "Complete", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted", running: "Generating", queued: "Waiting", cancelling: "Cancelling"};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function notice(text) {
  const node = $("#notice");
  node.textContent = text; node.hidden = false;
  clearTimeout(noticeTimer); noticeTimer = setTimeout(() => { node.hidden = true; }, 7000);
}
async function api(url, options = {}) {
  const response = await fetch(url, {...options, headers: {"X-GenIA-Request": "1", ...options.headers}});
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}
function image(src, alt, className) {
  const node = element("img", className); node.src = src; node.alt = alt; node.loading = "lazy"; return node;
}
function button(text, className, action) {
  const node = element("button", className, text); node.type = "button";
  node.addEventListener("click", async () => { try { await action(node); } catch (error) { notice(error.message); } });
  return node;
}
function elapsed(job) {
  const seconds = Math.max(0, Math.floor(((job.finished || Date.now() / 1000) - (job.started || job.created))));
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}
function size(bytes) { return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`; }
function optimizationProgress(text) {
  // tqdm writes carriage-return updates, sometimes with ANSI cursor controls.
  const clean = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
  const pattern = /Finetune obj\s+(\d+):\s*\d+(?:\.\d+)?%\|[^|\r\n]*\|\s*(\d+)\s*\/\s*(\d+)\s*\[/g;
  let latest = null;
  for (const match of clean.matchAll(pattern)) {
    const completed = Number(match[2]), total = Number(match[3]);
    if (Number.isSafeInteger(total) && total > 0 && completed >= 0 && completed <= total) {
      latest = {completed, total};
    }
  }
  return latest;
}
async function refreshLog(log, {force = false, onData} = {}) {
  if ((log.hidden && !force) || pendingLogs.has(log)) return;
  pendingLogs.add(log);
  try {
    const data = await api(`/api/jobs/${log.dataset.jobId}/log`);
    if (!log.isConnected) return;
    onData?.(data.text);
    if (log.hidden) return;
    const text = data.text || "Worker has not written a log yet.";
    if (text === log.textContent) return;
    // Measure after the request: the user may have scrolled while it was in flight.
    const scrollTop = log.scrollTop;
    const follow = !log.textContent || log.scrollHeight - log.clientHeight - scrollTop <= 24;
    // The server returns a rolling tail. If its beginning was trimmed, retain
    // the first visible line where possible instead of jumping through the log.
    const previous = log.textContent;
    const lineHeight = parseFloat(getComputedStyle(log).lineHeight) || 15;
    let removedLines = 0;
    if (previous && !text.startsWith(previous)) {
      const overlap = previous.indexOf(text.slice(0, 256));
      if (overlap > 0) removedLines = (previous.slice(0, overlap).match(/\n/g) || []).length;
    }
    log.textContent = text;
    log.scrollTop = follow ? log.scrollHeight : Math.max(0, scrollTop - removedLines * lineHeight);
  } finally { pendingLogs.delete(log); }
}
function addLog(parent, job, actions = parent) {
  const log = element("pre", "logs"); log.hidden = true;
  const toggle = button("View log", "log-button", async () => {
    log.hidden = !log.hidden; toggle.textContent = log.hidden ? "View log" : "Hide log";
    toggle.setAttribute("aria-expanded", String(!log.hidden));
    if (!log.hidden) await refreshLog(log);
  });
  toggle.setAttribute("aria-expanded", "false");
  actions.append(toggle);
  parent.append(log);
  log.dataset.jobId = job.id;
}

function renderQueue(jobs) {
  const root = $("#queue");
  const ids = new Set(jobs.map(job => job.id));
  for (const [id, entry] of queueCards) {
    if (!ids.has(id)) { entry.item.remove(); queueCards.delete(id); }
  }
  $("#queue-count").textContent = jobs.length;
  const empty = $(".queue-empty", root);
  if (jobs.length) empty?.remove();
  else if (!empty) root.append(element("p", "queue-empty", "All clear. Ready for your next idea."));
  let position = 0;
  for (const [index, job] of jobs.entries()) {
    let entry = queueCards.get(job.id);
    if (!entry) {
      const item = element("article", "queue-item");
      const row = element("div", "queue-row");
      const info = element("div");
      const meta = element("div", "queue-meta");
      info.append(element("div", "queue-name", job.title), meta);
      row.append(image(job.input_url, job.title), info);
      const stage = element("div", "queue-stage");
      stage.setAttribute("role", "status");
      const progress = element("progress", "queue-progress");
      const counter = element("div", "queue-counter");
      counter.hidden = true;
      progress.title = "Progress for the current optimization pass, not the entire generation";
      const actions = element("div", "queue-actions");
      const cancel = button("Cancel", "cancel", async node => {
        node.disabled = true;
        try { await api(`/api/jobs/${job.id}/cancel`, {method: "POST"}); await refresh(); }
        finally { node.disabled = false; }
      });
      actions.append(cancel);
      item.append(row, stage, progress, counter, actions); addLog(item, job, actions);
      entry = {item, meta, stage, progress, counter, cancel}; queueCards.set(job.id, entry);
    }
    entry.meta.textContent = job.status === "queued" ? `Waiting · position ${++position}` : `${labels[job.status]} · ${elapsed(job)}`;
    const stageLabel = job.progress?.label || "Reconstruction in progress";
    if (entry.stage.textContent !== stageLabel) entry.stage.textContent = stageLabel;
    entry.stage.hidden = entry.progress.hidden = job.status !== "running";
    const stageCode = job.progress?.stage;
    if (entry.stageCode !== stageCode || job.status !== "running") {
      entry.progress.removeAttribute("value");
      entry.counter.hidden = true;
      entry.progress.removeAttribute("aria-valuetext");
    }
    entry.stageCode = stageCode;
    entry.status = job.status;
    entry.progress.setAttribute("aria-label", stageLabel);
    entry.cancel.disabled = job.status === "cancelling";
    if (root.children[index] !== entry.item) root.insertBefore(entry.item, root.children[index] || null);
    const log = $(".logs", entry.item);
    refreshLog(log, {
      force: job.status === "running" && stageCode === "finetune",
      onData: text => {
        // An old request can finish after the pipeline advances or is cancelled.
        if (entry.status !== "running" || entry.stageCode !== "finetune" || stageCode !== "finetune") return;
        const measured = optimizationProgress(text);
        if (!measured) return;
        const {completed, total} = measured;
        const percent = Math.floor(100 * completed / total);
        entry.progress.max = total;
        entry.progress.value = completed;
        const detail = `${completed} / ${total} iterations · ${percent}%`;
        entry.progress.setAttribute("aria-valuetext", detail);
        entry.counter.textContent = detail;
        entry.counter.hidden = false;
      },
    }).catch(error => { if (!log.textContent && !log.hidden) log.textContent = error.message; });
  }
}

function makeCard(job) {
  const card = element("article", "card");
  const stage = element("div", "preview");
  const previews = job.assets.filter(a => a.kind === "preview");
  const models = [...job.assets.filter(a => a.kind === "splat"), ...job.assets.filter(a => a.kind === "model")];
  const poster = image(previews[0]?.url || job.input_url, `${job.title} preview`);
  const badge = element("span", `status ${job.status}`, labels[job.status]);
  stage.append(poster, badge);
  const body = element("div", "card-body");
  const date = new Date(job.created * 1000).toLocaleString(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
  body.append(element("h3", "card-title", job.title), element("div", "card-meta", `${date} · Seed ${job.seed}${job.started ? ` · ${elapsed(job)}` : ""}`));
  if (job.error) body.append(element("p", "error", job.error));
  let selected = models[0];
  if (previews.length > 1) {
    const strip = element("div", "thumbnails");
    previews.forEach((asset, i) => {
      const thumb = button("", "", () => { if (viewer?.card === card) viewer.close(); poster.src = asset.url; });
      thumb.setAttribute("aria-label", `Preview ${i + 1}`); thumb.append(image(asset.url, `View ${i + 1}`)); strip.append(thumb);
    });
    body.append(strip);
  }
  const actions = element("div", "card-actions");
  if (models.length) {
    const downloadLabel = asset => asset.kind === "splat" ? "Download splats ↓" : "Download GLB ↓";
    const download = element("a", "download", downloadLabel(selected));
    download.href = `${selected.url}?download=1`;
    const load = button(`◇ Load 3D · ${size(selected.bytes)}`, "load-model", async node => {
      if (viewer) viewer.close();
      node.disabled = true;
      const controller = new AbortController();
      let renderer;
      let closed = false;
      const loading = element("span", "viewer-loading", "Loading 3D…");
      const close = () => {
        if (closed) return;
        closed = true;
        controller.abort(); renderer?.dispose(); stage.replaceChildren(poster, badge);
        node.disabled = false; if (viewer?.card === card) viewer = null;
      };
      viewer = {card, close};
      const closeButton = button("Close 3D ×", "close-model", close);
      stage.replaceChildren(loading, closeButton);
      try {
        // Neither renderer nor model data is requested until this click.
        const {createViewer} = selected.kind === "splat"
          ? await import("./splat-viewer.js") : await import("./viewer.js");
        if (controller.signal.aborted) return;
        renderer = await createViewer(stage, selected.url, controller.signal,
          text => { loading.textContent = text; }, selected.view_rotation);
        if (controller.signal.aborted) { renderer.dispose(); return; }
        loading.remove();
      } catch (error) {
        close(); if (error.name !== "AbortError") throw error;
      }
    });
    if (models.length > 1) {
      const select = element("select", "mesh-select"); select.setAttribute("aria-label", "3D representation to inspect");
      models.forEach((asset, i) => { const option = element("option", "", `${asset.kind === "splat" ? "Splats" : "Mesh"} · ${asset.name} · ${size(asset.bytes)}`); option.value = i; select.append(option); });
      select.addEventListener("change", () => {
        if (viewer?.card === card) viewer.close(); selected = models[Number(select.value)];
        load.textContent = `◇ Load 3D · ${size(selected.bytes)}`; download.href = `${selected.url}?download=1`;
        download.textContent = downloadLabel(selected);
      });
      body.append(select);
    }
    actions.append(load, download);
  }
  const footer = element("div", "card-footer");
  body.append(actions, footer); addLog(body, job, footer);
  const remove = button("Delete", "delete-button", async node => {
    node.disabled = true;
    try {
      try { await api(`/api/jobs/${job.id}`, {method: "DELETE"}); }
      catch (error) { if (error.status !== 404) throw error; }
      removeGeneration(job.id);
      notice("Generation and saved files deleted.");
      await refresh();
    } finally { node.disabled = false; }
  });
  remove.title = "Delete this generation and all its saved files";
  footer.append(remove);
  card.append(stage, body); return card;
}

function removeGeneration(id) {
  deletedIds.add(id);
  const card = cards.get(id);
  if (card && viewer?.card === card) viewer.close();
  card?.remove();
  cards.delete(id);
  if (history.delete(id)) $("#total").textContent = Math.max(0, Number($("#total").textContent) - 1);
  $("#empty").hidden = history.size > 0;
}

function renderHistory(jobs) {
  // Ignore stale poll/page responses that were already in flight during deletion.
  for (const job of jobs) if (!deletedIds.has(job.id)) history.set(job.id, job);
  const container = $("#history");
  const sorted = [...history.values()].sort((a, b) => b.seq - a.seq);
  for (let i = 0; i < sorted.length; i++) {
    const job = sorted[i];
    if (!cards.has(job.id)) cards.set(job.id, makeCard(job));
    const card = cards.get(job.id);
    // Do not reinsert unchanged nodes: preserve the live WebGL canvas and scroll position.
    if (container.children[i] !== card) container.insertBefore(card, container.children[i] || null);
  }
  $("#empty").hidden = sorted.length > 0;
}
async function refresh() {
  const since = history.size ? Math.min(...[...history.values()].map(job => job.seq)) : null;
  const data = await api(since === null ? "/api/jobs" : `/api/jobs?since=${since}`);
  if (data.history_ids && since !== null) {
    const liveIds = new Set(data.history_ids);
    // Also remove generations deleted from another tab, including older pages.
    for (const job of history.values()) {
      if (job.seq >= since && !liveIds.has(job.id)) removeGeneration(job.id);
    }
  }
  const currentIds = new Set(data.queue.map(job => job.id));
  // A long-running older job may finish below the newest history page after
  // newer queued jobs were cancelled. Reconcile it even if pagination ended.
  const finishedIds = [...activeIds].filter(id => !currentIds.has(id) && !history.has(id));
  const finished = await Promise.all(finishedIds.map(id => api(`/api/jobs/${id}`).catch(error => {
    if (error.status === 404) return null;
    throw error;
  })));
  activeIds = currentIds;
  data.history.push(...finished.filter(job => job && !["queued", "running", "cancelling"].includes(job.status)));
  renderQueue(data.queue); renderHistory(data.history);
  if (!$("#more").dataset.expanded) next = data.next;
  $("#more").hidden = next === null;
  $("#total").textContent = Object.entries(data.counts).filter(([key]) => !["queued", "running", "cancelling"].includes(key)).reduce((sum, [, value]) => sum + value, 0);
  $("#connection").textContent = data.worker === "running" ? "Worker active" : "Connected";
  $("#connection").classList.remove("offline");
}

$("#more").addEventListener("click", async () => {
  const node = $("#more"); node.disabled = true;
  try { const data = await api(`/api/jobs?before=${next}`); renderHistory(data.history); next = data.next; node.dataset.expanded = "1"; node.hidden = next === null; }
  catch (error) { notice(error.message); }
  finally { node.disabled = false; }
});
function selectedImage() {
  const file = $("#image").files[0]; if (!file) return;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file); $("#upload-preview").src = previewURL;
  $("#upload-preview").hidden = false; $("#drop-hint").hidden = true;
  $("#filename").textContent = `${file.name} · ${size(file.size)}`;
}
$("#image").addEventListener("change", selectedImage);
const dropzone = $("#dropzone");
dropzone.addEventListener("dragover", event => { event.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", event => {
  event.preventDefault(); dropzone.classList.remove("dragging");
  if (event.dataTransfer.files.length) {
    const transfer = new DataTransfer(); transfer.items.add(event.dataTransfer.files[0]);
    $("#image").files = transfer.files; selectedImage();
  }
});
$("#upload-form").addEventListener("submit", async event => {
  event.preventDefault(); const submit = $("#submit"); submit.disabled = true; submit.textContent = "Uploading…";
  try { await api("/api/jobs", {method: "POST", body: new FormData(event.target)}); notice("Added to the queue."); await refresh(); }
  catch (error) { notice(error.message); }
  finally { submit.disabled = false; submit.textContent = "Add to queue ↗"; }
});
async function poll() {
  if (polling || document.hidden) return;
  polling = true;
  try { await refresh(); }
  catch { $("#connection").textContent = "Reconnecting…"; $("#connection").classList.add("offline"); }
  finally { polling = false; }
}
poll(); setInterval(poll, 3000);
document.addEventListener("visibilitychange", poll);
window.addEventListener("pagehide", () => viewer?.close());
