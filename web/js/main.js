import * as THREE from 'three';
import { Viewer } from './viewer.js';
import { SceneGraph } from './scenegraph.js';
import { MATCAPS, SHADING_MODES, matcap } from './shading.js';
import {
  convertUpload, filesFromDrop, filesFromInput, getHealth, getUpload, isDirectGltf, openPath, uploadFiles,
} from './upload.js';

const $ = (id) => document.getElementById(id);
const viewer = new Viewer($('viewport'));

// source: 'upload' (files sent to the server), 'path' (read in place) or 'file' (a dropped .glb)
let current = { source: null, uploadId: null, path: null, root: null };
let purpose = 'render';
/** Variant selection overrides for the open scene: { prim path: { set: variant } }. */
let variants = {};
let report = null;
let localPaths = false;
let busy = false;

// -- small helpers ---------------------------------------------------------------

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function fmtCount(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)} k`;
  return Math.round(n).toLocaleString();
}

const basename = (p) => (p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop();
const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
  },
};

// -- status: loading card + toast -----------------------------------------------------

function setLoading(step, { detail = '', progress = null } = {}) {
  $('loading').hidden = false;
  $('loading-step').textContent = step;
  $('loading-detail').textContent = detail;
  const bar = $('loading-progress');
  if (progress === null) bar.removeAttribute('value');
  else bar.value = progress;
}

const hideLoading = () => { $('loading').hidden = true; };

let toastTimer = 0;
function toast(message, kind = 'error') {
  $('toast-text').textContent = message;
  $('toast').hidden = false;
  $('toast').classList.toggle('ok', kind === 'ok');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, kind === 'ok' ? 2500 : 12000);
}
$('toast-close').addEventListener('click', () => { $('toast').hidden = true; });

// -- screens -----------------------------------------------------------------------------

function showOpen() {
  const hasScene = !!viewer.model;
  $('welcome').hidden = false;
  $('welcome').classList.toggle('modal', hasScene);
  $('welcome-close').hidden = !hasScene;
  renderRecent();
  setTimeout(() => (localPaths ? $('path-input') : $('open-folder')).focus(), 0);
}

function hideOpen() {
  if (!viewer.model) return; // nothing to go back to
  $('welcome').hidden = true;
  $('viewport').focus({ preventScroll: true });
}

function showSceneChrome() {
  $('welcome').hidden = true;
  $('topbar').hidden = false;
  const noStage = current.source === 'file';
  $('tab-scene-btn').disabled = noStage;
  if (noStage && panel.tab === 'scene') panel.tab = 'info';
  setPanel(panel.open);
  if (!store.get('usdv.hintSeen', false)) $('nav-hint').hidden = false;
}

// -- side panel: scene graph · display · info --------------------------------------

const panel = { open: store.get('usdv.panel', true), tab: store.get('usdv.tab', 'scene') };

function setPanel(open, tab = panel.tab) {
  if (tab === 'scene' && $('tab-scene-btn').disabled) tab = 'info';
  panel.open = open;
  panel.tab = tab;
  store.set('usdv.panel', open);
  store.set('usdv.tab', tab);
  $('panel').hidden = !open || $('topbar').hidden;
  document.body.classList.toggle('panel-open', open);
  $('btn-panel').setAttribute('aria-pressed', String(open));
  for (const b of $('panel').querySelectorAll('.tab')) b.setAttribute('aria-selected', String(b.dataset.tab === tab));
  for (const name of ['scene', 'view', 'info']) $(`tab-${name}`).hidden = name !== tab;
  if (open && tab === 'scene') graph.activate();
  else graph.deactivate();
}

/** A shortcut for one tab: open it, or close the panel if it is already showing. */
function togglePanelTab(tab) {
  if (panel.open && panel.tab === tab) setPanel(false);
  else setPanel(true, tab);
}

// the panel's width: dragged at its left edge, remembered
function setPanelWidth(width) {
  const max = Math.max(320, Math.min(900, window.innerWidth * 0.7));
  const w = Math.round(Math.min(Math.max(width, 320), max));
  $('panel').style.setProperty('--panel-w', `${w}px`);
  return w;
}
setPanelWidth(store.get('usdv.panelWidth', 420));
$('panel-resize').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  const handle = e.currentTarget;
  handle.setPointerCapture(e.pointerId);
  document.body.classList.add('resizing-panel');
  const right = window.innerWidth - $('panel').getBoundingClientRect().right;
  const move = (ev) => store.set('usdv.panelWidth', setPanelWidth(window.innerWidth - right - ev.clientX));
  const up = () => {
    document.body.classList.remove('resizing-panel');
    handle.removeEventListener('pointermove', move);
    handle.removeEventListener('pointerup', up);
  };
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', up);
});
$('panel-resize').addEventListener('dblclick', () => store.set('usdv.panelWidth', setPanelWidth(420)));

// -- recent scenes --------------------------------------------------------------------

function remember(entry) {
  const key = (e) => (e.type === 'path' ? `p:${e.path}` : `u:${e.uploadId}:${e.root}`);
  const items = store.get('usdv.recent', []).filter((e) => key(e) !== key(entry));
  items.unshift({ ...entry, time: Date.now() });
  store.set('usdv.recent', items.slice(0, 8));
}

function renderRecent() {
  const items = store.get('usdv.recent', []).filter((e) => e.type === 'upload' || localPaths);
  $('recent-block').hidden = items.length === 0;
  $('recent-list').replaceChildren(...items.map((e) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = e.type === 'path'
      ? '<svg viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>'
      : '<svg viewBox="0 0 24 24"><path d="M12 15V4M7.5 8.5 12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>';
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = e.name;
    const where = document.createElement('span');
    where.className = 'where';
    where.textContent = e.type === 'path' ? e.path : 'uploaded (kept on the server for a while)';
    b.title = e.type === 'path' ? e.path : e.name;
    b.append(name, where);
    b.addEventListener('click', () => {
      if (e.type === 'path') run(() => openPathAndLoad(e.path, e.root));
      else run(() => restoreUpload(e.uploadId, e.root));
    });
    li.append(b);
    return li;
  }));
}

// -- report -> info tab --------------------------------------------------------------

/** "1.5" / "12" / "120": a length that reads well in a tile. */
function fmtLength(m) {
  if (m >= 100) return m.toFixed(0);
  if (m >= 10) return m.toFixed(0);
  if (m >= 1) return m.toFixed(1);
  return m.toFixed(2);
}

/** Warnings are written for developers: show file names and prim names, not paths. */
function friendly(text) {
  return String(text)
    // file paths (absolute or relative, either slash) -> file name
    .replace(/(?:[A-Za-z]:)?(?:[\w.@ -]*[\\/])+([\w.@-]+\.[A-Za-z0-9]{2,5})/g, '$1')
    // prim paths -> prim name
    .replace(/(?:\/[A-Za-z_][\w-]*){2,}/g, (m) => m.slice(m.lastIndexOf('/') + 1))
    .replace(/^Material ([^:]+): /, '$1: ');
}

const ICON_OK = '<svg class="issue__icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.6 2.6L16 9.6"/></svg>';
const ICON_BAD = '<svg class="issue__icon" viewBox="0 0 24 24"><path d="M12 3.5 21.5 20h-19z"/><path d="M12 10v4.2M12 17v.1"/></svg>';
const ICON_WARN = '<svg class="issue__icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7.6v5.2M12 16v.1"/></svg>';
const CHEVRON = '<svg class="chev" viewBox="0 0 16 16"><path d="m6 3.5 4.5 4.5L6 12.5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** A folded issue: a headline, one line of advice, and the list behind it. */
function issueBox(kind, title, sub, items) {
  const box = el('details', `issue issue--${kind}`);
  const summary = el('summary');
  summary.innerHTML = kind === 'bad' ? ICON_BAD : ICON_WARN;
  const text = el('div', 'issue__text');
  text.append(el('div', 'issue__title', title));
  if (sub) text.append(el('div', 'issue__sub', sub));
  summary.append(text);
  summary.insertAdjacentHTML('beforeend', CHEVRON);
  box.append(summary);
  const ul = el('ul', 'filelist');
  for (const item of items) {
    const li = el('li', undefined, item.label);
    li.title = item.title;
    ul.append(li);
  }
  box.append(ul);
  return box;
}

function showReport(report) {
  const s = report?.stats;
  const view = viewer.stats();

  const name = report ? basename(report.root) : basename(current.path);
  $('scene-name').textContent = name || 'Scene';
  $('scene-sub').textContent = report?.purpose === 'proxy' ? 'Proxy geometry' : '';
  $('scene-sub').hidden = !$('scene-sub').textContent;
  document.title = `${name} · USD Viewer`;

  // the file
  const full = report?.source === 'path' ? report.path : (report?.root || current.path || '');
  const folder = basename(full.replace(/[\\/][^\\/]*$/, ''));
  $('info-name').textContent = name;
  $('info-where').textContent = !report ? 'glTF file' : report.source === 'path' ? `in ${folder || full}` : 'Uploaded';
  $('info-where').title = full;
  $('copy-path').hidden = !full;

  // four numbers that matter
  const size = view.size;
  const tiles = [
    [fmtCount(view.fullTriangles || s?.triangles || 0), 'Triangles'],
    [s ? fmtCount(s.meshPrims) : '–', s?.instances ? 'Objects (incl. instances)' : 'Objects'],
    [s ? String(s.materials) : '–', s?.textures ? `Materials · ${s.textures} textures` : 'Materials'],
    [`${fmtLength(size.x)} × ${fmtLength(size.y)} × ${fmtLength(size.z)}`, 'Size in metres'],
  ];
  $('stats').replaceChildren(...tiles.map(([value, label]) => {
    const tile = el('div', 'tile');
    const v = el('div', 'tile__value', value);
    v.title = value;
    tile.append(v, el('div', 'tile__label', label));
    return tile;
  }));

  // problems, in plain words
  const missing = report?.unresolved || [];
  const warnings = report ? [...report.compositionErrors, ...report.warnings] : [];
  const issues = [];
  if (missing.length) {
    const advice = report?.source === 'upload'
      ? (localPaths ? 'Drop the whole project folder, or open the file by its path.' : 'Drop the whole project folder.')
      : 'Referenced, but not on disk.';
    issues.push(issueBox('bad', `${missing.length} missing ${missing.length === 1 ? 'file' : 'files'}`, advice,
      missing.map((m) => ({ label: basename(m), title: m }))));
  }
  if (warnings.length) {
    issues.push(issueBox('warn', `${warnings.length} ${warnings.length === 1 ? 'note' : 'notes'}`,
      'Things shown differently than in Houdini.', warnings.map((w) => ({ label: friendly(w), title: w }))));
  }
  if (!issues.length && report) {
    const ok = el('div', 'issue issue--ok');
    ok.innerHTML = ICON_OK;
    ok.append(el('div', 'issue__title', 'Everything loaded'));
    issues.push(ok);
  }
  $('issues').replaceChildren(...issues);
  $('issues').hidden = !issues.length;
  const dot = missing.length + warnings.length > 0;
  for (const id of ['issues-dot', 'info-dot']) {
    $(id).hidden = !dot;
    $(id).classList.toggle('bad', missing.length > 0);
  }
  if (missing.length && !panel.open) setPanel(true, 'info');

  // layers, by name
  const layers = report?.layers || [];
  $('layers-block').hidden = layers.length === 0;
  $('layers-count').textContent = layers.length;
  $('layers-list').replaceChildren(...layers.map((layer) => {
    const li = el('li', undefined, basename(layer));
    li.title = layer;
    return li;
  }));

  const foot = [];
  if (report) {
    foot.push(`${report.upAxis}-up`);
    foot.push(`converted in ${report.seconds.toFixed(1)} s`);
    foot.push(fmtBytes(report.glbBytes));
  }
  $('info-foot').textContent = foot.join(' · ');
}

/**
 * Root layer choices: the pipeline's assemblies, as usd-refgraph lists them
 * (shot roots, then sets, then assets; the server decides). A root opened some
 * other way (a link, a recent entry) stays listed so the selector shows it.
 */
function showRoots(candidates, selected) {
  const options = [...candidates];
  if (selected && !options.includes(selected)) options.push(selected);
  $('root-field').hidden = options.length < 2;
  $('root-select').replaceChildren(...options.map((f) => {
    const o = document.createElement('option');
    o.value = f;
    // the file name reads better than a path; the folder only tells sets and assets apart
    o.textContent = candidates.includes(f) ? basename(f) : `${basename(f)} (not an assembly)`;
    o.title = f;
    return o;
  }));
  if (selected) $('root-select').value = selected;
}

// -- loading flows -------------------------------------------------------------------------

function updateHash() {
  const params = new URLSearchParams();
  if (current.source === 'upload') params.set('upload', current.uploadId);
  else if (current.source === 'path') params.set('path', current.path);
  else {
    history.replaceState(null, '', location.pathname);
    return;
  }
  if (current.root) params.set('root', current.root);
  if (purpose !== 'render') params.set('purpose', purpose);
  if (Object.keys(variants).length) params.set('variants', JSON.stringify(variants));
  history.replaceState(null, '', `#${params}`);
}

/**
 * `same`: the same scene re-composed (another variant or purpose): keep the
 * camera, the scene graph's open branches and the selection.
 */
async function loadReport(next, { same = false } = {}) {
  report = next;
  updateHash();
  const label = basename(next.root);
  const progress = (p) => setLoading(`Downloading ${label}`, {
    detail: `${fmtBytes(next.glbBytes * p)} of ${fmtBytes(next.glbBytes)}`, progress: p,
  });
  if (!same) progress(0);
  await viewer.loadGlb(next.glbUrl, same ? undefined : progress, { keepView: same && !!viewer.model });
  showSceneChrome();
  showReport(next);
  graph.setScene(next.sceneId, { keep: same });
  renderVariants();
  viewer.select(graph.selection);
  hideLoading();
}

/** What changed since the last load decides whether to keep the view. */
function sameScene(source, key) {
  return current.source === source && current.key === key;
}

async function convertAndLoad(uploadId, root, { quiet = false } = {}) {
  const same = sameScene('upload', `${uploadId}:${root}`);
  if (!same) variants = {};
  if (!quiet) setLoading(`Converting ${basename(root)}`, { detail: 'Composing the stage with OpenUSD…' });
  const next = await convertUpload(uploadId, root, purpose, nonEmpty(variants));
  current = { source: 'upload', uploadId, path: null, root: next.root, key: `${uploadId}:${root}` };
  remember({ type: 'upload', uploadId, root: next.root, name: basename(next.root) });
  await loadReport(next, { same });
}

async function restoreUpload(uploadId, root) {
  setLoading('Opening upload');
  const summary = await getUpload(uploadId);
  showRoots(summary.rootCandidates, root || summary.root);
  await convertAndLoad(uploadId, root || summary.root);
}

async function openPathAndLoad(path, root = null, { quiet = false, withVariants = null } = {}) {
  const same = sameScene('path', `${path}:${root}`);
  if (!same) variants = withVariants ?? {};
  if (!quiet) setLoading(`Converting ${basename(root || path)}`, { detail: 'Composing the stage with OpenUSD…' });
  const next = await openPath(path, root, purpose, nonEmpty(variants));
  current = { source: 'path', uploadId: null, path, root: next.folder ? next.root : null, key: `${path}:${root}` };
  if (next.folder) showRoots(next.rootCandidates, next.root);
  else $('root-field').hidden = true;
  $('path-input').value = path;
  remember({ type: 'path', path, root: current.root, name: basename(next.root) });
  await loadReport(next, { same });
}

const nonEmpty = (v) => (Object.keys(v).length ? v : null);

/** Convert the current scene again (after switching purpose or a variant). */
function reloadCurrent(options = {}) {
  if (current.source === 'upload') return run(() => convertAndLoad(current.uploadId, current.root, options));
  if (current.source === 'path') return run(() => openPathAndLoad(current.path, current.root, options));
  return null;
}

// -- variants ---------------------------------------------------------------------------

/** Re-compose with one variant selection changed; the files on disk are not touched. */
function setVariant(primPath, set, value) {
  if (busy || !report) return;
  variants = { ...variants, [primPath]: { ...(variants[primPath] || {}), [set]: value } };
  // the converted result is cached per selection, so most switches are quick:
  // show progress in place and only bring up the loading card if it takes a while
  setVariantBusy(true);
  const slow = setTimeout(() => setLoading(`Switching ${set} to ${value}`, { detail: 'Composing the stage with OpenUSD…' }), 700);
  const done = reloadCurrent({ quiet: true });
  Promise.resolve(done).finally(() => {
    clearTimeout(slow);
    setVariantBusy(false);
  });
}

function setVariantBusy(on) {
  for (const chip of $('variants').querySelectorAll('.vchip')) chip.classList.toggle('busy', on);
  for (const c of $('variants').querySelectorAll('select, button')) c.disabled = on;
  graph.setBusy(on);
}

/** Step the default prim's variant set `index` by `delta`, wrapping around. */
function cycleVariant(index = lastVariantSet, delta = 1) {
  const dv = report?.defaultVariants;
  const set = dv?.sets[index];
  if (!set) return;
  lastVariantSet = index;
  const at = Math.max(0, set.options.indexOf(set.selection));
  const value = set.options[(at + delta + set.options.length) % set.options.length];
  setVariant(dv.prim, set.name, value);
}
let lastVariantSet = 0;

const CHEVRON_LEFT = '<svg viewBox="0 0 24 24"><path d="m14.5 6-6 6 6 6"/></svg>';
const CHEVRON_RIGHT = '<svg viewBox="0 0 24 24"><path d="m9.5 6 6 6-6 6"/></svg>';

/** The default prim's variant sets in the top bar, each with previous / next. */
function renderVariants() {
  const dv = report?.defaultVariants;
  const host = $('variants');
  host.hidden = !dv;
  if (!dv) {
    host.replaceChildren();
    return;
  }
  if (lastVariantSet >= dv.sets.length) lastVariantSet = 0;
  host.replaceChildren(...dv.sets.map((set, index) => {
    const chip = document.createElement('div');
    chip.className = 'vchip';
    chip.title = `Variant set "${set.name}" on ${dv.prim}. [ and ] cycle it.`;
    const name = document.createElement('span');
    name.className = 'vname';
    name.textContent = set.name;
    const prev = document.createElement('button');
    prev.type = 'button';
    prev.innerHTML = CHEVRON_LEFT;
    prev.title = 'Previous variant ([)';
    prev.addEventListener('click', () => cycleVariant(index, -1));
    const select = document.createElement('select');
    select.setAttribute('aria-label', `Variant set ${set.name}`);
    select.replaceChildren(...set.options.map((option) => {
      const o = document.createElement('option');
      o.value = option;
      o.textContent = option;
      return o;
    }));
    select.value = set.selection ?? '';
    select.addEventListener('change', () => {
      lastVariantSet = index;
      setVariant(dv.prim, set.name, select.value);
    });
    const next = document.createElement('button');
    next.type = 'button';
    next.innerHTML = CHEVRON_RIGHT;
    next.title = 'Next variant (])';
    next.addEventListener('click', () => cycleVariant(index, 1));
    chip.append(name, prev, select, next);
    return chip;
  }));
}

async function run(task) {
  if (busy) return;
  busy = true;
  $('toast').hidden = true;
  try {
    await task();
  } catch (err) {
    console.error(err);
    hideLoading();
    toast(err.message || String(err));
  } finally {
    busy = false;
  }
}

function openFiles(files) {
  if (!files.length) return;
  run(async () => {
    if (isDirectGltf(files)) {
      const url = URL.createObjectURL(files[0].file);
      try {
        setLoading(`Loading ${files[0].path}`);
        await viewer.loadGlb(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      current = { source: 'file', uploadId: null, path: files[0].path, root: null };
      report = null;
      variants = {};
      $('root-field').hidden = true;
      updateHash();
      showSceneChrome();
      showReport(null);
      graph.setScene(null);
      renderVariants();
      hideLoading();
      return;
    }
    const total = files.reduce((n, f) => n + f.file.size, 0);
    const label = `Uploading ${files.length} file${files.length === 1 ? '' : 's'}`;
    setLoading(label, { detail: fmtBytes(total), progress: 0 });
    const summary = await uploadFiles(files, (p) => setLoading(label, {
      detail: `${fmtBytes(total * p)} of ${fmtBytes(total)}`, progress: p,
    }));
    showRoots(summary.rootCandidates, summary.root);
    await convertAndLoad(summary.uploadId, summary.root);
  });
}

async function restoreFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  setPurpose(params.get('purpose') === 'proxy' ? 'proxy' : 'render');
  const uploadId = params.get('upload');
  const path = params.get('path');
  let wanted = {};
  try { wanted = JSON.parse(params.get('variants') || '{}') || {}; } catch { /* ignore a bad link */ }
  const key = path ? `${path}:${params.get('root')}` : `${uploadId}:${params.get('root')}`;
  if (current.key === key && JSON.stringify(wanted) === JSON.stringify(variants)) return; // already showing it
  if (path) await run(() => openPathAndLoad(path, params.get('root'), { withVariants: wanted }));
  else if (uploadId) await run(() => restoreUpload(uploadId, params.get('root')));
}

// -- settings ---------------------------------------------------------------------------------

function setPurpose(value) {
  purpose = value;
  $('opt-proxy').checked = value === 'proxy';
  $('btn-proxy').setAttribute('aria-pressed', String(value === 'proxy'));
}

function togglePurpose() {
  if (busy) return;
  setPurpose(purpose === 'proxy' ? 'render' : 'proxy');
  if (!reloadCurrent()) updateHash();
}

// -- scene graph ------------------------------------------------------------------------

const graph = new SceneGraph($('graph-body'), {
  onSelect: (sel) => viewer.select(sel),
  onFrame: (path) => {
    if (!viewer.frameSelection({ path, instance: graph.selection?.path === path ? graph.selection.instance : null })) {
      toast('Nothing under this prim is drawn in the view', 'ok');
    }
  },
  onToast: (message, kind) => toast(message, kind),
  geometry: (sel) => viewer.selection.geometry(sel),
  onVariant: (path, set, value) => setVariant(path, set, value),
});

/** Select from the 3D view (or clear with null); the tree opens onto the prim. */
function selectFromView(hit) {
  if (!hit) {
    viewer.select(null);
    void graph.reveal(null);
    return;
  }
  const sel = { path: hit.path, instance: hit.instance ?? null };
  viewer.select(sel);
  void graph.reveal(sel.path, sel.instance);
}

function frame() {
  const sel = graph.selection;
  if (!sel || !viewer.frameSelection(sel)) viewer.frameScene();
}

const settings = store.get('usdv.settings', {});
function saveSetting(key, value) {
  settings[key] = value;
  store.set('usdv.settings', settings);
}

function bindSwitch(id, key, apply) {
  const input = $(id);
  if (key in settings) input.checked = settings[key];
  apply(input.checked);
  input.addEventListener('change', () => {
    apply(input.checked);
    saveSetting(key, input.checked);
  });
}

bindSwitch('opt-grid', 'grid', (on) => viewer.setGrid(on));
bindSwitch('opt-shadows', 'shadows', (on) => viewer.setShadows(on));
bindSwitch('opt-double', 'doubleSided', (on) => viewer.setDoubleSided(on));
let showStats = false;
bindSwitch('opt-stats', 'stats', (on) => { showStats = on; });
$('opt-proxy').addEventListener('change', () => {
  $('opt-proxy').checked = purpose === 'proxy'; // togglePurpose decides
  togglePurpose();
});

// -- viewport shading: material · matcap · wireframe ----------------------------------

function setShading(mode, look = settings.matcap || 'clay') {
  if (!SHADING_MODES.includes(mode)) mode = 'material';
  viewer.setShading(mode, look);
  for (const b of document.querySelectorAll('[data-shading]')) {
    b.setAttribute('aria-checked', String(b.dataset.shading === mode));
  }
  for (const b of $('matcaps').children) b.setAttribute('aria-checked', String(b.dataset.look === look));
  $('matcaps').hidden = mode !== 'matcap';
  saveSetting('shading', mode);
  saveSetting('matcap', look);
}

$('matcaps').replaceChildren(...MATCAPS.map((m) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'swatch';
  b.dataset.look = m.id;
  b.title = m.label;
  b.setAttribute('role', 'radio');
  b.setAttribute('aria-label', m.label);
  const img = document.createElement('img');
  img.src = matcap(m.id).url;
  img.alt = '';
  const label = document.createElement('span');
  label.textContent = m.label;
  b.append(img, label);
  b.addEventListener('click', () => setShading('matcap', m.id));
  return b;
}));
for (const b of document.querySelectorAll('[data-shading]')) {
  b.addEventListener('click', () => setShading(b.dataset.shading));
}
setShading(settings.shading || 'material', settings.matcap || 'clay');

function cycleShading() {
  const at = SHADING_MODES.indexOf(viewer.options.shading);
  setShading(SHADING_MODES[(at + 1) % SHADING_MODES.length]);
}

const DETAIL_HINTS = {
  fast: 'Simplifies sooner. Smoothest on very heavy scenes.',
  balanced: 'Full detail up close, simpler far away.',
  high: 'Full detail further out.',
  full: 'Always full detail. Can be slow on heavy scenes.',
};

function setDetail(value) {
  viewer.setDetail(value);
  for (const b of $('detail').querySelectorAll('button')) {
    b.setAttribute('aria-checked', String(b.dataset.value === value));
  }
  $('detail-hint').textContent = DETAIL_HINTS[value] || '';
  saveSetting('detail', value);
}
$('detail').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) setDetail(b.dataset.value);
});
setDetail(settings.detail || 'balanced');

function showExposure(stops) {
  $('exposure-value').textContent = stops === 0 ? '0' : `${stops > 0 ? '+' : ''}${stops.toFixed(1)}`;
}
$('opt-exposure').value = settings.exposure ?? 0;
viewer.setExposure(2 ** Number($('opt-exposure').value));
showExposure(Number($('opt-exposure').value));
$('opt-exposure').addEventListener('input', (e) => {
  viewer.setExposure(2 ** Number(e.target.value));
  showExposure(Number(e.target.value));
  saveSetting('exposure', Number(e.target.value));
});
$('opt-exposure').addEventListener('dblclick', (e) => {
  e.target.value = 0;
  e.target.dispatchEvent(new Event('input'));
});


// -- wiring -------------------------------------------------------------------------------------

$('open-folder').addEventListener('click', () => $('folder-input').click());
$('open-files').addEventListener('click', () => $('files-input').click());
for (const id of ['folder-input', 'files-input']) {
  $(id).addEventListener('change', (e) => {
    openFiles(filesFromInput(e.target));
    e.target.value = '';
  });
}

$('path-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const path = $('path-input').value.trim().replace(/^"(.*)"$/, '$1');
  if (path) run(() => openPathAndLoad(path));
});

$('root-select').addEventListener('change', (e) => {
  if (current.source === 'upload') run(() => convertAndLoad(current.uploadId, e.target.value));
  else if (current.source === 'path') run(() => openPathAndLoad(current.path, e.target.value));
});

$('btn-open').addEventListener('click', showOpen);
$('welcome-close').addEventListener('click', hideOpen);
$('welcome').addEventListener('pointerdown', (e) => { if (e.target === $('welcome')) hideOpen(); });
$('btn-frame').addEventListener('click', frame);
$('btn-panel').addEventListener('click', () => setPanel(!panel.open));
$('panel-close').addEventListener('click', () => setPanel(false));
for (const b of $('panel').querySelectorAll('.tab')) b.addEventListener('click', () => setPanel(true, b.dataset.tab));
$('copy-path').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('info-where').title);
    toast('Path copied', 'ok');
  } catch {
    toast('Could not copy to the clipboard');
  }
});

// click (not drag) in the view selects what is under the cursor
let pressAt = null;
$('viewport').addEventListener('pointerdown', (e) => {
  pressAt = e.button === 0 && !e.shiftKey ? { x: e.clientX, y: e.clientY, t: performance.now() } : null;
});
$('viewport').addEventListener('pointerup', (e) => {
  const p = pressAt;
  pressAt = null;
  if (!p || e.button !== 0 || !viewer.model || !graph.sceneId) return;
  if (Math.hypot(e.clientX - p.x, e.clientY - p.y) > 4 || performance.now() - p.t > 400) return;
  const rect = $('viewport').getBoundingClientRect();
  selectFromView(viewer.pickObject(e.clientX - rect.left, e.clientY - rect.top));
});
$('btn-proxy').addEventListener('click', togglePurpose);
$('btn-help').addEventListener('click', () => { $('help').hidden = false; });
$('help-close').addEventListener('click', () => { $('help').hidden = true; });
$('help').addEventListener('pointerdown', (e) => { if (e.target === $('help')) $('help').hidden = true; });

function isTyping(e) {
  const t = e.target;
  return t.isContentEditable || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT'
    || (t.tagName === 'INPUT' && !['checkbox', 'range', 'button'].includes(t.type));
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (!$('help').hidden) $('help').hidden = true;
    else if (!$('welcome').hidden) hideOpen();
    else if (graph.selection) selectFromView(null);
    return;
  }
  if (isTyping(e)) return;
  // brackets by character: their key differs between layouts (AltGr+8/9 on Nordic
  // keyboards, which arrives with Ctrl+Alt held)
  if ((e.key === '[' || e.key === ']') && viewer.model && $('welcome').hidden) {
    cycleVariant(lastVariantSet, e.key === ']' ? 1 : -1);
    e.preventDefault();
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (!$('welcome').hidden && !viewer.model) return;
  const click = (id) => $(id).click();
  switch (e.code) {
    case 'KeyF': frame(); break;
    case 'KeyZ': cycleShading(); break;
    case 'KeyT': if (graph.sceneId) togglePanelTab('scene'); break;

    case 'KeyG': click('opt-grid'); break;
    case 'KeyL': click('opt-shadows'); break;
    case 'KeyB': click('opt-double'); break;
    case 'KeyP': togglePurpose(); break;
    case 'KeyI': togglePanelTab('info'); break;
    case 'KeyO': showOpen(); break;
    case 'Slash': case 'F1': $('help').hidden = !$('help').hidden; break;
    default: return;
  }
  e.preventDefault();
  if (e.target !== $('viewport')) e.target.blur?.();
});

// drag & drop anywhere
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth += 1;
  document.body.classList.add('drag-over');
  if ($('welcome').hidden) $('drop-overlay').hidden = false;
});
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    $('drop-overlay').hidden = true;
    document.body.classList.remove('drag-over');
  }
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('drop-overlay').hidden = true;
  document.body.classList.remove('drag-over');
  openFiles(await filesFromDrop(e.dataTransfer));
});

// navigation feedback: cursor, pivot marker, first-use hint
const nav = viewer.nav;
let dragMode = null;
nav.addEventListener('dragstart', (e) => {
  dragMode = e.detail.mode;
  document.body.classList.add(`dragging-${dragMode}`);
});
nav.addEventListener('dragend', (e) => {
  document.body.classList.remove(`dragging-${e.detail.mode}`);
  dragMode = null;
  $('pivot').hidden = true;
});
let interactions = 0;
nav.addEventListener('interact', () => {
  interactions += 1;
  if (interactions === 6 && !$('nav-hint').hidden) {
    $('nav-hint').classList.add('fade');
    setTimeout(() => { $('nav-hint').hidden = true; }, 700);
    store.set('usdv.hintSeen', true);
  }
});

// HUD
const pivotScreen = new THREE.Vector3();
let lastHud = 0;
let frames = 0;
let fpsWindowStart = performance.now();
let fps = 0;
viewer.onFrame = (rendered) => {
  const now = performance.now();
  if (rendered) frames += 1;
  if (now - fpsWindowStart > 500) {
    fps = (frames * 1000) / (now - fpsWindowStart);
    frames = 0;
    fpsWindowStart = now;
  }
  if (dragMode === 'orbit') {
    pivotScreen.copy(nav.pivot).project(viewer.camera);
    const visible = pivotScreen.z < 1;
    $('pivot').hidden = !visible;
    if (visible) {
      $('pivot').style.left = `${((pivotScreen.x + 1) / 2) * window.innerWidth}px`;
      $('pivot').style.top = `${((1 - pivotScreen.y) / 2) * window.innerHeight}px`;
    }
  }
  if (now - lastHud < 250 || !viewer.model) return;
  lastHud = now;
  $('hud').hidden = !(showStats || nav.isFlying);
  $('hud-fps').hidden = !showStats;
  $('hud-tris').hidden = !showStats;
  if (showStats) {
    const s = viewer.stats();
    $('hud-fps').textContent = fps >= 3 ? `${Math.round(fps)} fps` : 'idle';
    $('hud-tris').textContent = `${fmtCount(s.drawnTriangles)} triangles`;
  }
  $('hud-speed').textContent = nav.isFlying ? `Flying ${nav.speed.toPrecision(2)} m/s` : '';
};

// -- start ----------------------------------------------------------------------------------------

window.addEventListener('hashchange', restoreFromHash);
getHealth().then((h) => {
  localPaths = !!h.localPaths;
  $('path-form').hidden = !localPaths;
}).finally(async () => {
  showOpen();
  await restoreFromHash();
});

window.usdViewer = viewer; // debugging handle
window.usdGraph = graph;
