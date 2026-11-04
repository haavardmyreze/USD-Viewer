import * as THREE from 'three';
import { Viewer } from './viewer.js';
import { SceneGraph } from './scenegraph.js';
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

function list(ul, items) {
  ul.replaceChildren(...items.map((t) => {
    const li = document.createElement('li');
    li.textContent = t;
    return li;
  }));
}

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
  $('inspector').hidden = false;
  $('btn-graph').hidden = current.source === 'file';
  if (current.source !== 'file' && store.get('usdv.graph', false) && $('graph').hidden) setGraph(true);
  if (current.source === 'file' && !$('graph').hidden) setGraph(false);
  $('hud').hidden = false;
  if (!store.get('usdv.hintSeen', false)) $('nav-hint').hidden = false;
}

function setInspector(open) {
  document.body.classList.toggle('no-inspector', !open);
  $('btn-inspector').setAttribute('aria-pressed', String(open));
  store.set('usdv.inspector', open);
}

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

// -- report -> inspector ----------------------------------------------------------------

function showReport(report) {
  const s = report?.stats;
  const view = viewer.stats();

  const name = report ? basename(report.root) : current.path;
  $('scene-name').textContent = name || 'Scene';
  const sub = [];
  if (report) {
    sub.push(report.source === 'path' ? 'Opened in place' : 'Uploaded');
    sub.push(report.purpose === 'proxy' ? 'proxy geometry' : 'render geometry');
  } else {
    sub.push('glTF file');
  }
  $('scene-sub').textContent = sub.join(' · ');
  $('scene-path').textContent = report?.source === 'path' ? report.path : (report?.root || '');
  $('scene-path').hidden = !$('scene-path').textContent;
  document.title = `${name} · USD Viewer`;

  const rows = [];
  rows.push(['Triangles', fmtCount(view.fullTriangles || s?.triangles || 0)]);
  if (s) {
    if (s.instances) rows.push(['Instances', fmtCount(s.meshPrims)]);
    else rows.push(['Mesh prims', fmtCount(s.meshPrims)]);
    if (s.pointInstancers) rows.push(['Point instancers', fmtCount(s.pointInstancers)]);
    rows.push(['Unique meshes', fmtCount(s.uniqueMeshes)]);
    rows.push(['Materials · textures', `${s.materials} · ${s.textures}`]);
  }
  const size = view.size;
  rows.push(['Size', `${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)} m`]);
  if (report) {
    rows.push(['Up axis · units', `${report.upAxis} · ${report.metersPerUnit} m`]);
    rows.push(['Layers', fmtCount(report.layers.length)]);
    rows.push(['Converted', `${report.seconds.toFixed(1)} s · ${fmtBytes(report.glbBytes)}`]);
  }
  $('stats').replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    dd.title = v;
    return [dt, dd];
  }));

  // issues
  const missing = report?.unresolved || [];
  const warnings = report ? [...report.compositionErrors, ...report.warnings] : [];
  $('issues-block').hidden = missing.length + warnings.length === 0;
  $('issues-count').textContent = missing.length + warnings.length;
  $('missing-details').hidden = missing.length === 0;
  $('missing-count').textContent = missing.length;
  list($('missing-list'), missing);
  $('warnings-details').hidden = warnings.length === 0;
  $('warnings-count').textContent = warnings.length;
  list($('warnings-list'), warnings);
  $('warnings-details').open = missing.length === 0 && warnings.length <= 3;
  $('issues-dot').hidden = missing.length + warnings.length === 0;
  $('issues-dot').classList.toggle('bad', missing.length > 0);

  const explain = $('missing-explain');
  explain.hidden = missing.length === 0;
  if (report?.source === 'upload') {
    explain.textContent = 'These files are outside what was dropped. The browser can only send what you give it: '
      + 'drop the folder that contains everything (e.g. the project root)'
      + (localPaths ? ", or open the root layer's path to read it in place." : '.');
  } else {
    explain.textContent = 'These files are referenced by the scene but do not exist on disk.';
  }
  if (missing.length && store.get('usdv.inspector', true) === false) setInspector(true);

  const layers = report?.layers || [];
  $('layers-block').hidden = layers.length === 0;
  $('layers-count').textContent = layers.length;
  list($('layers-list'), layers);
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
    o.textContent = candidates.includes(f) ? f : `${f}  (not an assembly)`;
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

function setGraph(open) {
  $('graph').hidden = !open;
  document.body.classList.toggle('graph-open', open);
  $('btn-graph').setAttribute('aria-pressed', String(open));
  store.set('usdv.graph', open);
  if (open) graph.activate();
  else graph.deactivate();
}

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
$('opt-proxy').addEventListener('change', () => {
  $('opt-proxy').checked = purpose === 'proxy'; // togglePurpose decides
  togglePurpose();
});

function setDetail(value) {
  viewer.setDetail(value);
  for (const b of $('detail').querySelectorAll('button')) {
    b.setAttribute('aria-checked', String(b.dataset.value === value));
  }
  saveSetting('detail', value);
}
$('detail').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) setDetail(b.dataset.value);
});
setDetail(settings.detail || 'balanced');

$('opt-exposure').value = settings.exposure ?? 0;
viewer.setExposure(2 ** Number($('opt-exposure').value));
$('opt-exposure').addEventListener('input', (e) => {
  viewer.setExposure(2 ** Number(e.target.value));
  saveSetting('exposure', Number(e.target.value));
});
$('opt-exposure').addEventListener('dblclick', (e) => {
  e.target.value = 0;
  e.target.dispatchEvent(new Event('input'));
});

setInspector(store.get('usdv.inspector', true));

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
$('btn-graph').addEventListener('click', () => setGraph($('graph').hidden));
$('graph-close').addEventListener('click', () => setGraph(false));

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
$('btn-inspector').addEventListener('click', () => setInspector(document.body.classList.contains('no-inspector')));
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
    case 'KeyT': if (graph.sceneId) setGraph($('graph').hidden); break;

    case 'KeyG': click('opt-grid'); break;
    case 'KeyL': click('opt-shadows'); break;
    case 'KeyB': click('opt-double'); break;
    case 'KeyP': togglePurpose(); break;
    case 'KeyI': click('btn-inspector'); break;
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
  const s = viewer.stats();
  $('hud-fps').textContent = fps >= 3 ? `${Math.round(fps)} fps` : 'still';
  $('hud-tris').textContent = `${fmtCount(s.drawnTriangles)} tris drawn`;
  $('hud-speed').textContent = nav.isFlying ? `fly ${nav.speed.toPrecision(2)} m/s` : '';
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
