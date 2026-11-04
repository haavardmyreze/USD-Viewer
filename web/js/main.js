import { Viewer } from './viewer.js';
import {
  convertUpload, filesFromDrop, filesFromInput, getHealth, getUpload, isDirectGltf, openPath, uploadFiles,
} from './upload.js';

const $ = (id) => document.getElementById(id);

const viewer = new Viewer($('viewport'));
// source: 'upload' (files sent to the server) or 'path' (read in place on the server machine)
let current = { source: null, uploadId: null, path: null, root: null };
let purpose = 'render';
let localPaths = false;
let busy = false;

// -- status ------------------------------------------------------------------

function setStatus(text, { error = false, progress = null } = {}) {
  $('status-section').hidden = !text && progress === null;
  $('status').textContent = text || '';
  $('status').classList.toggle('error', error);
  const bar = $('progress');
  bar.hidden = progress === null;
  if (progress !== null) {
    if (progress < 0) bar.removeAttribute('value'); // indeterminate
    else bar.value = progress;
  }
}

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

const fmtInt = (n) => Math.round(n).toLocaleString();

function fillList(detailsId, countId, listId, items) {
  $(detailsId).hidden = !items || items.length === 0;
  if (!items) return;
  $(countId).textContent = items.length;
  const ul = $(listId);
  ul.replaceChildren(...items.map((t) => {
    const li = document.createElement('li');
    li.textContent = t;
    return li;
  }));
}

function showReport(report) {
  const s = report ? report.stats : null;
  const view = viewer.stats();
  const rows = [];
  if (report) {
    rows.push(['Root', report.root]);
    rows.push(['Layers', fmtInt(report.layers.length)]);
    rows.push(['Geometry', report.purpose === 'proxy' ? 'proxy purpose' : 'render purpose']);
    if (s.pointInstancers) {
      rows.push(['Point instancers', fmtInt(s.pointInstancers)]);
    }
    rows.push(['Mesh instances', fmtInt(s.meshPrims)]);
    rows.push(['Unique meshes', fmtInt(s.uniqueMeshes)]);
    rows.push(['Triangles', fmtInt(s.triangles)]);
    rows.push(['Materials', fmtInt(s.materials)]);
    rows.push(['Textures', fmtInt(s.textures)]);
    rows.push(['Up axis / units', `${report.upAxis} · ${report.metersPerUnit} m`]);
    rows.push(['GLB size', fmtBytes(report.glbBytes)]);
    rows.push(['Convert time', `${report.seconds.toFixed(2)} s`]);
  } else {
    rows.push(['Meshes', fmtInt(view.meshes)]);
    rows.push(['Triangles', fmtInt(view.triangles)]);
  }
  rows.push(['Scene size', `${view.size.x.toFixed(1)} × ${view.size.y.toFixed(1)} × ${view.size.z.toFixed(1)} m`]);

  $('stats').replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    dd.title = v;
    return [dt, dd];
  }));
  $('info-section').hidden = false;

  fillList('unresolved-details', 'unresolved-count', 'unresolved-list', report?.unresolved);
  const warnings = report ? [...report.compositionErrors, ...report.warnings] : [];
  fillList('warnings-details', 'warnings-count', 'warnings-list', warnings);
  fillList('layers-details', 'layers-count', 'layers-list', report?.layers);
  showMissingNotice(report);
}

function showMissingNotice(report) {
  const missing = report?.unresolved?.length || 0;
  $('missing-section').hidden = missing === 0;
  if (!missing) return;
  const n = `${missing} file${missing === 1 ? '' : 's'}`;
  if (report.source === 'upload') {
    $('missing-text').textContent = `${n} referenced by the scene ${missing === 1 ? 'is' : 'are'} outside what was `
      + 'dropped. The browser can only send what you give it, so drop the folder that contains everything '
      + '(e.g. the project root)'
      + (localPaths ? ", or paste the root layer's path above to read it in place." : '.');
  } else {
    $('missing-text').textContent = `${n} referenced by the scene could not be found on disk. `
      + 'See Missing assets below.';
  }
}

function showRoots(summary) {
  const sel = $('root-select');
  const options = summary.usdFiles.length > 1 ? summary.usdFiles : [];
  $('root-section').hidden = options.length === 0;
  const candidates = new Set(summary.rootCandidates);
  sel.replaceChildren(...[...summary.rootCandidates, ...options.filter((o) => !candidates.has(o))].map((f) => {
    const o = document.createElement('option');
    o.value = f;
    o.textContent = candidates.has(f) ? f : `${f}  (referenced by another layer)`;
    return o;
  }));
}

// -- loading -----------------------------------------------------------------

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
  history.replaceState(null, '', `#${params}`);
}

async function loadReport(report) {
  updateHash();
  setStatus(`Loading ${fmtBytes(report.glbBytes)} …`, { progress: 0 });
  await viewer.loadGlb(report.glbUrl, (p) => setStatus(`Loading ${fmtBytes(report.glbBytes)} …`, { progress: p }));
  showReport(report);
  setStatus('');
  updateClickHint();
}

async function convertAndLoad(uploadId, root) {
  setStatus(`Converting ${root} …`, { progress: -1 });
  const report = await convertUpload(uploadId, root, purpose);
  current = { source: 'upload', uploadId, path: null, root: report.root };
  $('root-select').value = report.root;
  await loadReport(report);
}

async function openPathAndLoad(path, root = null) {
  setStatus(`Converting ${root || path} …`, { progress: -1 });
  const report = await openPath(path, root, purpose);
  current = { source: 'path', uploadId: null, path, root: report.folder ? report.root : null };
  if (report.folder) {
    showRoots({ usdFiles: report.rootCandidates, rootCandidates: report.rootCandidates });
    $('root-select').value = report.root;
  } else {
    $('root-section').hidden = true;
  }
  $('path-input').value = path;
  await loadReport(report);
}

/** Convert the current scene again (e.g. after switching purpose). */
function reloadCurrent() {
  if (current.source === 'upload') return run(() => convertAndLoad(current.uploadId, current.root));
  if (current.source === 'path') return run(() => openPathAndLoad(current.path, current.root));
  return undefined;
}

async function run(task) {
  if (busy) return;
  busy = true;
  try {
    await task();
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), { error: true });
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
        setStatus(`Loading ${files[0].path} …`, { progress: -1 });
        await viewer.loadGlb(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      current = { source: null, uploadId: null, path: null, root: null };
      $('root-section').hidden = true;
      updateHash();
      showReport(null);
      setStatus('');
      updateClickHint();
      return;
    }

    const total = files.reduce((n, f) => n + f.file.size, 0);
    const label = `Uploading ${files.length} file${files.length === 1 ? '' : 's'} (${fmtBytes(total)})`;
    setStatus(`${label} …`, { progress: 0 });
    const summary = await uploadFiles(files, (p) => setStatus(`${label} …`, { progress: p }));
    showRoots(summary);
    await convertAndLoad(summary.uploadId, summary.root);
  });
}

async function restoreFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  purpose = params.get('purpose') === 'proxy' ? 'proxy' : 'render';
  $('opt-proxy').checked = purpose === 'proxy';
  const uploadId = params.get('upload');
  const path = params.get('path');
  if (path) {
    await run(() => openPathAndLoad(path, params.get('root')));
  } else if (uploadId) {
    await run(async () => {
      const summary = await getUpload(uploadId);
      showRoots(summary);
      await convertAndLoad(uploadId, params.get('root') || summary.root);
    });
  }
}

// -- UI wiring -----------------------------------------------------------------

$('open-folder').addEventListener('click', () => $('folder-input').click());
$('open-files').addEventListener('click', () => $('files-input').click());
for (const id of ['folder-input', 'files-input']) {
  $(id).addEventListener('change', (e) => {
    openFiles(filesFromInput(e.target));
    e.target.value = '';
  });
}

$('root-select').addEventListener('change', (e) => {
  if (current.source === 'upload') run(() => convertAndLoad(current.uploadId, e.target.value));
  else if (current.source === 'path') run(() => openPathAndLoad(current.path, e.target.value));
});

$('path-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const path = $('path-input').value.trim().replace(/^"(.*)"$/, '$1');
  if (path) run(() => openPathAndLoad(path));
  $('path-input').blur();
});

$('opt-proxy').addEventListener('change', (e) => {
  if (busy) {
    e.target.checked = purpose === 'proxy';
    return;
  }
  purpose = e.target.checked ? 'proxy' : 'render';
  if (!reloadCurrent()) updateHash();
});

$('toggle-panel').addEventListener('click', togglePanel);
function togglePanel() {
  const collapsed = $('panel').classList.toggle('collapsed');
  $('toggle-panel').innerHTML = collapsed ? '+' : '&minus;';
}

$('opt-grid').addEventListener('change', (e) => viewer.setGrid(e.target.checked));
$('opt-double').addEventListener('change', (e) => viewer.setDoubleSided(e.target.checked));

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox') return;
  if (e.target.tagName === 'SELECT' || e.ctrlKey || e.metaKey) return;
  switch (e.code) {
    case 'KeyF': viewer.frameScene(); break;
    case 'KeyG': $('opt-grid').click(); break;
    case 'KeyB': $('opt-double').click(); break;
    case 'KeyI': togglePanel(); break;
    case 'KeyP': $('opt-proxy').click(); break;
    default: return;
  }
  // keep focus off the toggled checkbox so Space doesn't re-toggle it
  e.target.blur?.();
});

// drag & drop anywhere
let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth += 1;
  $('drop-overlay').hidden = false;
});
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) $('drop-overlay').hidden = true;
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('drop-overlay').hidden = true;
  openFiles(await filesFromDrop(e.dataTransfer));
});

// HUD
function updateClickHint() {
  $('click-hint').hidden = !viewer.model || viewer.controls.locked;
}
viewer.controls.addEventListener('lockchange', updateClickHint);

const hudSpeed = $('hud-speed');
const hudPos = $('hud-pos');
let lastHud = 0;
viewer.onFrame = () => {
  const now = performance.now();
  if (now - lastHud < 100) return;
  lastHud = now;
  const p = viewer.camera.position;
  hudSpeed.textContent = `speed ${viewer.controls.speed.toPrecision(3)} m/s`;
  hudPos.textContent = `${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}`;
};

// debugging handle
window.usdViewer = viewer;

// a pasted #path=… / #upload=… link (replaceState from this page doesn't fire this)
window.addEventListener('hashchange', restoreFromHash);

getHealth().then((h) => {
  localPaths = !!h.localPaths;
  $('path-section').hidden = !localPaths;
}).finally(restoreFromHash);
