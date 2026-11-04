import { Viewer } from './viewer.js';
import {
  convertUpload, filesFromDrop, filesFromInput, getUpload, isDirectGltf, uploadFiles,
} from './upload.js';

const $ = (id) => document.getElementById(id);

const viewer = new Viewer($('viewport'));
let current = { uploadId: null, root: null };
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
    rows.push(['Mesh prims', fmtInt(s.meshPrims)]);
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

async function convertAndLoad(uploadId, root) {
  setStatus(`Converting ${root} …`, { progress: -1 });
  const report = await convertUpload(uploadId, root);
  current = { uploadId, root: report.root };
  $('root-select').value = report.root;
  history.replaceState(null, '', `#upload=${uploadId}&root=${encodeURIComponent(report.root)}`);

  setStatus(`Loading ${fmtBytes(report.glbBytes)} …`, { progress: 0 });
  await viewer.loadGlb(report.glbUrl, (p) => setStatus(`Loading ${fmtBytes(report.glbBytes)} …`, { progress: p }));
  showReport(report);
  setStatus('');
  updateClickHint();
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
      current = { uploadId: null, root: null };
      $('root-section').hidden = true;
      history.replaceState(null, '', location.pathname);
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
  const uploadId = params.get('upload');
  if (!uploadId) return;
  await run(async () => {
    const summary = await getUpload(uploadId);
    showRoots(summary);
    await convertAndLoad(uploadId, params.get('root') || summary.root);
  });
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
  if (current.uploadId) run(() => convertAndLoad(current.uploadId, e.target.value));
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

restoreFromHash();
