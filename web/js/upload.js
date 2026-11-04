// Collecting files from drag-and-drop / file inputs and talking to the server.

const IGNORED = /(^|\/)(\.DS_Store|Thumbs\.db|__MACOSX)(\/|$)|(^|\/)\._/;

/** A picked file together with its path relative to the dropped folder. */
export class PickedFile {
  constructor(file, path) {
    this.file = file;
    this.path = path.replace(/^\/+/, '');
  }
}

function readAllEntries(dirEntry) {
  const reader = dirEntry.createReader();
  return new Promise((resolve, reject) => {
    const out = [];
    const next = () => reader.readEntries((batch) => {
      if (!batch.length) return resolve(out);
      out.push(...batch);
      next(); // readEntries returns at most ~100 entries per call
    }, reject);
    next();
  });
}

async function walkEntry(entry, prefix, out) {
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej));
    out.push(new PickedFile(file, prefix + entry.name));
  } else if (entry.isDirectory) {
    const children = await readAllEntries(entry);
    for (const child of children) await walkEntry(child, `${prefix}${entry.name}/`, out);
  }
}

/** Files from a drop event, preserving folder structure. */
export async function filesFromDrop(dataTransfer) {
  const items = [...(dataTransfer.items || [])];
  const entries = items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  const out = [];
  if (entries.length) {
    for (const entry of entries) await walkEntry(entry, '', out);
  } else {
    for (const f of dataTransfer.files) out.push(new PickedFile(f, f.name));
  }
  return clean(out);
}

/** Files from an <input type=file> (with or without webkitdirectory). */
export function filesFromInput(input) {
  return clean([...input.files].map((f) => new PickedFile(f, f.webkitRelativePath || f.name)));
}

function clean(files) {
  return files.filter((f) => !IGNORED.test(f.path));
}

export function isDirectGltf(files) {
  return files.length === 1 && /\.(glb|gltf)$/i.test(files[0].path);
}

/** Upload with progress. Resolves to the server's upload summary. */
export function uploadFiles(files, onProgress) {
  const form = new FormData();
  for (const f of files) {
    form.append('paths', f.path);
    form.append('files', f.file, f.file.name);
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/uploads');
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(errorText(xhr.response, xhr.status)));
    };
    xhr.onerror = () => reject(new Error('Network error while uploading'));
    xhr.send(form);
  });
}

export async function getUpload(uploadId) {
  const res = await fetch(`/api/uploads/${encodeURIComponent(uploadId)}`);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorText(body, res.status));
  return body;
}

export async function convertUpload(uploadId, root) {
  const res = await fetch(`/api/uploads/${encodeURIComponent(uploadId)}/convert`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root }),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorText(body, res.status));
  return body;
}

function errorText(body, status) {
  const detail = body && body.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ');
  return `Server error (${status})`;
}
