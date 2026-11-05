// The info card: who made what you just clicked, and what state it is in.
//
// Clicking geometry in the view opens a small card docked at the top left,
// opposite the side panel, so it never covers what was clicked. It reads the
// pipeline's own bookkeeping, the way usd-refgraph does: every published layer
// carries a record in customLayerData (artist, status, comment, workfile, ROP,
// publish time), and the naming convention says which asset a layer belongs to
// and which block of it. The server (GET .../info) walks up from the prim to
// the reference that brought its asset in and returns, from the inside out, the
// asset, its material, and the set or shot that placed it.
//
// The card says the few things you look for at a glance (what it is, its
// status, who touched it last and when) and keeps the rest one click away:
// a layer row opens onto its full record, the material onto its textures.
//
// Status reads as in usd-refgraph: amber placeholder, blue production ready,
// green locked; an entity is only as finished as its least finished layer.

import { houdiniIcon, primIcon } from './houdini.js';

const ICONS = {
  target: '<circle cx="8" cy="8" r="5.6"/><circle cx="8" cy="8" r="1.6"/><path d="M8 .8v2.4M8 12.8v2.4M.8 8h2.4M12.8 8h2.4"/>',
  copy: '<rect x="5.4" y="5.4" width="8" height="8" rx="1.5"/><path d="M10.6 5.4V4A1.4 1.4 0 0 0 9.2 2.6H4A1.4 1.4 0 0 0 2.6 4v5.2A1.4 1.4 0 0 0 4 10.6h1.4"/>',
  close: '<path d="m4.5 4.5 7 7M11.5 4.5l-7 7"/>',
  chevron: '<path d="m6 3.5 4.5 4.5L6 12.5"/>',
  alert: '<path d="M8 2.2 14.4 13.4H1.6z"/><path d="M8 6.6v3.2M8 11.7h.01"/>',
};

export const STATUS_LABEL = {
  placeholder: 'Placeholder',
  production_ready: 'Production ready',
  locked: 'Locked',
  unknown: 'No status',
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function icon(name) {
  const span = document.createElement('span');
  span.innerHTML = `<svg class="rg-icon" viewBox="0 0 16 16" aria-hidden="true">${ICONS[name]}</svg>`;
  return span.firstChild;
}

function iconButton(name, title, onClick) {
  const b = el('button', 'iconbtn');
  b.type = 'button';
  b.title = title;
  b.setAttribute('aria-label', title);
  b.appendChild(icon(name));
  b.addEventListener('click', onClick);
  return b;
}

function dot(status) {
  const d = el('span', `sdot sdot--${status || 'unknown'}`);
  d.title = STATUS_LABEL[status] ?? STATUS_LABEL.unknown;
  return d;
}

/** `5 min ago`, `yesterday`, `3 days ago`: as usd-refgraph words it. */
export function relative(ms) {
  if (!ms) return '';
  const elapsed = Date.now() - ms;
  if (elapsed >= 0 && elapsed < 60_000) return 'just now';
  if (elapsed >= 0 && elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)} min ago`;
  if (elapsed >= 0 && elapsed < 86_400_000) {
    const h = Math.floor(elapsed / 3_600_000);
    return `${h} ${h === 1 ? 'hour' : 'hours'} ago`;
  }
  const day = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(new Date()) - day(new Date(ms))) / 86_400_000);
  if (days <= 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 30) return `${days} days ago`;
  if (days < 365) return `${Math.round(days / 30)} months ago`;
  return `${Math.round(days / 365)} years ago`;
}

function moment(ms) {
  return new Date(ms).toLocaleString(undefined, {
    day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function when(ms) {
  const t = el('time', 'when', relative(ms));
  t.dateTime = new Date(ms).toISOString();
  t.title = moment(ms);
  return t;
}

const capitalize = (text) => (text ? text[0].toUpperCase() + text.slice(1) : text);
const fmtCount = (n) => Number(n).toLocaleString();
const leaf = (path) => path.slice(path.lastIndexOf('/') + 1) || '/';
/** Non-breaking hyphens keep `env-column` in one piece when it wraps. */
const nobreak = (text) => text.replace(/-/g, '‑');

/** `Environment asset`, `Set`, `Shot`: what an entity is, in words. */
function entityKind(entity) {
  if (!entity.tier) return null;
  if (entity.tier === 'asset' && entity.category) return `${capitalize(entity.category)} asset`;
  return capitalize(entity.tier);
}

/** A layer's short name: `assembly`, or its block (`model`, `lookdev`). */
function blockName(layer) {
  return layer.role === 'assembly' ? 'assembly' : layer.block ?? layer.name;
}

/** The record published most recently among an entity's layers. */
function latest(entity) {
  return entity.layers
    .map((layer) => layer.record)
    .filter((record) => record?.exportedAt)
    .sort((a, b) => b.exportedAt - a.exportedAt)[0] ?? null;
}

export class InfoCard {
  constructor(host, callbacks) {
    this.callbacks = callbacks;
    this.root = el('aside', 'infocard');
    this.root.id = 'infocard';
    this.root.hidden = true;
    this.root.setAttribute('aria-label', 'Selection info');
    this.root.setAttribute('aria-live', 'polite');
    host.appendChild(this.root);
    this.request = 0;
    this.sel = null;
    this.open = new Set(); // what is unfolded, kept as the selection moves
    // a press on the card is not a click in the view
    for (const type of ['pointerdown', 'pointerup', 'wheel', 'dblclick']) {
      this.root.addEventListener(type, (e) => e.stopPropagation());
    }
  }

  get visible() {
    return !this.root.hidden && !this.root.classList.contains('is-leaving');
  }

  /** Show the card for a selection ({ path, instance, proto }). */
  async show(sceneId, sel) {
    const ticket = ++this.request;
    this.sel = sel;
    const query = new URLSearchParams({ path: sel.path });
    if (Number.isInteger(sel.instance)) query.set('instance', String(sel.instance));
    if (sel.proto) query.set('proto', sel.proto);
    // only a slow answer shows the waiting state, so a quick one does not flash
    const slow = setTimeout(() => {
      if (ticket === this.request) this._render(null, sel);
    }, 150);
    let info = null;
    let error = null;
    try {
      const res = await fetch(`/api/scenes/${encodeURIComponent(sceneId)}/info?${query}`);
      if (!res.ok) {
        let msg = `${res.status} ${res.statusText}`;
        try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
        throw new Error(msg);
      }
      info = await res.json();
    } catch (err) {
      error = err.message || String(err);
    } finally {
      clearTimeout(slow);
    }
    if (ticket !== this.request) return;
    this._render(info, sel, error);
  }

  hide() {
    this.request++;
    this.sel = null;
    if (!this.visible) return;
    this.root.classList.add('is-leaving');
    const done = () => {
      if (!this.root.classList.contains('is-leaving')) return; // shown again meanwhile
      this.root.classList.remove('is-leaving');
      this.root.hidden = true;
      this.root.replaceChildren();
    };
    this.root.addEventListener('animationend', done, { once: true });
    setTimeout(done, 220); // reduced motion: no animation to end
  }

  // -- content --------------------------------------------------------------

  _render(info, sel, error = null) {
    const entering = !this.visible;
    this.root.classList.remove('is-leaving');
    this.root.replaceChildren();
    this.root.hidden = false;
    if (entering) {
      this.root.classList.remove('is-entering');
      void this.root.offsetWidth; // restart the entrance
      this.root.classList.add('is-entering');
    }

    const entities = info?.entities ?? [];
    const asset = entities.find((e) => e.arc !== 'root' && e.arc !== 'sublayer') ?? null;
    const outer = entities.filter((e) => e !== asset);
    this.root.appendChild(this._head(info, sel, asset));

    if (error) {
      this.root.appendChild(el('p', 'ic__note', `Could not read this prim: ${error}`));
      return;
    }
    if (!info) {
      this.root.appendChild(el('p', 'ic__note ic__note--wait', 'Reading the layers…'));
      return;
    }

    const body = el('div', 'ic__body');
    if (asset) {
      body.appendChild(this._status(asset));
      const where = this._where(info, asset);
      if (where) body.appendChild(where);
      body.appendChild(this._layers(asset));
    } else {
      body.appendChild(el('p', 'ic__note', 'Authored directly in the scene, not part of a published asset.'));
    }
    const facts = this._facts(info, outer);
    if (facts) body.appendChild(facts);
    const issues = this._issues(info);
    if (issues) body.appendChild(issues);
    this.root.appendChild(body);
  }

  /** The asset's name and what it is; the prim itself only as a tooltip. */
  _head(info, sel, asset) {
    const head = el('div', 'ic__head');
    const prim = info?.prim;
    head.appendChild(houdiniIcon(primIcon(prim?.typeName ?? '', prim?.kind), undefined, 'hicon ic__icon'));

    const title = el('div', 'ic__title');
    const primName = Number.isInteger(sel.instance) ? `${leaf(sel.path)} [${sel.instance}]` : leaf(sel.path);
    title.appendChild(el('div', 'ic__name', nobreak(asset?.tier ? asset.name : primName)));
    const kind = asset ? entityKind(asset) : prim?.typeName;
    if (kind) title.appendChild(el('div', 'ic__sub', kind));
    title.title = sel.path;
    head.appendChild(title);

    const tools = el('div', 'ic__tools');
    tools.append(
      iconButton('target', 'Frame it (F)', () => this.callbacks.onFrame?.(this.sel)),
      iconButton('close', 'Close (Esc)', () => this.callbacks.onClose?.()),
    );
    head.appendChild(tools);
    return head;
  }

  /** The one line read first: status, who published last, and when. */
  _status(entity) {
    const line = el('div', 'ic__status');
    const status = el('span', `ic__state ic__state--${entity.status}`);
    status.append(dot(entity.status), el('span', undefined, STATUS_LABEL[entity.status]));
    line.appendChild(status);
    const last = latest(entity);
    if (last) {
      const who = el('span', 'ic__who');
      if (last.artist) who.append(last.artist, ' · ');
      who.append(when(last.exportedAt));
      line.appendChild(who);
    }
    return line;
  }

  /** Which variant, and how many copies: one quiet line. */
  _where(info, asset) {
    const parts = [];
    for (const set of asset.variants ?? []) {
      if (set.selection && set.options.length > 1) parts.push(`${set.name} ${set.selection}`);
    }
    const copies = info.copies;
    if (copies?.kind === 'pointinstancer') parts.push(`${fmtCount(copies.same)} of ${fmtCount(copies.total)} instances`);
    else if (copies?.kind === 'instance') parts.push(`${fmtCount(copies.same)}${copies.truncated ? '+' : ''} instances`);
    if (!parts.length) return null;
    const line = el('div', 'ic__where', parts.join(' · '));
    if (copies?.instancer) line.title = `Placed by the point instancer ${copies.instancer}`;
    return line;
  }

  /** One row per published layer, assembly first; a row opens to its record. */
  _layers(entity) {
    const list = el('div', 'ic__layers');
    // the artist goes on each row only when more than one person worked on it
    const showArtist = (entity.artists?.length ?? 0) > 1;
    for (const layer of entity.layers) list.appendChild(this._layerRow(layer, showArtist));
    return list;
  }

  _layerRow(layer, showArtist) {
    const record = layer.record;
    const key = `layer:${layer.path}`;
    const row = el('details', 'ic__layer');
    row.open = this.open.has(key);
    row.addEventListener('toggle', () => {
      if (row.open) this.open.add(key);
      else this.open.delete(key);
    });

    const summary = el('summary');
    summary.title = layer.name;
    summary.appendChild(dot(record?.status));
    summary.appendChild(el('span', 'ic__block', blockName(layer)));
    const meta = el('span', 'ic__meta');
    if (!record) meta.append('no record');
    else {
      if (showArtist && record.artist) meta.append(record.artist, record.exportedAt ? ' · ' : '');
      if (record.exportedAt) meta.append(when(record.exportedAt));
    }
    summary.appendChild(meta);
    summary.appendChild(this._caret());
    row.appendChild(summary);

    const more = el('div', 'ic__more');
    if (record?.comment) more.appendChild(el('p', 'ic__comment', record.comment));
    const facts = el('dl', 'ic__facts');
    const add = (label, value, mono = false) => {
      if (!value) return;
      facts.append(el('dt', undefined, label));
      const dd = el('dd', mono ? 'is-mono' : undefined);
      if (typeof value === 'string') dd.textContent = value;
      else dd.append(value);
      facts.append(dd);
    };
    if (record) add('Status', STATUS_LABEL[record.status] + (record.statusRaw ? ` (“${record.statusRaw}”)` : ''));
    add('By', record?.artist);
    add('Published', record?.exportedAt ? moment(record.exportedAt) : null);
    add('Workfile', record?.hipFile, true);
    add('ROP', record?.ropPath, true);
    for (const [k, v] of Object.entries(record?.extra ?? {})) add(k, v);
    const file = el('span', 'ic__file');
    const path = el('span', 'ic__path', layer.name);
    path.title = layer.path;
    file.append(path, iconButton('copy', 'Copy the file path', () => this._copy(layer.path, 'File path copied')));
    add('File', file, true);
    more.appendChild(facts);
    row.appendChild(more);
    return row;
  }

  /** Material and set, one line each; the material opens onto its textures. */
  _facts(info, outer) {
    const rows = [];
    for (const material of info.materials ?? []) rows.push(this._materialRow(material));
    for (const entity of outer) rows.push(this._outerRow(entity));
    if (!rows.length) return null;
    const box = el('div', 'ic__rows');
    box.append(...rows);
    return box;
  }

  _materialRow(material) {
    const key = `material:${material.path}`;
    const row = el('details', 'ic__row');
    row.open = this.open.has(key);
    row.addEventListener('toggle', () => {
      if (row.open) this.open.add(key);
      else this.open.delete(key);
    });
    const summary = el('summary');
    summary.appendChild(el('span', 'ic__label', 'Material'));
    const value = el('span', 'ic__value');
    value.append(material.name);
    if (material.entity) {
      // shared from a material library: say which, and in what state
      value.append(el('span', 'ic__from', ` · ${material.entity.name}`), dot(material.entity.status));
    }
    value.title = material.path;
    summary.appendChild(value);
    if (material.textures.length) summary.appendChild(this._caret());
    row.appendChild(summary);

    if (material.textures.length) {
      const list = el('ul', 'ic__textures');
      for (const tex of material.textures) {
        const li = el('li', tex.exists ? undefined : 'is-missing');
        const name = el('span', 'ic__texName', tex.name);
        name.title = tex.path;
        li.append(name, el('span', 'ic__texMeta', tex.exists ? (tex.channel ?? '') : 'missing'));
        list.appendChild(li);
      }
      row.appendChild(list);
    }
    return row;
  }

  _outerRow(entity) {
    const row = el('div', 'ic__row');
    const line = el('div', 'ic__line');
    line.appendChild(el('span', 'ic__label', entityKind(entity) ?? 'Scene'));
    const value = el('span', 'ic__value');
    value.append(nobreak(entity.name), dot(entity.status));
    const last = latest(entity);
    if (last) value.append(el('span', 'ic__from', ' · '), when(last.exportedAt));
    value.title = entity.layers.map((l) => `${l.name}${l.record?.artist ? ` (${l.record.artist})` : ''}`).join('\n');
    line.appendChild(value);
    row.appendChild(line);
    return row;
  }

  /** Only what needs fixing: missing textures and layers nobody signed. */
  _issues(info) {
    const items = [];
    for (const material of info.materials ?? []) {
      const missing = material.textures.filter((t) => !t.exists).length;
      if (missing) items.push({ level: 'bad', text: `${missing} missing ${missing === 1 ? 'texture' : 'textures'}` });
    }
    for (const entity of info.entities) {
      for (const issue of entity.issues ?? []) if (issue.level !== 'info') items.push(issue);
    }
    if (!items.length) return null;
    const list = el('ul', 'ic__issues');
    for (const item of items) {
      const li = el('li', `ic__issue ic__issue--${item.level}`);
      li.append(icon('alert'), el('span', undefined, item.text));
      list.appendChild(li);
    }
    return list;
  }

  _caret() {
    const caret = el('span', 'ic__caret');
    caret.appendChild(icon('chevron'));
    return caret;
  }

  async _copy(text, done) {
    try {
      await navigator.clipboard.writeText(text);
      this.callbacks.onToast?.(done, 'ok');
    } catch {
      this.callbacks.onToast?.('Could not copy to the clipboard');
    }
  }
}
