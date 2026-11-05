// The info card: who made what you just clicked, and what state it is in.
//
// Clicking geometry in the view opens a small floating card beside the cursor.
// It reads the pipeline's own bookkeeping, the way usd-refgraph does: every
// published layer carries a record in customLayerData (artist, status, comment,
// workfile, ROP, publish time), and the naming convention says which asset a
// layer belongs to and which block of it. The server (GET .../info) walks up
// from the prim to the reference that brought its asset in and returns, from the
// inside out:
//
//   the asset         env-column: its assembly and blocks, each with its record
//   where it sits     a PointInstancer's instance count, the variant picked
//   its look          the bound material (and the library it came from), textures
//   placed by         the set or shot around it, down to the stage's root layer
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

const TIER_LABEL = { asset: 'asset', set: 'set', shot: 'shot' };

const GAP = 18; // between the cursor and the card
const EDGE = 12; // from the window's edges

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

function pill(status) {
  const p = el('span', `spill spill--${status || 'unknown'}`);
  p.append(dot(status), el('span', undefined, STATUS_LABEL[status] ?? STATUS_LABEL.unknown));
  return p;
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

function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

function fmtCount(n) {
  return Number(n).toLocaleString();
}

/** `Environment asset`, `Set`, `Shot`: what an entity is, in words. */
function entityKind(entity) {
  const tier = TIER_LABEL[entity.tier];
  if (!tier) return 'Not a pipeline name';
  if (entity.tier === 'asset' && entity.category) return `${capitalize(entity.category)} ${tier}`;
  return capitalize(tier);
}

export class InfoCard {
  constructor(host, callbacks) {
    this.host = host;
    this.callbacks = callbacks;
    this.root = el('div', 'infocard');
    this.root.id = 'infocard';
    this.root.hidden = true;
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', 'Selection info');
    host.appendChild(this.root);
    this.request = 0;
    this.sel = null;
    this.anchor = null;
    this.open = new Set(); // expanded layer rows, by path, kept as the selection moves
    // a press on the card is not a click in the view
    for (const type of ['pointerdown', 'pointerup', 'wheel', 'dblclick']) {
      this.root.addEventListener(type, (e) => e.stopPropagation());
    }
    window.addEventListener('resize', () => this._place());
  }

  get visible() {
    return !this.root.hidden;
  }

  /** Show the card for a selection picked at window point `anchor` ({ x, y }). */
  async show(sceneId, sel, anchor) {
    const ticket = ++this.request;
    this.sel = sel;
    this.anchor = anchor;
    const query = new URLSearchParams({ path: sel.path });
    if (Number.isInteger(sel.instance)) query.set('instance', String(sel.instance));
    if (sel.proto) query.set('proto', sel.proto);
    // only a slow answer shows the waiting state, so a quick one does not flash
    const slow = setTimeout(() => {
      if (ticket === this.request) this._render(null, sel);
    }, 120);
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
    if (this.root.hidden || this.root.classList.contains('is-leaving')) return;
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

  // -- layout ---------------------------------------------------------------

  /** Beside the cursor, on whichever side has room, clear of the side panel. */
  _place() {
    if (this.root.hidden || !this.anchor) return;
    const panel = document.getElementById('panel');
    const right = panel && !panel.hidden ? panel.getBoundingClientRect().left - EDGE : window.innerWidth - EDGE;
    const top = 64; // under the top bar
    const bottom = window.innerHeight - EDGE;
    const w = this.root.offsetWidth;
    const h = Math.min(this.root.scrollHeight, bottom - top);
    let x = this.anchor.x + GAP;
    if (x + w > right) x = this.anchor.x - GAP - w; // flip to the left of the cursor
    x = Math.max(EDGE, Math.min(x, right - w));
    let y = this.anchor.y - 44; // the header level with the cursor
    y = Math.max(top, Math.min(y, bottom - h));
    this.root.style.left = `${Math.round(x)}px`;
    this.root.style.top = `${Math.round(y)}px`;
    this.root.style.maxHeight = `${bottom - top}px`;
    this.root.style.transformOrigin = `${x > this.anchor.x ? 'left' : 'right'} ${Math.round(this.anchor.y - y)}px`;
  }

  // -- content --------------------------------------------------------------

  _render(info, sel, error = null) {
    const wasHidden = this.root.hidden || this.root.classList.contains('is-leaving');
    const scroll = wasHidden ? 0 : this.root.scrollTop;
    this.root.classList.remove('is-leaving');
    this.root.replaceChildren();
    this.root.hidden = false;
    if (wasHidden) {
      // restart the entrance
      this.root.classList.remove('is-entering');
      void this.root.offsetWidth;
      this.root.classList.add('is-entering');
    }

    const entities = info?.entities ?? [];
    const asset = entities.find((e) => e.arc !== 'root' && e.arc !== 'sublayer') ?? null;
    this.root.appendChild(this._head(info, sel, asset ?? entities[0] ?? null));

    if (error) {
      this.root.appendChild(el('p', 'ic__empty', `Could not read this prim: ${error}`));
    } else if (!info) {
      this.root.appendChild(el('div', 'ic__loading', 'Reading the layers…'));
    } else {
      const body = el('div', 'ic__body');
      if (asset) {
        body.appendChild(this._summary(asset));
        const placement = this._placement(info, asset);
        if (placement) body.appendChild(placement);
        body.appendChild(this._layers(asset, 'Published layers'));
      } else {
        body.appendChild(el('p', 'ic__empty', 'Not part of a published asset: it is authored directly in the scene.'));
      }
      if (info.materials?.length) body.appendChild(this._materials(info.materials));
      const outer = entities.filter((e) => e !== asset);
      if (outer.length) body.appendChild(this._outer(outer));
      const issues = this._issues(info);
      if (issues) body.appendChild(issues);
      this.root.appendChild(body);
    }
    this._place();
    this.root.scrollTop = scroll;
  }

  _head(info, sel, entity) {
    const head = el('div', 'ic__head');
    const prim = info?.prim;
    const name = sel.path.slice(sel.path.lastIndexOf('/') + 1) || '/';
    head.appendChild(houdiniIcon(primIcon(prim?.typeName ?? '', prim?.kind), undefined, 'hicon ic__icon'));

    const title = el('div', 'ic__title');
    const titleText = entity?.tier ? entity.name : (Number.isInteger(sel.instance) ? `${name} [${sel.instance}]` : name);
    // non-breaking hyphens keep `env-column` in one piece when it wraps
    title.appendChild(el('div', 'ic__name', titleText.replace(/-/g, '‑')));
    const sub = [];
    if (entity?.tier) sub.push(entityKind(entity));
    if (prim) sub.push(Number.isInteger(sel.instance) ? `${prim.name} [${sel.instance}]` : prim.name);
    title.appendChild(el('div', 'ic__sub', sub.join(' · ') || (prim?.typeName ?? '')));
    title.title = sel.path;
    head.appendChild(title);

    const tools = el('div', 'ic__tools');
    tools.append(
      iconButton('target', 'Frame it (F)', () => this.callbacks.onFrame?.(this.sel)),
      iconButton('copy', 'Copy the prim path', () => this._copy(sel.path, 'Prim path copied')),
      iconButton('close', 'Close (Esc)', () => this.callbacks.onClose?.()),
    );
    head.appendChild(tools);
    return head;
  }

  /** Status, and who published it last and when, in one line. */
  _summary(entity) {
    const box = el('div', 'ic__summary');
    box.appendChild(pill(entity.status));
    const line = el('span', 'ic__who');
    if (entity.artists?.length) {
      line.append('by ', el('strong', undefined, entity.artists.join(', ')));
    }
    if (entity.lastPublished) {
      line.append(entity.artists?.length ? ' · ' : 'published ', when(entity.lastPublished));
    }
    if (line.childNodes.length) box.appendChild(line);
    return box;
  }

  /** Where this copy sits: instance counts and the variants picked for it. */
  _placement(info, asset) {
    const facts = el('dl', 'ic__facts');
    const add = (key, value) => {
      if (!value) return;
      facts.append(el('dt', undefined, key));
      const dd = el('dd');
      if (typeof value === 'string') dd.textContent = value;
      else dd.append(value);
      facts.append(dd);
    };
    const copies = info.copies;
    if (copies?.kind === 'pointinstancer') {
      // the prototype's share of the instancer's points
      const line = el('span', undefined, `${fmtCount(copies.same)} of ${fmtCount(copies.total)} instances`);
      const inst = el('span', 'ic__dim', ` · ${copies.instancer.slice(copies.instancer.lastIndexOf('/') + 1)}`);
      line.title = `Placed by the point instancer ${copies.instancer}`;
      line.append(inst);
      add('Copies', line);
    } else if (copies?.kind === 'instance') {
      add('Copies', `${fmtCount(copies.same)}${copies.truncated ? '+' : ''} instances of the same prototype`);
    }
    for (const set of asset.variants ?? []) {
      if (!set.options.length) continue;
      const v = el('span', 'ic__variant');
      v.append(el('span', 'ic__chip ic__chip--on', set.selection ?? 'none'));
      const others = set.options.filter((o) => o !== set.selection);
      if (others.length) v.append(el('span', 'ic__dim', ` of ${set.options.length}`));
      v.title = set.options.join(', ');
      add(`Variant ${set.name}`, v);
    }
    if (asset.introducedBy) add('Placed by', asset.introducedBy);
    return facts.childElementCount ? facts : null;
  }

  /** One row per published layer, assembly first; a row opens to its full record. */
  _layers(entity, heading) {
    const section = el('section', 'ic__section');
    const h = el('h3', undefined, heading);
    h.appendChild(el('span', 'ic__count', String(entity.layers.length)));
    section.appendChild(h);
    const list = el('div', 'ic__layers');
    for (const layer of entity.layers) list.appendChild(this._layerRow(layer));
    section.appendChild(list);
    return section;
  }

  _layerRow(layer) {
    const record = layer.record;
    const row = el('div', 'ic__layer');
    const isOpen = this.open.has(layer.path);
    row.classList.toggle('is-open', isOpen);

    const top = el('button', 'ic__layerTop');
    top.type = 'button';
    top.setAttribute('aria-expanded', String(isOpen));
    top.title = layer.name;
    top.appendChild(dot(record?.status));
    const label = el('span', 'ic__block', layer.role === 'assembly' ? 'assembly' : layer.block ?? layer.name);
    top.appendChild(label);
    const meta = el('span', 'ic__layerMeta');
    if (record?.artist) meta.append(record.artist);
    if (record?.exportedAt) {
      if (record.artist) meta.append(' · ');
      meta.append(when(record.exportedAt));
    }
    if (!record) meta.append(el('span', 'ic__dim', 'no publish record'));
    top.appendChild(meta);
    const caret = el('span', 'ic__caret');
    caret.appendChild(icon('chevron'));
    top.appendChild(caret);
    row.appendChild(top);

    if (record?.comment) row.appendChild(el('p', 'ic__comment', record.comment));

    const more = el('dl', 'ic__facts ic__more');
    const add = (key, value, mono = false) => {
      if (!value) return;
      more.append(el('dt', undefined, key));
      const dd = el('dd', mono ? 'is-mono' : undefined);
      if (typeof value === 'string') dd.textContent = value;
      else dd.append(value);
      more.append(dd);
    };
    add('Status', record ? STATUS_LABEL[record.status] + (record.statusRaw ? ` (“${record.statusRaw}”)` : '') : null);
    add('Published', record?.exportedAt ? moment(record.exportedAt) : null);
    add('Workfile', record?.hipFile, true);
    add('ROP', record?.ropPath, true);
    for (const [key, value] of Object.entries(record?.extra ?? {})) add(key, value);
    const file = el('span', 'ic__file');
    const path = el('span', 'ic__path', layer.path);
    path.title = layer.path;
    file.append(path, iconButton('copy', 'Copy the file path', () => this._copy(layer.path, 'File path copied')));
    add('File', file, true);
    if (layer.modified && (!record?.exportedAt || Math.abs(layer.modified - record.exportedAt) > 120_000)) {
      add('Modified', moment(layer.modified));
    }
    row.appendChild(more);

    top.addEventListener('click', () => {
      const open = !row.classList.contains('is-open');
      row.classList.toggle('is-open', open);
      top.setAttribute('aria-expanded', String(open));
      if (open) this.open.add(layer.path);
      else this.open.delete(layer.path);
      this._place();
    });
    return row;
  }

  _materials(materials) {
    const section = el('section', 'ic__section');
    section.appendChild(el('h3', undefined, materials.length === 1 ? 'Material' : 'Materials'));
    for (const material of materials) {
      const box = el('div', 'ic__material');
      const head = el('div', 'ic__matHead');
      head.appendChild(houdiniIcon('SCENEGRAPH__primtype__material', undefined, 'hicon ic__matIcon'));
      const name = el('span', 'ic__matName', material.name);
      name.title = material.path;
      head.appendChild(name);
      if (material.entity) {
        // shared from a material library: say whose, and in what state
        const from = el('span', 'ic__from');
        from.append('from ', el('strong', undefined, material.entity.name), ' ', dot(material.entity.status));
        const tip = [STATUS_LABEL[material.entity.status]];
        if (material.entity.artists?.length) tip.push(`by ${material.entity.artists.join(', ')}`);
        if (material.entity.lastPublished) tip.push(relative(material.entity.lastPublished));
        from.title = tip.join(' · ');
        head.appendChild(from);
      }
      box.appendChild(head);
      if (material.textures.length) {
        const list = el('ul', 'ic__textures');
        for (const tex of material.textures) {
          const li = el('li', tex.exists ? undefined : 'is-missing');
          const n = el('span', 'ic__texName', tex.name);
          n.title = tex.path;
          li.appendChild(n);
          const what = [tex.channel, tex.resolution].filter(Boolean).join(' · ');
          li.appendChild(el('span', 'ic__texMeta', tex.exists ? what : 'missing'));
          list.appendChild(li);
        }
        box.appendChild(list);
      }
      section.appendChild(box);
    }
    return section;
  }

  /** What the asset sits in: the set or shot, outermost last. Folded to one line each. */
  _outer(entities) {
    const section = el('section', 'ic__section');
    section.appendChild(el('h3', undefined, 'Placed in'));
    for (const entity of entities) {
      const box = el('details', 'ic__outer');
      box.open = this.open.has(`entity:${entity.name}`);
      box.addEventListener('toggle', () => {
        if (box.open) this.open.add(`entity:${entity.name}`);
        else this.open.delete(`entity:${entity.name}`);
        this._place();
      });
      const summary = el('summary');
      summary.appendChild(dot(entity.status));
      summary.appendChild(el('span', 'ic__outerName', entity.name.replace(/-/g, '‑')));
      summary.appendChild(el('span', 'ic__dim', entityKind(entity)));
      const last = entity.layers
        .filter((l) => l.record?.exportedAt)
        .sort((a, b) => b.record.exportedAt - a.record.exportedAt)[0];
      if (last) {
        const meta = el('span', 'ic__layerMeta');
        meta.append(`${last.role === 'assembly' ? 'assembly' : last.block ?? last.name}`);
        if (last.record.artist) meta.append(` by ${last.record.artist}`);
        meta.append(' · ', when(last.record.exportedAt));
        summary.appendChild(meta);
      }
      const caret = el('span', 'ic__caret');
      caret.appendChild(icon('chevron'));
      summary.appendChild(caret);
      box.appendChild(summary);
      const list = el('div', 'ic__layers');
      for (const layer of entity.layers) list.appendChild(this._layerRow(layer));
      box.appendChild(list);
      section.appendChild(box);
    }
    return section;
  }

  /** Things worth fixing: missing textures, layers nobody signed, odd names. */
  _issues(info) {
    const items = [];
    for (const entity of info.entities) {
      for (const issue of entity.issues ?? []) items.push(issue);
    }
    for (const material of info.materials ?? []) {
      const missing = material.textures.filter((t) => !t.exists);
      if (missing.length) {
        items.unshift({ level: 'bad', text: `${material.name}: ${missing.length} missing ${missing.length === 1 ? 'texture' : 'textures'}` });
      }
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

  async _copy(text, done) {
    try {
      await navigator.clipboard.writeText(text);
      this.callbacks.onToast?.(done, 'ok');
    } catch {
      this.callbacks.onToast?.('Could not copy to the clipboard');
    }
  }
}
