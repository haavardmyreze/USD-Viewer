// The scene graph panel: the composed prim hierarchy of the open scene.
//
// Ported from usd-refgraph's Scene tab, so it reads the same way: Houdini's
// Scene Graph Tree icons, connector lines from each open prim down to its
// children, `over`s in italics, deactivated prims struck through, arcs and
// variant selections flagged on the row, and a picked prim's opinions listed
// strongest first.
//
// What the viewer adds: picking a prim highlights its geometry in the 3D view,
// and picking geometry in the 3D view opens the tree onto its prim.
//
// The tree is read a level at a time (/prims), a branch at a time for expand
// all (/subtree) and a prim's detail on demand (/prim), so a heavy stage is
// never walked whole. It opens collapsed; Expand all opens up to a budget,
// materials excepted.

import { ARC_ICON, houdiniIcon, primIcon } from './houdini.js';

const ROOT = '/';

/** Stroked 16x16 line icons, as in usd-refgraph's icons.ts. */
const ICONS = {
  chevronRight: '<path d="m6 3.5 4.5 4.5L6 12.5"/>',
  expandAll: '<path d="m4 3.4 4 3.6 4-3.6"/><path d="m4 9 4 3.6L12 9"/>',
  collapseAll: '<path d="m4 7 4-3.6L12 7"/><path d="m4 12.6 4-3.6 4 3.6"/>',
  copy: '<rect x="5.4" y="5.4" width="8" height="8" rx="1.5"/><path d="M10.6 5.4V4A1.4 1.4 0 0 0 9.2 2.6H4A1.4 1.4 0 0 0 2.6 4v5.2A1.4 1.4 0 0 0 4 10.6h1.4"/>',
  target: '<circle cx="8" cy="8" r="5.6"/><circle cx="8" cy="8" r="1.6"/><path d="M8 .8v2.4M8 12.8v2.4M.8 8h2.4M12.8 8h2.4"/>',
  close: '<path d="m4.5 4.5 7 7M11.5 4.5l-7 7"/>',
  search: '<circle cx="7" cy="7" r="4.6"/><path d="m10.6 10.6 3 3"/>',
};

const ARC_NAME = {
  root: 'local', sublayer: 'sublayer', reference: 'reference', payload: 'payload',
  variant: 'variant', inherit: 'inherit', specialize: 'specialize', relocate: 'relocate',
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
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
  b.appendChild(icon(name));
  b.title = title;
  b.setAttribute('aria-label', title);
  b.addEventListener('click', onClick);
  return b;
}

function facts() {
  const root = el('dl', 'facts');
  return {
    root,
    add(key, value, mono = false) {
      if (value === null || value === undefined || value === '') return this;
      root.appendChild(el('dt', undefined, key));
      const dd = el('dd', mono ? 'is-mono' : undefined);
      if (typeof value === 'string') dd.textContent = value;
      else dd.appendChild(value);
      root.appendChild(dd);
      return this;
    },
  };
}

class ApiError extends Error {}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new ApiError(msg);
  }
  return res.json();
}

export class SceneGraph {
  /**
   * @param host       element the panel renders into
   * @param callbacks  onSelect(selection | null), onFrame(path), onToast(msg),
   *                   geometry(selection) -> { objects, instances, triangles } | null
   */
  constructor(host, callbacks) {
    this.host = host;
    this.callbacks = callbacks;
    this.root = el('div', 'scene');
    this.host.replaceChildren(this.root);
    this.sceneId = null;
    this.state = null;
    this.root.addEventListener('keydown', (e) => this._onKey(e));
  }

  /** Show the tree of a converted scene; null when there is no USD stage (a .glb). */
  /**
   * `keep` carries what was open and selected across, for the same stage
   * re-composed (another variant), so the tree reads as the same tree.
   */
  setScene(sceneId, { keep = false } = {}) {
    if (sceneId === this.sceneId) return;
    const before = this.state;
    this.sceneId = sceneId;
    if (!keep) {
      this.query = '';
      this.results = null;
      if (this.searchInput) this.searchInput.value = '';
    }
    this.state = sceneId ? {
      levels: new Map(), pending: new Set(), errors: new Map(), open: new Set(),
      selected: null, instance: null, detail: null, primed: false, expanding: false,
    } : null;
    const s = this.state;
    if (s && keep && before) {
      for (const p of before.open) s.open.add(p);
      s.selected = before.selected;
      s.instance = before.instance;
      s.primed = before.primed;
      if (this.visible) {
        for (const p of [ROOT, ...s.open]) void this._fetchLevel(s, p);
        if (s.selected) void this._loadDetail(s, s.selected);
      }
    } else if (s && this.visible) void this._fetchLevel(s, ROOT);
    this.render();
  }

  /** The current selection, { path, instance } or null. */
  get selection() {
    const s = this.state;
    return s?.selected ? { path: s.selected, instance: s.instance } : null;
  }

  /** Disable variant buttons while a switch is converting. */
  setBusy(busy) {
    this.busy = busy;
    this.render();
  }

  /** Called when the panel is shown: the first level is fetched lazily. */
  activate() {
    this.visible = true;
    const s = this.state;
    if (s) {
      for (const p of [ROOT, ...s.open]) {
        if (!s.levels.has(p) && !s.pending.has(p)) void this._fetchLevel(s, p);
      }
      if (s.selected && !s.detail) void this._loadDetail(s, s.selected);
    }
    this.render();
  }

  deactivate() {
    this.visible = false;
  }

  /** Select a prim picked in the 3D view (or null), opening the tree onto it. */
  async reveal(path, instance = null) {
    const s = this.state;
    if (!s) return;
    if (!path) {
      this._setSelected(null);
      return;
    }
    s.selected = path;
    s.instance = instance;
    s.detail = null;
    this.render();
    void this._loadDetail(s, path);
    try {
      const out = await getJson(this._url('reveal', path));
      if (s !== this.state) return;
      for (const level of out.levels) {
        // keep a level that was already listed in full; add the path's prim if it was cut
        const had = s.levels.get(level.primPath);
        if (!had) s.levels.set(level.primPath, level);
        else {
          for (const child of level.children) {
            if (!had.children.some((c) => c.path === child.path)) had.children.push(child);
          }
        }
        if (level.primPath !== ROOT) s.open.add(level.primPath);
      }
      s.primed = true;
      this.render();
      this._scrollToSelected();
    } catch (err) {
      if (s === this.state) this.callbacks.onToast?.(describe(err));
    }
  }

  // -- data -----------------------------------------------------------------

  _url(kind, path) {
    return `/api/scenes/${encodeURIComponent(this.sceneId)}/${kind}?path=${encodeURIComponent(path)}`;
  }

  async _fetchLevel(state, primPath) {
    if (state.pending.has(primPath)) return;
    state.pending.add(primPath);
    state.errors.delete(primPath);
    try {
      const level = await getJson(this._url('prims', primPath));
      state.levels.set(primPath, level);
      // the tree opens collapsed: top-level prims only, opened as you go
      if (primPath === ROOT) state.primed = true;
    } catch (err) {
      state.errors.set(primPath, describe(err));
    } finally {
      state.pending.delete(primPath);
    }
    if (state === this.state) this.render();
  }

  _toggle(prim) {
    const s = this.state;
    if (s.open.has(prim.path)) s.open.delete(prim.path);
    else {
      s.open.add(prim.path);
      if (!s.levels.has(prim.path)) void this._fetchLevel(s, prim.path);
    }
    this.render();
  }

  /**
   * Open every prim beneath `primPath`. Materials stay closed, and everything
   * inside them, unless the branch being expanded is the material itself.
   */
  async _expandAll(primPath = ROOT, { initial = false } = {}) {
    const s = this.state;
    if (!s || s.expanding) return;
    s.expanding = true;
    this.render();
    try {
      const subtree = await getJson(this._url('subtree', primPath));
      if (s !== this.state) return;
      const listed = new Set(subtree.levels.map((l) => l.primPath));
      const closed = new Set();
      for (const level of subtree.levels) {
        s.levels.set(level.primPath, level);
        s.errors.delete(level.primPath);
        const insideClosed = closed.has(level.primPath);
        for (const child of level.children) {
          if (insideClosed || child.typeName === 'Material') {
            closed.add(child.path);
            continue;
          }
          if (child.childCount === 0 || !listed.has(child.path)) continue;
          s.open.add(child.path);
        }
      }
      if (primPath !== ROOT) s.open.add(primPath);
      s.primed = true;
      if (subtree.truncated && !initial) {
        this.callbacks.onToast?.(`Expanded the first ${subtree.prims.toLocaleString()} prims. Open the rest a branch at a time.`);
      }
    } catch (err) {
      if (s === this.state) this.callbacks.onToast?.(describe(err));
    } finally {
      s.expanding = false;
    }
    if (s === this.state) this.render();
  }

  _collapseAll(primPath = ROOT) {
    const s = this.state;
    if (primPath === ROOT) s.open.clear();
    else {
      const prefix = `${primPath}/`;
      for (const p of s.open) if (p === primPath || p.startsWith(prefix)) s.open.delete(p);
    }
    this.render();
  }

  _setSelected(path) {
    const s = this.state;
    s.selected = path;
    s.instance = null;
    s.detail = null;
    this.render();
    this.callbacks.onSelect?.(path ? { path } : null);
    if (path) void this._loadDetail(s, path);
  }

  /** A click picks the prim, or puts the selection away if it is the one picked. */
  _clickPrim(path) {
    const s = this.state;
    this._setSelected(s.selected === path && s.instance === null ? null : path);
  }

  async _loadDetail(state, path) {
    try {
      const detail = await getJson(this._url('prim', path));
      if (state !== this.state || state.selected !== path) return;
      state.detail = detail;
      this.render();
    } catch (err) {
      if (state === this.state) this.callbacks.onToast?.(describe(err));
    }
  }

  // -- rendering ------------------------------------------------------------

  render() {
    const s = this.state;
    if (!s) {
      this.root.replaceChildren(el('p', 'blank', 'No scene graph for a .glb file.'));
      return;
    }
    const focused = document.activeElement?.closest?.('.prim[data-path]');
    const refocus = focused && this.root.contains(focused) ? focused.dataset.path : undefined;
    const scroller = this.root.querySelector('.scene__tree');
    const scrollTop = scroller?.scrollTop ?? 0;

    const tree = el('div', 'scene__tree');
    tree.setAttribute('role', 'tree');
    tree.setAttribute('aria-label', 'Prims');
    if (this.query) this._buildResults(tree, s);
    else this._buildLevel(tree, s, ROOT, []);
    const children = [this._buildBar(), tree];
    if (s.selected) children.push(this._buildDetail(s));
    this.root.replaceChildren(...children);
    tree.scrollTop = this.query ? 0 : scrollTop;

    if (refocus) this.root.querySelector(`.prim[data-path="${CSS.escape(refocus)}"]`)?.focus({ preventScroll: true });
  }

  _scrollToSelected() {
    const s = this.state;
    if (!s?.selected) return;
    const row = this.root.querySelector(`.prim[data-path="${CSS.escape(s.selected)}"]`);
    row?.scrollIntoView({ block: 'center' });
  }

  /** Find box and expand / collapse. Built once, so typing keeps its focus. */
  _buildBar() {
    const s = this.state;
    if (!this.bar) {
      this.bar = el('div', 'scene__bar');
      const search = el('label', 'search');
      search.appendChild(icon('search'));
      const input = el('input');
      input.type = 'search';
      input.placeholder = 'Find';
      input.spellcheck = false;
      input.setAttribute('aria-label', 'Find a prim by name');
      input.addEventListener('input', () => {
        this.query = input.value.trim().toLowerCase();
        this._search();
        this.render();
      });
      input.addEventListener('keydown', (e) => {
        e.stopPropagation(); // typing must not fly the camera or toggle settings
        if (e.key === 'Escape') {
          input.value = '';
          this.query = '';
          this.render();
          input.blur();
        } else if (e.key === 'Enter') {
          this.root.querySelector('.scene__tree .prim[data-path]')?.click();
        }
      });
      search.appendChild(input);
      this.searchInput = input;
      this.bar.appendChild(search);
      this.tools = el('div', 'scene__tools');
      this.expandBtn = iconButton('expandAll', 'Expand all', () => void this._expandAll());
      this.collapseBtn = iconButton('collapseAll', 'Collapse all', () => this._collapseAll());
      this.tools.append(this.expandBtn, this.collapseBtn);
      this.bar.appendChild(this.tools);
    }
    this.expandBtn.disabled = s.expanding || !s.levels.has(ROOT) || !!this.query;
    this.collapseBtn.disabled = s.expanding || !s.open.size || !!this.query;
    return this.bar;
  }

  /** Ask the server for prims matching the find box (the tree may be mostly closed). */
  _search() {
    clearTimeout(this._searchTimer);
    const query = this.query;
    const state = this.state;
    if (!query || !state) {
      this.results = null;
      return;
    }
    this.results = { query, pending: true, items: [] };
    this._searchTimer = setTimeout(async () => {
      try {
        const url = `/api/scenes/${encodeURIComponent(this.sceneId)}/find?q=${encodeURIComponent(query)}`;
        const out = await getJson(url);
        if (this.query !== query || this.state !== state) return;
        this.results = { query, pending: false, items: out.results, truncated: out.truncated };
      } catch (err) {
        if (this.query !== query) return;
        this.results = { query, pending: false, items: [], error: describe(err) };
      }
      this.render();
    }, 180);
  }

  /** The find box's matches: name, and the two levels above for context. */
  _buildResults(host, s) {
    const r = this.results;
    if (!r || r.pending) {
      host.appendChild(el('p', 'blank', 'Searching…'));
      return;
    }
    if (r.error) {
      host.appendChild(el('p', 'blank', r.error));
      return;
    }
    if (!r.items.length) {
      host.appendChild(el('p', 'blank', 'No prim with that name.'));
      return;
    }
    for (const prim of r.items) {
      const row = el('button', `prim prim--result${s.selected === prim.path ? ' is-on' : ''}`);
      row.type = 'button';
      row.dataset.path = prim.path;
      row.title = prim.path;
      row.appendChild(houdiniIcon(primIcon(prim.typeName, prim.kind), undefined, 'hicon prim__icon'));
      row.appendChild(el('span', 'prim__name', prim.name));
      row.appendChild(el('span', 'prim__where', parentOf(prim.path).split('/').slice(-2).join('/') || '/'));
      row.addEventListener('click', async () => {
        // show it in the tree: reveal reads and opens the levels down to it
        this.query = '';
        this.results = null;
        if (this.searchInput) this.searchInput.value = '';
        await this.reveal(prim.path);
        this.callbacks.onSelect?.({ path: prim.path });
      });
      host.appendChild(row);
    }
    if (r.truncated) host.appendChild(el('p', 'blank', 'Showing the first matches. Type more to narrow it down.'));
  }

  /**
   * One prim's children, and theirs where open. `rails` says, for each ancestor
   * level, whether that ancestor still has siblings below it, which is whether
   * a vertical line keeps running down past the rows drawn now.
   */
  _buildLevel(host, s, parent, rails) {
    const level = s.levels.get(parent);
    const error = s.errors.get(parent);
    if (error) {
      host.appendChild(note(error, rails, 'error'));
      return;
    }
    if (!level) {
      host.appendChild(note('Loading…', rails));
      return;
    }
    if (!level.children.length && parent === ROOT) {
      host.appendChild(el('p', 'blank', 'This scene is empty.'));
      return;
    }
    const more = level.total - level.children.length;
    level.children.forEach((prim, index) => {
      const last = index === level.children.length - 1 && more <= 0;
      host.appendChild(this._buildRow(s, prim, rails, last));
      if (s.open.has(prim.path)) this._buildLevel(host, s, prim.path, [...rails, !last]);
    });
    if (more > 0) host.appendChild(note(`${more.toLocaleString()} more not shown`, rails));
  }

  _buildRow(s, prim, rails, last) {
    const open = s.open.has(prim.path);
    const branch = prim.childCount > 0;
    const depth = rails.length;
    const selected = s.selected === prim.path;

    const classes = ['prim'];
    if (selected) classes.push('is-on');
    else if (s.selected && s.selected.startsWith(`${prim.path}/`)) classes.push('is-ancestor');
    if (!prim.active) classes.push('is-inactive');
    if (prim.specifier !== 'def') classes.push(`is-${prim.specifier}`);
    if (prim.instanceProxy) classes.push('is-proxy');
    const row = el('button', classes.join(' '));
    row.type = 'button';
    row.dataset.path = prim.path;
    row.setAttribute('role', 'treeitem');
    row.setAttribute('aria-level', String(depth + 1));
    row.setAttribute('aria-selected', String(selected));
    if (branch) row.setAttribute('aria-expanded', String(open));
    row.title = rowTitle(prim);

    for (const guide of guides(rails, last)) row.appendChild(guide);

    const caret = el('span', `prim__caret${open ? ' is-open' : ''}${!branch && depth > 0 ? ' is-leaf' : ''}`);
    if (branch) {
      caret.appendChild(icon('chevronRight'));
      caret.title = 'Shift-click to open or close everything inside';
      caret.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!e.shiftKey) this._toggle(prim);
        else if (open) this._collapseAll(prim.path);
        else void this._expandAll(prim.path);
      });
    }
    row.appendChild(caret);
    row.appendChild(houdiniIcon(primIcon(prim.typeName, prim.kind), undefined, 'hicon prim__icon'));
    row.appendChild(el('span', 'prim__name', prim.name));

    // only what an artist acts on: the variant, and where the content comes from
    const flags = el('span', 'prim__flags');
    if (prim.variants) {
      const selections = Object.entries(prim.variants);
      const pill = el('span', 'prim__variant');
      pill.appendChild(el('span', undefined,
        selections.length === 1 ? selections[0][1] || '(none)' : `${selections.length} variants`));
      pill.title = selections.map(([set, value]) => `${set}: ${value || '(none)'}`).join('\n');
      flags.appendChild(pill);
    }
    if (prim.arcs.includes('reference')) {
      flags.appendChild(houdiniIcon('LOP__reference', 'Brought in by a reference', 'hicon hicon--flag'));
    }
    if (prim.arcs.includes('payload')) {
      flags.appendChild(houdiniIcon('SCENEGRAPH__payloads', prim.unloaded ? 'Payload, not loaded' : 'Brought in by a payload',
        `hicon hicon--flag${prim.unloaded ? ' is-unloaded' : ''}`));
    }
    if (prim.instance) {
      flags.appendChild(houdiniIcon('SCENEGRAPH__primtype__instances', 'Instance: shares its contents with copies', 'hicon hicon--flag'));
    }
    if (flags.childElementCount) row.appendChild(flags);

    row.addEventListener('click', (e) => {
      if (e.detail > 1) return; // the second click of a double-click
      this._clickPrim(prim.path);
    });
    row.addEventListener('dblclick', () => {
      if (s.selected !== prim.path) this._setSelected(prim.path);
      this.callbacks.onFrame?.(prim.path);
    });
    return row;
  }

  /**
   * The picked prim: name and what it is first, then its variants. Paths,
   * schemas and the layers it comes from are folded away under "Details".
   */
  _buildDetail(s) {
    const section = el('div', 'scene__detail pdetail');
    const path = s.selected;
    const detail = s.detail;
    const prim = this._findPrim(path);
    const typeName = detail?.typeName ?? prim?.typeName ?? '';
    const kind = detail?.kind ?? prim?.kind;

    const head = el('div', 'pdetail__head');
    head.appendChild(houdiniIcon(primIcon(typeName, kind), undefined, 'hicon pdetail__icon'));
    const title = el('div', 'pdetail__title');
    const name = path.slice(path.lastIndexOf('/') + 1) || '/';
    title.appendChild(el('div', 'pdetail__name', s.instance !== null ? `${name} [${s.instance}]` : name));
    title.appendChild(el('div', 'pdetail__sub', [typeName || 'Prim', kind ? capitalize(kind) : null].filter(Boolean).join(' · ')));
    title.title = path;
    head.appendChild(title);
    const actions = el('div', 'pdetail__actions');
    actions.append(
      iconButton('target', 'Frame it (F)', () => this.callbacks.onFrame?.(path)),
      iconButton('copy', 'Copy its path', () => void this._copy(path)),
      iconButton('close', 'Deselect (Esc)', () => this._setSelected(null)),
    );
    head.appendChild(actions);
    section.appendChild(head);

    const chips = el('div', 'chips');
    const chip = (text, accent = false, tip = '') => {
      const c = el('span', `chip${accent ? ' chip--accent' : ''}`, text);
      if (tip) c.title = tip;
      chips.appendChild(c);
    };
    if (s.instance !== null) chip(`Instance ${s.instance}`, true, 'One copy placed by this point instancer');
    const geo = this.callbacks.geometry?.({ path, instance: s.instance });
    if (geo) {
      chip(geo.objects ? `${fmtCount(geo.instances)} ${geo.instances === 1 ? 'object' : 'objects'} · ${fmtCount(geo.triangles)} triangles`
        : 'Nothing visible');
    }
    if (detail && !detail.active) chip('Deactivated');
    if (detail?.instanceProxy) chip('Inside an instance', false, 'Shares its contents with other copies; edit the original');
    if (detail?.unloaded) chip('Not loaded');
    if (chips.childElementCount) section.appendChild(chips);

    if (!detail) return section;

    // variants: switchable, except inside an instance (edits there would not apply)
    const sets = detail.variantSets.filter((set) => set.options.length);
    if (sets.length) {
      const switchable = !detail.instanceProxy && this.callbacks.onVariant;
      const box = el('div', 'vsets');
      for (const set of sets) {
        const row = el('div', 'vset');
        row.appendChild(el('span', 'vset__label', set.name));
        const seg = el('div', 'seg');
        seg.setAttribute('role', 'radiogroup');
        seg.setAttribute('aria-label', `Variant set ${set.name}`);
        for (const option of set.options) {
          const b = el('button', `seg__btn${option === set.selection ? ' is-on' : ''}`, option);
          b.type = 'button';
          b.setAttribute('role', 'radio');
          b.setAttribute('aria-checked', String(option === set.selection));
          b.disabled = !!this.busy || !switchable;
          b.addEventListener('click', () => {
            if (option !== set.selection) this.callbacks.onVariant(path, set.name, option);
          });
          seg.appendChild(b);
        }
        row.appendChild(seg);
        box.appendChild(row);
      }
      section.appendChild(box);
    }

    const more = el('details', 'disclosure');
    more.open = !!this.detailsOpen;
    more.addEventListener('toggle', () => { this.detailsOpen = more.open; });
    more.appendChild(el('summary', undefined, 'Details'));
    const f = facts();
    f.add('Path', path, true);
    if (detail.specifier !== 'def') f.add('Specifier', detail.specifier);
    if (detail.appliedSchemas.length) f.add('Schemas', detail.appliedSchemas.join(', '), true);
    f.add('Properties', `${detail.attributeCount} attributes, ${detail.relationshipCount} relationships`);
    const other = detail.arcs.filter((a) => a === 'inherit' || a === 'specialize');
    if (other.length) f.add('Also', other.map((a) => `${a}s`).join(', '));
    more.appendChild(f.root);

    const heading = el('h3', undefined, 'Comes from');
    heading.title = 'Every layer with something to say about this prim, strongest first';
    more.appendChild(heading);
    const list = el('div', 'arc-list');
    for (const opinion of detail.opinions) {
      const row = el('div', `arc${opinion.root ? ' arc--here' : ''}`);
      const pip = el('span', 'arc__pip');
      const glyph = ARC_ICON[opinion.arc];
      if (glyph) pip.appendChild(houdiniIcon(glyph, undefined, 'hicon hicon--pip'));
      row.appendChild(pip);
      const main = el('div', 'arc__main');
      main.appendChild(el('div', 'arc__name', opinion.name));
      row.appendChild(main);
      row.appendChild(el('span', 'arc__kind', opinion.root ? 'root layer' : ARC_NAME[opinion.arc] ?? opinion.arc));
      row.title = [opinion.layerPath, opinion.specPath, opinion.introducedBy ? `via ${opinion.introducedBy}` : '']
        .filter(Boolean).join('\n');
      list.appendChild(row);
    }
    more.appendChild(list);
    section.appendChild(more);
    return section;
  }

  async _copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      this.callbacks.onToast?.('Prim path copied', 'ok');
    } catch {
      this.callbacks.onToast?.('Could not copy to the clipboard');
    }
  }

  // -- keyboard -------------------------------------------------------------

  /** Arrow keys walk the tree the way a file browser's outline does. */
  _onKey(e) {
    const row = e.target.closest?.('.prim[data-path]');
    if (!row || e.ctrlKey || e.metaKey || e.altKey) return;
    const rows = [...this.root.querySelectorAll('.prim[data-path]')];
    const index = rows.indexOf(row);
    const path = row.dataset.path;
    const prim = this._findPrim(path);
    const s = this.state;
    const focus = (target) => {
      if (!target) return;
      target.focus();
      this._setSelected(target.dataset.path);
    };
    const handled = () => {
      e.preventDefault();
      e.stopPropagation(); // keep the viewport's fly keys out of it
    };

    if (e.key === '*' || (e.shiftKey && e.key === 'ArrowRight')) {
      if (prim && prim.childCount > 0) void this._expandAll(path);
      return handled();
    }
    if (e.shiftKey && e.key === 'ArrowLeft') {
      this._collapseAll(path);
      return handled();
    }
    switch (e.key) {
      case 'ArrowDown': focus(rows[index + 1]); break;
      case 'ArrowUp': focus(rows[index - 1]); break;
      case 'ArrowRight':
        if (prim && prim.childCount > 0 && !s.open.has(path)) this._toggle(prim);
        else if (s.open.has(path)) focus(rows[index + 1]);
        break;
      case 'ArrowLeft':
        if (prim && s.open.has(path)) this._toggle(prim);
        else focus(rows.find((r) => r.dataset.path === parentOf(path)));
        break;
      case 'Enter':
        this.callbacks.onFrame?.(path);
        break;
      default:
        return undefined;
    }
    return handled();
  }

  _findPrim(path) {
    return this.state.levels.get(parentOf(path))?.children.find((p) => p.path === path);
  }
}

// -- helpers ------------------------------------------------------------------

/** A line in the tree that is not a prim: always the last thing at its level. */
function note(text, rails, variant) {
  const node = el('div', `prim prim--note${variant ? ' prim--error' : ''}`);
  for (const guide of guides(rails, true)) node.appendChild(guide);
  node.appendChild(el('span', 'prim__caret'));
  node.appendChild(el('span', 'prim__name', text));
  return node;
}

/**
 * The connector lines in front of a row. Column k sits under the caret of the
 * ancestor at depth k; the last column is this row's own branch: a tee when
 * more siblings follow, an elbow when it is the last of them.
 */
function guides(rails, last) {
  const cells = [];
  for (let column = 0; column < rails.length; column++) {
    const own = column === rails.length - 1;
    const kind = own ? (last ? 'elbow' : 'tee') : rails[column + 1] ? 'pipe' : 'blank';
    cells.push(el('span', `guide guide--${kind}`));
  }
  return cells;
}

function rowTitle(prim) {
  const lines = [prim.path];
  const what = [prim.specifier !== 'def' ? prim.specifier : null, prim.typeName || null,
    prim.kind ? `kind ${prim.kind}` : null].filter(Boolean).join(' · ');
  if (what) lines.push(what);
  if (!prim.active) lines.push('Deactivated');
  if (prim.childCount) lines.push(`${prim.childCount} ${prim.childCount === 1 ? 'child' : 'children'}`);
  lines.push('Double-click to frame it');
  return lines.join('\n');
}

function parentOf(path) {
  const cut = path.lastIndexOf('/');
  return cut <= 0 ? ROOT : path.slice(0, cut);
}

function describe(err) {
  return err instanceof ApiError ? err.message : 'Could not read the scene graph';
}

function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

function fmtCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)} k`;
  return Math.round(n).toLocaleString();
}
