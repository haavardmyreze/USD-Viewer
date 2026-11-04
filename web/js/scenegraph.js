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
// never walked whole. It opens expanded up to a budget, materials excepted.

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

const countBadge = (value) => el('span', 'badge', typeof value === 'number' ? value.toLocaleString() : String(value));

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
      if (primPath === ROOT && !state.primed) {
        state.primed = true;
        if (state === this.state) void this._expandAll(ROOT, { initial: true });
      }
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
      this.root.replaceChildren(el('p', 'blank', 'The scene graph needs a USD scene. A .glb file has none.'));
      return;
    }
    const focused = document.activeElement?.closest?.('.prim[data-path]');
    const refocus = focused && this.root.contains(focused) ? focused.dataset.path : undefined;
    const scroller = this.root.querySelector('.scene__tree');
    const scrollTop = scroller?.scrollTop ?? 0;

    const top = s.levels.get(ROOT);
    const tree = el('div', 'scene__tree');
    tree.setAttribute('role', 'tree');
    tree.setAttribute('aria-label', 'Prims');
    this._buildLevel(tree, s, ROOT, []);
    const children = [this._buildBar(top), tree];
    if (s.selected) children.push(this._buildDetail(s));
    this.root.replaceChildren(...children);
    tree.scrollTop = scrollTop;

    if (refocus) this.root.querySelector(`.prim[data-path="${CSS.escape(refocus)}"]`)?.focus({ preventScroll: true });
  }

  _scrollToSelected() {
    const s = this.state;
    if (!s?.selected) return;
    const row = this.root.querySelector(`.prim[data-path="${CSS.escape(s.selected)}"]`);
    row?.scrollIntoView({ block: 'nearest' });
  }

  _buildBar(top) {
    const s = this.state;
    const bar = el('div', 'scene__bar');
    const heading = el('h3', undefined, 'Prims');
    if (top) heading.appendChild(countBadge(top.total));
    bar.appendChild(heading);
    const tools = el('div', 'scene__tools');
    const expand = iconButton('expandAll', 'Expand all', () => void this._expandAll());
    expand.disabled = s.expanding || !top;
    const collapse = iconButton('collapseAll', 'Collapse all', () => this._collapseAll());
    collapse.disabled = s.expanding || !s.open.size;
    tools.append(expand, collapse);
    bar.appendChild(tools);
    return bar;
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
      host.appendChild(note('Composing…', rails));
      return;
    }
    if (!level.children.length && parent === ROOT) {
      host.appendChild(el('p', 'blank', 'This stage composes no prims.'));
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
      caret.title = 'Shift-click to expand or collapse everything beneath';
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

    const flags = el('span', 'prim__flags');
    if (prim.variants) {
      const selections = Object.entries(prim.variants);
      const pill = el('span', 'prim__variant');
      pill.appendChild(houdiniIcon('LOP__setvariant', undefined, 'hicon hicon--flag'));
      pill.appendChild(el('span', undefined,
        selections.length === 1 ? selections[0][1] || '(none)' : String(selections.length)));
      pill.title = selections.map(([set, value]) => `Variant ${set} = ${value || '(none)'}`).join('\n');
      flags.appendChild(pill);
    }
    for (const arc of prim.arcs) {
      if (arc === 'reference') {
        flags.appendChild(houdiniIcon('LOP__reference', 'Has an authored reference', 'hicon hicon--flag'));
      } else if (arc === 'payload') {
        flags.appendChild(houdiniIcon('SCENEGRAPH__payloads', prim.unloaded ? 'Payload, not loaded' : 'Payload, loaded',
          `hicon hicon--flag${prim.unloaded ? ' is-unloaded' : ''}`));
      } else {
        const f = el('span', 'prim__flag', arc);
        f.title = `Has an authored ${arc}`;
        flags.appendChild(f);
      }
    }
    if (prim.instance) {
      flags.appendChild(houdiniIcon('SCENEGRAPH__primtype__instances', 'Instanceable, sharing one prototype', 'hicon hicon--flag'));
    }
    if (flags.childElementCount) row.appendChild(flags);
    row.appendChild(el('span', 'prim__type', prim.typeName));

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

  _buildDetail(s) {
    const section = el('div', 'scene__detail');
    const path = s.selected;
    const detail = s.detail;

    const head = el('div', 'insp__path');
    const text = el('span', 'insp__pathText trunc', `‎${path}${s.instance !== null ? `[${s.instance}]` : ''}`);
    text.title = path;
    head.appendChild(text);
    head.appendChild(iconButton('target', 'Frame in view (F)', () => this.callbacks.onFrame?.(path)));
    head.appendChild(iconButton('copy', 'Copy prim path', () => void this._copy(path)));
    head.appendChild(iconButton('close', 'Clear selection (Esc)', () => this._setSelected(null)));
    section.appendChild(head);

    if (!detail) {
      section.appendChild(el('p', 'blank', 'Reading prim…'));
      return section;
    }

    const f = facts();
    f.add('Type', detail.typeName || '(typeless)', true);
    f.add('Kind', detail.kind, true);
    if (detail.specifier !== 'def') f.add('Specifier', detail.specifier);
    if (!detail.active) f.add('Active', 'no');
    if (s.instance !== null) f.add('Instance', `#${s.instance} of this instancer`);
    if (detail.instance) f.add('Instance', 'instanceable');
    if (detail.instanceProxy) f.add('Instance', 'inside an instance');
    if (detail.arcs.includes('payload')) f.add('Payload', detail.unloaded ? 'not loaded' : 'loaded');
    // variant selections inside an instance cannot be overridden from its proxy
    const switchable = !detail.instanceProxy && this.callbacks.onVariant;
    if (!switchable) {
      for (const set of detail.variantSets) {
        const value = el('span', undefined, set.selection ?? '(none)');
        value.title = `Options: ${set.options.join(', ') || 'none'}`;
        f.add(`{${set.name}}`, value, true);
      }
    }
    if (detail.appliedSchemas.length) f.add('Schemas', detail.appliedSchemas.join(', '), true);
    f.add('Properties', `${detail.attributeCount} attr · ${detail.relationshipCount} rel`);
    const geo = this.callbacks.geometry?.({ path, instance: s.instance });
    if (geo) {
      f.add('In view', geo.objects
        ? `${fmtCount(geo.instances)} ${geo.instances === 1 ? 'object' : 'objects'} · ${fmtCount(geo.triangles)} tris`
        : 'no geometry');
    }
    section.appendChild(f.root);

    if (switchable && detail.variantSets.some((set) => set.options.length)) {
      const vh = el('h3', undefined, 'Variants');
      vh.title = 'Switching re-composes the stage with this selection; the files are not changed';
      section.appendChild(vh);
      const sets = el('div', 'vsets');
      for (const set of detail.variantSets) {
        if (!set.options.length) continue;
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
          b.disabled = !!this.busy;
          b.addEventListener('click', () => {
            if (option !== set.selection) this.callbacks.onVariant(path, set.name, option);
          });
          seg.appendChild(b);
        }
        row.appendChild(seg);
        sets.appendChild(row);
      }
      section.appendChild(sets);
    }

    const heading = el('h3', undefined, 'Opinions');
    heading.appendChild(countBadge(detail.opinions.length));
    heading.title = 'Every layer with something to say about this prim, strongest first';
    section.appendChild(heading);
    const list = el('div', 'arc-list');
    for (const opinion of detail.opinions) {
      const row = el('div', `arc${opinion.root ? ' arc--here' : ''}`);
      const pip = el('span', 'arc__pip');
      const glyph = ARC_ICON[opinion.arc];
      if (glyph) pip.appendChild(houdiniIcon(glyph, undefined, 'hicon hicon--pip'));
      row.appendChild(pip);
      const main = el('div', 'arc__main');
      main.appendChild(el('div', 'arc__name', opinion.name));
      const meta = [opinion.specPath];
      if (opinion.introducedBy) meta.push(`from ${opinion.introducedBy}`);
      main.appendChild(el('div', 'arc__meta', meta.join('  ·  ')));
      row.appendChild(main);
      row.appendChild(el('span', 'arc__kind', opinion.root ? 'root layer' : ARC_NAME[opinion.arc] ?? opinion.arc));
      row.title = opinion.layerPath;
      list.appendChild(row);
    }
    section.appendChild(list);
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

function fmtCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)} k`;
  return Math.round(n).toLocaleString();
}
