// Selection: which geometry belongs to a USD prim, and highlighting it.
//
// The converter tags every glTF node with the prim the scene graph selects for
// it (extras.usdPath): the gprim itself, or for PointInstancer instances the
// instancer, with each instance's authored index (usdInstanceIds) and the gprim
// inside the prototype (usdProto).
//
// Other copies of the same instance get a fainter highlight in another colour:
// the other instances of the same prototype in a PointInstancer (found here),
// and the other copies of a native instance (listed by the server: setPeers).
//
// Highlights are a translucent tint drawn over the geometry, batched so their
// cost follows what is highlighted, not how it is split up: every highlighted
// copy of one geometry (thousands of native instances, a PointInstancer's
// instances) is one instanced draw. A batch holds only the highlighted copies
// the LOD manager currently draws at that geometry's level, read from its
// per-instance state and repacked when it changes, so one highlighted
// instance out of a million costs one instance.
//
// Each selection is a generation of batches that fades in (the selection
// first, its peers a moment later) and out on its own, so the old highlight
// can fade while the new one appears.
import * as THREE from 'three';
import { HELPER_LAYER } from './picking.js';

/** Per highlight role: tint, base opacity and extra opacity at silhouettes. */
const LOOKS = {
  selected: { tint: new THREE.Color(0x0a84ff), alpha: 0.32, rim: 0.38, order: 11 }, // refgraph's selection blue
  peer: { tint: new THREE.Color(0xffa63d), alpha: 0.16, rim: 0.28, order: 10 }, // the other copies: amber, faint
};
/** Fade timings, in ms. */
const FADE_IN = 220;
const FADE_OUT = 160;
const PEER_DELAY = 140;
const smooth = (t) => t * t * (3 - 2 * t);

const vertexShader = /* glsl */`
#include <common>
varying vec3 vViewNormal;
void main() {
  #include <begin_vertex>
  #include <project_vertex>
  #ifdef HAS_NORMAL
    vec3 n = normal;
    #ifdef USE_INSTANCING
      n = mat3( instanceMatrix ) * n;
    #endif
    vViewNormal = normalize( normalMatrix * n );
  #else
    vViewNormal = vec3( 0.0, 0.0, 1.0 );
  #endif
}
`;

const fragmentShader = /* glsl */`
uniform vec3 uTint;
uniform vec2 uAlpha;
uniform float uFade;
varying vec3 vViewNormal;
void main() {
  // a little stronger at grazing angles, so silhouettes read
  float rim = 1.0 - abs( normalize( vViewNormal ).z );
  gl_FragColor = vec4( uTint, ( uAlpha.x + uAlpha.y * rim * rim ) * uFade );
}
`;

const IDENTITY = new THREE.Matrix4().toArray();
const _a = new THREE.Matrix4();
const _b = new THREE.Matrix4();

function trianglesOf(mesh) {
  const g = mesh.geometry;
  return (g.index ? g.index.count : g.attributes.position.count) / 3;
}

export class Selection {
  constructor(lod) {
    this.lod = lod;
    this.entries = [];
    this.byMesh = new Map();
    this.current = null;
    this.selected = new Set(); // entries of the current selection
    /** Holds the overlay batches; the viewer adds it to the scene. */
    this.root = new THREE.Group();
    this.root.name = 'usdv highlights';
    this.shown = null; // the current generation
    this.generations = new Set(); // the current one and any still fading out
    this._lodVersion = -1;
  }

  /** Index the nodes of a freshly loaded model. */
  build(model) {
    for (const g of this.generations) g.dispose();
    this.generations.clear();
    this.shown = null;
    this.current = null;
    this.selected = new Set();
    this.entries = [];
    this.byMesh = new Map();
    model.traverse((o) => {
      const path = o.userData?.usdPath;
      if (typeof path !== 'string') return;
      const meshes = o.isMesh ? [o] : o.children.filter((c) => c.isMesh);
      if (!meshes.length) return;
      const ids = o.userData.usdInstanceIds;
      const entry = {
        path,
        proto: o.userData.usdProto ?? null,
        instance: Number.isInteger(o.userData.usdInstance) ? o.userData.usdInstance : null,
        ids: ids instanceof Float32Array ? ids : null,
        meshes,
        level: o.userData.usdvLod?.level ?? 0,
        triangles: meshes.reduce((n, m) => n + trianglesOf(m), 0),
      };
      this.entries.push(entry);
      for (const m of meshes) this.byMesh.set(m, entry);
    });
  }

  /** The prim (and instance) a picked mesh / instance slot belongs to. */
  describe(mesh, slot) {
    const entry = this.byMesh.get(mesh);
    if (!entry) return null;
    let instance = entry.instance;
    if (mesh.isInstancedMesh) {
      const original = this.lod.instanceInSlot(mesh, slot);
      if (entry.ids) instance = Math.round(entry.ids[original]);
    }
    return { path: entry.path, instance, proto: entry.proto };
  }

  /**
   * Entries a selection covers, each with the original instances it covers
   * (null = all of them).
   */
  _match(sel) {
    if (!sel) return [];
    const prefix = sel.path === '/' ? '/' : `${sel.path}/`;
    const under = (p) => p && (p === sel.path || p.startsWith(prefix));
    const out = [];
    for (const entry of this.entries) {
      if (sel.instance !== null && sel.instance !== undefined) {
        if (entry.path !== sel.path) continue;
        if (entry.ids) {
          const mask = [];
          for (let i = 0; i < entry.ids.length; i++) if (Math.round(entry.ids[i]) === sel.instance) mask.push(i);
          if (mask.length) out.push({ entry, instances: mask });
        } else if (entry.instance === sel.instance) out.push({ entry, instances: null });
        continue;
      }
      if (under(entry.path) || under(entry.proto)) out.push({ entry, instances: null });
    }
    return out;
  }

  /** How much geometry a selection covers (full-detail level only). */
  geometry(sel) {
    let objects = 0;
    let triangles = 0;
    for (const { entry, instances } of this._match(sel)) {
      if (entry.level !== 0) continue;
      const mesh = entry.meshes[0];
      const n = instances ? instances.length : (mesh.isInstancedMesh ? (entry.ids?.length ?? this._count(mesh)) : 1);
      objects += n;
      triangles += n * entry.triangles;
    }
    return { objects, instances: objects, triangles };
  }

  _count(mesh) {
    return this.lod.groupOf.get(mesh)?.group.count ?? mesh.count;
  }

  /** World-space bounds of a selection, or null when it has no geometry. */
  bounds(sel) {
    const box = new THREE.Box3();
    const b = new THREE.Box3();
    const m = new THREE.Matrix4();
    const inst = new THREE.Matrix4();
    for (const { entry, instances } of this._match(sel)) {
      if (entry.level !== 0) continue;
      for (const mesh of entry.meshes) {
        const managed = this.lod.groupOf.get(mesh);
        if (mesh.isInstancedMesh && managed) {
          const g = managed.group;
          const list = instances ?? Array.from({ length: g.count }, (_, i) => i);
          if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
          for (const i of list) {
            m.copy(g.world).multiply(inst.fromArray(g.matrices, i * 16));
            box.union(b.copy(mesh.geometry.boundingBox).applyMatrix4(m));
          }
        } else {
          if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
          box.union(b.copy(mesh.geometry.boundingBox).applyMatrix4(mesh.matrixWorld));
        }
      }
    }
    return box.isEmpty() ? null : box;
  }

  /**
   * Highlight a selection ({ path, instance? }) or nothing (null). A
   * PointInstancer instance also lights up the instancer's other instances of
   * the same prototype.
   */
  set(sel) {
    this.clear();
    this.current = sel;
    const matched = this._match(sel);
    this.selected = new Set(matched.map((m) => m.entry));
    if (!matched.length) return;
    const g = new Generation(this.root);
    this.shown = g;
    this.generations.add(g);
    for (const { entry, instances } of matched) this._add(g, 'selected', entry, instances);
    if (sel.instance === null || sel.instance === undefined) return;

    const protos = new Set(matched.map((m) => m.entry.proto));
    for (const entry of this.entries) {
      if (entry.path !== sel.path || !protos.has(entry.proto)) continue;
      if (entry.ids) {
        const others = [];
        for (let i = 0; i < entry.ids.length; i++) if (Math.round(entry.ids[i]) !== sel.instance) others.push(i);
        if (others.length) this._add(g, 'peer', entry, others);
      } else if (entry.instance !== sel.instance) this._add(g, 'peer', entry, null);
    }
  }

  /**
   * Also highlight, faintly, the prims at `paths` and everything beneath them
   * (the other copies of a native instance). Added to the current selection.
   */
  setPeers(paths) {
    const g = this.shown;
    if (!g || !paths?.length) return;
    const wanted = new Set(paths);
    for (const entry of this.entries) {
      if (this.selected.has(entry)) continue;
      // the entry's prim or one of its ancestors is a peer
      for (let p = entry.path; p.length > 1; p = p.slice(0, Math.max(p.lastIndexOf('/'), 1))) {
        if (wanted.has(p)) {
          this._add(g, 'peer', entry, null);
          break;
        }
      }
    }
  }

  /** Fade the current highlight out. */
  clear() {
    this.shown?.fadeOut(performance.now());
    this.shown = null;
    this.current = null;
    this.selected = new Set();
  }

  /** Step the fades to time `now`. Returns whether any is still running. */
  animate(now) {
    let running = false;
    for (const g of this.generations) {
      if (g.animate(now)) running = true;
      else if (g.closing) {
        g.dispose();
        this.generations.delete(g);
      }
    }
    return running;
  }

  /** Follow the LOD manager's latest choices. Call after lod.update(). */
  sync() {
    const lodChanged = this.lod.version !== this._lodVersion;
    this._lodVersion = this.lod.version;
    for (const g of this.generations) g.pack(lodChanged);
  }

  /** Highlight `instances` (original indices; null = all) of an entry's meshes. */
  _add(generation, role, entry, instances) {
    for (const mesh of entry.meshes) {
      const managed = this.lod.groupOf.get(mesh);
      let source;
      if (managed) {
        const { group, level } = managed;
        source = {
          world: group.world,
          matrices: group.instanced ? group.matrices : null,
          assigned: group.assigned,
          level,
          count: group.count,
        };
      } else {
        source = {
          world: mesh.matrixWorld,
          matrices: mesh.isInstancedMesh ? mesh.instanceMatrix.array.slice(0, mesh.count * 16) : null,
          assigned: null,
          level: 0,
          count: mesh.isInstancedMesh ? mesh.count : 1,
        };
      }
      source.indices = instances && source.matrices ? Int32Array.from(instances) : null;
      generation.add(role, mesh, source);
    }
  }
}

/** One selection's highlight: its batches, and a fade per role. */
class Generation {
  constructor(root) {
    this.root = root;
    this.batches = new Map();
    this.materials = new Map();
    this.closing = false;
    const now = performance.now();
    this.roles = {
      selected: { fade: { value: 0 }, anim: { from: 0, to: 1, start: now, duration: FADE_IN } },
      peer: { fade: { value: 0 }, anim: null }, // starts with the first peer
    };
    this.createdAt = now;
  }

  add(role, mesh, source) {
    const material = this._material(role, mesh);
    const key = `${role}:${mesh.geometry.uuid}:${material.uuid}`;
    let batch = this.batches.get(key);
    if (!batch) {
      batch = new Batch(mesh.geometry, material, this.root);
      this.batches.set(key, batch);
    }
    batch.add(source);
    // peers fade in a moment after the selection, or when they arrive (from the server)
    const r = this.roles[role];
    if (role === 'peer' && !r.started && !this.closing) {
      r.started = true;
      r.anim = { from: 0, to: 1, start: Math.max(performance.now(), this.createdAt + PEER_DELAY), duration: FADE_IN };
    }
  }

  _material(role, mesh) {
    const side = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material)?.side ?? THREE.FrontSide;
    const normal = !!mesh.geometry.attributes.normal;
    const key = `${role}:${side}:${normal}`;
    let m = this.materials.get(key);
    if (!m) {
      const look = LOOKS[role];
      m = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
          uTint: { value: look.tint },
          uAlpha: { value: new THREE.Vector2(look.alpha, look.rim) },
          uFade: this.roles[role].fade, // shared by the role's materials: one value to animate
        },
        defines: normal ? { HAS_NORMAL: '' } : {},
        transparent: true,
        depthWrite: false,
        depthFunc: THREE.LessEqualDepth,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -2,
        side,
      });
      m.userData.order = look.order;
      this.materials.set(key, m);
    }
    return m;
  }

  fadeOut(now) {
    this.closing = true;
    for (const r of Object.values(this.roles)) {
      r.anim = { from: r.fade.value, to: 0, start: now, duration: FADE_OUT };
    }
  }

  /** Returns whether a fade is still running. */
  animate(now) {
    let running = false;
    for (const r of Object.values(this.roles)) {
      const a = r.anim;
      if (!a) continue;
      const t = Math.min(Math.max((now - a.start) / a.duration, 0), 1);
      r.fade.value = a.from + (a.to - a.from) * smooth(t);
      if (t < 1) running = true;
      else r.anim = null;
    }
    return running;
  }

  pack(lodChanged) {
    for (const b of this.batches.values()) if (b.dirty || (lodChanged && b.followsLod)) b.pack();
  }

  dispose() {
    for (const b of this.batches.values()) b.dispose();
    for (const m of this.materials.values()) m.dispose();
    this.batches.clear();
    this.materials.clear();
  }
}

/**
 * Every highlighted copy of one geometry, in one role: an InstancedMesh of
 * the copies drawn now. When all copies share a world transform (one
 * instancer, say) it becomes the mesh's own and instance matrices are copied
 * as they are; otherwise each copy's full transform is written.
 */
class Batch {
  constructor(geometry, material, root) {
    this.geometry = geometry;
    this.material = material;
    this.root = root;
    this.sources = [];
    this.world = undefined; // the shared world transform; null once they differ
    this.capacity = 0;
    this.mesh = null;
    this.dirty = true;
    this.followsLod = false; // whether any copy's visibility comes from the LOD manager
  }

  add(source) {
    this.sources.push(source);
    if (source.assigned) this.followsLod = true;
    if (this.world === undefined) this.world = source.world;
    else if (this.world && !this.world.equals(source.world)) this.world = null;
    this.dirty = true;
  }

  _ensureCapacity() {
    const needed = this.sources.reduce((n, s) => n + (s.indices ? s.indices.length : s.count), 0);
    if (this.mesh && needed <= this.capacity) return;
    this.mesh?.removeFromParent();
    this.mesh?.dispose();
    const mesh = new THREE.InstancedMesh(this.geometry, this.material, Math.max(needed, 1));
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.name = 'usdv highlight';
    mesh.frustumCulled = false; // only drawn copies are packed
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.renderOrder = this.material.userData.order;
    mesh.layers.set(HELPER_LAYER);
    mesh.matrixAutoUpdate = false;
    this.root.add(mesh);
    this.mesh = mesh;
    this.capacity = Math.max(needed, 1);
  }

  pack() {
    this._ensureCapacity();
    const out = this.mesh.instanceMatrix.array;
    const shared = !!this.world;
    let n = 0;
    for (const s of this.sources) {
      const { indices, assigned, level, matrices } = s;
      const len = indices ? indices.length : s.count;
      for (let k = 0; k < len; k++) {
        const i = indices ? indices[k] : k;
        if (assigned && assigned[i] !== level) continue; // culled, or drawn at another level
        if (shared) {
          if (matrices) out.set(matrices.subarray(i * 16, i * 16 + 16), n * 16);
          else out.set(IDENTITY, n * 16);
        } else {
          _a.copy(s.world);
          if (matrices) _a.multiply(_b.fromArray(matrices, i * 16));
          _a.toArray(out, n * 16);
        }
        n += 1;
      }
    }
    this.mesh.count = n;
    this.mesh.visible = n > 0;
    if (n) this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.matrix.copy(shared ? this.world : _a.identity());
    this.mesh.matrixWorld.copy(this.mesh.matrix); // the root sits at the scene origin
    this.dirty = false;
  }

  dispose() {
    this.mesh?.removeFromParent();
    this.mesh?.dispose(); // the geometry is the model's: left alone
  }
}
