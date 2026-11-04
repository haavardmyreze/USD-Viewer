// Per-instance frustum culling and level-of-detail selection.
//
// The converter writes simplified copies of heavy meshes into a second glTF
// scene, tagged extras.usdvLod = { group, level }. Each frame the camera moves,
// every instance of a group is tested against the view frustum and assigned the
// level that suits its size on screen. The visible instances of each level are
// packed into that level's InstancedMesh, so hidden or tiny objects cost nothing.
import * as THREE from 'three';

const _frustum = new THREE.Frustum();
const _projView = new THREE.Matrix4();
const _sphere = new THREE.Sphere();
const _m = new THREE.Matrix4();
const _inst = new THREE.Matrix4();

function meshesOf(object) {
  if (object.isMesh) return [object];
  return object.children.filter((c) => c.isMesh);
}

function triangleCount(mesh) {
  const g = mesh.geometry;
  return (g.index ? g.index.count : g.attributes.position.count) / 3;
}

export class LodManager {
  constructor() {
    this.groups = [];
    /** Managed mesh -> { group, level }. */
    this.groupOf = new Map();
    /** Screen-height fraction below which LOD1 is used; each further level halves it. */
    this.threshold = 0.45;
    this.stats = { triangles: 0, visible: 0, culled: 0, total: 0 };
  }

  /** Find LOD groups (and plain instanced meshes) under `root`. */
  setup(root) {
    this.groups = [];
    this.groupOf = new Map();
    root.updateMatrixWorld(true);
    const byId = new Map();
    const grouped = new Set();
    root.traverse((o) => {
      const info = o.userData?.usdvLod;
      if (!info) return;
      if (!byId.has(info.group)) byId.set(info.group, []);
      const meshes = meshesOf(o);
      byId.get(info.group)[info.level] = meshes;
      meshes.forEach((m) => grouped.add(m));
    });
    for (const levels of byId.values()) {
      if (levels.length && levels.every(Boolean)) this._addGroup(levels);
    }
    // instanced meshes without LODs still benefit from per-instance culling
    root.traverse((o) => {
      if (o.isInstancedMesh && !grouped.has(o)) this._addGroup([[o]]);
    });
  }

  _addGroup(levels) {
    const base = levels[0];
    const first = base[0];
    const instanced = !!first.isInstancedMesh;
    const count = instanced ? first.count : 1;

    const box = new THREE.Box3();
    for (const m of base) {
      if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
      box.union(m.geometry.boundingBox);
    }
    const local = box.getBoundingSphere(new THREE.Sphere());

    // world-space bounding sphere of every instance
    const matrices = instanced ? first.instanceMatrix.array.slice() : null;
    // per-instance colours (e.g. a Points prim's displayColor) are packed like the matrices
    const colors = instanced && first.instanceColor ? first.instanceColor.array.slice() : null;
    const spheres = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      _m.copy(first.matrixWorld);
      if (instanced) _m.multiply(_inst.fromArray(matrices, i * 16));
      _sphere.copy(local).applyMatrix4(_m);
      spheres.set([_sphere.center.x, _sphere.center.y, _sphere.center.z, _sphere.radius], i * 4);
    }

    for (const meshes of levels) {
      for (const m of meshes) {
        m.frustumCulled = false; // culled per instance here
        if (instanced) {
          m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
          if (colors && !m.instanceColor) m.instanceColor = new THREE.InstancedBufferAttribute(colors.slice(), 3);
          m.instanceColor?.setUsage(THREE.DynamicDrawUsage);
          m.count = 0;
        }
        m.visible = false;
      }
    }

    const group = {
      levels,
      instanced,
      count,
      localBox: box,
      world: first.matrixWorld.clone(),
      matrices,
      colors,
      spheres,
      assigned: new Int8Array(count).fill(-2), // -2 = never assigned, -1 = culled
      tris: levels.map((meshes) => meshes.reduce((n, m) => n + triangleCount(m), 0)),
      // slots[level][slot] = original instance index drawn in that slot
      slots: levels.map(() => new Int32Array(Math.max(count, 1))),
      version: 0, // bumped whenever the packing changes
    };
    this.groups.push(group);
    levels.forEach((meshes, level) => meshes.forEach((m) => this.groupOf.set(m, { group, level })));
  }

  /** Original instance index drawn in `slot` of a managed mesh. */
  instanceInSlot(mesh, slot) {
    const entry = this.groupOf.get(mesh);
    if (!entry || !entry.group.instanced) return slot;
    return entry.group.slots[entry.level][slot] ?? slot;
  }

  /** World-space bounds of everything managed (all instances, ignoring culling). */
  expandBox(box) {
    const b = new THREE.Box3();
    for (const g of this.groups) {
      for (let i = 0; i < g.count; i++) {
        _m.copy(g.world);
        if (g.instanced) _m.multiply(_inst.fromArray(g.matrices, i * 16));
        box.union(b.copy(g.localBox).applyMatrix4(_m));
      }
    }
    return box;
  }

  /**
   * Make one instance of every level visible, so a single render uploads all
   * geometry to the GPU. Otherwise each level uploads the first time it is
   * needed, which stalls the frame mid-flight. Call update() afterwards.
   */
  showAllLevels() {
    for (const g of this.groups) {
      for (const meshes of g.levels) {
        for (const m of meshes) {
          m.visible = true;
          if (g.instanced) {
            m.instanceMatrix.array.set(g.matrices.subarray(0, 16), 0);
            if (g.colors && m.instanceColor) {
              m.instanceColor.array.set(g.colors.subarray(0, 3), 0);
              m.instanceColor.needsUpdate = true;
            }
            m.count = 1;
            g.slots[g.levels.indexOf(meshes)][0] = 0;
            m.instanceMatrix.needsUpdate = true;
          }
        }
      }
      g.assigned.fill(-2);
      g.version += 1;
    }
  }

  /**
   * Choose what to draw.
   * @param camera   the view camera
   * @param detail   >1 shows more detail (e.g. 2 for the still frame after moving)
   * @param options  { cull: false, level: n } forces every instance to one level (shadow pass)
   */
  update(camera, detail = 1, options = {}) {
    const cull = options.cull !== false;
    const forced = options.level;
    if (cull) {
      _projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      _frustum.setFromProjectionMatrix(_projView);
    }
    const cam = camera.position;
    const k = 1 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const t1 = this.threshold / detail;
    const stats = { triangles: 0, visible: 0, culled: 0, total: 0 };

    for (const g of this.groups) {
      const nLevels = g.levels.length;
      const counts = new Array(nLevels).fill(0);
      let changed = false;
      const s = g.spheres;
      for (let i = 0; i < g.count; i++) {
        const o = i * 4;
        let level;
        _sphere.center.set(s[o], s[o + 1], s[o + 2]);
        _sphere.radius = s[o + 3];
        if (cull && !_frustum.intersectsSphere(_sphere)) {
          level = -1;
        } else if (forced !== undefined) {
          level = Math.min(forced, nLevels - 1);
        } else {
          // fraction of the screen height the instance's bounding sphere covers
          const dist = Math.max(_sphere.center.distanceTo(cam), 1e-9);
          const cover = (_sphere.radius * k) / dist;
          level = 0;
          let t = t1;
          while (level < nLevels - 1 && cover < t) {
            level += 1;
            t *= 0.5;
          }
        }
        if (g.assigned[i] !== level) changed = true;
        g.assigned[i] = level;
        if (level < 0) {
          stats.culled += 1;
          continue;
        }
        if (g.instanced) {
          const slot = counts[level];
          g.slots[level][slot] = i;
          for (const m of g.levels[level]) {
            m.instanceMatrix.array.set(g.matrices.subarray(i * 16, i * 16 + 16), slot * 16);
            if (g.colors && m.instanceColor) m.instanceColor.array.set(g.colors.subarray(i * 3, i * 3 + 3), slot * 3);
          }
        }
        counts[level] += 1;
        stats.visible += 1;
        stats.triangles += g.tris[level];
      }
      stats.total += g.count;
      if (changed) g.version += 1;
      for (let l = 0; l < nLevels; l++) {
        for (const m of g.levels[l]) {
          m.visible = counts[l] > 0;
          if (g.instanced) {
            m.count = counts[l];
            // only upload when the packing changed
            if (changed && counts[l] > 0) {
              m.instanceMatrix.needsUpdate = true;
              if (m.instanceColor) m.instanceColor.needsUpdate = true;
            }
          }
        }
      }
    }
    this.stats = stats;
    return stats;
  }
}
