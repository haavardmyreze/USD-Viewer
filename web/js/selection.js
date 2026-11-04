// Selection: which geometry belongs to a USD prim, and highlighting it.
//
// The converter tags every glTF node with the prim the scene graph selects for
// it (extras.usdPath): the gprim itself, or for PointInstancer instances the
// instancer, with each instance's authored index (usdInstanceIds) and the gprim
// inside the prototype (usdProto).
//
// Highlights are overlays: a child of each selected mesh that shares its
// geometry buffers and instance matrices and draws a translucent tint on top.
// As a child it follows the mesh's LOD visibility; per-instance selections are
// a mask attribute repacked whenever the LOD manager repacks the instances.
import * as THREE from 'three';
import { HELPER_LAYER } from './picking.js';

const TINT = new THREE.Color(0x0a84ff); // refgraph's selection blue

const vertexShader = /* glsl */`
#include <common>
#ifdef USE_SEL_MASK
attribute float usdvSel;
#endif
varying vec3 vViewNormal;
void main() {
  #include <begin_vertex>
  #include <project_vertex>
  vec3 n = normal;
  #ifdef USE_INSTANCING
    n = mat3( instanceMatrix ) * n;
  #endif
  vViewNormal = normalize( normalMatrix * n );
  #ifdef USE_SEL_MASK
    if ( usdvSel < 0.5 ) gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 ); // outside the clip volume
  #endif
}
`;

const fragmentShader = /* glsl */`
uniform vec3 uTint;
varying vec3 vViewNormal;
void main() {
  // a little stronger at grazing angles, so silhouettes read
  float rim = 1.0 - abs( normalize( vViewNormal ).z );
  gl_FragColor = vec4( uTint, 0.32 + 0.38 * rim * rim );
}
`;

function makeMaterial(masked, side) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: { uTint: { value: TINT } },
    defines: masked ? { USE_SEL_MASK: '' } : {},
    transparent: true,
    depthWrite: false,
    depthFunc: THREE.LessEqualDepth,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -2,
    side,
  });
}

function trianglesOf(mesh) {
  const g = mesh.geometry;
  return (g.index ? g.index.count : g.attributes.position.count) / 3;
}

export class Selection {
  constructor(lod) {
    this.lod = lod;
    this.entries = [];
    this.byMesh = new Map();
    this.overlays = new Map();
    this.active = [];
    this.current = null;
    this.materials = new Map();
  }

  /** Index the nodes of a freshly loaded model. */
  build(model) {
    this.clear();
    this.entries = [];
    this.byMesh = new Map();
    this.overlays = new Map();
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

  /** Highlight a selection ({ path, instance? }) or nothing (null). */
  set(sel) {
    this.clear();
    this.current = sel;
    for (const { entry, instances } of this._match(sel)) {
      for (const mesh of entry.meshes) {
        const masked = !!(instances && mesh.isInstancedMesh);
        const overlay = this._overlayFor(mesh, masked);
        if (masked) {
          const chosen = new Uint8Array(this._count(mesh));
          for (const i of instances) chosen[i] = 1;
          overlay.userData.chosen = chosen;
          overlay.userData.version = -1;
        }
        overlay.visible = true;
        this.active.push(overlay);
      }
    }
  }

  /** Follow the LOD manager's latest instance packing. Call after lod.update(). */
  sync() {
    for (const o of this.active) o.userData.sync?.();
  }

  clear() {
    for (const o of this.active) o.visible = false;
    this.active = [];
    this.current = null;
  }

  _material(masked, side) {
    const key = `${masked}:${side}`;
    if (!this.materials.has(key)) this.materials.set(key, makeMaterial(masked, side));
    return this.materials.get(key);
  }

  _overlayFor(mesh, masked) {
    const key = `${masked}`;
    let byKind = this.overlays.get(mesh);
    if (!byKind) {
      byKind = {};
      this.overlays.set(mesh, byKind);
    }
    if (byKind[key]) return byKind[key];

    const src = mesh.geometry;
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(src.index);
    geometry.setAttribute('position', src.attributes.position);
    if (src.attributes.normal) geometry.setAttribute('normal', src.attributes.normal);
    else geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(src.attributes.position.count * 3), 3));
    for (const group of src.groups) geometry.addGroup(group.start, group.count, group.materialIndex);
    const side = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material)?.side ?? THREE.FrontSide;
    const material = this._material(masked, side);

    let overlay;
    if (mesh.isInstancedMesh) {
      overlay = new THREE.InstancedMesh(geometry, material, 1);
      overlay.instanceMatrix = mesh.instanceMatrix; // shared: same packing, same upload
      if (masked) {
        const sel = new THREE.InstancedBufferAttribute(new Float32Array(mesh.instanceMatrix.count), 1);
        sel.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('usdvSel', sel);
      }
      const lod = this.lod;
      // run by sync() after each LOD update and before the render: attributes
      // are uploaded before onBeforeRender, so a mask set there lags a frame
      overlay.userData.sync = () => {
        overlay.count = mesh.count;
        if (!masked) return;
        const managed = lod.groupOf.get(mesh);
        const version = managed ? managed.group.version : 0;
        if (overlay.userData.version === version) return;
        overlay.userData.version = version;
        const chosen = overlay.userData.chosen;
        const attr = geometry.attributes.usdvSel;
        for (let slot = 0; slot < mesh.count; slot++) {
          attr.array[slot] = chosen[lod.instanceInSlot(mesh, slot)] ? 1 : 0;
        }
        attr.needsUpdate = true;
      };
    } else {
      overlay = new THREE.Mesh(geometry, material);
    }
    overlay.name = 'usdv selection';
    overlay.userData.usdvOverlay = true;
    overlay.frustumCulled = false;
    overlay.castShadow = false;
    overlay.receiveShadow = false;
    overlay.renderOrder = 10;
    overlay.layers.set(HELPER_LAYER);
    overlay.visible = false;
    overlay.matrixAutoUpdate = false; // identity: the parent's transform is the mesh's
    mesh.add(overlay);
    overlay.updateMatrixWorld(true);
    byKind[key] = overlay;
    return overlay;
  }
}
