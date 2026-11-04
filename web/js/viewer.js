import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Navigation } from './navigation.js';
import { HELPER_LAYER, Picker } from './picking.js';
import { LodManager } from './lod.js';
import { PivotGizmo } from './pivot.js';
import { Selection } from './selection.js';
import { ShadingMaterials } from './shading.js';
import { applyTriplanarToScene } from './triplanar.js';
import { applyShaderGraphs, workingMaterial } from './mtlx.js';

const SUN_DIRECTION = new THREE.Vector3(0.55, 0.7, 0.4).normalize();
/** Detail presets: multiplier on the screen size at which lower LODs kick in. */
export const DETAIL_LEVELS = { fast: 0.5, balanced: 1, high: 2.5, full: Infinity };
const SHADOW_TRIANGLE_BUDGET = 6e6;

/**
 * Renders on demand: a frame is drawn only when the camera or a setting changes.
 * While the camera moves, detail and resolution adapt to keep it smooth; once it
 * stops, one full-quality frame is drawn.
 */
export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: true, powerPreference: 'high-performance',
    });
    this.renderer = renderer;
    this.maxPixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.interactiveRatio = this.maxPixelRatio;
    renderer.setPixelRatio(this.maxPixelRatio);
    renderer.setClearColor(0x000000, 0); // the page background shows through
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false; // static scene: shadows are rendered once

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.55;
    pmrem.dispose();

    this.sun = new THREE.DirectionalLight(0xfff3e2, 2.0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0002;
    this.scene.add(this.sun, this.sun.target);
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x4a4236, 0.35));

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    this.camera.layers.enable(HELPER_LAYER);
    this.camera.position.set(6, 4, 8);

    this.picker = new Picker(renderer);
    this.nav = new Navigation(this.camera, canvas, (x, y) => this.pick(x, y));
    this.nav.lookAt(new THREE.Vector3());
    this.lod = new LodManager();
    this.selection = new Selection(this.lod);
    this.pivot = new PivotGizmo();
    this.scene.add(this.pivot.object);

    this.model = null;
    this.bounds = new THREE.Box3(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 2, 5));
    this.radius = 8;
    this.options = {
      grid: true, shadows: true, doubleSided: false, detail: 'balanced', shading: 'material', matcap: 'clay',
    };
    this._originalSides = new Map();
    this._originalMaterials = new Map(); // mesh -> its own material, while shading replaces it
    this._shading = new ShadingMaterials();
    this.ground = null;
    this.grid = null;

    this.loader = new GLTFLoader();
    this.clock = new THREE.Clock();
    this._lastView = new THREE.Matrix4();
    this._lastMove = 0;
    this._dirty = true;
    this._needsRefine = true;
    this._shadowsDirty = false;
    this._frameMs = 16;
    this._lastRenderAt = 0;
    this.lastFrame = { triangles: 0, refined: false };

    this._buildHelpers();
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
    renderer.setAnimationLoop(() => this._tick());
  }

  // -- loading ---------------------------------------------------------------

  /**
   * @param options.keepView  keep the camera where it is (e.g. after switching a
   *                          variant), instead of framing the new model
   */
  async loadGlb(url, onProgress, options = {}) {
    const gltf = await this.loader.loadAsync(url, (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    });
    gltf.userData.shaderGraphs = await applyShaderGraphs(gltf);
    for (const s of gltf.scenes) applyTriplanarToScene(s);
    await resolveInstanceIds(gltf);
    this.setModel(gltf, options);
    return gltf;
  }

  setModel(gltf, { keepView = false } = {}) {
    if (this.model) {
      this.selection.clear();
      this.pivot.hide();
      this._restoreMaterials();
      this.scene.remove(this.model);
      disposeObject(this.model);
    }
    this._originalSides.clear();
    this._originalMaterials.clear();
    const model = new THREE.Group();
    model.name = 'model';
    const main = gltf.scene;
    model.add(main);
    // simplified levels of detail live in a second glTF scene
    for (const s of gltf.scenes) if (s !== main && s.name === 'usdv_lods') model.add(s);
    model.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
      }
    });
    this.model = model;
    this.scene.add(model);
    model.updateMatrixWorld(true);
    // USD stage space -> view: the converter's up-axis and units correction
    this.stageToWorld = (main.getObjectByName('usd_stage_root') ?? main).matrixWorld.clone();
    this.lod.setup(model);
    const pickable = [];
    model.traverse((o) => { if (o.isMesh) pickable.push(o); });
    this.picker.register(pickable);
    this.selection.build(model);

    // bounds: plain meshes + every managed instance (ignores the LOD copies)
    const managed = new Set(this.lod.groups.flatMap((g) => g.levels.flat()));
    const box = new THREE.Box3();
    main.traverse((o) => {
      if (!o.isMesh || managed.has(o)) return;
      if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
      box.union(o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld));
    });
    this.lod.expandBox(box);
    if (box.isEmpty()) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    this.bounds = box;
    this.radius = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, 0.05);
    this.fullTriangles = this.lod.groups.reduce((n, g) => n + g.count * g.tris[0], 0);
    main.traverse((o) => {
      if (o.isMesh && !managed.has(o)) {
        const g = o.geometry;
        this.fullTriangles += (g.index ? g.index.count : g.attributes.position.count) / 3;
      }
    });

    this.nav.setSceneRadius(this.radius);
    this._placeHelpers();
    this._placeSun();
    this.setDoubleSided(this.options.doubleSided); // also applies the shading mode
    if (!keepView) this.frameScene(false);
    this.nav.update(0);
    this._warmUp();
    this.invalidate();
  }

  /** Compile shaders and upload every mesh and texture now, not mid-flight. */
  _warmUp() {
    this.lod.showAllLevels();
    this.camera.updateMatrixWorld();
    this._updateClipPlanes();
    // compile the scene's own materials too while another shading mode shows, and put
    // the plain glTF material back where a generated shader did not compile
    const shown = new Map();
    for (const [mesh, own] of this._originalMaterials) {
      shown.set(mesh, mesh.material);
      mesh.material = own;
    }
    this.renderer.compile(this.scene, this.camera);
    this.failedShaders = new Set();
    for (const [mesh, own] of this._originalMaterials) {
      const ok = workingMaterial(this.renderer, own);
      for (const [a, b] of (Array.isArray(own) ? own.map((m, i) => [m, ok[i]]) : [[own, ok]])) {
        if (a !== b) this.failedShaders.add(a.name);
      }
      this._originalMaterials.set(mesh, ok);
      mesh.material = shown.get(mesh) === own ? ok : shown.get(mesh);
    }
    if (this.failedShaders.size) this.renderer.compile(this.scene, this.camera);
    this.renderer.render(this.scene, this.camera);
  }

  // -- view ------------------------------------------------------------------

  /** Camera pose that shows the whole scene from a three-quarter view. */
  frameScene(animate = true) {
    const dir = new THREE.Vector3(0.62, 0.42, 1).normalize();
    this._frameSphere(this.bounds.getCenter(new THREE.Vector3()), this.radius, dir, animate ? 0.6 : 0, 0.9);
  }

  /** Fly to a selection ({ path, instance? }), keeping the viewing direction. */
  frameSelection(sel, animate = true) {
    const box = this.selection.bounds(sel);
    if (!box) return false;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const dir = this.camera.getWorldDirection(new THREE.Vector3()).negate();
    this._frameSphere(sphere.center, Math.max(sphere.radius, this.radius * 1e-4), dir, animate ? 0.5 : 0, 1.15);
    return true;
  }

  _frameSphere(center, radius, dir, duration, margin) {
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const aspect = Number.isFinite(this.camera.aspect) && this.camera.aspect > 0 ? this.camera.aspect : 1.6;
    const fitH = radius / Math.sin(fov / 2);
    const fitW = radius / Math.sin(Math.atan(Math.tan(fov / 2) * aspect));
    const dist = Math.max(fitH, fitW) * margin;
    this.nav.flyTo(center.clone().addScaledVector(dir, dist), center, duration);
    this.invalidate();
  }

  /** Highlight a selection ({ path, instance? }) or nothing (null). */
  select(sel) {
    this.selection.set(sel);
    this.selection.sync();
    this.pivot.hide();
    this.invalidate();
  }

  /**
   * What the server knows about the current selection (GET .../selection):
   * its pivot, drawn as a tripod, and the other copies of its instance, which
   * get a fainter highlight.
   */
  showSelectionInfo(info) {
    if (!this.model) return;
    if (info.pivot) {
      const frame = new THREE.Matrix4().fromArray(info.pivot.matrix); // USD row-major = three column-major
      frame.premultiply(this.stageToWorld);
      // sized to the selection, so it reaches about to the edge of what it moves
      const box = this.selection.bounds(this.selection.current);
      const radius = box ? box.getBoundingSphere(new THREE.Sphere()).radius : 0;
      const length = radius > 0 ? radius * 0.8 : this.radius * 0.05;
      this.pivot.show(frame, new THREE.Vector3().fromArray(info.pivot.point), length);
    } else this.pivot.hide();
    this.selection.setPeers(info.instances);
    this.selection.sync();
    this.invalidate();
  }

  /** The prim under canvas pixel (x, y): { path, instance, proto, point } or null. */
  pickObject(x, y) {
    if (!this.model) return null;
    this.camera.updateMatrixWorld();
    const hit = this.picker.pickObject(this.scene, this.camera, x, y);
    if (!hit?.object) return null;
    const info = this.selection.describe(hit.object, hit.slot);
    return info ? { ...info, point: hit.point } : null;
  }

  /** World point under canvas pixel (x, y), or null. */
  pick(x, y) {
    if (!this.model) return null;
    this.camera.updateMatrixWorld();
    return this.picker.pick(this.scene, this.camera, x, y);
  }

  invalidate() {
    this._dirty = true;
    this._needsRefine = true;
  }

  setGrid(on) {
    this.options.grid = on;
    if (this.grid) this.grid.visible = on;
    this.invalidate();
  }

  setShadows(on) {
    this.options.shadows = on;
    this.sun.castShadow = on;
    if (this.ground) this.ground.visible = on;
    this._shadowsDirty = on;
    this.invalidate();
  }

  setExposure(v) {
    this.renderer.toneMappingExposure = v;
    this.invalidate();
  }

  setDetail(name) {
    if (name in DETAIL_LEVELS) this.options.detail = name;
    this.invalidate();
  }

  setDoubleSided(on) {
    this.options.doubleSided = on;
    this.model?.traverse((o) => {
      if (!o.isMesh || o.userData.usdvOverlay) return;
      const own = this._originalMaterials.get(o) ?? o.material;
      for (const m of Array.isArray(own) ? own : [own]) {
        if (!this._originalSides.has(m)) this._originalSides.set(m, m.side);
        m.side = on ? THREE.DoubleSide : this._originalSides.get(m);
        m.needsUpdate = true;
      }
    });
    this._applyShading(); // shading materials follow each mesh's sides
    this.invalidate();
  }

  /**
   * Viewport shading, as in Blender: 'material' (the scene's own), 'matcap'
   * 'plain' (no material: neutral grey, lit) or 'matcap' (one lit-sphere look
   * on everything, `look` picks which).
   */
  setShading(mode, look = this.options.matcap) {
    this.options.shading = mode;
    this.options.matcap = look;
    this._applyShading();
    this.invalidate();
  }

  _applyShading() {
    if (!this.model) return;
    const { shading, matcap } = this.options;
    this.model.traverse((o) => {
      if (!o.isMesh || o.userData.usdvOverlay) return;
      if (!this._originalMaterials.has(o)) this._originalMaterials.set(o, o.material);
      const own = this._originalMaterials.get(o);
      if (shading === 'material') {
        o.material = own;
        return;
      }
      const side = (Array.isArray(own) ? own[0] : own)?.side ?? THREE.FrontSide;
      const m = this._shading.get(shading, matcap, side);
      o.material = Array.isArray(own) ? own.map(() => m) : m;
    });
  }

  _restoreMaterials() {
    for (const [mesh, own] of this._originalMaterials) mesh.material = own;
    this._originalMaterials.clear();
  }

  stats() {
    return {
      fullTriangles: this.fullTriangles || 0,
      drawnTriangles: this.lastFrame.triangles,
      instances: this.lod.stats.total,
      visibleInstances: this.lod.stats.visible,
      size: this.bounds.getSize(new THREE.Vector3()),
      pixelRatio: this.renderer.getPixelRatio(),
      frameMs: this._frameMs,
    };
  }

  // -- helpers (ground shadow + grid) -------------------------------------------

  _buildHelpers() {
    const plane = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    this.ground = new THREE.Mesh(plane, new THREE.ShadowMaterial({ opacity: 0.3, depthWrite: false }));
    this.ground.receiveShadow = true;
    this.ground.layers.set(HELPER_LAYER);
    this.ground.renderOrder = -2;

    this.grid = new THREE.Mesh(plane, new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uSpacing: { value: 1 },
        uFade: { value: 50 },
        uColor: { value: new THREE.Color(0x9aa3b5) },
        uOpacity: { value: 0.32 },
      },
      vertexShader: /* glsl */`
        varying vec3 vWorld;
        void main() {
          vec4 w = modelMatrix * vec4( position, 1.0 );
          vWorld = w.xyz;
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: /* glsl */`
        varying vec3 vWorld;
        uniform float uSpacing;
        uniform float uFade;
        uniform float uOpacity;
        uniform vec3 uColor;
        float line( float spacing ) {
          vec2 c = vWorld.xz / spacing;
          vec2 g = abs( fract( c - 0.5 ) - 0.5 ) / fwidth( c );
          return 1.0 - min( min( g.x, g.y ), 1.0 );
        }
        void main() {
          float d = distance( cameraPosition.xz, vWorld.xz );
          float fade = 1.0 - smoothstep( uFade * 0.25, uFade, d );
          float a = max( line( uSpacing ) * 0.4, line( uSpacing * 10.0 ) ) * uOpacity * fade;
          if ( a < 0.004 ) discard;
          gl_FragColor = vec4( uColor, a );
        }`,
    }));
    this.grid.layers.set(HELPER_LAYER);
    this.grid.renderOrder = -1;
    this.scene.add(this.ground, this.grid);
  }

  _placeHelpers() {
    const c = this.bounds.getCenter(new THREE.Vector3());
    const r = this.radius;
    const y = this.bounds.min.y - r * 2e-4;
    for (const h of [this.ground, this.grid]) {
      h.position.set(c.x, y, c.z);
      h.scale.setScalar(r * 40);
    }
    // grid spacing: a power of ten that gives ~10-100 lines across the scene
    this.grid.material.uniforms.uSpacing.value = Math.pow(10, Math.floor(Math.log10(Math.max(r / 5, 1e-3))));
    this.grid.material.uniforms.uFade.value = r * 6;
    this.grid.visible = this.options.grid;
    this.ground.visible = this.options.shadows;
  }

  _placeSun() {
    const c = this.bounds.getCenter(new THREE.Vector3());
    const r = this.radius;
    this.sun.target.position.copy(c);
    this.sun.position.copy(c).addScaledVector(SUN_DIRECTION, r * 2);
    const cam = this.sun.shadow.camera;
    cam.left = -r * 1.02;
    cam.right = r * 1.02;
    cam.top = r * 1.02;
    cam.bottom = -r * 1.02;
    cam.near = r * 0.5;
    cam.far = r * 3.5;
    cam.updateProjectionMatrix();
    // about one shadow-map texel, so surfaces don't shadow themselves
    this.sun.shadow.normalBias = (2 * r) / this.sun.shadow.mapSize.x * 1.5;
    this.sun.updateMatrixWorld();
    this.sun.target.updateMatrixWorld();
    this._shadowsDirty = this.options.shadows;
  }

  /** Render the shadow map once, with every instance at a level that fits the budget. */
  _renderShadows() {
    const groups = this.lod.groups;
    const maxLevels = Math.max(1, ...groups.map((g) => g.levels.length));
    let level = 0;
    while (level < maxLevels - 1
      && groups.reduce((n, g) => n + g.count * g.tris[Math.min(level, g.levels.length - 1)], 0) > SHADOW_TRIANGLE_BUDGET) {
      level += 1;
    }
    this.lod.update(this.camera, 1, { cull: false, level });
    this.renderer.shadowMap.needsUpdate = true;
    this.renderer.render(this.scene, this.camera);
    this._shadowsDirty = false;
  }

  // -- frame loop ---------------------------------------------------------------

  _resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    if (!w || !h) return; // hidden (e.g. background tab): keep the last size
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  _updateClipPlanes() {
    // keep the near plane as far out as possible for depth precision
    const r = this.radius;
    const outside = this.bounds.distanceToPoint(this.camera.position);
    const near = THREE.MathUtils.clamp(outside * 0.3, r * 2e-4, r * 0.5);
    const far = outside + r * 4;
    if (Math.abs(near - this.camera.near) / near > 0.05 || Math.abs(far - this.camera.far) / far > 0.05) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  _setPixelRatio(ratio) {
    if (Math.abs(ratio - this.renderer.getPixelRatio()) < 0.02) return;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight, false);
  }

  _tick() {
    const now = performance.now();
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.nav.update(dt);
    this.camera.updateMatrixWorld();
    if (!this._lastView.equals(this.camera.matrixWorld)) {
      this._lastMove = now;
      this._dirty = true;
      this._needsRefine = true;
    }

    const refine = !this._dirty && this._needsRefine && now - this._lastMove > 200;
    if (!this._dirty && !refine) {
      this.onFrame?.(false);
      return;
    }

    if (refine) {
      this._setPixelRatio(this.maxPixelRatio);
    } else {
      // while moving: trade resolution for frame rate (consecutive frames only)
      const interval = now - this._lastRenderAt;
      if (interval < 250) {
        this._frameMs += (interval - this._frameMs) * 0.2;
        if (this._frameMs > 30 && this.interactiveRatio > 0.5) this.interactiveRatio *= 0.85;
        else if (this._frameMs < 18 && this.interactiveRatio < this.maxPixelRatio) {
          this.interactiveRatio = Math.min(this.maxPixelRatio, this.interactiveRatio * 1.08);
        }
      }
      this._setPixelRatio(this.interactiveRatio);
    }

    this._updateClipPlanes();
    if (this._shadowsDirty && this.options.shadows && this.model) this._renderShadows();
    const detail = DETAIL_LEVELS[this.options.detail] ?? 1;
    this.lod.update(this.camera, refine ? Math.max(detail * 2, 2) : detail);
    this.selection.sync();
    this.renderer.render(this.scene, this.camera);

    this.lastFrame = { triangles: this.renderer.info.render.triangles, refined: refine };
    this._lastView.copy(this.camera.matrixWorld);
    this._lastRenderAt = now;
    this._dirty = false;
    if (refine) this._needsRefine = false;
    this.onFrame?.(true);
  }
}

/**
 * Instance -> authored PointInstancer index tables are glTF accessors named in
 * node extras (usdInstanceIds); swap each index for its loaded Float32Array.
 */
async function resolveInstanceIds(gltf) {
  const pending = new Map();
  const nodes = [];
  for (const s of gltf.scenes) {
    s.traverse((o) => {
      if (typeof o.userData?.usdInstanceIds === 'number') nodes.push(o);
    });
  }
  for (const o of nodes) {
    const index = o.userData.usdInstanceIds;
    if (!pending.has(index)) pending.set(index, gltf.parser.getDependency('accessor', index));
  }
  await Promise.all(nodes.map(async (o) => {
    const attr = await pending.get(o.userData.usdInstanceIds);
    o.userData.usdInstanceIds = attr?.array instanceof Float32Array ? attr.array : null;
  }));
}

function disposeObject(root) {
  const textures = new Set();
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry?.dispose();
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (!m) continue;
      for (const v of Object.values(m)) if (v && v.isTexture) textures.add(v);
      m.dispose();
    }
    if (o.isInstancedMesh) o.dispose();
  });
  textures.forEach((t) => t.dispose());
}
