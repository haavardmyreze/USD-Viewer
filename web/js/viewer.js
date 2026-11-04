import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FlyControls } from './flyControls.js';
import { applyTriplanarToScene } from './triplanar.js';

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x2a2d33);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.7;
    pmrem.dispose();

    this.sun = new THREE.DirectionalLight(0xffffff, 1.2);
    this.sun.position.set(0.5, 1, 0.3);
    this.scene.add(this.sun, this.sun.target);
    this.scene.add(new THREE.HemisphereLight(0xdde6ff, 0x3a3326, 0.3));

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 10000);
    this.camera.position.set(0, 1.7, 5);
    this.controls = new FlyControls(this.camera, canvas);

    this.model = null;
    this.grid = null;
    this.bounds = new THREE.Box3(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 2, 5));
    this.forceDoubleSided = false;
    this._originalSides = new Map();

    this.loader = new GLTFLoader();
    this.clock = new THREE.Clock();

    window.addEventListener('resize', () => this._resize());
    this._resize();
    this.renderer.setAnimationLoop(() => this._tick());
  }

  async loadGlb(url, onProgress) {
    const gltf = await this.loader.loadAsync(url, (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    });
    applyTriplanarToScene(gltf.scene);
    this.setModel(gltf.scene);
    return gltf;
  }

  setModel(object) {
    if (this.model) {
      this.scene.remove(this.model);
      disposeObject(this.model);
    }
    this._originalSides.clear();
    this.model = object;
    this.scene.add(object);
    object.updateMatrixWorld(true);

    this.bounds = new THREE.Box3().setFromObject(object, true);
    if (this.bounds.isEmpty()) this.bounds.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    this._fitClipPlanes();
    this._placeSun();
    this.setDoubleSided(this.forceDoubleSided);
    if (this.grid) this.setGrid(true);
    this.frameScene();
  }

  stats() {
    let meshes = 0;
    let triangles = 0;
    this.model?.traverse((o) => {
      if (!o.isMesh) return;
      meshes += 1;
      const g = o.geometry;
      triangles += (g.index ? g.index.count : g.attributes.position.count) / 3;
    });
    const size = this.bounds.getSize(new THREE.Vector3());
    return { meshes, triangles, size };
  }

  /** Put the camera outside the scene looking at its centre, eye-level-ish. */
  frameScene() {
    const center = this.bounds.getCenter(new THREE.Vector3());
    const size = this.bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 0.1);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const dist = radius / Math.sin(fov / 2) * 0.9;

    const dir = new THREE.Vector3(0.6, 0.45, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.controls.lookAt(center);
    this.controls.stop();
    // crossing the scene in ~10 s at base speed feels right for most sizes
    this.controls.setSpeed(Math.max(radius / 5, 0.05));
  }

  setGrid(on) {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
      this.grid = null;
    }
    if (!on) return;
    const size = this.bounds.getSize(new THREE.Vector3());
    const extent = Math.max(size.x, size.z, 1) * 1.5;
    // grid lines on a "nice" metre spacing
    const step = Math.pow(10, Math.floor(Math.log10(extent / 10)));
    const divisions = Math.min(Math.ceil(extent / step), 1000);
    this.grid = new THREE.GridHelper(divisions * step, divisions, 0x6a7080, 0x41454e);
    const c = this.bounds.getCenter(new THREE.Vector3());
    // sit just below the lowest geometry so the grid never z-fights a floor
    this.grid.position.set(c.x, this.bounds.min.y - size.length() * 1e-3, c.z);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.6;
    this.scene.add(this.grid);
  }

  setDoubleSided(on) {
    this.forceDoubleSided = on;
    this.model?.traverse((o) => {
      if (!o.isMesh) return;
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
        if (!this._originalSides.has(m)) this._originalSides.set(m, m.side);
        m.side = on ? THREE.DoubleSide : this._originalSides.get(m);
        m.needsUpdate = true;
      }
    });
  }

  // -- internals -----------------------------------------------------------

  _fitClipPlanes() {
    const radius = Math.max(this.bounds.getSize(new THREE.Vector3()).length() / 2, 0.1);
    this.camera.near = Math.max(radius * 1e-5, 1e-4);
    this.camera.far = radius * 200;
    this.camera.updateProjectionMatrix();
  }

  _placeSun() {
    const c = this.bounds.getCenter(new THREE.Vector3());
    const r = this.bounds.getSize(new THREE.Vector3()).length();
    this.sun.target.position.copy(c);
    this.sun.position.copy(c).add(new THREE.Vector3(0.5, 1, 0.3).multiplyScalar(r));
  }

  _resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.controls.update(dt);
    this.renderer.render(this.scene, this.camera);
    this.onFrame?.();
  }
}

function disposeObject(root) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry?.dispose();
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (!m) continue;
      for (const v of Object.values(m)) if (v && v.isTexture) v.dispose();
      m.dispose();
    }
  });
}
