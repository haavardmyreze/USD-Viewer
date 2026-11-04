// GPU picking: what surface point is under the cursor?
//
// Renders view-space depth for the single pixel under the cursor into a float
// render target. That works for any amount of geometry and instancing, unlike
// CPU raycasting against millions of triangles.
import * as THREE from 'three';

const vertexShader = /* glsl */`
varying float vViewDepth;
void main() {
  vec4 p = vec4( position, 1.0 );
  #ifdef USE_INSTANCING
    p = instanceMatrix * p;
  #endif
  vec4 mv = modelViewMatrix * p;
  vViewDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const fragmentShader = /* glsl */`
varying float vViewDepth;
void main() {
  gl_FragColor = vec4( vViewDepth, 1.0, 0.0, 1.0 );
}
`;

/** Objects on this layer are drawn but never picked (ground, grid, helpers). */
export const HELPER_LAYER = 1;

export class Picker {
  constructor(renderer) {
    this.renderer = renderer;
    this.target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType, depthBuffer: true });
    this.material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, side: THREE.DoubleSide });
    this.pixel = new Float32Array(4);
    this.camera = new THREE.PerspectiveCamera();
    this._ndc = new THREE.Vector3();
    this._dir = new THREE.Vector3();
    this._forward = new THREE.Vector3();
  }

  /**
   * World-space point under (x, y) in CSS pixels of the canvas, or null.
   * `beforeRender` lets the caller prepare the scene (e.g. LOD selection).
   */
  pick(scene, camera, x, y) {
    const r = this.renderer;
    const canvas = r.domElement;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (x < 0 || y < 0 || x >= w || y >= h) return null;

    const cam = this.camera.copy(camera);
    cam.layers.set(0);
    cam.setViewOffset(w, h, Math.floor(x), Math.floor(y), 1, 1);
    cam.updateProjectionMatrix();

    const prev = {
      target: r.getRenderTarget(),
      override: scene.overrideMaterial,
      background: scene.background,
      clearAlpha: r.getClearAlpha(),
      clearColor: r.getClearColor(new THREE.Color()),
      shadowAuto: r.shadowMap.autoUpdate,
    };
    scene.overrideMaterial = this.material;
    scene.background = null;
    r.shadowMap.autoUpdate = false;
    r.setRenderTarget(this.target);
    r.setClearColor(0x000000, 0);
    r.clear();
    r.render(scene, cam);
    r.readRenderTargetPixels(this.target, 0, 0, 1, 1, this.pixel);

    r.setRenderTarget(prev.target);
    r.setClearColor(prev.clearColor, prev.clearAlpha);
    r.shadowMap.autoUpdate = prev.shadowAuto;
    scene.overrideMaterial = prev.override;
    scene.background = prev.background;

    if (this.pixel[1] < 0.5) return null; // nothing drawn there
    const depth = this.pixel[0];

    // ray through the pixel centre, scaled so its view-space depth matches
    this._ndc.set(((x + 0.5) / w) * 2 - 1, -((y + 0.5) / h) * 2 + 1, 0.5).unproject(camera);
    this._dir.copy(this._ndc).sub(camera.position).normalize();
    camera.getWorldDirection(this._forward);
    const along = depth / Math.max(this._dir.dot(this._forward), 1e-6);
    return camera.position.clone().addScaledVector(this._dir, along);
  }

  dispose() {
    this.target.dispose();
    this.material.dispose();
  }
}
