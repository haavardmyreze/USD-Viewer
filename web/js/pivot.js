// The selected prim's pivot: an axis tripod at its pivot point, turned with its
// frame (X red, Y green, Z blue, the prim's own USD axes).
//
// Drawn on top of everything at a constant size on screen, on the helper layer
// so it is never picked and casts no shadow.
import * as THREE from 'three';
import { HELPER_LAYER } from './picking.js';

const AXES = [
  { dir: new THREE.Vector3(1, 0, 0), color: 0xff3b55 },
  { dir: new THREE.Vector3(0, 1, 0), color: 0x7ed321 },
  { dir: new THREE.Vector3(0, 0, 1), color: 0x2f8cff },
];
/** Length of an axis on screen, in CSS pixels. */
const SCREEN_LENGTH = 64;

function material(color, opacity = 1) {
  return new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, depthTest: false, depthWrite: false, toneMapped: false,
  });
}

export class PivotGizmo {
  constructor() {
    const root = new THREE.Group();
    root.name = 'usdv pivot';
    const shaft = new THREE.CylinderGeometry(0.014, 0.014, 0.8, 8).translate(0, 0.4, 0);
    const tip = new THREE.ConeGeometry(0.05, 0.2, 14).translate(0, 0.9, 0);
    const up = new THREE.Vector3(0, 1, 0);
    for (const { dir, color } of AXES) {
      const axis = new THREE.Group();
      axis.quaternion.setFromUnitVectors(up, dir);
      const m = material(color);
      axis.add(new THREE.Mesh(shaft, m), new THREE.Mesh(tip, m));
      root.add(axis);
    }
    root.add(new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 10), material(0xffffff, 0.95)));
    root.traverse((o) => {
      o.layers.set(HELPER_LAYER);
      o.renderOrder = 20;
      o.frustumCulled = false;
    });
    root.visible = false;
    this.object = root;
    this._signs = new THREE.Vector3(1, 1, 1);
    this._scale = new THREE.Vector3();
    this._toCamera = new THREE.Vector3();
    this._forward = new THREE.Vector3();
  }

  /**
   * Place the tripod: `frame` is the prim's local-to-world matrix (three.js
   * world), `point` its pivot in that frame.
   */
  show(frame, point) {
    const o = this.object;
    frame.decompose(new THREE.Vector3(), o.quaternion, this._scale);
    // keep a mirrored frame's flipped axes; drop its scale (the size is on screen)
    this._signs.set(Math.sign(this._scale.x) || 1, Math.sign(this._scale.y) || 1, Math.sign(this._scale.z) || 1);
    o.position.copy(point).applyMatrix4(frame);
    o.visible = true;
  }

  hide() {
    this.object.visible = false;
  }

  /** Keep the tripod the same size on screen. Call before each render. */
  update(camera, viewportHeight) {
    const o = this.object;
    if (!o.visible) return;
    camera.getWorldDirection(this._forward);
    const depth = Math.max(this._toCamera.copy(o.position).sub(camera.position).dot(this._forward), camera.near);
    const worldPerPixel = (2 * depth * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / Math.max(viewportHeight, 1);
    const size = SCREEN_LENGTH * worldPerPixel;
    o.scale.copy(this._signs).multiplyScalar(size);
    o.updateMatrixWorld(true);
  }
}
