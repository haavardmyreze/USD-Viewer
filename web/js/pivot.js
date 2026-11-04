// The selected prim's pivot: an axis tripod at its pivot point, turned with its
// frame (X red, Y green, Z blue, the prim's own USD axes).
//
// A fixed size in world space (it does not change as the camera zooms), drawn on
// top of everything, on the helper layer so it is never picked and casts no shadow.
import * as THREE from 'three';
import { HELPER_LAYER } from './picking.js';

const AXES = [
  { dir: new THREE.Vector3(1, 0, 0), color: 0xff3b55 },
  { dir: new THREE.Vector3(0, 1, 0), color: 0x7ed321 },
  { dir: new THREE.Vector3(0, 0, 1), color: 0x2f8cff },
];

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
    this._scale = new THREE.Vector3();
  }

  /**
   * Place the tripod: `frame` is the prim's local-to-world matrix (three.js
   * world), `point` its pivot in that frame, `length` an axis's length in world
   * units.
   */
  show(frame, point, length) {
    const o = this.object;
    frame.decompose(new THREE.Vector3(), o.quaternion, this._scale);
    // keep a mirrored frame's flipped axes, but not its scale: the size is given
    o.scale.set(Math.sign(this._scale.x) || 1, Math.sign(this._scale.y) || 1, Math.sign(this._scale.z) || 1)
      .multiplyScalar(length);
    o.position.copy(point).applyMatrix4(frame);
    o.visible = true;
    o.updateMatrixWorld(true);
  }

  hide() {
    this.object.visible = false;
  }
}
