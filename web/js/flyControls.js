import * as THREE from 'three';

/**
 * First-person fly navigation.
 *
 *  - click the canvas to capture the mouse (pointer lock), Esc to release
 *  - or hold the right mouse button and drag to look without capturing
 *  - WASD / arrows move along the view direction, E/Q (or Space/C) move up/down
 *  - Shift = fast, Ctrl/Alt = slow, mouse wheel scales the base speed
 */
export class FlyControls extends EventTarget {
  constructor(camera, domElement) {
    super();
    this.camera = camera;
    this.dom = domElement;

    this.speed = 2; // metres per second
    this.lookSensitivity = 0.0022;
    this.damping = 12; // higher = snappier stops
    this.enabled = true;

    this.yaw = 0;
    this.pitch = 0;
    this.velocity = new THREE.Vector3();
    this.keys = new Set();
    this.dragging = false;

    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._forward = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._target = new THREE.Vector3();

    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onLockChange = this._onLockChange.bind(this);
    this._onBlur = () => this.keys.clear();

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('blur', this._onBlur);
    document.addEventListener('mousemove', this._onMouseMove);
    document.addEventListener('pointerlockchange', this._onLockChange);
    this.dom.addEventListener('pointerdown', this._onPointerDown);
    window.addEventListener('pointerup', this._onPointerUp);
    this.dom.addEventListener('wheel', this._onWheel, { passive: false });
    this.dom.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  get locked() {
    return document.pointerLockElement === this.dom;
  }

  /** Point the camera at a world position. */
  lookAt(target) {
    const dir = new THREE.Vector3().subVectors(target, this.camera.position).normalize();
    this.yaw = Math.atan2(-dir.x, -dir.z);
    this.pitch = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
    this._applyRotation();
  }

  setSpeed(speed) {
    this.speed = THREE.MathUtils.clamp(speed, 0.01, 100000);
    this.dispatchEvent(new Event('speedchange'));
  }

  stop() {
    this.velocity.set(0, 0, 0);
  }

  update(dt) {
    if (!this.enabled) return;
    const k = this.keys;
    const move = this._target.set(0, 0, 0);

    this.camera.getWorldDirection(this._forward);
    this._right.crossVectors(this._forward, this.camera.up).normalize();

    if (k.has('KeyW') || k.has('ArrowUp')) move.add(this._forward);
    if (k.has('KeyS') || k.has('ArrowDown')) move.sub(this._forward);
    if (k.has('KeyD') || k.has('ArrowRight')) move.add(this._right);
    if (k.has('KeyA') || k.has('ArrowLeft')) move.sub(this._right);
    if (k.has('KeyE') || k.has('Space')) move.y += 1;
    if (k.has('KeyQ') || k.has('KeyC')) move.y -= 1;

    let speed = this.speed;
    if (k.has('ShiftLeft') || k.has('ShiftRight')) speed *= 4;
    if (k.has('ControlLeft') || k.has('ControlRight') || k.has('AltLeft') || k.has('AltRight')) speed *= 0.25;

    if (move.lengthSq() > 0) move.normalize().multiplyScalar(speed);

    // exponential smoothing towards the target velocity
    const t = 1 - Math.exp(-this.damping * dt);
    this.velocity.lerp(move, t);
    if (move.lengthSq() === 0 && this.velocity.lengthSq() < 1e-8) this.velocity.set(0, 0, 0);

    this.camera.position.addScaledVector(this.velocity, dt);
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('blur', this._onBlur);
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('pointerlockchange', this._onLockChange);
    this.dom.removeEventListener('pointerdown', this._onPointerDown);
    window.removeEventListener('pointerup', this._onPointerUp);
    this.dom.removeEventListener('wheel', this._onWheel);
  }

  // -- internals -----------------------------------------------------------

  _applyRotation() {
    const limit = Math.PI / 2 - 0.001;
    this.pitch = THREE.MathUtils.clamp(this.pitch, -limit, limit);
    this._euler.set(this.pitch, this.yaw, 0);
    this.camera.quaternion.setFromEuler(this._euler);
  }

  _isTyping(e) {
    const el = e.target;
    return el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA') &&
      el.type !== 'checkbox';
  }

  _onKeyDown(e) {
    if (!this.enabled || this._isTyping(e) || e.metaKey) return;
    this.keys.add(e.code);
    // keep the browser from scrolling or opening menus while flying
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
    if ((e.code === 'AltLeft' || e.code === 'AltRight')) e.preventDefault();
  }

  _onKeyUp(e) {
    this.keys.delete(e.code);
  }

  _onPointerDown(e) {
    if (!this.enabled) return;
    if (e.button === 2) {
      this.dragging = true;
      this.dom.setPointerCapture?.(e.pointerId);
    } else if (e.button === 0 && !this.locked) {
      this.dom.requestPointerLock?.()?.catch?.(() => {});
    }
  }

  _onPointerUp(e) {
    if (e.button === 2) this.dragging = false;
  }

  _onMouseMove(e) {
    if (!this.enabled || !(this.locked || this.dragging)) return;
    this.yaw -= e.movementX * this.lookSensitivity;
    this.pitch -= e.movementY * this.lookSensitivity;
    this._applyRotation();
  }

  _onWheel(e) {
    e.preventDefault();
    const factor = Math.pow(1.15, -Math.sign(e.deltaY));
    this.setSpeed(this.speed * factor);
  }

  _onLockChange() {
    if (!this.locked) this.keys.clear();
    this.dispatchEvent(new Event('lockchange'));
  }
}
