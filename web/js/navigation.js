import * as THREE from 'three';

/**
 * Viewport navigation in the style of DCC / game-engine editors.
 *
 *   Left-drag              orbit around the point under the cursor
 *   Shift+left / middle    pan
 *   Right-drag             look around; W A S D / Q E fly while held (and any time)
 *   Wheel                  zoom towards the point under the cursor
 *   Right-drag + wheel     change fly speed
 *   Double-click           focus on the point under the cursor
 *   Touch                  one finger orbits, two fingers pan and pinch-zoom
 *
 * The camera never rolls. Events: 'dragstart' / 'dragend' ({ detail: { mode } }),
 * 'speed' when the fly speed changes, 'interact' on any user input.
 */
const MAX_PITCH = THREE.MathUtils.degToRad(89);
const FLY_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyC', 'Space',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
]);

export class Navigation extends EventTarget {
  /**
   * @param camera  THREE.PerspectiveCamera (no parent)
   * @param dom     element receiving pointer events (the canvas)
   * @param pick    (x, y) => THREE.Vector3 | null, CSS pixels relative to dom
   */
  constructor(camera, dom, pick) {
    super();
    this.camera = camera;
    this.dom = dom;
    this.pick = pick;
    this.enabled = true;

    this.yaw = 0;
    this.pitch = 0;
    this.pivot = new THREE.Vector3();
    this.sceneRadius = 10;
    this.speed = 2; // fly speed, metres per second
    this.orbitSensitivity = 0.0055; // radians per pixel
    this.lookSensitivity = 0.0028;
    this.damping = 10;

    this.velocity = new THREE.Vector3();
    this.keys = new Set();
    this.drag = null;
    this.touches = new Map();
    this.anim = null;
    this._wheelPick = { time: 0, x: -1, y: -1, point: null };

    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._v = new THREE.Vector3();
    this._q = new THREE.Quaternion();

    this._bind('pointerdown', dom, this._onPointerDown);
    this._bind('pointermove', dom, this._onPointerMove);
    this._bind('pointerup', dom, this._onPointerUp);
    this._bind('pointercancel', dom, this._onPointerUp);
    this._bind('wheel', dom, this._onWheel, { passive: false });
    this._bind('dblclick', dom, this._onDoubleClick);
    this._bind('contextmenu', dom, (e) => e.preventDefault());
    this._bind('keydown', window, this._onKeyDown);
    this._bind('keyup', window, (e) => this.keys.delete(e.code));
    this._bind('blur', window, () => this.keys.clear());
    dom.style.touchAction = 'none';
  }

  _bind(type, target, fn, opts) {
    target.addEventListener(type, fn.bind(this), opts);
  }

  // -- public ---------------------------------------------------------------

  /** Scale speeds to a scene of this bounding radius (metres). */
  setSceneRadius(radius) {
    this.sceneRadius = Math.max(radius, 1e-3);
    this.setSpeed(Math.max(radius / 6, 0.05));
  }

  setSpeed(speed) {
    this.speed = THREE.MathUtils.clamp(speed, 1e-3, 1e6);
    this.dispatchEvent(new Event('speed'));
  }

  /** Point the camera at a world position (no animation). */
  lookAt(target) {
    const d = this._v.subVectors(target, this.camera.position);
    if (d.lengthSq() < 1e-12) return;
    d.normalize();
    this.yaw = Math.atan2(-d.x, -d.z);
    this.pitch = THREE.MathUtils.clamp(Math.asin(d.y), -MAX_PITCH, MAX_PITCH);
    this._applyRotation();
  }

  /** Animate the camera to `position`, looking at `target`. */
  flyTo(position, target, duration = 0.45) {
    const from = { pos: this.camera.position.clone(), yaw: this.yaw, pitch: this.pitch };
    const d = new THREE.Vector3().subVectors(target, position).normalize();
    let yaw = Math.atan2(-d.x, -d.z);
    // turn the short way round
    while (yaw - from.yaw > Math.PI) yaw -= 2 * Math.PI;
    while (yaw - from.yaw < -Math.PI) yaw += 2 * Math.PI;
    const to = { pos: position.clone(), yaw, pitch: THREE.MathUtils.clamp(Math.asin(d.y), -MAX_PITCH, MAX_PITCH) };
    this.pivot.copy(target);
    this.velocity.set(0, 0, 0);
    if (duration <= 0) {
      this.camera.position.copy(to.pos);
      this.yaw = to.yaw;
      this.pitch = to.pitch;
      this._applyRotation();
      return;
    }
    this.anim = { from, to, t: 0, duration };
  }

  stop() {
    this.velocity.set(0, 0, 0);
    this.anim = null;
  }

  get isFlying() {
    return this.drag?.mode === 'look' || [...this.keys].some((k) => FLY_KEYS.has(k));
  }

  update(dt) {
    if (this.anim) {
      const a = this.anim;
      a.t = Math.min(a.t + dt / a.duration, 1);
      const s = a.t * a.t * (3 - 2 * a.t);
      this.camera.position.lerpVectors(a.from.pos, a.to.pos, s);
      this.yaw = a.from.yaw + (a.to.yaw - a.from.yaw) * s;
      this.pitch = a.from.pitch + (a.to.pitch - a.from.pitch) * s;
      this._applyRotation();
      if (a.t >= 1) this.anim = null;
      return;
    }

    const k = this.keys;
    const input = this._v.set(
      (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) - (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0),
      (k.has('KeyE') || k.has('Space') ? 1 : 0) - (k.has('KeyQ') || k.has('KeyC') ? 1 : 0),
      (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0) - (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0),
    );
    let speed = this.speed;
    if (k.has('ShiftLeft') || k.has('ShiftRight')) speed *= 4;
    if (k.has('ControlLeft') || k.has('ControlRight') || k.has('AltLeft') || k.has('AltRight')) speed *= 0.25;

    const target = new THREE.Vector3();
    if (input.lengthSq() > 0) {
      input.normalize();
      // horizontal moves follow the view; vertical moves are world up/down
      const vertical = input.y;
      input.y = 0;
      target.copy(input).applyQuaternion(this.camera.quaternion).multiplyScalar(speed);
      target.y += vertical * speed;
    }
    const blend = 1 - Math.exp(-this.damping * dt);
    this.velocity.lerp(target, blend);
    if (this.velocity.lengthSq() < 1e-10 && input.lengthSq() === 0) this.velocity.set(0, 0, 0);
    else this.camera.position.addScaledVector(this.velocity, dt);
  }

  // -- internals ------------------------------------------------------------

  _applyRotation() {
    this._euler.set(this.pitch, this.yaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(this._euler);
    this.camera.updateMatrixWorld();
  }

  _local(e) {
    const r = this.dom.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _interact() {
    this.anim = null;
    this.dispatchEvent(new Event('interact'));
  }

  /** Depth along the view direction of a world point. */
  _viewDepth(p) {
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    return fwd.dot(this._v.subVectors(p, this.camera.position));
  }

  /** A sensible pivot when nothing is under the cursor: keep the old one if it's in front. */
  _fallbackPivot() {
    const d = this._viewDepth(this.pivot);
    if (d > this.sceneRadius * 1e-3) return this.pivot.clone();
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    return this.camera.position.clone().addScaledVector(fwd, this.sceneRadius * 0.5);
  }

  _orbit(dx, dy) {
    const offset = new THREE.Vector3().subVectors(this.camera.position, this.pivot);
    const before = this.camera.quaternion.clone().invert();
    this.yaw -= dx * this.orbitSensitivity;
    this.pitch = THREE.MathUtils.clamp(this.pitch - dy * this.orbitSensitivity, -MAX_PITCH, MAX_PITCH);
    this._applyRotation();
    offset.applyQuaternion(before).applyQuaternion(this.camera.quaternion);
    this.camera.position.copy(this.pivot).add(offset);
    this.camera.updateMatrixWorld();
  }

  _look(dx, dy) {
    this.yaw -= dx * this.lookSensitivity;
    this.pitch = THREE.MathUtils.clamp(this.pitch - dy * this.lookSensitivity, -MAX_PITCH, MAX_PITCH);
    this._applyRotation();
  }

  _pan(dx, dy, depth) {
    const h = this.dom.clientHeight || 1;
    const perPixel = (2 * depth * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)) / h;
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    this.camera.position.addScaledVector(right, -dx * perPixel).addScaledVector(up, dy * perPixel);
    this.pivot.addScaledVector(right, -dx * perPixel).addScaledVector(up, dy * perPixel);
    this.camera.updateMatrixWorld();
  }

  /** Move towards `point` by a fraction of the distance (negative = away). */
  _dollyTowards(point, fraction) {
    const d = new THREE.Vector3().subVectors(point, this.camera.position);
    const minGap = this.sceneRadius * 1e-4;
    if (fraction > 0 && d.length() * (1 - fraction) < minGap) fraction = Math.max(0, 1 - minGap / d.length());
    this.camera.position.addScaledVector(d, fraction);
    this.camera.updateMatrixWorld();
  }

  _dragStart(e, mode) {
    const [x, y] = this._local(e);
    let depth = null;
    if (mode === 'orbit' || mode === 'pan') {
      const hit = this.pick(x, y);
      if (mode === 'orbit') this.pivot.copy(hit || this._fallbackPivot());
      depth = Math.max(this._viewDepth(hit || this._fallbackPivot()), this.sceneRadius * 1e-3);
    }
    this.drag = { mode, pointerId: e.pointerId, x: e.clientX, y: e.clientY, depth, moved: false };
    this.dom.setPointerCapture?.(e.pointerId);
    this.dispatchEvent(new CustomEvent('dragstart', { detail: { mode } }));
  }

  _dragEnd() {
    if (!this.drag) return;
    const { mode } = this.drag;
    this.dom.releasePointerCapture?.(this.drag.pointerId);
    this.drag = null;
    this.dispatchEvent(new CustomEvent('dragend', { detail: { mode } }));
  }

  _onPointerDown(e) {
    if (!this.enabled) return;
    this._interact();
    this.dom.focus?.({ preventScroll: true });
    if (e.pointerType === 'touch') {
      this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.touches.size === 1) this._dragStart(e, 'orbit');
      else if (this.touches.size === 2) {
        this._dragEnd();
        this._pinch = this._pinchState();
      }
      return;
    }
    if (this.drag) return;
    if (e.button === 0) this._dragStart(e, e.shiftKey ? 'pan' : 'orbit');
    else if (e.button === 1) {
      e.preventDefault();
      this._dragStart(e, 'pan');
    } else if (e.button === 2) this._dragStart(e, 'look');
  }

  _pinchState() {
    const [a, b] = [...this.touches.values()];
    return { mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, dist: Math.hypot(a.x - b.x, a.y - b.y) };
  }

  _onPointerMove(e) {
    if (e.pointerType === 'touch' && this.touches.has(e.pointerId)) {
      this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.touches.size === 2 && this._pinch) {
        const now = this._pinchState();
        const depth = Math.max(this._viewDepth(this.pivot), this.sceneRadius * 1e-3);
        this._pan(now.mid.x - this._pinch.mid.x, now.mid.y - this._pinch.mid.y, depth);
        if (this._pinch.dist > 0) this._dollyTowards(this.pivot, 1 - this._pinch.dist / now.dist);
        this._pinch = now;
        return;
      }
    }
    const d = this.drag;
    if (!d || e.pointerId !== d.pointerId) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    d.x = e.clientX;
    d.y = e.clientY;
    if (dx || dy) d.moved = true;
    if (d.mode === 'orbit') this._orbit(dx, dy);
    else if (d.mode === 'pan') this._pan(dx, dy, d.depth);
    else if (d.mode === 'look') this._look(dx, dy);
  }

  _onPointerUp(e) {
    if (e.pointerType === 'touch') {
      this.touches.delete(e.pointerId);
      if (this.touches.size < 2) this._pinch = null;
    }
    if (this.drag && e.pointerId === this.drag.pointerId) this._dragEnd();
  }

  _onWheel(e) {
    if (!this.enabled) return;
    e.preventDefault();
    this._interact();
    let delta = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1);
    if (this.drag?.mode === 'look') {
      this.setSpeed(this.speed * Math.pow(1.2, -Math.sign(delta)));
      return;
    }
    const [x, y] = this._local(e);
    // re-pick only when the cursor moved or a while has passed (picking costs a draw)
    const w = this._wheelPick;
    const now = performance.now();
    if (now - w.time > 300 || Math.hypot(x - w.x, y - w.y) > 4) {
      w.point = this.pick(x, y);
      w.time = now;
      w.x = x;
      w.y = y;
    }
    let point = w.point;
    if (!point) {
      // nothing under the cursor: zoom along the cursor ray as far as the pivot
      const ndc = new THREE.Vector3((x / this.dom.clientWidth) * 2 - 1, -(y / this.dom.clientHeight) * 2 + 1, 0.5);
      const dir = ndc.unproject(this.camera).sub(this.camera.position).normalize();
      const dist = Math.max(this.camera.position.distanceTo(this.pivot), this.sceneRadius * 0.05);
      point = this.camera.position.clone().addScaledVector(dir, dist);
    } else {
      this.pivot.copy(point);
    }
    delta = THREE.MathUtils.clamp(delta, -300, 300);
    const steps = -delta / 100;
    this._dollyTowards(point, 1 - Math.pow(0.8, steps));
  }

  _onDoubleClick(e) {
    if (!this.enabled) return;
    const [x, y] = this._local(e);
    const hit = this.pick(x, y);
    if (!hit) return;
    const dist = this.camera.position.distanceTo(hit);
    const keep = Math.max(dist * 0.35, this.sceneRadius * 0.01);
    const dir = new THREE.Vector3().subVectors(this.camera.position, hit).normalize();
    this.flyTo(hit.clone().addScaledVector(dir, keep), hit);
  }

  _onKeyDown(e) {
    if (!this.enabled || e.metaKey || isTyping(e)) return;
    this.keys.add(e.code);
    if (FLY_KEYS.has(e.code)) {
      this._interact();
      // keep the page from scrolling while flying
      if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
    }
  }
}

function isTyping(e) {
  const t = e.target;
  if (!t || !t.tagName) return false;
  if (t.isContentEditable) return true;
  if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return true;
  return t.tagName === 'INPUT' && !['checkbox', 'radio', 'range', 'button'].includes(t.type);
}
