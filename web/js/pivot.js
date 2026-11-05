// The selected prim's pivot: three lines from its pivot point along its own USD
// axes (X red, Y green, Z blue). Plain lines, no handles: it shows the frame, it
// is not something to drag.
//
// Each axis is AXIS_LENGTH long at scale 1 and follows the prim's scale on that
// axis, so the lines also show how the prim is scaled. The lines are a constant
// width on screen and drawn on top of everything, on the helper layer so they
// are never picked and cast no shadow.
import * as THREE from 'three';
import { HELPER_LAYER } from './picking.js';

/** An axis at scale 1, in metres. */
const AXIS_LENGTH = 0.2;
/** Line width and its dark outline, in CSS pixels. */
const WIDTH = 2;
const OUTLINE = 1;
const FADE_IN = 180; // ms
const COLORS = [0xff3b55, 0x7ed321, 0x2f8cff].map((c) => new THREE.Color(c));

const vertexShader = /* glsl */`
attribute vec3 aOther;  // the other end of this vertex's axis
attribute float aSide;  // which edge of the strip, -1 or 1
attribute float aWiden; // extra width (the outline), in CSS pixels
attribute vec4 aColor;
uniform vec2 uViewport; // drawing buffer, in device pixels
uniform float uPixelRatio;
uniform float uWidth;
varying vec4 vColor;
void main() {
  vColor = aColor;
  vec4 a = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  vec4 b = projectionMatrix * modelViewMatrix * vec4( aOther, 1.0 );
  // keep both ends in front of the camera
  const float EPS = 1e-5;
  if ( a.w < EPS && b.w < EPS ) {
    gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 ); // outside the clip volume
    return;
  }
  if ( a.w < EPS ) a = mix( a, b, ( EPS - a.w ) / ( b.w - a.w ) );
  if ( b.w < EPS ) b = mix( b, a, ( EPS - b.w ) / ( a.w - b.w ) );
  vec2 halfView = uViewport * 0.5;
  vec2 d = ( b.xy / b.w - a.xy / a.w ) * halfView;
  vec2 dir = length( d ) > 1e-4 ? normalize( d ) : vec2( 1.0, 0.0 );
  float width = ( uWidth + aWiden ) * uPixelRatio;
  a.xy += vec2( -dir.y, dir.x ) * aSide * width * 0.5 / halfView * a.w;
  gl_Position = a;
}
`;

const fragmentShader = /* glsl */`
uniform float uOpacity;
varying vec4 vColor;
void main() {
  gl_FragColor = vec4( vColor.rgb, vColor.a * uOpacity );
}
`;

export class PivotGizmo {
  constructor() {
    // per axis two strips (dark outline first, then the colour), 4 vertices each
    const strips = 6;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(strips * 4 * 3), 3));
    geometry.setAttribute('aOther', new THREE.BufferAttribute(new Float32Array(strips * 4 * 3), 3));
    const side = [];
    const widen = [];
    const color = [];
    const index = [];
    const order = [0, 1, 2, 3, 4, 5].map((k) => (k < 3 ? { axis: k, outline: true } : { axis: k - 3, outline: false }));
    order.forEach(({ axis, outline }, k) => {
      // vertices: start on both edges, then end on both edges; at the end the
      // direction runs the other way, so its sides are swapped
      side.push(-1, 1, 1, -1);
      for (let v = 0; v < 4; v++) {
        widen.push(outline ? 2 * OUTLINE : 0);
        if (outline) color.push(0, 0, 0, 0.4);
        else color.push(COLORS[axis].r, COLORS[axis].g, COLORS[axis].b, 1);
      }
      const o = k * 4;
      index.push(o, o + 1, o + 3, o, o + 3, o + 2);
    });
    geometry.setAttribute('aSide', new THREE.BufferAttribute(new Float32Array(side), 1));
    geometry.setAttribute('aWiden', new THREE.BufferAttribute(new Float32Array(widen), 1));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(color), 4));
    geometry.setIndex(index);
    this._order = order;

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uViewport: { value: new THREE.Vector2(1, 1) },
        uPixelRatio: { value: 1 },
        uWidth: { value: WIDTH },
        uOpacity: { value: 1 },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'usdv pivot';
    mesh.layers.set(HELPER_LAYER);
    mesh.renderOrder = 20;
    mesh.frustumCulled = false;
    mesh.visible = false;
    const size = new THREE.Vector2();
    mesh.onBeforeRender = (renderer) => {
      material.uniforms.uViewport.value.copy(renderer.getDrawingBufferSize(size));
      material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
      material.uniformsNeedUpdate = true;
    };
    this.object = mesh;
    this._shownAt = 0;
  }

  /**
   * Place the axes. `usdFrame` is the prim's local-to-world matrix in USD
   * stage space, `stageToWorld` takes stage space to the view (up axis, metres),
   * `point` is the pivot in the prim's frame.
   */
  show(usdFrame, stageToWorld, point) {
    const frame = stageToWorld.clone().multiply(usdFrame);
    const origin = point.clone().applyMatrix4(frame);
    const usdAxes = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    const axes = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    usdFrame.extractBasis(...usdAxes);
    frame.extractBasis(...axes);
    const ends = axes.map((dir, i) => {
      const scale = usdAxes[i].length(); // the prim's own scale on this axis, without units
      return origin.clone().addScaledVector(dir.normalize(), AXIS_LENGTH * scale);
    });

    const g = this.object.geometry;
    const pos = g.attributes.position;
    const other = g.attributes.aOther;
    this._order.forEach(({ axis }, k) => {
      for (let v = 0; v < 4; v++) {
        const atStart = v < 2;
        pos.setXYZ(k * 4 + v, ...(atStart ? origin : ends[axis]).toArray());
        other.setXYZ(k * 4 + v, ...(atStart ? ends[axis] : origin).toArray());
      }
    });
    pos.needsUpdate = true;
    other.needsUpdate = true;
    this.object.material.uniforms.uOpacity.value = 0;
    this._shownAt = performance.now();
    this.object.visible = true;
  }

  hide() {
    this.object.visible = false;
  }

  /** Step the fade-in to time `now`. Returns whether it is still running. */
  animate(now) {
    if (!this.object.visible) return false;
    const t = Math.min(Math.max((now - this._shownAt) / FADE_IN, 0), 1);
    this.object.material.uniforms.uOpacity.value = t * t * (3 - 2 * t);
    return t < 1;
  }
}
