// Viewport shading, as in Blender's viewport: the scene's own materials, a
// matcap (one lit-sphere image for everything, to read form), or wireframe.
//
// The matcaps are painted here on a canvas rather than shipped as images: a
// lit sphere is a few lines of shading maths, and this keeps them crisp at any
// size and free of licensing questions.
import * as THREE from 'three';

const SIZE = 256;

/**
 * Paint a lit sphere. `shade(n)` gets the view-space normal (x right, y up,
 * z towards the viewer) and returns linear [r, g, b] in 0..1.
 */
function paint(shade) {
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(SIZE, SIZE);
  const r = SIZE / 2;
  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      let x = (px + 0.5 - r) / r;
      let y = -(py + 0.5 - r) / r;
      let d = x * x + y * y;
      if (d > 1) { // outside the disc: repeat the rim, which is what edges sample
        const k = 1 / Math.sqrt(d);
        x *= k;
        y *= k;
        d = 1;
      }
      const z = Math.sqrt(Math.max(0, 1 - d));
      const [cr, cg, cb] = shade(x, y, z);
      const i = (py * SIZE + px) * 4;
      img.data[i] = toSrgb8(cr);
      img.data[i + 1] = toSrgb8(cg);
      img.data[i + 2] = toSrgb8(cb);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

function toSrgb8(v) {
  const c = Math.min(Math.max(v, 0), 1);
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
  return Math.round(s * 255);
}

const norm = (v) => {
  const l = Math.hypot(...v);
  return v.map((c) => c / l);
};
const KEY = norm([-0.45, 0.6, 0.66]); // key light: upper left, towards the viewer
const FILL = norm([0.6, -0.15, 0.8]);

function studio({ base, ambient = 0.18, fill = 0.25, spec = 0.0, shine = 40, rim = 0.0, rimColor = [1, 1, 1] }) {
  return (x, y, z) => {
    const n = [x, y, z];
    const dk = Math.max(0, n[0] * KEY[0] + n[1] * KEY[1] + n[2] * KEY[2]);
    const df = Math.max(0, n[0] * FILL[0] + n[1] * FILL[1] + n[2] * FILL[2]);
    // specular: reflect the key about the normal, compare with the view (0, 0, 1)
    const rz = 2 * dk * n[2] - KEY[2];
    const sp = spec * Math.max(0, rz) ** shine;
    const rm = rim * (1 - z) ** 3;
    const light = ambient + dk * (1 - ambient) + df * fill;
    return base.map((b, i) => b * light + sp + rm * rimColor[i]);
  };
}

/** A polished metal: a sky above the horizon line, dark ground below. */
function metal(x, y, z) {
  const ry = 2 * z * y; // reflected direction's y for a view along -z
  const sky = 0.55 + 0.4 * Math.max(0, ry);
  const ground = 0.08 + 0.06 * Math.max(0, 1 + ry);
  const t = 1 / (1 + Math.exp(-ry * 14)); // sharp horizon
  const v = ground * (1 - t) + sky * t;
  const dk = Math.max(0, x * KEY[0] + y * KEY[1] + z * KEY[2]);
  const hl = 1.6 * Math.max(0, 2 * dk * z - KEY[2]) ** 60;
  return [v * 0.95 + hl, v * 0.97 + hl, v + hl];
}

/** View-space normals as colours: shows facing and smoothing at a glance. */
function normals(x, y, z) {
  const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return [lin(x * 0.5 + 0.5), lin(y * 0.5 + 0.5), lin(z * 0.5 + 0.5)];
}

export const MATCAPS = [
  { id: 'clay', label: 'Clay', shade: studio({ base: [0.6, 0.55, 0.5], spec: 0.05, shine: 12, rim: 0.12 }) },
  { id: 'studio', label: 'Studio', shade: studio({ base: [0.72, 0.72, 0.74], spec: 0.35, shine: 30, rim: 0.2, fill: 0.35 }) },
  { id: 'ceramic', label: 'Ceramic', shade: studio({ base: [0.85, 0.84, 0.8], ambient: 0.28, spec: 0.9, shine: 90, rim: 0.08 }) },
  { id: 'metal', label: 'Metal', shade: metal },
  { id: 'red', label: 'Red wax', shade: studio({ base: [0.55, 0.06, 0.04], ambient: 0.12, spec: 0.5, shine: 50, rim: 0.25, rimColor: [1, 0.5, 0.4] }) },
  { id: 'normals', label: 'Normals', shade: normals },
];

const cache = new Map();

/** The matcap texture and a small preview image URL for one look. */
export function matcap(id) {
  const entry = MATCAPS.find((m) => m.id === id) ?? MATCAPS[0];
  if (!cache.has(entry.id)) {
    const canvas = paint(entry.shade);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    cache.set(entry.id, { texture, url: canvas.toDataURL('image/png') });
  }
  return cache.get(entry.id);
}

export const SHADING_MODES = ['material', 'matcap', 'wireframe'];

/** Shared materials, one per mode, look and face side. */
export class ShadingMaterials {
  constructor() {
    this.materials = new Map();
  }

  get(mode, look, side) {
    const key = `${mode}:${mode === 'matcap' ? look : ''}:${side}`;
    let m = this.materials.get(key);
    if (!m) {
      m = mode === 'matcap'
        ? new THREE.MeshMatcapMaterial({ matcap: matcap(look).texture, side })
        : new THREE.MeshBasicMaterial({
          color: 0xc6d0de, wireframe: true, transparent: true, opacity: 0.32, depthWrite: false, side: THREE.DoubleSide,
        });
      m.toneMapped = false; // like Blender: exposure does not change the matcap or the lines
      m.userData.usdvShading = true;
      this.materials.set(key, m);
    }
    return m;
  }
}
