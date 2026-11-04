// Shading networks compiled by the converter (server/mtlx_glsl.py) from MaterialX
// and UsdPreviewSurface materials.
//
// Each glTF material may carry extras.usdvShader: GLSL statements that evaluate
// the material's whole pattern network (textures, maths, noises, triplanar
// projection, ...) from MaterialX's own node implementations, and assign the
// results to three.js MeshPhysicalMaterial parameters (usdv_p_<name>). Here a
// MeshPhysicalMaterial is patched to run that code, so three.js still provides
// the lighting, shadows and environment.
//
// The network sees geometry the way it does in Houdini or usdview: object space
// is the prim's own local space (the mesh's POSITION / NORMAL attributes) and
// world space is the USD stage's world, in stage units and up axis. The glTF
// scene has a root node that converts the stage to Y-up metres; its inverse
// (usdvRootInverse) takes three.js world space back to USD world space.
import * as THREE from 'three';

/** Shared by every compiled material: three.js world -> USD world. */
const rootInverse = { value: new THREE.Matrix4() };
const root = { value: new THREE.Matrix4() };

// three.js uniforms the generated code can shadow with a local of the same name
const SHADOWED = [
  'diffuse', 'opacity', 'roughness', 'metalness', 'emissive', 'ior', 'specularIntensity', 'specularColor',
  'clearcoat', 'clearcoatRoughness', 'sheenColor', 'sheenRoughness', 'transmission', 'thickness',
  'attenuationColor', 'attenuationDistance', 'iridescence', 'iridescenceIOR', 'iridescenceThicknessMaximum',
];

function vertexCode(cfg) {
  const needs = new Set(cfg.needs);
  const head = [];
  const body = [];
  const has = (n) => needs.has(n);
  const usesT = has('Tobject') || has('Tworld') || has('Bobject') || has('Bworld');
  if (usesT) head.push('#ifndef USE_TANGENT', 'attribute vec4 tangent;', '#endif');
  body.push('mat4 usdvO2W = usdvRootInverse * modelMatrix;');
  body.push('#ifdef USE_INSTANCING', 'usdvO2W = usdvO2W * instanceMatrix;', '#endif');
  if (has('Pobject')) { head.push('varying vec3 usdv_vPobject;'); body.push('usdv_vPobject = transformed;'); }
  if (has('Pworld') || has('Vworld')) {
    head.push('varying vec3 usdv_vPworld;');
    body.push('usdv_vPworld = ( usdvO2W * vec4( transformed, 1.0 ) ).xyz;');
  }
  if (has('Nobject') || has('Bobject')) { head.push('varying vec3 usdv_vNobject;'); body.push('usdv_vNobject = objectNormal;'); }
  if (has('Nworld') || has('Bworld')) {
    head.push('varying vec3 usdv_vNworld;');
    body.push('usdv_vNworld = transpose( inverse( mat3( usdvO2W ) ) ) * objectNormal;');
  }
  if (usesT) {
    head.push('varying vec4 usdv_vTobject;', 'varying vec3 usdv_vTworld;');
    body.push('usdv_vTobject = tangent;', 'usdv_vTworld = mat3( usdvO2W ) * tangent.xyz;');
  }
  if (has('UV0')) { head.push('varying vec2 usdv_vUV0;'); body.push('usdv_vUV0 = vec2( uv.x, 1.0 - uv.y );'); }
  if (has('M')) { head.push('varying mat4 usdv_vO2W;'); body.push('usdv_vO2W = usdvO2W;'); }
  for (const attr of Object.keys(cfg.attributes)) {
    head.push(`attribute vec4 ${attr};`, `varying vec4 ${attr}_v;`);
    body.push(`${attr}_v = ${attr};`);
  }
  return { head: ['uniform mat4 usdvRootInverse;', ...head].join('\n'), body: `{\n${body.join('\n')}\n}` };
}

function fragmentCode(cfg, lib) {
  const needs = new Set(cfg.needs);
  const has = (n) => needs.has(n);
  const usesT = has('Tobject') || has('Tworld') || has('Bobject') || has('Bworld');
  const head = ['uniform mat4 usdvRootInverse;', 'uniform mat4 usdvRoot;'];
  if (has('Pobject')) head.push('varying vec3 usdv_vPobject;');
  if (has('Pworld') || has('Vworld')) head.push('varying vec3 usdv_vPworld;');
  if (has('Nobject') || has('Bobject')) head.push('varying vec3 usdv_vNobject;');
  if (has('Nworld') || has('Bworld')) head.push('varying vec3 usdv_vNworld;');
  if (usesT) head.push('varying vec4 usdv_vTobject;', 'varying vec3 usdv_vTworld;');
  if (has('UV0')) head.push('varying vec2 usdv_vUV0;');
  if (has('M')) head.push('varying mat4 usdv_vO2W;');
  for (const attr of Object.keys(cfg.attributes)) head.push(`varying vec4 ${attr}_v;`);
  if (cfg.uniforms.length) head.push(`uniform vec4 usdvU[${cfg.uniforms.length}];`);
  for (const t of cfg.textures) head.push(`uniform sampler2D ${t.uniform};`);
  head.push('vec3 usdvWorldToView( vec3 v ) { return normalize( mat3( viewMatrix ) * normalize( mat3( usdvRoot ) * v ) ); }');
  for (const k of cfg.chunks) head.push(`// ---- ${k}\n${lib[k] ?? ''}`);

  const pre = [];
  if (has('Pobject')) pre.push('vec3 usdv_Pobject = usdv_vPobject;');
  if (has('Pworld') || has('Vworld')) pre.push('vec3 usdv_Pworld = usdv_vPworld;');
  if (has('Nobject') || has('Bobject')) pre.push('vec3 usdv_Nobject = normalize( usdv_vNobject );');
  if (has('Nworld') || has('Bworld')) pre.push('vec3 usdv_Nworld = normalize( usdv_vNworld );');
  if (usesT) {
    pre.push('vec3 usdv_Tobject = normalize( usdv_vTobject.xyz );', 'vec3 usdv_Tworld = normalize( usdv_vTworld );');
    pre.push('float usdv_Tsign = usdv_vTobject.w < 0.0 ? -1.0 : 1.0;');
    if (has('Bobject')) pre.push('vec3 usdv_Bobject = normalize( cross( usdv_Nobject, usdv_Tobject ) ) * usdv_Tsign;');
    if (has('Bworld')) pre.push('vec3 usdv_Bworld = normalize( cross( usdv_Nworld, usdv_Tworld ) ) * usdv_Tsign;');
  }
  if (has('UV0')) pre.push('vec2 usdv_UV0 = usdv_vUV0;');
  if (has('Vworld')) {
    pre.push('vec3 usdv_Vworld = normalize( usdv_Pworld - ( usdvRootInverse * vec4( cameraPosition, 1.0 ) ).xyz );');
  }
  if (has('M')) pre.push('mat4 usdv_ObjToWorld = usdv_vO2W;', 'mat4 usdv_WorldToObj = inverse( usdv_vO2W );');
  pre.push(cfg.main);
  for (const name of SHADOWED) {
    if (name in cfg.params) pre.push(`${cfg.params[name]} ${name} = usdv_p_${name};`);
  }
  if ('alphaCutoff' in cfg.params) pre.push('float alphaTest = usdv_p_alphaCutoff;');
  // three.js computes an anisotropy frame from UVs it does not have here
  pre.push('#if defined( USE_ANISOTROPY ) && ! defined( USE_UV )', 'vec2 vUv = vec2( 0.0 );', '#endif');

  const normal = 'normal' in cfg.params ? `
normal = usdvWorldToView( usdv_p_normal );
#ifdef DOUBLE_SIDED
normal *= faceDirection;
#endif` : '';
  const coatNormal = 'clearcoatNormal' in cfg.params ? `
#ifdef USE_CLEARCOAT
clearcoatNormal = usdvWorldToView( usdv_p_clearcoatNormal );
#ifdef DOUBLE_SIDED
clearcoatNormal *= faceDirection;
#endif
#endif` : '';
  const anisotropy = 'anisotropyStrength' in cfg.params ? `
#ifdef USE_ANISOTROPY
{
  material.anisotropy = saturate( usdv_p_anisotropyStrength );
  material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
  vec3 aT = usdvWorldToView( usdv_Tworld );
  aT = normalize( aT - normal * dot( aT, normal ) );
  vec3 aB = cross( normal, aT );
  vec2 dir = vec2( cos( usdv_p_anisotropyRotation ), sin( usdv_p_anisotropyRotation ) );
  material.anisotropyT = aT * dir.x + aB * dir.y;
  material.anisotropyB = aB * dir.x - aT * dir.y;
}
#endif` : '';
  const occlusion = 'occlusion' in cfg.params ? `
{
  float usdvAO = clamp( usdv_p_occlusion, 0.0, 1.0 );
  reflectedLight.indirectDiffuse *= usdvAO;
  #if defined( USE_CLEARCOAT )
    clearcoatSpecularIndirect *= usdvAO;
  #endif
  #if defined( USE_SHEEN )
    sheenSpecularIndirect *= usdvAO;
  #endif
  #if defined( USE_ENVMAP ) && defined( STANDARD )
    reflectedLight.indirectSpecular *= computeSpecularOcclusion( saturate( dot( geometryNormal, geometryViewDir ) ), usdvAO, material.roughness );
  #endif
}` : '';
  return { head: head.join('\n'), pre: pre.join('\n'), normal, coatNormal, anisotropy, occlusion };
}

function patch(source, search, replacement, label, errors) {
  if (!source.includes(search)) {
    errors.push(`could not patch ${label}`);
    return source;
  }
  return source.replace(search, replacement);
}

/**
 * Build the viewer material for one glTF material that carries a shader graph.
 * `fallback` is the plain glTF material three.js created from the same entry.
 */
function buildMaterial(cfg, fallback, textures, lib) {
  const f = cfg.features || {};
  const m = new THREE.MeshPhysicalMaterial({
    name: fallback.name,
    side: fallback.side,
    // the shader computes these; values > 0 only switch the features on
    clearcoat: f.clearcoat ? 1 : 0,
    sheen: f.sheen ? 1 : 0,
    transmission: f.transmission ? 1 : 0,
    iridescence: f.iridescence ? 1 : 0,
    anisotropy: f.anisotropy ? 1 : 0,
    alphaTest: f.alphaTest ? 0.5 : 0,
  });
  if (f.transparent) {
    m.transparent = true;
    m.depthWrite = false;
  }
  m.userData = { ...fallback.userData, usdvFallback: fallback, usdvGraph: true };
  const uniforms = {
    usdvRootInverse: rootInverse,
    usdvRoot: root,
  };
  if (cfg.uniforms.length) {
    uniforms.usdvU = { value: cfg.uniforms.map((u) => new THREE.Vector4(...u)) };
  }
  for (const t of cfg.textures) uniforms[t.uniform] = { value: textures.get(t.index) ?? null };

  const vs = vertexCode(cfg);
  const fs = fragmentCode(cfg, lib);
  const key = `usdv:${cfg.key}:${cfg.uniforms.length}`;
  m.onBeforeCompile = (shader) => {
    const errors = [];
    Object.assign(shader.uniforms, uniforms);
    let v = vs.head + '\n' + shader.vertexShader;
    v = patch(v, '#include <project_vertex>', `#include <project_vertex>\n${vs.body}`, 'vertex', errors);
    shader.vertexShader = v;
    let s = fs.head + '\n' + shader.fragmentShader;
    s = patch(s, 'void main() {', `void main() {\n${fs.pre}\n`, 'main', errors);
    if (fs.normal) s = patch(s, '#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${fs.normal}`, 'normal', errors);
    if (fs.coatNormal) {
      s = patch(s, '#include <clearcoat_normal_fragment_maps>',
        `#include <clearcoat_normal_fragment_maps>\n${fs.coatNormal}`, 'coat normal', errors);
    }
    if (fs.anisotropy) {
      s = patch(s, '#include <lights_physical_fragment>', `#include <lights_physical_fragment>\n${fs.anisotropy}`,
        'anisotropy', errors);
    }
    if (fs.occlusion) s = patch(s, '#include <aomap_fragment>', `#include <aomap_fragment>\n${fs.occlusion}`, 'occlusion', errors);
    shader.fragmentShader = s;
    if (errors.length) console.warn(`usd-viewer: material ${m.name}: ${errors.join(', ')}`);
  };
  m.customProgramCacheKey = () => key;
  return m;
}

/**
 * Replace the plain glTF materials of a loaded scene with their compiled shader
 * graphs. Returns the number of materials replaced.
 */
export async function applyShaderGraphs(gltf) {
  const lib = gltf.parser.json.extras?.usdvGlsl ?? {};
  const meshes = [];
  for (const s of gltf.scenes) {
    s.traverse((o) => {
      if (!o.isMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      if (mats.some((m) => m?.userData?.usdvShader)) meshes.push(o);
    });
    // three.js world -> USD world: the inverse of the converter's stage root node
    s.updateMatrixWorld(true);
    const stageRoot = s.getObjectByName('usd_stage_root');
    if (stageRoot) {
      root.value.copy(stageRoot.matrixWorld);
      rootInverse.value.copy(stageRoot.matrixWorld).invert();
    }
  }
  if (!meshes.length) return 0;

  const wanted = new Set();
  for (const o of meshes) {
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      for (const t of m?.userData?.usdvShader?.textures ?? []) wanted.add(t.index);
    }
  }
  const textures = new Map();
  await Promise.all([...wanted].map(async (index) => {
    try {
      const tex = await gltf.parser.getDependency('texture', index);
      tex.colorSpace = THREE.NoColorSpace; // decoded in the shader, as MaterialX does
      textures.set(index, tex);
    } catch (e) {
      console.warn('usd-viewer: texture', index, e);
    }
  }));

  const built = new Map();
  const swap = (m) => {
    const cfg = m?.userData?.usdvShader;
    if (!cfg) return m;
    if (!built.has(m)) built.set(m, buildMaterial(cfg, m, textures, lib));
    return built.get(m);
  };
  for (const o of meshes) {
    o.material = Array.isArray(o.material) ? o.material.map(swap) : swap(o.material);
  }
  return built.size;
}

/**
 * After the scene's shaders were compiled: put the plain glTF material back on
 * any mesh whose generated shader the browser rejected. Returns the names of the
 * materials that fell back.
 */
export function shaderFailed(renderer, material) {
  if (!material?.userData?.usdvGraph) return false;
  const program = renderer.properties.get(material)?.currentProgram;
  if (!program) return false;
  program.getUniforms(); // runs three.js's compile check once
  const d = program.diagnostics;
  if (!d || d.runnable !== false) return false;
  if (!material.userData.usdvReported) {
    material.userData.usdvReported = true;
    console.warn(`usd-viewer: the shader of ${material.name} did not compile; using its plain material\n`
      + `${d.fragmentShader?.log || ''}${d.vertexShader?.log || ''}${d.programLog || ''}`);
  }
  return true;
}

/** `material` itself, or its plain glTF material when its shader failed to compile. */
export function workingMaterial(renderer, material) {
  if (Array.isArray(material)) return material.map((m) => workingMaterial(renderer, m));
  return shaderFailed(renderer, material) ? material.userData.usdvFallback : material;
}
