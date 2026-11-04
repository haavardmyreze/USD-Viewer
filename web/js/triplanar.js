// Triplanar texturing for materials whose USD shading network projects textures
// by position (e.g. MaterialX / Karma triplanar nodes) instead of using UVs.
//
// The converter writes the textures as ordinary glTF maps and puts the projection
// settings in material.extras.usdvTriplanar = { scale, space, flipNormalY }.
// Here the standard three.js shader is patched so those maps are sampled three
// times (along X, Y and Z) and blended by the surface normal.
import * as THREE from 'three';

const SAMPLED_MAPS = [
  ['map', 'vMapUv'],
  ['roughnessMap', 'vRoughnessMapUv'],
  ['metalnessMap', 'vMetalnessMapUv'],
  ['aoMap', 'vAoMapUv'],
  ['emissiveMap', 'vEmissiveMapUv'],
];

const VERTEX_HEAD = /* glsl */`
varying vec3 vTriPos;
varying vec3 vTriNormal;
varying mat3 vTriToView;
`;

const VERTEX_BODY = /* glsl */`
{
  vec4 triP = vec4( transformed, 1.0 );
  vec3 triN = objectNormal;
  mat3 triToWorld = mat3( modelMatrix );
  #ifdef USE_INSTANCING
    triToWorld = triToWorld * mat3( instanceMatrix );
  #endif
  #ifdef TRIPLANAR_WORLD
    #ifdef USE_INSTANCING
      triP = instanceMatrix * triP;
    #endif
    triP = modelMatrix * triP;
    triN = triToWorld * triN;
    vTriToView = mat3( viewMatrix );
  #else
    vTriToView = mat3( viewMatrix ) * triToWorld;
  #endif
  vTriPos = triP.xyz;
  vTriNormal = triN;
}
`;

const FRAGMENT_HEAD = /* glsl */`
varying vec3 vTriPos;
varying vec3 vTriNormal;
varying mat3 vTriToView;
uniform float triScale;
uniform float triSharpness;
uniform float triFlipY;

vec3 triWeights( vec3 n ) {
  vec3 w = pow( abs( n ), vec3( triSharpness ) );
  return w / max( dot( w, vec3( 1.0 ) ), 1e-5 );
}

vec4 triSample( sampler2D tex, vec3 p, vec3 w ) {
  return texture2D( tex, p.zy ) * w.x + texture2D( tex, p.xz ) * w.y + texture2D( tex, p.xy ) * w.z;
}

// Whiteout-blended triplanar normal mapping (normal in projection space)
vec3 triNormal( sampler2D tex, vec3 p, vec3 n, vec3 w, vec2 strength ) {
  vec3 tx = texture2D( tex, p.zy ).xyz * 2.0 - 1.0;
  vec3 ty = texture2D( tex, p.xz ).xyz * 2.0 - 1.0;
  vec3 tz = texture2D( tex, p.xy ).xyz * 2.0 - 1.0;
  vec2 s = strength * vec2( 1.0, triFlipY );
  tx.xy *= s; ty.xy *= s; tz.xy *= s;
  tx = vec3( tx.xy + n.zy, abs( tx.z ) * n.x );
  ty = vec3( ty.xy + n.xz, abs( ty.z ) * n.y );
  tz = vec3( tz.xy + n.xy, abs( tz.z ) * n.z );
  return normalize( tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z );
}
`;

const FRAGMENT_MAIN_PRELUDE = /* glsl */`
  vec3 triN = normalize( vTriNormal );
  vec3 triW = triWeights( triN );
  vec3 triP = vTriPos * triScale;
`;

const NORMAL_MAPS = /* glsl */`
#if defined( USE_NORMALMAP_TANGENTSPACE )
  {
    vec3 triMapped = triNormal( normalMap, triP, triN, triW, abs( normalScale ) );
    normal = normalize( vTriToView * triMapped );
    #ifdef DOUBLE_SIDED
      normal *= faceDirection;
    #endif
  }
#else
  #include <normal_fragment_maps>
#endif
`;

function replaceOnce(source, search, replacement, label) {
  if (!source.includes(search)) {
    console.warn(`triplanar: could not patch ${label}`);
    return source;
  }
  return source.replace(search, replacement);
}

function chunkWithTriplanar(name) {
  let chunk = THREE.ShaderChunk[name];
  for (const [map, uv] of SAMPLED_MAPS) {
    chunk = chunk.replaceAll(`texture2D( ${map}, ${uv} )`, `triSample( ${map}, triP, triW )`);
  }
  return chunk;
}

/** Patch one material in place. Returns true if it uses triplanar projection. */
export function applyTriplanar(material) {
  const cfg = material?.userData?.usdvTriplanar;
  if (!cfg || material.userData.triplanarApplied) return false;
  material.userData.triplanarApplied = true;

  const uniforms = {
    triScale: { value: cfg.scale || 1 },
    // higher = sharper transition between projections
    triSharpness: { value: cfg.sharpness || 4 },
    // maps are assumed OpenGL-style (+Y); an "invert" on the USD normal node means
    // a DirectX-style map, so green is flipped. (three's own tangent frame and its
    // normalScale sign are not used here, hence abs(normalScale) in the shader.)
    triFlipY: { value: cfg.flipNormalY ? -1 : 1 },
  };
  material.defines = { ...(material.defines || {}) };
  if (cfg.space === 'world') material.defines.TRIPLANAR_WORLD = '';

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = VERTEX_HEAD + replaceOnce(
      shader.vertexShader, '#include <project_vertex>', `#include <project_vertex>\n${VERTEX_BODY}`, 'vertex');

    let fs = FRAGMENT_HEAD + shader.fragmentShader;
    fs = replaceOnce(fs, 'void main() {', `void main() {\n${FRAGMENT_MAIN_PRELUDE}`, 'main');
    for (const name of ['map_fragment', 'roughnessmap_fragment', 'metalnessmap_fragment',
      'aomap_fragment', 'emissivemap_fragment']) {
      fs = fs.replace(`#include <${name}>`, chunkWithTriplanar(name));
    }
    fs = replaceOnce(fs, '#include <normal_fragment_maps>', NORMAL_MAPS, 'normal map');
    shader.fragmentShader = fs;
  };
  material.customProgramCacheKey = () => `usdv-triplanar-${cfg.space}`;
  material.needsUpdate = true;
  return true;
}

/** Patch every triplanar material in a loaded scene. Returns how many were patched. */
export function applyTriplanarToScene(root) {
  const seen = new Set();
  let count = 0;
  root.traverse((o) => {
    if (!o.isMesh) return;
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (!m || seen.has(m)) continue;
      seen.add(m);
      if (applyTriplanar(m)) count += 1;
    }
  });
  return count;
}
