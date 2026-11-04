// Houdini's own icons, for anything that describes a USD prim.
//
// Artists read a scene graph in Houdini's Scene Graph Tree every day, so the
// tree draws each prim with the icon Solaris gives it, as usd-refgraph does.
// The files are SideFX's official SVGs, copied unchanged into
// web/vendor/houdini/ under their Houdini names; they are full-colour artwork,
// drawn as images and never recoloured.

const BASE = 'vendor/houdini/';

/** Every icon in vendor/houdini/, by file name without extension. */
const NAMES = new Set(`COMMON__usd LOP__reference LOP__setvariant LOP__sublayer SCENEGRAPH__cylinderlight
SCENEGRAPH__disklight SCENEGRAPH__distantlight SCENEGRAPH__domelight SCENEGRAPH__geometrylight SCENEGRAPH__kind__
SCENEGRAPH__kind__assembly SCENEGRAPH__kind__component SCENEGRAPH__kind__group SCENEGRAPH__kind__leaf
SCENEGRAPH__kind__model SCENEGRAPH__kind__subcomponent SCENEGRAPH__payloads SCENEGRAPH__pointlight
SCENEGRAPH__portallight SCENEGRAPH__primtype__ SCENEGRAPH__primtype__backdrop SCENEGRAPH__primtype__basiscurves
SCENEGRAPH__primtype__blendshape SCENEGRAPH__primtype__camera SCENEGRAPH__primtype__capsule SCENEGRAPH__primtype__cone
SCENEGRAPH__primtype__cube SCENEGRAPH__primtype__cylinder SCENEGRAPH__primtype__field3dasset
SCENEGRAPH__primtype__generativeprocedural SCENEGRAPH__primtype__geomsubset SCENEGRAPH__primtype__hermitecurves
SCENEGRAPH__primtype__houdinifieldasset SCENEGRAPH__primtype__houdinilayerinfo SCENEGRAPH__primtype__instances
SCENEGRAPH__primtype__joint SCENEGRAPH__primtype__karmalightfilter SCENEGRAPH__primtype__karmaskyatmosphere
SCENEGRAPH__primtype__karmaskydomelight SCENEGRAPH__primtype__light SCENEGRAPH__primtype__lightfilter
SCENEGRAPH__primtype__lightportal SCENEGRAPH__primtype__material SCENEGRAPH__primtype__mesh
SCENEGRAPH__primtype__mesh__subdiv SCENEGRAPH__primtype__nodegraph SCENEGRAPH__primtype__nurbscurves
SCENEGRAPH__primtype__nurbspatch SCENEGRAPH__primtype__openvdbasset SCENEGRAPH__primtype__physics
SCENEGRAPH__primtype__physicsscene SCENEGRAPH__primtype__plane SCENEGRAPH__primtype__pointinstancer
SCENEGRAPH__primtype__points SCENEGRAPH__primtype__procedural SCENEGRAPH__primtype__renderpass
SCENEGRAPH__primtype__renderproduct SCENEGRAPH__primtype__rendersettings SCENEGRAPH__primtype__rendervar
SCENEGRAPH__primtype__scope SCENEGRAPH__primtype__shader SCENEGRAPH__primtype__skelanimation
SCENEGRAPH__primtype__skeleton SCENEGRAPH__primtype__skelroot SCENEGRAPH__primtype__sphere
SCENEGRAPH__primtype__tetmesh SCENEGRAPH__primtype__unknown SCENEGRAPH__primtype__volume SCENEGRAPH__primtype__xform
SCENEGRAPH__rectlight SCENEGRAPH__shapedlight SCENEGRAPH__spherelight`.split(/\s+/));

/** A Houdini icon by its file name, e.g. `LOP__reference`. */
export function houdiniIcon(name, title, className = 'hicon') {
  const img = document.createElement('img');
  img.className = className;
  img.src = BASE + (NAMES.has(name) ? name : 'SCENEGRAPH__primtype__unknown') + '.svg';
  img.alt = '';
  img.draggable = false;
  img.setAttribute('aria-hidden', 'true');
  if (title) img.title = title;
  return img;
}

/**
 * The Scene Graph Tree icon for a prim type. Houdini names these after the
 * schema, lowercased; lights carry no `primtype` prefix and versioned schemas
 * (`DomeLight_1`) share their unversioned icon.
 */
export function primTypeIcon(typeName) {
  if (!typeName) return 'SCENEGRAPH__primtype__';
  const base = typeName.replace(/_\d+$/, '').toLowerCase();
  for (const name of [`SCENEGRAPH__primtype__${base}`, `SCENEGRAPH__${base}`]) {
    if (NAMES.has(name)) return name;
  }
  return 'SCENEGRAPH__primtype__unknown';
}

/** The kind stars, for the kinds Houdini draws. */
export function kindIcon(kind) {
  if (!kind) return null;
  const name = `SCENEGRAPH__kind__${kind.toLowerCase()}`;
  return NAMES.has(name) ? name : null;
}

const GROUPING_TYPES = new Set(['', 'Xform', 'Scope']);

/**
 * The one icon a row shows: on an Xform, Scope or typeless prim the kind's
 * stars stand in for the type icon, since there the kind is what matters.
 */
export function primIcon(typeName, kind) {
  const stars = GROUPING_TYPES.has(typeName) ? kindIcon(kind) : null;
  return stars ?? primTypeIcon(typeName);
}

/** How an opinion's layer was brought in, as the LOP that authors that arc. */
export const ARC_ICON = {
  root: 'COMMON__usd',
  sublayer: 'LOP__sublayer',
  reference: 'LOP__reference',
  payload: 'SCENEGRAPH__payloads',
  variant: 'LOP__setvariant',
};
