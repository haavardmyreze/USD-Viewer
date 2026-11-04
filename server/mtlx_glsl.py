"""MaterialX (and UsdPreviewSurface) shading networks -> GLSL for the viewer.

The converter's plain glTF materials can only carry one texture per channel, so
any maths, mixing, noise or projection between a texture and the surface is
lost. This module instead turns the whole pattern network into GLSL that the
viewer splices into three.js's physically based material.

The code is generated from MaterialX's own library, the same one Houdini's
viewport, usdview (Storm) and MaterialXView compile:

- nodes implemented as node graphs in the library (triplanarprojection, mix,
  ramps, UsdUVTexture, ...) are expanded into their graphs;
- nodes with a GLSL implementation use MaterialX's own source (inline
  expressions, or functions from the genglsl .glsl files, e.g. the noises);
- the few nodes MaterialX implements in C++ (position, normal, tangent,
  texcoord, geompropvalue, transforms, ...) are written here, against the
  geometry the viewer provides in USD's own spaces: object space is the prim's
  local space and world space is the USD stage's world, in stage units and
  stage up axis, exactly what the network sees in Houdini.

Only the surface shader itself (standard_surface, open_pbr_surface,
UsdPreviewSurface, gltf_pbr, surface_unlit) is not compiled from MaterialX: its
inputs are mapped onto three.js's MeshPhysicalMaterial, which provides the
lighting, shadows and environment of the viewer.

The generated code follows a small contract with web/js/mtlx.js: geometry is
available as locals named usdv_Pobject, usdv_Pworld, usdv_Nobject, ... (see
GEOM), primvars as attributes named in `attributes`, textures as sampler
uniforms usdvTex<n>, authored values in the vec4 array usdvU (so that materials
that differ only in values share one compiled program), and the results are
assigned to locals named usdv_p_<param> for each three.js parameter.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field

try:  # MaterialX is optional: without it only the plain glTF materials are written
    import MaterialX as mx
except ImportError:  # pragma: no cover
    mx = None

GLSL_TYPE = {
    "float": "float", "integer": "int", "boolean": "bool",
    "vector2": "vec2", "vector3": "vec3", "vector4": "vec4",
    "color3": "vec3", "color4": "vec4",
    "matrix33": "mat3", "matrix44": "mat4",
    "string": "int", "filename": "sampler2D",
}
WIDTH = {"float": 1, "integer": 1, "boolean": 1, "vector2": 2, "vector3": 3, "vector4": 4,
         "color3": 3, "color4": 4}
COLOR_TYPES = ("color3", "color4")

# geometric properties the viewer provides (web/js/mtlx.js declares them)
GEOM = {
    "Pobject": ("usdv_Pobject", "vector3"), "Pworld": ("usdv_Pworld", "vector3"),
    "Nobject": ("usdv_Nobject", "vector3"), "Nworld": ("usdv_Nworld", "vector3"),
    "Tobject": ("usdv_Tobject", "vector3"), "Tworld": ("usdv_Tworld", "vector3"),
    "Bobject": ("usdv_Bobject", "vector3"), "Bworld": ("usdv_Bworld", "vector3"),
    "UV0": ("usdv_UV0", "vector2"), "Vworld": ("usdv_Vworld", "vector3"),
}

MAX_TEXTURES = 12  # leaves units for three's environment, shadow and transmission maps
MAX_DEPTH = 64

# ---------------------------------------------------------------------------
# MaterialX library
# ---------------------------------------------------------------------------


def _extra_library_paths() -> list[str]:
    """Folders with more node definitions (e.g. a DCC's own MaterialX nodes).

    USDV_MATERIALX_PATH plus the variables USD itself reads for MaterialX
    (Houdini sets PXR_MTLX_PLUGIN_SEARCH_PATHS for its Karma nodes).
    """
    out = []
    for var in ("USDV_MATERIALX_PATH", "PXR_MTLX_PLUGIN_SEARCH_PATHS", "PXR_MTLX_STDLIB_SEARCH_PATHS"):
        for p in os.environ.get(var, "").split(os.pathsep):
            if p and os.path.isdir(p) and p not in out:
                out.append(p)
    return out


class Library:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "Library | None":
        if mx is None:
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self.doc = mx.createDocument()
        search = mx.getDefaultDataSearchPath()
        mx.loadLibraries(mx.getDefaultDataLibraryFolders(), search, self.doc)
        self.extra_files: list[str] = []
        for folder in _extra_library_paths():
            for root, _dirs, files in os.walk(folder):
                for f in sorted(files):
                    if f.endswith(".mtlx"):
                        path = os.path.join(root, f)
                        try:
                            mx.readFromXmlFile(self.doc, path)
                            self.extra_files.append(path)
                        except Exception:  # noqa: BLE001 - a broken file costs only its nodes
                            pass
        self.genglsl_dirs = []
        for root in search.asString().split(mx.PATH_LIST_SEPARATOR if hasattr(mx, "PATH_LIST_SEPARATOR") else os.pathsep):
            for lib in ("stdlib", "pbrlib", "nprlib", "cmlib", "bxdf", "lights"):
                d = os.path.join(root, "libraries", lib, "genglsl")
                if os.path.isdir(d):
                    self.genglsl_dirs.append(d)
        for p in self.extra_files:
            d = os.path.dirname(p)
            for sub in (d, os.path.join(d, "genglsl")):
                if os.path.isdir(sub) and sub not in self.genglsl_dirs:
                    self.genglsl_dirs.append(sub)
        self.version = mx.getVersionString()
        self._nodedefs = {nd.getName(): nd for nd in self.doc.getNodeDefs()}
        self._impl_cache: dict = {}
        self._file_cache: dict = {}
        self._sig_cache: dict = {}
        self._inputs_cache: dict = {}

    def nodedef(self, name: str):
        return self._nodedefs.get(name)

    def find_nodedef(self, category: str, out_type: str | None = None):
        """Nodedef for a node category (and output type), preferring the default version."""
        best = None
        for nd in self.doc.getNodeDefs():
            if nd.getNodeString() != category:
                continue
            if out_type and nd.getType() != out_type:
                continue
            if best is None or nd.getAttribute("isdefaultversion") == "true":
                best = nd
        return best

    def inputs(self, nd):
        """Active inputs (inherited ones included), in declaration order."""
        key = nd.getName()
        if key not in self._inputs_cache:
            self._inputs_cache[key] = list(nd.getActiveInputs())
        return self._inputs_cache[key]

    def outputs(self, nd):
        return list(nd.getActiveOutputs())

    def implementation(self, nd):
        """("ng", NodeGraph) | ("src", code) | ("file", path, function) | ("native", None) | (None, None)."""
        key = nd.getName()
        if key in self._impl_cache:
            return self._impl_cache[key]
        impl = nd.getImplementation("genglsl")
        result = (None, None)
        if impl is not None:
            if impl.isA(mx.NodeGraph):
                result = ("ng", impl)
            elif impl.getAttribute("sourcecode"):
                result = ("src", impl.getAttribute("sourcecode"))
            elif impl.getAttribute("file"):
                path = self._find_file(impl.getAttribute("file"), impl)
                result = ("file", path, impl.getAttribute("function")) if path else (None, None)
            else:
                result = ("native", None)
        self._impl_cache[key] = result
        return result

    def _find_file(self, name: str, element=None) -> str | None:
        if element is not None:
            src = element.getSourceUri()
            if src:
                d = os.path.dirname(src)
                for cand in (os.path.join(d, name), os.path.join(d, "genglsl", name)):
                    if os.path.isfile(cand):
                        return cand
        for d in self.genglsl_dirs:
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
        return None

    def glsl_chunks(self, path: str) -> list[tuple[str, str]]:
        """A .glsl file and its includes as (key, text) chunks, dependencies first."""
        if path in self._file_cache:
            return self._file_cache[path]
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        chunks: list[tuple[str, str]] = []
        body = []
        for line in text.splitlines():
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if m:
                inc = m.group(1).replace("$fileTransformUv", "mx_transform_uv_vflip.glsl")
                inc_path = os.path.normpath(os.path.join(os.path.dirname(path), inc))
                if not os.path.isfile(inc_path):
                    inc_path = self._find_file(inc) or ""
                if inc_path:
                    for c in self.glsl_chunks(inc_path) or [("unsupported", None)]:
                        if c not in chunks:
                            chunks.append(c)
                continue
            body.append(line)
        code = "\n".join(body)
        code = code.replace("$texSamplerSignature", "sampler2D tex_sampler")
        code = code.replace("$texSamplerSampler2D", "tex_sampler")
        if "$" in code:  # needs shader generator context (closures, light loops): not a pattern
            code = None
        if code is None:
            chunks = None
        else:
            chunks.append((self._chunk_key(path), code))
        self._file_cache[path] = chunks
        return chunks

    def _chunk_key(self, path: str) -> str:
        parts = os.path.normpath(path).split(os.sep)
        tail = "/".join(parts[-3:])
        return tail

    def signature(self, path: str, function: str) -> list[tuple[str, str, str]] | None:
        """Parameters of a GLSL function: [(qualifier, type, name)]."""
        key = (path, function)
        if key in self._sig_cache:
            return self._sig_cache[key]
        chunks = self.glsl_chunks(path)
        if not chunks or any(c[1] is None for c in chunks):
            self._sig_cache[key] = None
            return None
        code = chunks[-1][1]
        m = re.search(r"\bvoid\s+" + re.escape(function) + r"\s*\(([^)]*)\)", code)
        params = None
        if m:
            params = []
            for p in m.group(1).split(","):
                words = p.split()
                if not words:
                    continue
                qual = words[0] if words[0] in ("in", "out", "inout") else ""
                rest = words[1:] if qual else words
                params.append((qual, rest[0], rest[-1]))
        self._sig_cache[key] = params
        return params

    def math_chunk(self) -> tuple[str, str] | None:
        p = self._find_file(os.path.join("lib", "mx_math.glsl"))
        return self.glsl_chunks(p)[-1] if p else None


# ---------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------


def _num(v) -> str:
    f = float(v)
    if f != f:  # nan
        return "0.0"
    if f in (float("inf"), float("-inf")):
        return "1e30" if f > 0 else "-1e30"
    s = repr(f)
    if "e" in s or "E" in s:
        m, e = s.lower().split("e")
        if "." not in m:
            m += ".0"
        return f"{m}e{int(e)}"
    return s if "." in s else s + ".0"


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (int, float, bool)):
        return [v]
    if isinstance(v, str):
        return [float(x) for x in v.replace(";", ",").split(",") if x.strip()]
    try:
        flat = []
        for x in v:
            if isinstance(x, (int, float)):
                flat.append(x)
            else:
                flat.extend(_as_list(x))
        return flat
    except TypeError:
        return [v]


def literal(mtype: str, value) -> str:
    """GLSL literal for a MaterialX-typed value (strings handled by the caller)."""
    if mtype == "boolean":
        if isinstance(value, str):
            return "true" if value.strip().lower() in ("true", "1") else "false"
        return "true" if value else "false"
    if mtype == "integer":
        try:
            return str(int(float(_as_list(value)[0]))) if value not in (None, "") else "0"
        except (ValueError, IndexError, TypeError):
            return "0"
    if mtype == "float":
        vals = _as_list(value)
        return _num(vals[0] if vals else 0.0)
    if mtype in WIDTH:
        n = WIDTH[mtype]
        vals = _as_list(value)
        if not vals:
            vals = [0.0] * n
        if len(vals) == 1:
            vals = vals * n
        vals = (vals + [0.0] * n)[:n]
        return f"{GLSL_TYPE[mtype]}({', '.join(_num(x) for x in vals)})"
    if mtype in ("matrix33", "matrix44"):
        n = 9 if mtype == "matrix33" else 16
        vals = _as_list(value)
        if len(vals) != n:
            return f"{GLSL_TYPE[mtype]}(1.0)"
        return f"{GLSL_TYPE[mtype]}({', '.join(_num(x) for x in vals)})"
    return "0"


def _default(mtype: str) -> str:
    if mtype in ("matrix33", "matrix44"):
        return f"{GLSL_TYPE[mtype]}(1.0)"
    if mtype == "boolean":
        return "false"
    if mtype in ("integer", "string"):
        return "0"
    return literal(mtype, 0.0)


def convert_expr(expr: str, src: str, dst: str) -> str:
    """Convert a GLSL expression between MaterialX types (implicit USD connections)."""
    if src == dst or GLSL_TYPE.get(src) == GLSL_TYPE.get(dst):
        return expr
    sw, dw = WIDTH.get(src), WIDTH.get(dst)
    if dst == "float" and src in ("integer", "boolean"):
        return f"float({expr})"
    if dst == "integer" and src in ("float", "boolean"):
        return f"int({expr})"
    if dst == "boolean" and src in ("float", "integer"):
        return f"(({expr}) != {'0.0' if src == 'float' else '0'})"
    if sw is None or dw is None:
        return _default(dst)
    if src in ("integer", "boolean"):
        expr, sw = f"float({expr})", 1
    if dst in ("integer", "boolean"):
        return convert_expr(f"({expr}).x", "float", dst) if sw > 1 else convert_expr(expr, "float", dst)
    if sw == 1:
        return f"{GLSL_TYPE[dst]}({expr})" if dw > 1 else expr
    if dw == 1:
        return f"({expr}).x"
    if dw < sw:
        return f"({expr}).{'xyzw'[:dw]}"
    if sw == 2:
        return f"vec3({expr}, 0.0)" if dw == 3 else f"vec4({expr}, 0.0, 1.0)"
    return f"vec4({expr}, 1.0)"


# ---------------------------------------------------------------------------
# graph representation
# ---------------------------------------------------------------------------


@dataclass
class Binding:
    """What feeds a node input.

    kind: "value" (value, authored), "conn" (node, output), "geom" (name),
    "asset" (an image file; value is the asset path, colorspace its tag),
    "unset" (use the nodedef's default).
    """
    kind: str
    value: object = None
    node: "Node | None" = None
    output: str = "out"
    authored: bool = False
    colorspace: str | None = None
    mtype: str | None = None  # type of the value / upstream output when known
    anchor: object = None  # asset: the layer the path is relative to (for UDIM tiles)


@dataclass(eq=False)
class Node:
    nd: object  # mx.NodeDef
    inputs: dict = field(default_factory=dict)  # name -> Binding
    name: str = ""
    # a library node graph: its bindings live in a scope (see _NGScope)
    scope: object = None


class _NGScope:
    """One expansion of a library node graph for one outer node."""

    def __init__(self, ng, outer: Node, outer_scope):
        self.ng = ng
        self.outer = outer
        self.outer_scope = outer_scope
        self.nodes: dict[str, Node] = {}


# ---------------------------------------------------------------------------
# code generation
# ---------------------------------------------------------------------------


@dataclass
class ShaderResult:
    main: str
    chunks: list  # [(key, text)] library code, dependencies first
    params: dict  # three.js parameter -> glsl type (assigned as usdv_p_<name>)
    needs: set
    attributes: dict  # attribute name -> primvar name
    textures: list  # [{"uniform": "usdvTex0", ...texture info}]
    uniforms: list  # [[x, y, z, w]] values of usdvU
    features: dict
    warnings: list
    key: str = ""


class GenError(Exception):
    pass


class Generator:
    """Generates GLSL statements for one material."""

    def __init__(self, lib: Library, *, texture_cb, primvar_cb, frame: float = 1.0, fps: float = 24.0):
        self.lib = lib
        self.texture_cb = texture_cb
        self.primvar_cb = primvar_cb
        self.frame = frame
        self.fps = fps
        self.lines: list[str] = []
        self.chunks: list[tuple[str, str]] = []
        self.memo: dict = {}
        self.n = 0
        self.uniforms: list[list[float]] = []
        self.uniform_index: dict = {}
        self.needs: set[str] = set()
        self.attributes: dict[str, str] = {}
        self.textures: list[dict] = []
        self.texture_index: dict = {}
        self.warnings: list[str] = []
        self._active: set = set()
        mc = lib.math_chunk()
        if mc:
            self.chunks.append(mc)

    # -- bookkeeping -------------------------------------------------------

    def var(self, prefix="n") -> str:
        self.n += 1
        return f"usdv_{prefix}{self.n}"

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def add_chunks(self, chunks) -> None:
        for c in chunks:
            if c not in self.chunks:
                self.chunks.append(c)

    def uniform_expr(self, mtype: str, value) -> str:
        """An authored value as a slot of usdvU, so equal networks share a program."""
        n = WIDTH[mtype]
        vals = [float(x) for x in _as_list(value)]
        if len(vals) == 1 and n > 1:
            vals = vals * n
        vals = (vals + [0.0] * 4)[:4]
        key = (mtype, tuple(vals))
        idx = self.uniform_index.get(key)
        if idx is None:
            idx = len(self.uniforms)
            self.uniforms.append(vals)
            self.uniform_index[key] = idx
        ref = f"usdvU[{idx}]"
        return {1: f"{ref}.x", 2: f"{ref}.xy", 3: f"{ref}.xyz", 4: ref}[n]

    def geom(self, name: str) -> tuple[str, str]:
        if name in GEOM:
            self.needs.add(name)
            if name in ("Bobject", "Bworld"):
                self.needs.update(("Tobject" if name == "Bobject" else "Tworld",
                                   "Nobject" if name == "Bobject" else "Nworld"))
            return GEOM[name]
        if name.startswith("UV") and name[2:].isdigit():
            return self.texcoord(int(name[2:]))
        # a geompropdef naming another primvar
        gpd = self.lib.doc.getGeomPropDef(name)
        if gpd is not None and gpd.getGeomProp() == "geompropvalue":
            t = gpd.getType()
            return self.primvar(gpd.getAttribute("geomprop") or name, t, _default(t))
        self.warn(f"geometric property {name} is not available; using 0")
        return "vec3(0.0)", "vector3"

    def texcoord(self, index: int) -> tuple[str, str]:
        if index == 0:
            self.needs.add("UV0")
            return GEOM["UV0"]
        # the converter also tries uv<n>, st_<n> and map<n+1> on each mesh
        return self.primvar(f"st{index}", "vector2", "vec2(0.0)")

    def primvar(self, name: str, mtype: str, default: str) -> tuple[str, str]:
        """A primvar read as a vertex attribute.

        Attributes narrower than vec4 are written with w = 0, so a prim that lacks
        the primvar (WebGL's default attribute is (0, 0, 0, 1)) gets `default`.
        """
        attr = self.primvar_cb(name, mtype)
        if not attr:
            return default, mtype
        self.attributes[attr] = name
        n = WIDTH.get(mtype, 1)
        v = f"{attr}_v"
        if n == 4:
            expr = v
        else:
            swz = {1: ".x", 2: ".xy", 3: ".xyz"}[n]
            raw = v + swz
            if mtype == "integer":
                raw = f"int({raw})"
            elif mtype == "boolean":
                raw = f"({raw} != 0.0)"
            expr = f"({v}.w > 0.5 ? {default} : {raw})"
        return expr, mtype

    # -- inputs ------------------------------------------------------------

    def _string_value(self, binding: Binding | None, inp) -> str:
        if binding is not None and binding.kind == "value" and binding.value is not None:
            return str(binding.value)
        if binding is not None and binding.kind == "conn":
            return str(self.const(binding) or "")
        return inp.getValueString() if inp is not None else ""

    def const(self, binding: Binding | None):
        """The constant value behind a binding (through constant / dot nodes), or None."""
        for _ in range(16):
            if binding is None:
                return None
            if binding.kind == "value":
                return binding.value
            if binding.kind != "conn" or binding.node is None or isinstance(binding.node, _Expr):
                return None
            node = binding.node
            cat = node.nd.getNodeString()
            if cat not in ("constant", "dot"):
                return None
            binding = node.inputs.get("value" if cat == "constant" else "in")
            while binding is not None and binding.kind == "iface":
                node = node.scope.outer
                binding = node.inputs.get(binding.value)
        return None

    def resolved(self, node: Node, name: str):
        """(binding, owner, declaration) of an input, following library graph interfaces.

        binding is None when nothing is authored; declaration is the input of the
        nodedef whose default then applies.
        """
        decl = node.nd.getActiveInput(name)
        b = node.inputs.get(name)
        owner = node
        while b is not None and b.kind == "iface":
            iname = b.value
            owner = owner.scope.outer
            b = owner.inputs.get(iname)
            decl = owner.nd.getActiveInput(iname)
        return b, owner, decl

    def input_expr(self, node: Node, inp, depth: int) -> str:
        """GLSL expression for one nodedef input of `node`."""
        name = inp.getName()
        mtype = inp.getType()
        binding = node.inputs.get(name)
        owner = node
        # interface bindings point at the outer node's input; follow them
        while binding is not None and binding.kind == "iface":
            outer = owner.scope.outer
            iname = binding.value
            binding = outer.inputs.get(iname)
            owner = outer
            if binding is None:
                # the outer input is unset: use the outer nodedef's declaration
                oin = owner.nd.getActiveInput(iname)
                if oin is None:
                    return _default(mtype)
                return self._declared_default(owner, oin, mtype, depth)
        if binding is None or binding.kind == "unset":
            return self._declared_default(node, inp, mtype, depth)
        return self.binding_expr(binding, mtype, depth, inp=inp, owner=owner)

    def _declared_default(self, node: Node, inp, mtype: str, depth: int) -> str:
        gp = inp.getDefaultGeomPropString()
        if gp:
            expr, gtype = self.geom(gp)
            return convert_expr(expr, gtype, mtype)
        itype = inp.getType()
        if itype == "string":
            return str(self._enum_index(inp, inp.getValueString()))
        if itype == "filename":
            return ""  # no file: the caller substitutes the default
        val = inp.getValueString()
        if not val:
            return convert_expr(_default(itype), itype, mtype)
        return convert_expr(literal(itype, val), itype, mtype)

    @staticmethod
    def _enum_index(inp, value: str) -> int:
        enum = [e.strip() for e in (inp.getAttribute("enum") or "").split(",")] if inp is not None else []
        values = [e.strip() for e in (inp.getAttribute("enumvalues") or "").split(",")] if inp is not None else []
        v = (value or "").strip()
        if v in enum:
            i = enum.index(v)
            if values and i < len(values):
                try:
                    return int(float(values[i]))
                except ValueError:
                    return i
            return i
        try:
            return int(float(v))
        except ValueError:
            return 0

    def binding_expr(self, b: Binding, mtype: str, depth: int, inp=None, owner=None) -> str:
        if b.kind == "geom":
            expr, gtype = self.geom(b.value)
            return convert_expr(expr, gtype, mtype)
        if b.kind == "conn":
            expr, otype = self.output_expr(b.node, b.output, depth + 1)
            return convert_expr(expr, otype, mtype)
        if b.kind == "asset":
            return ""  # handled by the texture node itself
        if b.kind == "value":
            if mtype == "string":
                return str(self._enum_index(inp, str(b.value)))
            if mtype == "filename":
                return ""
            if b.authored and mtype in WIDTH and mtype not in ("integer", "boolean"):
                try:
                    return self.uniform_expr(mtype, b.value)
                except (TypeError, ValueError):
                    pass
            src = b.mtype if b.mtype in GLSL_TYPE else mtype
            try:
                return convert_expr(literal(src, b.value), src, mtype)
            except (TypeError, ValueError):
                return _default(mtype)
        return _default(mtype)

    # -- nodes -------------------------------------------------------------

    def output_expr(self, node: Node, output: str, depth: int = 0) -> tuple[str, str]:
        """(expression, MaterialX type) of one output of a node."""
        if isinstance(node, _Expr):
            return node.expr, node.mtype
        if depth > MAX_DEPTH:
            raise GenError("network too deep")
        key = (id(node), output)
        if key in self.memo:
            return self.memo[key]
        if id(node) in self._active:
            raise GenError("cycle in shading network")
        self._active.add(id(node))
        try:
            result = self._emit(node, output, depth)
        finally:
            self._active.discard(id(node))
        self.memo[key] = result
        return result

    def _out_type(self, nd, output: str) -> str:
        o = nd.getActiveOutput(output) if output else None
        if o is None:
            outs = self.lib.outputs(nd)
            o = outs[0] if outs else None
        return o.getType() if o is not None else "float"

    def _emit(self, node: Node, output: str, depth: int) -> tuple[str, str]:
        nd = node.nd
        outs = self.lib.outputs(nd)
        if not output or nd.getActiveOutput(output) is None:
            output = outs[0].getName() if outs else "out"
        otype = self._out_type(nd, output)
        if otype == "filename":
            raise GenError("filename values cannot be computed")
        if otype in ("surfaceshader", "BSDF", "EDF", "VDF", "displacementshader", "volumeshader",
                     "lightshader", "material", "closure"):
            raise GenError(f"{nd.getNodeString()} is a shader, not a pattern")
        native = self._native(node, output, otype, depth)
        if native is not None:
            return native
        kind, *impl = self.lib.implementation(nd)
        if kind == "ng":
            return self._emit_ng(node, impl[0], output, otype, depth)
        if kind == "src":
            return self._emit_src(node, impl[0], otype, depth)
        if kind == "file":
            return self._emit_file(node, impl[0], impl[1], output, depth)
        raise GenError(f"no GLSL implementation for {nd.getName()}")

    def _emit_src(self, node: Node, code: str, otype: str, depth: int) -> tuple[str, str]:
        nd = node.nd
        exprs = {}
        for inp in self.lib.inputs(nd):
            if "{{" + inp.getName() + "}}" in code:
                exprs[inp.getName()] = self.input_expr(node, inp, depth)

        def sub(m):
            return f"({exprs.get(m.group(1), '0')})"

        body = re.sub(r"\{\{(\w+)\}\}", sub, code)
        v = self.var()
        self.lines.append(f"{GLSL_TYPE[otype]} {v} = {body};")
        return v, otype

    def _emit_file(self, node: Node, path: str, function: str, output: str, depth: int,
                   sampler: str | None = None) -> tuple[str, str]:
        nd = node.nd
        sig = self.lib.signature(path, function)
        if sig is None:
            raise GenError(f"{nd.getName()} has no usable GLSL implementation")
        inputs = self.lib.inputs(nd)
        outs = self.lib.outputs(nd)
        has_file = any(i.getType() == "filename" for i in inputs)
        tex = None
        if has_file:
            tex = self._texture_for(node, depth)
            if tex is None:
                # no image: MaterialX returns the default value
                dflt = nd.getActiveInput("default")
                expr = self.input_expr(node, dflt, depth) if dflt is not None else _default(outs[0].getType())
                return expr, outs[0].getType()
            sampler = tex["uniform"]
        args = []
        n_in = len(inputs)
        out_params = [p for p in sig if p[0] == "out"]
        in_params = [p for p in sig if p[0] != "out"]
        for i, p in enumerate(in_params):
            if i < n_in:
                inp = inputs[i]
                if inp.getType() == "filename":
                    args.append(sampler or "usdvTexNone")
                    continue
                expr = self.input_expr(node, inp, depth)
                if tex is not None and inp.getName() == "texcoord" and tex.get("udim"):
                    expr = f"usdv_udim({expr}, vec2({_num(tex['udim'][0])}, {_num(tex['udim'][1])}))"
                    self._add_udim_chunk()
                args.append(expr)
            else:
                pname = p[2]
                args.append({"uv_scale": "vec2(1.0)", "uv_offset": "vec2(0.0)"}.get(pname, _default_glsl(p[1])))
        result = None
        out_vars = []
        for o, p in zip(outs, out_params):
            v = self.var()
            self.lines.append(f"{GLSL_TYPE.get(o.getType(), p[1])} {v};")
            out_vars.append(v)
            self.memo[(id(node), o.getName())] = (v, o.getType())
            if o.getName() == output:
                result = (v, o.getType())
        self.add_chunks(self.lib.glsl_chunks(path))
        self.lines.append(f"{function}({', '.join(args + out_vars)});")
        if result is None:
            result = (out_vars[0], outs[0].getType()) if out_vars else ("0.0", "float")
        if tex is not None:
            result = self._post_texture(node, tex, result, depth)
            self.memo[(id(node), output)] = result
        return result

    def _add_udim_chunk(self):
        self.add_chunks([("usdv/udim", UDIM_GLSL)])

    def _texture_for(self, node: Node, depth: int) -> dict | None:
        """Register the image a texture node reads; returns its sampler info or None."""
        nd = node.nd
        file_inp = next(i for i in self.lib.inputs(nd) if i.getType() == "filename")
        b = node.inputs.get(file_inp.getName())
        owner = node
        while b is not None and b.kind == "iface":
            owner = owner.scope.outer
            b = owner.inputs.get(b.value)
        if b is None or b.kind not in ("asset", "value") or not b.value:
            return None

        def mode(name, default):
            inp = nd.getActiveInput(name)
            if inp is None:
                return default
            binding = node.inputs.get(name)
            o = node
            while binding is not None and binding.kind == "iface":
                o = o.scope.outer
                binding = o.inputs.get(binding.value)
            val = binding.value if binding is not None and binding.kind == "value" else inp.getValueString()
            return str(val or default)

        out_type = self.lib.outputs(nd)[0].getType()
        info = {
            "asset": b.value,
            "anchor": b.anchor,
            "colorspace": b.colorspace,
            "color": out_type in COLOR_TYPES,
            "wrapS": mode("uaddressmode", "periodic"),
            "wrapT": mode("vaddressmode", "periodic"),
            "filter": mode("filtertype", "linear"),
        }
        key = (str(getattr(b.value, "path", b.value)), str(getattr(b.value, "resolvedPath", "")), info["colorspace"],
               info["color"], info["wrapS"], info["wrapT"], info["filter"])
        if key in self.texture_index:
            return self.texture_index[key]
        reg = self.texture_cb(info)
        if reg is None:
            self.texture_index[key] = None
            return None
        same = next((t for t in self.textures if t["texture"] == reg["texture"]), None)
        if same is not None:
            tex = {**same, **{k: v for k, v in reg.items() if k != "texture"}}
            tex["uniform"] = same["uniform"]
        else:
            if len(self.textures) >= MAX_TEXTURES:
                raise GenError(f"more than {MAX_TEXTURES} textures")
            tex = {**reg, "uniform": f"usdvTex{len(self.textures)}"}
            self.textures.append(tex)
        tex["constant"] = (info["wrapS"] in ("constant", "black"), info["wrapT"] in ("constant", "black"))
        self.texture_index[key] = tex
        return tex

    def _post_texture(self, node: Node, tex: dict, result, depth: int):
        """Constant address mode and colour space decoding after an image read."""
        v, otype = result
        nd = node.nd
        cu, cv = tex.get("constant", (False, False))
        tc_inp = nd.getActiveInput("texcoord")
        dflt = nd.getActiveInput("default")
        if (cu or cv) and tc_inp is not None and dflt is not None:
            tc = self.input_expr(node, tc_inp, depth)
            conds = []
            if cu:
                conds.append(f"tcq.x < 0.0 || tcq.x > 1.0")
            if cv:
                conds.append(f"tcq.y < 0.0 || tcq.y > 1.0")
            d = self.input_expr(node, dflt, depth)
            self.lines.append(f"{{ vec2 tcq = {tc}; if ({' || '.join(conds)}) {v} = {d}; }}")
        transform = tex.get("transform")
        if transform and otype in COLOR_TYPES:
            ndname = f"ND_{transform}_to_lin_rec709_{otype}"
            cnd = self.lib.nodedef(ndname)
            if cnd is not None:
                cnode = Node(cnd, {"in": Binding("conn", node=_Expr(v, otype))}, name=ndname)
                return self.output_expr(cnode, "out", depth + 1)
        return v, otype

    def _emit_ng(self, node: Node, ng, output: str, otype: str, depth: int) -> tuple[str, str]:
        scope = getattr(node, "_ng_scope", None)
        if scope is None:
            scope = _NGScope(ng, node, node.scope)
            node._ng_scope = scope
        out = ng.getOutput(output) or (ng.getOutputs()[0] if ng.getOutputs() else None)
        if out is None:
            raise GenError(f"{ng.getName()} has no output {output}")
        return self._ng_port(scope, out, otype, depth)

    def _ng_port(self, scope: _NGScope, port, want_type: str, depth: int) -> tuple[str, str]:
        """Value of an NG output (or an internal node input)."""
        iface = port.getInterfaceName()
        if iface:
            oin = scope.outer.nd.getActiveInput(iface)
            if oin is None:
                return _default(want_type), want_type
            return convert_expr(self.input_expr(scope.outer, oin, depth), oin.getType(), want_type), want_type
        nodename = port.getNodeName()
        if nodename:
            inner = self._ng_node(scope, nodename)
            expr, t = self.output_expr(inner, port.getOutputString() or "", depth + 1)
            return convert_expr(expr, t, want_type), want_type
        val = port.getValueString()
        return (literal(want_type, val) if val else _default(want_type)), want_type

    def _ng_node(self, scope: _NGScope, name: str) -> Node:
        if name in scope.nodes:
            return scope.nodes[name]
        el = scope.ng.getNode(name)
        if el is None:
            raise GenError(f"{scope.ng.getName()}: node {name} not found")
        nd = el.getNodeDef()
        if nd is None:
            raise GenError(f"{scope.ng.getName()}: no nodedef for {el.getCategory()} ({el.getType()})")
        node = Node(nd, {}, name=name, scope=scope)
        scope.nodes[name] = node
        for inp in el.getInputs():
            iname = inp.getName()
            if inp.getInterfaceName():
                node.inputs[iname] = Binding("iface", value=inp.getInterfaceName())
            elif inp.getNodeName():
                node.inputs[iname] = Binding("conn", node=self._ng_node(scope, inp.getNodeName()),
                                             output=inp.getOutputString() or "")
            elif inp.hasValueString():
                node.inputs[iname] = Binding("value", value=inp.getValueString(), mtype=inp.getType())
        return node

    # -- nodes MaterialX implements in C++ --------------------------------------

    def _native(self, node: Node, output: str, otype: str, depth: int):
        cat = node.nd.getNodeString()
        nd = node.nd

        def sval(name, default=""):
            b, _owner, decl = self.resolved(node, name)
            c = self.const(b)
            if c is not None and c != "":
                return str(c)
            if b is None and decl is not None and decl.getValueString():
                return decl.getValueString()
            return default

        if cat in ("position", "normal", "tangent", "bitangent"):
            space = sval("space", "object")
            letter = {"position": "P", "normal": "N", "tangent": "T", "bitangent": "B"}[cat]
            name = letter + ("world" if space == "world" else "object")
            expr, gtype = self.geom(name)
            return convert_expr(expr, gtype, otype), otype
        if cat == "texcoord":
            try:
                index = int(float(sval("index", "0")))
            except ValueError:
                index = 0
            expr, t = self.texcoord(index)
            return convert_expr(expr, t, otype), otype
        if cat == "geomcolor":
            expr, _ = self.primvar("displayColor", "color3", "vec3(1.0)")
            if otype == "color4":
                op, _ = self.primvar("displayOpacity", "float", "1.0")
                return f"vec4({expr}, {op})", otype
            return convert_expr(expr, "color3", otype), otype
        if cat in ("geompropvalue", "geompropvalueuniform", "UsdPrimvarReader"):
            prop = sval("geomprop" if cat != "UsdPrimvarReader" else "varname", "")
            fallback_name = "default" if cat != "UsdPrimvarReader" else "fallback"
            if otype in ("string", "filename"):
                return "0", otype
            d = nd.getActiveInput(fallback_name)
            dexpr = self.input_expr(node, d, depth) if d is not None else _default(otype)
            if not prop:
                return dexpr, otype
            expr, _ = self.primvar(prop, otype, dexpr)
            return expr, otype
        if cat == "frame":
            return _num(self.frame), "float"
        if cat == "time":
            fps_inp = nd.getActiveInput("fps")
            fps = self.input_expr(node, fps_inp, depth) if fps_inp is not None else _num(self.fps)
            return f"({_num(self.frame)} / {fps})", "float"
        if cat == "viewdirection":
            space = sval("space", "world")
            self.needs.add("Vworld")
            if space == "world":
                return "usdv_Vworld", otype
            self.needs.add("M")
            return "normalize((usdv_WorldToObj * vec4(usdv_Vworld, 0.0)).xyz)", otype
        if cat in ("transformpoint", "transformvector", "transformnormal"):
            vin = nd.getActiveInput("in")
            v = self.input_expr(node, vin, depth)
            frm = sval("fromspace", "object")
            to = sval("tospace", "world")
            frm = "world" if frm == "world" else "object"
            to = "world" if to == "world" else "object"
            if frm == to:
                return v, otype
            self.needs.add("M")
            mat = "usdv_ObjToWorld" if frm == "object" else "usdv_WorldToObj"
            if cat == "transformpoint":
                return f"({mat} * vec4({v}, 1.0)).xyz", otype
            if cat == "transformvector":
                return f"(mat3({mat}) * {v})", otype
            inv = "usdv_WorldToObj" if frm == "object" else "usdv_ObjToWorld"
            return f"normalize(transpose(mat3({inv})) * {v})", otype
        return None


class _Expr(Node):
    """A precomputed expression used as a node (e.g. the input of a colour transform)."""

    def __init__(self, expr: str, mtype: str):
        super().__init__(nd=None)
        self.expr = expr
        self.mtype = mtype


def _default_glsl(gtype: str) -> str:
    return {"float": "0.0", "int": "0", "bool": "false", "vec2": "vec2(0.0)", "vec3": "vec3(0.0)",
            "vec4": "vec4(0.0)", "mat3": "mat3(1.0)", "mat4": "mat4(1.0)"}.get(gtype, "0.0")


def _default_one(mtype: str) -> str:
    return literal(mtype, 1.0) if mtype in WIDTH else _default(mtype)


UDIM_GLSL = """\
// UDIM tiles packed into one atlas: tile (u, v) sits at column u, row v from the bottom
vec2 usdv_udim(vec2 uv, vec2 grid)
{
    vec2 tile = clamp(floor(uv), vec2(0.0), grid - 1.0);
    return (tile + fract(uv)) / grid;
}
"""

# ---------------------------------------------------------------------------
# surface shaders -> three.js MeshPhysicalMaterial
# ---------------------------------------------------------------------------

# three.js parameters the viewer can shadow, with their GLSL types
THREE_PARAMS = {
    "diffuse": "vec3", "opacity": "float", "roughness": "float", "metalness": "float",
    "emissive": "vec3", "ior": "float", "specularIntensity": "float", "specularColor": "vec3",
    "clearcoat": "float", "clearcoatRoughness": "float", "sheenColor": "vec3", "sheenRoughness": "float",
    "transmission": "float", "thickness": "float", "attenuationColor": "vec3", "attenuationDistance": "float",
    "iridescence": "float", "iridescenceIOR": "float", "iridescenceThicknessMaximum": "float",
    "anisotropyStrength": "float", "anisotropyRotation": "float",
    "normal": "vec3", "clearcoatNormal": "vec3", "occlusion": "float", "alphaCutoff": "float",
}

SURFACES = ("standard_surface", "open_pbr_surface", "UsdPreviewSurface", "gltf_pbr", "surface_unlit",
            "disney_principled", "mix")


@dataclass
class _Param:
    expr: str
    const: object = None  # python value when the input is unconnected


class SurfaceMapper:
    def __init__(self, gen: Generator):
        self.gen = gen

    def inp(self, node: Node, name: str, want: str | None = None) -> _Param:
        nd = node.nd
        i = nd.getActiveInput(name)
        if i is None:
            return _Param(_default(want or "float"), 0.0)
        mtype = i.getType()
        b = node.inputs.get(name)
        const = None
        if b is None or b.kind == "unset":
            const = i.getValueString() if not i.getDefaultGeomPropString() else None
            const = _parse_const(mtype, const) if const is not None else None
        elif b.kind == "value":
            const = _parse_const(mtype, b.value)
        expr = self.gen.input_expr(node, i, 0)
        if want and want != mtype:
            expr = convert_expr(expr, mtype, want)
        return _Param(expr, const)

    def connected(self, node: Node, name: str) -> bool:
        b = node.inputs.get(name)
        return b is not None and b.kind in ("conn", "geom")

    def map(self, node: Node) -> tuple[dict, dict]:
        cat = node.nd.getNodeString()
        fn = getattr(self, "_" + cat, None)
        if fn is None:
            raise GenError(f"surface shader {cat} is not supported")
        return fn(node)

    # every mapping returns ({param: expr}, {feature flags})

    def _standard_surface(self, n):
        p, f = {}, {}
        base, color = self.inp(n, "base"), self.inp(n, "base_color")
        sub, sub_color = self.inp(n, "subsurface"), self.inp(n, "subsurface_color")
        diffuse = f"({base.expr} * {color.expr})"
        if not _is_zero(sub):
            diffuse = f"mix({diffuse}, {base.expr} * {sub_color.expr}, {sub.expr})"
        coat, coat_color = self.inp(n, "coat"), self.inp(n, "coat_color")
        if not _is_zero(coat):
            diffuse = f"({diffuse} * mix(vec3(1.0), {coat_color.expr}, {coat.expr}))"
        p["diffuse"] = diffuse
        p["metalness"] = self.inp(n, "metalness").expr
        p["roughness"] = self.inp(n, "specular_roughness").expr
        p["specularIntensity"] = self.inp(n, "specular").expr
        p["specularColor"] = self.inp(n, "specular_color").expr
        p["ior"] = self.inp(n, "specular_IOR").expr
        if not _is_zero(coat):
            p["clearcoat"] = coat.expr
            p["clearcoatRoughness"] = self.inp(n, "coat_roughness").expr
            f["clearcoat"] = True
            if self.connected(n, "coat_normal"):
                p["clearcoatNormal"] = self.inp(n, "coat_normal").expr
        sheen = self.inp(n, "sheen")
        if not _is_zero(sheen):
            p["sheenColor"] = f"({sheen.expr} * {self.inp(n, 'sheen_color').expr})"
            p["sheenRoughness"] = self.inp(n, "sheen_roughness").expr
            f["sheen"] = True
        tr = self.inp(n, "transmission")
        if not _is_zero(tr):
            p["transmission"] = tr.expr
            p["attenuationColor"] = self.inp(n, "transmission_color").expr
            depth = self.inp(n, "transmission_depth")
            p["attenuationDistance"] = f"max({depth.expr}, 1e-4)" if not _is_zero(depth) else "1e30"
            f["transmission"] = True
        tf = self.inp(n, "thin_film_thickness")
        if not _is_zero(tf):
            p["iridescence"] = f"step(1e-3, {tf.expr})"
            p["iridescenceIOR"] = self.inp(n, "thin_film_IOR").expr
            p["iridescenceThicknessMaximum"] = tf.expr
            f["iridescence"] = True
        an = self.inp(n, "specular_anisotropy")
        if not _is_zero(an):
            p["anisotropyStrength"] = an.expr
            p["anisotropyRotation"] = f"({self.inp(n, 'specular_rotation').expr} * 6.283185307)"
            f["anisotropy"] = True
        em = self.inp(n, "emission")
        if not _is_zero(em):
            p["emissive"] = f"({em.expr} * {self.inp(n, 'emission_color').expr})"
        op = self.inp(n, "opacity")
        p["opacity"] = f"dot({op.expr}, vec3(1.0 / 3.0))"
        f["transparent"] = not _is_one(op)
        if self.connected(n, "normal"):
            p["normal"] = self.inp(n, "normal").expr
        f["thinWalled"] = _truthy(self.inp(n, "thin_walled").const)
        return p, f

    def _open_pbr_surface(self, n):
        p, f = {}, {}
        base_w, color = self.inp(n, "base_weight"), self.inp(n, "base_color")
        diffuse = f"({base_w.expr} * {color.expr})"
        sub = self.inp(n, "subsurface_weight")
        if not _is_zero(sub):
            diffuse = f"mix({diffuse}, {base_w.expr} * {self.inp(n, 'subsurface_color').expr}, {sub.expr})"
        coat = self.inp(n, "coat_weight")
        if not _is_zero(coat):
            diffuse = f"({diffuse} * mix(vec3(1.0), {self.inp(n, 'coat_color').expr}, {coat.expr}))"
        p["diffuse"] = diffuse
        p["metalness"] = self.inp(n, "base_metalness").expr
        p["roughness"] = self.inp(n, "specular_roughness").expr
        p["specularIntensity"] = self.inp(n, "specular_weight").expr
        p["specularColor"] = self.inp(n, "specular_color").expr
        p["ior"] = self.inp(n, "specular_ior").expr
        if not _is_zero(coat):
            p["clearcoat"] = coat.expr
            p["clearcoatRoughness"] = self.inp(n, "coat_roughness").expr
            f["clearcoat"] = True
            if self.connected(n, "geometry_coat_normal"):
                p["clearcoatNormal"] = self.inp(n, "geometry_coat_normal").expr
        fuzz = self.inp(n, "fuzz_weight")
        if not _is_zero(fuzz):
            p["sheenColor"] = f"({fuzz.expr} * {self.inp(n, 'fuzz_color').expr})"
            p["sheenRoughness"] = self.inp(n, "fuzz_roughness").expr
            f["sheen"] = True
        tr = self.inp(n, "transmission_weight")
        if not _is_zero(tr):
            p["transmission"] = tr.expr
            p["attenuationColor"] = self.inp(n, "transmission_color").expr
            depth = self.inp(n, "transmission_depth")
            p["attenuationDistance"] = f"max({depth.expr}, 1e-4)" if not _is_zero(depth) else "1e30"
            f["transmission"] = True
        tfw = self.inp(n, "thin_film_weight")
        if not _is_zero(tfw):
            p["iridescence"] = tfw.expr
            p["iridescenceIOR"] = self.inp(n, "thin_film_ior").expr
            p["iridescenceThicknessMaximum"] = f"({self.inp(n, 'thin_film_thickness').expr} * 1000.0)"
            f["iridescence"] = True
        an = self.inp(n, "specular_roughness_anisotropy")
        if not _is_zero(an):
            p["anisotropyStrength"] = an.expr
            p["anisotropyRotation"] = "0.0"
            f["anisotropy"] = True
        em = self.inp(n, "emission_luminance")
        if not _is_zero(em):
            p["emissive"] = f"({em.expr} * {self.inp(n, 'emission_color').expr})"
        op = self.inp(n, "geometry_opacity")
        p["opacity"] = op.expr
        f["transparent"] = not _is_one(op)
        if self.connected(n, "geometry_normal"):
            p["normal"] = self.inp(n, "geometry_normal").expr
        f["thinWalled"] = _truthy(self.inp(n, "geometry_thin_walled").const)
        return p, f

    def _UsdPreviewSurface(self, n):
        p, f = {}, {}
        p["diffuse"] = self.inp(n, "diffuseColor").expr
        spec_wf = self.inp(n, "useSpecularWorkflow")
        if _truthy(spec_wf.const):
            p["metalness"] = "0.0"
            p["specularColor"] = self.inp(n, "specularColor").expr
            p["specularIntensity"] = "1.0"
        else:
            p["metalness"] = self.inp(n, "metallic").expr
        p["roughness"] = self.inp(n, "roughness").expr
        p["ior"] = self.inp(n, "ior").expr
        cc = self.inp(n, "clearcoat")
        if not _is_zero(cc):
            p["clearcoat"] = cc.expr
            p["clearcoatRoughness"] = self.inp(n, "clearcoatRoughness").expr
            f["clearcoat"] = True
        em = self.inp(n, "emissiveColor")
        if not _is_zero(em):
            p["emissive"] = em.expr
        op = self.inp(n, "opacity")
        thr = self.inp(n, "opacityThreshold")
        p["opacity"] = op.expr
        if not _is_zero(thr):
            p["alphaCutoff"] = thr.expr
            f["alphaTest"] = True
        else:
            f["transparent"] = not _is_one(op)
        if self.connected(n, "normal"):
            # tangent-space normal in [-1, 1], as MaterialX's UsdPreviewSurface graph reads it
            nrm = self.inp(n, "normal").expr
            g = self.gen
            for name in ("Nworld", "Tworld", "Bworld"):
                g.geom(name)
            p["normal"] = f"normalize(usdv_Tworld * ({nrm}).x + usdv_Bworld * ({nrm}).y + usdv_Nworld * ({nrm}).z)"
        if self.connected(n, "occlusion"):
            p["occlusion"] = self.inp(n, "occlusion").expr
        return p, f

    def _gltf_pbr(self, n):
        p, f = {}, {}
        p["diffuse"] = self.inp(n, "base_color").expr
        p["metalness"] = self.inp(n, "metallic").expr
        p["roughness"] = self.inp(n, "roughness").expr
        p["ior"] = self.inp(n, "ior").expr
        p["specularIntensity"] = self.inp(n, "specular").expr
        p["specularColor"] = self.inp(n, "specular_color").expr
        if self.connected(n, "occlusion"):
            p["occlusion"] = self.inp(n, "occlusion").expr
        cc = self.inp(n, "clearcoat")
        if not _is_zero(cc):
            p["clearcoat"] = cc.expr
            p["clearcoatRoughness"] = self.inp(n, "clearcoat_roughness").expr
            f["clearcoat"] = True
            if self.connected(n, "clearcoat_normal"):
                p["clearcoatNormal"] = self.inp(n, "clearcoat_normal").expr
        sheen = self.inp(n, "sheen_color")
        if not _is_zero(sheen):
            p["sheenColor"] = sheen.expr
            p["sheenRoughness"] = self.inp(n, "sheen_roughness").expr
            f["sheen"] = True
        tr = self.inp(n, "transmission")
        if not _is_zero(tr):
            p["transmission"] = tr.expr
            p["thickness"] = self.inp(n, "thickness").expr
            p["attenuationColor"] = self.inp(n, "attenuation_color").expr
            p["attenuationDistance"] = self.inp(n, "attenuation_distance").expr
            f["transmission"] = True
        ir = self.inp(n, "iridescence")
        if not _is_zero(ir):
            p["iridescence"] = ir.expr
            p["iridescenceIOR"] = self.inp(n, "iridescence_ior").expr
            p["iridescenceThicknessMaximum"] = self.inp(n, "iridescence_thickness").expr
            f["iridescence"] = True
        em = self.inp(n, "emissive")
        if not _is_zero(em):
            p["emissive"] = f"({em.expr} * {self.inp(n, 'emissive_strength').expr})"
        alpha = self.inp(n, "alpha")
        mode = self.inp(n, "alpha_mode")
        mode_v = int(mode.const or 0) if isinstance(mode.const, (int, float)) else 0
        p["opacity"] = alpha.expr
        if mode_v == 1:
            p["alphaCutoff"] = self.inp(n, "alpha_cutoff").expr
            f["alphaTest"] = True
        elif mode_v == 2:
            f["transparent"] = True
        else:
            p["opacity"] = "1.0"
        if self.connected(n, "normal"):
            p["normal"] = self.inp(n, "normal").expr
        return p, f

    def _surface_unlit(self, n):
        p, f = {}, {}
        p["diffuse"] = "vec3(0.0)"
        p["metalness"] = "0.0"
        p["roughness"] = "1.0"
        p["specularIntensity"] = "0.0"
        p["emissive"] = f"({self.inp(n, 'emission').expr} * {self.inp(n, 'emission_color').expr})"
        op = self.inp(n, "opacity")
        p["opacity"] = op.expr
        f["transparent"] = not _is_one(op)
        f["unlit"] = True
        return p, f

    def _disney_principled(self, n):
        p, f = {}, {}
        p["diffuse"] = self.inp(n, "baseColor").expr
        p["metalness"] = self.inp(n, "metallic").expr
        p["roughness"] = self.inp(n, "roughness").expr
        p["specularIntensity"] = f"({self.inp(n, 'specular').expr} * 2.0)"
        cc = self.inp(n, "clearcoat")
        if not _is_zero(cc):
            p["clearcoat"] = cc.expr
            p["clearcoatRoughness"] = f"(1.0 - {self.inp(n, 'clearcoatGloss').expr})"
            f["clearcoat"] = True
        sh = self.inp(n, "sheen")
        if not _is_zero(sh):
            p["sheenColor"] = f"vec3({sh.expr})"
            p["sheenRoughness"] = "0.5"
            f["sheen"] = True
        return p, f

    def _mix(self, n):
        fg = self._surface_input(n, "fg")
        bg = self._surface_input(n, "bg")
        m = self.inp(n, "mix").expr
        pf, ff = self.map(fg) if fg is not None else ({}, {})
        pb, fb = self.map(bg) if bg is not None else ({}, {})
        p = {}
        for k in set(pf) | set(pb):
            a = pb.get(k, _three_default(k))
            b = pf.get(k, _three_default(k))
            p[k] = f"normalize(mix({a}, {b}, {m}))" if k in ("normal", "clearcoatNormal") else f"mix({a}, {b}, {m})"
        f = {k: bool(ff.get(k) or fb.get(k)) for k in set(ff) | set(fb)}
        return p, f

    def _surface_input(self, n, name):
        b = n.inputs.get(name)
        if b is None or b.kind != "conn":
            return None
        return b.node


def _three_default(name: str) -> str:
    return {"diffuse": "vec3(0.8)", "opacity": "1.0", "roughness": "0.5", "metalness": "0.0",
            "emissive": "vec3(0.0)", "ior": "1.5", "specularIntensity": "1.0", "specularColor": "vec3(1.0)",
            "clearcoat": "0.0", "clearcoatRoughness": "0.0", "sheenColor": "vec3(0.0)", "sheenRoughness": "1.0",
            "transmission": "0.0", "thickness": "0.0", "attenuationColor": "vec3(1.0)",
            "attenuationDistance": "1e30", "iridescence": "0.0", "iridescenceIOR": "1.3",
            "iridescenceThicknessMaximum": "400.0", "anisotropyStrength": "0.0", "anisotropyRotation": "0.0",
            "normal": "usdv_Nworld", "clearcoatNormal": "usdv_Nworld", "occlusion": "1.0",
            "alphaCutoff": "0.0"}.get(name, "0.0")


def _parse_const(mtype: str, value):
    if value is None:
        return None
    if mtype in ("string", "filename"):
        return value
    if mtype == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1")
        return bool(value)
    try:
        vals = _as_list(value)
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else tuple(vals)


def _is_zero(p: _Param) -> bool:
    c = p.const
    if c is None:
        return False
    vals = c if isinstance(c, tuple) else (c,)
    try:
        return all(float(v) == 0.0 for v in vals)
    except (TypeError, ValueError):
        return False


def _is_one(p: _Param) -> bool:
    c = p.const
    if c is None:
        return False
    vals = c if isinstance(c, tuple) else (c,)
    try:
        return all(float(v) >= 0.999 for v in vals)
    except (TypeError, ValueError):
        return False


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1")
    try:
        return bool(float(v)) if v is not None else False
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def generate(surface: Node, *, texture_cb, primvar_cb, frame=1.0, fps=24.0) -> ShaderResult:
    """GLSL for a surface shader node and its upstream network."""
    lib = Library.get()
    if lib is None:
        raise GenError("MaterialX is not installed")
    gen = Generator(lib, texture_cb=texture_cb, primvar_cb=primvar_cb, frame=frame, fps=fps)
    mapper = SurfaceMapper(gen)
    params, features = mapper.map(surface)
    lines = list(gen.lines)
    out_params = {}
    for name, expr in params.items():
        t = THREE_PARAMS[name]
        lines.append(f"{t} usdv_p_{name} = {expr};")
        out_params[name] = t
    main = "\n".join(lines)
    needs = set(gen.needs)
    if "normal" in out_params or "clearcoatNormal" in out_params:
        needs.add("Nworld")
    if "anisotropyStrength" in out_params:  # the viewer orients anisotropy along the tangent
        needs.update(("Tworld", "Nworld"))
    res = ShaderResult(main=main, chunks=list(gen.chunks), params=out_params, needs=needs,
                       attributes=dict(gen.attributes), textures=list(gen.textures),
                       uniforms=list(gen.uniforms), features=features, warnings=list(gen.warnings))
    key_src = "\n".join([main, ",".join(sorted(needs)), ",".join(sorted(res.attributes)),
                         str(len(res.uniforms)), ",".join(t["uniform"] for t in res.textures),
                         ",".join(k for k, _ in res.chunks), repr(sorted(features.items()))])
    res.key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    return res


# ---------------------------------------------------------------------------
# UsdShade networks -> Node graph
# ---------------------------------------------------------------------------

# USD shader ids that MaterialX's library defines under another name
_USD_IDS = {
    "UsdPreviewSurface": "ND_UsdPreviewSurface_surfaceshader",
    "UsdUVTexture": "ND_UsdUVTexture_23",
    "UsdTransform2d": "ND_UsdTransform2d",
    "UsdPrimvarReader_float": "ND_UsdPrimvarReader_float",
    "UsdPrimvarReader_float2": "ND_UsdPrimvarReader_vector2",
    "UsdPrimvarReader_float3": "ND_UsdPrimvarReader_vector3",
    "UsdPrimvarReader_float4": "ND_UsdPrimvarReader_vector4",
    "UsdPrimvarReader_normal": "ND_UsdPrimvarReader_vector3",
    "UsdPrimvarReader_point": "ND_UsdPrimvarReader_vector3",
    "UsdPrimvarReader_vector": "ND_UsdPrimvarReader_vector3",
    "UsdPrimvarReader_int": "ND_UsdPrimvarReader_integer",
    "UsdPrimvarReader_string": "ND_UsdPrimvarReader_string",
    "UsdPrimvarReader_matrix": "ND_UsdPrimvarReader_matrix44",
}

_SDF_TO_MX = {
    "float": "float", "double": "float", "half": "float", "int": "integer", "bool": "boolean",
    "float2": "vector2", "double2": "vector2", "half2": "vector2", "texCoord2f": "vector2", "texCoord2d": "vector2",
    "float3": "vector3", "double3": "vector3", "half3": "vector3", "vector3f": "vector3", "vector3d": "vector3",
    "normal3f": "vector3", "normal3d": "vector3", "point3f": "vector3", "point3d": "vector3",
    "texCoord3f": "vector3", "color3f": "color3", "color3d": "color3", "color3h": "color3",
    "float4": "vector4", "double4": "vector4", "half4": "vector4", "color4f": "color4", "color4d": "color4",
    "matrix3d": "matrix33", "matrix4d": "matrix44", "string": "string", "token": "string", "asset": "filename",
}


def sdf_to_mx(type_name) -> str | None:
    return _SDF_TO_MX.get(str(type_name.scalarType if hasattr(type_name, "scalarType") else type_name))


def _py_value(v):
    """A USD value as plain Python (tuples for vectors, flat rows for matrices)."""
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return tuple(_as_list(v))
    except Exception:  # noqa: BLE001
        return str(v)


class UsdNetwork:
    """Builds Nodes from UsdShade shader prims, resolving through NodeGraph boundaries."""

    def __init__(self, lib: Library, time, warn):
        self.lib = lib
        self.time = time
        self.warn = warn
        self.cache: dict = {}

    def node(self, shader, want_type: str | None = None) -> Node | None:
        from pxr import UsdShade

        path = shader.GetPath().pathString
        if path in self.cache:
            return self.cache[path]
        self.cache[path] = None  # cycle guard
        sid = str(shader.GetIdAttr().Get() or "")
        if not sid:
            src = shader.GetPrim().GetAttribute("info:implementationSource").Get()
            if src and str(src) != "id":
                self.warn(f"{path}: shaders from {src} files (OSL / MDL) are not supported")
                return None
        nd = self.lib.nodedef(_USD_IDS.get(sid, sid))
        rename = {}
        if nd is None:
            nd, rename = self._substitute(shader, sid, want_type)
            if nd is None:
                return None
        node = Node(nd, {}, name=path)
        self.cache[path] = node
        color_space = None
        for inp in shader.GetInputs():
            name = inp.GetBaseName()
            if name == "sourceColorSpace":  # UsdUVTexture: applies to its file
                attrs = inp.GetValueProducingAttributes()
                v = attrs[0].Get(self.time) if attrs and not UsdShade.Output.IsOutput(attrs[0]) else None
                color_space = str(v) if v else None
                continue
            targets = rename.get(name, [name])
            binding = self._binding(inp)
            if binding is None:
                continue
            for t in targets:
                if nd.getActiveInput(t) is not None:
                    node.inputs[t] = binding
        if color_space and color_space != "auto":
            for b in node.inputs.values():
                if b.kind == "asset" and not b.colorspace:
                    b.colorspace = color_space
        return node

    def _binding(self, inp) -> Binding | None:
        from pxr import Sdf, UsdShade

        attrs = inp.GetValueProducingAttributes()
        if not attrs:
            return None
        a = attrs[0]
        if UsdShade.Output.IsOutput(a):
            up = UsdShade.Shader(a.GetPrim())
            if not up:
                return None
            out_type = sdf_to_mx(a.GetTypeName())
            upnode = self.node(up, out_type)
            if upnode is None:
                return None
            return Binding("conn", node=upnode, output=a.GetBaseName(), mtype=out_type)
        v = a.Get(self.time)
        if v is None:
            return None
        if isinstance(v, Sdf.AssetPath):
            anchor = None
            try:
                stack = a.GetPropertyStack(self.time)
                anchor = stack[0].layer if stack else None
            except Exception:  # noqa: BLE001
                pass
            return Binding("asset", value=v, colorspace=_color_space(a) or _color_space(inp.GetAttr()),
                           authored=True, anchor=anchor)
        return Binding("value", value=_py_value(v), authored=True, mtype=sdf_to_mx(a.GetTypeName()))

    def _substitute(self, shader, sid: str, want_type: str | None):
        """A standard node for a DCC-specific one MaterialX does not define (e.g. Karma's)."""
        low = sid.lower()
        out_type = want_type
        if out_type is None:
            outs = shader.GetOutputs()
            out_type = sdf_to_mx(outs[0].GetTypeName()) if outs else None
        out_type = out_type or "color3"
        path = shader.GetPath().pathString
        if "triplanar" in low:
            nd = self.lib.nodedef(f"ND_triplanarprojection_{out_type}")
            if nd is not None:
                extra = " (hex tiling is not reproduced)" if "hex" in low else ""
                self.warn(f"{_short(path)}: {sid} is not a MaterialX node; shown as MaterialX "
                          f"triplanarprojection{extra}")
                return nd, {"file": ["filex", "filey", "filez"], "layer": ["layerx", "layery", "layerz"]}
        if "hextiled" in low:
            kind = "hextilednormalmap" if "normal" in low else "hextiledimage"
            nd = self.lib.nodedef(f"ND_{kind}_{'vector3' if kind.endswith('map') else out_type}")
            if nd is not None:
                self.warn(f"{_short(path)}: {sid} is not a MaterialX node; shown as MaterialX {kind}")
                return nd, {}
        if any(k in low for k in ("image", "texture", "tex")) and out_type in WIDTH:
            nd = self.lib.nodedef(f"ND_image_{out_type}")
            if nd is not None and shader.GetInput("file"):
                self.warn(f"{_short(path)}: {sid} is not a MaterialX node; shown as MaterialX image")
                return nd, {}
        # unknown node: pass its first input of the same type through
        dot = self.lib.nodedef(f"ND_dot_{out_type}")
        for inp in shader.GetInputs():
            if sdf_to_mx(inp.GetTypeName()) == out_type and inp.GetValueProducingAttributes():
                if dot is not None:
                    self.warn(f"{_short(path)}: unknown shader {sid!r}; its input {inp.GetBaseName()!r} "
                              "is passed through")
                    return dot, {inp.GetBaseName(): ["in"]}
        self.warn(f"{_short(path)}: unknown shader {sid!r}; its output is left at the default")
        return None, {}


def _short(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    return "/".join(parts[-2:]) if len(parts) > 1 else path


def _color_space(attr) -> str | None:
    try:
        if attr.HasColorSpace():
            cs = str(attr.GetColorSpace())
            if cs:
                return cs
    except Exception:  # noqa: BLE001
        pass
    try:
        from pxr import Usd
        cs = Usd.ColorSpaceAPI.ComputeColorSpaceName(attr)
        if cs:
            return str(cs)
    except Exception:  # noqa: BLE001
        pass
    return None


def surface_node(material, network: UsdNetwork):
    """The surface shader of a Material, MaterialX first (as Houdini's viewport and Karma pick it)."""
    from pxr import UsdShade

    candidates = []
    for out in material.GetSurfaceOutputs():
        ctx = out.GetBaseName().split(":")[0] if ":" in out.GetBaseName() else ""
        for a in out.GetValueProducingAttributes():
            if not UsdShade.Output.IsOutput(a):
                continue
            s = UsdShade.Shader(a.GetPrim())
            sid = str(s.GetIdAttr().Get() or "") if s else ""
            rank = 0 if sid.startswith("ND_") else 1 if sid == "UsdPreviewSurface" else 2
            if ctx in ("mtlx", "") or sid.startswith("ND_") or sid == "UsdPreviewSurface":
                candidates.append((rank, s))
    for _rank, s in sorted(candidates, key=lambda c: c[0]):
        node = network.node(s, "surfaceshader")
        if node is not None and node.nd.getNodeString() in SURFACES:
            return node
    return None
