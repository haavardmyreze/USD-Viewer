"""The composed prim hierarchy of an opened scene, read a level at a time.

Adapted from usd-refgraph's scene tree: the viewer asks for one prim's children
when a row is expanded, a whole branch for "expand all", and one prim's detail
when it is picked, so a heavy stage is never walked whole. Every prim is listed,
not just the renderable ones: an `over` that never got a `def`, a deactivated
prim and a class are exactly what someone debugging composition wants to see.
"""

from __future__ import annotations

import glob
import os
import threading
from collections import OrderedDict
from typing import Any

from pxr import Gf, Pcp, Sdf, Usd, UsdGeom, UsdShade

from . import pipeline

#: Children returned for one prim before the list is cut short.
CHILD_LIMIT = 500
#: Prims listed by one expand-all before the rest is left collapsed.
SUBTREE_LIMIT = 5000
#: Find: results returned, and prims looked at before giving up.
FIND_LIMIT = 200
FIND_BUDGET = 500_000
#: Other copies of an instance listed for one selection before the list is cut short.
PEER_LIMIT = 20_000
#: Stages kept open between requests: opening is the slow part.
MAX_OPEN_STAGES = 3

#: Instance proxies are walked into so an instanced asset does not read as empty.
PREDICATE = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

SPECIFIER = {Sdf.SpecifierDef: "def", Sdf.SpecifierOver: "over", Sdf.SpecifierClass: "class"}


class SceneGraphError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class StageCache:
    """A few recently used stages, keyed by scene id (which changes with the files)."""

    def __init__(self, size: int = MAX_OPEN_STAGES) -> None:
        self.size = size
        self.lock = threading.RLock()
        self._stages: OrderedDict[str, Usd.Stage] = OrderedDict()

    def get(self, key: str, root_path: str, variants: dict | None = None) -> Usd.Stage:
        with self.lock:
            stage = self._stages.get(key)
            if stage is not None:
                self._stages.move_to_end(key)
                return stage
            if not os.path.isfile(root_path):
                raise SceneGraphError(404, f"The scene's root layer is gone: {root_path}")
            try:
                # opened like the conversion (payloads loaded, the same variant
                # selections), so the tree shows what is rendered
                from .usd_to_glb import refresh_layers, variant_session_layer
                refresh_layers()
                stage = Usd.Stage.Open(Sdf.Layer.FindOrOpen(root_path), variant_session_layer(variants),
                                       Usd.Stage.LoadAll)
            except Exception as exc:  # noqa: BLE001
                first = str(exc).strip().splitlines()[0] if str(exc) else root_path
                raise SceneGraphError(422, f"USD could not compose this file: {first}") from exc
            if stage is None:
                raise SceneGraphError(422, f"USD could not compose this file: {root_path}")
            self._stages[key] = stage
            while len(self._stages) > self.size:
                self._stages.popitem(last=False)
            return stage


STAGES = StageCache()


class SceneGraph:
    """Queries against one scene's stage. `display` shortens layer paths for display."""

    def __init__(self, key: str, root_path: str, display=None, variants: dict | None = None) -> None:
        self.stage = STAGES.get(key, root_path, variants)
        self.display = display or (lambda p: p)

    def children(self, prim_path: str = "/", limit: int = CHILD_LIMIT) -> dict[str, Any]:
        with STAGES.lock:
            return self._level(self._prim(prim_path), limit)[0]

    def subtree(self, prim_path: str = "/", max_prims: int = SUBTREE_LIMIT) -> dict[str, Any]:
        """Every level beneath `prim_path`, breadth first, up to a budget."""
        with STAGES.lock:
            queue = [self._prim(prim_path)]
            levels: list[dict[str, Any]] = []
            listed = 0
            truncated = False
            while queue:
                prim = queue.pop(0)
                if listed >= max_prims:
                    truncated = True
                    break
                level, shown = self._level(prim, CHILD_LIMIT)
                levels.append(level)
                listed += len(shown)
                queue.extend(child for child in shown if child.GetFilteredChildren(PREDICATE))
            return {"levels": levels, "truncated": truncated, "prims": listed}

    def reveal(self, prim_path: str) -> dict[str, Any]:
        """The levels from the root down to `prim_path`'s parent, so the tree can open
        onto a prim picked in the 3D view. Each level is sure to list the next prim on
        the way, even past the child limit."""
        with STAGES.lock:
            target = self._prim(prim_path)
            chain = []
            p = target
            while p and not p.IsPseudoRoot():
                chain.append(p)
                p = p.GetParent()
            chain.reverse()
            levels = []
            parent = self.stage.GetPseudoRoot()
            for prim in chain:
                level, shown = self._level(parent, CHILD_LIMIT)
                if prim not in shown:
                    level["children"].append(_summary(prim))
                levels.append(level)
                parent = prim
            return {"levels": levels, "path": str(target.GetPath())}

    def find(self, query: str, limit: int = FIND_LIMIT, budget: int = FIND_BUDGET) -> dict[str, Any]:
        """Prims whose name contains `query` (any case), shallowest first.

        Walks the stage, so it reaches prims no one has expanded yet; it stops
        after `budget` prims so a huge shot cannot stall the server.
        """
        needle = query.strip().lower()
        if not needle:
            return {"results": [], "truncated": False}
        with STAGES.lock:
            hits: list[Usd.Prim] = []
            scanned = 0
            truncated = False
            for prim in Usd.PrimRange(self.stage.GetPseudoRoot(), PREDICATE):
                if prim.IsPseudoRoot():
                    continue
                scanned += 1
                if scanned > budget:
                    truncated = True
                    break
                if needle in prim.GetName().lower():
                    hits.append(prim)
            hits.sort(key=lambda p: (p.GetPath().pathElementCount, str(p.GetPath())))
            if len(hits) > limit:
                truncated = True
            return {"results": [_summary(p) for p in hits[:limit]], "truncated": truncated}

    def detail(self, prim_path: str) -> dict[str, Any]:
        with STAGES.lock:
            prim = self._prim(prim_path)
            out = _summary(prim)
            variant_sets = prim.GetVariantSets()
            out["variantSets"] = [
                {
                    "name": name,
                    "selection": variant_sets.GetVariantSelection(name) or None,
                    "options": list(variant_sets.GetVariantSet(name).GetVariantNames()),
                }
                for name in variant_sets.GetNames()
            ]
            out["appliedSchemas"] = list(prim.GetAppliedSchemas())
            out["attributeCount"] = len(prim.GetAttributes())
            out["relationshipCount"] = len(prim.GetRelationships())
            out["opinions"] = self._opinions(prim)
            return out

    def selection(self, prim_path: str, instance: int | None = None) -> dict[str, Any]:
        """What the 3D view draws for a selection besides the highlight.

        `pivot`: the prim's frame in stage space (USD's row-major local-to-world
        matrix, at the time the conversion uses) and its pivot point in that frame
        (an authored `xformOp:translate:pivot`, else the origin); for a
        PointInstancer instance, that instance's frame. None when the prim has no
        transform of its own.

        `instances`: the other prims that are the same native instance, i.e. the
        same prim of the same prototype under every other instance of it.
        """
        with STAGES.lock:
            prim = self._prim(prim_path)
            if prim.IsPseudoRoot():
                return {"pivot": None, "instances": [], "truncated": False}
            time = Usd.TimeCode.EarliestTime()
            peers, truncated = self._instance_peers(prim)
            return {"pivot": _pivot(prim, instance, time), "instances": peers, "truncated": truncated}

    def info(self, prim_path: str, instance: int | None = None, proto: str | None = None) -> dict[str, Any]:
        """Who made what was clicked, and what state it is in: the info card.

        `entities` runs from the inside out: the asset the prim belongs to, then
        whatever placed it (a set, a shot), ending with the stage's own root
        entity. Each lists its published layers with their publish records
        (customLayerData), read the way usd-refgraph reads them. `materials` are
        the ones bound to the clicked geometry, with the textures they read.
        """
        with STAGES.lock:
            prim = self._prim(prim_path)
            if prim.IsPseudoRoot():
                raise SceneGraphError(400, "Pick a prim, not the stage")
            target = prim
            copies = None
            if instance is not None and prim.IsA(UsdGeom.PointInstancer):
                copies, proto_root = _instancer_copies(prim, instance)
                if proto_root:
                    target = proto_root
            if proto:
                inner = self.stage.GetPrimAtPath(proto) if Sdf.Path.IsValidPathString(proto) else None
                if inner:
                    target = inner
            if copies is None and (prim.IsInstance() or prim.IsInstanceProxy()):
                peers, truncated = self._instance_peers(prim)
                if peers:
                    copies = {"kind": "instance", "same": len(peers) + 1, "truncated": truncated}

            entities = self._entities(target)
            asset = entities[0]["name"] if entities else None
            materials = self._materials(target, asset)
            out: dict[str, Any] = {
                "prim": _summary(prim),
                "target": str(target.GetPath()),
                "entities": entities,
                "materials": materials,
            }
            if instance is not None:
                out["instance"] = instance
            if copies:
                out["copies"] = copies
            return out

    # -- helpers -------------------------------------------------------------

    def _entities(self, target: Usd.Prim) -> list[dict[str, Any]]:
        """The pipeline entities `target` belongs to, innermost first.

        Walks up from the prim: the nearest prim that references (or payloads) a
        published layer is where its asset enters the scene, and the layers of
        that arc's layer stack (the assembly and its sublayered blocks, plus any
        nested arcs on the same prim) are that asset's publishes. Further up, the
        same again for whatever placed it, and last the root layer stack.
        """
        found: OrderedDict[str, dict[str, Any]] = OrderedDict()

        def layers_of(entity: str) -> OrderedDict:
            return found[entity]["_layers"]

        def take(layer: Sdf.Layer, entry: dict[str, Any]) -> None:
            if layer.anonymous:
                return
            name, _ = pipeline.split_entity(layer.realPath or layer.identifier)
            if name not in found:
                found[name] = dict(entry, name=name, _layers=OrderedDict())
            layers_of(name).setdefault(layer.identifier, layer)

        prim = target
        while prim and not prim.IsPseudoRoot():
            for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs():
                if arc.IsAncestral() or arc.GetArcType() not in (Pcp.ArcTypeReference, Pcp.ArcTypePayload):
                    continue
                introducing = arc.GetIntroducingLayer()
                entry = {
                    "primPath": str(prim.GetPath()),
                    "arc": "payload" if arc.GetArcType() == Pcp.ArcTypePayload else "reference",
                    "introducedBy": os.path.basename(introducing.realPath or introducing.identifier)
                    if introducing and not introducing.anonymous else None,
                    "_prim": prim,
                }
                for layer in arc.GetTargetNode().layerStack.layers:
                    take(layer, entry)
            prim = prim.GetParent()

        # the root layer stack: the stage's own entity, and anything it sublayers
        # that has a say about where the asset was placed (a set inside a shot)
        stack = [layer for layer in self.stage.GetLayerStack(includeSessionLayers=False) if not layer.anonymous]
        root_layer = self.stage.GetRootLayer()
        root_name, _ = pipeline.split_entity(root_layer.realPath or root_layer.identifier)
        anchor = Sdf.Path(next(reversed(found.values()))["primPath"]) if found else target.GetPath()
        root_entry = {"primPath": "/", "arc": "root", "introducedBy": None, "_prim": None}
        for layer in stack:
            name, _ = pipeline.split_entity(layer.realPath or layer.identifier)
            if name == root_name or name in found or layer.GetPrimAtPath(anchor):
                take(layer, dict(root_entry, arc="root" if name == root_name else "sublayer"))
        if root_name in found:  # the stage's own entity is the outermost
            found.move_to_end(root_name)

        return [self._entity(entry) for entry in found.values()]

    def _entity(self, entry: dict[str, Any]) -> dict[str, Any]:
        name = entry["name"]
        layers = [self._layer(layer) for layer in entry["_layers"].values()]
        layers.sort(key=lambda layer: layer["role"] != "assembly")  # stable: blocks keep their order
        role, tier = next(((layer["role"], layer["tier"]) for layer in layers if layer["role"] == "assembly"),
                          (layers[0]["role"], layers[0]["tier"]) if layers else ("other", None))
        records = [layer["record"] for layer in layers if layer.get("record")]
        artists: list[str] = []
        for record in sorted(records, key=lambda r: -(r.get("exportedAt") or 0)):
            if record.get("artist") and record["artist"] not in artists:
                artists.append(record["artist"])
        issues = []
        unrecorded = [layer["name"] for layer in layers if not layer.get("record")]
        if unrecorded:
            issues.append({"level": "warn", "text": f"No publish record in {', '.join(unrecorded)}"})
        odd = [layer["name"] for layer in layers if layer["role"] == "other"]
        if odd:
            issues.append({"level": "info", "text": f"Outside the naming convention: {', '.join(odd)}"})
        for layer in layers:
            raw = (layer.get("record") or {}).get("statusRaw")
            if raw:
                issues.append({"level": "info", "text": f"{layer['name']}: status “{raw}” is not one the pipeline knows"})

        out: dict[str, Any] = {
            "name": name,
            "tier": tier if role in ("assembly", "block") else None,
            "category": pipeline.category_of(name),
            "primPath": entry["primPath"],
            "arc": entry["arc"],
            "status": pipeline.rollup_status([r["status"] for r in records]),
            "artists": artists,
            "layers": layers,
            "issues": issues,
        }
        published = [r["exportedAt"] for r in records if r.get("exportedAt")]
        if published:
            out["lastPublished"] = max(published)
        if entry.get("introducedBy"):
            out["introducedBy"] = entry["introducedBy"]
        prim = entry.get("_prim")
        if prim:
            variant_sets = prim.GetVariantSets()
            out["variants"] = [
                {"name": set_name, "selection": variant_sets.GetVariantSelection(set_name) or None,
                 "options": list(variant_sets.GetVariantSet(set_name).GetVariantNames())}
                for set_name in variant_sets.GetNames()
            ]
        return out

    def _layer(self, layer: Sdf.Layer) -> dict[str, Any]:
        real = layer.realPath or layer.identifier
        role, tier = pipeline.classify(real)
        _, block = pipeline.split_entity(real)
        out: dict[str, Any] = {
            "name": os.path.basename(real),
            "path": self.display(real),
            "role": role,
            "tier": tier,
            "block": block,
        }
        record = pipeline.read_record(layer.customLayerData)
        if record:
            out["record"] = record
        try:
            out["modified"] = int(os.path.getmtime(real) * 1000)
        except OSError:
            pass
        return out

    def _materials(self, target: Usd.Prim, asset: str | None, limit: int = 3, budget: int = 2000) -> list[dict[str, Any]]:
        """The materials bound to the target, or to the geometry beneath it, with the
        textures each reads."""
        seen: OrderedDict[str, Usd.Prim] = OrderedDict()
        for n, prim in enumerate(Usd.PrimRange(target, PREDICATE)):
            if n >= budget or len(seen) >= limit:
                break
            if not prim.IsA(UsdGeom.Imageable):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material and str(material.GetPath()) not in seen:
                seen[str(material.GetPath())] = material.GetPrim()
        out = []
        for path, prim in seen.items():
            entry: dict[str, Any] = {"path": path, "name": prim.GetName(), "textures": self._textures(prim)}
            owners = self._entities(prim)
            # a material from somewhere else than the asset itself, e.g. a shared library
            if owners and owners[0]["name"] != asset:
                entity = owners[0]
                entry["entity"] = {"name": entity["name"], "status": entity["status"],
                                   "artists": entity["artists"], "lastPublished": entity.get("lastPublished")}
            out.append(entry)
        return out

    def _textures(self, material: Usd.Prim) -> list[dict[str, Any]]:
        textures: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for prim in Usd.PrimRange(material, PREDICATE):
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attr.Get()
                if not value or not value.path:
                    continue
                authored = value.path
                resolved = value.resolvedPath
                exists = bool(resolved) and os.path.isfile(resolved)
                if "<UDIM>" in authored and not resolved:
                    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
                    if stack:
                        pattern = stack[0].layer.ComputeAbsolutePath(authored)
                        exists = bool(glob.glob(pattern.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")))
                        resolved = pattern
                if authored in textures:
                    continue
                entry: dict[str, Any] = {
                    "name": os.path.basename(authored),
                    "path": self.display(resolved) if resolved else authored,
                    "exists": exists,
                    "entity": pipeline.split_entity(authored)[0],
                }
                entry.update(pipeline.texture_info(authored))
                textures[authored] = entry
        return list(textures.values())

    def _instance_peers(self, prim: Usd.Prim, limit: int = PEER_LIMIT) -> tuple[list[str], bool]:
        if prim.IsInstance():
            in_proto = prim.GetPrototype()
        elif prim.IsInstanceProxy():
            in_proto = prim.GetPrimInPrototype()
            # nested instancing: down to the innermost prototype, so copies of it
            # placed outside the outer prototype are found too
            while in_proto and in_proto.IsInstanceProxy():
                in_proto = in_proto.GetPrimInPrototype()
        else:
            return [], False
        if not in_proto:
            return [], False

        own = prim.GetPath()
        out: list[str] = []
        seen: set[Sdf.Path] = set()
        pending = [in_proto.GetPath()]
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            root = path.GetPrefixes()[0]  # the prototype, e.g. /__Prototype_1
            proto = self.stage.GetPrimAtPath(root)
            if not proto or not proto.IsPrototype():
                continue
            rel = path.MakeRelativePath(root)
            for inst in proto.GetInstances():
                target = inst.GetPath() if rel == Sdf.Path(".") else inst.GetPath().AppendPath(rel)
                if inst.IsInPrototype():
                    pending.append(target)  # inside another prototype: its copies are the copies
                elif target != own:
                    out.append(str(target))
                    if len(out) >= limit:
                        return sorted(out), True
        return sorted(out), False


    def _level(self, prim: Usd.Prim, limit: int) -> tuple[dict[str, Any], list[Usd.Prim]]:
        kids = prim.GetFilteredChildren(PREDICATE)
        shown = list(kids[:limit])
        default = self.stage.GetDefaultPrim()
        return {
            "primPath": str(prim.GetPath()),
            "defaultPrim": str(default.GetPath()) if default else None,
            "children": [_summary(child) for child in shown],
            "total": len(kids),
        }, shown

    def _prim(self, prim_path: str) -> Usd.Prim:
        if not prim_path or not Sdf.Path.IsValidPathString(prim_path):
            raise SceneGraphError(400, f"Not a prim path: {prim_path}")
        path = Sdf.Path(prim_path)
        prim = (self.stage.GetPseudoRoot() if path == Sdf.Path.absoluteRootPath
                else self.stage.GetPrimAtPath(path))
        if not prim:
            raise SceneGraphError(404, f"No such prim: {prim_path}")
        return prim

    def _opinions(self, prim: Usd.Prim) -> list[dict[str, Any]]:
        """Every layer holding an opinion on the prim, strongest first, labelled with
        the arc that brought it in. The session layer is left out."""
        out: list[dict[str, Any]] = []
        root_layers = {layer.identifier for layer in self.stage.GetLayerStack(includeSessionLayers=False)}
        root_id = self.stage.GetRootLayer().identifier
        for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs():
            node = arc.GetTargetNode()
            kind = str(arc.GetArcType()).rsplit("ArcType", 1)[-1].lower()
            introducing = arc.GetIntroducingLayer()
            for layer in node.layerStack.layers:
                if layer.anonymous or not layer.GetPrimAtPath(node.path):
                    continue
                real = layer.realPath or layer.identifier
                this_arc = kind
                # the root layer stack is the root layer plus its sublayers
                if kind == "root" and layer.identifier != root_id and layer.identifier in root_layers:
                    this_arc = "sublayer"
                entry: dict[str, Any] = {
                    "layerPath": self.display(real),
                    "name": os.path.basename(real),
                    "specPath": str(node.path),
                    "arc": this_arc,
                    "root": layer.identifier == root_id,
                }
                if introducing and not introducing.anonymous and kind != "root":
                    entry["introducedBy"] = os.path.basename(introducing.realPath or introducing.identifier)
                out.append(entry)
        return out


def _summary(prim: Usd.Prim) -> dict[str, Any]:
    """What a tree row needs: cheap queries only, since a level can be long."""
    arcs: list[str] = []
    if prim.HasAuthoredReferences():
        arcs.append("reference")
    if prim.HasAuthoredPayloads():
        arcs.append("payload")
    if prim.HasAuthoredInherits():
        arcs.append("inherit")
    if prim.HasAuthoredSpecializes():
        arcs.append("specialize")
    variant_sets = prim.GetVariantSets()
    variants = {name: variant_sets.GetVariantSelection(name) for name in variant_sets.GetNames()}

    out: dict[str, Any] = {
        "name": prim.GetName(),
        "path": str(prim.GetPath()),
        "typeName": str(prim.GetTypeName()),
        "specifier": SPECIFIER.get(prim.GetSpecifier(), "def"),
        "active": prim.IsActive(),
        "childCount": len(prim.GetFilteredChildren(PREDICATE)),
        "arcs": arcs,
    }
    kind = Usd.ModelAPI(prim).GetKind()
    if kind:
        out["kind"] = kind
    if variants:
        out["variants"] = variants
    if prim.IsInstance():
        out["instance"] = True
    if prim.IsInstanceProxy():
        out["instanceProxy"] = True
    if prim.HasAuthoredPayloads() and not prim.IsLoaded():
        out["unloaded"] = True
    return out


def _pivot(prim: Usd.Prim, instance: int | None, time: Usd.TimeCode) -> dict[str, Any] | None:
    cache = UsdGeom.XformCache(time)
    if instance is not None and prim.IsA(UsdGeom.PointInstancer):
        xforms = UsdGeom.PointInstancer(prim).ComputeInstanceTransformsAtTime(
            time, time, UsdGeom.PointInstancer.ExcludeProtoXform, UsdGeom.PointInstancer.IgnoreMask)
        if xforms is None or not 0 <= instance < len(xforms):
            return None
        world = Gf.Matrix4d(xforms[instance]) * cache.GetLocalToWorldTransform(prim)
        return {"matrix": _flat(world), "point": [0.0, 0.0, 0.0]}
    if not prim.IsA(UsdGeom.Xformable):
        return None
    point = [0.0, 0.0, 0.0]
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        # XformCommonAPI's pivot: translate:pivot ... !invert!translate:pivot puts
        # the pivot at that offset in the prim's own frame
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and not op.IsInverseOp() \
                and op.SplitName()[-1] == "pivot":
            value = op.Get(time)
            if value is not None:
                point = [float(v) for v in value]
            break
    return {"matrix": _flat(cache.GetLocalToWorldTransform(prim)), "point": point}


def _instancer_copies(prim: Usd.Prim, instance: int) -> tuple[dict[str, Any] | None, Usd.Prim | None]:
    """How many of a PointInstancer's instances share this one's prototype, and that prototype."""
    instancer = UsdGeom.PointInstancer(prim)
    indices = instancer.GetProtoIndicesAttr().Get(Usd.TimeCode.EarliestTime())
    if indices is None or not 0 <= instance < len(indices):
        return None, None
    index = int(indices[instance])
    targets = instancer.GetPrototypesRel().GetForwardedTargets()
    proto = prim.GetStage().GetPrimAtPath(targets[index]) if 0 <= index < len(targets) else None
    copies = {
        "kind": "pointinstancer",
        "same": sum(1 for i in indices if int(i) == index),
        "total": len(indices),
        "instancer": str(prim.GetPath()),
    }
    if proto:
        copies["prototype"] = proto.GetName()
    return copies, proto or None


def _flat(m: Gf.Matrix4d) -> list[float]:
    return [float(m[r][c]) for r in range(4) for c in range(4)]
