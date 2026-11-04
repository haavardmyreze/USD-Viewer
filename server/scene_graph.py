"""The composed prim hierarchy of an opened scene, read a level at a time.

Adapted from usd-refgraph's scene tree: the viewer asks for one prim's children
when a row is expanded, a whole branch for "expand all", and one prim's detail
when it is picked, so a heavy stage is never walked whole. Every prim is listed,
not just the renderable ones: an `over` that never got a `def`, a deactivated
prim and a class are exactly what someone debugging composition wants to see.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

from pxr import Sdf, Usd

#: Children returned for one prim before the list is cut short.
CHILD_LIMIT = 500
#: Prims listed by one expand-all before the rest is left collapsed.
SUBTREE_LIMIT = 5000
#: Find: results returned, and prims looked at before giving up.
FIND_LIMIT = 200
FIND_BUDGET = 500_000
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

    # -- helpers -------------------------------------------------------------

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
