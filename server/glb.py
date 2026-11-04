"""Minimal glTF 2.0 binary (GLB) writer.

Only what the USD converter needs: meshes with indexed triangle primitives,
PBR materials, embedded PNG/JPEG textures and a flat node list.
"""

from __future__ import annotations

import json
import struct

import numpy as np

# glTF constants
FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

_TYPE_FOR_WIDTH = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}


class GlbBuilder:
    def __init__(self) -> None:
        self._bin = bytearray()
        self.gltf: dict = {
            "asset": {"version": "2.0", "generator": "usd-viewer converter"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
        }

    # -- raw data ---------------------------------------------------------

    def _add_view(self, data: bytes, target: int | None = None) -> int:
        while len(self._bin) % 4:
            self._bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self._bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self._bin.extend(data)
        self.gltf["bufferViews"].append(view)
        return len(self.gltf["bufferViews"]) - 1

    def add_attribute(self, values: np.ndarray, target: int | None = ARRAY_BUFFER) -> int:
        """Add a float attribute, shape (N,) or (N, width).

        Vertex attributes use target=ARRAY_BUFFER; per-instance data
        (EXT_mesh_gpu_instancing) must not set a target.
        """
        arr = np.ascontiguousarray(values, dtype=np.float32)
        width = 1 if arr.ndim == 1 else arr.shape[1]
        view = self._add_view(arr.tobytes(), target)
        acc = {
            "bufferView": view,
            "componentType": FLOAT,
            "count": int(arr.shape[0]),
            "type": _TYPE_FOR_WIDTH[width],
        }
        if arr.shape[0]:
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            acc["min"] = np.atleast_1d(mins).astype(float).tolist()
            acc["max"] = np.atleast_1d(maxs).astype(float).tolist()
        self.gltf["accessors"].append(acc)
        return len(self.gltf["accessors"]) - 1

    def add_indices(self, indices: np.ndarray) -> int:
        arr = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
        view = self._add_view(arr.tobytes(), ELEMENT_ARRAY_BUFFER)
        self.gltf["accessors"].append(
            {
                "bufferView": view,
                "componentType": UNSIGNED_INT,
                "count": int(arr.shape[0]),
                "type": "SCALAR",
            }
        )
        return len(self.gltf["accessors"]) - 1

    # -- scene objects ----------------------------------------------------

    def add_image(self, data: bytes, mime: str, name: str | None = None) -> int:
        view = self._add_view(data)
        img = {"bufferView": view, "mimeType": mime}
        if name:
            img["name"] = name
        self.gltf["images"].append(img)
        return len(self.gltf["images"]) - 1

    def add_sampler(self, wrap_s: int, wrap_t: int) -> int:
        sampler = {"wrapS": wrap_s, "wrapT": wrap_t}
        for i, existing in enumerate(self.gltf["samplers"]):
            if existing == sampler:
                return i
        self.gltf["samplers"].append(sampler)
        return len(self.gltf["samplers"]) - 1

    def add_texture(self, image: int, sampler: int) -> int:
        texture = {"source": image, "sampler": sampler}
        for i, existing in enumerate(self.gltf["textures"]):
            if existing == texture:
                return i
        self.gltf["textures"].append(texture)
        return len(self.gltf["textures"]) - 1

    def add_material(self, material: dict) -> int:
        self.gltf["materials"].append(material)
        return len(self.gltf["materials"]) - 1

    def add_mesh(self, name: str, primitives: list[dict]) -> int:
        self.gltf["meshes"].append({"name": name, "primitives": primitives})
        return len(self.gltf["meshes"]) - 1

    def add_node(self, node: dict, root: bool = False) -> int:
        self.gltf["nodes"].append(node)
        idx = len(self.gltf["nodes"]) - 1
        if root:
            self.gltf["scenes"][0]["nodes"].append(idx)
        return idx

    # -- output -----------------------------------------------------------

    def to_bytes(self) -> bytes:
        gltf = {k: v for k, v in self.gltf.items() if v != []}
        if self._bin:
            while len(self._bin) % 4:
                self._bin.append(0)
            gltf["buffers"] = [{"byteLength": len(self._bin)}]

        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4:
            json_bytes += b" "

        chunks = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        if self._bin:
            chunks += struct.pack("<II", len(self._bin), 0x004E4942) + bytes(self._bin)
        header = struct.pack("<III", 0x46546C67, 2, 12 + len(chunks))
        return header + chunks
