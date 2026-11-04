import json
import struct

import numpy as np

_DTYPES = {5126: np.float32, 5125: np.uint32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def parse_glb(data: bytes):
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    json_len, json_type = struct.unpack("<II", data[12:20])
    assert json_type == 0x4E4F534A
    gltf = json.loads(data[20:20 + json_len])
    rest = data[20 + json_len:]
    binary = b""
    if rest:
        bin_len, bin_type = struct.unpack("<II", rest[:8])
        assert bin_type == 0x004E4942
        binary = rest[8:8 + bin_len]
    return gltf, binary


def accessor(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    width = _WIDTH[acc["type"]]
    arr = np.frombuffer(binary, dtype=_DTYPES[acc["componentType"]],
                        count=acc["count"] * width, offset=view["byteOffset"])
    return arr.reshape(acc["count"], width) if width > 1 else arr
