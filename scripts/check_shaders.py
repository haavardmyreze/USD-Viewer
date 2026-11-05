"""Check the compiled shading networks in a real browser (WebGL2, headless Chromium).

    python scripts/check_shaders.py

Needs Node.js with Playwright and a Chromium for it (npm i -g playwright &&
npx playwright install chromium). It:

1. compiles the GLSL of every MaterialX library node that computes a value;
2. loads tests/make_shader_scenes.py's scenes in the viewer and checks that no
   material had to fall back to its plain glTF version;
3. reads back pixels and compares them with values computed in USD's own spaces:
   world / object position, texture coordinates, image orientation, UDIM tiles,
   and a world-space MaterialX triplanarprojection evaluated on the CPU.

Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import mtlx_glsl  # noqa: E402
from tests import make_shader_scenes  # noqa: E402

BROWSER_ARGS = "['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']"

COMPILE_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const browser = await chromium.launch({ args: ARGS });
  const page = await browser.newPage();
  const failed = await page.evaluate((cases) => {
    const gl = document.createElement('canvas').getContext('webgl2');
    const head = `#version 300 es
precision highp float; precision highp int;
in vec3 v; out vec4 fragColor;
uniform vec4 usdvU[4]; uniform sampler2D usdvTex0; in vec4 _usdpv0_v;
`;
    const out = [];
    for (const c of cases) {
      const src = head + c.chunks.join('\n') + `
void main() {
vec3 usdv_Pobject = v, usdv_Pworld = v, usdv_Nobject = v, usdv_Nworld = v, usdv_Tobject = v, usdv_Tworld = v;
vec3 usdv_Bobject = v, usdv_Bworld = v, usdv_Vworld = v; vec2 usdv_UV0 = v.xy;
mat4 usdv_ObjToWorld = mat4(1.0), usdv_WorldToObj = mat4(1.0);
vec4 acc = vec4(0.0);
${c.main}
fragColor = acc;
}`;
      const sh = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) out.push([c.name, gl.getShaderInfoLog(sh)]);
      gl.deleteShader(sh);
    }
    return out;
  }, cases);
  console.log(JSON.stringify(failed));
  await browser.close();
})();
""".replace("ARGS", BROWSER_ARGS)

VIEW_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base, scene, outFile, camPos, target] = process.argv.slice(2);
  const browser = await chromium.launch({ args: ARGS });
  const page = await browser.newPage({ viewport: { width: 800, height: 600 }, deviceScaleFactor: 1 });
  await page.goto(`${base}/#path=${encodeURIComponent(scene)}`);
  await page.waitForFunction(() => window.usdViewer?.model, null, { timeout: 180000 });
  await page.waitForTimeout(500);
  const res = await page.evaluate(async ([cp, tg]) => {
    const THREE = await import('three');
    const v = window.usdViewer;
    const own = [...v._originalMaterials.values()].flat();
    const failed = [...(v.failedShaders || [])];
    const graphs = own.filter((m) => m.userData.usdvGraph).length;
    v.setGrid(false);
    v.setShadows(false);
    v.renderer.setPixelRatio(1);
    v.renderer.toneMapping = THREE.NoToneMapping;
    v.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    v.nav.flyTo(new THREE.Vector3(...cp), new THREE.Vector3(...tg), 0);
    v.nav.update(0.1);
    v.camera.updateMatrixWorld();
    v.lod.update(v.camera, Infinity);
    v.renderer.render(v.scene, v.camera);
    const gl = v.renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const cw = v.canvas.clientWidth, ch = v.canvas.clientHeight;
    const samples = [];
    for (let gy = 0; gy < 60; gy++) for (let gx = 0; gx < 80; gx++) {
      const x = (gx + 0.5) * cw / 80, y = (gy + 0.5) * ch / 60;
      const p = v.pick(x, y);
      if (!p) continue;
      const i = ((h - 1 - Math.floor(y * h / ch)) * w + Math.floor(x * w / cw)) * 4;
      samples.push({ c: [px[i] / 255, px[i + 1] / 255, px[i + 2] / 255], p: [p.x, p.y, p.z] });
    }
    return { samples, failed, graphs, total: own.length };
  }, [camPos.split(',').map(Number), target.split(',').map(Number)]);
  fs.writeFileSync(outFile, JSON.stringify(res));
  await browser.close();
})();
""".replace("ARGS", BROWSER_ARGS)


def node(script: Path, *args: str) -> str:
    env = dict(os.environ)
    try:
        npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, check=True).stdout.strip()
        env["NODE_PATH"] = os.pathsep.join(p for p in (env.get("NODE_PATH"), npm_root) if p)
    except (OSError, subprocess.CalledProcessError):
        pass
    r = subprocess.run(["node", str(script), *args], capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        raise SystemExit(f"node {script.name} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def compile_all_nodes(work: Path) -> list:
    lib = mtlx_glsl.Library.get()
    shader_types = ("surfaceshader", "BSDF", "EDF", "VDF", "displacementshader", "volumeshader",
                    "lightshader", "material", "filename")
    cases = []
    for nd in lib.doc.getNodeDefs():
        outs = lib.outputs(nd)
        if not outs or any(o.getType() in shader_types for o in outs):
            continue
        gen = mtlx_glsl.Generator(lib, texture_cb=lambda info: {"texture": 0},
                                  primvar_cb=lambda name, mtype: "_usdpv0")
        try:
            res = [gen.output_expr(mtlx_glsl.Node(nd, {}), o.getName()) for o in outs]
        except mtlx_glsl.GenError:
            continue  # reported by tests/test_shaders.py
        sink = []
        for e, t in res:
            g = mtlx_glsl.GLSL_TYPE.get(t)
            if g in ("vec2", "vec3", "vec4"):
                sink.append(f"acc += vec4({e}{', 0.0' * (4 - int(g[-1]))});")
            elif g == "float":
                sink.append(f"acc += vec4({e});")
            elif g == "int":
                sink.append(f"acc += vec4(float({e}));")
            elif g == "bool":
                sink.append(f"acc += vec4({e} ? 1.0 : 0.0);")
            elif g == "mat3":
                sink.append(f"acc += vec4(({e})[0], 0.0);")
            elif g == "mat4":
                sink.append(f"acc += ({e})[0];")
        cases.append({"name": nd.getName(), "chunks": [c[1] for c in gen.chunks],
                      "main": "\n".join(gen.lines + sink)})
    (work / "cases.json").write_text(json.dumps(cases))
    (work / "compile.cjs").write_text(COMPILE_JS)
    failed = json.loads(node(work / "compile.cjs", str(work / "cases.json")).strip().splitlines()[-1])
    print(f"library nodes: {len(cases) - len(failed)}/{len(cases)} compile as WebGL2 GLSL")
    for name, log in failed[:20]:
        print(f"  {name}: {log.strip().splitlines()[0] if log.strip() else ''}")
    return failed


def view(work: Path, base: str, scene: Path, cam: str, target: str) -> dict:
    out = work / f"{scene.stem}.json"
    node(work / "view.cjs", base, str(scene), str(out), cam, target)
    return json.loads(out.read_text())


def to_usd(p, s=0.01):  # three.js world -> USD world for the Z-up, centimetre test scenes
    x, y, z = p
    return np.array([x / s, -z / s, y / s])


def check_spaces(res) -> list[str]:
    quads = {(0, 0): (0, 0, 1), (0, 1): (1, 0, 0), (1, 1): (0, 1, 0), (1, 0): (1, 1, 1)}
    udim = {(0, 0): (1, 0, 0), (1, 0): (0, 1, 0), (0, 1): (0, 0, 1), (1, 1): (0, 0, 0)}
    errs: dict[str, list] = {}
    for smp in res["samples"]:
        P, c = to_usd(smp["p"]), np.array(smp["c"])
        x, y, z = P
        if abs(z - 25) > 1:
            continue
        # quads span 80 units from x, y multiples of 10 / 20: skip antialiased edge pixels
        if min(abs(x - round(x / 10) * 10), abs(y - round(y / 10) * 10)) < 1.0:
            continue
        wrap = False
        if 130 <= x <= 210 and 20 <= y <= 100:
            exp, k, wrap = np.mod(P * 0.01, 1.0), "world position", True
        elif 0 <= x <= 80 and 30 <= y <= 110:
            exp, k, wrap = np.mod((P - np.array([-20, 20, 5])) / 2 * 0.01, 1.0), "object position", True
        elif 0 <= x <= 80 and 120 <= y <= 200:
            exp, k = np.array([x / 80, (y - 120) / 80, 0]), "texcoord"
        elif 100 <= x <= 180 and 120 <= y <= 200:
            u, v = (x - 100) / 80, (y - 120) / 80
            if min(abs(u - .5), abs(v - .5), u, v, 1 - u, 1 - v) < 0.03:
                continue
            exp, k = np.array(quads[(int(u > .5), int(v > .5))], float), "image orientation"
        elif 200 <= x <= 280 and 120 <= y <= 200:
            u, v = (x - 200) / 40, (y - 120) / 40
            if min(abs(u - 1), abs(v - 1), u, v, 2 - u, 2 - v) < 0.05:
                continue
            exp, k = np.array(udim[(int(u), int(v))], float), "UDIM"
        elif 300 <= x <= 380 and 120 <= y <= 200:
            u, v = (x - 300) / 80, (y - 120) / 80
            if min(abs(u - .5), abs(v - .5), u, v, 1 - u, 1 - v) < 0.03:
                continue
            red = quads[(int(u > .5), int(v > .5))][0]
            exp, k = np.array([red, red, red], float), "1.38 swizzle"
        else:
            continue
        e = np.abs(c - exp)
        if wrap:
            e = np.minimum(e, 1 - e)
        errs.setdefault(k, []).append(float(e.max()))
    problems = []
    for k in ("world position", "object position", "texcoord", "image orientation", "UDIM", "1.38 swizzle"):
        v = np.array(errs.get(k, []))
        ok = len(v) > 50 and np.percentile(v, 95) < 0.01
        print(f"  {k:18s} {'ok ' if ok else 'BAD'} n={len(v)} p95 error={np.percentile(v, 95) if len(v) else -1:.4f}")
        if not ok:
            problems.append(k)
    return problems


def check_triplanar(res, scene: Path, image: Path) -> list[str]:
    from PIL import Image
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene))
    M = np.array(UsdGeom.Xformable(stage.GetPrimAtPath("/World/Rock/Shape"))
                 .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    Minv = np.linalg.inv(M)
    img = np.asarray(Image.open(image).convert("RGB"), dtype=np.float64) / 255.0
    img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    H, W = img.shape[:2]

    def sample(u, v):  # mx_image: periodic, v up
        u, v = u % 1.0, v % 1.0
        return img[min(int((1 - v) * H), H - 1), min(int(u * W), W - 1)], min(abs(u - .5), abs(v - .5), u, v, 1 - u, 1 - v)

    errs = []
    for smp in res["samples"]:
        P = to_usd(smp["p"])
        local = (np.append(P, 1) @ Minv)[:3]
        n = (local / np.linalg.norm(local)) @ np.linalg.inv(M[:3, :3]).T
        n /= np.linalg.norm(n)
        pos = P / 40.0
        w = np.abs(n) / np.abs(n).sum()
        w = w ** (1 / 0.3)
        w /= w.sum()
        if w.max() < 0.97:
            continue  # blend zone: the mesh's interpolated normals differ from the analytic sphere
        (cx, ex), (cy, ey), (cz, ez) = sample(pos[1], pos[2]), sample(pos[0], pos[2]), sample(pos[0], pos[1])
        if min(e for e, wi in ((ex, w[0]), (ey, w[1]), (ez, w[2])) if wi > 0.02) < 0.02:
            continue  # texel edge
        errs.append(np.abs(np.array(smp["c"]) - (cx * w[0] + cy * w[1] + cz * w[2])).max())
    v = np.array(errs)
    ok = len(v) > 50 and np.percentile(v, 95) < 0.02
    print(f"  {'triplanar (world)':18s} {'ok ' if ok else 'BAD'} n={len(v)} p95 error={np.percentile(v, 95):.4f}")
    return [] if ok else ["triplanar"]


def main() -> int:
    if not shutil.which("node"):
        print("Node.js is needed (with Playwright) to run the browser checks")
        return 2
    work = Path(tempfile.mkdtemp(prefix="usdv-shaders-"))
    scenes = work / "scenes"
    make_shader_scenes.main(str(scenes))
    (work / "view.cjs").write_text(VIEW_JS)
    problems = []
    if compile_all_nodes(work):
        problems.append("library nodes")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "USDV_LOCAL_PATHS": "1", "USDV_DATA_DIR": str(work / "data")}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"{base}/api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        res = view(work, base, scenes / "nodes.usda", "3,3,8", "2.5,2.5,0")
        print(f"node scene: {res['graphs']}/{res['total']} materials use their compiled network")
        if res["failed"] or res["graphs"] != res["total"]:
            print(f"  fell back: {res['failed']}")
            problems.append("node scene")
        print("spaces:")
        problems += check_spaces(view(work, base, scenes / "spaces.usda", "1.9,5,-0.6", "1.9,0,-1.1"))
        problems += check_triplanar(view(work, base, scenes / "triplanar.usda", "0.75,0.75,0.8", "0.15,0.3,0.07"),
                                    scenes / "triplanar.usda", scenes / "quads.png")
    finally:
        server.terminate()
    print("FAILED: " + ", ".join(problems) if problems else "all shader checks passed")
    if not problems:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
