"""Publish a small, explicit set of generation artifacts; never expose a run directory."""
import json
import math
from pathlib import Path

from PIL import Image


def safe_file(directory, relative):
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


def splat_orientations(directory):
    """Small initial-view hints; object transforms use PyTorch3D row vectors."""
    try:
        path = safe_file(directory, "output/final/poses.json")
        if path.stat().st_size > 1024 * 1024:
            return {}
        poses = json.loads(path.read_text())
        rotations = {}
        for obj in poses.get("objects", []):
            transforms = obj.get("transforms", [])
            if not transforms:
                continue
            q = transforms[0].get("rotation", [])
            if len(q) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in q):
                continue
            length = math.hypot(*q)
            if not length:
                continue
            w, x, y, z = (v / length for v in q)
            r = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                 [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                 [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
            # Object -> first P3D camera -> OpenGL camera (X and Z reversed).
            rotations[Path(obj.get("ply", "")).name] = [
                r[col][row] * (-1 if row != 1 else 1)
                for row in range(3) for col in range(3)]
        cameras = poses.get("cameras", [])
        if cameras:
            c2w = cameras[0].get("c2w", [])
            if len(c2w) == 4 and all(len(row) == 4 for row in c2w):
                # World exports use OpenCV/R3 axes: X right, Y down, Z forward.
                r = [c2w[col][row] * (1 if row == 0 else -1)
                     for row in range(3) for col in range(3)]
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in r):
                    rotations["world.compressed.ply"] = rotations["world.ply"] = r
        return rotations
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def splat_assets(directory):
    """Discover final splats, including exports from jobs saved before the viewer existed.

    Only scan this fixed export directory; do not read/decompress model payloads.
    Filename-based keys stay stable if another artifact disappears.
    """
    directory = Path(directory)
    paths = sorted((directory / "output/final/gaussians").glob("*.ply"),
                   key=lambda p: (p.name.startswith("world."), p.name))
    assets = {}
    orientations = splat_orientations(directory) if paths else {}
    for path in paths:
        if path.name.startswith("background."):
            continue
        relative = str(path.relative_to(directory))
        try:
            path = safe_file(directory, relative)
            size = path.stat().st_size
        except (FileNotFoundError, OSError):
            continue
        assets[f"splat-{path.name}"] = {"path": relative, "name": path.name,
                                       "bytes": size, "kind": "splat"}
        if path.name in orientations:
            assets[f"splat-{path.name}"]["view_rotation"] = orientations[path.name]
    return assets


def collect(directory):
    directory = Path(directory)
    final = directory / "output" / "final"
    meshes = sorted((final / "meshes").glob("*.glb"))
    # Prefer the individual object, which is centered for inspection, over world space.
    meshes.sort(key=lambda p: (p.stem == "world", p.name))
    if not meshes:
        raise RuntimeError("Generation finished without exporting a GLB mesh. See the worker log.")
    assets = splat_assets(directory)
    for i, path in enumerate(meshes):
        relative = str(path.relative_to(directory))
        safe_file(directory, relative)
        assets[f"model-{i}"] = {"path": relative, "name": path.name,
                                 "bytes": path.stat().st_size, "kind": "model"}
    previews = sorted((final / "renders_train").glob("*.png"))[:1]
    previews += sorted((final / "decoded_renders").glob("*.png"))[:4]
    previews += sorted((final / "renders_synth_nvs").glob("*.png"))[:4]
    for i, path in enumerate(previews[:6]):
        safe_file(directory, str(path.relative_to(directory)))
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((768, 768), Image.Resampling.LANCZOS)
            destination = directory / f"preview-{i}.jpg"
            im.save(destination, quality=85)
        assets[f"preview-{i}"] = {"path": destination.name, "name": path.stem,
                                   "bytes": destination.stat().st_size, "kind": "preview"}
    return assets


def log_tail(directory, limit=32_768):
    try:
        with (Path(directory) / "worker.log").open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
