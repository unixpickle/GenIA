"""Subprocess entry point: mask preparation, the existing pipeline, and export."""
import argparse
import atexit
import json
import os
import select
import signal
import sys
import threading
from pathlib import Path

from .progress import write_stage


def watch_parent():
    """The supervisor holds stdin open. Its death closes the pipe, including on SIGKILL.

    Only used in a dedicated session started by JobQueue. This prevents an expensive
    orphaned model process from surviving a crashed server and competing after restart.
    """
    if os.getpgrp() != os.getpid():
        raise RuntimeError("The model worker must run in its own process group.")
    descriptor = sys.stdin.fileno()
    stopped = threading.Event()

    def watch():
        # Never hold a BufferedReader lock in a daemon thread: Python acquires
        # those locks during shutdown, which used to abort successful workers.
        while not stopped.is_set():
            readable, _, _ = select.select([descriptor], [], [], 0.2)
            if readable and not os.read(descriptor, 4096):
                if not stopped.is_set():
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                return

    thread = threading.Thread(target=watch, daemon=True, name="parent-watch")
    thread.start()

    def stop():
        stopped.set()
        thread.join()

    # Also covers callers that do not explicitly stop the watchdog.
    atexit.register(stop)
    return stop


def pipeline_overrides(directory, seed):
    # JSON quoting is also valid Hydra string quoting, including paths with spaces.
    return [
        "+experiment=mono_static", "dataset=image", "pipeline=mono_ours", "processing=moge",
        "processing.resume_from_cache=false", f"processing.seed={seed}",
        "output.save_cache=false", "output.suppress_intermediate_renders=true",
        "output.suppress_intermediate_metrics=true", "shape_init.shape_cache_dir=null",
        "appearance_init=perframe", "global_pose_refine@global_pose_refine_1=low_rgb",
        "dataset.path=" + json.dumps(str(directory / "input")), "dataset.scene_name=scene",
        "output.output_dir=" + json.dumps(str(directory / "output")),
        # Still images and meshes are the web UI outputs. Avoid rendering unused videos.
        "output.save_viz_orbit=false", "output.save_viz_world_space=false",
        "output.save_viz_track_overlays=false",
    ]


def prepare_mask(directory):
    scene = directory / "input" / "scene"
    mask_path = scene / "0000_mask.png"
    if mask_path.exists():
        return
    write_stage(directory, "masking")
    print("Preparing foreground mask with RMBG…", flush=True)
    import numpy as np
    import torch
    from PIL import Image
    from actionmesh.preprocessing.background_removal import BackgroundRemover
    from genia.core.device import empty_cache, get_device

    with Image.open(scene / "0000.png") as image:
        rgb = np.array(image.convert("RGB"))
    model = BackgroundRemover().eval().to(get_device())
    with torch.inference_mode():
        mask = (model(rgb)[:, :, 3] >= 128).astype(np.uint8)
    if not mask.any():
        raise ValueError("Automatic masking found no foreground. Upload a white-on-black mask.")
    Image.fromarray(mask).save(mask_path)
    del model
    empty_cache()


def generate(directory):
    write_stage(directory, "starting")
    options = json.loads((directory / "request.json").read_text())
    prepare_mask(directory)
    write_stage(directory, "loading_models")
    print("Starting single-image reconstruction…", flush=True)
    from hydra import compose, initialize_config_module
    from genia.core.run import run

    with initialize_config_module(config_module="genia.core.configs", version_base="1.2"):
        config = compose(config_name="genia", overrides=pipeline_overrides(directory, options["seed"]))
    run(config, on_stage=lambda stage: write_stage(directory, stage))
    print("Generation complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    stop_watch = watch_parent()
    try:
        generate(args.directory.resolve())
    finally:
        stop_watch()


if __name__ == "__main__":
    main()
