"""Small atomic stage snapshots shared by the worker and HTTP server."""
import json
import logging
import time

LABELS = {
    "starting": "Starting generation",
    "masking": "Finding the foreground object",
    "loading_models": "Loading reconstruction models",
    "preprocessing": "Estimating depth and preparing the image",
    "gt_shapes_inversion": "Preparing reference geometry",
    "shape_init": "Generating the 3D shape",
    "appearance_init": "Generating appearance",
    "pose_init": "Estimating object pose",
    "global_pose_refine_1": "Aligning the object to the image",
    "appearance_init_2": "Refining appearance",
    "finetune": "Optimizing shape and appearance",
    "global_pose_refine_2": "Refining final alignment",
    "exporting": "Exporting the model and renders",
    "previews": "Preparing gallery previews",
}


def write_stage(directory, stage):
    snapshot = {"stage": stage, "label": LABELS[stage], "started": time.time()}
    try:
        temporary = directory / "progress.tmp"
        temporary.write_text(json.dumps(snapshot))
        temporary.replace(directory / "progress.json")
    except OSError:
        # Reporting must never cause an otherwise valid generation to fail.
        logging.getLogger(__name__).warning("Could not save generation progress", exc_info=True)


def read_stage(directory):
    try:
        value = json.loads((directory / "progress.json").read_text())
        stage = value["stage"]
        return {"stage": stage, "label": LABELS[stage]}
    except (OSError, ValueError, KeyError, TypeError):
        return {"stage": "starting", "label": LABELS["starting"]}
