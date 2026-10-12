<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->

# GenIA for Apple silicon

A macOS port of [GenIA](https://github.com/facebookresearch/GenIA), with local
Metal operators and a browser UI for image-to-3D reconstruction.

Upload an image, queue a reconstruction, and inspect the optimized Gaussian splats
or exported mesh in your browser. Model inference and optimization run locally on
your Mac through PyTorch MPS.

## Install

You need an Apple silicon Mac with macOS 14 or newer and [uv](https://docs.astral.sh/uv/). The installer
creates a Python 3.12 environment and installs the pinned dependencies and local
model implementations. No CUDA toolkit or recursive submodule checkout is needed.

```bash
git clone https://github.com/unixpickle/GenIA.git
cd GenIA
bash install_macos.sh
uv pip install --python .venv/bin/python -e '.[webui]'
```

The port has been tested on an M2 Max with 32 GB of unified memory and macOS 26.
Reconstruction is memory intensive and can take tens of minutes per image;
requirements vary with the scene and settings.

## Download models

Request access to [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects)
and accept its licence, then sign in with a Hugging Face account that has access:

```bash
.venv/bin/hf auth login
.venv/bin/python -m genia.core.download_weights --cache-root /Volumes/MLData4/GenIA
```

Replace `/Volumes/MLData4/GenIA` with your preferred model-cache directory and use
the same path when starting the server. Downloads reuse existing cached files.
The command fetches the models for both static and dynamic reconstruction.

## Start the WebUI

```bash
.venv/bin/python -m genia.core.webui --cache-root /Volumes/MLData4/GenIA
```

Open **http://127.0.0.1:7860**.

- **Upload and queue:** add an image, optionally supply a foreground mask, and
  choose a seed. Transparent images can supply their own mask; otherwise the
  worker removes the background automatically.
- **Follow progress:** see the current stage, optimization iterations, and live
  logs. Cancel a queued or running task at any time.
- **Browse results:** previous generations retain their previews, logs, and
  downloadable Gaussian PLY and GLB files. Delete results from their history card.
- **Inspect in 3D:** click **Load 3D** to view the optimized splats. Switch to the
  mesh for comparison. Models download only on demand, and closing the viewer
  releases its resources.

The viewer uses WebGL 2 and works over HTTP, including from a phone on your LAN.
It needs no Node build or CDN. The WebUI currently supports single-image static
reconstruction; multiview and video workflows remain available through the CLI.

To listen on a LAN address:

```bash
.venv/bin/python -m genia.core.webui \
  --host 10.9.0.4 --port 7860 \
  --cache-root /Volumes/MLData4/GenIA
```

Use an address assigned to your Mac, then open **http://10.9.0.4:7860** from another
device. The server has no authentication, so keep it on a trusted network.

Queue state, uploads, and generated files persist in `results/webui/`. Use
`--data-dir` to choose another location. One background process handles one
reconstruction at a time; cancelling an active task stops that process, and the
next task starts in a fresh worker.

See the [WebUI guide](docs/webui.md) for masks, viewer limits, storage, and server
options.

## About the port

The macOS implementation includes local model sources and Metal replacements for
geometry, sparse operations, Gaussian rendering, and mesh extraction. MPS
fine-tuning uses FP32 by default. Single-image, multiview, and dynamic pipelines
have completed on Apple silicon; detailed validation and known limits are in the
[macOS port guide](docs/macos-port.md).

- [Setup, model caches, CLI demos, and validation](docs/macos-port.md)
- [Pipeline configuration](docs/configuration.md)
- [Local model sources and provenance](vendor/README.md)

Run the WebUI checks or the complete operator and pipeline test suite:

```bash
.venv/bin/python -m pytest tests/test_webui*.py -q
.venv/bin/python -m pytest tests -q
```

## License

[CC BY-NC 4.0](LICENSE). Third-party code and model weights retain their respective
licences; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
