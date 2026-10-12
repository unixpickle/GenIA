# Local generation studio

Install the main GenIA environment and download its model weights first (see
[macOS setup](macos-port.md)). The UI adds Flask; it needs no Node build, CDN, or
browser downloads beyond files served by GenIA itself.

```bash
uv pip install --python .venv/bin/python -e '.[webui]'
.venv/bin/python -m genia.core.webui --cache-root /Volumes/MLData4/GenIA
```

Open **http://127.0.0.1:7860**. On another platform, use the Python executable in
your GenIA environment. Omit `--cache-root` to keep the existing Hugging Face and
torch cache environment. The flag uses the same directory layout as
`genia.core.download_weights --cache-root` and passes cache paths to workers before
any model library imports. It does not move or duplicate existing weights.

## Using the UI

1. Upload or drop a PNG, JPEG, or WebP image. Generations use the uploaded filename;
   you can change the seed under **Generation options**.
2. Click **Add to queue**. You can upload more images while a generation runs.
   The active job shows its current pipeline stage. During shape/appearance
   optimization, the bar shows actual completed iterations and a percentage for
   the current optimization pass, read from the worker log. Other stages show an
   activity indicator; this is not an overall generation percentage or ETA.
3. Use **Cancel** to remove a waiting job or stop the active generation. **View log**
   shows the last 32 KiB of worker output; active logs refresh while open.
   Refreshes preserve the open log and your scroll position. New output is followed
   automatically only while you are at the bottom of the log.
4. Finished jobs appear in the collection with image previews, elapsed time, seed,
   and downloads. Failed, interrupted, and cancelled jobs remain in history.
5. Click **Load 3D** to display the optimized **splats** inline. The selector also
   offers the exported **mesh** for comparison and GLB download. Existing completed
   generations automatically gain splat viewing when their final PLY is present;
   no rerun or database migration is needed. Drag to rotate, scroll
   to zoom (or pinch on a touchscreen), and double-click to reset. Keyboard arrows rotate, `+`/`-` zoom, and `R`
   resets. **Close 3D**, or opening another model, releases its WebGL resources.
6. Use **Delete** on a history card to permanently remove the generation, its input,
   logs, previews, and model files. Active jobs must be cancelled and finish stopping
   first. Failed, interrupted, and cancelled jobs can also be deleted.

Only thumbnail images are requested while browsing history, and images use browser
lazy loading. The viewer JavaScript and selected PLY or GLB are requested on the explicit Load 3D
action. There is no model prefetch, CDN, or telemetry. The gallery loads 12 entries
at a time and preserves open viewers when refreshing queue status. The viewer is
for GenIA's Gaussian PLY and uncompressed, vertex-colored triangle GLB exports;
textured or compressed third-party meshes need an external viewer. Your browser
needs **WebGL 2**, which works over the existing HTTP LAN URL without HTTPS or WebGPU.

The splat renderer displays the optimized Gaussian appearance directly, without
mesh lighting. The saved first camera pose sets its initial orientation when
available, so objects open facing the input view. It supports the exporter's compressed PLY and binary little-endian
3DGS PLY, including anisotropic scale, rotation, opacity, and DC color. Higher-order
spherical harmonics are not evaluated; files containing them show a **Base colors**
label. The current static pipeline's compressed exports use DC color. A browser
worker decodes and sorts splats, keeping that work off the UI thread. Rotation
requests are coalesced to one sort in flight, with a final update after dragging;
rapid motion may briefly use the previous depth ordering. Closing the viewer aborts
the download, terminates the worker, and frees GPU resources. Rendering is demand
driven and resolution is capped on dense mobile screens. Files over 256 MiB or
2 million splats are rejected with a message; downloads remain available.

The UI currently runs **single-image static reconstruction**, using the existing
`mono_static` / `mono_ours` recipe with MoGe depth, per-frame appearance, and the
`low_rgb` first pose refinement. It exports meshes and still previews and skips
unused orbit/world videos. Video and multiview inputs remain available through the
existing demo commands. The collection records jobs submitted through this UI;
it does not scan or import unrelated previous CLI runs.

### Foreground masks

An optional mask must match the image dimensions: white foreground, black background.
Transparent input images use their alpha channel automatically. Otherwise the worker
runs the existing local BRIA RMBG 1.4 implementation before reconstruction. Its weights
are downloaded to the configured model cache on first use if absent (or fetched ahead
of time by the full `genia.core.download_weights` command). The `--no-actionmesh`
download option does not currently include RMBG; use an uploaded mask or transparency
if only the static-model cache is installed and the machine is offline.

Each upload is one object. The optional mask is binary, not a multi-object label map.
Image orientation is applied before checking mask alignment. Images are limited to
24 megapixels and resized to at most 2048 pixels on the longest edge; requests are
limited to 32 MiB including the optional mask. The queue holds up to 100 waiting jobs.

## Worker and persistence

The HTTP server never imports torch or loads model weights. A supervisor thread
claims jobs from a SQLite FIFO queue and starts **one fresh Python subprocess per
generation**. Masking, sampling, optimization, and export all happen there. This
releases model/GPU memory between jobs, at the cost of loading models again for each
generation. Expect the same substantial reconstruction time as the CLI pipeline;
this is not a fast preview mode.

Cancelling an active job marks it as cancelling, kills its entire process group,
waits for the process to exit, then marks it cancelled. The next queued job starts
in a new worker. Partial output stays on disk for diagnosis but is not published as
a finished model. Cancellation of a completed job is an idempotent no-op.

Closing the browser leaves the server and queue running. Stopping the server kills
the current worker and marks that job interrupted, while waiting jobs persist for
the next launch. A pipe watchdog kills the worker if its parent server crashes.
On restart, stale running jobs become interrupted, not silently successful or
automatically retried. Submit the image again to retry. An exclusive file lock
prevents two servers from consuming the same queue.

By default, data lives in `results/webui/` (ignored by git):

```text
jobs.sqlite3          persistent queue, history, and artifact metadata
server.lock          exclusive server lock
jobs/<id>/
  request.json       normalized upload options
  input/scene/       pipeline image and mask
  input.jpg          input thumbnail
  worker.log         worker stdout/stderr
  output/            ordinary GenIA pipeline outputs
  preview-*.jpg      small gallery previews
```

There is no automatic history cleanup; use **Delete** to remove individual generations.
Keep the data directory to preserve results;
back it up or remove it while the server is stopped if you want to reset the studio.

## Server options

```bash
.venv/bin/python -m genia.core.webui --help
.venv/bin/python -m genia.core.webui --port 7861 --data-dir results/another-studio --device mps
```

`--device` accepts `mps`, `cuda`, or `cpu`; the default uses GenIA's existing automatic
selection and honors `GENIA_DEVICE`. The server defaults to loopback. You can bind
an explicit LAN address with `--host`, but this is a local tool without authentication;
use a trusted network. Writes require a custom request header and cross-site browser
requests are rejected. Uploaded filenames never determine storage paths, and artifact
routes expose only published previews, meshes, splats, and input thumbnails, with byte-range
support for downloads.

## Code layout and checks

`core/webui/app.py` owns HTTP, `store.py` owns SQLite state transitions, `queue.py`
owns subprocess lifetime, `worker.py` adapts the existing pipeline, `uploads.py`
normalizes inputs, and `artifacts.py` publishes safe result files. `static/app.js`
owns UI state. `static/viewer.js` implements the independent GLB viewer;
`splat-viewer.js` owns WebGL rendering and orbit controls, `splat-worker.js` owns
background work, and `splat-data.js` owns PLY decoding and depth sorting.
Importing the server package is lightweight. Static assets are included in wheels.

```bash
.venv/bin/python -m pytest tests/test_webui*.py -q
```

These checks cover HTTP uploads and limits, mask handling, persistence, FIFO order,
cancellation of real subprocesses, worker replacement, failures, shutdown/recovery,
the parent-death watchdog, asset allowlisting/range requests, and the actual GLB
parser, splat decoding against the Python exporter, depth sorting, and viewer
lifecycle. JavaScript checks use Node or the macOS JavaScriptCore executable when
available. An optional GLES test uses ANGLE/SwiftShader to compare the actual splat
shaders against independent CPU projection and compositing; it skips when those
libraries are unavailable. This does not replace checking touch interaction and
performance on the target phone.
