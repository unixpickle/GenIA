"""HTTP boundary: uploads, queue state, bounded logs, and allowlisted artifacts."""
import json
import secrets
import shutil
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from .artifacts import log_tail, safe_file, splat_assets
from .queue import JobQueue
from .progress import read_stage
from .store import Store
from .uploads import prepare_upload


def create_app(data_dir, *, environment=None, command=None, hosts=None):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config.update(MAX_CONTENT_LENGTH=32 * 1024 * 1024, MAX_FORM_PARTS=8,
                      MAX_FORM_MEMORY_SIZE=512_000,
                      TRUSTED_HOSTS=hosts or ["localhost", "127.0.0.1", "[::1]"])
    store = Store(Path(data_dir))
    queue = JobQueue(store, environment=environment, command=command)
    app.extensions.update(genia_store=store, genia_queue=queue)

    @app.before_request
    def protect_writes():
        # Browsers cannot send this header cross-origin without a CORS preflight.
        # No CORS headers are emitted. Also block cross-origin reads by fetch metadata.
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            abort(403, "Open the UI directly from this server.")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("X-GenIA-Request") != "1":
                abort(403, "Missing X-GenIA-Request header.")

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; connect-src 'self'; object-src 'none'; "
            "worker-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        if request.path.startswith("/api/") and "/assets/" not in request.path:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    def get_job(job_id):
        job = store.get(job_id)
        if job is None:
            abort(404, "Generation not found.")
        return job

    def published_assets(job):
        assets = dict(job["assets"])
        if job["status"] == "succeeded":
            assets.update(splat_assets(store.jobs_dir / job["id"]))
        return assets

    def public(job):
        job = dict(job)
        job["progress"] = read_stage(store.jobs_dir / job["id"]) if job["status"] == "running" else None
        job["input_url"] = f"/api/jobs/{job['id']}/assets/input"
        job["assets"] = [{"key": key, **{k: v for k, v in asset.items() if k != "path"},
                          "url": f"/api/jobs/{job['id']}/assets/{key}"}
                         for key, asset in published_assets(job).items()]
        return job

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/jobs")
    def jobs():
        try:
            before = int(request.args["before"]) if "before" in request.args else None
            since = int(request.args["since"]) if "since" in request.args else None
            limit = int(request.args.get("limit", 12))
            if not 1 <= limit <= 48 or any(value is not None and not 1 <= value < 2**63
                                         for value in (before, since)):
                raise ValueError
        except ValueError:
            abort(400, "Invalid pagination parameters.")
        data = store.snapshot(before, limit, since)
        data["queue"] = [public(j) for j in data["queue"]]
        data["history"] = [public(j) for j in data["history"]]
        data["worker"] = "stopping" if queue.stopping else "running" if queue.current else "idle"
        return jsonify(data)

    @app.post("/api/jobs")
    def upload():
        image = request.files.get("image")
        if image is None or not image.filename:
            abort(400, "Choose an image to upload.")
        try:
            seed = int(request.form.get("seed", "42"))
            if not 0 <= seed <= 2**32 - 1:
                raise ValueError
        except ValueError:
            abort(400, "Seed must be an integer from 0 to 4294967295.")
        title = Path(image.filename).stem.strip()[:120] or "Generation"
        job_id = secrets.token_hex(16)
        directory = store.jobs_dir / job_id
        mask = request.files.get("mask")
        try:
            options = prepare_upload(directory, image.stream, mask.stream if mask and mask.filename else None)
            (directory / "request.json").write_text(json.dumps({**options, "seed": seed}))
            job = store.enqueue(job_id, title, seed)
        except ValueError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            abort(400, str(exc))
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        queue.wake()
        return jsonify(public(job)), 201

    @app.get("/api/jobs/<job_id>")
    def detail(job_id):
        return jsonify(public(get_job(job_id)))

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel(job_id):
        job = queue.cancel(job_id)
        if job is None:
            abort(404, "Generation not found.")
        return jsonify(public(job))

    @app.delete("/api/jobs/<job_id>")
    def delete(job_id):
        try:
            deleted = store.delete(job_id)
        except ValueError as exc:
            abort(409, str(exc))
        except OSError:
            app.logger.exception("Could not delete generation %s", job_id)
            abort(500, "Could not remove all generation files. The history entry was kept; please retry.")
        if not deleted:
            abort(404, "Generation not found.")
        return jsonify(deleted=job_id)

    @app.get("/api/jobs/<job_id>/log")
    def log(job_id):
        get_job(job_id)
        return jsonify(text=log_tail(store.jobs_dir / job_id))

    @app.get("/api/jobs/<job_id>/assets/<key>")
    def asset(job_id, key):
        job = get_job(job_id)
        entry = {"path": "input.jpg", "name": "input.jpg"} if key == "input" else published_assets(job).get(key)
        if entry is None:
            abort(404)
        try:
            path = safe_file(store.jobs_dir / job_id, entry["path"])
        except FileNotFoundError:
            abort(404)
        return send_file(path, conditional=True, max_age=86400,
                         mimetype={".glb": "model/gltf-binary", ".ply": "application/octet-stream"}.get(path.suffix, "image/jpeg"),
                         as_attachment=request.args.get("download") == "1", download_name=entry["name"])

    return app
