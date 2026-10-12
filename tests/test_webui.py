"""HTTP, durable queue, and actual subprocess lifecycle tests; no model weights needed."""
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("flask")
from genia.core.webui.app import create_app
from genia.core.webui.artifacts import collect, log_tail, safe_file
from genia.core.webui.queue import JobQueue
from genia.core.webui.store import Store
from genia.core.webui.worker import pipeline_overrides

HEADERS = {"X-GenIA-Request": "1"}


def png(mode="RGB", size=(32, 24), color="white"):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path)


def upload(client, **kwargs):
    return client.post("/api/jobs", headers=HEADERS,
                       data={"image": (png(), "object.png"), **kwargs})


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError("Timed out waiting for worker state")


def enqueue(store, title="object"):
    import secrets
    job_id = secrets.token_hex(16)
    directory = store.jobs_dir / job_id
    directory.mkdir()
    return store.enqueue(job_id, title, 42)


def test_upload_persistence_alpha_and_cancel(app):
    client = app.test_client()
    source = Image.new("RGBA", (40, 30), (255, 0, 0, 0))
    source.paste((255, 0, 0, 255), (8, 5, 30, 25))
    stream = io.BytesIO(); source.save(stream, "PNG"); stream.seek(0)
    response = upload(client, image=(stream, "../../danger.png"), title="<script>test</script>", seed="123")
    assert response.status_code == 201
    job = response.json
    assert job["title"] == "danger" and job["seed"] == 123
    store = app.extensions["genia_store"]
    directory = store.jobs_dir / job["id"]
    with Image.open(directory / "input/scene/0000_mask.png") as mask:
        assert set(mask.tobytes()) == {0, 1}
    assert json.loads((directory / "request.json").read_text())["mask"] == "transparency"
    assert client.get(job["input_url"]).mimetype == "image/jpeg"
    assert Store(store.root).get(job["id"])["status"] == "queued"
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=HEADERS).json["status"] == "cancelled"
    # Repeated cancellation is idempotent, and it never touches terminal states.
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=HEADERS).json["status"] == "cancelled"
    assert client.get("/api/jobs").json["history"][0]["id"] == job["id"]


@pytest.mark.parametrize("data", [
    {"image": (io.BytesIO(b"not an image"), "fake.png")},
    {"seed": "-1"}, {"seed": "nan"}, {"seed": str(2**32)},
    {"mask": (png(size=(10, 10)), "mask.png")},
    {"mask": (png(color="black"), "mask.png")},
    {"image": (png("RGBA", color=(0, 0, 0, 0)), "empty.png")},
])
def test_invalid_uploads_leave_no_jobs(app, data):
    assert upload(app.test_client(), **data).status_code == 400
    assert app.extensions["genia_store"].snapshot()["counts"] == {}
    assert not list(app.extensions["genia_store"].jobs_dir.iterdir())


def test_oversize_request_and_missing_image(app):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    client = app.test_client()
    assert client.post("/api/jobs", headers=HEADERS, data={}).status_code == 400
    assert upload(client, title="x" * 3000).status_code == 413


def test_mask_thresholding_and_image_downscale(app):
    response = upload(app.test_client(), image=(png(size=(3000, 1000)), "a.png"),
                      mask=(png(size=(3000, 1000), color=(200, 200, 200)), "m.png"))
    directory = app.extensions["genia_store"].jobs_dir / response.json["id"]
    with Image.open(directory / "input/scene/0000.png") as image:
        assert image.size == (2048, 683)
    with Image.open(directory / "input/scene/0000_mask.png") as mask:
        assert set(mask.tobytes()) == {1}


def test_http_security_and_static_assets(app):
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/viewer.js").status_code == 200
    assert client.post("/api/jobs", data={}).status_code == 403
    assert client.get("/api/jobs", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/jobs", base_url="http://attacker.example").status_code == 400
    assert client.get("/api/jobs?limit=10000").status_code == 400
    assert client.get("/api/jobs?before=oops").status_code == 400
    assert client.get(f"/api/jobs?before={2**80}").status_code == 400
    assert client.get("/api/jobs/no-such-job").status_code == 404
    assert client.get("/api/jobs").headers["Cache-Control"] == "no-store"
    assert "script-src 'self'" in client.get("/").headers["Content-Security-Policy"]


def test_cursor_history_and_fifo_claim(app):
    store = app.extensions["genia_store"]
    jobs = [enqueue(store, str(i)) for i in range(5)]
    assert store.claim()["id"] == jobs[0]["id"]
    for job in jobs:
        store.cancel(job["id"])
    store.finish(jobs[0]["id"], "cancelled")
    client = app.test_client()
    first = client.get("/api/jobs?limit=2").json
    second = client.get(f"/api/jobs?limit=2&before={first['next']}").json
    assert [j["title"] for j in first["history"]] == ["4", "3"]
    assert [j["title"] for j in second["history"]] == ["2", "1"]


def test_artifact_allowlist_ranges_and_bounded_log(app, tmp_path):
    client = app.test_client()
    job = upload(client).json
    store = app.extensions["genia_store"]
    directory = store.jobs_dir / job["id"]
    final = directory / "output/final"
    (final / "meshes").mkdir(parents=True)
    (final / "meshes/001.glb").write_bytes(b"glTF" + b"0" * 100)
    (final / "renders_train").mkdir()
    Image.new("RGB", (1000, 1000), "red").save(final / "renders_train/000.png")
    store.claim(); store.finish(job["id"], "succeeded", assets=collect(directory))
    data = client.get(f"/api/jobs/{job['id']}").json
    assert all("path" not in a for a in data["assets"])
    model_url = next(a["url"] for a in data["assets"] if a["kind"] == "model")
    response = client.get(model_url, headers={"Range": "bytes=0-3"})
    assert response.status_code == 206 and response.data == b"glTF"
    assert "attachment" in client.get(model_url + "?download=1").headers["Content-Disposition"]
    assert client.get(f"/api/jobs/{job['id']}/assets/request.json").status_code == 404
    with Image.open(directory / "preview-0.jpg") as image:
        assert image.size == (768, 768)
    secret = tmp_path / "secret"; secret.write_text("hidden")
    (directory / "escape").symlink_to(secret)
    with pytest.raises(FileNotFoundError): safe_file(directory, "escape")
    with pytest.raises(FileNotFoundError): safe_file(directory, "../../secret")
    (directory / "worker.log").write_text("x" * 100_000)
    assert len(log_tail(directory)) == 32_768
    assert len(client.get(f"/api/jobs/{job['id']}/log").json["text"]) == 32_768


def test_recovery_and_exclusive_server_lock(tmp_path):
    store = Store(tmp_path)
    stale = enqueue(store)
    store.claim()
    queue = JobQueue(store)
    queue.start()
    other = JobQueue(Store(tmp_path))
    try:
        assert store.get(stale["id"])["status"] == "interrupted"
        with pytest.raises(RuntimeError, match="Another server"):
            other.start()
    finally:
        queue.stop()
    other.start(); other.stop()


def test_real_worker_cancel_then_next_success_and_failure(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text('''import sys,time,os
from pathlib import Path
from genia.core.webui.worker import watch_parent
watch_parent()
p=Path(sys.argv[1])
(p/'pid').write_text(str(os.getpid()))
if (p/'slow').exists():
    time.sleep(60)
if (p/'fail').exists():
    print('intentional model failure',flush=True)
    sys.exit(7)
(p/'output/final/meshes').mkdir(parents=True)
(p/'output/final/meshes/001.glb').write_bytes(b'glTF')
''')
    store = Store(tmp_path / "data")
    first, second, third = [enqueue(store) for _ in range(3)]
    (store.jobs_dir / first["id"] / "slow").touch()
    (store.jobs_dir / third["id"] / "fail").touch()
    queue = JobQueue(store, command=lambda directory: [sys.executable, str(script), str(directory)])
    queue.start()
    try:
        pid_file = store.jobs_dir / first["id"] / "pid"
        wait_for(pid_file.exists)
        pid = int(pid_file.read_text())
        assert store.get(second["id"])["status"] == "queued"
        queue.cancel(first["id"])
        wait_for(lambda: store.get(first["id"])["status"] == "cancelled")
        with pytest.raises(ProcessLookupError): os.kill(pid, 0)
        wait_for(lambda: store.get(second["id"])["status"] == "succeeded")
        wait_for(lambda: store.get(third["id"])["status"] == "failed")
        assert "code 7" in store.get(third["id"])["error"]
        assert "intentional model failure" in log_tail(store.jobs_dir / third["id"])
        assert store.get(second["id"])["assets"]["model-0"]["kind"] == "model"
        # Cancelling an already-completed job must not change its result.
        assert queue.cancel(second["id"])["status"] == "succeeded"
    finally:
        queue.stop()


def test_shutdown_interrupts_active_but_keeps_waiting_jobs(tmp_path):
    store = Store(tmp_path)
    first, second = enqueue(store), enqueue(store)
    queue = JobQueue(store, command=lambda _: [sys.executable, "-c", "import time; time.sleep(60)"])
    queue.start()
    wait_for(lambda: store.get(first["id"])["status"] == "running")
    queue.stop()
    assert store.get(first["id"])["status"] == "interrupted"
    assert store.get(second["id"])["status"] == "queued"


def test_parent_watchdog_exits_when_pipe_closes(tmp_path):
    # Same watchdog as the real model worker, without loading any model packages.
    code = "from genia.core.webui.worker import watch_parent; import time; watch_parent(); print('ready',flush=True); time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, start_new_session=True)
    try:
        assert child.stdout.readline() == b"ready\n"
        child.stdin.close()
        assert child.wait(timeout=5) == -signal.SIGKILL
    finally:
        if child.poll() is None: os.killpg(child.pid, signal.SIGKILL)
        child.wait(); child.stdout.close()


@pytest.mark.parametrize("explicit_stop", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_parent_watchdog_does_not_abort_interpreter_shutdown(explicit_stop, raises):
    # Keep the parent pipe OPEN until after worker exit. communicate() would close
    # it first and only exercise the parent-death path, missing this regression.
    code = """from genia.core.webui.worker import watch_parent
import threading,time
stop=watch_parent()
print('ready',flush=True)
time.sleep(.05)
"""
    if explicit_stop:
        code += "stop()\nassert not any(t.name=='parent-watch' for t in threading.enumerate())\n"
    if raises:
        code += "raise RuntimeError('intentional generation failure')\n"
    child = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        assert child.stdout.readline() == b"ready\n"
        assert child.wait(timeout=5) == (1 if raises else 0)
        error = child.stderr.read().decode()
        assert "_enter_buffered_busy" not in error
        assert "Fatal Python error" not in error
        if raises:
            assert "intentional generation failure" in error
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        child.stdin.close(); child.stdout.close(); child.stderr.close()


def test_cancellation_kills_worker_descendants(tmp_path):
    store = Store(tmp_path / "data")
    job = enqueue(store)
    heartbeat = store.jobs_dir / job["id"] / "heartbeat"
    child_code = f"import time; from pathlib import Path; p=Path({str(heartbeat)!r});\nwhile True: p.write_text(str(time.monotonic())); time.sleep(.02)"
    parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(60)"
    queue = JobQueue(store, command=lambda _: [sys.executable, "-c", parent_code])
    queue.start()
    try:
        wait_for(heartbeat.exists)
        queue.cancel(job["id"])
        wait_for(lambda: store.get(job["id"])["status"] == "cancelled")
        last_write = heartbeat.stat().st_mtime_ns
        time.sleep(.15)
        assert heartbeat.stat().st_mtime_ns == last_write
    finally:
        queue.stop()


@pytest.mark.parametrize("command", [lambda _: ["/nonexistent/genia-worker"],
                                    lambda _: [sys.executable, "-c", "pass"]])
def test_startup_or_missing_output_failure_does_not_stall_queue(tmp_path, command):
    store = Store(tmp_path)
    first, second = enqueue(store), enqueue(store)
    queue = JobQueue(store, command=command)
    queue.start()
    try:
        wait_for(lambda: store.get(second["id"])["status"] == "failed")
        assert store.get(first["id"])["status"] == "failed"
    finally:
        queue.stop()


def test_full_queue_rejects_upload_without_leaking_files(app):
    store = app.extensions["genia_store"]
    for _ in range(100): enqueue(store)
    response = upload(app.test_client())
    assert response.status_code == 400 and "queue is full" in response.json["error"]
    assert len(list(store.jobs_dir.iterdir())) == 100


def test_pipeline_recipe_matches_existing_image_pipeline(tmp_path):
    from hydra import compose, initialize_config_module
    directory = tmp_path / "spaces and 'quotes'"
    with initialize_config_module(config_module="genia.core.configs", version_base="1.2"):
        config = compose(config_name="genia", overrides=pipeline_overrides(directory, 17))
    assert config.dataset.name == "image"
    assert config.dataset.path == str(directory / "input")
    assert config.output.output_dir == str(directory / "output")
    assert config.processing.seed == 17
    assert config.output.save_output_mesh
    assert config.output.save_output_renders
    assert not config.output.save_viz_orbit


def test_server_import_does_not_load_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; from genia.core.webui.app import create_app; assert 'torch' not in sys.modules"], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_live_stage_visible_only_for_running_jobs(app):
    from genia.core.webui.progress import write_stage
    client = app.test_client()
    job = upload(client).json
    store = app.extensions["genia_store"]
    directory = store.jobs_dir / job["id"]
    assert job["progress"] is None
    store.claim()
    assert client.get(f"/api/jobs/{job['id']}").json["progress"]["stage"] == "starting"
    write_stage(directory, "finetune")
    progress = client.get("/api/jobs").json["queue"][0]["progress"]
    assert progress == {"stage": "finetune", "label": "Optimizing shape and appearance"}
    assert not (directory / "progress.tmp").exists()
    (directory / "progress.json").write_text("{broken")
    assert client.get(f"/api/jobs/{job['id']}").json["progress"]["stage"] == "starting"
    write_stage(directory, "exporting")
    store.finish(job["id"], "failed", "Example failure")
    assert client.get(f"/api/jobs/{job['id']}").json["progress"] is None


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled", "interrupted"])
def test_delete_removes_history_and_all_files(app, status):
    client = app.test_client()
    job = upload(client).json
    other = upload(client).json
    store = app.extensions["genia_store"]
    directory = store.jobs_dir / job["id"]
    (directory / "output").mkdir()
    (directory / "output/model.glb").write_bytes(b"model")
    (directory / "worker.log").write_text("log")
    store.claim()
    store.finish(job["id"], status)
    response = client.delete(f"/api/jobs/{job['id']}", headers=HEADERS)
    assert response.status_code == 200 and response.json == {"deleted": job["id"]}
    assert not directory.exists()
    assert Store(store.root).get(job["id"]) is None
    assert store.get(other["id"])["status"] == "queued"
    assert (store.jobs_dir / other["id"]).exists()
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.get(job["input_url"]).status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}", headers=HEADERS).status_code == 404
    snapshot = client.get(f"/api/jobs?since={job['seq']}").json
    assert job["id"] not in snapshot["history_ids"]
    assert snapshot["counts"] == {"queued": 1}


@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_delete_rejects_active_jobs(app, status):
    client = app.test_client()
    job = upload(client).json
    store = app.extensions["genia_store"]
    if status != "queued":
        store.claim()
    if status == "cancelling":
        store.cancel(job["id"])
    response = client.delete(f"/api/jobs/{job['id']}", headers=HEADERS)
    assert response.status_code == 409
    assert "Cancel" in response.json["error"]
    assert store.get(job["id"])["status"] == status
    assert (store.jobs_dir / job["id"] / "input.jpg").exists()


def test_delete_failure_keeps_record_and_can_be_retried(app, monkeypatch):
    from genia.core.webui import store as store_module
    client = app.test_client()
    job = upload(client).json
    store = app.extensions["genia_store"]
    store.cancel(job["id"])
    with monkeypatch.context() as patch:
        def fail(path):
            raise PermissionError("Test cleanup failure")
        patch.setattr(store_module.shutil, "rmtree", fail)
        response = client.delete(f"/api/jobs/{job['id']}", headers=HEADERS)
    assert response.status_code == 500
    assert store.get(job["id"])["status"] == "cancelled"
    assert client.delete(f"/api/jobs/{job['id']}", headers=HEADERS).status_code == 200


def test_delete_requires_write_header_and_tolerates_missing_files(app):
    client = app.test_client()
    store = app.extensions["genia_store"]
    job = enqueue(store)
    store.cancel(job["id"])
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 403
    (store.jobs_dir / job["id"]).rmdir()
    assert client.delete(f"/api/jobs/{job['id']}", headers=HEADERS).status_code == 200
    assert client.get("/api/jobs?since=-1").status_code == 400


def test_delete_does_not_follow_symlinks(app, tmp_path):
    client = app.test_client()
    store = app.extensions["genia_store"]
    job = enqueue(store)
    store.cancel(job["id"])
    directory = store.jobs_dir / job["id"]
    directory.rmdir()
    target = tmp_path / "external"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    directory.symlink_to(target, target_is_directory=True)
    assert client.delete(f"/api/jobs/{job['id']}", headers=HEADERS).status_code == 200
    assert (target / "keep.txt").read_text() == "keep"
    assert not directory.is_symlink()


def test_worker_forwards_actual_pipeline_stages(tmp_path, monkeypatch):
    import types
    from genia.core.webui import worker
    from genia.core.webui.progress import read_stage
    (tmp_path / "request.json").write_text('{"seed":42}')
    observed = []
    monkeypatch.setattr(worker, "prepare_mask", lambda directory: None)
    def fake_run(config, on_stage):
        assert config.processing.seed == 42
        for stage in ["loading_models", "preprocessing", "shape_init", "finetune", "exporting"]:
            on_stage(stage)
            observed.append(read_stage(tmp_path)["stage"])
    monkeypatch.setitem(sys.modules, "genia.core.run", types.SimpleNamespace(run=fake_run))
    worker.generate(tmp_path)
    assert observed == ["loading_models", "preprocessing", "shape_init", "finetune", "exporting"]


def test_pipeline_reports_enabled_stages_before_work(tmp_path, monkeypatch):
    """Run the actual runner with model work stubbed, including disabled blocks."""
    import contextlib
    import types
    from hydra import compose, initialize_config_module
    from genia.core import run as runner
    from genia.core.manifest import BLOCKS
    from genia.core.utils import config as config_utils
    from genia.core.utils import pipeline_state

    with initialize_config_module(config_module="genia.core.configs", version_base="1.2"):
        cfg = compose(config_name="genia", overrides=pipeline_overrides(tmp_path, 42))
    stages = []
    executed = []
    state = types.SimpleNamespace(check_consistency=lambda: None)
    class FakePipeline:
        pipeline_obj = None
        def __init__(self, config):
            assert stages[-1] == "loading_models"
        def bind(self, *args):
            assert stages[-1] == "preprocessing"
        def new_state(self):
            return state
        def run_block(self, block, state):
            if config_utils.block_config(cfg, block).enabled:
                assert stages[-1] == block
                executed.append(block)
    def final(*args, **kwargs):
        assert stages[-1] == "exporting"
    timer = types.SimpleNamespace(reset=lambda: None, start_block=lambda _: None,
                                  end_block=lambda _: None)
    replacements = {
        "genia.core.pipeline": {"Pipeline": FakePipeline},
        "genia.core.evaluator": {"Evaluator": lambda device: None},
        "genia.core.final": {"run_final": final},
        "genia.core.state": {"SAM3DState": object},
        "genia.core.preprocessing": {
            "resolve_run_assets": lambda cfg: (1, [0], [0], [0]),
            "run_preprocessing": lambda *args: None},
        "genia.core.utils.timing": {"get_timer": lambda: timer,
                                    "memory_snapshot": lambda *args: contextlib.nullcontext()},
    }
    for name, values in replacements.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**values))
    monkeypatch.setattr(runner, "get_device", lambda: "cpu")
    monkeypatch.setattr(config_utils, "print_config_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_state, "save_pipeline_cache", lambda *args: pytest.fail("Unexpected cache write"))
    runner.run(cfg, on_stage=stages.append)
    expected = [block for block, section in BLOCKS if cfg[section].enabled]
    assert executed == expected
    assert stages == ["loading_models", "preprocessing", *expected, "exporting"]
