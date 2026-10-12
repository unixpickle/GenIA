"""Run the local UI: python -m genia.core.webui."""
import argparse
import signal
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="GenIA local generation queue and gallery")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: localhost only)")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("results/webui"), help="Persistent uploads, queue, and results")
    parser.add_argument("--cache-root", type=Path, help="Model cache root, e.g. /Volumes/MLData4/GenIA")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], help="Default: automatic device selection")
    args = parser.parse_args()
    try:
        from werkzeug.serving import make_server
        from .app import create_app
    except ImportError as exc:
        parser.exit(1, f"Web UI dependencies are missing ({exc}). Run: uv pip install 'flask>=3.1,<4'\n")
    environment = {}
    if args.cache_root:
        root = args.cache_root.expanduser().resolve()
        environment.update(HF_HUB_CACHE=str(root / "huggingface" / "hub"),
                           HF_XET_CACHE=str(root / "huggingface" / "xet"), TORCH_HOME=str(root / "torch"))
    if args.device:
        environment["GENIA_DEVICE"] = args.device
    # Explicit LAN binding is supported for a trusted network; there is no authentication.
    hosts = ["localhost", "127.0.0.1", "[::1]", args.host]
    app = create_app(args.data_dir.expanduser(), environment=environment, hosts=hosts)
    queue = app.extensions["genia_queue"]
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        with make_server(args.host, args.port, app, threaded=True) as server:
            queue.start()
            print(f"GenIA → http://{args.host}:{args.port}  |  Data: {args.data_dir.resolve()}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        queue.stop()


if __name__ == "__main__":
    main()
