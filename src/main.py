from __future__ import annotations

import argparse
import json
import socket
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.config import get_nested, load_config
from src.datasets.paths import PROJECT_ROOT
from src.utils.json import write_json


GUI_DIR = PROJECT_ROOT / "src" / "gui"
JAEGER_AGENT_HOST = "localhost"
JAEGER_AGENT_PORT = 6831
JAEGER_UI_HOST = "127.0.0.1"
JAEGER_UI_PORT = 16686


def add_search_args(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    parser.add_argument("--clip-model", default=get_nested(config, "models", "clip", "openai/clip-vit-base-patch32"))
    parser.add_argument("--clap-model", default=get_nested(config, "models", "clap", "laion/clap-htsat-unfused"))
    parser.add_argument("--projection-type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--projection-hidden-dim", type=int, default=1024)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--qdrant-url", default=get_nested(config, "qdrant", "url", "http://localhost:6333"))
    parser.add_argument("--collection-name", default=get_nested(config, "qdrant", "collection_name", "reflectra_music_clap"))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-checkpoint", default=None)
    parser.add_argument("--reranker-type", choices=["mlp", "attention"], default=get_nested(config, "reranker", "type", "mlp"))
    parser.add_argument("--reranker-hidden-dim", type=int, default=get_nested(config, "reranker", "hidden_dim", 512))
    parser.add_argument("--reranker-dropout", type=float, default=get_nested(config, "reranker", "dropout", 0.1))
    parser.add_argument("--timing-json", default=None)
    parser.add_argument(
        "--timing-plot",
        nargs="?",
        const="",
        default=None,
        help="Write a timing plot. Optional path; defaults to plots/<trace>.png.",
    )
    parser.add_argument("--show-timing-plot", action="store_true")
    parser.add_argument(
        "--jaeger",
        action="store_true",
        help="Export OpenTelemetry spans to Jaeger.",
    )
    parser.add_argument(
        "--otel-service-name",
        default="reflectra",
        help="OpenTelemetry service name.",
    )
    parser.add_argument(
        "--timing-dir",
        default="plots",
        help="Directory used for timing plots when --timing-plot is not provided.",
    )


def config_arg_parser() -> tuple[argparse.ArgumentParser, dict[str, Any]]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    return config_parser, load_config(config_args.config)


def build_parser() -> argparse.ArgumentParser:
    config_parser, config = config_arg_parser()

    parser = argparse.ArgumentParser(
        description="Reflectra image-to-song search.",
        parents=[config_parser],
    )
    add_search_args(parser, config)
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument("--output", default=None)

    return parser


def build_gui_parser() -> argparse.ArgumentParser:
    config_parser, config = config_arg_parser()
    parser = argparse.ArgumentParser(
        description="Launch Reflectra desktop GUI and local search server.",
        parents=[config_parser],
    )
    add_search_args(parser, config)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--gui-dir", default=str(GUI_DIR))
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Run only the local HTTP server; do not launch the desktop shell.",
    )
    return parser


def load_model(args: argparse.Namespace):
    from src.models.reflectra_model import ReflectraModel

    if args.use_reranker and not args.reranker_checkpoint:
        raise ValueError("--use-reranker requires --reranker-checkpoint")

    model = ReflectraModel(
        clip_model_name=args.clip_model,
        clap_model_name=args.clap_model,
        projection_type=args.projection_type,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        device=args.device,
        projection_checkpoint=args.checkpoint,
        use_reranker=args.use_reranker,
        reranker_type=args.reranker_type,
        reranker_hidden_dim=args.reranker_hidden_dim,
        reranker_dropout=args.reranker_dropout,
        reranker_checkpoint=args.reranker_checkpoint,
        reranker_top_k=args.candidate_k,
    )
    model.eval()
    return model


def configure_tracing(args: argparse.Namespace) -> None:
    if not getattr(args, "jaeger", False):
        return

    from src.opentelemetry.telemetry import configure_jaeger_exporter

    configure_jaeger_exporter(
        service_name=args.otel_service_name,
        agent_host_name=JAEGER_AGENT_HOST,
        agent_port=JAEGER_AGENT_PORT,
    )


def search_image(
    args: argparse.Namespace,
    image_path: str,
    model: Any | None = None,
) -> dict[str, Any]:
    from src.vector_db.qdrant_store import get_qdrant_client, search_vector

    configure_tracing(args)
    client = get_qdrant_client(url=args.qdrant_url)
    model = model or load_model(args)

    if args.use_reranker:
        from src.vector_db.rerank_search import search_image_with_rerank

        if model.reranker is None:
            raise RuntimeError("Reranker was requested but model.reranker is not loaded.")

        search_result = search_image_with_rerank(
            client=client,
            collection_name=args.collection_name,
            model=model,
            reranker=model.reranker,
            image_path=image_path,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            timing_plot_path=resolve_timing_plot_path(args.timing_plot, args.timing_dir, "image_search_with_rerank"),
            timing_json_path=args.timing_json,
            show_timing_plot=args.show_timing_plot,
            return_timings=True,
        )
        results = search_result["results"]
        timings = search_result["timings"]
    else:
        import torch

        from src.opentelemetry.telemetry import StageTimer

        timer = StageTimer("image_search")
        with timer.stage("encode_query"):
            with torch.no_grad():
                query_embed = model.encode_image([image_path])[0]

        with timer.stage("check_db"):
            candidates = search_vector(
                client=client,
                collection_name=args.collection_name,
                query_vector=query_embed.cpu().numpy().tolist(),
                limit=args.final_k,
            )

        with timer.stage("format_results"):
            results = [
                {
                    "payload": point.payload,
                    "bi_encoder_score": float(point.score),
                    "rerank_score": None,
                }
                for point in candidates
            ]

        if args.timing_json:
            timer.save_json(args.timing_json)
        timing_plot_path = resolve_timing_plot_path(args.timing_plot, args.timing_dir, "image_search")
        if timing_plot_path is not None:
            timer.save_plot(timing_plot_path, show=args.show_timing_plot)
        timings = timer.as_dicts()

    return {
        "image": image_path,
        "collection_name": args.collection_name,
        "reranker_used": bool(args.use_reranker),
        "results": results,
        "timings": timings,
    }


def resolve_timing_plot_path(
    value: str | None,
    timing_dir: str,
    trace_name: str,
) -> str | None:
    if value is None:
        return None
    if value:
        return value
    return str(Path(timing_dir) / f"{trace_name}.png")


def parse_multipart_image(body: bytes, content_type: str) -> tuple[bytes, str]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Expected multipart/form-data with boundary.")

    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    delimiter = f"--{boundary}".encode()

    for part in body.split(delimiter):
        if b"Content-Disposition:" not in part or b'name="image"' not in part:
            continue

        header, _, data = part.partition(b"\r\n\r\n")
        if not data:
            continue

        data = data.rstrip(b"\r\n-")
        filename = "upload.png"
        for item in header.decode(errors="ignore").split(";"):
            item = item.strip()
            if item.startswith("filename="):
                filename = item.split("=", 1)[1].strip('"') or filename

        return data, filename

    raise ValueError("No image file field found in multipart body.")


def dependency_status(args: argparse.Namespace) -> dict[str, Any]:
    from src.vector_db.qdrant_store import get_qdrant_client

    status: dict[str, Any] = {
        "ok": True,
        "qdrant": {
            "ok": True,
            "url": args.qdrant_url,
            "message": "Qdrant is reachable.",
        },
        "jaeger": {
            "enabled": bool(args.jaeger),
            "ok": True,
            "ui_url": f"http://{JAEGER_UI_HOST}:{JAEGER_UI_PORT}",
            "message": "Jaeger tracing is not enabled.",
        },
    }

    try:
        client = get_qdrant_client(url=args.qdrant_url)
        client.get_collections()
    except Exception as exc:
        status["ok"] = False
        status["qdrant"] = {
            "ok": False,
            "url": args.qdrant_url,
            "message": f"Qdrant is not reachable at {args.qdrant_url}. Run scripts/setup.sh.",
            "error": str(exc),
        }

    if args.jaeger:
        jaeger_ok = is_tcp_port_open(JAEGER_UI_HOST, JAEGER_UI_PORT)
        status["jaeger"] = {
            "enabled": True,
            "ok": jaeger_ok,
            "ui_url": f"http://{JAEGER_UI_HOST}:{JAEGER_UI_PORT}",
            "message": (
                "Jaeger UI is ready."
                if jaeger_ok
                else "Jaeger UI is not reachable. Run scripts/setup.sh."
            ),
        }
        if not jaeger_ok:
            status["ok"] = False

    return status


def is_tcp_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


class ReflectraRequestHandler(BaseHTTPRequestHandler):
    server_version = "ReflectraGUI/0.1"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self.send_json(
                200,
                dependency_status(self.server.search_args),  # type: ignore[attr-defined]
            )
            return

        self.send_json(200, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/api/search":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            image_bytes, filename = parse_multipart_image(
                body=body,
                content_type=self.headers.get("Content-Type", ""),
            )

            suffix = Path(filename).suffix or ".png"
            upload_dir = Path(tempfile.gettempdir()) / "reflectra_gui_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                dir=upload_dir,
                delete=False,
            ) as upload_file:
                upload_file.write(image_bytes)
                image_path = upload_file.name

            result = search_image(
                args=self.server.search_args,  # type: ignore[attr-defined]
                image_path=image_path,
                model=self.server.model,  # type: ignore[attr-defined]
            )
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def build_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    configure_tracing(args)
    model = load_model(args)
    gui_dir = Path(args.gui_dir).expanduser().resolve()
    if not gui_dir.exists():
        raise FileNotFoundError(f"GUI directory does not exist: {gui_dir}")

    server = ThreadingHTTPServer((args.host, args.port), ReflectraRequestHandler)
    server.gui_dir = gui_dir  # type: ignore[attr-defined]
    server.search_args = args  # type: ignore[attr-defined]
    server.model = model  # type: ignore[attr-defined]
    return server


def serve(args: argparse.Namespace) -> None:
    server = build_server(args)
    print(f"[INFO] Reflectra GUI: http://{args.host}:{args.port}")
    print("[INFO] Press Ctrl+C to stop.")
    server.serve_forever()


def gui_main() -> None:
    args = build_gui_parser().parse_args()

    if args.server_only:
        serve(args)
        return

    from src.gui.app import run_app

    server_state: dict[str, Any] = {
        "server": None,
        "error": None,
    }

    def backend_worker() -> None:
        try:
            server = build_server(args)
            server_state["server"] = server
            print(f"[INFO] Reflectra backend ready: http://{args.host}:{args.port}")
            server.serve_forever()
        except Exception as exc:
            server_state["error"] = exc
            print("[ERROR] Reflectra backend failed to start:")
            traceback.print_exc()

    thread = threading.Thread(target=backend_worker, daemon=True)
    thread.start()
    print(f"[INFO] Reflectra GUI opened; backend warming at http://{args.host}:{args.port}")

    try:
        raise SystemExit(
            run_app(
                backend_url=f"http://{args.host}:{args.port}",
                gui_dir=Path(args.gui_dir).expanduser().resolve(),
            )
        )
    finally:
        server = server_state.get("server")
        if server is not None:
            server.shutdown()


def main() -> None:
    args = build_parser().parse_args()
    configure_tracing(args)
    result = search_image(args=args, image_path=args.image)
    print(json.dumps(result, indent=2))

    if args.output:
        write_json(Path(args.output), result)
        print(f"Saved search result to: {args.output}")


if __name__ == "__main__":
    main()
