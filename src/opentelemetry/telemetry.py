from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    from opentelemetry import trace
except ModuleNotFoundError:
    trace = None


_TRACING_CONFIGURED = False


def configure_jaeger_exporter(
    service_name: str = "reflectra",
    agent_host_name: str = "localhost",
    agent_port: int = 6831,
) -> None:
    """
    Export OpenTelemetry spans to a local Jaeger agent.
    """

    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return

    if trace is None:
        raise RuntimeError(
            "OpenTelemetry is not installed. Reinstall dependencies before using --jaeger."
        )

    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        package_name = getattr(exc, "name", None) or "opentelemetry-exporter-jaeger-thrift"
        raise RuntimeError(
            f"Jaeger tracing dependency '{package_name}' is missing or incompatible. "
            "Run `pip install -e .` to install the pinned Reflectra tracing dependencies."
        ) from exc

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name})
    )
    exporter = JaegerExporter(
        agent_host_name=agent_host_name,
        agent_port=agent_port,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True


@dataclass
class StageTiming:
    stage: str
    seconds: float


class StageTimer:
    def __init__(self, trace_name: str) -> None:
        self.trace_name = trace_name
        self.timings: list[StageTiming] = []
        self._tracer = trace.get_tracer(__name__) if trace is not None else None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        span_context = (
            self._tracer.start_as_current_span(name)
            if self._tracer is not None
            else None
        )

        if span_context is None:
            try:
                yield
            finally:
                self.timings.append(
                    StageTiming(stage=name, seconds=time.perf_counter() - start)
                )
            return

        with span_context as span:
            try:
                yield
            finally:
                seconds = time.perf_counter() - start
                span.set_attribute("reflectra.stage", name)
                span.set_attribute("reflectra.stage.seconds", seconds)
                self.timings.append(StageTiming(stage=name, seconds=seconds))

    def as_dicts(self) -> list[dict[str, float | str]]:
        return [
            {
                "stage": timing.stage,
                "seconds": timing.seconds,
            }
            for timing in self.timings
        ]

    def save_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "trace_name": self.trace_name,
                    "timings": self.as_dicts(),
                },
                output_file,
                indent=2,
            )
        return path

    def save_plot(self, output_path: str | Path, show: bool = False) -> Path:
        import matplotlib.pyplot as plt

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        stages = [timing.stage for timing in self.timings]
        seconds = [timing.seconds for timing in self.timings]

        plt.figure(figsize=(9, 4.5))
        bars = plt.bar(stages, seconds, color="#4C78A8")
        plt.ylabel("Seconds")
        plt.title(f"Stage Time: {self.trace_name}")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, seconds):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close()

        return path
