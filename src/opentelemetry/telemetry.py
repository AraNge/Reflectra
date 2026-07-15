from __future__ import annotations

import json
import os
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
    def trace(self) -> Iterator[None]:
        if self._tracer is None:
            yield
            return

        with self._tracer.start_as_current_span(self.trace_name) as span:
            span.set_attribute("reflectra.trace_name", self.trace_name)
            try:
                yield
            finally:
                span.set_attribute("reflectra.stage.count", len(self.timings))

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
        if not show:
            os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "reflectra_matplotlib"))
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        stages = [timing.stage for timing in self.timings]
        seconds = [timing.seconds for timing in self.timings]
        total_seconds = sum(seconds)
        max_total = total_seconds or 1.0
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#138a22",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#17becf",
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        bottom = 0.0
        for index, (stage, value) in enumerate(zip(stages, seconds)):
            color = colors[index % len(colors)]
            ax.bar(
                [0],
                [value],
                bottom=[bottom],
                width=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=f"{stage} (≈ {value:.3f}s)",
            )

            if value >= max_total * 0.045:
                ax.text(
                    0,
                    bottom + value / 2,
                    f"{value:.3f}s",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=15,
                    fontweight="bold",
                )
            else:
                ax.annotate(
                    f"{value:.3f}s",
                    xy=(0.31, bottom + value),
                    xytext=(0.42, bottom + max_total * 0.04),
                    arrowprops={"arrowstyle": "-", "color": color},
                    color=color,
                    fontsize=13,
                    fontweight="bold",
                )
            bottom += value

        ax.text(
            0,
            total_seconds + max_total * 0.025,
            f"Total: {total_seconds:.3f}s",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_title(f"Stage Time: {self.trace_name}", fontsize=24, pad=18)
        ax.set_ylabel("Seconds", fontsize=15)
        ax.set_xticks([0], ["Total"])
        ax.set_xlim(-0.55, 0.85)
        ax.set_ylim(0, max_total * 1.2)
        ax.grid(axis="y", linestyle=(0, (4, 4)), color="#cfcfcf", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=13)

        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return path


def force_flush_traces(timeout_millis: int = 5000) -> None:
    if trace is None:
        return

    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is None:
        return
    try:
        force_flush(timeout_millis=timeout_millis)
    except TypeError:
        force_flush()
