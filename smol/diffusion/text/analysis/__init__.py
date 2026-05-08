from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "smol").exists() and (candidate / "runs").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root containing both 'smol/' and 'runs/'.")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs(runs_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not runs_root.exists():
        return runs

    for experiment_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            logs_dir = run_dir / "logs"
            summary_path = run_dir / "run_summary.json"
            metrics_path = logs_dir / "train_metrics.jsonl"
            events_path = logs_dir / "events.jsonl"

            summary = load_json(summary_path) if summary_path.exists() else {}
            metrics = load_jsonl(metrics_path)
            events = load_jsonl(events_path)
            checkpoint_steps = [event.get("step") for event in events if event.get("message") == "checkpoint_saved"]

            runs.append(
                {
                    "experiment_name": experiment_dir.name,
                    "run_stamp": run_dir.name,
                    "run_dir": run_dir,
                    "summary": summary,
                    "metrics": metrics,
                    "events": events,
                    "num_metrics": len(metrics),
                    "last_step": metrics[-1]["step"] if metrics else None,
                    "checkpoint_steps": checkpoint_steps,
                }
            )

    runs.sort(key=lambda run: (run["experiment_name"], run["run_stamp"]))
    return runs


def describe_run(run: dict[str, Any]) -> str:
    return (
        f"{run['experiment_name']}/{run['run_stamp']} | "
        f"steps={run['last_step']} | metrics={run['num_metrics']} | "
        f"checkpoints={len(run['checkpoint_steps'])}"
    )


def select_runs(
    runs: list[dict[str, Any]],
    *,
    experiment_name: str | None = None,
    run_stamps: list[str] | None = None,
    max_auto_runs: int = 3,
) -> list[dict[str, Any]]:
    filtered = runs
    if experiment_name is not None:
        filtered = [run for run in filtered if run["experiment_name"] == experiment_name]
    if run_stamps:
        wanted = set(run_stamps)
        filtered = [run for run in filtered if run["run_stamp"] in wanted]
    elif max_auto_runs > 0:
        filtered = filtered[-max_auto_runs:]
    return filtered


def metric_series(run: dict[str, Any], key: str) -> tuple[list[Any], list[Any]]:
    xs: list[Any] = []
    ys: list[Any] = []
    for row in run["metrics"]:
        value = row.get(key)
        if value is None:
            continue
        xs.append(row["step"])
        ys.append(value)
    return xs, ys


def maybe_smooth_series(values: list[Any], sigma: float) -> list[Any]:
    if sigma <= 0 or len(values) < 2:
        return values
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(values, sigma=sigma, mode="nearest").tolist()


def available_metric_keys(runs: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for run in runs:
        for row in run["metrics"]:
            keys.update(row.keys())
    return sorted(keys)


def expand_internal_metric_keys(
    runs: list[dict[str, Any]],
    *,
    layer_index: int,
    base_names: list[str],
    summaries: list[str],
) -> list[str]:
    available = set(available_metric_keys(runs))
    metric_keys: list[str] = []
    for base_name in base_names:
        prefix = f"internals/layer_{layer_index}/{base_name}"
        for summary in summaries:
            key = f"{prefix}/{summary}"
            if key in available:
                metric_keys.append(key)
    return metric_keys


def configure_plotly_renderer() -> str:
    import plotly.io as pio

    if "VSCODE_PID" in os.environ:
        renderer = "vscode"
    else:
        renderer = "notebook_connected"
    pio.renderers.default = renderer
    return renderer


def show_figure(fig: "go.Figure") -> None:
    try:
        fig.show()
    except Exception:
        from IPython.display import HTML, display

        display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))


def add_checkpoint_lines(fig: "go.Figure", run: dict[str, Any], row: int, col: int, color: str) -> None:
    for step in run["checkpoint_steps"]:
        if step is None:
            continue
        fig.add_vline(
            x=step,
            line_width=1,
            line_dash="dot",
            line_color=color,
            opacity=0.35,
            row=row,
            col=col,
        )


def plot_metric_grid(
    runs: list[dict[str, Any]],
    metric_keys: list[str],
    *,
    title: str,
    smoothing_sigma: float = 0.0,
    height_per_row: int = 280,
) -> "go.Figure":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=len(metric_keys), cols=1, shared_xaxes=True, subplot_titles=metric_keys)
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
    ]

    for run_index, run in enumerate(runs):
        color = palette[run_index % len(palette)]
        label = f"{run['experiment_name']}/{run['run_stamp']}"
        for row_index, key in enumerate(metric_keys, start=1):
            xs, ys = metric_series(run, key)
            if not xs:
                continue
            smoothed_ys = maybe_smooth_series(ys, smoothing_sigma)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=smoothed_ys,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=row_index == 1,
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        "run=%{fullData.name}<br>"
                        f"metric={key}<br>"
                        "step=%{x}<br>"
                        "value=%{y}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=1,
            )
            add_checkpoint_lines(fig, run, row_index, 1, color)

    fig.update_layout(
        title=f"{title} (gaussian sigma={smoothing_sigma})" if smoothing_sigma > 0 else title,
        height=max(1, len(metric_keys)) * height_per_row,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 20, "t": 90, "b": 50},
    )
    fig.update_xaxes(title_text="step", row=len(metric_keys), col=1)
    return fig


def plot_metric_panel_grid(
    runs: list[dict[str, Any]],
    metric_keys: list[str],
    *,
    title: str,
    smoothing_sigma: float = 0.0,
    columns: int = 3,
    height_per_row: int = 280,
) -> "go.Figure":
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not metric_keys:
        raise ValueError("metric_keys must not be empty")

    columns = max(1, columns)
    rows = math.ceil(len(metric_keys) / columns)
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=metric_keys)
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
    ]

    for run_index, run in enumerate(runs):
        color = palette[run_index % len(palette)]
        label = f"{run['experiment_name']}/{run['run_stamp']}"
        for metric_index, key in enumerate(metric_keys):
            xs, ys = metric_series(run, key)
            if not xs:
                continue
            row_index = (metric_index // columns) + 1
            col_index = (metric_index % columns) + 1
            smoothed_ys = maybe_smooth_series(ys, smoothing_sigma)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=smoothed_ys,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=metric_index == 0,
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        "run=%{fullData.name}<br>"
                        f"metric={key}<br>"
                        "step=%{x}<br>"
                        "value=%{y}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=col_index,
            )
            add_checkpoint_lines(fig, run, row_index, col_index, color)

    fig.update_layout(
        title=f"{title} (gaussian sigma={smoothing_sigma})" if smoothing_sigma > 0 else title,
        height=max(1, rows) * height_per_row,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 20, "t": 90, "b": 50},
    )
    for row_index in range(1, rows + 1):
        for col_index in range(1, columns + 1):
            if (row_index - 1) * columns + col_index > len(metric_keys):
                continue
            fig.update_xaxes(title_text="step", row=row_index, col=col_index)
    return fig
