import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/tables/reference_table_sums/scz_total_reference_table.csv")

ELECTRIC_RYG = LinearSegmentedColormap.from_list(
    "electric_ryg",
    [
        (0.00, "#ff2a2a"),
        (0.20, "#ff5e00"),
        (0.42, "#ffd400"),
        (0.65, "#c8ff00"),
        (0.82, "#3dff5a"),
        (1.00, "#00d957"),
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the total SCz reference table as a 3D surface.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV file containing the total SCz reference table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/plots/reference_table/reference_table_3d_scz_total.png"),
        help="Path where the 3D plot image will be saved.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=28.0,
        help="3D view elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-135.0,
        help="3D view azimuth angle.",
    )
    parser.add_argument(
        "--surface-x-shift",
        type=float,
        default=0.0,
        help="Shift applied to the reference surface X axis values.",
    )
    parser.add_argument(
        "--surface-y-shift",
        type=float,
        default=0.0,
        help="Shift applied to the reference surface Y axis values.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay summed real-data corner medians as cyan floating points and lines.",
    )
    parser.add_argument(
        "--overlay-table",
        type=Path,
        default=Path("data/processed/tables/corner_maps/corner_scz_total_map_macroway_two_way_table_exact.csv"),
        help="Sparse two-way table CSV used when --overlay is enabled.",
    )
    parser.add_argument(
        "--overlay-x-shift",
        type=float,
        default=-5.0,
        help="Shift applied to overlay X coordinates.",
    )
    parser.add_argument(
        "--overlay-y-shift",
        type=float,
        default=-5.0,
        help="Shift applied to overlay Y coordinates.",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="data/processed/plots/reference_table/reference_table_3d_scz_total.html",
        default=None,
        help="Also export an interactive HTML plot. Optionally pass a custom output path.",
    )
    return parser.parse_args()


def colorscale_from_cmap(cmap: LinearSegmentedColormap, steps: int = 11) -> list[list[object]]:
    scale: list[list[object]] = []
    for idx in range(steps):
        t = idx / (steps - 1)
        r, g, b, _ = cmap(t)
        scale.append([t, f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"])
    return scale


def load_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, index_col=0)
    x_values = np.asarray([float(col) for col in df.columns], dtype="float64")
    y_values = np.asarray([float(idx) for idx in df.index], dtype="float64")
    z_values = df.to_numpy(dtype="float64")
    return x_values, y_values, z_values


def load_sparse_table_points(table_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(table_path, index_col=0)
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []

    for y_label, row in table.iterrows():
        y_value = float(y_label)
        for x_label, cell in row.items():
            if pd.isna(cell):
                continue
            x_values.append(float(x_label))
            y_values.append(y_value)
            z_values.append(float(cell))

    if not x_values:
        raise ValueError(f"Overlay table has no numeric points: {table_path}")

    return (
        np.asarray(x_values, dtype="float64"),
        np.asarray(y_values, dtype="float64"),
        np.asarray(z_values, dtype="float64"),
    )


def build_interactive_html(
    html_path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    overlay_enabled: bool,
    overlay_table: Path,
    overlay_x_shift: float,
    overlay_y_shift: float,
    elev: float,
    azim: float,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    surface_text = [[f"{value:.3f}" for value in row] for row in z_values.tolist()]
    plotly_traces: list[dict[str, object]] = [
        {
            "type": "surface",
            "x": x_values.tolist(),
            "y": y_values.tolist(),
            "z": z_values.tolist(),
            "colorscale": colorscale_from_cmap(ELECTRIC_RYG),
            "showscale": True,
            "colorbar": {"title": "Value"},
            "opacity": 0.97,
            "contours": {
                "z": {"show": True, "usecolormap": True, "project": {"z": True}, "width": 1},
            },
            "hovertemplate": "X=%{x}<br>Y=%{y}<br>Value=%{z:.3f}<extra></extra>",
        },
        {
            "type": "scatter3d",
            "mode": "text",
            "x": x_grid.ravel().tolist(),
            "y": y_grid.ravel().tolist(),
            "z": (z_values.ravel() + 0.02).tolist(),
            "text": [item for row in surface_text for item in row],
            "textfont": {"size": 10, "color": "#111111"},
            "hoverinfo": "skip",
            "showlegend": False,
        },
    ]

    if overlay_enabled:
        overlay_x, overlay_y, overlay_z = load_sparse_table_points(overlay_table)
        overlay_x = overlay_x + overlay_x_shift
        overlay_y = overlay_y + overlay_y_shift
        base_z = np.full_like(overlay_z, float(z_values.min()) - 0.03)
        plotly_traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "x": overlay_x.tolist(),
                "y": overlay_y.tolist(),
                "z": overlay_z.tolist(),
                "text": [f"{value:.3f}" for value in overlay_z],
                "textposition": "top center",
                "textfont": {"size": 11, "color": "#00e5ff"},
                "marker": {
                    "size": 5,
                    "color": "#00e5ff",
                    "line": {"color": "#101010", "width": 1},
                },
                "name": "Summed real corner medians",
                "hovertemplate": "X=%{x}<br>Y=%{y}<br>Real value=%{z:.3f}<extra></extra>",
            }
        )
        for x_value, y_value, z_value, z_start in zip(
            overlay_x.tolist(),
            overlay_y.tolist(),
            overlay_z.tolist(),
            base_z.tolist(),
            strict=False,
        ):
            plotly_traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "x": [x_value, x_value],
                    "y": [y_value, y_value],
                    "z": [z_start, z_value],
                    "line": {"color": "#00e5ff", "width": 3},
                    "hoverinfo": "skip",
                    "showlegend": False,
                }
            )

    layout = {
        "title": "Total SCz Reference Table 3D Surface",
        "scene": {
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
            "zaxis": {"title": "Value"},
            "camera": {
                "eye": {
                    "x": float(np.cos(np.deg2rad(azim)) * 1.8),
                    "y": float(np.sin(np.deg2rad(azim)) * 1.8),
                    "z": float(0.8 + np.sin(np.deg2rad(elev))),
                }
            },
        },
        "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Total SCz Reference Table 3D Surface</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body style="margin:0;background:#ffffff;">
  <div id="plot" style="width:100vw;height:100vh;"></div>
  <script>
    const data = {json.dumps(plotly_traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('plot', data, layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    x_values, y_values, z_values = load_table(args.input)
    shifted_x = x_values + args.surface_x_shift
    shifted_y = y_values + args.surface_y_shift

    html_path = Path(args.html) if args.html else None
    if html_path:
        build_interactive_html(
            html_path=html_path,
            x_values=shifted_x,
            y_values=shifted_y,
            z_values=z_values,
            overlay_enabled=args.overlay,
            overlay_table=args.overlay_table,
            overlay_x_shift=args.overlay_x_shift,
            overlay_y_shift=args.overlay_y_shift,
            elev=args.elev,
            azim=args.azim,
        )

    x_grid, y_grid = np.meshgrid(shifted_x, shifted_y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_values,
        cmap=ELECTRIC_RYG,
        edgecolor="black",
        linewidth=0.35,
        antialiased=True,
        alpha=0.96,
    )

    ax.contour(
        x_grid,
        y_grid,
        z_values,
        zdir="z",
        offset=float(z_values.min()) - 0.05,
        cmap=ELECTRIC_RYG,
        levels=12,
        linewidths=1.0,
    )

    for x_value, y_value, z_value in zip(
        x_grid.ravel(),
        y_grid.ravel(),
        z_values.ravel(),
        strict=False,
    ):
        ax.text(
            x_value,
            y_value,
            z_value + 0.02,
            f"{z_value:.3f}",
            fontsize=7,
            color="#111111",
            ha="center",
            va="bottom",
        )

    ax.set_title("Total SCz Reference Table 3D Surface")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")
    ax.view_init(elev=args.elev, azim=args.azim)

    colorbar = fig.colorbar(surface, shrink=0.72, pad=0.08)
    colorbar.set_label("Value")

    if args.overlay:
        overlay_x, overlay_y, overlay_z = load_sparse_table_points(args.overlay_table)
        overlay_x = overlay_x + args.overlay_x_shift
        overlay_y = overlay_y + args.overlay_y_shift
        base_z = np.full_like(overlay_z, float(z_values.min()) - 0.03)
        ax.scatter(
            overlay_x,
            overlay_y,
            overlay_z,
            s=58,
            c="#00e5ff",
            edgecolors="#101010",
            linewidths=0.9,
            depthshade=False,
            label="Summed real corner medians",
        )
        for x_value, y_value, z_value, z_start in zip(
            overlay_x,
            overlay_y,
            overlay_z,
            base_z,
            strict=False,
        ):
            ax.plot(
                [x_value, x_value],
                [y_value, y_value],
                [z_start, z_value],
                color="#00e5ff",
                linewidth=1.0,
                alpha=0.9,
            )
            ax.text(
                x_value,
                y_value,
                z_value + 0.02,
                f"{z_value:.3f}",
                fontsize=8,
                color="#00e5ff",
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#0b0b0b",
                    "edgecolor": "#00e5ff",
                    "linewidth": 0.8,
                    "alpha": 0.9,
                },
            )
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))

    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {args.output}")
    if html_path:
        print(f"Saved interactive HTML: {html_path}")
    plt.show()


if __name__ == "__main__":
    main()
