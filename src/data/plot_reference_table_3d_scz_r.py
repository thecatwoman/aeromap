import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


X_VALUES = np.array(
    [-10.0, -5.0, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0],
    dtype="float64",
)

Y_VALUES = np.array(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0],
    dtype="float64",
)

Z_VALUES = np.array(
    [
        [1.853, 1.878, 1.901, 1.911, 1.921, 1.930, 1.943, 1.948, 1.954, 1.955, 1.988, 1.992, 2.006, 2.021, 2.037],
        [1.901, 1.928, 1.952, 1.962, 1.972, 1.983, 1.996, 2.007, 2.014, 2.011, 2.015, 2.015, 2.012, 2.007, 2.000],
        [1.946, 1.976, 2.002, 2.013, 2.023, 2.037, 2.048, 2.067, 2.074, 2.066, 2.042, 2.037, 2.018, 1.993, 1.963],
        [1.972, 2.005, 2.032, 2.043, 2.053, 2.081, 2.116, 2.120, 2.124, 2.118, 2.074, 2.024, 1.984, 1.922, 1.849],
        [1.977, 1.999, 2.016, 2.022, 2.026, 2.050, 2.080, 2.089, 2.096, 2.092, 2.056, 2.009, 1.959, 1.911, 1.843],
        [1.938, 1.945, 1.947, 1.947, 1.944, 1.956, 1.976, 1.991, 2.004, 2.009, 2.003, 1.965, 1.923, 1.877, 1.832],
        [1.767, 1.799, 1.824, 1.834, 1.842, 1.850, 1.868, 1.890, 1.910, 1.920, 1.917, 1.903, 1.876, 1.835, 1.789],
        [1.706, 1.724, 1.737, 1.741, 1.744, 1.753, 1.773, 1.803, 1.837, 1.836, 1.835, 1.834, 1.837, 1.781, 1.733],
        [1.541, 1.583, 1.619, 1.635, 1.650, 1.668, 1.687, 1.720, 1.745, 1.750, 1.743, 1.735, 1.720, 1.693, 1.659],
        [1.485, 1.515, 1.539, 1.550, 1.560, 1.570, 1.600, 1.630, 1.650, 1.666, 1.649, 1.635, 1.620, 1.602, 1.582],
        [1.340, 1.387, 1.432, 1.453, 1.473, 1.478, 1.511, 1.534, 1.552, 1.563, 1.554, 1.544, 1.534, 1.522, 1.511],
        [1.265, 1.308, 1.350, 1.370, 1.390, 1.386, 1.420, 1.433, 1.450, 1.474, 1.470, 1.466, 1.461, 1.455, 1.449],
        [1.191, 1.231, 1.270, 1.290, 1.309, 1.296, 1.332, 1.335, 1.352, 1.389, 1.388, 1.389, 1.389, 1.389, 1.388],
    ],
    dtype="float64",
)


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
        description="Plot the ScZ_R reference table as a 3D surface.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/plots/reference_table/reference_table_3d_scz_r.png"),
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
        default=-5.0,
        help="Shift applied to the reference surface X axis values.",
    )
    parser.add_argument(
        "--surface-y-shift",
        type=float,
        default=-5.0,
        help="Shift applied to the reference surface Y axis values.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay real-data corner medians as cyan floating points and lines.",
    )
    parser.add_argument(
        "--overlay-table",
        type=Path,
        default=Path("data/processed/tables/corner_maps/corner_scz_r_map_macroway_two_way_table_exact.csv"),
        help="Sparse two-way table CSV used when --overlay is enabled.",
    )
    parser.add_argument(
        "--compare-overlay-table",
        type=Path,
        default=None,
        help="Optional second sparse two-way table CSV to overlay for comparison.",
    )
    parser.add_argument(
        "--overlay-label",
        default="Primary segment medians",
        help="Legend label for the primary overlay table.",
    )
    parser.add_argument(
        "--compare-overlay-label",
        default="Comparison segment medians",
        help="Legend label for the comparison overlay table.",
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
        const="data/processed/plots/reference_table/reference_table_3d_scz_r.html",
        default=None,
        help="Also export an interactive HTML plot. Optionally pass a custom output path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save outputs without opening the matplotlib window.",
    )
    return parser.parse_args()


def load_sparse_table_points(table_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(table_path, index_col=0)
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []

    for y_label, row in cast(Any, table).iterrows():
        y_value = float(str(y_label))
        for x_label, cell in cast(Any, row).items():
            if pd.isna(cell):
                continue
            x_values.append(float(str(x_label)))
            y_values.append(float(y_value))
            z_values.append(float(str(cell)))

    if not x_values:
        raise ValueError(f"Overlay table has no numeric points: {table_path}")

    return (
        np.asarray(x_values, dtype="float64"),
        np.asarray(y_values, dtype="float64"),
        np.asarray(z_values, dtype="float64"),
    )


def colorscale_from_cmap(cmap: LinearSegmentedColormap, steps: int = 11) -> list[list[object]]:
    scale: list[list[object]] = []
    for idx in range(steps):
        t = idx / (steps - 1)
        r, g, b, _ = cmap(t)
        scale.append([t, f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"])
    return scale


def build_interactive_html(
    html_path: Path,
    overlay_enabled: bool,
    overlay_table: Path,
    compare_overlay_table: Path | None,
    overlay_label: str,
    compare_overlay_label: str,
    surface_x_shift: float,
    surface_y_shift: float,
    overlay_x_shift: float,
    overlay_y_shift: float,
    elev: float,
    azim: float,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    shifted_x = X_VALUES + surface_x_shift
    shifted_y = Y_VALUES + surface_y_shift
    x_grid, y_grid = np.meshgrid(shifted_x, shifted_y)
    surface_text = [[f"{value:.3f}" for value in row] for row in Z_VALUES.tolist()]
    plotly_traces: list[dict[str, object]] = [
        {
            "type": "surface",
            "x": shifted_x.tolist(),
            "y": shifted_y.tolist(),
            "z": Z_VALUES.tolist(),
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
            "z": (Z_VALUES.ravel() + 0.012).tolist(),
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
        base_z = np.full_like(overlay_z, float(Z_VALUES.min()) - 0.03)
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
                    "name": overlay_label,
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
        if compare_overlay_table is not None:
            compare_x, compare_y, compare_z = load_sparse_table_points(compare_overlay_table)
            compare_x = compare_x + overlay_x_shift
            compare_y = compare_y + overlay_y_shift
            compare_base_z = np.full_like(compare_z, float(Z_VALUES.min()) - 0.03)
            plotly_traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers+text",
                    "x": compare_x.tolist(),
                    "y": compare_y.tolist(),
                    "z": compare_z.tolist(),
                    "text": [f"{value:.3f}" for value in compare_z],
                    "textposition": "top center",
                    "textfont": {"size": 11, "color": "#ff3b30"},
                    "marker": {
                        "size": 5,
                        "color": "#ff3b30",
                        "line": {"color": "#101010", "width": 1},
                    },
                    "name": compare_overlay_label,
                    "hovertemplate": "X=%{x}<br>Y=%{y}<br>Reference value=%{z:.3f}<extra></extra>",
                }
            )
            for x_value, y_value, z_value, z_start in zip(
                compare_x.tolist(),
                compare_y.tolist(),
                compare_z.tolist(),
                compare_base_z.tolist(),
                strict=False,
            ):
                plotly_traces.append(
                    {
                        "type": "scatter3d",
                        "mode": "lines",
                        "x": [x_value, x_value],
                        "y": [y_value, y_value],
                        "z": [z_start, z_value],
                        "line": {"color": "#ff3b30", "width": 3},
                        "hoverinfo": "skip",
                        "showlegend": False,
                    }
                )

    layout = {
        "title": "ScZ_R Reference Table 3D Surface",
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
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ScZ_R Reference Table 3D Surface</title>
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
    html_path = Path(args.html) if args.html else None
    if html_path:
        build_interactive_html(
            html_path=html_path,
            overlay_enabled=args.overlay,
            overlay_table=args.overlay_table,
            compare_overlay_table=args.compare_overlay_table,
            overlay_label=args.overlay_label,
            compare_overlay_label=args.compare_overlay_label,
            surface_x_shift=args.surface_x_shift,
            surface_y_shift=args.surface_y_shift,
            overlay_x_shift=args.overlay_x_shift,
            overlay_y_shift=args.overlay_y_shift,
            elev=args.elev,
            azim=args.azim,
        )

    shifted_x = X_VALUES + args.surface_x_shift
    shifted_y = Y_VALUES + args.surface_y_shift
    x_grid, y_grid = np.meshgrid(shifted_x, shifted_y)

    fig = plt.figure(figsize=(12, 8))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        Z_VALUES,
        cmap=ELECTRIC_RYG,
        edgecolor="black",
        linewidth=0.35,
        antialiased=True,
        alpha=0.96,
    )

    ax.contour(
        x_grid,
        y_grid,
        Z_VALUES,
        zdir="z",
        offset=float(Z_VALUES.min()) - 0.05,
        cmap=ELECTRIC_RYG,
        levels=12,
        linewidths=1.0,
    )

    for x_value, y_value, z_value in zip(
        x_grid.ravel(),
        y_grid.ravel(),
        Z_VALUES.ravel(),
        strict=False,
    ):
        ax.text(
            x_value,
            y_value,
            z_value + 0.012,
            f"{z_value:.3f}",
            fontsize=7,
            color="#111111",
            ha="center",
            va="bottom",
        )

    ax.set_title("ScZ_R Reference Table 3D Surface")
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
        base_z = np.full_like(overlay_z, float(Z_VALUES.min()) - 0.03)
        ax.scatter(
            overlay_x,
            overlay_y,
            overlay_z,
            s=58,
            c="#00e5ff",
            edgecolors="#101010",
            linewidths=0.9,
            depthshade=False,
            label=args.overlay_label,
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
        if args.compare_overlay_table is not None:
            compare_x, compare_y, compare_z = load_sparse_table_points(args.compare_overlay_table)
            compare_x = compare_x + args.overlay_x_shift
            compare_y = compare_y + args.overlay_y_shift
            compare_base_z = np.full_like(compare_z, float(Z_VALUES.min()) - 0.03)
            ax.scatter(
                compare_x,
                compare_y,
                compare_z,
                s=58,
                c="#ff3b30",
                edgecolors="#101010",
                linewidths=0.9,
                depthshade=False,
                label=args.compare_overlay_label,
            )
            for x_value, y_value, z_value, z_start in zip(
                compare_x,
                compare_y,
                compare_z,
                compare_base_z,
                strict=False,
            ):
                ax.plot(
                    [x_value, x_value],
                    [y_value, y_value],
                    [z_start, z_value],
                    color="#ff3b30",
                    linewidth=1.0,
                    alpha=0.9,
                )
                ax.text(
                    x_value,
                    y_value,
                    z_value + 0.02,
                    f"{z_value:.3f}",
                    fontsize=8,
                    color="#ff3b30",
                    ha="center",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "#0b0b0b",
                        "edgecolor": "#ff3b30",
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
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
