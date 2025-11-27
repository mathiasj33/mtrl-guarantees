import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


class StaticColorAxisBBox(mpatches.FancyBboxPatch):
    def set_edgecolor(self, color):
        if hasattr(self, "_original_edgecolor"):
            return
        self._original_edgecolor = color
        self._set_edgecolor(color)  # type: ignore

    def set_linewidth(self, w):
        super().set_linewidth(1.5)


class FancyAxes(maxes.Axes):
    name = "fancy_box_axes"
    _edgecolor: str

    def __init__(self, *args, **kwargs):
        self._edgecolor = kwargs.pop("edgecolor", None)
        self._linewidth = kwargs.pop("linewidth", None)
        super().__init__(*args, **kwargs)

    def _gen_axes_patch(self):
        return StaticColorAxisBBox(
            (0, 0),
            1.0,
            1.0,
            boxstyle="round, rounding_size=0.06, pad=0",
            edgecolor=self._edgecolor,
            linewidth=5,
        )


_color_map = {
    "red": "#f34121",
    "green": "#4caf50",
    "purple": "violet",
    "yellow": "#fdd835",
}


def draw_circle(ax, center, color, radius=0.4):
    circ = plt.Circle(center, radius, fc=to_rgba(color, 0.8), ec=color)  # type: ignore
    ax.add_patch(circ)


def draw_zones(ax, zone_positions, colors):
    for (x, y), color in zip(zone_positions, colors, strict=False):
        if color in _color_map:
            color = _color_map[color]
        draw_circle(ax, (x, y), color)


def draw_path(
    ax, points, color, linewidth=2, markersize=5, style="solid", draw_markers=False
):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    ax.plot(x, y, color=color, linestyle=style, marker=None, linewidth=linewidth)
    if draw_markers:
        for i in range(20, len(points), 20):
            ax.plot(x[i], y[i], marker="o", color=color, markersize=markersize)
        ax.plot(x[-1], y[-1], marker="o", color=color, markersize=markersize)


def draw_diamond(ax, center, color, size=0.18):
    diamond_shape = plt.Polygon(  # type: ignore
        [
            (center[0], center[1] + size),
            (center[0] + size, center[1]),
            (center[0], center[1] - size),
            (center[0] - size, center[1]),
        ],
        color=_color_map.get(color, color),
        zorder=10,
    )
    ax.add_patch(diamond_shape)


def draw_trajectories(
    zone_positions: list[list[tuple[float, float]]],
    colors: list[list[str]],
    paths: list[list[tuple[float, float]]],
    num_cols: int,
    num_rows: int,
):
    if len(zone_positions) != len(paths):
        raise ValueError("Number of zone positions and paths must be the same")
    if num_cols * num_rows < len(zone_positions):
        raise ValueError("Number of zone positions exceeds the number of subplots")
    fig = plt.figure(figsize=(20, 15))
    for i, (zone_poss, path, cs) in enumerate(
        zip(zone_positions, paths, colors, strict=True)
    ):
        ax = fig.add_subplot(
            num_rows,
            num_cols,
            i + 1,
            axes_class=FancyAxes,
            edgecolor="gray",
            linewidth=0.5,
        )
        setup_axis(ax)
        draw_zones(ax, zone_poss, cs)
        draw_diamond(ax, path[0], color="orange")
        draw_path(ax, path, color="green", linewidth=4)
    plt.tight_layout(pad=4)
    plt.show()


def setup_axis(ax):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    ax.grid(
        True, which="both", color="gray", linestyle="dashed", linewidth=1, alpha=0.5
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    color = "gray"
    ax.spines["top"].set_color(color)
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_color(color)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["bottom"].set_color(color)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color(color)
    ax.spines["left"].set_linewidth(0.5)
    # ax.xaxis.get_gridlines()[-1].set_clip_on(False)
    # ax.yaxis.get_gridlines()[0].set_clip_on(False)
    hide_ticks(ax.xaxis)
    hide_ticks(ax.yaxis)

    # ax.set_xticks(np.arange(-3, 4, 1))
    # ax.set_yticks(np.arange(-3, 4, 1))
    # ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Add border
    # ax.patch.set_edgecolor('white')
    # ax.patch.set_linewidth(.5)


def hide_ticks(axis):
    for tick in axis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
