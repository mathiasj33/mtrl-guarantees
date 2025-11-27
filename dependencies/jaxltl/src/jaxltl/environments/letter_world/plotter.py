"""
Plotting utilities for LetterWorld environment.
"""

import matplotlib.pyplot as plt
import numpy as np

from jaxltl.environments.zone_env.plotter import FancyAxes, draw_diamond


def draw_trajectories(
    letters_grids,
    paths,
    propositions,
    grid_size,
    num_cols,
    num_rows,
    path_color="green",
):
    if len(letters_grids) != len(paths):
        raise ValueError("Number of grids and paths must be the same")
    if num_cols * num_rows < len(letters_grids):
        raise ValueError("Number of plots exceeds the number of subplots")

    fig = plt.figure(figsize=(20, 15))
    for i, (grid, path) in enumerate(zip(letters_grids, paths, strict=False)):
        ax = fig.add_subplot(
            num_rows,
            num_cols,
            i + 1,
            axes_class=FancyAxes,
            edgecolor="gray",
            linewidth=0.5,
        )
        setup_axis(ax, grid_size)
        draw_letters(ax, grid, propositions, path)
        draw_path(ax, path, grid_size, color=path_color)

    plt.tight_layout(pad=5.0, h_pad=5.0)
    plt.show()


def setup_axis(ax, grid_size):
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add border like in ZoneEnv
    color = "gray"
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(0.5)

    # Set background color
    ax.set_facecolor((240 / 255, 240 / 255, 240 / 255))


def draw_letters(ax, grid, propositions, path):
    # grid shape: (G, G, L)
    rows, cols, letter_idxs = np.where(grid)

    for r, c, l_idx in zip(rows, cols, letter_idxs, strict=False):
        letter = propositions[l_idx]

        # Draw cell background
        cell_color = (200 / 255, 200 / 255, 200 / 255)
        rect = plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1, facecolor=cell_color, edgecolor=None, zorder=0
        )
        ax.add_patch(rect)

        # Text color
        ax.text(
            c,
            r,
            letter,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            zorder=3,
        )


def draw_path(ax, path, grid_size, linewidth=3, color=(0, 1, 0, 0.3)):
    if not path:
        return

    path = np.array(path)
    xs = path[:, 1]
    ys = path[:, 0]

    # Start
    draw_diamond(ax, (xs[0], ys[0]), "orange", size=0.2)

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        y1, x1 = p1
        y2, x2 = p2

        if abs(x1 - x2) > 1:
            if x1 > x2:
                ax.plot(
                    [x1, grid_size - 0.5], [y1, y1], color=color, linewidth=linewidth
                )
                ax.plot([-0.5, x2], [y2, y2], color=color, linewidth=linewidth)
            else:
                ax.plot([x1, -0.5], [y1, y1], color=color, linewidth=linewidth)
                ax.plot(
                    [grid_size - 0.5, x2], [y2, y2], color=color, linewidth=linewidth
                )
        elif abs(y1 - y2) > 1:
            if y1 > y2:
                ax.plot(
                    [x1, x1], [y1, grid_size - 0.5], color=color, linewidth=linewidth
                )
                ax.plot([x2, x2], [-0.5, y2], color=color, linewidth=linewidth)
            else:
                ax.plot([x1, x1], [y1, -0.5], color=color, linewidth=linewidth)
                ax.plot(
                    [x2, x2], [grid_size - 0.5, y2], color=color, linewidth=linewidth
                )
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
