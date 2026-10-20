from collections import Counter

import matplotlib

matplotlib.use("Agg")  # Save straight to PNG files without opening a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from log_merger import format_timestamp

# Blue scale for the heatmaps, and blue, green and orange for event rate, local bursts and cross-boundary bursts
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
EVENT_COLOR = "#2a78d6"
LOCAL_COLOR = "#1baf7a"
CROSS_COLOR = "#eb6834"
BLUE_SCALE = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab",
                                                         "#0d366b"])


def style_axes(axes, title, xLabel, yLabel):
    # Light background, gray axes and no top or right border
    axes.set_facecolor(SURFACE)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(GRID)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axes.set_title(title, fontsize=11, loc="left", color=TEXT_PRIMARY)
    axes.set_xlabel(xLabel, color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel(yLabel, color=TEXT_SECONDARY, fontsize=9)
    axes.grid(axis="y", color=GRID, linewidth=0.8)


def plot_dependency_heatmaps(matrices, titles, components, path):
    # Heatmaps of A, A^2 and A^3 side by side with the value printed in every cell
    figure, axesList = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.6), facecolor=SURFACE)
    for axes, matrix, title in zip(axesList, matrices, titles):
        peak = max(max(row) for row in matrix) or 1
        axes.imshow(matrix, cmap=BLUE_SCALE, vmin=0, vmax=peak)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                textColor = "#ffffff" if value > peak * 0.55 else TEXT_PRIMARY  # White text on dark cells
                axes.text(j, i, str(value), ha="center", va="center", fontsize=9, color=textColor)
        axes.set_xticks(range(len(components)), components)
        axes.set_yticks(range(len(components)), components)
        axes.set_xlabel("to component (Y)", color=TEXT_SECONDARY, fontsize=9)
        axes.set_ylabel("from component (X)", color=TEXT_SECONDARY, fontsize=9)
        axes.set_title(title, fontsize=11)
        axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_burst_timeline(timestamps, bursts, zoomStartMs, zoomEndMs, path):
    # Events per second over the whole run with the bursts shaded, plus a zoomed view so the short bursts are visible
    startMs = min(timestamps)
    perSecond = Counter((timestamp - startMs) // 1000 for timestamp in timestamps)
    seconds = list(range(max(perSecond) + 1))
    figure, (overview, zoom) = plt.subplots(2, 1, figsize=(11, 7), facecolor=SURFACE,
                                            gridspec_kw={"height_ratios": [1, 1.1]})

    style_axes(overview, f"Events per second from {format_timestamp(startMs)} (bursts shaded)",
               "seconds since first event", "events / second")
    overview.plot(seconds, [perSecond.get(second, 0) for second in seconds], color=EVENT_COLOR, linewidth=1.5)
    for start, end, eventCount, kind in bursts:
        color = LOCAL_COLOR if kind == "local" else CROSS_COLOR
        overview.axvspan((start - startMs) / 1000, (end - startMs) / 1000 + 1, color=color, alpha=0.35, linewidth=0, zorder=0)

    style_axes(zoom, f"Zoom: {format_timestamp(zoomStartMs)} to {format_timestamp(zoomEndMs)} (100 ms bins)",
               "seconds since zoom start", "events / 100 ms")
    binWidth = 100
    zoomPoints = [timestamp for timestamp in timestamps if zoomStartMs <= timestamp <= zoomEndMs]
    perBin = Counter((timestamp - zoomStartMs) // binWidth for timestamp in zoomPoints)
    bins = list(range((zoomEndMs - zoomStartMs) // binWidth + 1))
    zoom.bar([b * binWidth / 1000 for b in bins], [perBin.get(b, 0) for b in bins], width=binWidth / 1000 * 0.85,
             align="edge", color=EVENT_COLOR, linewidth=0, zorder=2)
    for start, end, eventCount, kind in bursts:
        if end >= zoomStartMs and start <= zoomEndMs:
            color = LOCAL_COLOR if kind == "local" else CROSS_COLOR
            zoom.axvspan((start - zoomStartMs) / 1000, (end - zoomStartMs) / 1000, color=color, alpha=0.3, linewidth=0, zorder=0)

    legendItems = [Patch(color=EVENT_COLOR, label="event rate"),
                   Patch(color=LOCAL_COLOR, alpha=0.5, label="local burst"),
                   Patch(color=CROSS_COLOR, alpha=0.5, label="cross-boundary burst")]
    overview.legend(handles=legendItems, frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
