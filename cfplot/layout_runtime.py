"""Runtime layout/axes operations for refactored contour rendering.

These helpers keep stateful calls to legacy cfplot plot orchestration out of
contour.py while preserving behaviour.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import matplotlib
import matplotlib.pyplot as plot

from .state import plotvars


def gopen(
    rows: int = 1,
    columns: int = 1,
    user_plot: int = 1,
    file: str = "cfplot.png",
    orientation: str = "landscape",
    figsize: list[float] | tuple[float, float] = (11.7, 8.3),
    left: float | None = None,
    right: float | None = None,
    top: float | None = None,
    bottom: float | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
    dpi: float | None = None,
    user_position: bool = False,
) -> None:
    """Open a contour-runtime graphics session compatible with cfplot.gopen."""
    plotvars.rows = rows
    plotvars.columns = columns
    if file != "cfplot.png":
        plotvars.file = file
    plotvars.orientation = orientation

    _open_figure(
        user_plot=user_plot,
        figsize=figsize,
        orientation=orientation,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
        dpi=dpi,
        user_position=user_position,
    )

    plotvars._contour_session_open = True


def gclose(view: bool = True) -> None:
    """Close a contour-runtime graphics session and save/view output."""
    plotvars.user_plot = 0

    interactive = bool(globals().get("__IPYTHON__", False))

    if matplotlib.is_interactive():
        interactive = True

    saveargs = {}
    if plotvars.tight:
        saveargs = {"bbox_inches": "tight"}

    file = plotvars.file
    figure = plotvars.master_plot or getattr(plotvars.plot, "figure", None)
    if figure is not None and file is not None:
        if os.path.splitext(file)[1].lower() not in (".ps", ".eps", ".png", ".pdf"):
            file = file + ".png"
        figure.savefig(
            file,
            orientation=plotvars.orientation,
            dpi=plotvars.dpi,
            **saveargs,
        )
        plot.close(figure)
    elif figure is not None:
        if view and plotvars.viewer == "display" and not interactive:
            disp = shutil.which("display")
            if disp is not None:
                tfile = "cfplot.png"
                figure.savefig(
                    tfile,
                    orientation=plotvars.orientation,
                    dpi=plotvars.dpi,
                    **saveargs,
                )
                matplotlib.pyplot.ioff()
                subprocess.Popen([disp, tfile])
            else:
                plotvars.viewer = "matplotlib"

        if view and (plotvars.viewer == "matplotlib" or interactive):
            matplotlib.pyplot.ion()
            plot.show()
            plot.close(figure)
        elif not view:
            plot.close(figure)

    plotvars.plot = None
    plotvars.twinx = None
    plotvars.twiny = None
    plotvars.plot_xmin = None
    plotvars.plot_xmax = None
    plotvars.plot_ymin = None
    plotvars.plot_ymax = None
    plotvars.graph_xmin = None
    plotvars.graph_xmax = None
    plotvars.graph_ymin = None
    plotvars.graph_ymax = None
    plotvars.gpos_called = False
    plotvars.mymap = None
    plotvars.titles_con_called = False
    plotvars.master_plot = None
    plotvars._contour_session_open = False


def ensure_xy_viewport() -> None:
    """Ensure a Cartesian viewport exists, matching legacy gopen/gpos behavior."""
    if plotvars.master_plot is None:
        _open_figure(user_plot=0)

    if plotvars.plot is None or (plotvars.rows > 1 or plotvars.columns > 1):
        if plotvars.gpos_called is False or plotvars.plot is None:
            _select_position(1)


def set_plot_limits(
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    ylog: bool,
    user_gset: int,
) -> None:
    """Apply graph limits through legacy gset interface."""
    plotvars.user_gset = user_gset
    plotvars.xmin = xmin
    plotvars.xmax = xmax
    plotvars.ymin = ymin
    plotvars.ymax = ymax
    plotvars.ylog = ylog

    time_xstr = False
    time_ystr = False
    try:
        float(xmin)
    except Exception:
        time_xstr = True
    try:
        float(ymin)
    except Exception:
        time_ystr = True

    if plotvars.plot is not None and not time_xstr and not time_ystr:
        plotvars.plot.axis([xmin, xmax, ymin, ymax])

    if ylog and plotvars.plot is not None:
        plotvars.plot.set_yscale("log")


def apply_axes(
    *,
    plot_type: int,
    xticks: Any,
    yticks: Any,
    xlabel: str | None,
    ylabel: str | None,
    xticklabels: Any | None,
    yticklabels: Any | None,
) -> None:
    """Apply axis labels/ticks through legacy axes handlers."""
    if plot_type == 1:
        from .map_runtime import _apply_map_axes

        _apply_map_axes(
            xticks=xticks,
            yticks=yticks,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
    else:
        _apply_xy_axes(
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xlabel=xlabel,
            ylabel=ylabel,
        )


def _open_figure(
    user_plot: int = 0,
    *,
    figsize: list[float] | tuple[float, float] = (11.7, 8.3),
    orientation: str | None = None,
    left: float | None = None,
    right: float | None = None,
    top: float | None = None,
    bottom: float | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
    dpi: float | None = None,
    user_position: bool = False,
) -> None:
    """Open a contour-owned figure and apply legacy-default layout."""
    if orientation is None:
        orientation = plotvars.orientation

    if orientation == "portrait":
        figshape = (figsize[1], figsize[0])
    elif orientation == "landscape":
        figshape = (figsize[0], figsize[1])
    else:
        raise Warning(
            "gopen error\n"
            "orientation incorrectly set\n"
            f"input value was {orientation}\n"
            "Valid options are portrait or landscape\n"
        )

    plotvars.master_plot = plot.figure(figsize=figshape)

    if left is None:
        left = 0.12
    if right is None:
        right = 0.92
    if top is None:
        top = 0.95
    if bottom is None:
        bottom = 0.1 if plotvars.rows >= 3 else 0.08
    if wspace is None:
        wspace = 0.2
    if hspace is None:
        hspace = 0.5 if plotvars.rows >= 3 else 0.2

    plotvars.master_plot.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )

    plotvars.user_plot = user_plot
    plotvars.gpos_called = False
    plotvars.plot_xmin = None
    plotvars.plot_xmax = None
    plotvars.plot_ymin = None
    plotvars.plot_ymax = None

    if dpi is not None:
        plotvars.dpi = dpi

    if not user_position and plotvars.rows == 1 and plotvars.columns == 1:
        _select_position(1)

    if plotvars.columns > 2 or plotvars.rows > 2:
        matplotlib.rcParams["xtick.major.size"] = 2
        matplotlib.rcParams["ytick.major.size"] = 2


def _select_position(pos: int) -> None:
    """Select subplot position in the contour-owned figure state."""
    if plotvars.master_plot is None:
        _open_figure(user_plot=0)

    max_pos = plotvars.rows * plotvars.columns
    if pos < 1 or pos > max_pos:
        raise Warning(
            "pos error - pos out of range:\n"
            f" range = 1 - {max_pos}\n"
            f" input pos was {pos}\n"
        )

    user_pos = all(
        value is not None
        for value in [
            plotvars.plot_xmin,
            plotvars.plot_xmax,
            plotvars.plot_ymin,
            plotvars.plot_ymax,
        ]
    )

    if not user_pos:
        plotvars.plot = plotvars.master_plot.add_subplot(
            plotvars.rows, plotvars.columns, pos
        )
    else:
        delta_x = plotvars.plot_xmax - plotvars.plot_xmin
        delta_y = plotvars.plot_ymax - plotvars.plot_ymin
        plotvars.plot = plotvars.master_plot.add_axes(
            [plotvars.plot_xmin, plotvars.plot_ymin, delta_x, delta_y]
        )

    plotvars.plot.tick_params(which="both", direction="out", right=True, top=True)
    plotvars.pos = pos
    plotvars.gpos_called = True
    plotvars.mymap = None
    plotvars.graph_xmin = None
    plotvars.graph_xmax = None
    plotvars.graph_ymin = None
    plotvars.graph_ymax = None
    plotvars.titles_con_called = False


def _apply_xy_axes(
    *,
    xticks: Any,
    xticklabels: Any | None,
    yticks: Any,
    yticklabels: Any | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    """Apply simple Cartesian axes labels and ticks."""
    if plotvars.plot is None:
        return

    if xlabel:
        plotvars.plot.set_xlabel(
            xlabel,
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
        )
    if ylabel:
        plotvars.plot.set_ylabel(
            ylabel,
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
        )

    if xticks is not None:
        plotvars.plot.set_xticks(xticks)
        plotvars.plot.set_xticklabels(
            xticklabels if xticklabels is not None else xticks,
            rotation=plotvars.xtick_label_rotation,
            horizontalalignment=plotvars.xtick_label_align,
        )

    if yticks is not None:
        plotvars.plot.set_yticks(yticks)
        plotvars.plot.set_yticklabels(
            yticklabels if yticklabels is not None else yticks,
            rotation=plotvars.ytick_label_rotation,
            horizontalalignment=plotvars.ytick_label_align,
        )

    for label in plotvars.plot.xaxis.get_ticklabels():
        label.set_fontsize(plotvars.axis_label_fontsize)
        label.set_fontweight(plotvars.axis_label_fontweight)
    for label in plotvars.plot.yaxis.get_ticklabels():
        label.set_fontsize(plotvars.axis_label_fontsize)
        label.set_fontweight(plotvars.axis_label_fontweight)

