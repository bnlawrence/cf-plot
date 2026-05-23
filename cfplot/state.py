"""Shared plotting state for cf-plot."""

from __future__ import annotations

import os
import sys
from typing import Any

import cartopy
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np

from . import utility


class pvars:
    """Stores plotting variables in `cfp.plotvars`."""

    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def __str__(self):
        a = None
        v = None
        out = [f"{a} = {repr(v)}"]
        for a, v in self.__dict__.items():
            return "\n".join(out)


try:
    disp = os.environ["DISPLAY"]
except Exception:
    matplotlib.use("Agg")

try:
    pre_existing_data_dir = os.environ["pre_existing_data_dir"]
    cartopy.config["pre_existing_data_dir"] = pre_existing_data_dir
except KeyError:
    pass


cscale1 = [
    "#0a3278",
    "#0f4ba5",
    "#1e6ec8",
    "#3ca0f0",
    "#50b4fa",
    "#82d2ff",
    "#a0f0ff",
    "#c8faff",
    "#e6ffff",
    "#fffadc",
    "#ffe878",
    "#ffc03c",
    "#ffa000",
    "#ff6000",
    "#ff3200",
    "#e11400",
    "#c00000",
    "#a50000",
]

global_fill = True
global_lines = True
global_blockfill = False
global_degsym = False
global_viewer = "display"

defaults_file = os.path.expanduser("~") + "/.cfplot_defaults"
if os.path.exists(defaults_file):
    with open(defaults_file) as file:
        for line in file:
            vals = line.split(" ")
            com, val = vals
            v = False
            if val.strip() == "True":
                v = True
            if com == "blockfill":
                global_blockfill = v
            if com == "lines":
                global_lines = v
            if com == "fill":
                global_fill = v
            if com == "degsym":
                global_degsym = v
            if com == "viewer":
                global_viewer = val.strip()


setvars_defaults = {
    "viewer": global_viewer,
    "file": None,
    "dpi": None,
    "tight": False,
    "tspace_year": None,
    "tspace_month": None,
    "tspace_day": None,
    "tspace_hour": None,
    "xtick_label_rotation": 0,
    "xtick_label_align": "center",
    "ytick_label_rotation": 0,
    "ytick_label_align": "right",
    "text_fontsize": 11,
    "axis_label_fontsize": 11,
    "colorbar_fontsize": 11,
    "title_fontsize": 15,
    "master_title_fontsize": 30,
    "legend_text_size": 11,
    "fontweight": "normal",
    "text_fontweight": "normal",
    "axis_label_fontweight": "normal",
    "colorbar_fontweight": "normal",
    "title_fontweight": "normal",
    "master_title_fontweight": "normal",
    "legend_text_weight": "normal",
    "master_title": None,
    "master_title_location": [0.5, 0.95],
    "legend_frame": True,
    "legend_frame_edge_color": "k",
    "legend_frame_face_color": None,
    "grid": False,
    "grid_x_spacing": 60,
    "grid_y_spacing": 30,
    "grid_zorder": 100,
    "grid_colour": "k",
    "grid_linestyle": "--",
    "grid_thickness": 1.0,
    "rotated_grid_spacing": 10,
    "rotated_deg_spacing": 0.75,
    "rotated_continents": True,
    "rotated_grid": True,
    "rotated_grid_thickness": 1.0,
    "rotated_labels": True,
    "feature_zorder": 999,
    "land_color": None,
    "ocean_color": None,
    "lake_color": None,
    "continent_color": None,
    "continent_thickness": None,
    "continent_linestyle": None,
    "axis_width": None,
    "degsym": global_degsym,
    "level_spacing": None,
    "cs_uniform": True,
}

plotvars_defaults = {
    "global_viewer": global_viewer,
    "plot_type": 1,
    "master_plot": None,
    "plot": None,
    "mymap": None,
    "proj": "cyl",
    "resolution": "110m",
    "norm": None,
    "lonmin": -180,
    "lonmax": 180,
    "latmin": -90,
    "latmax": 90,
    "xmin": None,
    "xmax": None,
    "ymin": None,
    "ymax": None,
    "plot_xmin": None,
    "plot_xmax": None,
    "plot_ymin": None,
    "plot_ymax": None,
    "graph_xmin": None,
    "graph_xmax": None,
    "graph_ymin": None,
    "graph_ymax": None,
    "levels": None,
    "levels_min": None,
    "levels_max": None,
    "levels_step": None,
    "levels_extend": "both",
    "xticks": None,
    "xticklabels": None,
    "xlabel": None,
    "yticks": None,
    "yticklabels": None,
    "ylabel": None,
    "cs": cscale1,
    "cscale_flag": 0,
    "pos": 1,
    "gpos_called": False,
    "boundinglat": 0,
    "lon_0": 0,
    "lat_0": 40,
    "xlog": None,
    "ylog": None,
    "twinx": False,
    "twiny": False,
    "xstep": None,
    "ystep": None,
    "user_mapset": 0,
    "user_gset": 0,
    "user_levs": 0,
    "user_plot": 0,
    "cs_user": "cscale1",
    "rows": 1,
    "columns": 1,
    "title": None,
    "titles_con_called": False,
    "orientation": "landscape",
    "aspect": "equal",
    "_contour_session_open": False,
    "_contour_animation_artists": [],
}

allvars_defaults = {**setvars_defaults, **plotvars_defaults}
plotvars = pvars(**allvars_defaults)

# Keep shared-state viewer defaults aligned with legacy startup behavior.
is_inline = "inline" in matplotlib.get_backend()
if is_inline:
    plotvars.viewer = None

if sys.platform == "darwin":
    plotvars.global_viewer = "matplotlib"
    plotvars.viewer = "matplotlib"


def setvars(**kwargs: Any) -> None:
    """Set shared plotting variables from defaults plus explicit overrides."""
    for def_var, def_value in setvars_defaults.items():
        setattr(plotvars, def_var, def_value)

    for set_var, set_value in kwargs.items():
        if set_var not in setvars_defaults:
            raise ValueError(
                f"Unrecognised keyword argument for setvars: {set_var}"
            )

        setattr(plotvars, set_var, set_value)

    if "grid" not in kwargs and (
        "grid_x_spacing" in kwargs or "grid_y_spacing" in kwargs
    ):
        plotvars.grid = (
            plotvars.grid_x_spacing != setvars_defaults["grid_x_spacing"]
            or plotvars.grid_y_spacing != setvars_defaults["grid_y_spacing"]
        )

    pyplot.ioff()


def get_colour_scale_map() -> list[str]:
    """Return the active colour scale trimmed for extend settings."""
    cscale_ncols = len(plotvars.cs)
    if plotvars.levels_extend == "both":
        return plotvars.cs[1 : cscale_ncols - 1]
    if plotvars.levels_extend == "min":
        return plotvars.cs[1:]
    if plotvars.levels_extend == "max":
        return plotvars.cs[: cscale_ncols - 1]
    return plotvars.cs


def apply_colour_scale(
    scale: str | None = None,
    ncols: int | None = None,
    white: Any = None,
    below: int | None = None,
    above: int | None = None,
    reverse: bool = False,
    uniform: bool = False,
) -> None:
    """Apply a colour scale to shared plotting state."""
    if scale is None or scale == "":
        scale = "scale1"

    red, green, blue = utility.load_colour_scale_rgb(scale)

    if reverse:
        red = red[::-1]
        green = green[::-1]
        blue = blue[::-1]

    if ncols is not None:
        positions = np.linspace(0, np.size(red) - 1, num=ncols, endpoint=True)
        red, green, blue = utility.interpolate_colour_channels(
            red, green, blue, positions
        )

    if below is not None or above is not None:
        npoints = np.size(red) // 2

        x_below: np.ndarray | float | list[float] = []
        lower = npoints if below is None else below
        if below is not None and uniform:
            lower = max(above, below)
        if below == 1:
            x_below = 0
        if lower > 1:
            x_below = ((npoints - 1) / float(lower - 1)) * np.arange(lower)

        x_above: np.ndarray | float | list[float] = []
        upper = npoints if above is None else above
        if above is not None and uniform:
            upper = max(above, below)
        if above == 1:
            x_above = npoints * 2 - 1
        if upper > 1:
            x_above = ((npoints - 1) / float(upper - 1)) * np.arange(upper) + npoints

        positions = np.append(x_below, x_above)
        red, green, blue = utility.interpolate_colour_channels(
            red, green, blue, positions
        )

        if uniform:
            midpoint = max(below, above)
            red = red[midpoint - below : midpoint + above]
            green = green[midpoint - below : midpoint + above]
            blue = blue[midpoint - below : midpoint + above]

    hexarr = [
        f"#{int(red[idx]):02x}{int(green[idx]):02x}{int(blue[idx]):02x}"
        for idx in np.arange(np.size(red))
    ]

    if white is not None:
        if np.size(white) == 1:
            hexarr[white] = "#ffffff"
        else:
            for col in white:
                hexarr[col] = "#ffffff"

    plotvars.cs = hexarr


def cscale(
    scale: str | None = None,
    ncols: int | None = None,
    white: Any = None,
    below: int | None = None,
    above: int | None = None,
    reverse: bool = False,
    uniform: bool = False,
) -> None:
    """Choose and manipulate colour maps in shared plotting state."""
    if scale is None:
        plotvars.cscale_flag = 0
        return

    plotvars.cs_user = scale
    plotvars.cscale_flag = 1

    vals = [ncols, white, below, above]
    if any(val is not None for val in vals):
        plotvars.cscale_flag = 2
    if reverse is not False or uniform is not False:
        plotvars.cscale_flag = 2

    apply_colour_scale(
        scale=scale,
        ncols=ncols,
        white=white,
        below=below,
        above=above,
        reverse=reverse,
        uniform=uniform,
    )
