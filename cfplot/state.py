"""Shared plotting state for cf-plot."""

from __future__ import annotations

import os
import sys
from typing import Any

import cartopy
import matplotlib
import matplotlib.pyplot as pyplot

from .colour.colourmaps import cscale1


class pvars:
    """Stores plotting variables in `cfp.plotvars`."""

    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def __str__(self):
        out = []
        for a, v in self.__dict__.items():
            out.append(f"{a} = {repr(v)}")
        return "\n".join(out)


def reset_runtime_state() -> None:
    """Reset shared plotting runtime state back to defaults."""
    plotvars.master_plot = None
    plotvars.plot = None
    plotvars.mymap = None
    plotvars.norm = None
    plotvars.image = None
    plotvars.rows = 1
    plotvars.columns = 1
    plotvars.pos = 1
    plotvars.gpos_called = False
    plotvars.user_plot = 0
    plotvars._contour_session_open = False
    plotvars._contour_animation_artists = []
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
    plotvars.titles_con_called = False


try:
    disp = os.environ["DISPLAY"]
except Exception:
    matplotlib.use("Agg")

try:
    pre_existing_data_dir = os.environ["pre_existing_data_dir"]
    cartopy.config["pre_existing_data_dir"] = pre_existing_data_dir
except KeyError:
    pass

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
