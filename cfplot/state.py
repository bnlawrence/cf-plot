"""Plotting state management.

Encapsulates the mutable state needed for plotting,
replacing the global plotvars pattern.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class PlotState:
    """Immutable holder for plotting configuration and state.
    
    Replaces the global plotvars pattern with explicit, passable state.
    """

    def __init__(
        self,
        plot: Any = None,
        mymap: Any = None,
        rows: int = 1,
        columns: int = 1,
        gpos_called: bool = False,
        user_plot: int = 0,
        title_fontsize: int | None = None,
        title_fontweight: str = "normal",
        plot_type: int = 0,
        levels: np.ndarray | None = None,
        cscale_flag: int = 0,
        cs: list | None = None,
        cs_user: str = "viridis",
        cs_uniform: bool = True,
        levels_extend: str = "neither",
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        user_gset: int = 0,
        user_mapset: int = 0,
        proj: str = "cyl",
        resolution: str = "m",
        text_fontsize: int | None = None,
        continent_color: str | None = "k",
        continent_thickness: float | None = 1.5,
        continent_linestyle: str = "solid",
        ocean_color: str | None = None,
        land_color: str | None = None,
        lake_color: str | None = None,
        feature_zorder: int = 1,
        tspace_year: int | None = None,
        tspace_day: int | None = None,
        tspace_hour: int | None = None,
        degsym: bool = True,
        user_levs: int = 0,
        levels_step: float | None = None,
        norm: Any = None,
        image: Any = None,
    ):
        """Initialize plot state.
        
        Parameters are stored as immutable attributes.
        """
        self.plot = plot or plt.gca()
        self.mymap = mymap
        self.rows = rows
        self.columns = columns
        self.gpos_called = gpos_called
        self.user_plot = user_plot
        self.title_fontsize = title_fontsize or 15
        self.title_fontweight = title_fontweight
        self.plot_type = plot_type
        self.levels = levels
        self.cscale_flag = cscale_flag
        self.cs = cs or []
        self.cs_user = cs_user
        self.cs_uniform = cs_uniform
        self.levels_extend = levels_extend
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.user_gset = user_gset
        self.user_mapset = user_mapset
        self.proj = proj
        self.resolution = resolution
        self.text_fontsize = text_fontsize or 11
        self.continent_color = continent_color
        self.continent_thickness = continent_thickness
        self.continent_linestyle = continent_linestyle
        self.ocean_color = ocean_color
        self.land_color = land_color
        self.lake_color = lake_color
        self.feature_zorder = feature_zorder
        self.tspace_year = tspace_year
        self.tspace_day = tspace_day
        self.tspace_hour = tspace_hour
        self.degsym = degsym
        self.user_levs = user_levs
        self.levels_step = levels_step
        self.norm = norm
        self.image = image
        
        # Mutable state for animation tracking
        self._contour_animation_artists: list[Any] = []

    def with_updates(self, **kwargs) -> "PlotState":
        """Create a new PlotState with updated attributes."""
        current = {
            "plot": self.plot,
            "mymap": self.mymap,
            "rows": self.rows,
            "columns": self.columns,
            "gpos_called": self.gpos_called,
            "user_plot": self.user_plot,
            "title_fontsize": self.title_fontsize,
            "title_fontweight": self.title_fontweight,
            "plot_type": self.plot_type,
            "levels": self.levels,
            "cscale_flag": self.cscale_flag,
            "cs": self.cs,
            "cs_user": self.cs_user,
            "cs_uniform": self.cs_uniform,
            "levels_extend": self.levels_extend,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "user_gset": self.user_gset,
            "user_mapset": self.user_mapset,
            "proj": self.proj,
            "resolution": self.resolution,
            "text_fontsize": self.text_fontsize,
            "continent_color": self.continent_color,
            "continent_thickness": self.continent_thickness,
            "continent_linestyle": self.continent_linestyle,
            "ocean_color": self.ocean_color,
            "land_color": self.land_color,
            "lake_color": self.lake_color,
            "feature_zorder": self.feature_zorder,
            "tspace_year": self.tspace_year,
            "tspace_day": self.tspace_day,
            "tspace_hour": self.tspace_hour,
            "degsym": self.degsym,
            "user_levs": self.user_levs,
            "levels_step": self.levels_step,
            "norm": self.norm,
            "image": self.image,
        }
        current.update(kwargs)
        return PlotState(**current)
