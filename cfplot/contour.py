"""Contour plotting module.

Provides the refactored contour plotting interface.

Architecture:
- ContourData: Immutable container for arrays and metadata
- ContourLayout: Manages viewport allocation
- ColourScale: Encapsulates colormap/level logic
- ContourRenderer: Base class for rendering strategy
- MapContourRenderer: Renders to map (Cartopy)
- XYContourRenderer: Renders to Cartesian axes

Module Independence:
This module is currently independent at the module level (no module-level
imports from cfplot.py), though functions do import locally from cfplot
for essential utilities like calculate_levels and axis helpers. This keeps
module boundaries clear while preserving functionality during gradual refactoring.

Future work will move more utilities (calculate_levels, _stimeaxis, etc.)
into standalone modules (see utility.py, state.py) and eliminate even the
function-level cfplot imports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import os

import cf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cartopy_util
import matplotlib.colors
import matplotlib.pyplot as plot
import numpy as np
from matplotlib.axes import Axes

from . import utility
from .blockfill import _bfill
from .colorbar import cbar
from .layout_runtime import (
    apply_axes,
    ensure_map_viewport,
    ensure_xy_viewport,
    set_plot_limits,
)
from .map_runtime import MapSet
from .state import (
    apply_colour_scale,
    get_colour_scale_map,
    global_blockfill,
    global_fill,
    global_lines,
    plotvars,
)


@dataclass(frozen=True)
class ContourData:
    """Read-only contour inputs after extraction and validation.
    
    Holds extracted, validated, and pre-processed arrays ready for rendering.
    Immutable by design to prevent unintended state mutations during plotting.
    """

    field: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    ptype: int = 0
    colorbar_title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    levels: np.ndarray | None = None
    mult: int = 0
    fmult: float = 1.0
    irregular: bool = False
    is_ugrid: bool = False
    is_orca: bool = False
    fill: bool = True
    lines: bool = True
    blockfill: bool = False
    xpole: float | None = None
    ypole: float | None = None

    @classmethod
    def from_cf_field(
        cls,
        f: cf.Field,
        colorbar_title: str | None,
        verbose: bool | None = None,
        proj: str = "cyl",
    ) -> "ContourData":
        """Extract and prepare CF field for contouring."""
        (
            field,
            x,
            y,
            ptype,
            cbar_title,
            xlabel,
            ylabel,
            xpole,
            ypole,
        ) = utility.cf_data_assign(f, colorbar_title, verbose=verbose, proj=proj)

        if colorbar_title is not None:
            cbar_title = colorbar_title

        return cls(
            field=np.asarray(field),
            x=x if x is None else np.asarray(x),
            y=y if y is None else np.asarray(y),
            ptype=ptype if ptype is not None else 0,
            colorbar_title=cbar_title or "",
            xlabel=xlabel or "",
            ylabel=ylabel or "",
            xpole=utility.to_float_or_none(xpole),
            ypole=utility.to_float_or_none(ypole),
        )

    @classmethod
    def from_arrays(
        cls,
        field: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "ContourData":
        """Create from raw numpy arrays with validation."""
        field = np.asarray(field)
        x = np.asarray(x) if x is not None else np.arange(field.shape[1])
        y = np.asarray(y) if y is not None else np.arange(field.shape[0])

        # Validate array dimensions - support both 1D coordinates and 2D (e.g., ORCA grids)
        if field.ndim not in (1, 2, 3):
            raise ValueError(f"Field must be 1D, 2D, or 3D, got shape {field.shape}")

        return cls(
            field=field,
            x=x,
            y=y,
            ptype=0,
            colorbar_title="",
            xlabel="",
            ylabel="",
        )


class ContourLayout:
    """Manage viewport and annotation geometry for contour plots.
    
    Separates concerns: layout calculates space, rendering uses it.
    Currently still delegates to legacy gopen/gset/gpos system.
    """

    def __init__(self, plotvars: Any):
        self.viewport: Axes | None = None
        self.colorbar_ax: Axes | None = None
        self.title_ax: Axes | None = None
        self._plotvars = plotvars
        self.colorbar_orientation: str = "horizontal"
        self.colorbar_position: list[float] | None = None

    def allocate_xy_viewport(
        self,
        colorbar_orientation: str | None,
        colorbar_position: list[float] | None,
    ) -> "ContourLayout":
        """Reserve viewport for Cartesian/non-map rendering.
        
        Coordinates with plotvars for multi-plot grids.
        """
        # Set colorbar orientation  
        self.colorbar_orientation = colorbar_orientation or "horizontal"
        self.colorbar_position = colorbar_position

        ensure_xy_viewport()

        # Store reference to current axes/map for later use
        self.viewport = plotvars.plot

        return self

    def allocate_map_viewport(
        self,
        colorbar_orientation: str | None,
        colorbar_position: list[float] | None,
    ) -> "ContourLayout":
        """Reserve viewport for map rendering without map-axis operations.

        This intentionally keeps map setup in a dedicated flow where projection
        creation (mapset/_set_map) happens after base viewport selection.
        """
        self.colorbar_orientation = colorbar_orientation or "horizontal"
        self.colorbar_position = colorbar_position

        ensure_map_viewport()

        self.viewport = plotvars.plot

        return self

    def allocate(
        self,
        colorbar_orientation: str | None,
        colorbar_position: list[float] | None,
    ) -> "ContourLayout":
        """Backward-compatible alias for Cartesian viewport allocation."""
        return self.allocate_xy_viewport(colorbar_orientation, colorbar_position)

    def apply_title(
        self,
        title: str | None,
        dims_title: bool,
        fontsize: int | None,
        fontweight: str | None,
    ) -> None:
        """Apply title and dimension titles to plot."""
        pv = self._plotvars

        if title and title != "":
            if pv.plot_type == 1:
                _apply_map_title(
                    mymap=pv.mymap,
                    title=title,
                    proj=pv.proj,
                    boundinglat=pv.boundinglat,
                    lon_0=pv.lon_0,
                    lonmin=pv.lonmin,
                    lonmax=pv.lonmax,
                    latmin=pv.latmin,
                    latmax=pv.latmax,
                    title_fontsize=fontsize or pv.title_fontsize,
                    title_fontweight=fontweight or pv.title_fontweight,
                )
            else:
                if self.viewport:
                    self.viewport.set_title(
                        title,
                        y=1.03,
                        fontsize=fontsize or pv.title_fontsize,
                        fontweight=fontweight or pv.title_fontweight,
                    )

        if dims_title:
            _apply_dim_titles(
                plot=pv.plot,
                mymap=pv.mymap,
                plot_type=pv.plot_type,
                proj=pv.proj,
                lonmin=pv.lonmin,
                lonmax=pv.lonmax,
                latmin=pv.latmin,
                latmax=pv.latmax,
                axis_label_fontsize=pv.axis_label_fontsize,
                axis_label_fontweight=pv.axis_label_fontweight,
                title=dims_title if isinstance(dims_title, str) else None,
            )

    def apply_axis_labels(
        self,
        xlabel: str | None,
        ylabel: str | None,
        xticks: Any,
        yticks: Any,
        xticklabels: Any | None = None,
        yticklabels: Any | None = None,
    ) -> None:
        """Apply axis labels and ticks to plot."""
        if self.viewport is None:
            return

        apply_axes(
            plot_type=self._plotvars.plot_type,
            xticks=xticks,
            yticks=yticks,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )


class ColourScale:
    """Encapsulate level fitting, colormap selection, and cbar labels.
    
    Replaces the scattered cscale_flag (0/1/2) branching with explicit methods.
    """

    def __init__(self, plotvars: Any):
        self._plotvars = plotvars
        self._levels: np.ndarray | None = None
        self._includes_zero: bool = False
        self._levels_extend: str = "neither"

    def fit_to_levels(
        self,
        levels: np.ndarray,
        includes_zero: bool,
        levels_extend: str,
    ) -> "ColourScale":
        """Fit color scale to contour levels, handling zero if present."""
        self._levels = np.asarray(levels)
        self._includes_zero = includes_zero
        self._levels_extend = levels_extend

        # Replicate legacy cscale_flag == 0 logic (default colour scale).
        # If zero is present in levels, split scale1 around zero so
        # blue shades are strictly below zero and warm shades above.
        if self._plotvars.cscale_flag == 0:
            col_zero = 0
            includes_zero = False
            for cval in self._levels:
                if not includes_zero:
                    col_zero += 1
                if cval == 0:
                    includes_zero = True

            if includes_zero:
                cs_below = col_zero
                cs_above = np.size(self._levels) - col_zero + 1
                if self._plotvars.levels_extend in ("max", "neither"):
                    cs_below = cs_below - 1
                if self._plotvars.levels_extend in ("min", "neither"):
                    cs_above = cs_above - 1
                apply_colour_scale(
                    "scale1",
                    below=cs_below,
                    above=cs_above,
                    uniform=bool(self._plotvars.cs_uniform),
                )
            else:
                ncols = np.size(self._levels) + 1
                if self._plotvars.levels_extend in ("min", "max"):
                    ncols = ncols - 1
                elif self._plotvars.levels_extend == "neither":
                    ncols = ncols - 2
                apply_colour_scale("viridis", ncols=ncols)

            self._plotvars.cscale_flag = 0

        # Replicate cscale_flag == 1 logic (user-selected color map, fit to levels)
        if self._plotvars.cscale_flag == 1:
            ncols = np.size(self._levels) + 1
            if (
                self._plotvars.levels_extend == "min"
                or self._plotvars.levels_extend == "max"
            ):
                ncols = ncols - 1
            if self._plotvars.levels_extend == "neither":
                ncols = ncols - 2
            apply_colour_scale(self._plotvars.cs_user, ncols=ncols)
            self._plotvars.cscale_flag = 1

        return self

    def get_cmap(self) -> matplotlib.colors.ListedColormap:
        """Get colormap after fitting to levels."""
        colmap = get_colour_scale_map()
        cmap = matplotlib.colors.ListedColormap(colmap)

        if (
            self._plotvars.levels_extend == "min"
            or self._plotvars.levels_extend == "both"
        ):
            cmap.set_under(self._plotvars.cs[0])
        if (
            self._plotvars.levels_extend == "max"
            or self._plotvars.levels_extend == "both"
        ):
            cmap.set_over(self._plotvars.cs[-1])

        return cmap

    def colorbar_labels(
        self,
        levels: np.ndarray,
        orientation: str,
        n_columns: int,
        label_skip: int,
        custom_labels: list[str] | None,
    ) -> list[str]:
        """Generate colorbar labels from levels with skip/custom overrides."""
        if custom_labels is not None:
            return custom_labels

        # Legacy default: estimate skip for horizontal colour bars from the
        # total character count, and include fewer labels for readability.
        if label_skip is None:
            if orientation == "horizontal":
                nchars = sum(len(str(level)) for level in levels)
                label_skip = int(nchars / 80 + 1)
                if n_columns > 1:
                    label_skip = int(nchars * n_columns / 80)
            else:
                label_skip = 1

        if label_skip <= 1:
            return [str(level) for level in levels]

        if self._includes_zero:
            zero_positions = np.where(np.asarray(levels) == 0)[0]
            if np.size(zero_positions) > 0:
                zero_pos = int(zero_positions[0])
                labels = [levels[zero_pos]]

                i = zero_pos + label_skip
                while i <= len(levels) - 1:
                    labels = list(np.append(labels, levels[i]))
                    i += label_skip

                i = zero_pos - label_skip
                if i >= 0:
                    while i >= 0:
                        labels = list(np.append([levels[i]], labels))
                        i -= label_skip

                return [str(level) for level in labels]

        labels = [levels[0]]
        i = int(label_skip)
        while i <= len(levels) - 1:
            labels = list(np.append(labels, levels[i]))
            i += label_skip

        return [str(level) for level in labels]


class ContourRenderer:
    """Base renderer for shared contour drawing responsibilities."""

    def __init__(
        self,
        layout: ContourLayout,
        data: ContourData,
        colour_scale: ColourScale,
    ):
        self.layout = layout
        self.data = data
        self.cs = colour_scale
        self.frame_artists: list[Any] = []

    def render_filled(
        self, alpha: float, zorder: int, transform_first: bool | None
    ) -> None:
        """Render filled contours. Subclass implements plot-type-specific logic."""
        _ = (alpha, zorder, transform_first)

    def render_blockfill(
        self, fast: bool | None, alpha: float, zorder: int
    ) -> None:
        """Render block-filled contours."""
        _ = (fast, alpha, zorder)

    def render_lines(
        self,
        colors: Any,
        linewidths: Any,
        linestyles: Any,
        line_labels: bool,
        zero_thick: bool | int,
        zorder: int = 1,
    ) -> None:
        """Render contour lines and labels."""
        _ = (colors, linewidths, linestyles, line_labels, zero_thick, zorder)

    def render_colorbar(
        self,
        orientation: str | None,
        shrink: float | None,
        position: list[float] | None,
        fraction: float | None,
        thick: float | None,
        anchor: float | None,
        fontsize: int | None,
        fontweight: str | None,
        text_up_down: bool,
        text_down_up: bool,
        drawedges: bool,
        labels: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        """Render colorbar for filled contours."""
        _ = (
            orientation,
            shrink,
            position,
            fraction,
            thick,
            anchor,
            fontsize,
            fontweight,
            text_up_down,
            text_down_up,
            drawedges,
            labels,
            title,
        )


class MapContourRenderer(ContourRenderer):
    """Map renderer specialization for ptype == 1 (lon-lat plots).
    
    Handles Cartopy transformations, coastlines, and polar projections.
    """

    def render_filled(
        self, alpha: float, zorder: int, transform_first: bool | None
    ) -> None:
        """Render filled contours on a map with Cartopy."""
        if self.data.x is None or self.data.y is None or self.data.levels is None:
            return

        lons = self.data.x
        lats = self.data.y

        if transform_first is None and np.ndim(lons) == 1 and np.ndim(lats) == 1:
            if np.size(lons) >= 400:
                transform_first = True

        if transform_first and np.ndim(lons) == 1 and np.ndim(lats) == 1:
            lons, lats = np.meshgrid(lons, lats)

        cmap = self.cs.get_cmap()
        plotvars.image = plotvars.mymap.contourf(
            lons,
            lats,
            self.data.field * self.data.fmult,
            self.data.levels,
            extend=plotvars.levels_extend,
            cmap=cmap,
            norm=plotvars.norm,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
            transform_first=transform_first,
        )
        if hasattr(plotvars.image, "collections"):
            self.frame_artists.extend(list(plotvars.image.collections))

    def render_blockfill(
        self, fast: bool | None, alpha: float, zorder: int
    ) -> None:
        """Render block-filled contours on a map."""
        if self.data.x is None or self.data.y is None or self.data.levels is None:
            return

        _bfill(
            f=self.data.field * self.data.fmult,
            x=self.data.x,
            y=self.data.y,
            clevs=self.data.levels,
            lonlat=True,
            bound=0,
            alpha=alpha,
            fast=fast,
            zorder=zorder,
        )

    def render_lines(
        self,
        colors: Any,
        linewidths: Any,
        linestyles: Any,
        line_labels: bool,
        zero_thick: bool | int,
        zorder: int = 1,
    ) -> None:
        """Render contour lines on a map with Cartopy transform."""
        if self.data.x is None or self.data.y is None or self.data.levels is None:
            return

        cs = plotvars.mymap.contour(
            self.data.x,
            self.data.y,
            self.data.field * self.data.fmult,
            self.data.levels,
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )
        if hasattr(cs, "collections"):
            self.frame_artists.extend(list(cs.collections))

        if line_labels and not isinstance(self.data.levels, int):
            nd = utility.ndecs(self.data.levels)
            fmt = "%d"
            if nd != 0:
                fmt = "%1." + str(nd) + "f"
            plotvars.plot.clabel(
                cs,
                levels=self.data.levels,
                fmt=fmt,
                colors=colors,
                fontsize=plotvars.text_fontsize,
                zorder=zorder,
            )

        if zero_thick:
            cs0 = plotvars.mymap.contour(
                self.data.x,
                self.data.y,
                self.data.field * self.data.fmult,
                [-1e-32, 0],
                colors=colors,
                linewidths=zero_thick,
                linestyles=linestyles,
                alpha=1.0,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )
            if hasattr(cs0, "collections"):
                self.frame_artists.extend(list(cs0.collections))

    def render_colorbar(
        self,
        orientation: str | None,
        shrink: float | None,
        position: list[float] | None,
        fraction: float | None,
        thick: float | None,
        anchor: float | None,
        fontsize: int | None,
        fontweight: str | None,
        text_up_down: bool,
        text_down_up: bool,
        drawedges: bool,
        labels: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        """Render colorbar for map contour plots."""
        if self.data.levels is None:
            return

        cbar(
            labels=labels,
            orientation=orientation,
            position=position,
            shrink=shrink,
            title=title or self.data.colorbar_title,
            fontsize=fontsize,
            fontweight=fontweight,
            text_up_down=text_up_down,
            text_down_up=text_down_up,
            drawedges=drawedges,
            fraction=fraction,
            thick=thick,
            levs=self.data.levels,
            anchor=anchor,
        )


class XYContourRenderer(ContourRenderer):
    """Cartesian renderer specialization for non-map contour plots.
    
    Handles ptypes 0, 2-7 (simple XY, lat-height, lon-height, Hovmuller, rotated).
    """

    def render_filled(
        self, alpha: float, zorder: int, transform_first: bool | None
    ) -> None:
        """Render filled contours in Cartesian space."""
        _ = transform_first
        if self.data.x is None or self.data.y is None or self.data.levels is None:
            return

        cmap = self.cs.get_cmap()
        plotvars.image = plotvars.plot.contourf(
            self.data.x,
            self.data.y,
            self.data.field * self.data.fmult,
            self.data.levels,
            extend=plotvars.levels_extend,
            cmap=cmap,
            norm=plotvars.norm,
            alpha=alpha,
            zorder=zorder,
        )

    def render_blockfill(
        self, fast: bool | None, alpha: float, zorder: int
    ) -> None:
        """Render block-filled contours in Cartesian space."""
        if self.data.x is None or self.data.y is None or self.data.levels is None:
            return

        _bfill(
            f=self.data.field * self.data.fmult,
            x=self.data.x,
            y=self.data.y,
            clevs=self.data.levels,
            lonlat=False,
            bound=0,
            alpha=alpha,
            fast=fast,
            zorder=zorder,
        )

    def render_lines(
        self,
        colors: Any,
        linewidths: Any,
        linestyles: Any,
        line_labels: bool,
        zero_thick: bool | int,
        zorder: int = 1,
    ) -> None:
        """Render contour lines in Cartesian space."""
        if self.data.x is None or self.data.y is None or self.data.levels is None:
            return

        cs = plotvars.plot.contour(
            self.data.x,
            self.data.y,
            self.data.field * self.data.fmult,
            self.data.levels,
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            zorder=zorder,
        )
        if line_labels and not isinstance(self.data.levels, int):
            nd = utility.ndecs(self.data.levels)
            fmt = "%d"
            if nd != 0:
                fmt = "%1." + str(nd) + "f"
            plotvars.plot.clabel(
                cs,
                fmt=fmt,
                colors=colors,
                fontsize=plotvars.text_fontsize,
                zorder=zorder,
            )

        if zero_thick:
            plotvars.plot.contour(
                self.data.x,
                self.data.y,
                self.data.field * self.data.fmult,
                [-1e-32, 0],
                colors=colors,
                linewidths=zero_thick,
                linestyles=linestyles,
                alpha=1.0,
                zorder=zorder,
            )

    def render_colorbar(
        self,
        orientation: str | None,
        shrink: float | None,
        position: list[float] | None,
        fraction: float | None,
        thick: float | None,
        anchor: float | None,
        fontsize: int | None,
        fontweight: str | None,
        text_up_down: bool,
        text_down_up: bool,
        drawedges: bool,
        labels: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        """Render colorbar for Cartesian contour plots."""
        if self.data.levels is None:
            return

        cbar(
            labels=labels,
            orientation=orientation,
            position=position,
            shrink=shrink,
            title=title or self.data.colorbar_title,
            fontsize=fontsize,
            fontweight=fontweight,
            text_up_down=text_up_down,
            text_down_up=text_down_up,
            drawedges=drawedges,
            fraction=fraction,
            thick=thick,
            levs=self.data.levels,
            anchor=anchor,
        )


def _add_cyclic(field: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add a cyclic longitude column if the grid doesn't span the full 360°."""
    try:
        return cartopy_util.add_cyclic_point(field, lons)
    except Exception:
        # Promote to float64 and round to handle uneven spacing from numpy rounding.
        ndecs_max = max(
            len(str(float(v)).split(".")[-1].rstrip("0") or "0") for v in lons
        )
        lons64 = np.float64(lons).round(ndecs_max)
        return cartopy_util.add_cyclic_point(field, lons64)


def _can_use_new_xy_path(f: Any, kwargs: dict[str, Any]) -> bool:
    """Return True when the new XY renderer can safely handle this call."""
    unsupported = (
        "irregular",
        "face_lons",
        "face_lats",
        "face_connectivity",
        "orca",
        "swap_axes",
        "xlog",
        "transform_first",
    )
    for key in unsupported:
        if kwargs.get(key):
            return False

    ptype = kwargs.get("ptype", 0)
    if not isinstance(f, cf.Field) and ptype not in (0, 1, None):
        return False

    return True
def _apply_map_title(
    *,
    mymap: Any,
    title: str,
    proj: str,
    boundinglat: float,
    lon_0: float,
    lonmin: float,
    lonmax: float,
    latmin: float,
    latmax: float,
    title_fontsize: int,
    title_fontweight: str,
) -> None:
    """Draw a title on a map axes at the geographically correct position."""
    polar_range = 90 - abs(boundinglat)
    myprojs = ["cyl", "robin", "moll", "merc"]

    if proj in myprojs:
        lon_mid = lonmin + (lonmax - lonmin) / 2.0
        projs = [ccrs.PlateCarree, ccrs.Robinson, ccrs.Mollweide, ccrs.Mercator]
        myind = myprojs.index(proj)
        map_proj = projs[myind](central_longitude=lon_mid)
        xpt, ypt = map_proj.transform_point(lon_mid, latmax, ccrs.PlateCarree())
        ypt = ypt + (latmax - latmin) / 40.0
    elif proj == "npstere":
        mylon = lon_0 + 180
        mylat = boundinglat - polar_range / 15.0
        map_proj = ccrs.NorthPolarStereo(central_longitude=lon_0)
        xpt, ypt = map_proj.transform_point(mylon, mylat, ccrs.PlateCarree())
    elif proj == "spstere":
        mylon = lon_0
        mylat = boundinglat + polar_range / 15.0
        map_proj = ccrs.SouthPolarStereo(central_longitude=lon_0)
        xpt, ypt = map_proj.transform_point(mylon, mylat, ccrs.PlateCarree())
    elif proj == "lcc":
        lon_mid = lonmin + (lonmax - lonmin) / 2.0
        lat_0 = 40
        if latmin <= 0 and latmax <= 0:
            lat_0 = 40
        map_proj = ccrs.LambertConformal(
            central_longitude=lon_0,
            central_latitude=lat_0,
            cutoff=latmin,
        )
        xpt, ypt = map_proj.transform_point(lon_mid, latmax, ccrs.PlateCarree())
    else:
        return

    mymap.text(
        xpt,
        ypt,
        title,
        va="bottom",
        ha="center",
        rotation="horizontal",
        rotation_mode="anchor",
        fontsize=title_fontsize,
        fontweight=title_fontweight,
    )


def _apply_dim_titles(
    *,
    plot: Any,
    mymap: Any,
    plot_type: int,
    proj: str,
    lonmin: float,
    lonmax: float,
    latmin: float,
    latmax: float,
    axis_label_fontsize: int,
    axis_label_fontweight: str,
    title: str | None = None,
    title2: str | None = None,
    title3: str | None = None,
) -> None:
    """Draw a set of dimension titles on the active contour axes."""
    this_plot = mymap if plot_type == 1 else plot

    left, bottom, width, height = this_plot.get_position().bounds
    valign = "bottom"

    if plot_type == 1 and proj != "cyl":
        left -= 0.1
        myx = 1.25
        myy = 1.0
        valign = "top"
        if title3 is None:
            myx = 1.05
    elif plot_type == 1 and proj == "cyl":
        lonrange = lonmax - lonmin
        latrange = latmax - latmin
        if (lonrange / latrange) > 1.5:
            myx = 0.0
            myy = 1.02
        elif (lonrange / latrange) > 1.2:
            myx = 0.0
            myy = 1.02
            height -= 0.015
        else:
            left -= 0.1
            myx = 1.05
            myy = 1.0
            width -= 0.1
            valign = "top"
    else:
        height -= 0.1
        myx = 0.0
        myy = 1.02

    if title3 is None:
        this_plot.set_position([left, bottom, width, height])

    xspacing = 0.3
    yspacing = 0.0
    if myx in (1.05, 1.25):
        xspacing = 0.0
        yspacing = 0.2

    if title is not None:
        this_plot.text(
            myx,
            myy,
            title,
            va=valign,
            ha="left",
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
            transform=this_plot.transAxes,
        )

    if title2 is not None:
        this_plot.text(
            myx + xspacing,
            myy - yspacing,
            title2,
            va=valign,
            ha="left",
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
            transform=this_plot.transAxes,
        )

    if title3 is not None:
        this_plot.text(
            myx + xspacing * 2,
            myy - yspacing * 2,
            title3,
            va=valign,
            ha="left",
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
            transform=this_plot.transAxes,
        )


def _clear_animation_artists(plotvars: Any) -> None:
    """Remove artists from previous animation frame if present."""
    artists = getattr(plotvars, "_contour_animation_artists", None)
    if not artists:
        return
    for artist in artists:
        try:
            artist.remove()
        except Exception:
            continue
    plotvars._contour_animation_artists = []


def _rotated_vloc(
    *, lons: np.ndarray, lats: np.ndarray, xvec: np.ndarray, yvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map rotated-grid lon/lat points into plot coordinates.

    This mirrors the legacy location helper but uses safer index lookup so
    empty intersections do not raise during rotated projection plotting.
    """
    if any(val is None for val in [xvec, yvec, lons, lats]):
        errstr = (
            "\nvloc error\n"
            "xvec, yvec, lons, lats all need to be passed to vloc to\n"
            "generate a set of location points\n"
        )
        raise Warning(errstr)

    xvec = np.asarray(xvec, dtype=float).copy()
    yvec = np.asarray(yvec, dtype=float).copy()
    lons = np.asarray(lons, dtype=float).copy()
    lats = np.asarray(lats, dtype=float).copy()

    xarr = np.zeros(np.size(lons))
    yarr = np.zeros(np.size(lats))

    for i in np.arange(np.size(xvec)):
        xvec[i] = ((xvec[i] + 180) % 360) - 180
    for i in np.arange(np.size(lons)):
        lons[i] = ((lons[i] + 180) % 360) - 180

    if np.nanmax(xvec) > 150:
        for i in np.arange(np.size(xvec)):
            xvec[i] = (xvec[i] + 360.0) % 360.0
        pts = np.where(xvec < 0.0)
        xvec[pts] = xvec[pts] + 360.0
        for i in np.arange(np.size(lons)):
            lons[i] = (lons[i] + 360.0) % 360.0
        pts = np.where(lons < 0.0)
        lons[pts] = lons[pts] + 360.0

    for i in np.arange(np.size(lons)):
        if (lons[i] < np.min(xvec)) or (lons[i] > np.max(xvec)):
            xarr[i] = np.nan
        else:
            xpt = int(np.searchsorted(xvec, lons[i], side="right") - 1)
            xpt = max(0, min(xpt, np.size(xvec) - 2))
            xarr[i] = xpt + (lons[i] - xvec[xpt]) / (xvec[xpt + 1] - xvec[xpt])

        if (lats[i] < np.min(yvec)) or (lats[i] > np.max(yvec)):
            yarr[i] = np.nan
        else:
            ypt = int(np.searchsorted(yvec, lats[i], side="right") - 1)
            ypt = max(0, min(ypt, np.size(yvec) - 2))
            yarr[i] = ypt + (lats[i] - yvec[ypt]) / (yvec[ypt + 1] - yvec[ypt])

    return (xarr, yarr)


def _render_rotated_grid_axes(
    *,
    xpole: float,
    ypole: float,
    xvec: np.ndarray,
    yvec: np.ndarray,
    xticks: Any = None,
    xticklabels: Any = None,
    yticks: Any = None,
    yticklabels: Any = None,
    axes: bool = True,
    xaxis: bool = True,
    yaxis: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Draw rotated-grid axes and optional graticule labels."""
    import matplotlib.lines

    spacing = plotvars.rotated_grid_spacing
    degspacing = plotvars.rotated_deg_spacing
    continents = plotvars.rotated_continents
    grid = plotvars.rotated_grid
    labels = plotvars.rotated_labels
    grid_thickness = plotvars.rotated_grid_thickness

    yvec = np.asarray(yvec)
    xvec = np.asarray(xvec)
    if yvec[0] > yvec[np.size(yvec) - 1]:
        yvec = yvec[::-1]

    set_plot_limits(
        xmin=0,
        xmax=float(np.size(xvec) - 1),
        ymin=0,
        ymax=float(np.size(yvec) - 1),
        ylog=False,
        user_gset=0,
    )

    if continents:
        import cartopy.io.shapereader as shpreader
        import shapefile

        shpfilename = shpreader.natural_earth(
            resolution=plotvars.resolution,
            category="physical",
            name="coastline",
        )
        reader = shapefile.Reader(shpfilename)
        shapes = [s.points for s in reader.shapes()]
        for shape in shapes:
            lons, lats = list(zip(*shape))
            lons = np.array(lons)
            lats = np.array(lats)
            rotated_transform = ccrs.RotatedPole(
                pole_latitude=ypole, pole_longitude=xpole
            )
            points = rotated_transform.transform_points(
                ccrs.PlateCarree(), lons, lats
            )
            xout = np.array(points)[:, 0]
            yout = np.array(points)[:, 1]
            xpts, ypts = _rotated_vloc(lons=xout, lats=yout, xvec=xvec, yvec=yvec)
            plotvars.plot.plot(
                xpts,
                ypts,
                linewidth=plotvars.continent_thickness or 1.5,
                color=plotvars.continent_color or "k",
            )

    if xticks is None:
        lons = -180 + np.arange(360 / spacing + 1) * spacing
    else:
        lons = xticks
    if yticks is None:
        lats = -90 + np.arange(180 / spacing + 1) * spacing
    else:
        lats = yticks

    xlim = plotvars.plot.get_xlim()
    spacing_x = (xlim[1] - xlim[0]) / 20
    ylim = plotvars.plot.get_ylim()
    spacing_y = (ylim[1] - ylim[0]) / 20
    spacing = min(spacing_x, spacing_y)

    rotated_transform = ccrs.RotatedPole(pole_latitude=ypole, pole_longitude=xpole)

    if axes:
        if xaxis:
            for val in np.arange(np.size(lons)):
                ipts = max(2, int(179.0 / degspacing))
                lona = np.zeros(ipts) + lons[val]
                lata = -90 + np.arange(ipts) * degspacing
                points = rotated_transform.transform_points(
                    ccrs.PlateCarree(), lona, lata
                )
                xout = np.array(points)[:, 0]
                yout = np.array(points)[:, 1]
                xpts, ypts = _rotated_vloc(lons=xout, lats=yout, xvec=xvec, yvec=yvec)
                if grid:
                    plotvars.plot.plot(
                        xpts, ypts, ":", linewidth=grid_thickness, color="k"
                    )

                if labels and np.size(ypts[5:]) > np.sum(np.isnan(ypts[5:])):
                    ymin = np.nanmin(ypts[5:])
                    loc = np.where(ypts == ymin)[0]
                    if np.size(loc) > 1:
                        loc = loc[1]
                    if loc > 0 and np.isfinite(xpts[loc]):
                        xpos = float(np.asarray(xpts[loc]).reshape(-1)[0])
                        line = matplotlib.lines.Line2D(
                            [xpos, xpos], [0, -spacing / 2], color="k"
                        )
                        plotvars.plot.add_line(line)
                        line.set_clip_on(False)
                        xticklabel = (
                            utility.mapaxis(lons[val], lons[val], axis_type=1, degsym=plotvars.degsym)[1][0]
                            if xticklabels is None
                            else xticklabels[val]
                        )
                        plotvars.plot.text(
                            xpos,
                            -spacing,
                            xticklabel,
                            horizontalalignment="center",
                            verticalalignment="top",
                            fontsize=plotvars.text_fontsize,
                            fontweight=plotvars.text_fontweight,
                        )

        if yaxis:
            for val in np.arange(np.size(lats)):
                ipts = max(2, int(359.0 / degspacing))
                lata = np.zeros(ipts) + lats[val]
                lona = -180.0 + np.arange(ipts) * degspacing
                points = rotated_transform.transform_points(
                    ccrs.PlateCarree(), lona, lata
                )
                xout = np.array(points)[:, 0]
                yout = np.array(points)[:, 1]
                xpts, ypts = _rotated_vloc(lons=xout, lats=yout, xvec=xvec, yvec=yvec)

                if grid:
                    plotvars.plot.plot(
                        xpts, ypts, ":", linewidth=grid_thickness, color="k"
                    )

                if labels and np.size(xpts[5:]) > np.sum(np.isnan(xpts[5:])):
                    xmin = np.nanmin(xpts[5:])
                    loc = np.where(xpts == xmin)[0]
                    if np.size(loc) == 1 and loc > 0 and np.isfinite(ypts[loc]):
                        ypos = float(np.asarray(ypts[loc]).reshape(-1)[0])
                        line = matplotlib.lines.Line2D(
                            [0, -spacing / 2], [ypos, ypos], color="k"
                        )
                        plotvars.plot.add_line(line)
                        line.set_clip_on(False)
                        yticklabel = (
                            utility.mapaxis(lats[val], lats[val], axis_type=2, degsym=plotvars.degsym)[1][0]
                            if yticklabels is None
                            else yticklabels[val]
                        )
                        plotvars.plot.text(
                            -spacing,
                            ypos,
                            yticklabel,
                            horizontalalignment="right",
                            verticalalignment="center",
                            fontsize=plotvars.text_fontsize,
                            fontweight=plotvars.text_fontweight,
                        )

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


def _render_ptype6_rotated_pole(
    *,
    f: Any,
    data: ContourData,
    kwargs: dict[str, Any],
    clevs: np.ndarray,
    cs: ColourScale,
    cbar_labels: list[str] | Any,
    colorbar_title: str,
    fill: bool,
    lines: bool,
    blockfill: bool,
    line_labels: bool,
    zero_thick: bool | int,
    colors: Any,
    linewidths: Any,
    linestyles: Any,
    alpha: float,
    zorder: int,
) -> bool:
    """Render ptype 6 (rotated pole) for cylindrical transformed-map mode."""

    if data.x is None or data.y is None or data.levels is None:
        return False

    if plotvars.user_plot == 0:
        ensure_xy_viewport()

    rotated_pole = f.ref("grid_mapping_name:rotated_latitude_longitude", default=None)
    xpole = ypole = None
    transform = None
    if rotated_pole:
        xpole = utility.to_float_or_none(rotated_pole.get("grid_north_pole_longitude"))
        ypole = utility.to_float_or_none(rotated_pole.get("grid_north_pole_latitude"))

    if plotvars.proj == "rotated":
        xpts = np.arange(np.size(data.x))
        ypts = np.arange(np.size(data.y))
        plotargs: dict[str, Any] = {}
        plot = plotvars.plot
        set_plot_limits(
            xmin=0,
            xmax=float(np.size(xpts) - 1),
            ymin=0,
            ymax=float(np.size(ypts) - 1),
            ylog=False,
            user_gset=plotvars.user_gset,
        )
    elif plotvars.proj == "cyl":
        xpts = data.x
        ypts = data.y

        if not rotated_pole:
            return False
        if xpole is None or ypole is None:
            return False

        transform = ccrs.RotatedPole(pole_latitude=ypole, pole_longitude=xpole)

        map_runtime = MapSet(plotvars)
        if plotvars.user_mapset != 1:
            if np.ndim(xpts) == 1:
                lonpts, latpts = np.meshgrid(xpts, ypts)
            else:
                lonpts = xpts
                latpts = ypts
            points = ccrs.PlateCarree().transform_points(
                transform, lonpts.flatten(), latpts.flatten()
            )
            lons = np.array(points)[:, 0]
            lats = np.array(points)[:, 1]

            map_runtime.configure(
                lonmin=float(np.min(lons)),
                lonmax=float(np.max(lons)),
                latmin=float(np.min(lats)),
                latmax=float(np.max(lats)),
                user_mapset=0,
                resolution=plotvars.resolution,
            )
        map_runtime.ensure_map_axes()

        plotargs = {"transform": transform}
        plot = plotvars.mymap
    else:
        return False

    frame_artists: list[Any] = []

    if fill:
        cmap = cs.get_cmap()
        cset = plot.contourf(
            xpts,
            ypts,
            data.field * data.fmult,
            clevs,
            extend=plotvars.levels_extend,
            cmap=cmap,
            norm=plotvars.norm,
            alpha=alpha,
            zorder=zorder,
            **plotargs,
        )
        if hasattr(cset, "collections"):
            frame_artists.extend(list(cset.collections))

    if blockfill:
        _bfill(
            f=data.field * data.fmult,
            x=xpts,
            y=ypts,
            clevs=clevs,
            lonlat=False,
            bound=0,
            alpha=alpha,
            fast=kwargs.get("blockfill_fast", None),
            zorder=zorder,
            transform=transform,
        )

    if lines:
        cs_lines = plot.contour(
            xpts,
            ypts,
            data.field * data.fmult,
            clevs,
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            zorder=zorder,
            **plotargs,
        )
        if hasattr(cs_lines, "collections"):
            frame_artists.extend(list(cs_lines.collections))
        if line_labels and not isinstance(clevs, int):
            nd = utility.ndecs(clevs)
            fmt = "%d"
            if nd != 0:
                fmt = "%1." + str(nd) + "f"
            plot.clabel(
                cs_lines,
                fmt=fmt,
                colors=colors,
                zorder=zorder,
                fontsize=plotvars.text_fontsize,
            )
        if zero_thick:
            cs0 = plot.contour(
                xpts,
                ypts,
                data.field * data.fmult,
                [-1e-32, 0],
                colors=colors,
                linewidths=zero_thick,
                linestyles=linestyles,
                alpha=alpha,
                zorder=zorder,
                **plotargs,
            )
            if hasattr(cs0, "collections"):
                frame_artists.extend(list(cs0.collections))

    if kwargs.get("axes", True):
        if plotvars.proj == "cyl":
            apply_axes(
                plot_type=1,
                xticks=kwargs.get("xticks", None),
                yticks=kwargs.get("yticks", None),
                xlabel=kwargs.get("xlabel", None),
                ylabel=kwargs.get("ylabel", None),
                xticklabels=kwargs.get("xticklabels", None),
                yticklabels=kwargs.get("yticklabels", None),
            )
        else:
                _render_rotated_grid_axes(
                xpole=xpole,
                ypole=ypole,
                xvec=data.x,
                yvec=data.y,
                xticks=kwargs.get("xticks", None),
                xticklabels=kwargs.get("xticklabels", None),
                yticks=kwargs.get("yticks", None),
                yticklabels=kwargs.get("yticklabels", None),
                axes=True,
                xaxis=kwargs.get("xaxis", True),
                yaxis=kwargs.get("yaxis", True),
                xlabel=kwargs.get("xlabel", None),
                ylabel=kwargs.get("ylabel", None),
            )

    if kwargs.get("colorbar", True) and (fill or blockfill):
        cbar(
            labels=cbar_labels,
            orientation=kwargs.get("colorbar_orientation", None) or "horizontal",
            position=kwargs.get("colorbar_position", None),
            shrink=kwargs.get("colorbar_shrink", None),
            title=colorbar_title,
            fontsize=kwargs.get("colorbar_fontsize", None),
            fontweight=kwargs.get("colorbar_fontweight", None),
            text_up_down=kwargs.get("colorbar_text_up_down", False),
            text_down_up=kwargs.get("colorbar_text_down_up", False),
            drawedges=kwargs.get("colorbar_drawedges", True),
            fraction=kwargs.get("colorbar_fraction", None),
            thick=kwargs.get("colorbar_thick", None),
            levs=clevs,
            anchor=kwargs.get("colorbar_anchor", None),
            verbose=kwargs.get("verbose", None),
        )

    title = kwargs.get("title", "") or ""
    if title != "":
        _apply_map_title(
            mymap=plotvars.mymap,
            title=title,
            proj=plotvars.proj,
            boundinglat=plotvars.boundinglat,
            lon_0=plotvars.lon_0,
            lonmin=plotvars.lonmin,
            lonmax=plotvars.lonmax,
            latmin=plotvars.latmin,
            latmax=plotvars.latmax,
            title_fontsize=plotvars.title_fontsize,
            title_fontweight=plotvars.title_fontweight,
        )

    file = plotvars.file
    session_open = bool(getattr(plotvars, "_contour_session_open", False))
    if file is not None and not session_open:
        saveargs = {}
        if plotvars.tight:
            saveargs = {"bbox_inches": "tight"}
        if os.path.splitext(file)[1].lower() not in (".ps", ".eps", ".png", ".pdf"):
            file = file + ".png"
        figure = plotvars.master_plot or getattr(plotvars.plot, "figure", None)
        if figure is not None:
            figure.savefig(
                file,
                orientation=plotvars.orientation,
                dpi=plotvars.dpi,
                **saveargs,
            )
            import matplotlib.pyplot as _plt

            _plt.close(figure)

    plotvars._contour_animation_artists = frame_artists
    return True


def _render_with_new_xy(f: Any, x: Any, y: Any, kwargs: dict[str, Any]) -> bool:
    """Attempt rendering via new XY renderer and return True on success.
    
    Note: Imports from cfplot are local (inside function) to maintain
    module-level independence while preserving current functionality.
    """
    if isinstance(f, cf.Field) and (x is not None or y is not None):
        field_arr = np.asarray(f.array)
        x_arr = np.asarray(x.array) if isinstance(x, cf.Field) else x
        y_arr = np.asarray(y.array) if isinstance(y, cf.Field) else y
        data = ContourData.from_arrays(
            field=field_arr,
            x=None if x_arr is None else np.asarray(x_arr),
            y=None if y_arr is None else np.asarray(y_arr),
        )
        data = replace(data, ptype=kwargs.get("ptype", 0) or 0)
    elif isinstance(f, cf.Field):
        # Legacy parity: when mapset is user-defined for polar stereographic
        # plots, subset latitude before data extraction and level generation.
        if plotvars.user_mapset:
            if plotvars.proj == "npstere":
                f = f.subspace(Y=cf.wi(plotvars.boundinglat, 90.0))
            elif plotvars.proj == "spstere":
                f = f.subspace(Y=cf.wi(-90.0, plotvars.boundinglat))

        data = ContourData.from_cf_field(
            f=f,
            colorbar_title=kwargs.get("colorbar_title", None),
            verbose=kwargs.get("verbose", None),
            proj=getattr(plotvars, "proj", "cyl"),
        )
        # Implemented CF extraction targets include generic Cartesian (ptype 0),
        # map, and selected non-map ptypes.
        if data.ptype not in (0, 1, 2, 3, 4, 5, 6):
            return False
    else:
        data = ContourData.from_arrays(field=np.asarray(f), x=x, y=y)
        data = replace(data, ptype=kwargs.get("ptype", 0) or 0)

    # Keep legacy behavior for axis-routing logic by setting active plot type.
    plotvars.plot_type = data.ptype

    fill = kwargs.get("fill", global_fill)
    lines = kwargs.get("lines", global_lines)
    blockfill = kwargs.get("blockfill", global_blockfill)
    line_labels = kwargs.get("line_labels", True)
    zero_thick = kwargs.get("zero_thick", False)
    colors = kwargs.get("colors", "k")
    linewidths = kwargs.get("linewidths", None)
    linestyles = kwargs.get("linestyles", None)
    alpha = kwargs.get("alpha", 1.0)
    zorder = kwargs.get("zorder", 1)

    if blockfill:
        fill = False

    colorbar = kwargs.get("colorbar", True)
    if not fill and not blockfill:
        colorbar = False

    if plotvars.levels is None:
        clevs, mult, fmult = utility.calculate_levels(
            field=data.field,
            level_spacing=kwargs.get("level_spacing", "linear"),
            levels_step=plotvars.levels_step,
            verbose=kwargs.get("verbose", None),
        )
    else:
        clevs = np.asarray(plotvars.levels)
        mult = 0
        fmult = 1

    cs = ColourScale(plotvars).fit_to_levels(
        levels=np.asarray(clevs),
        includes_zero=bool(np.any(np.asarray(clevs) == 0)),
        levels_extend=plotvars.levels_extend,
    )

    import matplotlib
    matplotlib.rcParams["contour.negative_linestyle"] = "solid"

    _cb_orient = kwargs.get("colorbar_orientation", None)
    if _cb_orient is None:
        if data.ptype == 1 and plotvars.proj in ("npstere", "spstere"):
            _cb_orient = "vertical"
        else:
            _cb_orient = "horizontal"
    colorbar_orientation = _cb_orient
    colorbar_label_skip = kwargs.get("colorbar_label_skip", None)

    if colorbar_label_skip is None:
        if colorbar_orientation == "horizontal":
            nchars = 0
            for lev in clevs:
                nchars = nchars + len(str(lev))
            colorbar_label_skip = int(nchars / 80 + 1)
            if plotvars.columns > 1:
                colorbar_label_skip = int(nchars * (plotvars.columns) / 80)
        else:
            colorbar_label_skip = 1

    includes_zero = bool(np.any(np.asarray(clevs) == 0))
    if colorbar_label_skip > 1:
        if includes_zero:
            zero_pos = [i for i, mylev in enumerate(clevs) if mylev == 0][0]
            cbar_levels = clevs[zero_pos]
            i = zero_pos + colorbar_label_skip
            while i <= len(clevs) - 1:
                cbar_levels = np.append(cbar_levels, clevs[i])
                i = i + colorbar_label_skip
            i = zero_pos - colorbar_label_skip
            if i >= 0:
                while i >= 0:
                    cbar_levels = np.append(clevs[i], cbar_levels)
                    i = i - colorbar_label_skip
        else:
            cbar_levels = clevs[0]
            i = int(colorbar_label_skip)
            while i <= len(clevs) - 1:
                cbar_levels = np.append(cbar_levels, clevs[i])
                i = i + colorbar_label_skip
    else:
        cbar_levels = clevs

    if colorbar_label_skip is None:
        colorbar_label_skip = 1

    clabels = []
    for lev in np.atleast_1d(cbar_levels):
        clabels.append(str(lev))
        if colorbar_label_skip > 1:
            for _ in np.arange(colorbar_label_skip - 1):
                clabels.append("")

    cbar_labels = kwargs.get("colorbar_labels")
    if cbar_labels is None:
        cbar_labels = clabels

    colorbar_title = kwargs.get("colorbar_title", data.colorbar_title)
    if mult != 0:
        colorbar_title = f"{colorbar_title} *10^{{{mult}}}"

    # ptype 6 has its own rendering/axes flow and must bypass generic XY
    # layout to avoid non-map axes state assumptions.
    if data.ptype == 6:
        data = replace(
            data,
            levels=np.asarray(clevs),
            mult=mult,
            fmult=fmult,
            fill=fill,
            lines=lines,
            blockfill=blockfill,
        )
        return _render_ptype6_rotated_pole(
            f=f,
            data=data,
            kwargs=kwargs,
            clevs=np.asarray(clevs),
            cs=cs,
            cbar_labels=cbar_labels,
            colorbar_title=colorbar_title,
            fill=fill,
            lines=lines,
            blockfill=blockfill,
            line_labels=line_labels,
            zero_thick=zero_thick,
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            alpha=alpha,
            zorder=zorder,
        )

    if plotvars.user_plot == 0:
        ensure_xy_viewport()

    xmin = kwargs.get("xmin", float(np.nanmin(data.x)))
    xmax = kwargs.get("xmax", float(np.nanmax(data.x)))
    ymin = kwargs.get("ymin", float(np.nanmin(data.y)))
    ymax = kwargs.get("ymax", float(np.nanmax(data.y)))

    # Legacy parity for latitude/longitude/time-pressure plots:
    # pressure-like coordinates are rendered with pressure decreasing upward.
    if data.ptype in (2, 3, 7) and kwargs.get("user_gset", plotvars.user_gset) == 0:
        positive = "down"
        if isinstance(f, cf.Field):
            myz = utility.find_z(f)
            if myz is not None and hasattr(f.construct(myz), "positive"):
                positive = f.construct(myz).positive
        if "theta" in (data.ylabel or "").split(" "):
            positive = "up"
        if "height" in (data.ylabel or "").split(" "):
            positive = "up"

        if data.ptype == 2:
            if xmin < -80 and xmin >= -90:
                xmin = -90
            if xmax > 80 and xmax <= 90:
                xmax = 90

        if positive == "down":
            ymin = float(np.nanmax(data.y))
            ymax = float(np.nanmin(data.y))
            if ymax < 10:
                ymax = 0
        else:
            ymin = float(np.nanmin(data.y))
            ymax = float(np.nanmax(data.y))

    # Respect user gset for Hovmuller plots with date-string y limits.
    tmin = None
    tmax = None
    if isinstance(f, cf.Field) and data.ptype in (4, 5):
        if all(
            val is not None
            for val in [plotvars.xmin, plotvars.xmax, plotvars.ymin, plotvars.ymax]
        ):
            tmin = plotvars.ymin
            tmax = plotvars.ymax
            xmin = plotvars.xmin
            xmax = plotvars.xmax

            ref_time = f.construct("T").units
            ref_calendar = f.construct("T").calendar
            time_units = cf.Units(ref_time, ref_calendar)
            t = cf.Data(cf.dt(plotvars.ymin), units=time_units)
            ymin = t.array
            t = cf.Data(cf.dt(plotvars.ymax), units=time_units)
            ymax = t.array

    if kwargs.get("ylog", False) and ymax == 0:
        ymax = 1
    set_plot_limits(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        ylog=bool(kwargs.get("ylog", False)),
        user_gset=kwargs.get("user_gset", plotvars.user_gset),
    )

    if tmin is not None and tmax is not None:
        plotvars.ymin = tmin
        plotvars.ymax = tmax

    xticks = kwargs.get("xticks", None)
    yticks = kwargs.get("yticks", None)
    xticklabels = kwargs.get("xticklabels", None)
    yticklabels = kwargs.get("yticklabels", None)

    default_xlabel = data.xlabel or ""
    default_ylabel = data.ylabel or ""

    if data.ptype == 1:
        map_runtime = MapSet(plotvars)

        animation = bool(kwargs.get("animation", False))
        reuse_map_background = bool(kwargs.get("reuse_map_background", False))
        clear_previous_frame = bool(kwargs.get("clear_previous_frame", False))
        draw_static_map = not (animation and reuse_map_background)

        if clear_previous_frame:
            _clear_animation_artists(plotvars)

        mylonmin = float(np.nanmin(data.x))
        mylonmax = float(np.nanmax(data.x))
        mylatmin = float(np.nanmin(data.y))
        mylatmax = float(np.nanmax(data.y))
        lonrange = mylonmax - mylonmin
        latrange = mylatmax - mylatmin
        if lonrange > 360.0:
            mylonmax = mylonmin + 360.0

        if draw_static_map:
            if not ((lonrange > 350 and latrange > 160) or plotvars.user_mapset == 1):
                map_runtime.configure(
                    lonmin=mylonmin,
                    lonmax=mylonmax,
                    latmin=mylatmin,
                    latmax=mylatmax,
                    user_mapset=0,
                    resolution=plotvars.resolution,
                )
            map_runtime.ensure_map_axes()

        # Add a cyclic longitude column when the grid is near-global but
        # doesn't close on itself (no explicit bounds), to avoid a gap at
        # the wrap-around seam in Cartopy.
        if (
            data.x is not None
            and data.y is not None
            and np.ndim(data.x) == 1
            and np.ndim(data.y) == 1
            and not blockfill
        ):
            lonrange_data = float(np.nanmax(data.x)) - float(np.nanmin(data.x))
            if 350.0 < lonrange_data < 360.0:
                new_field, new_x = _add_cyclic(data.field, data.x)
                data = replace(data, field=new_field, x=new_x)

        if np.ndim(data.y) == 1 and data.y[0] > data.y[-1]:
            data = replace(data, y=data.y[::-1], field=np.flipud(data.field))

        xticks = kwargs.get("xticks", None)
        yticks = kwargs.get("yticks", None)
        xticklabels = kwargs.get("xticklabels", None)
        yticklabels = kwargs.get("yticklabels", None)

    if isinstance(f, cf.Field) and data.ptype in (4, 5):
        time_ticks, time_labels, time_label = utility.timeaxis(
            dtimes=f.construct("T"),
            user_gset=plotvars.user_gset,
            xmin=plotvars.xmin,
            xmax=plotvars.xmax,
            ymin=plotvars.ymin,
            ymax=plotvars.ymax,
            tspace_year=getattr(plotvars, "tspace_year", None),
            tspace_hour=getattr(plotvars, "tspace_hour", None),
            tspace_day=getattr(plotvars, "tspace_day", None),
        )
        if data.ptype == 4:
            lonlat_ticks, lonlat_labels = utility.mapaxis(
                min_val=xmin, max_val=xmax, axis_type=1, degsym=plotvars.degsym
            )
            default_xlabel = default_xlabel or "Longitude"
        else:
            lonlat_ticks, lonlat_labels = utility.mapaxis(
                min_val=xmin, max_val=xmax, axis_type=2, degsym=plotvars.degsym
            )
            default_xlabel = default_xlabel or "Latitude"

        default_ylabel = time_label or default_ylabel or "time"

        if xticks is None:
            xticks = lonlat_ticks
            xticklabels = lonlat_labels
        if yticks is None:
            yticks = time_ticks
            yticklabels = time_labels
    else:
        if data.ptype == 2:
            if xticks is None:
                xticks, xticklabels = utility.mapaxis(
                    min_val=xmin,
                    max_val=xmax,
                    axis_type=2,
                    degsym=plotvars.degsym,
                )
            if yticks is None:
                if kwargs.get("ylog", False):
                    ylo = min(ymin, ymax)
                    yhi = max(ymin, ymax)
                    yticks = [
                        tick
                        for tick in (1000, 100, 10, 1)
                        if ylo <= tick <= yhi
                    ]
                else:
                    ystep = 100.0
                    yrange = abs(ymax - ymin)
                    if yrange < 1:
                        ystep = yrange / 10.0 if yrange != 0 else 0.1
                    if yrange > 1:
                        ystep = 1.0
                    if yrange > 10:
                        ystep = 10.0
                    if yrange > 100:
                        ystep = 100.0
                    if yrange > 1000:
                        ystep = 200.0
                    if yrange > 2000:
                        ystep = 500.0
                    if yrange > 5000:
                        ystep = 1000.0
                    if yrange > 15000:
                        ystep = 5000.0
                    yticks = utility.gvals(
                        dmin=min(ymin, ymax),
                        dmax=max(ymin, ymax),
                        mystep=ystep,
                        mod=False,
                    )[0]
        elif data.ptype == 3:
            if xticks is None:
                xticks, xticklabels = utility.mapaxis(
                    min_val=xmin,
                    max_val=xmax,
                    axis_type=1,
                    degsym=plotvars.degsym,
                )
            if yticks is None:
                if kwargs.get("ylog", False):
                    ylo = min(ymin, ymax)
                    yhi = max(ymin, ymax)
                    yticks = [
                        tick
                        for tick in (1000, 100, 10, 1)
                        if ylo <= tick <= yhi
                    ]
                else:
                    ystep = 100.0
                    yrange = abs(ymax - ymin)
                    if yrange < 1:
                        ystep = yrange / 10.0 if yrange != 0 else 0.1
                    if yrange > 1:
                        ystep = 1.0
                    if yrange > 10:
                        ystep = 10.0
                    if yrange > 100:
                        ystep = 100.0
                    if yrange > 1000:
                        ystep = 200.0
                    if yrange > 2000:
                        ystep = 500.0
                    if yrange > 5000:
                        ystep = 1000.0
                    if yrange > 15000:
                        ystep = 5000.0
                    yticks = utility.gvals(
                        dmin=min(ymin, ymax),
                        dmax=max(ymin, ymax),
                        mystep=ystep,
                        mod=False,
                    )[0]
        else:
            if xticks is None:
                xticks = utility.gvals(dmin=xmin, dmax=xmax, mod=False)[0]
            if yticks is None:
                yticks = utility.gvals(dmin=ymax, dmax=ymin, mod=False)[0]

    if data.ptype == 1:
        layout = ContourLayout(plotvars).allocate_map_viewport(
            colorbar_orientation=colorbar_orientation,
            colorbar_position=kwargs.get("colorbar_position", None),
        )
    else:
        layout = ContourLayout(plotvars).allocate_xy_viewport(
            colorbar_orientation=colorbar_orientation,
            colorbar_position=kwargs.get("colorbar_position", None),
        )
        layout.apply_axis_labels(
            xlabel=kwargs.get("xlabel", default_xlabel),
            ylabel=kwargs.get("ylabel", default_ylabel),
            xticks=xticks,
            yticks=yticks,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )

    data = replace(
        data,
        levels=np.asarray(clevs),
        mult=mult,
        fmult=fmult,
        fill=fill,
        lines=lines,
        blockfill=blockfill,
    )
    if data.ptype == 1:
        renderer = MapContourRenderer(layout=layout, data=data, colour_scale=cs)
    else:
        renderer = XYContourRenderer(layout=layout, data=data, colour_scale=cs)

    if fill:
        renderer.render_filled(alpha=alpha, zorder=zorder, transform_first=None)
    if blockfill:
        renderer.render_blockfill(
            fast=kwargs.get("blockfill_fast", None), alpha=alpha, zorder=zorder
        )
    if lines:
        renderer.render_lines(
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            line_labels=line_labels,
            zero_thick=zero_thick,
            zorder=zorder,
        )

    if data.ptype == 1:
        map_runtime = MapSet(plotvars)

        animation = bool(kwargs.get("animation", False))
        reuse_map_background = bool(kwargs.get("reuse_map_background", False))
        draw_static_map = not (animation and reuse_map_background)

        if draw_static_map:
            apply_axes(
                plot_type=1,
                xticks=kwargs.get("xticks", None),
                yticks=kwargs.get("yticks", None),
                xlabel=kwargs.get("xlabel", None),
                ylabel=kwargs.get("ylabel", None),
                xticklabels=kwargs.get("xticklabels", None),
                yticklabels=kwargs.get("yticklabels", None),
            )

            feature = cfeature.NaturalEarthFeature(
                name="land",
                category="physical",
                scale=plotvars.resolution,
                facecolor="none",
            )
            plotvars.mymap.add_feature(
                feature,
                edgecolor=plotvars.continent_color or "k",
                linewidth=plotvars.continent_thickness or 1.5,
                linestyle=plotvars.continent_linestyle or "solid",
                zorder=kwargs.get("zorder", 1),
            )
            if plotvars.ocean_color is not None:
                plotvars.mymap.add_feature(
                    cfeature.OCEAN,
                    edgecolor="face",
                    facecolor=plotvars.ocean_color,
                    zorder=plotvars.feature_zorder,
                )
            if plotvars.land_color is not None:
                plotvars.mymap.add_feature(
                    cfeature.LAND,
                    edgecolor="face",
                    facecolor=plotvars.land_color,
                    zorder=plotvars.feature_zorder,
                )
            if plotvars.lake_color is not None:
                plotvars.mymap.add_feature(
                    cfeature.LAKES,
                    edgecolor="face",
                    facecolor=plotvars.lake_color,
                    zorder=plotvars.feature_zorder,
                )
            if kwargs.get("grid", False):
                map_runtime.draw_grid()

            map_runtime.draw_polar_axes()

        # Persist only dynamic contour artists for animation updates.
        plotvars._contour_animation_artists = list(renderer.frame_artists)

    if colorbar:
        renderer.render_colorbar(
            orientation=colorbar_orientation,
            shrink=kwargs.get("colorbar_shrink", None),
            position=kwargs.get("colorbar_position", None),
            fraction=kwargs.get("colorbar_fraction", None),
            thick=kwargs.get("colorbar_thick", None),
            anchor=kwargs.get("colorbar_anchor", None),
            fontsize=kwargs.get("colorbar_fontsize", None),
            fontweight=kwargs.get("colorbar_fontweight", None),
            text_up_down=kwargs.get("colorbar_text_up_down", False),
            text_down_up=kwargs.get("colorbar_text_down_up", False),
            drawedges=kwargs.get("colorbar_drawedges", True),
            labels=list(cbar_labels),
            title=colorbar_title,
        )

    file = plotvars.file
    session_open = bool(getattr(plotvars, "_contour_session_open", False))
    if file is not None and not session_open:
        saveargs = {}
        if plotvars.tight:
            saveargs = {"bbox_inches": "tight"}
        if os.path.splitext(file)[1].lower() not in (".ps", ".eps", ".png", ".pdf"):
            file = file + ".png"
        figure = plotvars.master_plot or getattr(plotvars.plot, "figure", None)
        if figure is not None:
            figure.savefig(
                file,
                orientation=plotvars.orientation,
                dpi=plotvars.dpi,
                **saveargs,
            )
            plot.close(figure)

    if data.ptype == 1:
        title = kwargs.get("title", "") or ""
        animation = bool(kwargs.get("animation", False))
        reuse_map_background = bool(kwargs.get("reuse_map_background", False))
        if title != "" and not (animation and reuse_map_background):
            _apply_map_title(
                mymap=plotvars.mymap,
                title=title,
                proj=plotvars.proj,
                boundinglat=plotvars.boundinglat,
                lon_0=plotvars.lon_0,
                lonmin=plotvars.lonmin,
                lonmax=plotvars.lonmax,
                latmin=plotvars.latmin,
                latmax=plotvars.latmax,
                title_fontsize=plotvars.title_fontsize,
                title_fontweight=plotvars.title_fontweight,
            )
    else:
        layout.apply_title(
            title=kwargs.get("title", "") or "",
            dims_title=bool(kwargs.get("titles", False)),
            fontsize=plotvars.title_fontsize,
            fontweight=plotvars.title_fontweight,
        )

    return True


def con(f=None, x=None, y=None, **kwargs):
    """Contour entrypoint coordinating through new object architecture.
    
    Gradually extracts logic from legacy _legacy_con into structured classes
    while preserving behavior. Eventually rendering will be split into 
    MapContourRenderer and XYContourRenderer subclasses.
    
    For now, orchestration uses the new classes for data and styling,
    then delegates to legacy renderer.
    """
    # Refactor mode: unsupported cases should fail explicitly rather than
    # silently routing through legacy code.
    if not _can_use_new_xy_path(f=f, kwargs=kwargs):
        raise NotImplementedError(
            "Contour case not implemented in refactored renderer yet"
        )

    if _render_with_new_xy(f=f, x=x, y=y, kwargs=kwargs):
        return None

    raise NotImplementedError(
        "Contour case not implemented in refactored renderer yet"
    )

