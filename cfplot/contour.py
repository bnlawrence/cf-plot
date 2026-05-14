"""Contour refactor boundary.

This module introduces the object model used to refactor contour plotting out
of cfplot.py while preserving the current behavior during migration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cf
import matplotlib.colors
import numpy as np
from matplotlib.axes import Axes


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
        plotvars: Any,
        verbose: bool | None,
    ) -> "ContourData":
        """Extract and prepare CF field for contouring.
        
        Delegates to legacy _cf_data_assign for now, wrapping result as immutable.
        """
        from .cfplot import _cf_data_assign

        # Extract data using legacy path
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
        ) = _cf_data_assign(f, colorbar_title, verbose=verbose)

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
            xpole=xpole,
            ypole=ypole,
        )

    @classmethod
    def from_arrays(
        cls,
        field: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "ContourData":
        """Create from raw numpy arrays with validation."""
        from .cfplot import _check_data

        field = np.asarray(field)
        x = np.asarray(x) if x is not None else np.arange(field.shape[1])
        y = np.asarray(y) if y is not None else np.arange(field.shape[0])

        _check_data(field, x, y)

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
        from .cfplot import gopen, gpos, plotvars

        # Set colorbar orientation  
        self.colorbar_orientation = colorbar_orientation or "horizontal"
        self.colorbar_position = colorbar_position

        # Call gpos(1) if multi-plot grid not already initialized
        if plotvars.rows > 1 or plotvars.columns > 1:
            if plotvars.gpos_called is False:
                gpos(1)

        # Open a new plot if necessary
        if plotvars.user_plot == 0:
            gopen(user_plot=0)

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
        from .cfplot import gopen, gpos, plotvars

        self.colorbar_orientation = colorbar_orientation or "horizontal"
        self.colorbar_position = colorbar_position

        # If map axes already exists, do not reopen/reset the viewport.
        if plotvars.mymap is None:
            if plotvars.rows > 1 or plotvars.columns > 1:
                if plotvars.gpos_called is False:
                    gpos(1)

            if plotvars.user_plot == 0:
                gopen(user_plot=0)

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
        from .cfplot import plotvars, _map_title, _dim_titles

        if title and title != "":
            if self._plotvars.plot_type == 1:
                _map_title(title)
            else:
                if self.viewport:
                    self.viewport.set_title(
                        title,
                        y=1.03,
                        fontsize=fontsize or plotvars.title_fontsize,
                        fontweight=fontweight or plotvars.title_fontweight,
                    )

        if dims_title:
            _dim_titles(title=dims_title if isinstance(dims_title, str) else None)

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
        from .cfplot import axes_plot, _plot_map_axes

        if self.viewport is None:
            return

        # For map plots, use specialized axis handler
        if self._plotvars.plot_type == 1:
            _plot_map_axes(
                axes=True,
                xaxis=True,
                yaxis=True,
                xticks=xticks,
                yticks=yticks,
                user_xlabel=xlabel or None,
                user_ylabel=ylabel or None,
                verbose=None,
            )
        else:
            # For non-map plots, use standard axes_plot
            axes_plot(
                xticks=xticks if xticks is not None else [],
                xticklabels=xticklabels if xticklabels is not None else xticks,
                yticks=yticks if yticks is not None else [],
                yticklabels=yticklabels if yticklabels is not None else yticks,
                xlabel=xlabel or "",
                ylabel=ylabel or "",
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

        # Delegate to legacy cscale system for now
        from .cfplot import cscale

        # Replicate cscale_flag == 0 logic (revert to default)
        if self._plotvars.cscale_flag == 0:
            col_zero = 0
            for cval in self._levels:
                if self._includes_zero is False:
                    col_zero = col_zero + 1
                if cval == 0:
                    self._includes_zero = True

            if self._includes_zero:
                cs_below = col_zero
                cs_above = np.size(self._levels) - col_zero + 1
                if (
                    self._plotvars.levels_extend == "max"
                    or self._plotvars.levels_extend == "neither"
                ):
                    cs_below = cs_below - 1
                if (
                    self._plotvars.levels_extend == "min"
                    or self._plotvars.levels_extend == "neither"
                ):
                    cs_above = cs_above - 1
                uniform = True
                if self._plotvars.cs_uniform is False:
                    uniform = False
                cscale("scale1", below=cs_below, above=cs_above, uniform=uniform)
            else:
                ncols = np.size(self._levels) + 1
                if (
                    self._plotvars.levels_extend == "min"
                    or self._plotvars.levels_extend == "max"
                ):
                    ncols = ncols - 1
                if self._plotvars.levels_extend == "neither":
                    ncols = ncols - 2
                cscale("viridis", ncols=ncols)

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
            cscale(self._plotvars.cs_user, ncols=ncols)
            self._plotvars.cscale_flag = 1

        return self

    def get_cmap(self) -> matplotlib.colors.ListedColormap:
        """Get colormap after fitting to levels."""
        from .cfplot import _cscale_get_map

        colmap = _cscale_get_map()
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
        _ = (orientation, n_columns)
        if custom_labels is not None:
            return custom_labels

        labels: list[str] = []
        for idx, level in enumerate(levels):
            if label_skip > 1 and idx % label_skip != 0:
                labels.append("")
            else:
                labels.append(str(level))
        return labels


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
    ) -> None:
        """Render contour lines and labels."""
        _ = (colors, linewidths, linestyles, line_labels, zero_thick)

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
        from .cfplot import ccrs, plotvars

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
        from .cfplot import _bfill

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
    ) -> None:
        """Render contour lines on a map with Cartopy transform."""
        from .cfplot import ccrs, ndecs, plotvars

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
        )
        if hasattr(cs, "collections"):
            self.frame_artists.extend(list(cs.collections))

        if line_labels and not isinstance(self.data.levels, int):
            nd = ndecs(self.data.levels)
            fmt = "%d"
            if nd != 0:
                fmt = "%1." + str(nd) + "f"
            plotvars.plot.clabel(
                cs,
                levels=self.data.levels,
                fmt=fmt,
                colors=colors,
                fontsize=plotvars.text_fontsize,
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
            )
            if hasattr(cs0, "collections"):
                self.frame_artists.extend(list(cs0.collections))


class XYContourRenderer(ContourRenderer):
    """Cartesian renderer specialization for non-map contour plots.
    
    Handles ptypes 0, 2-7 (simple XY, lat-height, lon-height, Hovmuller, rotated).
    """

    def render_filled(
        self, alpha: float, zorder: int, transform_first: bool | None
    ) -> None:
        """Render filled contours in Cartesian space."""
        from .cfplot import plotvars

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
        from .cfplot import _bfill

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
    ) -> None:
        """Render contour lines in Cartesian space."""
        from .cfplot import ndecs, plotvars

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
        )
        if line_labels and not isinstance(self.data.levels, int):
            nd = ndecs(self.data.levels)
            fmt = "%d"
            if nd != 0:
                fmt = "%1." + str(nd) + "f"
            plotvars.plot.clabel(cs, fmt=fmt, colors=colors, fontsize=plotvars.text_fontsize)

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
        from .cfplot import cbar

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
    if not isinstance(f, cf.Field) and ptype not in (0, None):
        return False

    return True


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


def _render_with_new_xy(f: Any, x: Any, y: Any, kwargs: dict[str, Any]) -> bool:
    """Attempt rendering via new XY renderer and return True on success."""
    from .cfplot import (
        _gvals,
        _mapaxis,
        _timeaxis,
        calculate_levels,
        global_blockfill,
        global_fill,
        global_lines,
        gopen,
        gset,
        plotvars,
    )

    if isinstance(f, cf.Field):
        data = ContourData.from_cf_field(
            f=f,
            colorbar_title=kwargs.get("colorbar_title", None),
            plotvars=plotvars,
            verbose=kwargs.get("verbose", None),
        )
        # Implemented CF extraction targets: map and selected non-map ptypes.
        if data.ptype not in (1, 2, 3, 4, 5):
            return False
    else:
        data = ContourData.from_arrays(field=np.asarray(f), x=x, y=y)

    # Keep legacy behavior for axis-routing logic by setting active plot type.
    plotvars.plot_type = data.ptype if isinstance(f, cf.Field) else kwargs.get("ptype", 0)

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
        clevs, mult, fmult = calculate_levels(
            field=data.field,
            level_spacing=kwargs.get("level_spacing", "linear"),
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

    colorbar_orientation = kwargs.get("colorbar_orientation", None) or "horizontal"
    colorbar_label_skip = kwargs.get("colorbar_label_skip", 1)
    if colorbar_label_skip is None:
        colorbar_label_skip = 1

    cbar_labels = kwargs.get("colorbar_labels")
    if cbar_labels is None:
        cbar_labels = cs.colorbar_labels(
            levels=np.asarray(clevs),
            orientation=colorbar_orientation,
            n_columns=plotvars.columns,
            label_skip=colorbar_label_skip,
            custom_labels=None,
        )

    if plotvars.user_plot == 0:
        gopen(user_plot=0)

    xmin = kwargs.get("xmin", float(np.nanmin(data.x)))
    xmax = kwargs.get("xmax", float(np.nanmax(data.x)))
    ymin = kwargs.get("ymin", float(np.nanmin(data.y)))
    ymax = kwargs.get("ymax", float(np.nanmax(data.y)))

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

    if kwargs.get("ylog", False):
        if ymax == 0:
            ymax = 1
        gset(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            ylog=True,
            user_gset=kwargs.get("user_gset", plotvars.user_gset),
        )
    else:
        gset(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
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

    if isinstance(f, cf.Field) and data.ptype == 1:
        from .cfplot import _set_map, mapset

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
            if (lonrange > 350 and latrange > 160) or plotvars.user_mapset == 1:
                _set_map()
            else:
                mapset(
                    lonmin=mylonmin,
                    lonmax=mylonmax,
                    latmin=mylatmin,
                    latmax=mylatmax,
                    user_mapset=0,
                    resolution=plotvars.resolution,
                )
                _set_map()

        if np.ndim(data.y) == 1 and data.y[0] > data.y[-1]:
            data = replace(data, y=data.y[::-1], field=np.flipud(data.field))

        xticks = kwargs.get("xticks", None)
        yticks = kwargs.get("yticks", None)
        xticklabels = kwargs.get("xticklabels", None)
        yticklabels = kwargs.get("yticklabels", None)

    if isinstance(f, cf.Field) and data.ptype in (4, 5):
        time_ticks, time_labels, time_label = _timeaxis(f.construct("T"))
        if data.ptype == 4:
            lonlat_ticks, lonlat_labels = _mapaxis(min=xmin, max=xmax, type=1)
            default_xlabel = default_xlabel or "Longitude"
        else:
            lonlat_ticks, lonlat_labels = _mapaxis(min=xmin, max=xmax, type=2)
            default_xlabel = default_xlabel or "Latitude"

        default_ylabel = time_label or default_ylabel or "time"

        if xticks is None:
            xticks = lonlat_ticks
            xticklabels = lonlat_labels
        if yticks is None:
            yticks = time_ticks
            yticklabels = time_labels
    else:
        if xticks is None:
            xticks = _gvals(dmin=xmin, dmax=xmax, mod=False)[0]
        if yticks is None:
            yticks = _gvals(dmin=ymax, dmax=ymin, mod=False)[0]

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
        )

    if data.ptype == 1:
        from .cfplot import (
            _plot_map_axes,
            cfeature,
            map_grid,
            plotvars,
        )

        animation = bool(kwargs.get("animation", False))
        reuse_map_background = bool(kwargs.get("reuse_map_background", False))
        draw_static_map = not (animation and reuse_map_background)

        if draw_static_map:
            _plot_map_axes(
                axes=kwargs.get("axes", True),
                xaxis=kwargs.get("xaxis", True),
                yaxis=kwargs.get("yaxis", True),
                xticks=kwargs.get("xticks", None),
                xticklabels=kwargs.get("xticklabels", None),
                yticks=kwargs.get("yticks", None),
                yticklabels=kwargs.get("yticklabels", None),
                user_xlabel=kwargs.get("xlabel", None),
                user_ylabel=kwargs.get("ylabel", None),
                verbose=kwargs.get("verbose", None),
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
                map_grid()

        # Persist only dynamic contour artists for animation updates.
        plotvars._contour_animation_artists = list(renderer.frame_artists)

    colorbar_title = kwargs.get("colorbar_title", "")
    if mult != 0:
        colorbar_title = f"{colorbar_title} *10^{{{mult}}}"

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

    if data.ptype == 1:
        title = kwargs.get("title", "") or ""
        animation = bool(kwargs.get("animation", False))
        reuse_map_background = bool(kwargs.get("reuse_map_background", False))
        if title != "" and not (animation and reuse_map_background):
            from .cfplot import _map_title

            _map_title(title)
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

