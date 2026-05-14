"""Contour refactor boundary.

This module introduces the object model used to refactor contour plotting out
of cfplot.py while preserving the current behavior during migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cf
import matplotlib.colors
import numpy as np
from matplotlib.axes import Axes


@dataclass(frozen=True)
class ContourData:
    """Read-only contour inputs after extraction and validation."""

    field: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    ptype: int | None = None
    colorbar_title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    levels: np.ndarray | None = None
    mult: int = 0
    fmult: float = 1.0
    irregular: bool | None = None
    is_ugrid: bool = False
    is_orca: bool = False

    @classmethod
    def from_cf_field(
        cls,
        f: cf.Field,
        colorbar_title: str | None,
        plotvars: Any,
        verbose: bool | None,
    ) -> "ContourData":
        # Placeholder for stepwise migration from cfplot._legacy_con.
        _ = (plotvars, verbose)
        return cls(
            field=np.asarray(f.array),
            x=None,
            y=None,
            colorbar_title=colorbar_title,
        )

    @classmethod
    def from_arrays(
        cls, field: np.ndarray, x: np.ndarray | None, y: np.ndarray | None
    ) -> "ContourData":
        return cls(field=np.asarray(field), x=x, y=y)


class ContourLayout:
    """Manage viewport and annotation geometry for contour plots."""

    def __init__(self, plotvars: Any):
        self.viewport: Axes | None = None
        self.colorbar_ax: Axes | None = None
        self.title_ax: Axes | None = None
        self._plotvars = plotvars

    def allocate(
        self,
        colorbar_orientation: str | None,
        colorbar_position: list[float] | None,
    ) -> "ContourLayout":
        _ = (colorbar_orientation, colorbar_position)
        return self

    def apply_title(
        self,
        title: str | None,
        dims_title: bool,
        fontsize: int | None,
        fontweight: str | None,
    ) -> None:
        _ = (title, dims_title, fontsize, fontweight)

    def apply_axis_labels(
        self,
        xlabel: str | None,
        ylabel: str | None,
        xticks: Any,
        yticks: Any,
    ) -> None:
        _ = (xlabel, ylabel, xticks, yticks)


class ColourScale:
    """Encapsulate level fitting, colormap selection, and cbar labels."""

    def __init__(self, plotvars: Any):
        self._plotvars = plotvars

    def fit_to_levels(
        self,
        levels: np.ndarray,
        includes_zero: bool,
        levels_extend: str,
    ) -> "ColourScale":
        _ = (levels, includes_zero, levels_extend)
        return self

    def get_cmap(self) -> matplotlib.colors.ListedColormap:
        # Transitional fallback. Legacy code still defines the effective map.
        return matplotlib.colors.ListedColormap(["#0a3278", "#a50000"])

    def colorbar_labels(
        self,
        levels: np.ndarray,
        orientation: str,
        n_columns: int,
        label_skip: int,
        custom_labels: list[str] | None,
    ) -> list[str]:
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

    def render_filled(
        self, alpha: float, zorder: int, transform_first: bool | None
    ) -> None:
        _ = (alpha, zorder, transform_first)

    def render_blockfill(
        self, fast: bool | None, alpha: float, zorder: int
    ) -> None:
        _ = (fast, alpha, zorder)

    def render_lines(
        self,
        colors: Any,
        linewidths: Any,
        linestyles: Any,
        line_labels: bool,
        zero_thick: bool | int,
    ) -> None:
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
    ) -> None:
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
        )


class MapContourRenderer(ContourRenderer):
    """Map renderer specialization for ptype == 1."""


class XYContourRenderer(ContourRenderer):
    """Cartesian renderer specialization for non-map contour plots."""


def con(f=None, x=None, y=None, **kwargs):
    """Transitional contour entrypoint delegated to legacy implementation."""

    from .cfplot import _legacy_con

    return _legacy_con(f=f, x=x, y=y, **kwargs)

