"""
cf-plot: code-light plotting for earth science and aligned research

Documentation is hosted and found at: https://ncas-cms.github.io/cf-plot/
"""

__author__ = "Andy Heaps, Sadie Bartholomew, Bryan Lawrence"
__date__ = "16th May, 2026"
__version__ = "4.0.0"

from .colorbar import cbar
from .colour import cscale
from .contour import con
from .contour import levs
from .layout_runtime import gclose, gopen, gset
from .layout_runtime import gpos
from .line import lineplot
from .map_runtime import mapset
from .state import plotvars, setvars
from .stipple import stipple
from .stream import stream
from .trajectory import traj
from .state import reset_runtime_state
from .utility import gvals as _gvals_impl, mapaxis as _mapaxis_impl, regrid
from .vector import vect
from .rotated_runtime import _render_rotated_grid_axes


def reset():
    gset()
    cscale()
    levs()
    mapset()
    setvars()
    reset_runtime_state()


def _gvals(*args, **kwargs):
    return _gvals_impl(*args, **kwargs)


def rgaxes(*, xpole=None, ypole=None, xvec=None, yvec=None, xticks=None, xticklabels=None, yticks=None, yticklabels=None, axes=True, xaxis=True, yaxis=True, xlabel=None, ylabel=None):
    return _render_rotated_grid_axes(
        xpole=xpole,
        ypole=ypole,
        xvec=xvec,
        yvec=yvec,
        xticks=xticks,
        xticklabels=xticklabels,
        yticks=yticks,
        yticklabels=yticklabels,
        axes=axes,
        xaxis=xaxis,
        yaxis=yaxis,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def _mapaxis(min=None, max=None, type=None):
    return _mapaxis_impl(
        min_val=min,
        max_val=max,
        axis_type=type,
        degsym=bool(plotvars.degsym),
    )


__all__ = [
    "cbar",
    "con",
    "cscale",
    "gclose",
    "gopen",
    "gpos",
    "gset",
    "levs",
    "lineplot",
    "mapset",
    "plotvars",
    "regrid",
    "reset",
    "rgaxes",
    "setvars",
    "stipple",
    "stream",
    "traj",
    "vect",
    "_gvals",
    "_mapaxis",
]
