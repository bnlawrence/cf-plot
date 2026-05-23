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
from .graphic import gpos
from .layout_runtime import gclose, gopen, gset
from .line import lineplot
from .map_runtime import mapset
from .state import plotvars, setvars
from .stipple import stipple
from .stream import stream
from .trajectory import traj
from .utility import gvals as _gvals_impl, mapaxis as _mapaxis_impl, regrid
from .vector import vect
from .rotated_runtime import _render_rotated_grid_axes


def reset():
    gset()
    cscale()
    levs()
    mapset()
    setvars()

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


def _which(program):
    """Check if an executable command is available on PATH."""

    import os

    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    def ext_candidates(fpath):
        yield fpath
        for ext in os.environ.get("PATHEXT", "").split(os.pathsep):
            yield fpath + ext

    for path in os.environ.get("PATH", "").split(os.pathsep):
        exe_file = os.path.join(path, program)
        for candidate in ext_candidates(exe_file):
            if is_exe(candidate):
                return candidate

    return None


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
    "_which",
]
