"""Compatibility exports for legacy mapping helpers.

This module preserves main-side imports while the refactor branch continues to
own the actual implementations in cfplot.py.
"""

from .cfplot import _map_title, _mapaxis, _plot_map_axes, _set_map, axes_plot, map_grid

__all__ = [
    "_map_title",
    "_mapaxis",
    "_plot_map_axes",
    "_set_map",
    "axes_plot",
    "map_grid",
]