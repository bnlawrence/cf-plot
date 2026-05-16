"""
cf-plot is a set of Python plotting routines for the contour, vector
and line plots that climate researchers commonly make.

See the cf-plot home page http://ajheaps.github.io/cf-plot for a
gallery of plots and how to use cf-plot.
"""

__author__ = "Andy Heaps, Sadie Bartholomew, Bryan Lawrence"
__date__ = "16th May, 2026"
__version__ = "4.0.0"

from .cfplot import *  # noqa: F403, F401
from .layout_runtime import gclose, gopen

# Internal-use methods required for testing
from .cfplot import _gvals
from .cfplot import _mapaxis
from .cfplot import _which
