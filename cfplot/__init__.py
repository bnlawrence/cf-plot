"""
cf-plot: code-light plotting for earth science and aligned research

Documentation is hosted and found at: https://ncas-cms.github.io/cf-plot/
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
