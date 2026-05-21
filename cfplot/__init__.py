"""
cf-plot: code-light plotting for earth science and aligned research

Documentation is hosted and found at: https://ncas-cms.github.io/cf-plot/
"""

__author__ = "Andy Heaps, Sadie Bartholomew, Bryan Lawrence"
__date__ = "16th May, 2026"
__version__ = "4.0.0"

import os

from .cfplot import *  # noqa: F403, F401
from .layout_runtime import gclose, gopen
from .line import lineplot
from .state import plotvars
from .trajectory import traj
from .utility import mapaxis as _mapaxis_impl
from .utils import _gvals


def _mapaxis(min=None, max=None, type=None):
	return _mapaxis_impl(
		min_val=min,
		max_val=max,
		axis_type=type,
		degsym=bool(plotvars.degsym),
	)


def _which(program):
	"""Check if an executable command is available on PATH."""

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
