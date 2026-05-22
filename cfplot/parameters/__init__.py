from .parameters import (
    allvars_defaults,
    axes,
    cscale,
    cscale1,
    global_blockfill,
    global_fill,
    global_lines,
    gset,
    levs,
    mapset,
    plotvars_defaults,
    pvars,
    reset,
    setvars,
    setvars_defaults,
    viridis,
)

from ..state import plotvars as plotvars
from . import parameters as _parameters_module

# Ensure parameter APIs mutate/read the shared runtime state object.
_parameters_module.plotvars = plotvars
