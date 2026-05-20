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
    plotvars,
    plotvars_defaults,
    pvars,
    reset,
    setvars,
    setvars_defaults,
    viridis,
)

# Keep legacy package imports pointed at the refactor state/mutator layer.
from ..cfplot import cscale as cscale
from ..cfplot import gset as gset
from ..cfplot import mapset as mapset
from ..state import plotvars as plotvars
