"""
cf-plot: code-light plotting for earth science and aligned research

Copyright 2025, National Centre for Atmospheric Science (NCAS).

"""

import os
import subprocess
import sys
from copy import deepcopy
from distutils.version import StrictVersion

import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.util as cartopy_util
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plot
import numpy as np
from scipy.interpolate import griddata
import shapely.geometry as sgeom
from matplotlib.collections import PolyCollection

import cf

from .calculate import (
    calculate_levels,
)
from .colour import (
    _process_color_scales,
    cbar,
)
from .graphic import (
    _which,
    gclose,
    gopen,
    gpos,
)
from .mapping import (
    _cscale_get_map,
    _mapaxis,
    _map_title,
    _plot_map_axes,
    _set_map,
    axes_plot,
    map_grid,
)
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
from .utils import (
    _bfill,
    _bfill_ugrid,
    _cf_data_assign,
    _dim_titles,
    _gvals,
    _supscr,
    _timeaxis,
    add_cyclic,
    cf_var_name,
    cf_var_name_titles,
    find_dim_names,
    find_pos_in_array,
    find_z,
    fix_floats,
    generate_titles,
    irregular_window,
    max_ndecs_data,
    ndecs,
    pcon,
    polar_regular_grid,
    regrid,
    rgaxes,
    stipple_points,
    vloc,
)
from .validate import (
    _check_data,
    check_well_formed,
    orca_check,
)
