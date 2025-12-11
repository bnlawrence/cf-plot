"""
cf-plot: code-light plotting for earth science and aligned research

Documentation is hosted and found at: https://ncas-cms.github.io/cf-plot/
"""

__author__ = "Andy Heaps"
__maintainer__ ="Sadie Bartholomew"
__date__ = "28th April 2025"
__version__ = "3.4.0"

import os

import cartopy
import cf
from distutils.version import StrictVersion
import matplotlib

# Import module functionality -------------------------------------------
# Imports to export functions: cfp.<module>.<function> -> cfp.<function>
# either as intended going forward or to preserve existing API.
# TODO review what should be available at module level.
from .contour import con
from .vector import vect
from .stipple import stipple
from .trajectory import traj
from .line import lineplot
from .stream import stream

from .calculate import (
    calculate_levels,
)
from .colour import (
    # Internal fuctions: don't expose, but leave commented here to track:
    # _cscale_get_map,
    # _process_color_scales,
    cbar,
)
from .graphic import (
    # Internal fuctions: don't expose, but leave commented here to track:
    # _which,
    gclose,
    gopen,
    gpos,
)
from .mapping import (
    # Internal fuctions: don't expose, but leave commented here to track:
    # _mapaxis,
    # _map_title,
    # _plot_map_axes,
    # _set_map,
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
    # Internal fuctions: don't expose, but leave commented here to track:
    # _bfill,
    # _bfill_ugrid,
    # _cf_data_assign,
    # _dim_titles,
    # _gvals,
    # _supscr,
    # _timeaxis,
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


# Process versions and display ------------------------------------------

# Check for the minimum cf-python version
cf_version_min = "3.17.0"
errstr = (
    f"cf-python >= {cf_version_min} needs to be installed to use cf-plot"
)
if StrictVersion(cf.__version__) < StrictVersion(cf_version_min):
    raise Warning(errstr)
# TODO add these checks for all other dependencies too?

# Check for a display and use the Agg backing store if none is present
# This is for batch mode processing
try:
    disp = os.environ["DISPLAY"]
except Exception:
    matplotlib.use("Agg")

# Check for user setting of pre_existing_data_dir pointing to central
# cartopy setup
# This is used in the cfview simple setup process
try:
    pre_existing_data_dir = os.environ["pre_existing_data_dir"]
    cartopy.config["pre_existing_data_dir"] = pre_existing_data_dir
except KeyError:
    pass
