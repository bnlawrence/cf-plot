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


def stream(
    u=None,
    v=None,
    x=None,
    y=None,
    density=None,
    linewidth=None,
    color=None,
    arrowsize=None,
    arrowstyle=None,
    minlength=None,
    maxlength=None,
    axes=True,
    xaxis=True,
    yaxis=True,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    zorder=None,
):
    """
    | Plot a streamplot to show fluid flow and 2D field gradients.
    |
    | u=None - u wind
    | v=None - v wind
    | x=None - x locations of u and v
    | y=None - y locations of u and v
    | density=None - controls the closeness of streamlines. When density = 1,
    |                the domain is divided into a 30x30 grid
    | linewidth=None - the width of the stream lines. With a 2D array the
    |                  line width can be varied across the grid. The array
    |                  must have the same shape as u and v
    | color=None - the streamline color
    | arrowsize=None - scaling factor for the arrow size
    | arrowstyle=None - arrow style specification
    | minlength=None - minimum length of streamline in axes coordinates
    | maxlength=None - maximum length of streamline in axes coordinates
    | axes=True - plot x and y axes
    | xaxis=True - plot xaxis
    | yaxis=True - plot y axis
    | xticks=None - xtick positions
    | xticklabels=None - xtick labels
    | yticks=None - y tick positions
    | yticklabels=None - ytick labels
    | xlabel=None - label for x axis
    | ylabel=None - label for y axis
    | title=None - title for plot
    | zorder=None - plotting order
    |
    :Returns:
     None
    """

    colorbar_title = ""
    if title is None:
        title = ""
    text_fontsize = plotvars.text_fontsize
    continent_thickness = plotvars.continent_thickness
    continent_color = plotvars.continent_color
    if text_fontsize is None:
        text_fontsize = 11
    if continent_thickness is None:
        continent_thickness = 1.5
    if continent_color is None:
        continent_color = "k"
    title_fontsize = plotvars.title_fontsize
    if title_fontsize is None:
        title_fontsize = 15
    resolution_orig = plotvars.resolution
    rotated_vect = False

    # Set potential user axis labels
    user_xlabel = xlabel
    user_ylabel = ylabel

    # Set any additional arguments to streamplot
    plotargs = {}
    if density is not None:
        plotargs["density"] = density
    if linewidth is not None:
        plotargs["linewidth"] = linewidth
    if color is not None:
        plotargs["color"] = color
    if arrowsize is not None:
        plotargs["arrowsize"] = arrowsize
    if arrowstyle is not None:
        plotargs["arrowstyle"] = arrowstyle
    if minlength is not None:
        plotargs["minlength"] = minlength
    if maxlength is not None:
        plotargs["maxlength"] = maxlength

    # Extract required data
    # If a cf-python field
    if isinstance(u, cf.Field):

        # Check data is 2D
        ndims = np.squeeze(u.data).ndim
        if ndims != 2:
            errstr = (
                "\n\ncfp.vect error need a 2 dimensonal u field to make "
                "vectors\n"
                f"received {np.squeeze(u.data).ndim}"
            )
            if ndims == 1:
                errstr += " dimension\n\n"
            else:
                errstr += " dimensions\n\n"
            raise TypeError(errstr)

        (
            u_data,
            u_x,
            u_y,
            ptype,
            colorbar_title,
            xlabel,
            ylabel,
            xpole,
            ypole,
        ) = _cf_data_assign(u, colorbar_title, rotated_vect=rotated_vect)
    elif isinstance(u, cf.FieldList):
        raise TypeError("Can't plot a field list")
    else:
        # field=f #field data passed in as f
        _check_data(u, x, y)
        u_data = deepcopy(u)
        u_x = deepcopy(x)
        u_y = deepcopy(y)
        xlabel = ""
        ylabel = ""

    if isinstance(v, cf.Field):

        # Check data is 2D
        ndims = np.squeeze(v.data).ndim
        if ndims != 2:
            errstr = (
                "\n\ncfp.vect error need a 2 dimensonal v field to make "
                "vectors\n"
                f"received {np.squeeze(v.data).ndim}"
            )
            if ndims == 1:
                errstr += " dimension\n\n"
            else:
                errstr += " dimensions\n\n"
            raise TypeError(errstr)

        (
            v_data,
            v_x,
            v_y,
            ptype,
            colorbar_title,
            xlabel,
            ylabel,
            xpole,
            ypole,
        ) = _cf_data_assign(v, colorbar_title, rotated_vect=rotated_vect)
    elif isinstance(v, cf.FieldList):
        raise TypeError("Can't plot a field list")
    else:
        # field=f #field data passed in as f
        _check_data(v, x, y)
        v_data = deepcopy(v)
        xlabel = ""
        ylabel = ""

    # Reset xlabel and ylabel values with user defined labels in specified
    if user_xlabel is not None:
        xlabel = user_xlabel
    if user_ylabel is not None:
        ylabel = user_ylabel

    # Retrieve any user defined axis labels
    if xlabel == "" and plotvars.xlabel is not None:
        xlabel = plotvars.xlabel
    if ylabel == "" and plotvars.ylabel is not None:
        ylabel = plotvars.ylabel
    if xticks is None and plotvars.xticks is not None:
        xticks = plotvars.xticks
        if plotvars.xticklabels is not None:
            xticklabels = plotvars.xticklabels
        else:
            xticklabels = list(map(str, xticks))
    if yticks is None and plotvars.yticks is not None:
        yticks = plotvars.yticks
        if plotvars.yticklabels is not None:
            yticklabels = plotvars.yticklabels
        else:
            yticklabels = list(map(str, yticks))

    # Open a new plot if necessary
    if plotvars.user_plot == 0:
        gopen(user_plot=0)

    # Call gpos(1) if not already called
    if plotvars.rows > 1 or plotvars.columns > 1:
        if plotvars.gpos_called is False:
            gpos(1)

    # Set plot type if user specified
    if ptype is not None:
        plotvars.plot_type = ptype

    lonrange = np.nanmax(u_x) - np.nanmin(u_x)
    latrange = np.nanmax(u_y) - np.nanmin(u_y)

    if plotvars.plot_type == 1:
        # Set up mapping
        if (lonrange > 350 and latrange > 170) or plotvars.user_mapset == 1:
            _set_map()
        else:
            mapset(
                lonmin=np.nanmin(u_x),
                lonmax=np.nanmax(u_x),
                latmin=np.nanmin(u_y),
                latmax=np.nanmax(u_y),
                user_mapset=0,
                resolution=resolution_orig,
            )
            _set_map()

        mymap = plotvars.mymap

    # Map streamplot
    if plotvars.plot_type == 1:

        plotvars.mymap.streamplot(
            u_x, u_y, u_data, v_data, transform=ccrs.PlateCarree(), **plotargs
        )

        # axes
        _plot_map_axes(
            axes=axes,
            xaxis=xaxis,
            yaxis=yaxis,
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            user_xlabel=user_xlabel,
            user_ylabel=user_ylabel,
            verbose=False,
        )

        # Coastlines
        continent_thickness = plotvars.continent_thickness
        continent_color = plotvars.continent_color
        continent_linestyle = plotvars.continent_linestyle
        if continent_thickness is None:
            continent_thickness = 1.5
        if continent_color is None:
            continent_color = "k"
        if continent_linestyle is None:
            continent_linestyle = "solid"

        feature = cfeature.NaturalEarthFeature(
            name="land",
            category="physical",
            scale=plotvars.resolution,
            facecolor="none",
        )
        mymap.add_feature(
            feature,
            edgecolor=continent_color,
            linewidth=continent_thickness,
            linestyle=continent_linestyle,
        )

        # Title
        if title is not None:
            _map_title(title)

    ##########
    # Save plot
    ##########
    if plotvars.user_plot == 0:
        gset()
        cscale()
        gclose()

    if plotvars.user_mapset == 0:
        mapset()
        mapset(resolution=resolution_orig)

