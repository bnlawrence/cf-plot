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


def stipple(
    f=None,
    x=None,
    y=None,
    min=None,
    max=None,
    size=80,
    color="k",
    pts=50,
    marker=".",
    edgecolors="k",
    alpha=1.0,
    ylog=False,
    zorder=1,
):
    """
    | Put markers on a plot to indicate value of interest.
    |
    | f=None - cf field or field
    | x=None - x points for field
    | y=None - y points for field
    | min=None - minimum threshold for stipple
    | max=None - maximum threshold for stipple
    | size=80 - default size for stipples
    | color='k' - default colour for stipples
    | pts=50 - number of points in the x direction
    | marker='.' - default marker for stipples
    | edegecolors='k' - outline colour
    | alpha=1.0 - transparency setting - default is off
    | ylog=False - set to True if a log pressure stipple plot
    |              is required
    | zorder=2 - plotting order
    |
    :Returns:
       None
    |
    """

    if plotvars.plot_type not in [1, 2, 3]:
        errstr = (
            "\n stipple error - only X-Y, X-Z and Y-Z \n"
            "stipple supported at the present time\n"
            "Please raise a feature request if you see this error.\n"
        )
        raise Warning(errstr)

    # Extract required data for contouring
    # If a cf-python field
    if isinstance(f, cf.Field):
        colorbar_title = ""
        (
            field,
            xpts,
            ypts,
            ptype,
            colorbar_title,
            xlabel,
            ylabel,
            xpole,
            ypole,
        ) = _cf_data_assign(f, colorbar_title)
    elif isinstance(f, cf.FieldList):
        raise TypeError("Can't plot a field list")
    else:
        field = f  # field data passed in as f
        _check_data(field, x, y)
        xpts = x
        ypts = y

    if plotvars.plot_type == 1:
        # Cylindrical projection
        # Add cyclic information if missing.
        lonrange = np.nanmax(xpts) - np.nanmin(xpts)
        if lonrange < 360:
            # field, xpts = cartopy_util.add_cyclic_point(field, xpts)
            field, xpts = add_cyclic(field, xpts)

        # if plotvars.proj == 'cyl':
        if plotvars.proj in ["cyl", "robin", "merc", "ortho", "moll"]:
            # Calculate interpolation points
            xnew, ynew = stipple_points(
                xmin=np.nanmin(xpts),
                xmax=np.nanmax(xpts),
                ymin=np.nanmin(ypts),
                ymax=np.nanmax(ypts),
                pts=pts,
                stype=2,
            )

            # Calculate points in map space
            xnew_map = xnew
            ynew_map = ynew

        if plotvars.proj == "npstere" or plotvars.proj == "spstere":
            # Calculate interpolation points
            xnew, ynew, xnew_map, ynew_map = polar_regular_grid()
            # Convert longitudes to be 0 to 360
            # negative longitudes are incorrectly regridded in polar
            # stereographic projection
            xnew = np.mod(xnew + 360.0, 360.0)

    if plotvars.plot_type >= 2 and plotvars.plot_type <= 3:

        # Flip data if a lat-height plot and lats start at the north pole
        if plotvars.plot_type == 2:
            if xpts[0] > xpts[-1]:
                xpts = xpts[::-1]
                field = np.fliplr(field)

        # Calculate interpolation points
        ymin = np.nanmin(ypts)
        ymax = np.nanmax(ypts)
        if ylog:
            ymin = np.log10(ymin)
            ymax = np.log10(ymax)

        xnew, ynew = stipple_points(
            xmin=np.nanmin(xpts),
            xmax=np.nanmax(xpts),
            ymin=ymin,
            ymax=ymax,
            pts=pts,
            stype=2,
        )

        if ylog:
            ynew = 10**ynew

    # Get values at the new points
    vals = regrid(f=field, x=xpts, y=ypts, xnew=xnew, ynew=ynew)

    # Work out which of the points are valid
    valid_points = np.array([], dtype="int64")
    for i in np.arange(np.size(vals)):
        if vals[i] >= min and vals[i] <= max:
            valid_points = np.append(valid_points, i)

    if plotvars.plot_type == 1:
        proj = ccrs.PlateCarree()

        if np.size(valid_points) > 0:
            plotvars.mymap.scatter(
                xnew[valid_points],
                ynew[valid_points],
                s=size,
                c=color,
                marker=marker,
                edgecolors=edgecolors,
                alpha=alpha,
                transform=proj,
                zorder=zorder,
            )

    if plotvars.plot_type >= 2 and plotvars.plot_type <= 3:
        plotvars.plot.scatter(
            xnew[valid_points],
            ynew[valid_points],
            s=size,
            c=color,
            marker=marker,
            edgecolors=edgecolors,
            alpha=alpha,
            zorder=zorder,
        )


def vect(
    u=None,
    v=None,
    x=None,
    y=None,
    scale=None,
    stride=None,
    pts=None,
    key_length=None,
    key_label=None,
    ptype=None,
    title=None,
    magmin=None,
    width=0.02,
    headwidth=3,
    headlength=5,
    headaxislength=4.5,
    pivot="middle",
    key_location=[0.95, -0.06],
    key_show=True,
    axes=True,
    xaxis=True,
    yaxis=True,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    ylog=False,
    color="k",
    zorder=3,
    titles=None,
    alpha=1.0,
):
    """
    | Plot vectors.
    |
    | u=None - u wind
    | v=None - v wind
    | x=None - x locations of u and v
    | y=None - y locations of u and v
    | scale=None - data units per arrow length unit.  A smaller values gives
    |              a larger vector. Generally takes one value but in the case
    |              of two supplied values the second vector scaling applies to
    |              the v field.
    | stride=None - plot vector every stride points. Can take two values one
    |               for x and one for y.
    | pts=None - use bilinear interpolation to interpolate vectors onto a new
    |            grid - takes one or two values.
    |            If one value is passed then this is used for both the x and
    |            y axes.
    | magmin=None - don't plot any vects with less than this magnitude.
    | key_length=None - length of the key.  Generally takes one value but in
    |                   the case of two supplied values the second vector
    |                   scaling applies to the v field.
    | key_label=None - label for the key. Generally takes one value but in the
    |                  case of two supplied values the second vector scaling
    |                  applies to the v field.
    | key_location=[0.9, -0.06] - location of the vector key relative to the
    |                             plot in normalised coordinates.
    | key_show=True - draw the key.  Set to False if not required.
    | ptype=0 - plot type - not needed for cf fields.
    |                       0 = no specific plot type,
    |                       1 = longitude-latitude,
    |                       2 = latitude - height,
    |                       3 = longitude - height,
    |                       4 = latitude - time,
    |                       5 = longitude - time
    |                       6 = rotated pole
    |
    | title=None - plot title
    | width=0.005 - shaft width in arrow units; default is 0.005 times the
    |               width of the plot
    | headwidth=3 - head width as multiple of shaft width, default is 3
    | headlength=5 - head length as multiple of shaft width, default is 5
    | headaxislength=4.5 - head length at shaft intersection, default is 4.5
    | pivot='middle' - the part of the arrow that is at the grid point; the
    |                  arrow rotates about this point
                       takes 'tail', 'middle', 'tip'
    | axes=True - plot x and y axes
    | xaxis=True - plot xaxis
    | yaxis=True - plot y axis
    | xticks=None - xtick positions
    | xticklabels=None - xtick labels
    | yticks=None - y tick positions
    | yticklabels=None - ytick labels
    | xlabel=None - label for x axis
    | ylabel=None - label for y axis
    | ylog=False - log y axis
    | color='k' - colour for the vectors - default is black.
    | zorder=3 - plotting order
    | titles=None - generate dimension and cell_methods titles for plot
    | alpha=1.0 - transparency setting.  The default is no transparency.
    |
    :Returns:
     None
    |
    """

    # If the vector color is white set the quicker key colour to black
    # so that it can be seen
    qkey_color = color
    if qkey_color == "w" or qkey_color == "white":
        qkey_color = "k"

    colorbar_title = ""
    text_fontsize = plotvars.text_fontsize
    continent_thickness = plotvars.continent_thickness
    continent_color = plotvars.continent_color
    if text_fontsize is None:
        text_fontsize = 11
    if continent_thickness is None:
        continent_thickness = 1.5
    if continent_color is None:
        continent_color = "k"
    # ylog=plotvars.ylog
    title_fontsize = plotvars.title_fontsize
    title_fontweight = plotvars.title_fontweight
    if title_fontsize is None:
        title_fontsize = 15
    resolution_orig = plotvars.resolution

    # Set potential user axis labels
    user_xlabel = xlabel
    user_ylabel = ylabel

    rotated_vect = False
    if isinstance(u, cf.Field):
        if u.ref(
            "grid_mapping_name:rotated_latitude_longitude", default=False
        ):
            rotated_vect = True

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
                "received {np.squeeze(v.data).ndim}"
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
        v_x = deepcopy(x)
        xlabel = ""
        ylabel = ""

    # If a minimum magnitude is specified mask these data points
    if magmin is not None:
        mag = np.sqrt(u_data**2 + v_data**2)
        invalid = np.where(mag <= magmin)
        if np.size(invalid) > 0:
            u_data[invalid] = np.nan
            v_data[invalid] = np.nan

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

    if scale is None:
        scale = np.nanmax(u_data) / 4.0

    if key_length is None:
        key_length = scale

    # Calculate a set of dimension titles if requested
    if titles:
        title_dims = generate_titles(u)
        title_dims = f"u\n{title_dims}"
        title_dims2 = generate_titles(v)
        title_dims2 = f"v\n{title_dims2}"

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

        # u_data, u_x = cartopy_util.add_cyclic_point(u_data, u_x)
        u_data, u_x = add_cyclic(u_data, u_x)
        # v_data, v_x = cartopy_util.add_cyclic_point(v_data, v_x)
        v_data, v_x = add_cyclic(v_data, v_x)

    # stride data points to reduce vector density
    if stride is not None:
        if np.size(stride) == 1:
            xstride = stride
            ystride = stride
        if np.size(stride) == 2:
            xstride = stride[0]
            ystride = stride[1]

        u_x = u_x[0::xstride]
        u_y = u_y[0::ystride]
        u_data = u_data[0::ystride, 0::xstride]
        v_data = v_data[0::ystride, 0::xstride]

    # Map vectors
    if plotvars.plot_type == 1:
        lonmax = plotvars.lonmax
        proj = ccrs.PlateCarree()

        # Fix for high latitude vectors as described at
        # https://github.com/SciTools/cartopy/issues/1179
        if plotvars.proj != "cyl":
            u_src_crs = u_data / np.cos(u_y[:, np.newaxis] / 180 * np.pi)
            v_src_crs = v_data
            magnitude = np.ma.sqrt(u_data**2 + v_data**2)
            magn_src_crs = np.ma.sqrt(u_src_crs**2 + v_src_crs**2)

            u_data = u_src_crs * magnitude / magn_src_crs
            v_data = v_src_crs * magnitude / magn_src_crs

        if pts is None:
            quiv = plotvars.mymap.quiver(
                u_x,
                u_y,
                u_data,
                v_data,
                scale=scale,
                pivot=pivot,
                units="inches",
                width=width,
                headwidth=headwidth,
                headlength=headlength,
                headaxislength=headaxislength,
                color=color,
                transform=proj,
                alpha=alpha,
                zorder=zorder,
            )
        else:
            if plotvars.proj == "cyl":
                # **cartopy 0.16 fix for longitide points in cylindrical
                # projection when regridding to a number of points
                # Make points within the plotting region
                for pt in np.arange(np.size(u_x)):
                    if u_x[pt] > lonmax:
                        u_x[pt] = u_x[pt] - 360

            quiv = plotvars.mymap.quiver(
                u_x,
                u_y,
                u_data,
                v_data,
                scale=scale,
                pivot=pivot,
                units="inches",
                width=width,
                headwidth=headwidth,
                headlength=headlength,
                headaxislength=headaxislength,
                color=color,
                regrid_shape=pts,
                transform=proj,
                alpha=alpha,
                zorder=zorder,
            )

        # Make key_label if none exists
        if key_label is None:
            key_label = str(key_length)
        if isinstance(u, cf.Field):
            key_label = _supscr(key_label + u.units)
        if key_show:
            plotvars.mymap.quiverkey(
                quiv,
                key_location[0],
                key_location[1],
                key_length,
                key_label,
                labelpos="W",
                color=qkey_color,
                fontproperties={"size": str(plotvars.axis_label_fontsize)},
                coordinates="axes",
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

        # Titles for dimensions
        if titles:
            if plotvars.titles_con_called is False:
                _dim_titles(title=title_dims, title2=title_dims2)
            else:
                _dim_titles(title2=title_dims, title3=title_dims2)

    if plotvars.plot_type == 6:
        if u.ref("grid_mapping_name:rotated_latitude_longitude", False):
            proj = ccrs.PlateCarree()

            # Set up mapping
            if (
                lonrange > 350 and latrange > 170
            ) or plotvars.user_mapset == 1:
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

            quiv = plotvars.mymap.quiver(
                u_x,
                u_y,
                u_data,
                v_data,
                scale=scale * 10,
                transform=proj,
                pivot=pivot,
                units="inches",
                width=width,
                headwidth=headwidth,
                headlength=headlength,
                headaxislength=headaxislength,
                color=color,
                alpha=alpha,
                zorder=zorder,
            )

            # Make key_label if none exists
            if key_label is None:
                key_label = str(key_length)
            if isinstance(u, cf.Field):
                key_label = _supscr(key_label + u.units)

            if key_show:
                plotvars.mymap.quiverkey(
                    quiv,
                    key_location[0],
                    key_location[1],
                    key_length,
                    key_label,
                    labelpos="W",
                    color=qkey_color,
                    fontproperties={"size": str(plotvars.axis_label_fontsize)},
                    coordinates="axes",
                )

            # Axes on the native grid
            if plotvars.plot == "rotated":
                rgaxes(
                    xpole=xpole,
                    ypole=ypole,
                    xvec=x,
                    yvec=y,
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

            if plotvars.plot == "cyl":
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

            # Title
            if title is not None:
                _map_title(title)

            # Titles for dimensions
            if titles:
                _dim_titles(title=title_dims, titles2=title_dims2)

    ######################################
    # Latitude or longitude vs height plot
    ######################################
    if plotvars.plot_type == 2 or plotvars.plot_type == 3:

        user_gset = plotvars.user_gset
        if user_gset == 0:
            # Program selected data plot limits
            xmin = np.nanmin(u_x)
            xmax = np.nanmax(u_x)
            if plotvars.plot_type == 2:
                if xmin < -80 and xmin >= -90:
                    xmin = -90
                if xmax > 80 and xmax <= 90:
                    xmax = 90
            ymin = np.nanmin(u_y)
            if ymin <= 10:
                ymin = 0
            ymax = np.nanmax(u_y)
        else:
            # User specified plot limits
            xmin = plotvars.xmin
            xmax = plotvars.xmax
            if plotvars.ymin < plotvars.ymax:
                ymin = plotvars.ymin
                ymax = plotvars.ymax
            else:
                ymin = plotvars.ymax
                ymax = plotvars.ymin

        ystep = None
        if ymax == 1000:
            ystep = 100
        if ymax == 100000:
            ystep = 10000

        ytype = 0  # pressure or similar y axis
        if "theta" in ylabel.split(" "):
            ytype = 1
        if "height" in ylabel.split(" "):
            ytype = 1
            ystep = 100
            if (ymax - ymin) > 5000:
                ystep = 500.0
            if (ymax - ymin) > 10000:
                ystep = 1000.0
            if (ymax - ymin) > 50000:
                ystep = 10000.0

        # Set plot limits and draw axes
        if ylog != 1:
            if ytype == 1:
                gset(
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    user_gset=user_gset,
                )
            else:
                gset(
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymax,
                    ymax=ymin,
                    user_gset=user_gset,
                )

            # Set default x-axis labels
            lltype = 1
            if plotvars.plot_type == 2:
                lltype = 2
            llticks, lllabels = _mapaxis(min=xmin, max=xmax, type=lltype)

            heightticks = _gvals(
                dmin=ymin, dmax=ymax, mystep=ystep, mod=False
            )[0]
            heightlabels = heightticks

            if axes:
                if xaxis:
                    if xticks is not None:
                        llticks = xticks
                        lllabels = xticks
                        if xticklabels is not None:
                            lllabels = xticklabels
                else:
                    llticks = [100000000]
                    xlabel = ""

                if yaxis:
                    if yticks is not None:
                        heightticks = yticks
                        heightlabels = yticks
                        if yticklabels is not None:
                            heightlabels = yticklabels
                else:
                    heightticks = [100000000]
                    ylabel = ""

            else:
                llticks = [100000000]
                heightticks = [100000000]
                xlabel = ""
                ylabel = ""

            axes_plot(
                xticks=llticks,
                xticklabels=lllabels,
                yticks=heightticks,
                yticklabels=heightlabels,
                xlabel=xlabel,
                ylabel=ylabel,
            )

        # Log y axis
        if ylog:
            if ymin == 0:
                ymin = 1  # reset zero mb/height input to a small value
            gset(
                xmin=xmin,
                xmax=xmax,
                ymin=ymax,
                ymax=ymin,
                ylog=1,
                user_gset=user_gset,
            )
            llticks, lllabels = _mapaxis(
                min=xmin, max=xmax, type=plotvars.plot_type
            )

            if axes:
                if xaxis:
                    if xticks is not None:
                        llticks = xticks
                        lllabels = xticks
                        if xticklabels is not None:
                            lllabels = xticklabels
                else:
                    llticks = [100000000]
                    xlabel = ""

                if yaxis:
                    if yticks is not None:
                        heightticks = yticks
                        heightlabels = yticks
                        if yticklabels is not None:
                            heightlabels = yticklabels
                else:
                    heightticks = [100000000]
                    ylabel = ""

                if yticks is None:
                    axes_plot(
                        xticks=llticks,
                        xticklabels=lllabels,
                        xlabel=xlabel,
                        ylabel=ylabel,
                    )
                else:
                    axes_plot(
                        xticks=llticks,
                        xticklabels=lllabels,
                        yticks=heightticks,
                        yticklabels=heightlabels,
                        xlabel=xlabel,
                        ylabel=ylabel,
                    )

        # Regrid the data if requested
        if pts is not None:

            xnew, ynew = stipple_points(
                xmin=np.min(u_x),
                xmax=np.max(u_x),
                ymin=np.min(u_y),
                ymax=np.max(u_y),
                pts=pts,
                stype=1,
            )

            if ytype == 0:
                # Make y interpolation in log space as we have a pressure
                # coordinate
                u_vals = regrid(
                    f=u_data,
                    x=u_x,
                    y=np.log10(u_y),
                    xnew=xnew,
                    ynew=np.log10(ynew),
                )
                v_vals = regrid(
                    f=v_data,
                    x=u_x,
                    y=np.log10(u_y),
                    xnew=xnew,
                    ynew=np.log10(ynew),
                )
            else:
                u_vals = regrid(f=u_data, x=u_x, y=u_y, xnew=xnew, ynew=ynew)
                v_vals = regrid(f=v_data, x=u_x, y=u_y, xnew=xnew, ynew=ynew)

            u_x = xnew
            u_y = ynew
            u_data = u_vals
            v_data = v_vals

        # set scale and key lengths
        if np.size(scale) == 1:
            scale_u = scale
            scale_v = scale
        else:
            scale_u = scale[0]
            scale_v = scale[1]

        if np.size(key_length) == 2:
            key_length_u = key_length[0]
            key_length_v = key_length[1]
            # scale v data
            v_data = v_data * scale_u / scale_v
        else:
            key_length_u = key_length

        # Plot the vectors
        quiv = plotvars.plot.quiver(
            u_x,
            u_y,
            u_data,
            v_data,
            pivot=pivot,
            units="inches",
            scale=scale_u,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            headaxislength=headaxislength,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )

        # Plot single key
        if np.size(scale) == 1:
            # Single scale vector
            if key_label is None:
                key_label_u = str(key_length_u)
                if isinstance(u, cf.Field):
                    key_label_u = _supscr(f"{key_label_u} ({u.units})")
            else:
                key_label_u = key_label[0]
            if key_show:
                plotvars.plot.quiverkey(
                    quiv,
                    key_location[0],
                    key_location[1],
                    key_length_u,
                    key_label_u,
                    labelpos="W",
                    color=qkey_color,
                    fontproperties={"size": str(plotvars.axis_label_fontsize)},
                )

        # Plot two keys
        if np.size(scale) == 2:
            # translate from normalised units to plot units
            xpos = (
                key_location[0] * (plotvars.xmax - plotvars.xmin)
                + plotvars.xmin
            )
            ypos = (
                key_location[1] * (plotvars.ymax - plotvars.ymin)
                + plotvars.ymin
            )

            # horizontal and vertical spacings for offsetting vector reference
            # text
            xoffset = 0.01 * abs(plotvars.xmax - plotvars.xmin)
            yoffset = 0.01 * abs(plotvars.ymax - plotvars.ymin)

            # Assign key labels if necessary
            if key_label is None:
                key_label_u = str(key_length_u)
                key_label_v = str(key_length_v)
                if isinstance(u, cf.Field):
                    key_label_u = _supscr(f"{key_label_u} ({u.units})")
                if isinstance(v, cf.Field):
                    key_label_v = _supscr(f"{key_label_v} ({v.units})")
            else:
                key_label_u = _supscr(key_label[0])
                key_label_v = _supscr(key_label[1])

            # Plot reference vectors and keys
            if key_show:
                plotvars.plot.quiver(
                    xpos,
                    ypos,
                    key_length[0],
                    0,
                    pivot="tail",
                    units="inches",
                    scale=scale[0],
                    headaxislength=headaxislength,
                    width=width,
                    headwidth=headwidth,
                    headlength=headlength,
                    clip_on=False,
                    color=qkey_color,
                )

                plotvars.plot.quiver(
                    xpos,
                    ypos,
                    0,
                    key_length[1],
                    pivot="tail",
                    units="inches",
                    scale=scale[1],
                    headaxislength=headaxislength,
                    width=width,
                    headwidth=headwidth,
                    headlength=headlength,
                    clip_on=False,
                    color=qkey_color,
                )

                plotvars.plot.text(
                    xpos,
                    ypos + yoffset,
                    key_label_u,
                    horizontalalignment="left",
                    verticalalignment="top",
                )
                plotvars.plot.text(
                    xpos - xoffset,
                    ypos,
                    key_label_v,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

        if title is not None:
            plotvars.plot.set_title(
                title,
                y=1.03,
                fontsize=plotvars.title_fontsize,
                fontweight=title_fontweight,
            )

            # Titles for dimensions
            if titles:
                _dim_titles(title=title_dims, titles2=title_dims2)

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


def lineplot(
    f=None,
    x=None,
    y=None,
    fill=True,
    lines=True,
    line_labels=True,
    title=None,
    ptype=0,
    linestyle="-",
    linewidth=1.0,
    color=None,
    xlog=False,
    ylog=False,
    verbose=None,
    swap_xy=False,
    marker=None,
    markersize=5.0,
    markeredgecolor="k",
    markeredgewidth=0.5,
    label=None,
    legend_location="upper right",
    xunits=None,
    yunits=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    xname=None,
    yname=None,
    axes=True,
    xaxis=True,
    yaxis=True,
    titles=False,
    zorder=None,
):
    """
    | The interface to line plotting in cf-plot.
    |
    | The minimum use is lineplot(f) where f is a CF field.
    | If x and y are passed then an appropriate plot is made allowing
    | x vs data and y vs data plots.

    | When making a labelled line plot:
    | always have a label for each line
    | always put the legend location as an option to the last call to lineplot
    |
    | f - CF data used to make a line plot
    | x - x locations of data in y
    | y - y locations of data in x
    | linestyle='-' - line style
    | color=None - line color. Defaults to Matplotlib colour scheme
    |              unless specified
    | linewidth=1.0 - line width
    | marker=None - marker for points along the line
    | markersize=5.0 - size of the marker
    | markeredgecolor = 'k' - colour of edge around the marker
    | markeredgewidth = 0.5 - width of edge around the marker
    | xlog=False - log x-axis
    | ylog=False - log y-axis
    | label=None - line label - label for line
    | legend_location='upper right' - default location of legend
    |                 Other options are {'best': 0, 'center': 10,
    |                                    'center left': 6,
    |                                    'center right': 7, 'lower center': 8,
    |                                    'lower left': 3, 'lower right': 4,
    |                                    'right': 5,
    |                                    'upper center': 9, 'upper left': 2,
    |                                    'upper right': 1}
    | titles=False - set to True to have a dimensions title
    | verbose=None - change to 1 to get a verbose listing of what lineplot
    |                is doing
    | zorder=None - plotting order
    |
    | The following parameters override any CF data defaults:
    | title=None - plot title
    | xunits=None - x units
    | yunits=None - y units
    | xlabel=None - x name
    | ylabel=None - y name
    | xname=None - deprecated keyword
    | yname=None - deprecated keyword
    | xticks=None - x ticks
    | xticklabels=None - x tick labels
    | yticks=None - y ticks
    | yticklabels - y tick labels
    | axes=True - plot x and y axes
    | xaxis=True - plot xaxis
    | yaxis=True - plot y axis
    |
    |
    | When making a multiple line plot:
    | a) set the axis limits with gset before plotting the lines
    | b) the last call to lineplot is the one that any of the above
    |    axis overrides should be placed in.
    |
    """
    if verbose:
        print("lineplot - making a line plot")

    # Catch deprecated keywords
    if xname is not None or yname is not None:
        print(
            "\nlineplot error\n"
            "xname and yname are now deprecated keywords\n"
            "Please use xlabel and ylabel\n"
        )
        return

    ##################
    # Open a new plot is necessary
    ##################
    if plotvars.user_plot == 0:
        gopen(user_plot=0)

    # Call gpos(1) if not already called
    if plotvars.rows > 1 or plotvars.columns > 1:
        if plotvars.gpos_called is False:
            gpos(1)

    ##################
    # Extract required data
    # If a cf-python field
    ##################
    cf_field = False
    if f is not None:
        if isinstance(f, cf.Field):
            cf_field = True

            # Check data is 1D
            ndims = np.squeeze(f.data).ndim
            if ndims != 1:
                errstr = (
                    "\n\ncfp.lineplot error need a 1 dimensonal field to "
                    "make a line\n"
                    f"received {np.squeeze(f.data).ndim} dimensions\n\n"
                )
                raise TypeError(errstr)

            if x is not None:
                if isinstance(x, cf.Field):
                    errstr = (
                        "\n\ncfp.lineplot error - two or more cf-fields "
                        "passed for plotting.\n"
                        "To plot two cf-fields open a graphics plot with "
                        "cfp.gopen(), \n"
                        "plot the two fields separately with cfp.lineplot "
                        "and then close\n"
                        "the graphics plot with cfp.gclose()\n\n"
                    )
                    raise TypeError(errstr)

        elif isinstance(f, cf.FieldList):
            errstr = "\n\ncfp.lineplot - cannot plot a field list\n\n"
            raise TypeError(errstr)

    plot_xlabel = ""
    plot_ylabel = ""
    xlabel_units = ""
    ylabel_units = ""

    if cf_field:
        # Extract data
        if verbose:
            print("lineplot - CF field, extracting data")

        has_count = 0
        for mydim in list(f.dimension_coordinates()):
            if np.size(np.squeeze(f.construct(mydim).array)) > 1:
                has_count = has_count + 1
                x = np.squeeze(f.construct(mydim).array)

                # x label
                xlabel_units = str(getattr(f.construct(mydim), "Units", ""))
                plot_xlabel = (
                    f"{cf_var_name(field=f, dim=mydim)} ({xlabel_units})"
                )
                y = np.squeeze(f.array)

                # y label
                if hasattr(f, "id"):
                    plot_ylabel = f.id
                nc = f.nc_get_variable(False)
                if nc:
                    plot_ylabel = f.nc_get_variable()
                if hasattr(f, "short_name"):
                    plot_ylabel = f.short_name
                if hasattr(f, "long_name"):
                    plot_ylabel = f.long_name
                if hasattr(f, "standard_name"):
                    plot_ylabel = f.standard_name

                if hasattr(f, "Units"):
                    ylabel_units = str(f.Units)
                else:
                    ylabel_units = ""
                plot_ylabel += f" ({ylabel_units})"

        if has_count != 1:
            errstr = (
                "\n lineplot error - passed field is not suitable "
                "for plotting as a line\n"
            )
            for mydim in list(f.dimension_coordinates()):
                sn = getattr(f.construct(mydim), "standard_name", False)
                ln = getattr(f.construct(mydim), "long_name", False)
                if sn:
                    errstr += f"{mydim},{sn},{f.construct(mydim).size}\n"
                else:
                    # TODO SLB: replace simple 'else, if' statements such as
                    # this one, with many other examples in codebase, w/ 'elif'
                    if ln:
                        errstr += f"{mydim},{ln},{f.construct(mydim).size}\n"
            raise Warning(errstr)
    else:
        if verbose:
            print("lineplot - not a CF field, using passed data")
        errstr = ""
        if x is None or y is None:
            errstr = "lineplot error- must define both x and y"
        if f is not None:
            errstr += (
                "lineplot error- must define just x and y to make "
                "a lineplot"
            )
        if errstr != "":
            raise Warning(f"\n{errstr}\n")

    # Z on y-axis
    ztype = None
    if xlabel_units in [
        "mb",
        "mbar",
        "millibar",
        "decibar",
        "atmosphere",
        "atm",
        "pascal",
        "Pa",
        "hPa",
    ]:
        ztype = 1
    if xlabel_units in ["meter", "metre", "m", "kilometer", "kilometre", "km"]:
        ztype = 2

    myz = find_z(f)
    if cf_field and f.has_construct(myz):
        z_coord = f.construct(myz)
        if len(z_coord.array) > 1:
            zlabel = ""
            if hasattr(z_coord, "long_name"):
                zlabel = z_coord.long_name
            if hasattr(z_coord, "standard_name"):
                zlabel = z_coord.standard_name
            if zlabel == "atmosphere_hybrid_height_coordinate":
                ztype = 2

    if ztype is not None:
        x, y = y, x
        plot_xlabel, plot_ylabel = plot_ylabel, plot_xlabel

    # Set data values
    if verbose:
        print("lineplot - setting data values")
    xpts = np.squeeze(x)
    ypts = np.squeeze(y)
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)

    # Use accumulated plot limits if making a multiple line plot
    if plotvars.graph_xmin is None:
        plotvars.graph_xmin = minx
    else:
        if minx < plotvars.graph_xmin:
            plotvars.graph_xmin = minx

    if plotvars.graph_xmax is None:
        plotvars.graph_xmax = maxx
    else:
        if maxx > plotvars.graph_xmax:
            plotvars.graph_xmax = maxx

    if plotvars.graph_ymin is None:
        plotvars.graph_ymin = miny
    else:
        if miny < plotvars.graph_ymin:
            plotvars.graph_ymin = miny

    if plotvars.graph_ymax is None:
        plotvars.graph_ymax = maxy
    else:
        if maxy > plotvars.graph_ymax:
            plotvars.graph_ymax = maxy

    # Reset plot limits based on accumulated plot limits
    minx = plotvars.graph_xmin
    maxx = plotvars.graph_xmax
    miny = plotvars.graph_ymin
    maxy = plotvars.graph_ymax

    if cf_field and f.has_construct("T"):
        taxis = f.construct("T")

    if ztype == 1:
        miny = np.max(y)
        maxy = np.min(y)

    if ztype == 2:
        if cf_field and f.has_construct("Z"):
            if f.construct("Z").positive == "down":
                miny = np.max(y)
                maxy = np.min(y)

    # Use user set values if present
    time_xstr = False
    time_ystr = False
    if plotvars.xmin is not None:
        minx = plotvars.xmin
        miny = plotvars.ymin
        maxx = plotvars.xmax
        maxy = plotvars.ymax

        # Change from date string to a number if strings are passed
        try:
            float(minx)
        except Exception:
            time_xstr = True
        try:
            float(miny)
        except Exception:
            time_ystr = True

        if cf_field and f.has_construct("T"):
            taxis = f.construct("T")
            if time_xstr or time_ystr:
                ref_time = f.construct("T").units
                ref_calendar = f.construct("T").calendar
                time_units = cf.Units(ref_time, ref_calendar)

                if time_xstr:
                    t = cf.Data(cf.dt(minx), units=time_units)
                    minx = t.array
                    t = cf.Data(cf.dt(maxx), units=time_units)
                    maxx = t.array
                    taxis = cf.Data(
                        [cf.dt(plotvars.xmin), cf.dt(plotvars.xmax)],
                        units=time_units,
                    )

                if time_ystr:
                    t = cf.Data(cf.dt(miny), units=time_units)
                    miny = t.array
                    t = cf.Data(cf.dt(maxy), units=time_units)
                    maxy = t.array
                    taxis = cf.Data(
                        [cf.dt(plotvars.ymin), cf.dt(plotvars.ymax)],
                        units=time_units,
                    )

    # Set x and y labelling
    # Retrieve any user defined axis labels
    if plot_xlabel == "" and plotvars.xlabel is not None:
        plot_xlabel = plotvars.xlabel
    if plot_ylabel == "" and plotvars.ylabel is not None:
        plot_ylabel = plotvars.ylabel
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

    mod = False
    ymult = 0

    if xticks is None:
        if plot_xlabel[0:3].lower() == "lon":
            xticks, xticklabels = _mapaxis(minx, maxx, type=1)
        if plot_xlabel[0:3].lower() == "lat":
            xticks, xticklabels = _mapaxis(minx, maxx, type=2)
    if cf_field:
        if xticks is None:
            if f.has_construct("T"):
                if np.size(f.construct("T").array) > 1:
                    xticks, xticklabels, plot_xlabel = _timeaxis(taxis)
    if xticks is None:
        xticks, ymult = _gvals(dmin=minx, dmax=maxx, mod=mod)

        # Fix long floating point numbers if necessary
        fix_floats(xticks)
        xticklabels = xticks
    else:
        if xticklabels is None:
            xticklabels = []
            for val in xticks:
                xticklabels.append(f"{val}")

    if yticks is None:
        if abs(maxy - miny) > 1:
            if miny < maxy:
                yticks, ymult = _gvals(dmin=miny, dmax=maxy, mod=mod)
            if maxy < miny:
                yticks, ymult = _gvals(dmin=maxy, dmax=miny, mod=mod)

        else:
            yticks, ymult = _gvals(dmin=miny, dmax=maxy, mod=mod)

            # Fix long floating point numbers if necessary
            fix_floats(yticks)

    if yticklabels is None:
        yticklabels = []

        for val in yticks:
            yticklabels.append(str(round(val, 9)))

    if xlabel is not None:
        plot_xlabel = xlabel
        if xunits is not None:
            plot_xlabel += f"({xunits})"

    if ylabel is not None:
        plot_ylabel = ylabel
        if yunits is not None:
            plot_ylabel += f"({yunits})"

    if swap_xy:
        if verbose:
            print("lineplot - swapping x and y")

        xpts, ypts = ypts, xpts
        minx, miny = miny, minx
        maxx, maxy = maxy, maxx
        plot_xlabel, plot_ylabel = plot_ylabel, plot_xlabel
        xticks, yticks = yticks, xticks
        xticklabels, yticklabels = yticklabels, xticklabels

    if plotvars.user_gset == 1:
        if time_xstr is False and time_ystr is False:
            minx = plotvars.xmin
            maxx = plotvars.xmax
            miny = plotvars.ymin
            maxy = plotvars.ymax

    if axes:
        if xaxis is not True:
            xticks = [100000000]
            xticklabels = xticks
            plot_xlabel = ""

        if yaxis is not True:
            yticks = [100000000]
            yticklabels = yticks
            plot_ylabel = ""

    else:
        xticks = [100000000]
        xticklabels = xticks
        yticks = [100000000]
        yticklabels = yticks
        plot_xlabel = ""
        plot_ylabel = ""

    # Generate titles if requested
    if titles:
        title_dims = generate_titles(f)

    # Make graph
    if verbose:
        print("lineplot - making graph")

    xlabelalignment = plotvars.xtick_label_align
    ylabelalignment = plotvars.ytick_label_align

    if lines is False:
        linewidth = 0.0

    colorarg = {}
    if color is not None:
        colorarg = {"color": color}

    graph = plotvars.plot

    if plotvars.twinx:
        graph = graph.twinx()
        ylabelalignment = "left"

    if plotvars.twiny:
        graph = graph.twiny()

    # Reset y limits if minx = maxy
    if plotvars.xmin is None:
        if miny == maxy:
            miny = miny - 1.0
            maxy = maxy + 1.0

    graph.axis([minx, maxx, miny, maxy])
    graph.tick_params(direction="out", which="both", right=True, top=True)
    graph.set_xlabel(
        plot_xlabel,
        fontsize=plotvars.axis_label_fontsize,
        fontweight=plotvars.axis_label_fontweight,
    )
    graph.set_ylabel(
        plot_ylabel,
        fontsize=plotvars.axis_label_fontsize,
        fontweight=plotvars.axis_label_fontweight,
    )

    if plotvars.xlog or xlog:
        graph.set_xscale("log")
    if plotvars.ylog or ylog:
        graph.set_yscale("log")

    if xticks is not None:
        graph.set_xticks(xticks)
        graph.set_xticklabels(
            xticklabels,
            rotation=plotvars.xtick_label_rotation,
            horizontalalignment=xlabelalignment,
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
        )
    if yticks is not None:
        graph.set_yticks(yticks)
        graph.set_yticklabels(
            yticklabels,
            rotation=plotvars.ytick_label_rotation,
            horizontalalignment=ylabelalignment,
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
        )

    graph.plot(
        xpts,
        ypts,
        **colorarg,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        markeredgewidth=markeredgewidth,
        label=label,
        zorder=zorder,
    )

    # Set axis width if required
    if plotvars.axis_width is not None:
        for axis in ["top", "bottom", "left", "right"]:
            plotvars.plot.spines[axis].set_linewidth(plotvars.axis_width)

    # Add a legend if needed
    if label is not None:
        legend_properties = {
            "size": plotvars.legend_text_size,
            "weight": plotvars.legend_text_weight,
        }
        graph.legend(
            loc=legend_location,
            prop=legend_properties,
            frameon=plotvars.legend_frame,
            edgecolor=plotvars.legend_frame_edge_color,
            facecolor=plotvars.legend_frame_face_color,
        )

    # Set title
    if title is not None:
        graph.set_title(
            title,
            fontsize=plotvars.title_fontsize,
            fontweight=plotvars.title_fontweight,
        )

    # Titles for dimensions
    if titles:
        plotvars.plot = graph
        plotvars.plot_type = 0
        _dim_titles(title=title_dims)

    ##################
    # Save or view plot
    ##################
    if plotvars.user_plot == 0:
        if verbose:
            print("Saving or viewing plot")
        gclose()


def traj(
    f=None,
    title=None,
    ptype=0,
    linestyle="-",
    linewidth=1.0,
    linecolor="b",
    marker="o",
    markevery=1,
    markersize=5.0,
    markerfacecolor="r",
    markeredgecolor="g",
    markeredgewidth=1.0,
    latmax=None,
    latmin=None,
    axes=True,
    xaxis=True,
    yaxis=True,
    verbose=None,
    legend=False,
    legend_lines=False,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    colorbar=None,
    colorbar_position=None,
    colorbar_orientation="horizontal",
    colorbar_title=None,
    colorbar_text_up_down=False,
    colorbar_text_down_up=False,
    colorbar_drawedges=True,
    colorbar_fraction=None,
    colorbar_thick=None,
    colorbar_anchor=None,
    colorbar_shrink=None,
    colorbar_labels=None,
    vector=False,
    head_width=0.4,
    head_length=1.0,
    fc="k",
    ec="k",
    zorder=None,
):
    """
    | The interface to trajectory plotting in cf-plot.
    |
    | The minimum use is traj(f) where f is a CF field.
    |
    | f - CF data used to make a line plot
    | linestyle='-' - line style
    | linecolor='b' - line colour
    | linewidth=1.0 - line width
    | marker='o' - marker for points along the line
    | markersize=30 - size of the marker
    | markerfacecolor='b' - colour of the marker face
    | markeredgecolor='g' - colour of the marker edge
    | legend=False - plot different colour markers based on a set of user
    |                levels
    | zorder=None - order for plotting
    | verbose=None - Set to True to get a verbose listing of what traj is doing
    |
    | The following parameters override any CF data defaults:
    | title=None - plot title
    | axes=True - plot x and y axes
    | xaxis=True - plot xaxis
    | yaxis=True - plot y axis
    | xlabel=None - x name
    | ylabel=None - y name
    | xticks=None - x ticks
    | xticklabels=None - x tick labels
    | yticks=None - y ticks
    | yticklabels=None - y tick labels
    | colorbar=None - plot a colorbar
    | colorbar_position=None - position of colorbar
    |                          [xmin, ymin, x_extent,y_extent] in normalised
    |                          coordinates. Use when a common colorbar
    |                          is required for a set of plots. A typical set
    |                          of values would be [0.1, 0.05, 0.8, 0.02]
    | colorbar_orientation=None - orientation of the colorbar
    | colorbar_title=None - title for the colorbar
    | colorbar_text_up_down=False - if True horizontal colour bar labels
    |                               alternate above (start) and below the
    |                               colour bar
    | colorbar_text_down_up=False - if True horizontal colour bar labels
    |                               alternate below (start) and above the
    |                               colour bar
    | colorbar_drawedges=True - draw internal divisions in the colorbar
    | colorbar_fraction=None - space for the colorbar - default = 0.21, in
    |                          normalised coordinates
    | colorbar_thick=None - thickness of the colorbar - default = 0.015, in
    |                       normalised coordinates
    | colorbar_anchor=None - default=0.5 - anchor point of colorbar within
    |                        the fraction space. 0.0 = close to plot,
    |                        1.0 = further away
    | colorbar_shrink=None - value to shrink the colorbar by.  If the colorbar
    |                        exceeds the plot area then values of 1.0, 0.55
    |                        or 0.5m ay help it better fit the plot area.
    | colorbar_labels=None - labels for the colorbar. Default is to use the
    |                        levels defined
    |                        using cfp.levs
    | Vector options
    | vector=False - Draw vectors
    | head_width=2.0 - vector head width
    | head_length=2.0 - vector head length
    | fc='k' - vector face colour
    | ec='k' - vector edge colour

    """
    is_1d_data = False
    if f.ndim == 1:
        is_1d_data = True

    is_dsg = False
    if f.DSG or is_1d_data:  # registered as a DSG or if is 1D, assume is DSG
        is_dsg = True

    if verbose:
        print("traj - making a trajectory plot")

    if isinstance(f, cf.FieldList):
        errstr = (
            "\n\ncfp.traj - cannot make a trajectory plot from a field list "
            "- need to pass a field\n\n"
        )
        raise TypeError(errstr)

    # Read in data
    # Find the auxiliary lons and lats if provided
    has_lons = False
    has_lats = False
    for mydim in list(f.auxiliary_coordinates()):
        name = cf_var_name(field=f, dim=mydim)
        if name in ["longitude"]:
            lons = np.squeeze(f.construct(mydim).array)
            has_lons = True
        if name in ["latitude"]:
            lats = np.squeeze(f.construct(mydim).array)
            has_lats = True

    data = f.array

    # Raise an error if lons and lats not found in the input data
    if not has_lons or not has_lats:
        message = "\n\n\ntraj error\n"
        if not has_lons:
            message += "missing longitudes in the field auxiliary data\n"
        if not has_lats:
            message += "missing latitudes in the field auxiliary data\n"
        message += "\n\n\n"
        raise TypeError(message)

    if latmax is not None:
        pts = np.where(lats >= latmax)
        if np.size(pts) > 0:
            lons[pts] = np.nan
            lats[pts] = np.nan

    if latmin is not None:
        pts = np.where(lats <= latmin)
        if np.size(pts) > 0:
            lons[pts] = np.nan
            lats[pts] = np.nan

    # Set potential user axis labels
    user_xlabel = xlabel
    user_ylabel = ylabel

    user_xlabel = ""
    user_ylabel = ""

    # Set plotting parameters
    continent_thickness = 1.5
    continent_color = "k"
    continent_linestyle = "-"
    if plotvars.continent_thickness is not None:
        continent_thickness = plotvars.continent_thickness
    if plotvars.continent_color is not None:
        continent_color = plotvars.continent_color
    if plotvars.continent_linestyle is not None:
        continent_linestyle = plotvars.continent_linestyle
    land_color = plotvars.land_color
    ocean_color = plotvars.ocean_color
    lake_color = plotvars.lake_color

    ##################
    # Open a new plot is necessary
    ##################
    if plotvars.user_plot == 0:
        gopen(user_plot=0)

    # Call gpos(1) if not already called
    if plotvars.rows > 1 or plotvars.columns > 1:
        if plotvars.gpos_called is False:
            gpos(1)

    # Set up mapping
    if plotvars.user_mapset == 0:
        plotvars.lonmin = -180
        plotvars.lonmax = 180
        plotvars.latmin = -90
        plotvars.latmax = 90

    _set_map()
    mymap = plotvars.mymap

    # Set the plot limits
    gset(
        xmin=plotvars.lonmin,
        xmax=plotvars.lonmax,
        ymin=plotvars.latmin,
        ymax=plotvars.latmax,
        user_gset=0,
    )

    # Make lons and lats 2d if they are 1d
    ndim = np.ndim(lons)
    if ndim == 1:
        lons = lons.reshape(1, -1)
        lats = lats.reshape(1, -1)

    ntracks = np.shape(lons)[0]
    if ndim == 1:
        ntracks = 1

    if legend or legend_lines:
        # Check levels are not None
        levs = plotvars.levels
        if plotvars.levels is not None:
            if verbose:
                print(
                    "traj - plotting different colour markers based on a "
                    "user set of levels"
                )
            levs = plotvars.levels

        else:
            # Automatic levels
            if verbose:
                print("traj - generating automatic legend levels")
            dmin = np.nanmin(data)
            dmax = np.nanmax(data)
            levs, mult = _gvals(dmin=dmin, dmax=dmax, mod=False)

        # Add extend options to the levels if set
        if plotvars.levels_extend == "min" or plotvars.levels_extend == "both":
            levs = np.append(-1e-30, levs)
        if plotvars.levels_extend == "max" or plotvars.levels_extend == "both":
            levs = np.append(levs, 1e30)

        # Set the default colour scale
        if plotvars.cscale_flag == 0:
            cscale("viridis", ncols=np.size(levs) + 1)
            plotvars.cscale_flag = 0

        # User selected colour map but no mods so fit to levels
        if plotvars.cscale_flag == 1:
            cscale(plotvars.cs_user, ncols=np.size(levs) + 1)
            plotvars.cscale_flag = 1

    ##################################
    # Line, symbol and vector plotting
    ##################################
    for track in np.arange(ntracks):
        xpts = lons[track, :]
        ypts = lats[track, :]
        if is_1d_data:
            data2 = data
        else:
            data2 = data[track, :]

        xpts_orig = deepcopy(xpts)
        xpts = np.mod(xpts + 180, 360) - 180

        # Check if xpts are only within the remapped longitudes above
        if np.min(xpts) < -170 or np.max(xpts) > 170:
            xpts = xpts_orig

            for ix in np.arange(np.size(xpts) - 1):
                diff = xpts[ix + 1] - xpts[ix]
                if diff >= 60:
                    xpts[ix + 1] = xpts[ix + 1] - 360.0
                if diff <= -60:
                    xpts[ix + 1] = xpts[ix + 1] + 360.0

        # Plot lines and markers
        plot_linewidth = linewidth
        plot_markersize = markersize
        if legend:
            plot_markersize = 0.0

        if plot_linewidth > 0.0 or plot_markersize > 0.0:
            if verbose and track == 0 and linewidth > 0.0:
                print("plotting lines")
            if verbose and track == 0 and markersize > 0.0:
                print("plotting markers")

            if legend_lines is False:
                mymap.plot(
                    xpts,
                    ypts,
                    color=linecolor,
                    linewidth=plot_linewidth,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=markevery,
                    markersize=plot_markersize,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=markeredgewidth,
                    zorder=zorder,
                    clip_on=True,
                    transform=ccrs.PlateCarree(),
                )
            else:
                line_xpts = xpts.compressed()
                line_ypts = ypts.compressed()
                line_data = data2.compressed()

                for i in np.arange(np.size(line_xpts) - 1):
                    val = (line_data[i] + line_data[i + 1]) / 2.0

                    col = plotvars.cs[np.max(np.where(val > plotvars.levels))]
                    mymap.plot(
                        line_xpts[i : i + 2],
                        line_ypts[i : i + 2],
                        color=col,
                        linewidth=plot_linewidth,
                        linestyle=linestyle,
                        zorder=zorder,
                        clip_on=True,
                        transform=ccrs.PlateCarree(),
                    )

        # Plot vectors
        if vector:
            if verbose and track == 0:
                print("plotting vectors")
            if zorder is None:
                plot_zorder = 101
            else:
                plot_zorder = zorder
            if plotvars.proj == "cyl":
                if isinstance(xpts, np.ma.MaskedArray):
                    pts = np.ma.MaskedArray.count(xpts)
                else:
                    pts = xpts.size

                for pt in np.arange(pts - 1):
                    mymap.arrow(
                        xpts[pt],
                        ypts[pt],
                        xpts[pt + 1] - xpts[pt],
                        ypts[pt + 1] - ypts[pt],
                        head_width=head_width,
                        head_length=head_length,
                        fc=fc,
                        ec=ec,
                        length_includes_head=True,
                        zorder=plot_zorder,
                        clip_on=True,
                        transform=ccrs.PlateCarree(),
                    )

    # Plot different colour markers based on a user set of levels
    if legend:

        # For polar stereographic plots mask any points outside the
        # plotting limb
        if plotvars.proj == "npstere":
            pts = np.where(lats < plotvars.boundinglat)
            if np.size(pts) > 0:
                lats[pts] = np.nan

        if plotvars.proj == "spstere":
            pts = np.where(lats > plotvars.boundinglat)
            if np.size(pts) > 0:
                lats[pts] = np.nan

        for track in np.arange(ntracks):
            xpts = lons[track, :]
            ypts = lats[track, :]
            if is_1d_data:
                data2 = data
            else:
                data2 = data[track, :]

            for i in np.arange(np.size(levs) - 1):
                color = plotvars.cs[i]

                if np.ma.is_masked(data2):
                    pts = np.ma.where(
                        np.logical_and(data2 >= levs[i], data2 <= levs[i + 1])
                    )
                else:
                    pts = np.where(
                        np.logical_and(data2 >= levs[i], data2 <= levs[i + 1])
                    )

                if zorder is None:
                    plot_zorder = 101
                else:
                    plot_zorder = zorder
                if np.size(pts) > 0:

                    # Define the data to plot
                    if is_dsg or is_1d_data:
                        data_colours = [
                            plotvars.cs[
                                np.max(np.where(d > plotvars.levels))
                            ] for d in data2[pts]
                        ]
                    else:
                        data_colours = color

                    mymap.scatter(
                        xpts[pts],
                        ypts[pts],
                        s=markersize,
                        c=data_colours,
                        marker=marker,
                        edgecolors=markeredgecolor,
                        linewidths=plot_linewidth,
                        transform=ccrs.PlateCarree(),
                        zorder=plot_zorder,
                    )

    # Axes
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
        verbose=verbose,
    )

    # Coastlines
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

    if ocean_color is not None:
        mymap.add_feature(
            cfeature.OCEAN,
            edgecolor="face",
            facecolor=ocean_color,
            zorder=plotvars.feature_zorder,
        )
    if land_color is not None:
        mymap.add_feature(
            cfeature.LAND,
            edgecolor="face",
            facecolor=land_color,
            zorder=plotvars.feature_zorder,
        )
    if lake_color is not None:
        mymap.add_feature(
            cfeature.LAKES,
            edgecolor="face",
            facecolor=lake_color,
            zorder=plotvars.feature_zorder,
        )

    # Title
    if title is not None:
        _map_title(title)

    # Color bar
    plot_colorbar = False
    if colorbar is None and legend:
        plot_colorbar = True
    if colorbar is None and legend_lines:
        plot_colorbar = True
    if colorbar:
        plot_colorbar = True

    if plot_colorbar:
        if colorbar_title is None:
            colorbar_title = "No Name"
            if hasattr(f, "id"):
                colorbar_title = f.id
            nc = f.nc_get_variable(False)
            if nc:
                colorbar_title = f.nc_get_variable()
            if hasattr(f, "short_name"):
                colorbar_title = f.short_name
            if hasattr(f, "long_name"):
                colorbar_title = f.long_name
            if hasattr(f, "standard_name"):
                colorbar_title = f.standard_name

            if hasattr(f, "Units"):
                if str(f.Units) == "":
                    colorbar_title += ""
                else:
                    colorbar_title += f"({_supscr(str(f.Units))})"

        levs = plotvars.levels
        if colorbar_labels is not None:
            levs = colorbar_labels
        cbar(
            levs=levs,
            labels=levs,
            orientation=colorbar_orientation,
            position=colorbar_position,
            text_up_down=colorbar_text_up_down,
            text_down_up=colorbar_text_down_up,
            drawedges=colorbar_drawedges,
            fraction=colorbar_fraction,
            thick=colorbar_thick,
            shrink=colorbar_shrink,
            anchor=colorbar_anchor,
            title=colorbar_title,
            verbose=verbose,
        )

    ##########
    # Save plot
    ##########
    if plotvars.user_plot == 0:
        gclose()


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

