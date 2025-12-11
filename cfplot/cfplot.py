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

