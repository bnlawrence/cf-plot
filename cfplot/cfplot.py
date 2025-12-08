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
import shapely.geometry as sgeom
from matplotlib.collections import PolyCollection

from scipy.interpolate import griddata

from .colour import (
    cbar,
)
from .graphic import (
    gclose,
    gopen,
    gpos,
)
from .internal import (
    axes_plot,
    find_dim_names,
    find_z,
    _bfill,
    _bfill_ugrid,
    _cf_data_assign,
    _check_data,
    _cscale_get_map,
    _dim_titles,
    _gvals,
    _map_title,
    _mapaxis,
    _plot_map_axes,
    _set_map,
    _supscr,
    _timeaxis,
    _which,
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

# Check for the minimum cf-python version
cf_version_min = "3.17.0"
errstr = (
    f"\n\n cf-python > {cf_version_min}"
    "\n needs to be installed to use cf-plot \n\n"
)
try:
    import cf

    if StrictVersion(cf.__version__) < StrictVersion(cf_version_min):
        raise Warning(errstr)
except ImportError:
    raise Warning(errstr)


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


def con(
    f=None,
    x=None,
    y=None,
    fill=global_fill,
    lines=global_lines,
    line_labels=True,
    title=None,
    colorbar_title=None,
    colorbar=True,
    colorbar_label_skip=None,
    ptype=0,
    negative_linestyle="solid",
    blockfill=global_blockfill,
    zero_thick=False,
    colorbar_shrink=None,
    colorbar_orientation=None,
    colorbar_position=None,
    xlog=False,
    ylog=False,
    axes=True,
    xaxis=True,
    yaxis=True,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    colors="k",
    swap_axes=False,
    verbose=None,
    linewidths=None,
    alpha=1.0,
    colorbar_text_up_down=False,
    colorbar_fontsize=None,
    colorbar_fontweight=None,
    colorbar_text_down_up=False,
    colorbar_drawedges=True,
    colorbar_fraction=None,
    colorbar_thick=None,
    colorbar_anchor=None,
    colorbar_labels=None,
    linestyles=None,
    zorder=1,
    level_spacing=None,
    irregular=None,
    face_lons=False,
    face_lats=False,
    face_connectivity=False,
    titles=False,
    mytest=False,
    transform_first=None,
    blockfill_fast=None,
    nlevs=False,
    orca=None,
    orca_skip=None,
    grid=False,
):
    """
    | The interface to contouring in cf-plot.
    |
    | The minimum use is con(f)
    | where f is a 2 dimensional array. If a cf field is passed then an
    | appropriate plot will be produced i.e. a longitude-latitude or
    | latitude-height plot for example. If a 2d numeric array is passed then
    | the optional arrays x and y can be used to describe the x and y data
    | point locations.
    |
    | f - array to contour
    | x - x locations of data in f (optional)
    | y - y locations of data in f (optional)
    | fill=True - colour fill contours
    | lines=True - draw contour lines and labels
    | line_labels=True - label contour lines
    | title=title - title for the plot
    | ptype=0 - plot type - not needed for cf fields.
    |                       0 = no specific plot type,
    |                       1 = longitude-latitude,
    |                       2 = latitude - height,
    |                       3 = longitude - height,
    |                       4 = latitude - time,
    |                       5 = longitude - time
    |                       6 = rotated pole
    | negative_linestyle='solid' - set to one of 'solid', 'dashed'
    | zero_thick=False - add a thick zero contour line. Set to 3 for example.
    | blockfill=False - set to True for a blockfill plot
    | colorbar_title=colbar_title - title for the colour bar
    | colorbar=True - add a colour bar if a filled contour plot
    | colorbar_label_skip=None - skip colour bar labels. Set to 2 to skip
    |                            every other label.
    | colorbar_orientation=None - options are 'horizontal' and 'vertical'
    |                      The default for most plots is horizontal but
    |                      for polar stereographic plots this is vertical.
    | colorbar_shrink=None - value to shrink the colorbar by.  If the colorbar
    |                        exceeds the plot area then values of 1.0, 0.55
    |                        or 0.5m may help it better fit the plot area.
    | colorbar_position=None - position of colorbar
    |                          [xmin, ymin, x_extent,y_extent] in normalised
    |                          coordinates. Use when a common colorbar
    |                          is required for a set of plots. A typical set
    |                          of values would be [0.1, 0.05, 0.8, 0.02]
    | colorbar_fontsize=None - text size for colorbar labels and title
    | colorbar_fontweight=None - font weight for colorbar labels and title
    | colorbar_text_up_down=False - if True horizontal colour bar labels
    |                               alternate above (start) and below the
    |                               colour bar
    | colorbar_text_down_up=False - if True horizontal colour bar labels
    |                               alternate below (start) and above the
    |                               colour bar
    | colorbar_drawedges=True - draw internal divisions in the colorbar
    | colorbar_fraction=None - space for the colorbar - default = 0.21,
    |                          in normalised
    |                       coordinates
    | colorbar_thick=None - thickness of the colorbar - default = 0.015,
    |                       in normalised coordinates
    | colorbar_anchor=None - default=0.5 - anchor point of colorbar within
    |                        the fraction space.
    |                        0.0 = close to plot, 1.0 = further away
    | colorbar_labels=None - labels to use for colorbar. The default is to
    |                        use the contour levels as labels
    | colorbar_text_up_down=False - on a horizontal colorbar alternate the
    |                               labels top and bottom starting in the
    |                               up position
    | colorbar_text_down_up=False - on a horizontal colorbar alternate the
    |                               labels bottom and top starting in the
    |                               bottom position
    | colorbar_drawedges=True - draw internal delimeter lines in the colorbar
    | colors='k' - contour line colors - takes one or many values.
    | xlog=False - logarithmic x axis
    | ylog=False - logarithmic y axis
    | axes=True - plot x and y axes
    | xaxis=True - plot xaxis
    | yaxis=True - plot y axis
    | xticks=None - xtick positions
    | xticklabels=None - xtick labels
    | yticks=None - y tick positions
    | yticklabels=None - ytick labels
    | xlabel=None - label for x axis
    | ylabel=None - label for y axis
    | swap_axes=False - swap plotted axes - only valid for X, Y, Z vs T plots
    | verbose=None - change to 1 to get a verbose listing of what con
    |                is doing
    | linewidths=None - contour linewidths.  Either a single number for all
    |                   lines or array of widths
    | linestyles=None - takes 'solid', 'dashed', 'dashdot' or 'dotted'
    | alpha=1.0 - transparency setting.  The default is no transparency.
    | zorder=1 - order of drawing
    | level_spacing=None - Default of 'linear' level spacing.  Also takes
    |                      'log', 'loglike', 'outlier' and 'inspect'
    | irregular=None - flag for contouring irregular data
    | face_lons=None - longitude points for face vertices
    | face_lats=None - latitude points for face verticies
    | face_connectivity=None - connectivity for face verticies
    | titles=False - set to True to have a dimensions title
    | transform_first=None - Cartopy should transform the points before
    |                        calling the contouring algorithm, which can have
    |                        a significant impact on speed (it is much
    |                        faster to transform points than it is to
    |                        transform patches) If this is unset and the
    |                        number of points in the x direction is > 400
    |                        then it is set to True.
    | blockfill_fast=None - Use pcolormesh blockfill. This is possibly less
    |                       reliable that the usual code but is
    |                       faster for higher resolution datasets
    | nlevs=False - Let Matplotlib work out the levels for the contour plot
    | orca=None - User specifies this is an orca tripolar grid. Internally
    |             cf-plot tries to detect this by looking for a single
    |             discontinuity in the logitude 2D array. If found a fix
    |             it make to the longitudes so that they are no longer
    |             discontinuous.
    | orca_skip=None - Only plot every nth grid point in the 2D longitude
    |                  and latitude arrays.  This is useful for when
    |                  plotting his resolution data over the whole globe
    |                  which would otherwise be very slow to visualize.
    | grid=False - Draw a grid on the map using the parameters set by
    |              cfp.setvars. Defaults are grid_x_spacing=60,
    |              grid_y_spacing=30, grid_colour='k',
    |              grid_linestyle = '--', grid_thickness=1.0
    |
    :Returns:
     None

    """

    # Turn off divide warning in contour routine which is a numpy issue
    old_settings = np.seterr(all="ignore")
    np.seterr(divide="ignore")

    # Set potential user axis labels
    user_xlabel = xlabel
    user_ylabel = ylabel

    # Set blockfill to True if blockfill_fast is not None
    if blockfill_fast is not None:
        blockfill = True

    # Check if the field is a CF ugrid field with cell faces
    blockfill_ugrid = False
    if isinstance(f, cf.Field) and blockfill:
        if f.domain_topologies():
            if f.domain_topology("cell:face", default=None) is not None:
                face_lons_array = f.aux("X").bounds.array
                face_lats_array = f.aux("Y").bounds.array
                face_connectivity_array = f.domain_topology("cell:face").array
                blockfill_ugrid = True
                fill = False
                lines = False
                irregular = True
            else:
                errstr = (
                    "\n\nError - field does not contain the UGRID face "
                    "information to plot a blockfill plot\n\n\n"
                )
                raise TypeError(errstr)

    # Set blockfill_2d if blockfill and x and y are 2D
    blockfill_2d = False
    if blockfill and not isinstance(f, cf.Field):
        if np.ndim(x) == 2 and np.ndim(y) == 2:
            blockfill_2d = True

    # Call gpos(1) if not already called
    if plotvars.rows > 1 or plotvars.columns > 1:
        if plotvars.gpos_called is False:
            gpos(1)

    # Extract required data for contouring
    # If a cf-python field
    if isinstance(f, cf.Field):

        ndims = np.squeeze(f.data).ndim
        if ndims > 2:
            errstr = (
                "\n\ncfp.con error need a 1 or 2 dimensional field to "
                "contour\n"
                f"received {np.squeeze(f.data).ndim} dimensions\n\n"
                f"{f}"
            )
            raise TypeError(errstr)

        # Extract data
        if verbose:
            print("con - calling _cf_data_assign")

        # Subset the data if a user map is set
        # This is used to speed up the plotting
        # myfield is used for calculating the contour levels
        # myfield_extended is used to make the contour plot
        if plotvars.user_mapset and not blockfill_ugrid:
            if plotvars.proj == "npstere":
                f = f.subspace(Y=cf.wi(plotvars.boundinglat, 90.0))
            elif plotvars.proj == "spstere":
                f = f.subspace(Y=cf.wi(-90.0, plotvars.boundinglat))

        # Extract the data
        field, x, y, ptype, colorbar_title, xlabel, ylabel, xpole, ypole = (
            _cf_data_assign(f, colorbar_title, verbose=verbose)
        )

        if user_xlabel is not None:
            xlabel = user_xlabel
        if user_ylabel is not None:
            ylabel = user_ylabel
    elif isinstance(f, cf.FieldList):
        raise TypeError("\n\ncfp.con - cannot contour a field list\n\n")
    else:
        if verbose:
            print("con - using user assigned data")
        field = f  # field data passed in as f
        if x is None:
            x = np.arange(np.shape(field)[1])
        if y is None:
            y = np.arange(np.shape(field)[0])

        _check_data(field, x, y)
        xlabel = ""
        ylabel = ""

    # Assign irregular and orca keywords unless already set
    if irregular is None:
        if np.size(x) == np.size(np.unique(x)):
            irregular = False
        else:
            irregular = True
            if np.ndim(x) == 2 and np.ndim(y) == 2:
                if orca is None:
                    orca = orca_check(x)
                if orca:

                    # Apply orca_skip if set
                    if orca_skip is not None:
                        print("applying orca_skip value of ", orca_skip)
                        x = x[::orca_skip, ::orca_skip]
                        y = y[::orca_skip, ::orca_skip]
                        field = field[::orca_skip, ::orca_skip]

                    # orca grids have a discontinuity in the longitude grid
                    # use the method at
                    # https://gist.github.com/pelson/79cf31ef324774c97ae7
                    # to remove the discontinuity

                    fixed_x = x.copy()
                    for i, start in enumerate(
                        np.argmax(np.abs(np.diff(x)) > 180, axis=1)
                    ):
                        fixed_x[i, start + 1 :] += 360
                    x = fixed_x

    if np.ndim(x) == 2:
        irregular = False

    # Set contour line styles
    matplotlib.rcParams["contour.negative_linestyle"] = negative_linestyle

    # Set contour lines off on block plots
    if blockfill:
        fill = False
        field_orig = deepcopy(field)
        x_orig = deepcopy(x)
        y_orig = deepcopy(y)

        # Check number of colours and levels match if user has modified the
        # number of colours
        if plotvars.cscale_flag == 2:
            ncols = np.size(plotvars.cs)
            nintervals = np.size(plotvars.levels) - 1
            if plotvars.levels_extend == "min":
                nintervals += 1
            if plotvars.levels_extend == "max":
                nintervals += 1
            if plotvars.levels_extend == "both":
                nintervals += 2
            if ncols != nintervals:
                errstr = (
                    "\n\ncfp.con - blockfill error \n"
                    "need to match number of colours and contour intervals\n"
                    "Don't forget to take account of the colorbar "
                    "extensions\n\n"
                )
                raise TypeError(errstr)

    # Turn off colorbar if fill is turned off
    if not fill and not blockfill and not blockfill_ugrid:
        colorbar = False

    # Revert to default colour scale if cscale_flag flag is set
    if plotvars.cscale_flag == 0:
        plotvars.cs = cscale1

    # Set the orientation of the colorbar
    if plotvars.plot_type == 1:
        if plotvars.proj == "npstere" or plotvars.proj == "spstere":
            if colorbar_orientation is None:
                colorbar_orientation = "vertical"
    if colorbar_orientation is None:
        colorbar_orientation = "horizontal"

    # Store original map resolution
    resolution_orig = plotvars.resolution

    # Set size of color bar if not specified
    if colorbar_shrink is None:
        colorbar_shrink = 1.0
        if plotvars.proj == "npstere" or plotvars.proj == "spstere":
            colorbar_shrink = 0.8

    # Set plot type if user specified
    if ptype is not None:
        plotvars.plot_type = ptype

    # Get contour levels if none are defined
    spacing = "linear"
    if plotvars.level_spacing is not None:
        spacing = plotvars.level_spacing
    if level_spacing is not None:
        spacing = level_spacing

    if plotvars.levels is None:

        if isinstance(f, cf.Field):
            (
                field,
                x,
                y,
                ptype,
                colorbar_title,
                xlabel,
                ylabel,
                xpole,
                ypole,
            ) = _cf_data_assign(f, colorbar_title, verbose=verbose)
        clevs, mult, fmult = calculate_levels(
            field=field, level_spacing=spacing, verbose=verbose
        )

    else:
        clevs = plotvars.levels
        mult = 0
        fmult = 1

    # Set the colour scale if nothing is defined
    includes_zero = False
    if plotvars.cscale_flag == 0:
        col_zero = 0
        for cval in clevs:
            if includes_zero is False:
                col_zero = col_zero + 1
            if cval == 0:
                includes_zero = True

        if includes_zero:
            cs_below = col_zero
            cs_above = np.size(clevs) - col_zero + 1
            if (
                plotvars.levels_extend == "max"
                or plotvars.levels_extend == "neither"
            ):
                cs_below = cs_below - 1
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "neither"
            ):
                cs_above = cs_above - 1
            uniform = True
            if plotvars.cs_uniform is False:
                uniform = False
            cscale("scale1", below=cs_below, above=cs_above, uniform=uniform)
        else:
            ncols = np.size(clevs) + 1
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "max"
            ):
                ncols = ncols - 1
            if plotvars.levels_extend == "neither":
                ncols = ncols - 2
            cscale("viridis", ncols=ncols)

        plotvars.cscale_flag = 0

    # User selected colour map but no mods so fit to levels
    if plotvars.cscale_flag == 1:
        ncols = np.size(clevs) + 1
        if plotvars.levels_extend == "min" or plotvars.levels_extend == "max":
            ncols = ncols - 1
        if plotvars.levels_extend == "neither":
            ncols = ncols - 2
        cscale(plotvars.cs_user, ncols=ncols)
        plotvars.cscale_flag = 1

    # Set colorbar labels
    # Set a sensible label spacing if the user hasn't already done so
    if colorbar_label_skip is None:
        if colorbar_orientation == "horizontal":
            nchars = 0
            for lev in clevs:
                nchars = nchars + len(str(lev))
            colorbar_label_skip = int(nchars / 80 + 1)
            if plotvars.columns > 1:
                colorbar_label_skip = int(nchars * (plotvars.columns) / 80)
        else:
            colorbar_label_skip = 1

    if colorbar_label_skip > 1:
        if includes_zero:
            # include zero in the colorbar labels
            zero_pos = [i for i, mylev in enumerate(clevs) if mylev == 0][0]
            cbar_labels = clevs[zero_pos]
            i = zero_pos + colorbar_label_skip
            while i <= len(clevs) - 1:
                cbar_labels = np.append(cbar_labels, clevs[i])
                i = i + colorbar_label_skip
            i = zero_pos - colorbar_label_skip
            if i >= 0:
                while i >= 0:
                    cbar_labels = np.append(clevs[i], cbar_labels)
                    i = i - colorbar_label_skip
        else:
            cbar_labels = clevs[0]
            i = int(colorbar_label_skip)
            while i <= len(clevs) - 1:
                cbar_labels = np.append(cbar_labels, clevs[i])
                i = i + colorbar_label_skip

    else:
        cbar_labels = clevs

    if colorbar_label_skip is None:
        colorbar_label_skip = 1

    # Make a list of strings of the colorbar levels for later labelling
    clabels = []
    for i in cbar_labels:
        clabels.append(str(i))
        if colorbar_label_skip > 1:
            for skip in np.arange(colorbar_label_skip - 1):
                clabels.append("")

    if colorbar_labels is not None:
        cbar_labels = colorbar_labels
    else:
        cbar_labels = clabels

    # Turn off line_labels if the field is all the same
    # Matplotlib 3.2.2 throws an error if there are no line labels
    if np.nanmin(field) == np.nanmax(field):
        line_labels = False

    # Add mult to colorbar_title if used
    if colorbar_title is None:
        colorbar_title = ""
    if mult != 0:
        colorbar_title = colorbar_title + " *10$^{" + str(mult) + "}$"

    # Catch null title
    if title is None:
        title = ""
    if plotvars.title is not None:
        title = plotvars.title

    # Set plot variables
    title_fontsize = plotvars.title_fontsize
    text_fontsize = plotvars.text_fontsize
    if colorbar_fontsize is None:
        colorbar_fontsize = plotvars.colorbar_fontsize
    if colorbar_fontweight is None:
        colorbar_fontweight = plotvars.colorbar_fontweight
    continent_thickness = plotvars.continent_thickness
    continent_color = plotvars.continent_color
    continent_linestyle = plotvars.continent_linestyle
    land_color = plotvars.land_color
    ocean_color = plotvars.ocean_color
    lake_color = plotvars.lake_color
    title_fontweight = plotvars.title_fontweight
    if continent_thickness is None:
        continent_thickness = 1.5
    if continent_color is None:
        continent_color = "k"
    if continent_linestyle is None:
        continent_linestyle = "solid"
    cb_orient = colorbar_orientation

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

    # Calculate a set of dimension titles if requested
    if titles:
        plotvars.titles_con_called = True
        title_dims = generate_titles(f)
        if not colorbar:
            title_dims = colorbar_title + "\n" + title_dims

    # Check if data is well formed
    # i.e. dimensions have only recognizable X, Y, Z, T or a subset
    well_formed = False
    if isinstance(f, cf.Field):
        well_formed = check_well_formed(f)
        # TODO SLB - probably there should be an error raised here if not
        # 'well-formed'?

    if nlevs is not False:
        clevs = nlevs
        plotvars.levels_extend = "neither"
        if plotvars.cscale_flag == 0:
            if np.min(field) < 0 and np.max(field) > 0:
                cscale("scale1", ncols=nlevs)
            else:
                cscale("viridis", ncols=nlevs)
            plotvars.cscale_flag = 0
        else:
            cscale(plotvars.cs_user, ncols=nlevs)

    ##################
    # Map contour plot
    ##################
    if ptype == 1:
        if verbose:
            print("con - making a map plot")

        # Open a new plot if necessary
        if plotvars.user_plot == 0:
            gopen(user_plot=0)

        # Reset the stored mapping
        if plotvars.user_mapset == 0:
            plotvars.lonmin = -180
            plotvars.lonmax = 180
            plotvars.latmin = -90
            plotvars.latmax = 90

        # Set up mapping
        mylonmin = np.nanmin(x)
        mylonmax = np.nanmax(x)
        mylatmin = np.nanmin(y)
        mylatmax = np.nanmax(y)
        lonrange = mylonmax - mylonmin
        latrange = mylatmax - mylatmin

        if lonrange > 360.0:
            mylonmax = mylonmin + 360.0
            lonrange = 360.0

        if (lonrange > 350 and latrange > 160) or plotvars.user_mapset == 1:
            _set_map()
        else:
            mapset(
                lonmin=mylonmin,
                lonmax=mylonmax,
                latmin=mylatmin,
                latmax=mylatmax,
                user_mapset=0,
                resolution=resolution_orig,
            )

            _set_map()

        mymap = plotvars.mymap
        user_mapset = plotvars.user_mapset

        lonrange = np.nanmax(x) - np.nanmin(x)

        if not blockfill_ugrid and not blockfill_2d:
            if not irregular:
                if lonrange > 350 and np.ndim(y) == 1:
                    # Add cyclic information if missing.
                    if lonrange < 360:
                        # field, x = cartopy_util.add_cyclic_point(field, x)
                        # Call add_cyclic_point it spacing is regular
                        x_regular = True
                        xspacing = x[1] - x[0]
                        for ix in np.arange(len(x) - 1):
                            if x[ix + 1] - x[ix] != xspacing:
                                x_regular = False
                        if x_regular:
                            field, x = add_cyclic(field, x)

                        lonrange = np.nanmax(x) - np.nanmin(x)

                        # cartopy line drawing fix
                        if x[-1] - x[0] == 360.0:
                            x[-1] = x[-1] + 0.001

                    # Shift grid if needed
                    if plotvars.lonmin < np.nanmin(x):
                        # Cartopy feature at version 0.20.0
                        # -360 to 0 creates strange contours
                        vers = cartopy.__version__.split(".")
                        val = int(vers[0] + vers[1])
                        if val < 20:
                            x = x - 360
                    if plotvars.lonmin > np.nanmax(x):
                        x = x + 360
            elif not orca:
                # Get the irregular data within the map coordinates
                # Matplotlib tricontour cannot plot missing data so we need to
                # split the missing data into a separate field to deal with
                # this

                field_modified = deepcopy(field)
                pts_nan = np.where(np.isnan(field_modified))
                field_modified[pts_nan] = -1e30

                field_irregular, lons_irregular, lats_irregular = (
                    irregular_window(field_modified, x, y)
                )
                # pts_real  = np.where(np.isfinite(field_irregular))
                pts_real = np.where(field_irregular > -1e29)
                pts_nan = np.where(field_irregular < -1e29)

                field_irregular_nan = []
                lons_irregular_nan = []
                lats_irregular_nan = []
                if np.size(pts_nan) > 0:
                    field_irregular_nan = deepcopy(field_irregular)
                    lons_irregular_nan = deepcopy(lons_irregular)
                    lats_irregular_nan = deepcopy(lats_irregular)
                    field_irregular_nan[:] = 0
                    field_irregular_nan[pts_nan] = 1

                field_irregular_real = deepcopy(field_irregular[pts_real])
                lons_irregular_real = deepcopy(lons_irregular[pts_real])
                lats_irregular_real = deepcopy(lats_irregular[pts_real])

        if not irregular:
            # Flip latitudes and field if latitudes are in descending order
            if np.ndim(y) == 1:
                if y[0] > y[-1]:
                    y = y[::-1]
                    field = np.flipud(field)

        # Plotting a sub-area of the grid produces stray contour labels
        # in polar plots. Subsample the latitudes to remove this problem

        if plotvars.proj == "npstere" and np.ndim(y) == 1:
            if not blockfill_ugrid and not blockfill_2d:
                if irregular:
                    pts = np.where(lats_irregular > plotvars.boundinglat - 5)
                    pts = np.array(pts).flatten()
                    lons_irregular_real = lons_irregular_real[pts]
                    lats_irregular_real = lats_irregular_real[pts]
                    field_irregular_real = field_irregular_real[pts]
                else:
                    myypos = find_pos_in_array(
                        vals=y, val=plotvars.boundinglat
                    )
                    if myypos != -1:
                        y = y[myypos:]
                        field = field[myypos:, :]

        if plotvars.proj == "spstere" and np.ndim(y) == 1:
            if not blockfill_ugrid and not blockfill_2d:
                if irregular:
                    pts = np.where(
                        lats_irregular_real < plotvars.boundinglat + 5
                    )
                    lons_irregular_real = lons_irregular_real[pts]
                    lats_irregular_real = lats_irregular_real[pts]
                    field_irregular_real = field_irregular_real[pts]
                else:
                    myypos = find_pos_in_array(
                        vals=y, val=plotvars.boundinglat, above=True
                    )
                    if myypos != -1:
                        y = y[0 : myypos + 1]
                        field = field[0 : myypos + 1, :]

        # Set the longitudes and latitudes
        lons, lats = x, y

        # Set the plot limits
        if lonrange > 350:
            gset(
                xmin=plotvars.lonmin,
                xmax=plotvars.lonmax,
                ymin=plotvars.latmin,
                ymax=plotvars.latmax,
                user_gset=0,
            )
        else:
            if user_mapset == 1:
                gset(
                    xmin=plotvars.lonmin,
                    xmax=plotvars.lonmax,
                    ymin=plotvars.latmin,
                    ymax=plotvars.latmax,
                    user_gset=0,
                )
            else:
                gset(
                    xmin=np.nanmin(lons),
                    xmax=np.nanmax(lons),
                    ymin=np.nanmin(lats),
                    ymax=np.nanmax(lats),
                    user_gset=0,
                )

        # Filled contours
        if fill:
            if verbose:
                print("con - adding filled contours")
            # Get colour scale for use in contouring
            # If colour bar extensions are enabled then the colour map goes
            # from 1 to ncols-2.  The colours for the colour bar extensions
            # are then changed on the colorbar and plot after the plot is made
            colmap = _cscale_get_map()

            cmap = matplotlib.colors.ListedColormap(colmap)
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_under(plotvars.cs[0])
            if (
                plotvars.levels_extend == "max"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_over(plotvars.cs[-1])

            # For fast map contours add transform_first=True to contourf
            # command and make lons and lats 2D
            if (
                transform_first is None
                and np.ndim(lons) == 1
                and np.ndim(lats) == 1
            ):
                if np.size(lons) >= 400:
                    transform_first = True

            # Fast map contours are also needed when clevs is a integer
            if (
                isinstance(clevs, int)
                and plotvars.plot_type == 1
                and plotvars.proj == "cyl"
            ):
                transform_first = True

            if transform_first:
                if np.ndim(lons) == 1 and np.ndim(lats) == 1:
                    lons, lats = np.meshgrid(lons, lats)

            # Filled colour contours
            if not irregular or orca is True:
                plotvars.image = mymap.contourf(
                    lons,
                    lats,
                    field * fmult,
                    clevs,
                    extend=plotvars.levels_extend,
                    cmap=cmap,
                    norm=plotvars.norm,
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder,
                    transform_first=transform_first,
                )

            else:
                if np.size(field_irregular_real) > 0:
                    plotvars.image = mymap.tricontourf(
                        lons_irregular_real,
                        lats_irregular_real,
                        field_irregular_real * fmult,
                        clevs,
                        extend=plotvars.levels_extend,
                        cmap=cmap,
                        norm=plotvars.norm,
                        alpha=alpha,
                        transform=ccrs.PlateCarree(),
                        zorder=zorder,
                    )

        # Block fill
        if blockfill and not blockfill_ugrid:
            if verbose:
                print("con - adding blockfill")

            two_d = False
            if np.ndim(x) == 2 and np.ndim(y) == 2:
                two_d = True

            if isinstance(f, cf.Field):

                if f.ref(
                    "grid_mapping_name:transverse_mercator", default=False
                ):
                    # Special case for transverse mercator
                    _bfill(
                        f=f.squeeze() * fmult,
                        x=x,
                        y=y,
                        clevs=clevs,
                        lonlat=False,
                        alpha=alpha,
                        fast=blockfill_fast,
                        zorder=zorder,
                    )

                # elif orca:
                elif two_d:
                    _bfill(
                        f=field * fmult,
                        x=x,
                        y=y,
                        clevs=clevs,
                        lonlat=False,
                        alpha=alpha,
                        fast=blockfill_fast,
                        zorder=zorder,
                    )

                else:

                    if f.coord("X").has_bounds() and f.coord("Y").has_bounds():
                        xpts = np.squeeze(f.coord("X").bounds.array[:, 0])
                        ypts = np.squeeze(f.coord("Y").bounds.array[:, 0])
                        # Add last longitude point
                        xpts = np.append(
                            xpts, f.coord("X").bounds.array[-1, 1]
                        )
                        # Add last latitude point
                        ypts = np.append(
                            ypts, f.coord("Y").bounds.array[-1, 1]
                        )

                        _bfill(
                            f=field_orig * fmult,
                            x=xpts,
                            y=ypts,
                            clevs=clevs,
                            lonlat=True,
                            bound=1,
                            alpha=alpha,
                            fast=blockfill_fast,
                            zorder=zorder,
                        )
                    else:
                        _bfill(
                            f=field_orig * fmult,
                            x=x_orig,
                            y=y_orig,
                            clevs=clevs,
                            lonlat=True,
                            bound=0,
                            alpha=alpha,
                            fast=blockfill_fast,
                            zorder=zorder,
                        )

            else:
                _bfill(
                    f=field_orig * fmult,
                    x=x_orig,
                    y=y_orig,
                    clevs=clevs,
                    lonlat=True,
                    bound=0,
                    alpha=alpha,
                    fast=blockfill_fast,
                    zorder=zorder,
                )

        # Block fill for irregular
        if blockfill_ugrid and not blockfill_2d:
            if verbose:
                print("con - adding blockfill for irregular")
            _bfill_ugrid(
                f=field_orig * fmult,
                face_lons=face_lons_array,
                face_lats=face_lats_array,
                face_connectivity=face_connectivity_array,
                clevs=clevs,
                alpha=alpha,
                zorder=zorder,
            )

        # Contour lines and labels
        if lines:
            if verbose:
                print("con - adding contour lines and labels")

            if not irregular or blockfill_2d or orca:
                cs = mymap.contour(
                    lons,
                    lats,
                    field * fmult,
                    clevs,
                    colors=colors,
                    linewidths=linewidths,
                    linestyles=linestyles,
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder,
                )
            else:
                cs = mymap.tricontour(
                    lons_irregular_real,
                    lats_irregular_real,
                    field_irregular_real * fmult,
                    clevs,
                    colors=colors,
                    linewidths=linewidths,
                    linestyles=linestyles,
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder,
                )

            if line_labels and not isinstance(clevs, int):
                nd = ndecs(clevs)
                fmt = "%d"
                if nd != 0:
                    fmt = "%1." + str(nd) + "f"
                plotvars.plot.clabel(
                    cs,
                    levels=clevs,
                    fmt=fmt,
                    zorder=zorder,
                    colors=colors,
                    fontsize=text_fontsize,
                )

            # Thick zero contour line
            if zero_thick:
                cs = mymap.contour(
                    lons,
                    lats,
                    field * fmult,
                    [-1e-32, 0],
                    colors=colors,
                    linewidths=zero_thick,
                    linestyles=linestyles,
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder,
                )

        # Add a irregular mask if there is one
        if irregular and not blockfill_ugrid and not orca and not blockfill_2d:
            if np.size(field_irregular_nan) > 0:
                cmap_white = matplotlib.colors.ListedColormap([1.0, 1.0, 1.0])
                mymap.tricontourf(
                    lons_irregular_nan,
                    lats_irregular_nan,
                    field_irregular_nan,
                    [0.5, 1.5],
                    extend="neither",
                    cmap=cmap_white,
                    norm=plotvars.norm,
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder,
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

        # Coastlines and features
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
            zorder=zorder,
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

        if grid:
            map_grid()

        # Title
        if title != "":
            _map_title(title)

        # Titles for dimensions
        if titles:
            _dim_titles(title=title_dims)

        # Color bar
        if colorbar:
            cbar(
                labels=cbar_labels,
                orientation=cb_orient,
                position=colorbar_position,
                shrink=colorbar_shrink,
                title=colorbar_title,
                fontsize=colorbar_fontsize,
                fontweight=colorbar_fontweight,
                text_up_down=colorbar_text_up_down,
                text_down_up=colorbar_text_down_up,
                drawedges=colorbar_drawedges,
                fraction=colorbar_fraction,
                thick=colorbar_thick,
                anchor=colorbar_anchor,
                levs=clevs,
                verbose=verbose,
            )

        # Reset plot limits if not a user plot
        if plotvars.user_gset == 0:
            gset()

    ################################################
    # Latitude, longitude or time vs Z contour plots
    ################################################
    if ptype == 2 or ptype == 3 or ptype == 7:

        if verbose:
            if ptype == 2:
                print("con - making a latitude-pressure plot")
            if ptype == 3:
                print("con - making a longitude-pressure plot")
            if ptype == 7:
                print("con - making a time-pressure plot")

        # Work out which way is up
        positive = None
        myz = find_z(f)

        if isinstance(f, cf.Field) and well_formed:
            if hasattr(f.construct(myz), "positive"):
                positive = f.construct(myz).positive
            else:
                errstr = (
                    "\ncf-plot - data error \n"
                    "data needs a vertical coordinate direction"
                    " as required in CF data conventions"
                    "\nMaking a contour plot assuming positive is down\n\n"
                    "If this is incorrect the data needs to be modified to \n"
                    "include a correct value for the direction attribute\n"
                    "such as in f.coord('Z').positive='down'\n\n"
                )
                print(errstr)
                positive = "down"
        else:
            positive = "down"
            if "theta" in ylabel.split(" "):
                positive = "up"
            if "height" in ylabel.split(" "):
                positive = "up"

        if plotvars.user_plot == 0:
            gopen(user_plot=0)

        # Use gset parameter of ylog if user has set this
        if plotvars.ylog is True or plotvars.ylog == 1:
            ylog = True

        # Set plot limits
        user_gset = plotvars.user_gset
        if user_gset == 0:
            # Program selected data plot limits
            xmin = np.nanmin(x)
            if xmin < -80 and xmin >= -90:
                xmin = -90
            xmax = np.nanmax(x)
            if xmax > 80 and xmax <= 90:
                xmax = 90

            if positive == "down":
                ymin = np.nanmax(y)
                ymax = np.nanmin(y)
                if ymax < 10:
                    ymax = 0
            else:
                ymin = np.nanmin(y)
                ymax = np.nanmax(y)

        else:
            # Use user specified plot limits
            xmin = plotvars.xmin
            xmax = plotvars.xmax
            ymin = plotvars.ymin
            ymax = plotvars.ymax

        ystep = 100
        myrange = abs(ymax - ymin)

        if myrange < 1:
            ystep = abs(ymax - ymin) / 10.0
        if abs(ymax - ymin) > 1:
            ystep = 1
        if abs(ymax - ymin) > 10:
            ystep = 10
        if abs(ymax - ymin) > 100:
            ystep = 100
        if abs(ymax - ymin) > 1000:
            ystep = 200
        if abs(ymax - ymin) > 2000:
            ystep = 500
        if abs(ymax - ymin) > 5000:
            ystep = 1000
        if abs(ymax - ymin) > 15000:
            ystep = 5000

        # Work out ticks and tick labels
        if ylog is False or ylog == 0:
            heightticks = _gvals(
                dmin=min(ymin, ymax),
                dmax=max(ymin, ymax),
                mystep=ystep,
                mod=False,
            )[0]

            if myrange < 1 and myrange > 0.1:
                heightticks = np.arange(10) / 10.0

        else:
            heightticks = []
            for tick in 1000, 100, 10, 1:
                if tick >= min(ymin, ymax) and tick <= max(ymin, ymax):
                    heightticks.append(tick)
        heightlabels = heightticks

        if axes:
            if xaxis:
                if xticks is not None:
                    if xticklabels is None:
                        xticklabels = xticks
            else:
                xticks = [100000000]
                xticklabels = xticks
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
            xticks = [100000000]
            xticklabels = xticks
            heightticks = [100000000]
            heightlabels = heightticks
            xlabel = ""
            ylabel = ""

        if yticks is None:
            yticks = heightticks
            yticklabels = heightlabels

        # Time - height contour plot
        if ptype == 7:
            if isinstance(f, cf.Field):
                if plotvars.user_gset == 0:
                    tmin = f.construct("T").dtarray[0]
                    tmax = f.construct("T").dtarray[-1]
                else:
                    # Use user set values if present
                    tmin = plotvars.xmin
                    tmax = plotvars.xmax

                    ref_time = f.construct("T").units
                    ref_calendar = f.construct("T").calendar
                    time_units = cf.Units(ref_time, ref_calendar)
                    t = cf.Data(cf.dt(tmin), units=time_units)
                    xmin = t.array
                    t = cf.Data(cf.dt(tmax), units=time_units)
                    xmax = t.array

        if xticks is None and xaxis:
            if ptype == 2:
                xticks, xticklabels = _mapaxis(
                    min=xmin, max=xmax, type=2
                )  # lat-pressure
            if ptype == 3:
                xticks, xticklabels = _mapaxis(
                    min=xmin, max=xmax, type=1
                )  # lon-pressure

            if ptype == 7:
                # time-pressure
                if isinstance(f, cf.Field):

                    # Change plotvars.xmin and plotvars.xmax from a date string
                    # to a number
                    ref_time = f.construct("T").units
                    ref_calendar = f.construct("T").calendar
                    time_units = cf.Units(ref_time, ref_calendar)

                    t = cf.Data(cf.dt(tmin), units=time_units)
                    xmin = t.array
                    t = cf.Data(cf.dt(tmax), units=time_units)
                    xmax = t.array

                    taxis = cf.Data(
                        [cf.dt(tmin), cf.dt(tmax)], units=time_units
                    )
                    time_ticks, time_labels, tlabel = _timeaxis(taxis)

                    # Use user supplied labels if present
                    if user_xlabel is None:
                        xlabel = tlabel
                    if xticks is None:
                        xticks = time_ticks
                    if xticklabels is None:
                        xticklabels = time_labels

                else:
                    errstr = (
                        "\nNot a CF field\nPlease use ptype=0 and "
                        "specify axis labels manually\n"
                    )
                    raise Warning(errstr)

        # Set plot limits
        if ylog is False or ylog == 0:
            gset(
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, user_gset=user_gset
            )
        else:
            if ymax == 0:
                ymax = 1  # Avoid zero in a log plot
            gset(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                ylog=True,
                user_gset=user_gset,
            )

        # Label axes
        axes_plot(
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=heightticks,
            yticklabels=heightlabels,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        # Get colour scale for use in contouring
        # If colour bar extensions are enabled then the colour map goes
        # from 1 to ncols-2.  The colours for the colour bar extensions are
        # then changed on the colorbar and plot after the plot is made
        colmap = _cscale_get_map()

        # Filled contours
        if fill:
            colmap = _cscale_get_map()
            cmap = matplotlib.colors.ListedColormap(colmap)
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_under(plotvars.cs[0])
            if (
                plotvars.levels_extend == "max"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_over(plotvars.cs[-1])

            plotvars.image = plotvars.plot.contourf(
                x,
                y,
                field * fmult,
                clevs,
                extend=plotvars.levels_extend,
                cmap=cmap,
                norm=plotvars.norm,
                alpha=alpha,
                zorder=zorder,
            )

        # Block fill
        if blockfill:
            if isinstance(f, cf.Field):

                hasbounds = True

                if ptype == 2:
                    if f.coord("Y").has_bounds() and f.coord("Z").has_bounds():
                        xpts = np.squeeze(f.coord("Y").bounds.array)[:, 0]
                        xpts = np.append(
                            xpts, f.coord("Y").bounds.array[-1, 1]
                        )
                        ypts = np.squeeze(f.coord("Z").bounds.array)[:, 0]
                        ypts = np.append(
                            ypts, f.coord("Z").bounds.array[-1, 1]
                        )
                    else:
                        hasbounds = False

                if ptype == 3:
                    if f.coord("X").has_bounds() and f.coord("Z").has_bounds():
                        xpts = np.squeeze(f.coord("X").bounds.array)[:, 0]
                        xpts = np.append(
                            xpts, f.coord("X").bounds.array[-1, 1]
                        )
                        # Use 'noqa' to prevent PEP8 E501 being raised due to
                        # line length being too long. Can't prevent this
                        # straightforwardly given function name of that length.
                        ypts = np.squeeAllTrop_UpStrat_Eq_Total_AllWN_Timeseries_2ze(  # noqa: E501
                            f.coord("Z").bounds.array
                        )[
                            :, 0
                        ]
                        ypts = np.append(
                            xpts, f.coord("Z").bounds.array[-1, 1]
                        )
                    else:
                        hasbounds = False

                if ptype == 7:
                    if f.coord("T").has_bounds() and f.coord("Z").has_bounds():
                        xpts = np.squeeze(f.coord("T").bounds.array)[:, 0]
                        xpts = np.append(
                            xpts, f.coord("T").bounds.array[-1, 1]
                        )
                        ypts = np.squeeze(f.coord("Z").bounds.array)[:, 0]
                        ypts = np.append(
                            xpts, f.coord("Z").bounds.array[-1, 1]
                        )
                    else:
                        hasbounds = False

                if hasbounds:
                    _bfill(
                        f=field_orig * fmult,
                        x=xpts,
                        y=ypts,
                        clevs=clevs,
                        lonlat=False,
                        bound=1,
                        alpha=alpha,
                        fast=blockfill_fast,
                        zorder=zorder,
                    )
                else:
                    _bfill(
                        f=field_orig * fmult,
                        x=x_orig,
                        y=y_orig,
                        clevs=clevs,
                        lonlat=False,
                        bound=0,
                        alpha=alpha,
                        fast=blockfill_fast,
                        zorder=zorder,
                    )

            else:
                _bfill(
                    f=field_orig * fmult,
                    x=x_orig,
                    y=y_orig,
                    clevs=clevs,
                    lonlat=False,
                    bound=0,
                    alpha=alpha,
                    fast=blockfill_fast,
                    zorder=zorder,
                )

        # Contour lines and labels
        if lines:
            cs = plotvars.plot.contour(
                x,
                y,
                field * fmult,
                clevs,
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=zorder,
            )
            if line_labels and not isinstance(clevs, int):
                nd = ndecs(clevs)
                fmt = "%d"
                if nd != 0:
                    fmt = "%1." + str(nd) + "f"
                plotvars.plot.clabel(
                    cs,
                    fmt=fmt,
                    colors=colors,
                    zorder=zorder,
                    fontsize=text_fontsize,
                )

                # Thick zero contour line
                if zero_thick:
                    cs = plotvars.plot.contour(
                        x,
                        y,
                        field * fmult,
                        [-1e-32, 0],
                        colors=colors,
                        linewidths=zero_thick,
                        linestyles=linestyles,
                        alpha=alpha,
                        zorder=zorder,
                    )

        # Titles for dimensions
        if titles:
            _dim_titles(title=title_dims)

        # Color bar
        if colorbar:
            cbar(
                labels=cbar_labels,
                orientation=cb_orient,
                position=colorbar_position,
                shrink=colorbar_shrink,
                title=colorbar_title,
                fontsize=colorbar_fontsize,
                fontweight=colorbar_fontweight,
                text_up_down=colorbar_text_up_down,
                text_down_up=colorbar_text_down_up,
                drawedges=colorbar_drawedges,
                fraction=colorbar_fraction,
                thick=colorbar_thick,
                levs=clevs,
                anchor=colorbar_anchor,
                verbose=verbose,
            )

        # Title
        plotvars.plot.set_title(
            title, y=1.03, fontsize=title_fontsize, fontweight=title_fontweight
        )

        # Reset plot limits to those supplied by the user
        if user_gset == 1 and ptype == 7:
            gset(
                xmin=tmin, xmax=tmax, ymin=ymin, ymax=ymax, user_gset=user_gset
            )

        # reset plot limits if not a user plot
        if plotvars.user_gset == 0:
            gset()

    ########################
    # Hovmuller contour plot
    ########################
    if ptype == 4 or ptype == 5:
        if verbose:
            print("con - making a Hovmuller plot")
        yplotlabel = "Time"
        if ptype == 4:
            xplotlabel = "Longitude"
        if ptype == 5:
            xplotlabel = "Latitude"
        user_gset = plotvars.user_gset

        # Time strings set to None initially
        tmin = None
        tmax = None

        # Set plot limits
        if all(
            val is not None
            for val in [
                plotvars.xmin,
                plotvars.xmax,
                plotvars.ymin,
                plotvars.ymax,
            ]
        ):
            # Store time strings for later use
            tmin = plotvars.ymin
            tmax = plotvars.ymax

            # Check data has CF attributes needed
            check_units = check_units = True
            check_calendar = True
            check_Units_reftime = True
            if hasattr(f.construct("T"), "units") is False:
                check_units = False
            if hasattr(f.construct("T"), "calendar") is False:
                check_calendar = False
            if hasattr(f.construct("T"), "Units"):
                if not hasattr(f.construct("T").Units, "reftime"):
                    check_Units_reftime = False
            else:
                check_Units_reftime = False
            if False in [check_units, check_calendar, check_Units_reftime]:
                print(
                    "\nThe required CF time information to make the plot "
                    "is not available please fix the following before "
                    "trying to plot again"
                )
                if check_units is False:
                    print("Time axis missing: units")
                if check_calendar is False:
                    print("Time axis missing: calendar")
                if check_Units_reftime is False:
                    print("Time axis missing: Units.reftime")
                return

            # Change from date string in ymin and ymax to date as a float

            ref_time = f.construct("T").units
            ref_calendar = f.construct("T").calendar

            time_units = cf.Units(ref_time, ref_calendar)
            t = cf.Data(cf.dt(plotvars.ymin), units=time_units)
            ymin = t.array
            t = cf.Data(cf.dt(plotvars.ymax), units=time_units)
            ymax = t.array
            xmin = plotvars.xmin
            xmax = plotvars.xmax
        else:
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            ymin = np.nanmin(y)
            ymax = np.nanmax(y)

        # Extract axis labels
        if len(f.constructs("T")) > 1:
            errstr = (
                "\n\nTime axis error - only one time axis allowed\n "
                "Please list time axes with print(f.constructs())\n"
                "and remove the ones not needed for a hovmuller plot \n"
                "with f.del_construct('unwanted_time_axis')\n"
                "before trying to plot again\n\n\n\n"
            )
            raise TypeError(errstr)

        time_ticks, time_labels, ylabel = _timeaxis(f.construct("T"))

        if ptype == 4:
            lonlatticks, lonlatlabels = _mapaxis(min=xmin, max=xmax, type=1)
        if ptype == 5:
            lonlatticks, lonlatlabels = _mapaxis(min=xmin, max=xmax, type=2)

        if axes:
            if xaxis:
                if xticks is not None:
                    lonlatticks = xticks
                    lonlatlabels = xticks
                    if xticklabels is not None:
                        lonlatlabels = xticklabels
            else:
                lonlatticks = [100000000]
                xlabel = ""

            if yaxis:
                if yticks is not None:
                    timeticks = yticks
                    timelabels = yticks
                    if yticklabels is not None:
                        timelabels = yticklabels
            else:
                timeticks = [100000000]
                ylabel = ""

        else:
            timeticks = [100000000]
            xplotlabel = ""
            yplotlabel = ""

        if user_xlabel is not None:
            xplotlabel = user_xlabel
        if user_ylabel is not None:
            yplotlabel = user_ylabel

        # Use the automatically generated labels if none are supplied
        if ylabel is None:
            yplotlabel = "time"
        if np.size(time_ticks) > 0:
            timeticks = time_ticks
        if np.size(time_labels) > 0:
            timelabels = time_labels

        # Swap axes if requested
        if swap_axes:
            x, y = y, x
            field = np.flipud(np.rot90(field))
            xmin, ymin = ymin, xmin
            xmax, ymax = ymax, xmax
            xplotlabel, yplotlabel = yplotlabel, xplotlabel
            lonlatticks, timeticks = timeticks, lonlatticks
            lonlatlabels, timelabels = timelabels, lonlatlabels

        # Set plot limits
        if plotvars.user_plot == 0:
            gopen(user_plot=0)
        gset(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, user_gset=user_gset)

        # Revert to time strings if set
        if all(val is not None for val in [tmin, tmax]):
            plotvars.ymin = tmin
            plotvars.ymax = tmax

        # Set and label axes
        axes_plot(
            xticks=lonlatticks,
            xticklabels=lonlatlabels,
            yticks=timeticks,
            yticklabels=timelabels,
            xlabel=xplotlabel,
            ylabel=yplotlabel,
        )

        # Get colour scale for use in contouring
        # If colour bar extensions are enabled then the colour map goes
        # from 1 to ncols-2.  The colours for the colour bar extensions are
        # then changed on the colorbar and plot after the plot is made
        colmap = _cscale_get_map()

        # Filled contours
        if fill:
            colmap = _cscale_get_map()
            cmap = matplotlib.colors.ListedColormap(colmap)
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_under(plotvars.cs[0])
            if (
                plotvars.levels_extend == "max"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_over(plotvars.cs[-1])

            plotvars.image = plotvars.plot.contourf(
                x,
                y,
                field * fmult,
                clevs,
                extend=plotvars.levels_extend,
                cmap=cmap,
                norm=plotvars.norm,
                alpha=alpha,
                zorder=zorder,
            )

        # Block fill
        if blockfill:
            if isinstance(f, cf.Field):
                if f.coord("X").has_bounds():
                    if ptype == 4:
                        xpts = np.squeeze(f.coord("X").bounds.array)[:, 0]
                        xpts = np.append(
                            xpts, f.coord("X").bounds.array[-1, 1]
                        )
                    if ptype == 5:
                        xpts = np.squeeze(f.coord("Y").bounds.array)[:, 0]
                        xpts = np.append(
                            xpts, f.coord("Y").bounds.array[-1, 1]
                        )
                    ypts = np.squeeze(f.coord("T").bounds.array)[:, 0]
                    ypts = np.append(ypts, f.coord("T").bounds.array[-1, 1])
                    if swap_axes:
                        xpts, ypts = ypts, xpts
                        field_orig = np.flipud(np.rot90(field_orig))

                    _bfill(
                        f=field_orig * fmult,
                        x=xpts,
                        y=ypts,
                        clevs=clevs,
                        lonlat=False,
                        bound=1,
                        alpha=alpha,
                        fast=blockfill_fast,
                        zorder=zorder,
                    )
                else:
                    if swap_axes:
                        x_orig, y_orig = y_orig, x_orig
                        field_orig = np.flipud(np.rot90(field_orig))
                    _bfill(
                        f=field_orig * fmult,
                        x=x_orig,
                        y=y_orig,
                        clevs=clevs,
                        lonlat=False,
                        bound=0,
                        alpha=alpha,
                        fast=blockfill_fast,
                        zorder=zorder,
                    )

            else:
                if swap_axes:
                    x_orig, y_orig = y_orig, x_orig
                    field_orig = np.flipud(np.rot90(field_orig))
                _bfill(
                    f=field_orig * fmult,
                    x=x_orig,
                    y=y_orig,
                    clevs=clevs,
                    lonlat=False,
                    bound=0,
                    alpha=alpha,
                    fast=blockfill_fast,
                    zorder=zorder,
                )

        # Contour lines and labels
        if lines:
            cs = plotvars.plot.contour(
                x,
                y,
                field * fmult,
                clevs,
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyles,
                alpha=alpha,
            )
            if line_labels and not isinstance(clevs, int):
                nd = ndecs(clevs)
                fmt = "%d"
                if nd != 0:
                    fmt = "%1." + str(nd) + "f"
                plotvars.plot.clabel(
                    cs,
                    fmt=fmt,
                    colors=colors,
                    zorder=zorder,
                    fontsize=text_fontsize,
                )

                # Thick zero contour line
                if zero_thick:
                    cs = plotvars.plot.contour(
                        x,
                        y,
                        field * fmult,
                        [-1e-32, 0],
                        colors=colors,
                        linewidths=zero_thick,
                        linestyles=linestyles,
                        alpha=alpha,
                        zorder=zorder,
                    )
        # Titles for dimensions
        if titles:
            _dim_titles(title=title_dims)

        # Color bar
        if colorbar:
            cbar(
                labels=cbar_labels,
                orientation=cb_orient,
                position=colorbar_position,
                shrink=colorbar_shrink,
                title=colorbar_title,
                fontsize=colorbar_fontsize,
                fontweight=colorbar_fontweight,
                text_up_down=colorbar_text_up_down,
                text_down_up=colorbar_text_down_up,
                drawedges=colorbar_drawedges,
                fraction=colorbar_fraction,
                thick=colorbar_thick,
                levs=clevs,
                anchor=colorbar_anchor,
                verbose=verbose,
            )

        # Title
        plotvars.plot.set_title(
            title, y=1.03, fontsize=title_fontsize, fontweight=title_fontweight
        )

        # reset plot limits if not a user plot
        if user_gset == 0:
            gset()

    ###########################
    # Rotated pole contour plot
    ###########################
    if ptype == 6:

        # Extract x and y grid points
        if plotvars.proj == "cyl":
            xpts = x
            ypts = y
        else:
            xpts = np.arange(np.size(x))
            ypts = np.arange(np.size(y))

        if verbose:
            print("con - making a rotated pole plot")
        user_gset = plotvars.user_gset
        if plotvars.user_plot == 0:
            gopen(user_plot=0)

        # Set plot limits
        if plotvars.proj == "rotated":
            plotargs = {}
            gset(
                xmin=0,
                xmax=np.size(xpts) - 1,
                ymin=0,
                ymax=np.size(ypts) - 1,
                user_gset=user_gset,
            )
            plot = plotvars.plot

        # Set plot limits
        if plotvars.proj == "UKCP":
            plot = plotvars.plot
            plotargs = {}

        if plotvars.proj == "cyl":
            rotated_pole = f.ref(
                "grid_mapping_name:rotated_latitude_longitude"
            )
            xpole = rotated_pole["grid_north_pole_longitude"]
            ypole = rotated_pole["grid_north_pole_latitude"]

            transform = ccrs.RotatedPole(
                pole_latitude=ypole, pole_longitude=xpole
            )
            plotargs = {"transform": transform}
            if plotvars.user_mapset == 1:
                _set_map()
            else:
                if np.ndim(xpts) == 1:
                    lonpts, latpts = np.meshgrid(xpts, ypts)
                else:
                    lonpts = xpts
                    latpts = ypts

                points = ccrs.PlateCarree().transform_points(
                    transform, lonpts.flatten(), latpts.flatten()
                )
                lons = np.array(points)[:, 0]
                lats = np.array(points)[:, 1]

                mapset(
                    lonmin=np.min(lons),
                    lonmax=np.max(lons),
                    latmin=np.min(lats),
                    latmax=np.max(lats),
                    user_mapset=0,
                    resolution=resolution_orig,
                )
                _set_map()

            plotargs = {"transform": transform}
            plot = plotvars.mymap

        # Get colour scale for use in contouring
        # If colour bar extensions are enabled then the colour map goes
        # from 1 to ncols-2.  The colours for the colour bar extensions are
        # then changed on the colorbar and plot after the plot is made
        colmap = _cscale_get_map()

        # Filled contours
        if fill:
            colmap = _cscale_get_map()
            cmap = matplotlib.colors.ListedColormap(colmap)
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_under(plotvars.cs[0])
            if (
                plotvars.levels_extend == "max"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_over(plotvars.cs[-1])

            plot.contourf(
                xpts,
                ypts,
                field * fmult,
                clevs,
                extend=plotvars.levels_extend,
                cmap=cmap,
                norm=plotvars.norm,
                alpha=alpha,
                zorder=zorder,
                **plotargs,
            )

        # Block fill
        if blockfill:
            _bfill(
                f=field_orig * fmult,
                x=xpts,
                y=ypts,
                clevs=clevs,
                lonlat=False,
                bound=0,
                alpha=alpha,
                fast=blockfill_fast,
                zorder=zorder,
                transform=transform,
            )

        # Contour lines and labels
        if lines:
            cs = plot.contour(
                xpts,
                ypts,
                field * fmult,
                clevs,
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=zorder,
                **plotargs,
            )
            if line_labels and not isinstance(clevs, int):
                nd = ndecs(clevs)
                fmt = "%d"
                if nd != 0:
                    fmt = "%1." + str(nd) + "f"
                plot.clabel(
                    cs,
                    fmt=fmt,
                    colors=colors,
                    zorder=zorder,
                    fontsize=text_fontsize,
                )

            # Thick zero contour line
            if zero_thick:
                cs = plot.contour(
                    xpts,
                    ypts,
                    field * fmult,
                    [-1e-32, 0],
                    colors=colors,
                    linewidths=zero_thick,
                    linestyles=linestyles,
                    alpha=alpha,
                    zorder=zorder,
                    **plotargs,
                )

        # Titles for dimensions
        if titles:
            _dim_titles(title=title_dims)

        # Color bar
        if colorbar:
            cbar(
                labels=cbar_labels,
                orientation=cb_orient,
                position=colorbar_position,
                shrink=colorbar_shrink,
                title=colorbar_title,
                fontsize=colorbar_fontsize,
                fontweight=colorbar_fontweight,
                text_up_down=colorbar_text_up_down,
                text_down_up=colorbar_text_down_up,
                drawedges=colorbar_drawedges,
                fraction=colorbar_fraction,
                thick=colorbar_thick,
                levs=clevs,
                anchor=colorbar_anchor,
                verbose=verbose,
            )

        if plotvars.proj == "rotated" or plotvars.proj == "UKCP":
            # Remove Matplotlib default axis labels. Note we must do this
            # before we add in our custom (rotated or UKCP) axes labels
            # else they will also be wiped, along with the plot area!
            axes_plot(
                xticks=[100000000],
                xticklabels=[""],
                yticks=[100000000],
                yticklabels=[""],
                xlabel="",
                ylabel="",
            )

        # Rotated grid axes
        if axes:
            if plotvars.proj == "cyl":
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
            else:
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

        # Add title and coastlines for cylindrical projection
        if plotvars.proj == "cyl":
            # Coastlines
            feature = cfeature.NaturalEarthFeature(
                name="land",
                category="physical",
                scale=plotvars.resolution,
                facecolor="none",
            )
            plotvars.mymap.add_feature(
                feature,
                edgecolor=continent_color,
                linewidth=continent_thickness,
                linestyle=continent_linestyle,
                zorder=zorder,
            )

            # Title
            if title != "":
                _map_title(title)

        # Add title for native grid
        if plotvars.proj == "rotated":
            # Title
            plotvars.plot.set_title(
                title,
                y=1.03,
                fontsize=title_fontsize,
                fontweight=title_fontweight,
            )

        # reset plot limits if not a user plot
        if plotvars.user_gset == 0:
            gset()

    #############
    # Other plots
    #############
    if ptype == 0:
        if verbose:
            print("con - making an other plot")
        if plotvars.user_plot == 0:
            gopen(user_plot=0)
        user_gset = plotvars.user_gset

        # Set axis labels to None
        xplotlabel = None
        yplotlabel = None

        cf_field = False
        if f is not None:
            if isinstance(f, cf.Field):
                cf_field = True
                f = f.squeeze()

        # Work out axes if none are supplied
        if any(
            val is None
            for val in [
                plotvars.xmin,
                plotvars.xmax,
                plotvars.ymin,
                plotvars.ymax,
            ]
        ):
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            ymin = np.nanmin(y)
            ymax = np.nanmax(y)
        else:
            xmin = plotvars.xmin
            xmax = plotvars.xmax
            ymin = plotvars.ymin
            ymax = plotvars.ymax

        # Change from date string to a number if strings are passed
        time_xstr = False
        time_ystr = False

        try:
            float(xmin)
        except Exception:
            time_xstr = True
        try:
            float(ymin)
        except Exception:
            time_ystr = True

        xaxisticks = None
        yaxisticks = None
        xtimeaxis = False
        ytimeaxis = False

        if cf_field and f.has_construct("T"):
            if np.size(f.construct("T").array) > 1:

                taxis = f.construct("T")

                data_axes = f.get_data_axes()
                count = 1
                for d in data_axes:
                    i = f.constructs.domain_axis_identity(d)
                    try:
                        c = f.coordinate([i])
                        if np.size(c.array) > 1:
                            test_for_time_axis = False
                            sn = getattr(c, "standard_name", "NoName")
                            an = c.get_property("axis", "NoName")
                            if sn == "time" or an == "T":
                                test_for_time_axis = True

                            if count == 1:
                                if test_for_time_axis:
                                    ytimeaxis = True
                            elif count == 2:
                                if test_for_time_axis:
                                    xtimeaxis = True
                            count += 1
                    except ValueError:
                        print("no sensible coordinates for this axis")

                if time_xstr or time_ystr:
                    ref_time = f.construct("T").units
                    ref_calendar = f.construct("T").calendar
                    time_units = cf.Units(ref_time, ref_calendar)

                    if time_xstr:
                        t = cf.Data(cf.dt(xmin), units=time_units)
                        xmin = t.array
                        t = cf.Data(cf.dt(xmax), units=time_units)
                        xmax = t.array
                        taxis = cf.Data([xmin, xmax], units=time_units)
                        taxis.calendar = ref_calendar

                    if time_ystr:
                        t = cf.Data(cf.dt(ymin), units=time_units)
                        ymin = t.array
                        t = cf.Data(cf.dt(ymax), units=time_units)
                        ymax = t.array
                        taxis = cf.Data([ymin, ymax], units=time_units)
                        taxis.calendar = ref_calendar

                if xtimeaxis:
                    xaxisticks, xaxislabels, xplotlabel = _timeaxis(taxis)
                if ytimeaxis:
                    yaxisticks, yaxislabels, yplotlabel = _timeaxis(taxis)

        if cf_field:
            coords = list(f.coords())
            mycoords = []
            for coord in coords:
                if np.size(f.coord(coord).array) > 1:
                    mycoords.append(coord)
            mycoords.reverse()

            for icoord in np.arange(len(mycoords)):

                myaxisticks = None
                myaxislabels = None
                mylabel = None

                if f.coord(mycoords[icoord]).X:
                    myaxisticks, myaxislabels = _mapaxis(
                        np.min(f.coord("X").array),
                        np.max(f.coord("X").array),
                        type=1,
                    )
                    mylabel = "longitude"

                if f.coord(mycoords[icoord]).Y:
                    myaxisticks, myaxislabels = _mapaxis(
                        np.min(f.coord("Y").array),
                        np.max(f.coord("Y").array),
                        type=2,
                    )
                    mylabel = "latitude"

                if myaxisticks is not None:
                    if icoord == 0:
                        xaxisticks, xaxislabels, xlabel = (
                            myaxisticks,
                            myaxislabels,
                            mylabel,
                        )
                    if icoord == 1:
                        yaxisticks, yaxislabels, ylabel = (
                            myaxisticks,
                            myaxislabels,
                            mylabel,
                        )

        if xaxisticks is None:
            xaxisticks = _gvals(dmin=xmin, dmax=xmax, mod=False)[0]
            xaxislabels = xaxisticks

        if yaxisticks is None:
            yaxisticks = _gvals(dmin=ymax, dmax=ymin, mod=False)[0]
            yaxislabels = yaxisticks

        if user_xlabel is not None:
            xplotlabel = user_xlabel
        else:
            if xplotlabel is None:
                xplotlabel = xlabel
        if user_ylabel is not None:
            yplotlabel = user_ylabel
        else:
            if yplotlabel is None:
                yplotlabel = ylabel

        # Draw axes
        if axes:
            if xaxis:
                if xticks is not None:
                    xaxisticks = xticks
                    xaxislabels = xticks
                    if xticklabels is not None:
                        xaxislabels = xticklabels
            else:
                xaxisticks = [100000000]
                xlabel = ""

            if yaxis:
                if yticks is not None:
                    yaxisticks = yticks
                    yaxislabels = yticks
                    if yticklabels is not None:
                        yaxislabels = yticklabels
            else:
                yaxisticks = [100000000]
                ylabel = ""

        else:
            xaxisticks = [100000000]
            yaxisticks = [100000000]
            xlabel = ""
            ylabel = ""

        # Swap axes if requested
        if swap_axes:
            x, y = y, x
            field = np.flipud(np.rot90(field))
            xmin, ymin = ymin, xmin
            xmax, ymax = ymax, xmax
            xplotlabel, yplotlabel = yplotlabel, xplotlabel
            xaxisticks, yaxisticks = yaxisticks, xaxisticks
            xaxislabels, yaxislabels = yaxislabels, xaxislabels

        # Set plot limits and set default plot labels
        gset(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, user_gset=user_gset)

        # Draw axes
        axes_plot(
            xticks=xaxisticks,
            xticklabels=xaxislabels,
            yticks=yaxisticks,
            yticklabels=yaxislabels,
            xlabel=xplotlabel,
            ylabel=yplotlabel,
        )

        # Get colour scale for use in contouring
        # If colour bar extensions are enabled then the colour map goes
        # then from 1 to ncols-2.  The colours for the colour bar extensions
        # are changed on the colorbar and plot after the plot is made
        colmap = _cscale_get_map()

        # Filled contours
        if fill:
            colmap = _cscale_get_map()
            cmap = matplotlib.colors.ListedColormap(colmap)
            if (
                plotvars.levels_extend == "min"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_under(plotvars.cs[0])
            if (
                plotvars.levels_extend == "max"
                or plotvars.levels_extend == "both"
            ):
                cmap.set_over(plotvars.cs[-1])

            plotvars.image = plotvars.plot.contourf(
                x,
                y,
                field * fmult,
                clevs,
                extend=plotvars.levels_extend,
                cmap=cmap,
                norm=plotvars.norm,
                alpha=alpha,
                zorder=zorder,
            )

        # Block fill
        if blockfill:
            _bfill(
                f=field_orig * fmult,
                x=x_orig,
                y=y_orig,
                clevs=clevs,
                lonlat=False,
                bound=0,
                alpha=alpha,
                fast=blockfill_fast,
                zorder=zorder,
            )

        # Contour lines and labels
        if lines:
            cs = plotvars.plot.contour(
                x,
                y,
                field * fmult,
                clevs,
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=zorder,
            )
            if line_labels and not isinstance(clevs, int):
                nd = ndecs(clevs)
                fmt = "%d"
                if nd != 0:
                    fmt = "%1." + str(nd) + "f"
                plotvars.plot.clabel(
                    cs,
                    fmt=fmt,
                    colors=colors,
                    zorder=zorder,
                    fontsize=text_fontsize,
                )

            # Thick zero contour line
            if zero_thick:
                cs = plotvars.plot.contour(
                    x,
                    y,
                    field * fmult,
                    [-1e-32, 0],
                    colors=colors,
                    linewidths=zero_thick,
                    linestyles=linestyles,
                    alpha=alpha,
                    zorder=zorder,
                )

        # Titles for dimensions
        if titles:
            _dim_titles(title=title_dims)

        # Color bar
        if colorbar:
            cbar(
                labels=cbar_labels,
                orientation=cb_orient,
                position=colorbar_position,
                shrink=colorbar_shrink,
                title=colorbar_title,
                fontsize=colorbar_fontsize,
                fontweight=colorbar_fontweight,
                text_up_down=colorbar_text_up_down,
                text_down_up=colorbar_text_down_up,
                drawedges=colorbar_drawedges,
                fraction=colorbar_fraction,
                thick=colorbar_thick,
                levs=clevs,
                anchor=colorbar_anchor,
                verbose=verbose,
            )

        # Title
        plotvars.plot.set_title(
            title, y=1.03, fontsize=title_fontsize, fontweight=title_fontweight
        )

        # reset plot limits if not a user plot
        if plotvars.user_gset == 0:
            gset()

    ############################
    # Set axis width if required
    ############################
    if plotvars.axis_width is not None:
        for axis in ["top", "bottom", "left", "right"]:
            plotvars.plot.spines[axis].set_linewidth(plotvars.axis_width)

    ################################
    # Add a master title if reqested
    ################################
    if plotvars.master_title is not None:
        location = plotvars.master_title_location
        plotvars.master_plot.text(
            location[0],
            location[1],
            plotvars.master_title,
            horizontalalignment="center",
            fontweight=plotvars.master_title_fontweight,
            fontsize=plotvars.master_title_fontsize,
        )

    # Reset map resolution
    if plotvars.user_mapset == 0:
        mapset()
        mapset(resolution=resolution_orig)

    ##################
    # Save or view plot
    ##################

    if plotvars.user_plot == 0:
        if verbose:
            print("con - saving or viewing plot")

        np.seterr(**old_settings)  # reset to default numpy error settings

        gclose()



def ndecs(data=None):
    """
    | Finds the number of decimal places in an array.
    | Needed to make the colour bar match the contour line labelling.

    | data=data - input array of values

    :Returns:
    |  maximum number of necimal places
    |
    """

    maxdecs = 0

    for i in range(len(data)):
        number = data[i]
        a = str(number).split(".")
        if np.size(a) == 2:
            number_decs = len(a[1])
            if number_decs > maxdecs:
                maxdecs = number_decs

    return maxdecs


def pcon(mb=None, km=None, h=7.0, p0=1000):
    """
    | Convert pressure to height in kilometers and vice-versa.
    | This function uses the equation P=P0exp(-z/H) to translate
    | between pressure and height. In pcon the surface pressure P0 is set to
    | 1000.0mb and the scale height H is set to 7.0. The value of H can vary
    | from 6.0 in the polar regions to 8.5 in the tropics as well as
    | seasonally. The value of P0 could also be said to be 1013.25mb rather
    | than 1000.0mb.

    | As this relationship is approximate:
    | (i) Only use this for making the axis labels on y axis pressure plots
    | (ii) Put the converted axis on the right hand side to indicate that
    |      this isn't the primary unit of measure

    | print cfp.pcon(mb=[1000, 300, 100, 30, 10, 3, 1, 0.3])
    | [0. 8.42780963 16.11809565 24.54590528 32.2361913
    |  40.66400093 48.35428695, 56.78209658]

    | mb=None - input pressure
    | km=None - input height
    | h=7.0 - default value for h
    | p0=1000 - default value for p0

    :Returns:
     | pressure(mb) if height(km) input,
     | height(km) if pressure(mb) input
    """

    if all(val is None for val in [mb, km]) == 2:
        errstr = "pcon error - pcon must have mb or km input\n"
        raise Warning(errstr)

    if mb is not None:
        return h * (np.log(p0) - np.log(mb))
    if km is not None:
        return np.exp(-1.0 * (np.array(km) / h - np.log(p0)))


def regrid(f=None, x=None, y=None, xnew=None, ynew=None):
    """
    | Bilinear interpolation of a grid to new grid locations.
    |
    |     f=None - original field
    |     x=None - original field x values
    |     y=None - original field y values
    |     xnew=None - new x points
    |     ynew=None - new y points
    |
    :Returns:
       field values at requested locations
    |
    """
    # TODO SLB: what is this method for and why named/described as such???

    # Copy input arrays
    regrid_f = deepcopy(f)
    regrid_x = deepcopy(x)
    regrid_y = deepcopy(y)
    fieldout = []

    # Reverse xpts and field if necessary
    if regrid_x[0] > regrid_x[-1]:
        regrid_x = regrid_x[::-1]
        regrid_f = np.fliplr(regrid_f)

    # Reverse ypts and field if necessary
    if regrid_y[0] > regrid_y[-1]:
        regrid_y = regrid_y[::-1]
        regrid_f = np.flipud(regrid_f)

    # Iterate over the new grid to get the new grid values.
    for i in np.arange(np.size(xnew)):

        xval = xnew[i]
        yval = ynew[i]

        # Find position of new grid point in the x and y arrays
        myxpos = find_pos_in_array(vals=regrid_x, val=xval)
        myypos = find_pos_in_array(vals=regrid_y, val=yval)

        myxpos2 = myxpos + 1
        myypos2 = myypos + 1

        if myxpos2 != myxpos:
            alpha = (xnew[i] - regrid_x[myxpos]) / (
                regrid_x[myxpos2] - regrid_x[myxpos]
            )
        else:
            alpha = (xnew[i] - regrid_x[myxpos]) / 1e-30

        newval1 = regrid_f[myypos, myxpos] - regrid_f[myypos, myxpos2]
        newval1 = newval1 * alpha
        newval1 = regrid_f[myypos, myxpos] - newval1

        newval2 = regrid_f[myypos2, myxpos] - regrid_f[myypos2, myxpos2]
        newval2 = newval2 * alpha
        newval2 = regrid_f[myypos2, myxpos] - newval2

        if myypos2 != myypos:
            alpha2 = ynew[i] - regrid_y[myypos]
            alpha2 = alpha2 / (regrid_y[myypos2] - regrid_y[myypos])
        else:
            alpha2 = (ynew[i] - regrid_y[myypos]) / 1e-30

        newval3 = newval1 - (newval1 - newval2) * alpha2

        fieldout = np.append(fieldout, newval3)

    return fieldout


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


def stipple_points(
    xmin=None, xmax=None, ymin=None, ymax=None, pts=None, stype=None
):
    """
    | Calculate interpolation points.
    |
    | xmin=None - plot x minimum
    | ymax=None - plot x maximum
    | ymin=None - plot y minimum
    | ymax=None - plot x maximum
    | pts=None -  number of points in the x and y directions
    |             one number gives the same in both directions
    |
    | stype=None - type of grid.  1=regular, 2=offset
    |
    :Returns:
       stipple locations in x and y
    |
    """

    # Work out number of points in x and y directions
    if np.size(pts) == 1:
        pts_x = pts
        pts_y = pts
    if np.size(pts) == 2:
        pts_x = pts[0]
        pts_y = pts[1]

    # Create regularly spaced points
    xstep = (xmax - xmin) / float(pts_x)
    x1 = [xmin + xstep / 4]
    while (np.nanmax(x1) + xstep) < xmax - xstep / 10:
        x1 = np.append(x1, np.nanmax(x1) + xstep)

    x2 = [xmin + xstep * 3 / 4]
    while (np.nanmax(x2) + xstep) < xmax - xstep / 10:
        x2 = np.append(x2, np.nanmax(x2) + xstep)

    ystep = (ymax - ymin) / float(pts_y)
    y1 = [ymin + ystep / 2]
    while (np.nanmax(y1) + ystep) < ymax - ystep / 10:
        y1 = np.append(y1, np.nanmax(y1) + ystep)

    # Create interpolation points
    xnew = []
    ynew = []
    iy = 0

    for y in y1:
        iy = iy + 1
        if stype == 1:
            xnew = np.append(xnew, x1)
            y2 = np.zeros(np.size(x1))
            y2.fill(y)
            ynew = np.append(ynew, y2)

        if stype == 2:
            if iy % 2 == 0:
                xnew = np.append(xnew, x1)
                y2 = np.zeros(np.size(x1))
                y2.fill(y)
                ynew = np.append(ynew, y2)
            if iy % 2 == 1:
                xnew = np.append(xnew, x2)
                y2 = np.zeros(np.size(x2))
                y2.fill(y)
                ynew = np.append(ynew, y2)

    return xnew, ynew


def find_pos_in_array(vals=None, val=None, above=False):
    """
    | Find the position of a point in an array.
    |
    | vals - array values
    | val - value to find position of
    |
    :Returns:
      position in array
    |
    """

    pos = -1
    if above is False:
        for myval in vals:
            if val > myval:
                pos = pos + 1

    if above:
        for myval in vals:
            if val >= myval:
                pos = pos + 1

        if np.size(vals) - 1 > pos:
            pos = pos + 1

    return pos


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


def polar_regular_grid(pts=50):
    """
    | Return a regular grid over a polar stereographic area.
    |
    | pts=50 - number  of grid points in the x and y directions
    |
    :Returns:
     lons, lats of grid in degrees
     x, y locations of lons and lats
    """

    boundinglat = plotvars.boundinglat
    lon_0 = plotvars.lon_0

    if plotvars.proj == "npstere":
        thisproj = ccrs.NorthPolarStereo(central_longitude=lon_0)
    else:
        thisproj = ccrs.SouthPolarStereo(central_longitude=lon_0)

    # Find min and max of plotting region in device coordinates
    lons = np.array([lon_0 - 90, lon_0, lon_0 + 90, lon_0 + 180])
    lats = np.array([boundinglat, boundinglat, boundinglat, boundinglat])
    extent = thisproj.transform_points(ccrs.PlateCarree(), lons, lats)

    xmin = np.min(extent[:, 0])
    xmax = np.max(extent[:, 0])
    ymin = np.min(extent[:, 1])
    ymax = np.max(extent[:, 1])

    # Make up a stipple of points for cover the pole
    points_device = stipple_points(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, pts=pts, stype=2
    )

    xnew = np.array(points_device)[0, :]
    ynew = np.array(points_device)[1, :]

    points_polar = ccrs.PlateCarree().transform_points(thisproj, xnew, ynew)

    lons = np.array(points_polar)[:, 0]
    lats = np.array(points_polar)[:, 1]

    if plotvars.proj == "npstere":
        valid = np.where(lats >= boundinglat)
    else:
        valid = np.where(lats <= boundinglat)

    return lons[valid], lats[valid], xnew[valid], ynew[valid]


def cf_var_name(field=None, dim=None):
    """
    | Return the name from a supplied dimension in order.
    |
    | Names are returned in the following order:
    | * ncvar
    | * short_name
    | * long_name
    | * standard_name
    |
    | field=None - field
    | dim=None - dimension required - 'dim0', 'dim1' etc.
    |
    :Returns:
     name
    """

    # Check for multiple Z coordinates
    # Adjust dim if necessary
    if dim == "Z":
        z_count = 0
        z_names = []
        for mycoord in list(field.coords()):
            if field.coord(mycoord).Z:
                z_count += 1
                z_names.append(mycoord)

        if z_count > 1:
            dim = z_names[-1]

    id = getattr(field.construct(dim), "id", False)
    ncvar = field.construct(dim).nc_get_variable(False)
    short_name = getattr(field.construct(dim), "short_name", False)
    long_name = getattr(field.construct(dim), "long_name", False)
    standard_name = getattr(field.construct(dim), "standard_name", False)

    name = "No Name"
    if id:
        name = id
    if ncvar:
        name = ncvar
    if short_name:
        name = short_name
    if long_name:
        name = long_name
    if standard_name:
        name = standard_name

    return name


def cf_var_name_titles(field=None, dim=None):
    """
    | Return the name from a supplied dimension in order.
    |
    | Names are returned in the following order:
    | * standard_name
    | * long_name
    | * short_name
    | * ncvar

    | field=None - field
    | dim=None - dimension required - 'dim0', 'dim1' etc.
    |
    :Returns:
     name
    """
    # TODO SLB: combine with 'cf_var_name', which does reverse

    name = None
    units = None
    if field.has_construct(dim):

        id = getattr(field.construct(dim), "id", False)
        ncvar = field.construct(dim).nc_get_variable(False)
        short_name = getattr(field.construct(dim), "short_name", False)
        long_name = getattr(field.construct(dim), "long_name", False)
        standard_name = getattr(field.construct(dim), "standard_name", False)

        # name = 'No Name'
        if id:
            name = id
        if ncvar:
            name = ncvar
        if short_name:
            name = short_name
        if long_name:
            name = long_name
        if standard_name:
            name = standard_name

        units = getattr(field.construct(dim), "units", "")
        if len(units) > 0:
            units = f"({units})"
    return name, units



def vloc(xvec=None, yvec=None, lons=None, lats=None):
    """
    | Locate the positions of a set of points in a vector.
    |
    | xvec=None - data longitudes
    | yvec=None - data latitudes
    | lons=None - required longitude positions
    | lats=None - required latitude positions

    :Returns:
     locations of user points in the longitude and latitude points
    """

    # Check input parameters
    if any(val is None for val in [xvec, yvec, lons, lats]):
        errstr = (
            "\nvloc error\n"
            "xvec, yvec, lons, lats all need to be passed to vloc to\n"
            "generate a set of location points\n"
        )
        raise Warning(errstr)

    xarr = np.zeros(np.size(lons))
    yarr = np.zeros(np.size(lats))

    # Convert longitudes to -180 to 180.
    for i in np.arange(np.size(xvec)):
        xvec[i] = ((xvec[i] + 180) % 360) - 180
    for i in np.arange(np.size(lons)):
        lons[i] = ((lons[i] + 180) % 360) - 180

    # Centre around 180 degrees longitude if needed.
    if max(xvec) > 150:
        for i in np.arange(np.size(xvec)):
            xvec[i] = (xvec[i] + 360.0) % 360.0
        pts = np.where(xvec < 0.0)
        xvec[pts] = xvec[pts] + 360.0
        for i in np.arange(np.size(lons)):
            lons[i] = (lons[i] + 360.0) % 360.0
        pts = np.where(lons < 0.0)
        lons[pts] = lons[pts] + 360.0

    # Find position in array
    for i in np.arange(np.size(lons)):

        if (lons[i] < min(xvec)) or (lons[i] > max(xvec)):
            xpt = -1
        else:
            xpts = np.where(lons[i] >= xvec)
            xpt = np.nanmax(xpts)

        if (lats[i] < min(yvec)) or (lats[i] > max(yvec)):
            ypt = -1
        else:
            ypts = np.where(lats[i] >= yvec)
            ypt = np.nanmax(ypts)

        if xpt >= 0:
            xarr[i] = xpt + (lons[i] - xvec[xpt]) / (xvec[xpt + 1] - xvec[xpt])
        else:
            xarr[i] = None

        if (ypt >= 0) and ypt <= np.size(yvec) - 2:
            yarr[i] = ypt + (lats[i] - yvec[ypt]) / (yvec[ypt + 1] - yvec[ypt])
        else:
            yarr[i] = None

    return (xarr, yarr)


def rgaxes(
    xpole=None,
    ypole=None,
    xvec=None,
    yvec=None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    axes=None,
    xaxis=None,
    yaxis=None,
    xlabel=None,
    ylabel=None,
):
    """
    | Label rotated grid plots.
    |
    | xpole=None - location of xpole in degrees
    | ypole=None - location of ypole in degrees
    | xvec=None - location of x grid points
    | yvec=None - location of y grid points
    |
    | axes=True - plot x and y axes
    | xaxis=True - plot xaxis
    | yaxis=True - plot y axis
    | xticks=None - xtick positions
    | xticklabels=None - xtick labels
    | yticks=None - y tick positions
    | yticklabels=None - ytick labels
    | xlabel=None - label for x axis
    | ylabel=None - label for y axis
    |
    :Returns:
     name
    """

    spacing = plotvars.rotated_grid_spacing
    degspacing = plotvars.rotated_deg_spacing
    continents = plotvars.rotated_continents
    grid = plotvars.rotated_grid
    labels = plotvars.rotated_labels
    grid_thickness = plotvars.rotated_grid_thickness

    # Invert y array if going from north to south
    # Otherwise this gives nans for all output
    yvec_orig = yvec
    if yvec[0] > yvec[np.size(yvec) - 1]:
        yvec = yvec[::-1]

    gset(
        xmin=0,
        xmax=np.size(xvec) - 1,
        ymin=0,
        ymax=np.size(yvec) - 1,
        user_gset=0,
    )

    # Set continent thickness and color if not already set
    if plotvars.continent_thickness is None:
        continent_thickness = 1.5
    if plotvars.continent_color is None:
        continent_color = "k"

    # Draw continents
    if continents:

        import cartopy.io.shapereader as shpreader
        import shapefile

        shpfilename = shpreader.natural_earth(
            resolution=plotvars.resolution,
            category="physical",
            name="coastline",
        )
        reader = shapefile.Reader(shpfilename)
        shapes = [s.points for s in reader.shapes()]
        for shape in shapes:
            lons, lats = list(zip(*shape))
            lons = np.array(lons)
            lats = np.array(lats)

            rotated_transform = ccrs.RotatedPole(
                pole_latitude=ypole, pole_longitude=xpole
            )
            points = rotated_transform.transform_points(
                ccrs.PlateCarree(), lons, lats
            )
            xout = np.array(points)[:, 0]
            yout = np.array(points)[:, 1]

            xpts, ypts = vloc(lons=xout, lats=yout, xvec=xvec, yvec=yvec)
            plotvars.plot.plot(
                xpts,
                ypts,
                linewidth=continent_thickness,
                color=continent_color,
            )

    if xticks is None:
        lons = -180 + np.arange(360 / spacing + 1) * spacing
    else:
        lons = xticks
    if yticks is None:
        lats = -90 + np.arange(180 / spacing + 1) * spacing
    else:
        lats = yticks

    # Work out how far from plot to plot the longitude and latitude labels
    xlim = plotvars.plot.get_xlim()
    spacing_x = (xlim[1] - xlim[0]) / 20
    ylim = plotvars.plot.get_ylim()
    spacing_y = (ylim[1] - ylim[0]) / 20
    spacing = min(spacing_x, spacing_y)

    # Draw lines along a longitude
    if axes:
        if xaxis:
            for val in np.arange(np.size(lons)):
                ipts = 179.0 / degspacing
                lona = np.zeros(int(ipts)) + lons[val]
                lata = -90 + np.arange(ipts - 1) * degspacing

                rotated_transform = ccrs.RotatedPole(
                    pole_latitude=ypole, pole_longitude=xpole
                )
                points = rotated_transform.transform_points(
                    ccrs.PlateCarree(), lona, lata
                )
                xout = np.array(points)[:, 0]
                yout = np.array(points)[:, 1]

                xpts, ypts = vloc(lons=xout, lats=yout, xvec=xvec, yvec=yvec)
                if grid:
                    plotvars.plot.plot(
                        xpts, ypts, ":", linewidth=grid_thickness, color="k"
                    )

                if labels:
                    # Make a label unless the axis is all Nans
                    if np.size(ypts[5:]) > np.sum(np.isnan(ypts[5:])):
                        ymin = np.nanmin(ypts[5:])
                        loc = np.where(ypts == ymin)[0]
                        if np.size(loc) > 1:
                            loc = loc[1]

                        if loc > 0:
                            if np.isfinite(xpts[loc]):
                                line = matplotlib.lines.Line2D(
                                    [xpts[loc], xpts[loc]],
                                    [0, -spacing / 2],
                                    color="k",
                                )
                                plotvars.plot.add_line(line)
                                line.set_clip_on(False)
                                fw = plotvars.text_fontweight
                                if xticklabels is None:
                                    xticklabel = _mapaxis(
                                        lons[val], lons[val], type=1
                                    )[1][0]
                                else:
                                    xticklabel = xticks[val]

                                plotvars.plot.text(
                                    xpts[loc],
                                    -spacing,
                                    xticklabel,
                                    horizontalalignment="center",
                                    verticalalignment="top",
                                    fontsize=plotvars.text_fontsize,
                                    fontweight=fw,
                                )

    # Draw lines along a latitude
    if axes:
        if yaxis:
            for val in np.arange(np.size(lats)):
                ipts = 359.0 / degspacing
                lata = np.zeros(int(ipts)) + lats[val]
                lona = -180.0 + np.arange(ipts - 1) * degspacing

                rotated_transform = ccrs.RotatedPole(
                    pole_latitude=ypole, pole_longitude=xpole
                )
                points = rotated_transform.transform_points(
                    ccrs.PlateCarree(), lona, lata
                )
                xout = np.array(points)[:, 0]
                yout = np.array(points)[:, 1]
                xpts, ypts = vloc(lons=xout, lats=yout, xvec=xvec, yvec=yvec)

                if grid:
                    plotvars.plot.plot(
                        xpts, ypts, ":", linewidth=grid_thickness, color="k"
                    )

                if labels:
                    # Make a label unless the axis is all Nans
                    if np.size(xpts[5:]) > np.sum(np.isnan(xpts[5:])):
                        xmin = np.nanmin(xpts[5:])
                        loc = np.where(xpts == xmin)[0]
                        if np.size(loc) == 1:
                            if loc > 0:
                                if np.isfinite(ypts[loc]):
                                    line = matplotlib.lines.Line2D(
                                        [0, -spacing / 2],
                                        [ypts[loc], ypts[loc]],
                                        color="k",
                                    )
                                    plotvars.plot.add_line(line)
                                    line.set_clip_on(False)
                                    fw = plotvars.text_fontweight
                                    if yticklabels is None:
                                        yticklabel = _mapaxis(
                                            lats[val], lats[val], type=2
                                        )[1][0]
                                    else:
                                        yticklabel = yticks[val]

                                    plotvars.plot.text(
                                        -spacing,
                                        ypts[loc],
                                        yticklabel,
                                        horizontalalignment="right",
                                        verticalalignment="center",
                                        fontsize=plotvars.text_fontsize,
                                        fontweight=fw,
                                    )

    # Reset yvec
    yvec = yvec_orig


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


def add_cyclic(field, lons):
    """
    | A wrapper for `cartopy_util.add_cyclic_point(field, lons)`.
    |
    | This is needed for the case of when the longitudes are not evenly spaced
    | due to numpy rounding which causes an error from the cartopy wrapping
    | routine. In this case the longitudes are promoted to 64 bit and then
    | rounded to an appropriate number of decimal places before passing to
    | the cartopy add_cyclic routine.
    """

    try:
        field, lons = cartopy_util.add_cyclic_point(field, lons)
    except Exception:
        ndecs_max = max_ndecs_data(lons)
        lons = np.float64(lons).round(ndecs_max)
        field, lons = cartopy_util.add_cyclic_point(field, lons)

    return field, lons


def irregular_window(field, lons, lats):
    """TODO DOCS."""

    field_irregular = deepcopy(field)
    lons_irregular = deepcopy(lons)
    lats_irregular = deepcopy(lats)

    # Fix longitudes to be -180 to 180
    # lons_irregular = (
    #     (lons_irregular + plotvars.lonmin) % 360) + plotvars.lonmin

    # Test data to get appropiate longitude offset to perform remapping
    found_lon = False
    for ilon in [-360, 0, 360]:
        lons_test = lons_irregular + ilon
        if np.min(lons_test) <= plotvars.lonmin:
            found_lon = True
            lons_offset = ilon

    if found_lon:
        lons_irregular = lons_irregular + lons_offset
        pts = np.where(lons_irregular < plotvars.lonmin)
        lons_irregular[pts] = lons_irregular[pts] + 360.0
    else:
        errstr = (
            "\n\ncf-plot error - cannot determine grid offset in "
            "add_cyclic_irregular\n\n"
        )
        raise Warning(errstr)

    field_wrap = deepcopy(field_irregular)
    lons_wrap = deepcopy(lons_irregular)
    lats_wrap = deepcopy(lats_irregular)
    delta = 120.0

    pts_left = np.where(lons_wrap >= plotvars.lonmin + 360 - delta)
    lons_left = lons_wrap[pts_left] - 360.0
    lats_left = lats_wrap[pts_left]
    field_left = field_wrap[pts_left]

    field_wrap = np.concatenate([field_wrap, field_left])
    lons_wrap = np.concatenate([lons_wrap, lons_left])
    lats_wrap = np.concatenate([lats_wrap, lats_left])

    # Make a line of interpolated data on left hand side of plot and insert
    # this into the data on both the left and the right before contouring
    lons_new = np.zeros(181) + plotvars.lonmin
    lats_new = np.arange(181) - 90
    field_new = griddata(
        (lons_wrap, lats_wrap),
        field_wrap,
        (lons_new, lats_new),
        method="linear",
    )

    # Remove any non finite points in the interpolated data
    pts = np.where(np.isfinite(field_new))
    field_new = field_new[pts]
    lons_new = lons_new[pts]
    lats_new = lats_new[pts]

    # Add the interpolated data to the left
    field_irregular = np.concatenate([field_irregular, field_new])
    lons_irregular = np.concatenate([lons_irregular, lons_new])
    lats_irregular = np.concatenate([lats_irregular, lats_new])

    # Add to the right if a full globe is being plotted
    # The 359.99 here is needed or Cartopy will map 360 back to 0

    if plotvars.lonmax - plotvars.lonmin == 360:
        field_irregular = np.concatenate([field_irregular, field_new])
        lons_irregular = np.concatenate([lons_irregular, lons_new + 359.95])
        lats_irregular = np.concatenate([lats_irregular, lats_new])
    else:
        lons_new2 = np.zeros(181) + plotvars.lonmax
        lats_new2 = np.arange(181) - 90
        field_new2 = griddata(
            (lons_wrap, lats_wrap),
            field_wrap,
            (lons_new2, lats_new2),
            method="linear",
        )

        # Remove any non finite points in the interpolated data
        pts = np.where(np.isfinite(field_new2))
        field_new2 = field_new2[pts]
        lons_new2 = lons_new2[pts]
        lats_new2 = lats_new2[pts]

        # Add the interpolated data to the right
        field_irregular = np.concatenate([field_irregular, field_new2])
        lons_irregular = np.concatenate([lons_irregular, lons_new2])
        lats_irregular = np.concatenate([lats_irregular, lats_new2])

    # Finally remove any point off to the right of plotvars.lonmax
    pts = np.where(lons_irregular <= plotvars.lonmax)
    if np.size(pts) > 0:
        field_irregular = field_irregular[pts]
        lons_irregular = lons_irregular[pts]
        lats_irregular = lats_irregular[pts]

    return field_irregular, lons_irregular, lats_irregular


def max_ndecs_data(data):
    """TODO DOCS."""

    ndecs_max = 1
    data_ndecs = np.zeros(len(data))
    for i in np.arange(len(data)):
        data_ndecs[i] = len(str(data[i]).split(".")[1])

    if max(data_ndecs) >= ndecs_max:
        # Reset large decimal vales to zero
        if min(data_ndecs) < 10:
            pts = np.where(data_ndecs >= 10)
            data_ndecs[pts] = 0
            ndecs_max = int(max(data_ndecs))

    return ndecs_max


def fix_floats(data):
    """
    Fixes numpy rounding issues where 0.4 becomes 0.399999999999999999999.
    """

    # Return unchecked if any values have an e in them, for example 7.85e-8
    has_e = False
    for val in data:
        if "e" in str(val):
            has_e = True
    if has_e:
        return data

    data_ndecs = np.zeros(len(data))
    for i in np.arange(len(data)):
        data_ndecs[i] = len(str(float(data[i])).split(".")[1])

    if max(data_ndecs) >= 10:
        # Reset large decimal vales to zero
        if min(data_ndecs) < 10:
            pts = np.where(data_ndecs >= 10)
            data_ndecs[pts] = 0
            ndecs_max = int(max(data_ndecs))
            # Reset to new ndecs_max decimal places
            for i in np.arange(len(data)):
                data[i] = round(data[i], ndecs_max)
        else:
            # fix to two or more decimal places
            nd = 2
            data_range = 0.0
            data_temp = data

            while data_range == 0.0:
                data_temp = deepcopy(data)

                for i in np.arange(len(data_temp)):
                    data_temp[i] = round(data_temp[i], nd)

                data_range = np.max(data_temp) - np.min(data_temp)
                nd = nd + 1

            data = data_temp

    return data


def calculate_levels(field=None, level_spacing=None, verbose=None):
    """TODO DOCS."""

    dmin = np.nanmin(field)
    dmax = np.nanmax(field)

    tight = True

    field2 = deepcopy(field)

    if plotvars.user_levs == 1:
        # User defined
        if verbose:
            print("cfp.calculate_levels - using user defined contour levels")
        clevs = plotvars.levels
        mult = 0
        fmult = 1
    else:
        if plotvars.levels_step is None:
            # Automatic levels
            mult = 0
            fmult = 1
            if verbose:
                print(
                    "cfp.calculate_levels - generating automatic contour "
                    "levels"
                )

            if level_spacing == "outlier" or level_spacing == "inspect":
                hist = np.histogram(field, 100)[0]
                pts = np.size(field)
                rate = 0.01

                if sum(hist[1:-2]) == 0:
                    if hist[0] / hist[-1] < rate:
                        pts = np.where(field == dmin)
                        field2[pts] = dmax
                        dmin = np.nanmin(field2)

                    if hist[-1] / hist[0] < rate:
                        pts = np.where(field == dmax)
                        field2[pts] = dmin
                        dmax = np.nanmax(field2)

                clevs, mult = _gvals(dmin=dmin, dmax=dmax)
                fmult = 10**-mult
                tight = False

            if level_spacing == "linear":
                if isinstance(
                    np.ma.min(dmin), np.ma.core.MaskedConstant
                ) or isinstance(np.ma.min(dmax), np.ma.core.MaskedConstant):
                    errstr = (
                        "cf-plot calculate_levels error - data is entirely "
                        "masked\n"
                        "setting levels to 0 and 0.1 to produce a plot"
                    )
                    print(errstr)
                    dmin = 0.0
                    dmax = 0.1

                clevs, mult = _gvals(dmin=dmin, dmax=dmax)
                fmult = 10**-mult
                tight = False

            if level_spacing == "log" or level_spacing == "loglike":

                if dmin < 0.0 and dmax < 0.0:
                    dmin1 = abs(dmax)
                    dmax1 = abs(dmin)

                if dmin > 0.0 and dmax > 0.0:
                    dmin1 = abs(dmin)
                    dmax1 = abs(dmax)

                if dmin <= 0.0 and dmax >= 0.0:
                    dmax1 = max(abs(dmin), dmax)
                    pts = np.where(field < 0.0)
                    close_below = np.max(field[pts])
                    pts = np.where(field > 0.0)
                    close_above = np.min(field[pts])
                    dmin1 = min(abs(close_below), close_above)

                # Generate levels
                if level_spacing == "log":
                    clevs = []
                    for i in np.arange(31):
                        val = 10 ** (i - 30.0)
                        clevs.append("{:.0e}".format(val))

                if level_spacing == "loglike":
                    clevs = []
                    for i in np.arange(61):
                        val = 10 ** (i - 30.0)
                        clevs.append("{:.0e}".format(val))
                        clevs.append("{:.0e}".format(val * 2))
                        clevs.append("{:.0e}".format(val * 5))
                    clevs = np.float64(clevs)

                # Remove out of range levels
                clevs = np.float64(clevs)
                pts = np.where(
                    np.logical_and(clevs >= abs(dmin1), clevs <= abs(dmax1))
                )
                clevs = clevs[pts]

                if dmin < 0.0 and dmax < 0.0:
                    clevs = -1.0 * clevs[::-1]

                if dmin <= 0.0 and dmax >= 0.0:
                    clevs = np.concatenate([-1.0 * clevs[::-1], [0.0], clevs])

    # Use step to generate the levels
    if plotvars.levels_step is not None:
        if verbose:
            print(
                "calculate_levels - using specified step to generate "
                "contour levels"
            )

        step = plotvars.levels_step

        if isinstance(step, int):
            dmin = int(dmin)
            dmax = int(dmax)

        fmult = 1
        mult = 0
        clevs = []
        if dmin < 0:
            clevs = (np.arange(-1 * dmin / step + 1) * -step)[::-1]
        if dmax > 0:
            if np.size(clevs) > 0:
                clevs = np.concatenate(
                    (clevs[:-1], np.arange(dmax / step + 1) * step)
                )
            else:
                clevs = np.arange(dmax / step + 1) * step
        if isinstance(step, int):
            clevs = clevs.astype(int)

    # Remove any out of data range values
    if tight:
        pts = np.where(np.logical_and(clevs >= dmin, clevs <= dmax))
        clevs = clevs[pts]

    # Add an extra contour level if less than two levels are present
    if np.size(clevs) < 2:
        clevs.append(clevs[0] + 0.001)

    # Test for large numer of decimal places and fix if necessary
    if plotvars.levels is None:
        if isinstance(clevs[0], float):
            clevs = fix_floats(clevs)

    return (clevs, mult, fmult)


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


def generate_titles(f=None):
    """Generate a set of title dims to put at the top of plots."""

    mycoords = find_dim_names(f)
    # TODO SLB, see 'well_formed' dead code below in case this is important.
    # For now, us 'noqa' to prevent PEP8 F841 being raised due to lack of use.
    well_formed = check_well_formed(f)  # noqa: F841

    title_dims = ""
    if isinstance(f, cf.Field):
        for idim in np.arange(len(mycoords)):
            mycoord = mycoords[idim]
            if mycoord == "Z":
                mycoord = find_z(f)

            title, units = cf_var_name_titles(f, mycoord)
            if not f.coord(mycoord).T:
                values = f.construct(mycoord).array
                if len(values) > 1:
                    value = ""
                else:
                    value = str(values)
                title_dims += f"{mycoord}: {title} {value} {units}\n"

            else:
                values = f.construct(mycoord).dtarray

                if len(values) > 1:
                    value = ""
                else:
                    value = str(cf.Data(values).datetime_as_string)
                title_dims += f"{mycoord}: {title} {value}\n"

        if len(f.cell_methods()) > 0:
            title_dims += "cell_methods: "
            i = 0

            for method in f.cell_methods():
                if len(f.cell_methods()[method].get_axes()) > 0:
                    axis = f.cell_methods()[method].get_axes()[0]
                    try:
                        # Change domainaxis0 etc to an axis
                        myid = f.constructs.domain_axis_identity(axis)
                    except ValueError:
                        myid = axis

                    value = ""
                    if f.cell_methods()[method].has_method():
                        value = f.cell_methods()[method].get_method()

                    qualifiers = f.cell_methods()[method].qualifiers()
                    qualifier_text = ""
                    if len(qualifiers) > 0:
                        qualifier_text = str(qualifiers)

                    if i > 0:
                        title_dims += ", "

                    title_dims += f"{myid}: {value} {qualifier_text}"

                    i += 1

    return title_dims


def check_well_formed(field):
    """
    Check the coordinates are all of the form X, Y, Z, T.

    returns boolean
    """

    coords = list(field.coords())
    mycoords = deepcopy(coords)

    for i in np.arange(len(coords)):
        c = field.coord(coords[i])
        if c.X:
            mycoords[i] = "X"
        if c.Y:
            mycoords[i] = "Y"
        if c.Z:
            mycoords[i] = "Z"
        if c.T:
            mycoords[i] = "T"

    # Check if the coordtinates are all of the form X, Y, Z, T
    well_formed = True
    dimension_coords = [
        "dimensioncoordinate0",
        "dimensioncoordinate1",
        "dimensioncoordinate2",
        "dimensioncoordinate3",
    ]
    for i in np.arange(4):
        if dimension_coords[i] in mycoords:
            well_formed = False

    return well_formed


def orca_check(x, verbose=False):
    """Check input data to see if it is an orca ocean grid.

    We look for a single discontinuity in longitude where the data changes by
    greater that 120 degrees.
    """

    lons = deepcopy(x)

    # Only check for longitude range > 350 degrees
    if np.max(lons) - np.min(lons) < 350:
        return False

    nvpts = np.shape(lons)[0]

    lons_lower = lons[int(nvpts / 4), :]
    discont_lower_idx = np.where(abs(np.diff(lons_lower)) > 120)

    lons_mid = lons[int(nvpts / 2), :]
    discont_mid_idx = np.where(abs(np.diff(lons_mid)) > 120)

    lons_upper = lons[int(nvpts * 3 / 4), :]
    discont_upper_idx = np.where(abs(np.diff(lons_upper)) > 120)

    # Check for one discontinuity
    retval = False
    if (
        np.size(discont_lower_idx) == 1
        and np.size(discont_mid_idx) == 1
        and np.size(discont_upper_idx) == 1
    ):
        if verbose:
            print("orca_check - one discontinutity")
            print(discont_lower_idx, discont_mid_idx, discont_upper_idx)
        v1 = float(discont_lower_idx[0])
        v2 = float(discont_mid_idx[0])
        v3 = float(discont_upper_idx[0])

        spread = np.max(np.abs(np.diff([v1, v2, v3])))

        if verbose:
            print(
                f"orca_check spread is {np.max(np.abs(np.diff([v1, v2, v3])))}"
            )

        # Check for discontinuity spead of less than 20 places
        if spread <= 20:
            retval = True

    return retval


def map_grid():
    """Plot a grid on a map."""

    lons = (
        np.arange((360 / plotvars.grid_x_spacing) + 1)
        * plotvars.grid_x_spacing
    )
    lons = np.concatenate([lons - 360, lons])
    lats = (
        np.arange((180 / plotvars.grid_y_spacing) + 1)
        * plotvars.grid_y_spacing
        - 90
    )

    plotvars.mymap.gridlines(
        color=plotvars.grid_colour,
        linewidth=plotvars.grid_thickness,
        linestyle=plotvars.grid_linestyle,
        xlocs=lons,
        ylocs=lats,
        zorder=plotvars.grid_zorder,
    )


# ======================================================
# MISC. - TODO move these out of functional library code
# ======================================================


def _process_color_scales():
    """
    | Process colour scales to generate images of them for the web
    | documentation and the rst code for inclusion in the
    | colour_scale.rst file.
    |
    |
    | No inputs
    | This is an internal routine and not used by the user
    |
    :Returns:
     None
    """

    # Define scale categories
    uniform = ["viridis", "magma", "inferno", "plasma", "parula", "gray"]

    ncl_large = [
        "amwg256",
        "BkBlAqGrYeOrReViWh200",
        "BlAqGrYeOrRe",
        "BlAqGrYeOrReVi200",
        "BlGrYeOrReVi200",
        "BlRe",
        "BlueRed",
        "BlueRedGray",
        "BlueWhiteOrangeRed",
        "BlueYellowRed",
        "BlWhRe",
        "cmp_b2r",
        "cmp_haxby",
        "detail",
        "extrema",
        "GrayWhiteGray",
        "GreenYellow",
        "helix",
        "helix1",
        "hotres",
        "matlab_hot",
        "matlab_hsv",
        "matlab_jet",
        "matlab_lines",
        "ncl_default",
        "ncview_default",
        "OceanLakeLandSnow",
        "rainbow",
        "rainbow_white_gray",
        "rainbow_white",
        "rainbow_gray",
        "tbr_240_300",
        "tbr_stdev_0_30",
        "tbr_var_0_500",
        "tbrAvg1",
        "tbrStd1",
        "tbrVar1",
        "thelix",
        "ViBlGrWhYeOrRe",
        "wh_bl_gr_ye_re",
        "WhBlGrYeRe",
        "WhBlReWh",
        "WhiteBlue",
        "WhiteBlueGreenYellowRed",
        "WhiteGreen",
        "WhiteYellowOrangeRed",
        "WhViBlGrYeOrRe",
        "WhViBlGrYeOrReWh",
        "wxpEnIR",
        "3gauss",
        "3saw",
        "BrBG",
    ]

    ncl_meteoswiss = [
        "hotcold_18lev",
        "hotcolr_19lev",
        "mch_default",
        "perc2_9lev",
        "percent_11lev",
        "precip2_15lev",
        "precip2_17lev",
        "precip3_16lev",
        "precip4_11lev",
        "precip4_diff_19lev",
        "precip_11lev",
        "precip_diff_12lev",
        "precip_diff_1lev",
        "rh_19lev",
        "spread_15lev",
    ]

    ncl_color_blindness = [
        "StepSeq25",
        "posneg_2",
        "posneg_1",
        "BlueDarkOrange18",
        "BlueDarkRed18",
        "GreenMagenta16",
        "BlueGreen14",
        "BrownBlue12",
        "Cat12",
    ]

    ncl_small = [
        "amwg",
        "amwg_blueyellowred",
        "BlueDarkRed18",
        "BlueDarkOrange18",
        "BlueGreen14",
        "BrownBlue12",
        "Cat12",
        "cmp_flux",
        "cosam12",
        "cosam",
        "GHRSST_anomaly",
        "GreenMagenta16",
        "hotcold_18lev",
        "hotcolr_19lev",
        "mch_default",
        "nrl_sirkes",
        "nrl_sirkes_nowhite",
        "perc2_9lev",
        "percent_11lev",
        "posneg_2",
        "prcp_1",
        "prcp_2",
        "prcp_3",
        "precip_11lev",
        "precip_diff_12lev",
        "precip_diff_1lev",
        "precip2_15lev",
        "precip2_17lev",
        "precip3_16lev",
        "precip4_11lev",
        "precip4_diff_19lev",
        "radar",
        "radar_1",
        "rh_19lev",
        "seaice_1",
        "seaice_2",
        "so4_21",
        "spread_15lev",
        "StepSeq25",
        "sunshine_9lev",
        "sunshine_diff_12lev",
        "temp_19lev",
        "temp_diff_18lev",
        "temp_diff_1lev",
        "topo_15lev",
        "wgne15",
        "wind_17lev",
    ]

    orography = [
        "os250kmetres",
        "wiki_1_0_2",
        "wiki_1_0_3",
        "wiki_2_0",
        "wiki_2_0_reduced",
        "arctic",
    ]

    idl_guide = []
    for i in np.arange(1, 45):
        idl_guide.append("scale" + str(i))

    # TODO SLB improve string formatting below and consolidate
    for category in [
        "uniform",
        "ncl_meteoswiss",
        "ncl_small",
        "ncl_large",
        "ncl_color_blindness",
        "orography",
        "idl_guide",
    ]:
        if category == "uniform":
            scales = uniform
            div = "================== ====="
            chars = 10
            title = (
                "Perceptually uniform colour maps for use with continuous "
                "data"
            )
            print(title)
            print("----------------------------------------------")
            print("")
            print(div)
            print("Name               Scale")
            print(div)

        if category == "ncl_meteoswiss":
            scales = ncl_meteoswiss
            div = "================== ====="
            chars = 19
            print("NCAR Command Language - MeteoSwiss colour maps")
            print("----------------------------------------------")
            print("")
            print(div)
            print("Name               Scale")
            print(div)
        if category == "ncl_small":
            scales = ncl_small
            div = "=================== ====="
            chars = 20
            print("NCAR Command Language - small color maps (<50 colours)")
            print("------------------------------------------------------")
            print("")
            print(div)
            print("Name                Scale")
            print(div)
        if category == "ncl_large":
            scales = ncl_large
            div = "======================= ====="
            chars = 24
            print("NCAR Command Language - large colour maps (>50 colours)")
            print("-------------------------------------------------------")
            print("")
            print(div)
            print("Name                    Scale")
            print(div)
        if category == "ncl_color_blindness":
            scales = ncl_color_blindness
            div = "================ ====="
            chars = 17
            title = (
                "NCAR Command Language - Enhanced to help with colour "
                "blindness"
            )
            print(title)
            title = (
                "-----------------------------------------------------"
                "---------"
            )
            print(title)
            print("")
            print(div)
            print("Name             Scale")
            print(div)
            chars = 17
        if category == "orography":
            scales = orography
            div = "================ ====="
            chars = 17
            print("Orography/bathymetry colour scales")
            print("----------------------------------")
            print("")
            print(div)
            print("Name             Scale")
            print(div)
            chars = 17
        if category == "idl_guide":
            scales = idl_guide
            div = "======= ====="
            chars = 8
            print("IDL guide scales")
            print("----------------")
            print("")
            print(div)
            print("Name    Scale")
            print(div)
            chars = 8

        for scale in scales:
            # Make image of scale
            fig = plot.figure(figsize=(8, 0.5))
            ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.2])
            cscale(scale)
            cmap = matplotlib.colors.ListedColormap(plotvars.cs)
            cb1 = matplotlib.colorbar.ColorbarBase(
                ax1, cmap=cmap, orientation="horizontal", ticks=None
            )
            cb1.set_ticks([0.0, 1.0])
            cb1.set_ticklabels(["", ""])
            # TODO SLB, update path below to non-specific one
            file = (
                "/home/andy/cf-docs/cfplot_sphinx/images/"
                f"colour_scales/{scale}.png"
            )
            plot.savefig(file)
            plot.close()

            # Use convert to trim the png file to remove white space
            subprocess.call(["convert", "-trim", file, file])

            name_pad = scale
            while len(name_pad) < chars:
                name_pad = name_pad + " "
            fn = f"{name_pad}.. image:: images/colour_scales/{scale}.png"
            print(fn)

        print(div)
        print("")
        print("")
