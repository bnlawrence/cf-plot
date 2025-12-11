import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib
import numpy as np

import cf

from .calculate import (
    calculate_levels,
)
from .colour import (
    cbar,
)
from .graphic import (
    gclose,
    gopen,
)
from .mapping import (
    _cscale_get_map,
    _mapaxis,
    _map_title,
    _plot_map_axes,
    _set_map,
    axes_plot,
)
from .parameters import (
    cscale,
    cscale1,
    global_blockfill,
    global_fill,
    global_lines,
    gset,
    mapset,
    plotvars,
)
from .utils import (
    _cf_data_assign,
    _timeaxis,
    add_cyclic,
    find_pos_in_array,
    ndecs,
)
from .validate import (
    check_well_formed,
)


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
