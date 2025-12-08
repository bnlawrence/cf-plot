from copy import deepcopy

import cartopy.crs as ccrs
import cf
import numpy as np
import matplotlib.pyplot as plot

from ..parameters import (
    plotvars,
)


def axes_plot(
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
):
    """
    | A system function to specify axes plotting parameters.
    | Use xticks, yticks to specify the tick positions and xticklabels,
    | yticklabels to specify the associated labels.
    |
    | xticks=xticks - values for x ticks
    | xticklabels=xticklabels - labels for x tick marks
    | yticks=yticks - values for y ticks
    | yticklabels=yticklabels - labels for y tick marks
    | xlabel=xlabel - label for the x-axis
    | ylabel=ylabel - label for the y-axis
    | title=None - set title
    |

    :Returns:
     None
    """

    if plotvars.title is not None:
        title = plotvars.title
    title_fontsize = plotvars.title_fontsize
    text_fontsize = plotvars.text_fontsize
    axis_label_fontsize = plotvars.axis_label_fontsize
    if title_fontsize is None:
        title_fontsize = 15
    if text_fontsize is None:
        text_fontsize = 11
    if axis_label_fontsize is None:
        axis_label_fontsize = 11
    axis_label_fontweight = plotvars.axis_label_fontweight
    title_fontweight = plotvars.title_fontweight

    if (
        plotvars.plot_type == 1 or plotvars.plot_type == 6
    ) and plotvars.proj == "cyl":
        plot = plotvars.mymap
        lon_mid = plotvars.lonmin + (plotvars.lonmax - plotvars.lonmin) / 2.0
        plotargs = {"crs": ccrs.PlateCarree()}
    else:
        plot = plotvars.plot
        plotargs = {}

    if xlabel is not None:
        plotvars.plot.set_xlabel(
            xlabel,
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
        )
    if ylabel is not None:
        plotvars.plot.set_ylabel(
            ylabel,
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
        )

    xticklen = (plotvars.lonmax - plotvars.lonmin) * 0.007
    yticklen = (plotvars.latmax - plotvars.latmin) * 0.014

    # set the plot
    if plotvars.plot_type == 1 or plotvars.plot_type == 6:
        this_plot = plotvars.mymap
    else:
        this_plot = plotvars.plot

    if plotvars.plot_type == 6 and (
        plotvars.proj == "rotated" or plotvars.proj == "UKCP"
    ):
        this_plot = plotvars.plot

    # get the plot bounds
    left, bottom, width, height = this_plot.get_position().bounds

    lonrange = plotvars.lonmax - plotvars.lonmin
    lon_mid = plotvars.lonmin + (plotvars.lonmax - plotvars.lonmin) / 2.0

    # Set the ticks and tick labels
    if xticks is not None:
        # fudge min and max longitude tick positions or the labels wrap
        xticks_new = xticks
        if lonrange >= 360:
            xticks_new[0] = xticks_new[0] + 0.01
            xticks_new[-1] = xticks_new[-1] - 0.01

        plot.set_xticks(xticks_new, **plotargs)
        plot.set_xticklabels(
            xticklabels,
            rotation=plotvars.xtick_label_rotation,
            horizontalalignment=plotvars.xtick_label_align,
        )

        # Plot a corresponding tick on the top of the plot - cartopy feature?
        proj = ccrs.PlateCarree(central_longitude=lon_mid)
        if plotvars.plot_type == 1:
            for xval in xticks_new:
                xpt, ypt = proj.transform_point(
                    xval, plotvars.latmax, ccrs.PlateCarree()
                )
                ypt2 = ypt + yticklen
                plot.plot(
                    [xpt, xpt],
                    [ypt, ypt2],
                    color="k",
                    linewidth=0.8,
                    clip_on=False,
                )

    if yticks is not None:
        plot.set_yticks(yticks, **plotargs)
        plot.set_yticklabels(
            yticklabels,
            rotation=plotvars.ytick_label_rotation,
            horizontalalignment=plotvars.ytick_label_align,
        )

        # Plot a corresponding tick on the right of the plot - cartopy feature?
        if plotvars.plot_type == 1:
            proj = ccrs.PlateCarree(central_longitude=lon_mid)
            for ytick in yticks:
                xpt, ypt = proj.transform_point(
                    plotvars.lonmax - 0.001, ytick, ccrs.PlateCarree()
                )
                xpt2 = xpt + xticklen
                plot.plot(
                    [xpt, xpt2],
                    [ypt, ypt],
                    color="k",
                    linewidth=0.8,
                    clip_on=False,
                )

    # Set font size and weight
    for label in plot.xaxis.get_ticklabels():
        label.set_fontsize(axis_label_fontsize)
        label.set_fontweight(axis_label_fontweight)
    for label in plot.yaxis.get_ticklabels():
        label.set_fontsize(axis_label_fontsize)
        label.set_fontweight(axis_label_fontweight)

    # Title
    if title is not None:
        plot.set_title(
            title, y=1.03, fontsize=title_fontsize, fontweight=title_fontweight
        )


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


def find_dim_names(field):
    """Find the field dimension coordinate names.
    Ignores auxiliary coordinates (for now).

    returns:
    coordinates in the order [T, X, Y, Z]
    """

    # Get the field domain axes
    daxes = list(field.get_data_axes())

    # Get the field coordinates
    dcoords = list(field.coords())

    # Calculate the number of coordinates of type X, Y, Z and T
    nx = 0
    ny = 0
    nz = 0
    nt = 0
    for i in np.arange(len(dcoords)):
        if field.coord(dcoords[i]).X:
            nx += 1
        if field.coord(dcoords[i]).Y:
            ny += 1
        if field.coord(dcoords[i]).Z:
            nz += 1
        if field.coord(dcoords[i]).T:
            nt += 1

    # New test
    remove_aux = False

    # Strip out any auxiliary coordinates if the field is not a
    # trajectory field
    if remove_aux:
        for i in np.arange(len(dcoords)):
            if dcoords[i][:-1] == "auxiliarycoordinate":
                dcoords[i] = "aux"
        dcoords = list(filter(("aux").__ne__, dcoords))

    # Convert these into corresponding dimension coordinates
    if remove_aux:
        coords = []
        for i in np.arange(len(daxes)):
            val = daxes[i]
            coord = None
            for j in np.arange(len(dcoords)):
                if val == field.get_data_axes(dcoords[j])[0]:
                    coord = dcoords[j]

            if coord is not None:
                coords.append(coord)
            else:
                errstr = (
                    "find_data_names error  - cannot find a coordinate for "
                    f"{val}\nin the data\n"
                )
                raise Warning(errstr)
    else:
        coords = dcoords

    # Make a copy of coords in mycoords
    mycoords = deepcopy(coords)

    # Convert to X, Y, Z, T if coordinate is one of these
    # If the number of coordinates of this type is greater than 1 then don't
    # do this as f.coord('Z') gives an
    # error as there is more than one coordinate to return
    for i in np.arange(len(daxes)):
        if len(coords) - 1 < i:
            break
        if field.coord(coords[i]).X:
            if nx == 1:
                mycoords[i] = "X"
        if field.coord(coords[i]).Y:
            if ny == 1:
                mycoords[i] = "Y"
        if field.coord(coords[i]).Z:
            if nz == 1:
                mycoords[i] = "Z"
        if field.coord(coords[i]).T:
            if nt == 1:
                mycoords[i] = "T"

    # Return the reverse of the coordinates so that they are in the
    # order [X, Y, Z, T]
    mycoords.reverse()

    return mycoords


def find_z(f):
    """Find the Z coordinate if it exists."""

    # Return if f is undefined
    if f is None:
        return None

    myz = "Z"
    mycoords = find_dim_names(f)
    myz = None
    for mycoord in mycoords:
        if f.coord(mycoord).Z:
            myz = mycoord

    return myz


def _which(program):
    """Check if the ImageMagick display command is available."""

    def is_exe(fpath):
        """TODO DOCS."""
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    def ext_candidates(fpath):
        """TODO DOCS."""
        yield fpath
        for ext in os.environ.get("PATHEXT", "").split(os.pathsep):
            yield fpath + ext

    for path in os.environ["PATH"].split(os.pathsep):
        exe_file = os.path.join(path, program)
        for candidate in ext_candidates(exe_file):
            if is_exe(candidate):
                return candidate

    return None


def _dim_titles(title=None, title2=None, title3=None):
    """
    | An internal routine to draw a set of dimension titles on a plot.
    |
    | title=None - title to put on the plot
    | title2=None - additional title
    | title3=None - additional title
    |
    """

    # Logic for the supplied titles
    # if just title is supplied:
    #     title - contour or line title
    #
    # if both title and title2 are supplied:
    #     title u component of title
    #     title2 v component of the title
    #
    # if title2 and title3 are supplied:
    #     title2 u component of title
    #     title3 v component of title
    #
    # move the plot around if title3 is None

    # Get plot position
    if plotvars.plot_type == 1:
        this_plot = plotvars.mymap
    else:
        this_plot = plotvars.plot

    left, bottom, width, height = this_plot.get_position().bounds

    valign = "bottom"

    # Shift down if a cylindrical projection plot else to the left
    if plotvars.plot_type == 1 and plotvars.proj != "cyl":
        left -= 0.1
        myx = 1.25
        myy = 1.0
        valign = "top"
        if title3 is None:
            myx = 1.05
    elif plotvars.plot_type == 1 and plotvars.proj == "cyl":
        lonrange = plotvars.lonmax - plotvars.lonmin
        latrange = plotvars.latmax - plotvars.latmin
        if (lonrange / latrange) > 1.5:
            myx = 0.0
            myy = 1.02

        if (lonrange / latrange) > 1.2 and (lonrange / latrange) <= 1.5:
            myx = 0.0
            myy = 1.02
            height -= 0.015

        if (lonrange / latrange) <= 1.2:
            left -= 0.1
            # if title2 is not None:
            #    l = l - 0.1
            myx = 1.05
            myy = 1.0
            width -= 0.1
            valign = "top"
    else:
        height -= 0.1
        myx = 0.0
        myy = 1.02

    # Change the plot position if title3 is None
    if title3 is None:
        this_plot.set_position([left, bottom, width, height])

    # Set x and y spacing depending on the label location
    xspacing = 0.3
    yspacing = 0.0
    if myx == 1.05 or myx == 1.25:
        xspacing = 0.0
        yspacing = 0.2

    if title is not None:
        this_plot.text(
            myx,
            myy,
            title,
            va=valign,
            ha="left",
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
            transform=this_plot.transAxes,
        )

    if title2 is not None:
        this_plot.text(
            myx + xspacing,
            myy - yspacing,
            title2,
            va=valign,
            ha="left",
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
            transform=this_plot.transAxes,
        )

    if title3 is not None:
        this_plot.text(
            myx + xspacing * 2,
            myy - yspacing * 2,
            title3,
            va=valign,
            ha="left",
            fontsize=plotvars.axis_label_fontsize,
            fontweight=plotvars.axis_label_fontweight,
            transform=this_plot.transAxes,
        )


def _bfill_ugrid(
    f=None,
    face_lons=None,
    face_lats=None,
    face_connectivity=None,
    clevs=None,
    alpha=None,
    zorder=None,
):
    """
    | Block fill a irregular field with colour rectangles.
    | This is an internal routine and is not generally used by the user.
    |
    | f=None - field
    | face_lons=None - longitude points for face vertices
    | face_lats=None - latitude points for face verticies
    | face_connectivity=None - connectivity for face verticies
    | clevs=None - levels for filling
    | lonlat=False - lonlat data
    | bound=False - x and y are cf data boundaries
    | alpha=alpha - transparency setting 0 to 1
    | zorder=None - plotting order
    |
     :Returns:
       None
    |
    """

    # Colour faces according to value
    # Set faces to white initially
    cols = ["#000000" for x in range(len(face_connectivity))]

    levs = deepcopy(np.array(clevs))

    if plotvars.levels_extend == "min" or plotvars.levels_extend == "both":
        levs = np.concatenate([[-1e20], levs])
    ilevs_max = np.size(levs)
    if plotvars.levels_extend == "max" or plotvars.levels_extend == "both":
        levs = np.concatenate([levs, [1e20]])
    else:
        ilevs_max = ilevs_max - 1

    for ilev in np.arange(ilevs_max):
        lev = levs[ilev]
        col = plotvars.cs[ilev]
        pts = np.where(f.squeeze() >= lev)[0]

        if len(pts) > 0:
            if np.min(pts) >= 0:
                for val in np.arange(np.size(pts)):
                    pt = pts[val]
                    cols[pt] = col

    plotargs = {"transform": ccrs.PlateCarree()}

    coords_all = []

    nfaces = np.shape(face_connectivity)[0]

    coords_all = []
    for iface in np.arange(nfaces):
        lons = face_lons[iface, :]
        lats = face_lats[iface, :]

        # Wrapping in longitude
        if (np.max(lons) - np.min(lons)) > 100:
            if np.max(lons) > 180:
                for j in np.arange(len(lons)):
                    lons[j] = (lons[j] + 180) % 360 - 180
            else:
                for j in np.arange(len(lons)):
                    lons[j] = lons[j] % 360

        nverts = len(lons)

        # Add extra verticies if any of the points are at the north or
        # south pole
        if np.max(lats) == 90 or np.min(lats) == -90:

            geom = sgeom.Polygon(
                [(lons[k], lats[k]) for k in np.arange(nverts)]
            )
            geom_cyl = ccrs.PlateCarree().project_geometry(
                geom, ccrs.Geodetic()
            )

            # New method for shapely 2.0 +
            poly_mapped = sgeom.mapping(geom_cyl.geoms[0])

            coords = list(poly_mapped["coordinates"][0])
        else:
            coords = [(lons[k], lats[k]) for k in np.arange(nverts)]

        coords_all.append(coords)

    plotvars.mymap.add_collection(
        PolyCollection(
            coords_all,
            facecolors=cols,
            edgecolors=None,
            alpha=alpha,
            zorder=zorder,
            **plotargs,
        )
    )


def _mapaxis(min=None, max=None, type=None):
    """
    | Work out a sensible set of longitude and latitude
    | tick marks and labels. This is an internal routine and is not used
    | by the user.

    | min=None - minimum axis value
    | max=None - maximum axis value
    | type=None - 1 = longitude, 2 = latitude

    :Returns:
     longtitude/latitude ticks and longitude/latitude tick labels
    |
    """

    degsym = ""
    if plotvars.degsym:
        degsym = r"$\degree$"
    if type == 1:
        lonmin = min
        lonmax = max
        lonrange = lonmax - lonmin
        lonstep = 60
        if lonrange <= 180:
            lonstep = 30
        if lonrange <= 90:
            lonstep = 10
        if lonrange <= 30:
            lonstep = 5
        if lonrange <= 10:
            lonstep = 2
        if lonrange <= 5:
            lonstep = 1

        lons = np.arange(-720, 720 + lonstep, lonstep)
        lonticks = []
        for lon in lons:
            if lon >= lonmin and lon <= lonmax:
                lonticks.append(lon)

        lonlabels = []
        for lon in lonticks:
            lon2 = np.mod(lon + 180, 360) - 180
            if lon2 < 0 and lon2 > -180:
                if lon != 180:
                    lonlabels.append(str(abs(lon2)) + degsym + "W")

            if lon2 > 0 and lon2 <= 180:
                lonlabels.append(str(lon2) + degsym + "E")
            if lon2 == 0:
                lonlabels.append("0" + degsym)

            if lon == 180 or lon == -180:
                lonlabels.append("180" + degsym)

        return (lonticks, lonlabels)

    if type == 2:
        latmin = min
        latmax = max
        latrange = latmax - latmin
        latstep = 30
        if latrange <= 90:
            latstep = 10
        if latrange <= 30:
            latstep = 5
        if latrange <= 10:
            latstep = 2
        if latrange <= 5:
            latstep = 1

        lats = np.arange(-90, 90 + latstep, latstep)
        latticks = []
        for lat in lats:
            if lat >= latmin and lat <= latmax:
                latticks.append(lat)

        latlabels = []
        for lat in latticks:
            if lat < 0:
                latlabels.append(str(abs(lat)) + degsym + "S")
            if lat > 0:
                latlabels.append(str(lat) + degsym + "N")
            if lat == 0:
                latlabels.append("0" + degsym)

        return (latticks, latlabels)


def _timeaxis(dtimes=None):
    """
    | Work out a sensible set of time labels and tick
    | marks given a time span. This is an internal routine and is not used
    | by the user.

    | dtimes=None - data times as a CF variable

    :Returns:
     time ticks and labels
    |
    """

    time_units = dtimes.Units
    time_ticks = []
    time_labels = []
    axis_label = "Time"

    yearmin = min(dtimes.year.array)
    yearmax = max(dtimes.year.array)
    tmin = min(dtimes.dtarray)
    tmax = max(dtimes.dtarray)
    if hasattr(dtimes, "calendar"):
        calendar = dtimes.calendar
    else:
        calendar = "standard"

    if plotvars.user_gset != 0:
        if isinstance(plotvars.xmin, str):
            t = cf.Data(
                cf.dt(plotvars.xmin), units=time_units, calendar=calendar
            )
            yearmin = int(t.year)
            t = cf.Data(
                cf.dt(plotvars.xmax), units=time_units, calendar=calendar
            )
            yearmax = int(t.year)
            tmin = cf.dt(plotvars.xmin, calendar=calendar)
            tmax = cf.dt(plotvars.xmax, calendar=calendar)
        if isinstance(plotvars.ymin, str):
            t = cf.Data(
                cf.dt(plotvars.ymin), units=time_units, calendar=calendar
            )
            yearmin = int(t.year)
            t = cf.Data(
                cf.dt(plotvars.ymax), units=time_units, calendar=calendar
            )
            yearmax = int(t.year)
            tmin = cf.dt(plotvars.ymin, calendar=calendar)
            tmax = cf.dt(plotvars.ymax, calendar=calendar)

    # Years
    span = yearmax - yearmin
    if span > 4 and span < 3000:
        axis_label = "Time (year)"
        tvals = []
        if span <= 15:
            step = 1
        if span > 15:
            step = 2
        if span > 30:
            step = 5
        if span > 60:
            step = 10
        if span > 160:
            step = 20
        if span > 300:
            step = 50
        if span > 600:
            step = 100
        if span > 1300:
            step = 200

        if plotvars.tspace_year is not None:
            step = plotvars.tspace_year

        years = np.arange(yearmax / step + 2) * step
        tvals = years[np.where((years >= yearmin) & (years <= yearmax))]

        # Catch tvals if not properly defined and use gvals to generate some
        # year tick marks
        if np.size(tvals) < 2:
            tvals = gvals(dmin=yearmin, dmax=yearmax)[0]

        for year in tvals:
            time_ticks.append(
                np.min(
                    cf.Data(
                        cf.dt(f"{int(year)}-01-01 00:00:00"),
                        units=time_units,
                        calendar=calendar,
                    ).array
                )
            )
            time_labels.append(str(int(year)))

    # Months
    if yearmax - yearmin <= 4:
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        # Check number of labels with 1 month steps
        tsteps = 0
        for year in np.arange(yearmax - yearmin + 1) + yearmin:
            for month in np.arange(12):
                mytime = cf.dt(
                    f"{year}-{month + 1}-01 00:00:00",
                    calendar=calendar,
                )
                if mytime >= tmin and mytime <= tmax:
                    tsteps = tsteps + 1

        if tsteps < 17:
            mvals = np.arange(12)
        if tsteps >= 17:
            mvals = np.arange(4) * 3

        for year in np.arange(yearmax - yearmin + 1) + yearmin:
            for month in mvals:
                mytime = cf.dt(
                    f"{year}-{month + 1}-01 00:00:00",
                    calendar=calendar,
                )
                if mytime >= tmin and mytime <= tmax:
                    time_ticks.append(
                        np.min(
                            cf.Data(
                                mytime, units=time_units, calendar=calendar
                            ).array
                        )
                    )
                    time_labels.append(
                        str(months[month]) + " " + str(int(year))
                    )

    # Days and hours
    if np.size(time_ticks) <= 2:
        myday = cf.dt(
            int(tmin.year), int(tmin.month), int(tmin.day), calendar=calendar
        )
        not_found = 0
        hour_counter = 0
        span = 0
        while not_found <= 48:
            mydate = cf.Data(myday, dtimes.Units) + cf.Data(
                hour_counter, "hour"
            )
            if mydate >= tmin and mydate <= tmax:
                span = span + 1
            else:
                not_found = not_found + 1

            hour_counter = hour_counter + 1

        step = 1
        if span > 13:
            step = 1
        if span > 13:
            step = 4
        if span > 25:
            step = 6
        if span > 100:
            step = 12
        if span > 200:
            step = 24
        if span > 400:
            step = 48
        if span > 800:
            step = 96
        if plotvars.tspace_hour is not None:
            step = plotvars.tspace_hour
        if plotvars.tspace_day is not None:
            step = plotvars.tspace_day * 24

        not_found = 0
        hour_counter = 0
        axis_label = "Time (hour)"
        if span >= 24:
            axis_label = "Time"
        time_ticks = []
        time_labels = []

        while not_found <= 48:
            mytime = cf.Data(myday, dtimes.Units) + cf.Data(
                hour_counter, "hour"
            )
            if mytime >= tmin and mytime <= tmax:
                time_ticks.append(np.min(mytime.array))
                label = f"{mytime.year}-{mytime.month}-{mytime.day}"
                if hour_counter / 24 != int(hour_counter / 24):
                    label += f" {mytime.hour}:00:00"
                time_labels.append(label)
            else:
                not_found = not_found + 1

            hour_counter = hour_counter + step

    return (time_ticks, time_labels, axis_label)


def _supscr(text=None):
    """
    | Add superscript text formatting for `**` and `^`.
    | This is an internal routine used in titles and colour bars
    | and not used by the user.
    |
    | text=None - input text

    :Returns:
     Formatted text
    |
    """

    if text is None:
        errstr = "\n _supscr error - _supscr must have text input\n"
        raise Warning(errstr)

    tform = ""

    sup = 0
    for i in text:
        if i == "^":
            sup = 2
        if i == "*":
            sup = sup + 1

        if sup == 0:
            tform = tform + i
        if sup == 1:
            if i not in "*":
                tform = tform + "*" + i
                sup = 0
        if sup == 3:
            if i in "-0123456789":
                tform = tform + i
            else:
                tform = tform + "}$" + i
                sup = 0
        if sup == 2:
            tform = tform + "$^{"
            sup = 3

    if sup == 3:
        tform = tform + "}$"

    tform = tform.replace("m2", "m$^{2}$")
    tform = tform.replace("m3", "m$^{3}$")
    tform = tform.replace("m-2", "m$^{-2}$")
    tform = tform.replace("m-3", "m$^{-3}$")
    tform = tform.replace("s-1", "s$^{-1}$")
    tform = tform.replace("s-2", "s$^{-2}$")

    return tform


def _gvals(dmin=None, dmax=None, mystep=None, mod=True):
    """
    | Work out a sensible set of values between two limits.
    | This is an internal routine used for contour levels and axis
    | labelling and is not generally used by the user.

    | dmin = None - minimum
    | dmax = None - maximum
    | mystep = None - use this step
    | mod = True - modify data to make use of a multipler
    |
    """

    # Copies of inputs as these might be changed
    dmin1 = deepcopy(dmin)
    dmax1 = deepcopy(dmax)

    # Swap values if dmin1 > dmax1
    if dmax1 < dmin1:
        dmin1, dmax1 = dmax1, dmin1

    # Data range
    data_range = dmax1 - dmin1

    # field multiplier
    mult = 0
    vals = None

    # Return some values if dmin1 = dmax1
    if dmin1 == dmax1:
        vals = np.array([dmin1 - 1, dmin1, dmin1 + 1])
        mult = 0
        return vals, mult

    # Modify if requested or if out of range 0.001 to 2000000
    if data_range < 0.001:
        while dmax1 <= 3:
            dmin1 = dmin1 * 10.0
            dmax1 = dmax1 * 10.0
            data_range = dmax1 - dmin1
            mult = mult - 1

    if data_range > 2000000:
        while dmax1 > 10:
            dmin1 = dmin1 / 10.0
            dmax1 = dmax1 / 10.0
            data_range = dmax1 - dmin1
            mult = mult + 1

    if data_range >= 0.001 and data_range <= 2000000:

        # Calculate an appropriate step
        step = None
        test_steps = [
            0.0001,
            0.0002,
            0.0005,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            20000,
            50000,
            100000,
        ]

        if mystep is not None:
            step = mystep
        else:
            for val in test_steps:
                nvals = data_range / val

                if val < 1:
                    if nvals > 8:
                        step = val
                else:
                    if nvals > 11:
                        step = val

        # Return an error if no step found
        if step is None:
            errstr = "\n\n cfp._gvals - no valid step values found \n\n"
            errstr += "cfp._gvals(" + str(dmin1) + "," + str(dmax1) + ")\n\n"
            raise Warning(errstr)

        # values  < 0.0
        vals = None
        vals1 = None
        if dmin1 < 0.0:
            vals1 = (np.arange(-dmin1 / step) * -step)[::-1] - step

        # values  >= 0.0
        vals2 = None
        if dmax1 >= 0.0:
            vals2 = np.arange(dmax1 / step + 1) * step

        if vals1 is not None and vals2 is None:
            vals = vals1
        if vals2 is not None and vals1 is None:
            vals = vals2
        if vals1 is not None and vals2 is not None:
            vals = np.concatenate((vals1, vals2))

        # Round off decimal numbers so that
        # (np.arange(4) * -0.1)[3] = -0.30000000000000004 gives -0.3
        # as expected
        if step < 1:
            vals = vals.round(6)

        # Change values to integers for values >= 1
        if step >= 1:
            vals = vals.astype(int)

        pts = np.where(np.logical_and(vals >= dmin1, vals <= dmax1))
        if np.min(pts) > -1:
            vals = vals[pts]

        if mod is False:
            vals = vals * 10**mult
            mult = 0

    # Catch if no values have been defined
    if vals is None:
        vals = np.array([dmin, dmax])

    return (vals, mult)


def _cf_data_assign(
    f=None, colorbar_title=None, verbose=None, rotated_vect=False
):
    """
    | Check cf input data is okay and return data for contour plot.
    | This is an internal routine not used by the user.
    |
    | f=None - input cf field
    | colorbar_title=None - input colour bar title
    | rotated vect=False - return 1D x and y for rotated plot vectors
    | verbose=None - set to 1 to get a verbose idea of what the
    |          _cf_data_assign is doing

    :Returns:
     | f - data for contouring
     | x - x coordinates of data (optional)
     | y - y coordinates of data (optional)
     | ptype - plot type
     | colorbar_title - colour bar title
     | xlabel - x label for plot
     | ylabel - y label for plot
     |
    """

    # Check input data has the correct number of dimensions
    # Take into account rotated pole fields having extra dimensions
    ndim = len(f.domain_axes().filter_by_size(cf.gt(1)))
    if (
        f.ref("grid_mapping_name:rotated_latitude_longitude", default=False)
        is False
    ):
        if ndim > 2 or ndim < 1:
            print("")
            if ndim > 2:
                errstr = "_cf_data_assign error - data has too many dimensions"
            if ndim < 1:
                errstr = "_cf_data_assign error - data has too few dimensions"
            errstr += "\n cf-plot requires one or two dimensional data\n"
            for mydim in list(f.dimension_coordinates()):
                sn = getattr(f.construct(mydim), "standard_name", False)
                ln = getattr(f.construct(mydim), "long_name", False)
                if sn:
                    errstr += f"{mydim},{sn},{f.construct(mydim).size}\n"
                else:
                    if ln:
                        errstr += f"{mydim},{ln},{f.construct(mydim).size}\n"
            raise Warning(errstr)

    # Set up data arrays and variables
    lons = None
    lats = None
    height = None
    time = None
    xlabel = ""
    ylabel = ""
    has_lons = False
    has_lats = False
    has_height = False
    has_time = False
    xpole = None
    ypole = None
    ptype = None
    field = None
    x = None
    y = None

    # Check for multiple Z coordinates
    myz = find_z(f)

    # Extract coordinate data if a matching CF standard_name or axis is found
    for mycoord in f.coords():
        c = f.coord(mycoord)
        if c.X:
            lons = np.squeeze(f.construct(mycoord).array)
            if verbose:
                print("lons -", lons)
            if np.size(lons) > 1:
                has_lons = True

        if c.Y:
            lats = np.squeeze(f.construct(mycoord).array)
            if verbose:
                print("lats -", lats)
            if np.size(lats) > 1:
                has_lats = True

        if c.Z:
            height = np.squeeze(f.construct(mycoord).array)
            if verbose:
                print("height -", height)
            if np.size(height) > 1:
                has_height = True

        if c.T:
            time = np.squeeze(f.construct(mycoord).array)
            if verbose:
                print("time -", time)
            if np.size(time) > 1:
                has_time = True

    # assign field data
    field = np.squeeze(f.array)

    # Change Boolean data to integer
    if str(f.dtype) == "bool":
        warnstr = (
            "\n\n\n Warning - boolean data found - converting to "
            "integers\n\n\n"
        )
        print(warnstr)
        g = deepcopy(f)
        g.dtype = int
        field = np.squeeze(g.array)

    # Check what plot type is required.
    # 0=simple contour plot, 1=map plot, 2=latitude-height plot,
    # 3=longitude-time plot, 4=latitude-time plot.
    if has_lons and has_lats:
        ptype = 1
        x = lons
        y = lats

    if has_lats and has_height:
        ptype = 2
        x = lats
        y = height

        xname = cf_var_name(field=f, dim="Y")
        xunits = str(getattr(f.construct("Y"), "Units", ""))
        if xunits == "degrees_north":
            xunits = "degrees"
        if xunits != "":
            xlabel = f"{xname} ({xunits})"
        else:
            xlabel = xname

        yname = cf_var_name(field=f, dim=myz)
        yunits = str(getattr(f.construct(myz), "Units", ""))
        if yunits != "":
            ylabel = f"{yname} ({yunits})"
        else:
            ylabel = yname

    if has_lons and has_height:
        ptype = 3
        x = lons
        y = height

        xname = cf_var_name(field=f, dim="X")
        xunits = str(getattr(f.construct("X"), "Units", ""))
        if xunits == "degrees_east":
            xunits = "degrees"
        if xunits != "":
            xlabel = f"{xname} ({xunits})"
        else:
            xlabel = xname

        yname = cf_var_name(field=f, dim=myz)
        yunits = str(getattr(f.construct(myz), "Units", ""))
        if yunits != "":
            ylabel = f"{yname} ({yunits})"
        else:
            ylabel = yname

    if has_lons and has_time:
        ptype = 4
        x = lons
        y = time

        xname = cf_var_name(field=f, dim="X")
        xunits = str(getattr(f.construct("X"), "Units", ""))
        if xunits == "degrees_east":
            xunits = "degrees"
        if xunits != "":
            xlabel = f"{xname} ({xunits})"
        else:
            xlabel = xname

        yname = cf_var_name(field=f, dim="T")
        yunits = str(getattr(f.construct("T"), "Units", ""))
        if yunits != "":
            ylabel = f"{yname} ({yunits})"
        else:
            ylabel = yname

    if has_lats and has_time:
        ptype = 5
        x = lats
        y = time

        xname = cf_var_name(field=f, dim="Y")
        xunits = str(getattr(f.construct("Y"), "Units", ""))
        if xunits == "degrees_north":
            xunits = "degrees"
        if xunits != "":
            xlabel = f"{xname} ({xunits})"
        else:
            xlabel = xname

        yname = cf_var_name(field=f, dim="T")
        yunits = str(getattr(f.construct("T"), "Units", ""))
        if yunits != "":
            ylabel = f"{yname} ({yunits})"
        else:
            ylabel = yname

    # time height plot
    if has_height and has_time:
        ptype = 7
        x = time
        y = height

        xname = cf_var_name(field=f, dim="T")
        xunits = str(getattr(f.construct("T"), "Units", ""))
        if xunits != "":
            xlabel = f"{xname} ({xunits})"
        else:
            xlabel = xname

        yname = cf_var_name(field=f, dim="Z")
        yunits = str(getattr(f.construct("Z"), "Units", ""))
        if yunits != "":
            ylabel = f"{yname} ({yunits})"
        else:
            ylabel = yname

        # Rotate array to get it as time vs height
        field = np.rot90(field)
        field = np.flipud(field)

    # Rotated pole
    if f.ref("grid_mapping_name:rotated_latitude_longitude", default=False):
        ptype = 6

        rotated_pole = f.ref("grid_mapping_name:rotated_latitude_longitude")
        xpole = rotated_pole["grid_north_pole_longitude"]
        ypole = rotated_pole["grid_north_pole_latitude"]

        # Extract grid x and y coordinates
        for mydim in list(f.dimension_coordinates()):
            name = cf_var_name(field=f, dim=mydim)

            if name in ["grid_longitude", "longitude", "x"]:
                x = np.squeeze(f.construct(mydim).array)
                xunits = str(getattr(f.construct(mydim), "units", ""))
                xlabel = cf_var_name(field=f, dim=mydim)

            if name in ["grid_latitude", "latitude", "y"]:
                y = np.squeeze(f.construct(mydim).array)
                # Flip y and data if reversed
                if y[0] > y[-1]:
                    y = y[::-1]
                    field = np.flipud(field)
                yunits = str(getattr(f.construct(mydim), "Units", ""))
                ylabel = cf_var_name(field=f, dim=mydim) + yunits

    # Extract auxiliary lons and lats if they exist
    if ptype == 1 or ptype is None:
        if plotvars.proj != "rotated" and not rotated_vect:
            aux_lons = False
            aux_lats = False
            for mydim in list(f.auxiliary_coordinates()):
                name = cf_var_name(field=f, dim=mydim)
                if name in ["longitude"]:
                    xpts = np.squeeze(f.construct(mydim).array)
                    aux_lons = True
                if name in ["latitude"]:
                    ypts = np.squeeze(f.construct(mydim).array)
                    aux_lats = True

            if aux_lons and aux_lats:
                x = xpts
                y = ypts
                ptype = 1

    # UKCP grid
    if f.ref("grid_mapping_name:transverse_mercator", default=False):
        ptype = 1
        field = np.squeeze(f.array)

        # Find the auxiliary lons and lats if provided
        has_lons = False
        has_lats = False
        for mydim in list(f.auxiliary_coordinates()):
            name = cf_var_name(field=f, dim=mydim)
            if name in ["longitude"]:
                x = np.squeeze(f.construct(mydim).array)
                has_lons = True
            if name in ["latitude"]:
                y = np.squeeze(f.construct(mydim).array)
                has_lats = True

        # Calculate lons and lats if no auxiliary data for these
        if not has_lons or not has_lats:
            xpts = f.construct("X").array
            ypts = f.construct("Y").array
            field = np.squeeze(f.array)

            ref = f.ref("grid_mapping_name:transverse_mercator")
            false_easting = ref["false_easting"]
            false_northing = ref["false_northing"]
            central_longitude = ref["longitude_of_central_meridian"]
            central_latitude = ref["latitude_of_projection_origin"]
            scale_factor = ref["scale_factor_at_central_meridian"]

            # Set the transform
            transform = ccrs.TransverseMercator(
                false_easting=false_easting,
                false_northing=false_northing,
                central_longitude=central_longitude,
                central_latitude=central_latitude,
                scale_factor=scale_factor,
            )

            # Calculate the longitude and latitude points
            xvals, yvals = np.meshgrid(xpts, ypts)
            points = ccrs.PlateCarree().transform_points(
                transform, xvals, yvals
            )
            x = np.array(points)[:, :, 0]
            y = np.array(points)[:, :, 1]

    # None of the above
    if ptype is None:
        ptype = 0

        data_axes = f.get_data_axes()
        count = 1
        for d in data_axes:
            try:
                c = f.coordinate(filter_by_axis=[d])
                if np.size(c.array) > 1:
                    if count == 1:

                        y = c
                        mycoord = "dimensioncoordinate" + str(d[-1])
                        yunits = str(getattr(f.coord(mycoord), "Units", ""))
                        if yunits != "":
                            yunits = f"({yunits})"
                        ylabel = cf_var_name(field=f, dim=mycoord) + yunits
                    elif count == 2:
                        x = c
                        mycoord = "dimensioncoordinate" + str(d[-1])
                        xunits = str(getattr(f.coord(mycoord), "units", ""))
                        if xunits != "":
                            xunits = f"({xunits})"
                        xlabel = cf_var_name(field=f, dim=mycoord) + xunits
                    count += 1
            except ValueError:
                errstr = (
                    "\n\n_cf_data_assign - cannot find data to return\n\n"
                    f"{f.constructs.domain_axis_identity(d)}\n\n"
                )
                raise Warning(errstr)

    # Assign colorbar_title
    if colorbar_title is None:
        colorbar_title = "No Name"
        if hasattr(f, "id"):
            colorbar_title = f.id
        nc = f.nc_get_variable(None)
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
                colorbar_title = colorbar_title
            else:
                colorbar_title = f"{colorbar_title} ({_supscr(str(f.Units))})"

    # Return data
    return (field, x, y, ptype, colorbar_title, xlabel, ylabel, xpole, ypole)


def _check_data(field=None, x=None, y=None):
    """
    | Check user input contour data is correct.
    | This is an internal routine and is not used by the user.
    |
    | field=None - field
    | x=None - x points for field
    | y=None - y points for field
    |
    """

    # Input error trapping
    args = True
    if np.size(field) == 1:
        if field is None:
            errstr = (
                "\ncon error - a field for contouring must be "
                "passed with the f= flag\n"
            )
            args = False
    if np.size(x) == 1:
        if x is None:
            x = np.arange(np.shape(field)[1])
    if np.size(y) == 1:
        if y is None:
            y = np.arange(np.shape(field)[0])
    if not args:
        raise Warning(errstr)

    # Check input dimensions look okay.
    # All inputs 2D
    if np.ndim(field) == 2 and np.ndim(x) == 2 and np.ndim(y) == 2:
        xpts = np.shape(field)[1]
        ypts = np.shape(field)[0]
        if xpts != np.shape(x)[1] or xpts != np.shape(y)[1]:
            args = False
        if ypts != np.shape(x)[0] or ypts != np.shape(y)[0]:
            args = False
        if args:
            return

    # Field x and y all 1D
    if np.ndim(field) == 1 and np.ndim(x) == 1 and np.ndim(y) == 1:
        if np.size(x) != np.size(field):
            args = False
        if np.size(y) != np.size(field):
            args = False
        if args:
            return

    # Field 2D, x and y 1D
    if np.ndim(field) != 2:
        args = False
    if np.ndim(x) != 1:
        args = False
    if np.ndim(y) != 1:
        args = False
    if np.ndim(field) == 2:
        if np.size(x) != np.shape(field)[1]:
            args = False
        if np.size(y) != np.shape(field)[0]:
            args = False

    if args is False:
        errstr = (
            "\nInput arguments incorrectly shaped:\n"
            f"x has shape:{np.shape(x)}\n"
            f"y has shape:{np.shape(y)}\n"
            f"field has shape{np.shape(field)}\n\n"
            "Expected x=xpts, y=ypts, field=(ypts,xpts)\n"
            "x=npts, y=npts, field=npts\n"
            "or x=[ypts, xpts], y=[ypts, xpts], field=[ypts, xpts]\n"
        )
        raise Warning(errstr)


def _cscale_get_map():
    """
    | Return colour map for use in contour plots.
    | This is an internal routine and is not used by the user.
    |
    | This depends on the colour bar extensions.
    |
    :Returns:
         colour map
    |
    """
    cscale_ncols = np.size(plotvars.cs)
    if plotvars.levels_extend == "both":
        colmap = plotvars.cs[1 : cscale_ncols - 1]
    if plotvars.levels_extend == "min":
        colmap = plotvars.cs[1:]
    if plotvars.levels_extend == "max":
        colmap = plotvars.cs[: cscale_ncols - 1]
    if plotvars.levels_extend == "neither":
        colmap = plotvars.cs
    return colmap


def _bfill(
    f=None,
    x=None,
    y=None,
    clevs=False,
    lonlat=None,
    bound=False,
    alpha=1.0,
    single_fill_color=None,
    white=True,
    zorder=4,
    fast=None,
    transform=False,
    orca=False,
):
    """
    | Block fill a field with colour rectangles.
    | This is an internal routine and is not generally used by the user.
    |
    | f=None - field
    | x=None - x points for field
    | y=None - y points for field
    | clevs=None - levels for filling
    | lonlat=None - longitude and latitude data
    | bound=False - x and y are cf data boundaries
    | alpha=alpha - transparency setting 0 to 1
    | white=True - colour unplotted areas white
    | single_fill_color=None - colour for a blockfill between two levels
    |                        - makes maplotlib named colours or
    |                        - hexadecimal notation - '#d3d3d3' for grey
    | zorder=4 - plotting order
    | fast=None - use fast plotting with pcolormesh which is useful for
    |             larger datasets
    | transform=False - map transform supplied by calling routine
    | orca=False - data is orca data
    |
     :Returns:
       None
    |
    """

    # Set lonlat if not specified
    lonlat = False
    if plotvars.plot_type == 1:
        lonlat = True

    # If single_fill_color is defined then turn off whiting out the background.
    if single_fill_color is not None:
        white = False

    # Set 2D lon lat if data is that format
    two_d = False
    if not isinstance(f, cf.Field):
        if np.ndim(x) == 2 and np.ndim(x) == 2:
            two_d = True

    # Set the default map coordinates for the data to be PlateCarree
    plotargs = {}
    if lonlat:
        plotargs = {"transform": ccrs.PlateCarree()}

    # Set the field
    if isinstance(f, cf.Field):
        field = f.array
    else:
        field = f

    # Get colour scale for use in contouring
    # If colour bar extensions are enabled then the colour map goes
    # from 1 to ncols-2.  The colours for the colour bar extensions
    # are then changed on the colorbar and plot after the plot is made
    ncols_addition = 0
    if single_fill_color is None:
        colmap = _cscale_get_map()
        cmap = matplotlib.colors.ListedColormap(colmap)
        if plotvars.levels_extend in ["min", "both"]:
            cmap.set_under(plotvars.cs[0])
            clevs = np.append(-1e-30, clevs)
            ncols_addition += 1
        if plotvars.levels_extend in ["max", "both"]:
            cmap.set_over(plotvars.cs[-1])
            clevs = np.append(clevs, 1e30)
            ncols_addition += 1
    else:
        cols = single_fill_color
        cmap = matplotlib.colors.ListedColormap(cols)

    levels = np.array(deepcopy(clevs)).astype("float")

    # Colour array for storing the cell colour
    colarr = np.zeros([np.shape(field)[0], np.shape(field)[1]])
    for i in np.arange(np.size(levels) - 1):
        lev = levels[i]
        pts = np.where(np.logical_and(field >= lev, field < levels[i + 1]))
        colarr[pts] = int(i)

    # Change points that are masked back to -1
    if isinstance(field, np.ma.MaskedArray):
        pts = np.ma.where(field.mask)
        if np.size(pts) > 0:
            colarr[pts] = -1

    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N + ncols_addition)

    if isinstance(f, cf.Field):
        if f.ref("grid_mapping_name:transverse_mercator", default=False):
            lonlat = True

            # Case of transverse mercator of which UKCP is an example
            ref = f.ref("grid_mapping_name:transverse_mercator")
            false_easting = ref["false_easting"]
            false_northing = ref["false_northing"]
            central_longitude = ref["longitude_of_central_meridian"]
            central_latitude = ref["latitude_of_projection_origin"]
            scale_factor = ref["scale_factor_at_central_meridian"]

            transform = ccrs.TransverseMercator(
                false_easting=false_easting,
                false_northing=false_northing,
                central_longitude=central_longitude,
                central_latitude=central_latitude,
                scale_factor=scale_factor,
            )

            # Extract the axes and data
            xpts = np.append(
                f.dim("X").bounds.array[:, 0], f.dim("X").bounds.array[-1, 1]
            )
            ypts = np.append(
                f.dim("Y").bounds.array[:, 0], f.dim("Y").bounds.array[-1, 1]
            )
            field = np.squeeze(f.array)
            plotargs = {"transform": transform}

    else:

        if two_d is False:
            if bound:
                xpts = x
                ypts = y
            else:
                # Find x box boundaries
                xpts = x[0] - (x[1] - x[0]) / 2.0
                for ix in np.arange(np.size(x) - 1):
                    xpts = np.append(xpts, x[ix] + (x[ix + 1] - x[ix]) / 2.0)
                xpts = np.append(xpts, x[ix + 1] + (x[ix + 1] - x[ix]) / 2.0)

                # Find y box boundaries
                ypts = y[0] - (y[1] - y[0]) / 2.0
                for iy in np.arange(np.size(y) - 1):
                    ypts = np.append(ypts, y[iy] + (y[iy + 1] - y[iy]) / 2.0)
                ypts = np.append(ypts, y[iy + 1] + (y[iy + 1] - y[iy]) / 2.0)

            # Shift lon grid if needed
            if lonlat:
                # Extract upper bound and original rhs of box longitude
                # bounding points
                upper_bound = ypts[-1]

                # Reduce xpts and ypts by 1 or shifting of grid fails
                # The last points are the right / upper bounds for the
                # last data box
                xpts = xpts[0:-1]
                ypts = ypts[0:-1]

                if plotvars.lonmin < np.nanmin(xpts):
                    xpts = xpts - 360
                if plotvars.lonmin > np.nanmax(xpts):
                    xpts = xpts + 360

                # Add cyclic information if missing.
                lonrange = np.nanmax(xpts) - np.nanmin(xpts)
                if lonrange < 360 and lonrange > 350:
                    # field, xpts = cartopy_util.add_cyclic_point(field, xpts)
                    field, xpts = add_cyclic(field, xpts)

                right_bound = xpts[-1] + (xpts[-1] - xpts[-2])

                # Add end x and y end points
                xpts = np.append(xpts, right_bound)
                ypts = np.append(ypts, upper_bound)

        if two_d:
            # 2D lons and lats code
            if fast:
                xpts = x
                ypts = y
            else:
                nx = np.shape(x)[1]
                ny = np.shape(x)[0]

                for ix in np.arange(nx):
                    for iy in np.arange(ny):

                        # Calculate the local size difference and set the
                        # square points
                        if ix < nx - 2:
                            xdiff = (x[iy, ix + 1] - x[iy, ix]) / 2
                        else:
                            xdiff = (x[iy, ix] - x[iy, ix - 1]) / 2

                        if iy < ny - 2:
                            ydiff = (y[iy + 1, ix] - y[iy, ix]) / 2
                        else:
                            ydiff = (y[iy, ix] - y[iy - 1, ix]) / 2

                        xpts = [
                            x[iy, ix] - xdiff,
                            x[iy, ix] + xdiff,
                            x[iy, ix] + xdiff,
                            x[iy, ix] - xdiff,
                            x[iy, ix] - xdiff,
                        ]
                        ypts = [
                            y[iy, ix] - ydiff,
                            y[iy, ix] - ydiff,
                            y[iy, ix] + ydiff,
                            y[iy, ix] + ydiff,
                            y[iy, ix] - ydiff,
                        ]

                        # Plot the square
                        plotvars.mymap.add_patch(
                            mpatches.Polygon(
                                [
                                    [xpts[0], ypts[0]],
                                    [xpts[1], ypts[1]],
                                    [xpts[2], ypts[2]],
                                    [xpts[3], ypts[3]],
                                    [xpts[4], ypts[4]],
                                ],
                                facecolor=plotvars.cs[int(colarr[iy, ix])],
                                zorder=zorder,
                                transform=ccrs.PlateCarree(),
                            )
                        )

                return

    # Polar stereographic
    # Set points past plotting limb to be plotvars.boundinglat
    # Also set any lats past the pole to be the pole
    if plotvars.proj == "npstere":
        pts = np.where(ypts < plotvars.boundinglat)
        if np.size(pts) > 0:
            ypts[pts] = plotvars.boundinglat
        pts = np.where(ypts > 90.0)
        if np.size(pts) > 0:
            ypts[pts] = 90.0

    if plotvars.proj == "spstere":
        pts = np.where(ypts > plotvars.boundinglat)
        if np.size(pts) > 0:
            ypts[pts] = plotvars.boundinglat
        pts = np.where(ypts < -90.0)
        if np.size(pts) > 0:
            ypts[pts] = -90.0

    # Set the transform if not supplied to _bfill
    if transform:
        lonlat = True
    else:
        transform = ccrs.PlateCarree()

    if fast:
        if isinstance(clevs, int):
            norm = False

        if two_d:
            # Plot using pcolormesh if a 2D grid
            # field = f
            fixed_x = x.copy()
            for i, start in enumerate(
                np.argmax(np.abs(np.diff(x)) > 180, axis=1)
            ):
                fixed_x[i, start + 1 :] += 360
            plotvars.image = plotvars.mymap.pcolormesh(
                fixed_x, y, field, cmap=cmap, transform=transform, norm=norm
            )

        else:
            if lonlat:
                for offset in [0, 360.0]:
                    if isinstance(clevs, int):
                        plotvars.image = plotvars.mymap.pcolormesh(
                            xpts + offset,
                            ypts,
                            field,
                            transform=transform,
                            cmap=cmap,
                        )
                    else:
                        plotvars.image = plotvars.mymap.pcolormesh(
                            xpts + offset,
                            ypts,
                            field,
                            transform=transform,
                            cmap=cmap,
                            norm=norm,
                        )

            else:
                if isinstance(clevs, int):
                    plotvars.image = plotvars.plot.pcolormesh(
                        xpts, ypts, field, cmap=cmap
                    )
                else:
                    plotvars.image = plotvars.plot.pcolormesh(
                        xpts, ypts, field, cmap=cmap, norm=norm
                    )

    else:

        if plotvars.plot_type == 1 and plotvars.proj != "cyl":

            for i in np.arange(np.size(levels) - 1):
                allverts = []
                xy_stack = np.column_stack(np.where(colarr == i))

                for pt in np.arange(np.shape(xy_stack)[0]):
                    ix = xy_stack[pt][1]
                    iy = xy_stack[pt][0]
                    lons = [
                        xpts[ix],
                        xpts[ix + 1],
                        xpts[ix + 1],
                        xpts[ix],
                        xpts[ix],
                    ]
                    lats = [
                        ypts[iy],
                        ypts[iy],
                        ypts[iy + 1],
                        ypts[iy + 1],
                        ypts[iy],
                    ]

                    txpts, typts = lons, lats
                    verts = [
                        (txpts[0], typts[0]),
                        (txpts[1], typts[1]),
                        (txpts[2], typts[2]),
                        (txpts[3], typts[3]),
                        (txpts[4], typts[4]),
                    ]

                    allverts.append(verts)

                # Make the collection and add it to the plot.
                if single_fill_color is None:
                    color = plotvars.cs[i]
                else:
                    color = single_fill_color
                coll = PolyCollection(
                    allverts,
                    facecolor=color,
                    edgecolors=color,
                    alpha=alpha,
                    zorder=zorder,
                    **plotargs,
                )

                if lonlat:
                    plotvars.mymap.add_collection(coll)
                else:
                    plotvars.plot.add_collection(coll)
        else:
            for i in np.arange(np.size(levels) - 1):

                allverts = []
                xy_stack = np.column_stack(np.where(colarr == i))
                for pt in np.arange(np.shape(xy_stack)[0]):
                    ix = xy_stack[pt][1]
                    iy = xy_stack[pt][0]
                    verts = [
                        (xpts[ix], ypts[iy]),
                        (xpts[ix + 1], ypts[iy]),
                        (xpts[ix + 1], ypts[iy + 1]),
                        (xpts[ix], ypts[iy + 1]),
                        (xpts[ix], ypts[iy]),
                    ]

                    allverts.append(verts)

                # Make the collection and add it to the plot.
                if single_fill_color is None:
                    color = plotvars.cs[i]
                else:
                    color = single_fill_color

                coll = PolyCollection(
                    allverts,
                    facecolor=color,
                    edgecolors=color,
                    alpha=alpha,
                    zorder=zorder,
                    **plotargs,
                )

                if lonlat:
                    plotvars.mymap.add_collection(coll)
                else:
                    plotvars.plot.add_collection(coll)

        # Add white for undefined areas
        if white:
            allverts = []
            xy_stack = np.column_stack(np.where(colarr == -1))
            for pt in np.arange(np.shape(xy_stack)[0]):
                ix = xy_stack[pt][1]
                iy = xy_stack[pt][0]

                verts = [
                    (xpts[ix], ypts[iy]),
                    (xpts[ix + 1], ypts[iy]),
                    (xpts[ix + 1], ypts[iy + 1]),
                    (xpts[ix], ypts[iy + 1]),
                    (xpts[ix], ypts[iy]),
                ]

                allverts.append(verts)

            # Make the collection and add it to the plot.
            color = plotvars.cs[i]
            coll = PolyCollection(
                allverts,
                facecolor="#ffffff",
                edgecolors="#ffffff",
                alpha=alpha,
                zorder=zorder,
                **plotargs,
            )

            if lonlat:
                plotvars.mymap.add_collection(coll)
            else:
                plotvars.plot.add_collection(coll)


def _set_map():
    """
    | Set map and write into `plotvars.mymap`.
    | This is an internal routine and not used by the user.
    |
    | No inputs
    |
    |
    :Returns:
     None
    |
    """

    # Return if mymap is already set
    if plotvars.mymap is not None:
        return

    # Set up mapping
    extent = True
    lonmin = plotvars.lonmin
    lonmax = plotvars.lonmax
    latmin = plotvars.latmin
    latmax = plotvars.latmax
    lon_diff = lonmax - lonmin
    lon_mid = lonmin + lon_diff / 2.0

    vproj = plotvars.proj

    # Cartopy treats values as cyclic on the globe, so in order to avoid
    # errors where it thinks we are inputting the same value, e.g:
    #     UserWarning: Attempting to set identical low and high ylims makes
    #     transformation singular; automatically expanding.
    if vproj in ("cyl", "merc") and lon_diff == 360.0:
        lonmax += 0.01  # ask to plot a tiny extra increment to distinguish

    # Start of projection-specific logic
    if vproj == "cyl":
        proj = ccrs.PlateCarree(central_longitude=lon_mid)
    elif vproj == "merc":
        min_latitude = -80.0
        if lonmin > min_latitude:
            min_latitude = plotvars.lonmin
        max_latitude = 84.0
        if lonmax < max_latitude:
            max_latitude = plotvars.lonmax
        proj = ccrs.Mercator(
            central_longitude=plotvars.lon_0,
            min_latitude=min_latitude,
            max_latitude=max_latitude,
        )
    elif vproj == "npstere":
        proj = ccrs.NorthPolarStereo(central_longitude=plotvars.lon_0)
        # **cartopy 0.16 fix
        # Here we add in 0.01 to the longitude extent as this helps with
        # plotting lines and line labels
        lonmin = plotvars.lon_0 - 180
        lonmax = plotvars.lon_0 + 180.01
        latmin = plotvars.boundinglat
        latmax = 90
    elif vproj == "spstere":
        proj = ccrs.SouthPolarStereo(central_longitude=plotvars.lon_0)
        # **cartopy 0.16 fix
        # Here we add in 0.01 to the longitude extent as this helps with
        # plotting lines and line labels
        lonmin = plotvars.lon_0 - 180
        lonmax = plotvars.lonmax + 180.01
        latmin = -90
        latmax = plotvars.boundinglat
    elif vproj == "ortho":
        proj = ccrs.Orthographic(
            central_longitude=plotvars.lon_0, central_latitude=plotvars.lat_0
        )
        lonmin = plotvars.lon_0 - 180.0
        lonmax = plotvars.lon_0 + 180.01
        extent = False
    elif vproj == "moll":
        proj = ccrs.Mollweide(central_longitude=plotvars.lon_0)
        lonmin = plotvars.lon_0 - 180.0
        lonmax = plotvars.lon_0 + 180.01
        extent = False
    elif vproj == "robin":
        proj = ccrs.Robinson(central_longitude=plotvars.lon_0)
    elif vproj == "lcc":
        latmin = plotvars.latmin
        latmax = plotvars.latmax
        lonmin = plotvars.lonmin
        lonmax = plotvars.lonmax
        lon_0 = lonmin + (lonmax - lonmin) / 2.0
        lat_0 = latmin + (latmax - latmin) / 2.0
        cutoff = -40
        if lat_0 <= 0:
            cutoff = 40
        standard_parallels = [33, 45]
        if latmin <= 0 and latmax <= 0:
            standard_parallels = [-45, -33]
        proj = ccrs.LambertConformal(
            central_longitude=lon_0,
            central_latitude=lat_0,
            cutoff=cutoff,
            standard_parallels=standard_parallels,
        )
    elif vproj == "rotated":
        proj = ccrs.PlateCarree(central_longitude=lon_mid)
    elif vproj == "OSGB":
        proj = ccrs.OSGB()
    elif vproj == "EuroPP":
        proj = ccrs.EuroPP()
    elif vproj == "UKCP":
        # Special case of TransverseMercator for UKCP
        proj = ccrs.TransverseMercator()
    elif vproj == "TransverseMercator":
        proj = ccrs.TransverseMercator()
        lonmin = plotvars.lon_0 - 180.0
        lonmax = plotvars.lon_0 + 180.01
        extent = False
    elif vproj == "LambertCylindrical":
        proj = ccrs.LambertCylindrical()
        lonmin = plotvars.lonmin
        lonmax = plotvars.lonmax
        latmin = plotvars.latmin
        latmax = plotvars.latmax
    # End of projection-specific logic

    # Add a plot containing the projection
    if plotvars.plot_xmin:
        delta_x = plotvars.plot_xmax - plotvars.plot_xmin
        delta_y = plotvars.plot_ymax - plotvars.plot_ymin
        mymap = plotvars.master_plot.add_axes(
            [plotvars.plot_xmin, plotvars.plot_ymin, delta_x, delta_y],
            projection=proj,
        )
    else:
        mymap = plotvars.master_plot.add_subplot(
            plotvars.rows, plotvars.columns, plotvars.pos, projection=proj
        )

    # Set map extent
    set_extent = True
    if vproj in ["OSGB", "EuroPP", "UKCP", "robin", "lcc"]:
        set_extent = False
    if extent and set_extent:
        mymap.set_extent(
            [lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree()
        )

    # Update the scaling for certain projections
    if vproj == "cyl":
        mymap.set_aspect(plotvars.aspect)
    elif vproj == "lcc":
        # Special case of lcc
        mymap.set_extent(
            [lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree()
        )
    elif vproj == "UKCP":
        # Special case of TransverseMercator for UKCP
        mymap.set_extent([-11, 3, 49, 61], crs=ccrs.PlateCarree())
    elif vproj == "EuroPP":
        # EuroPP somehow needs some limits setting.
        mymap.set_extent([-12, 25, 30, 75], crs=ccrs.PlateCarree())

    # Remove any plotvars.plot axes leaving just the plotvars.mymap axes
    plotvars.plot.set_frame_on(False)
    plotvars.plot.set_xticks([])
    plotvars.plot.set_yticks([])

    # Store map
    plotvars.mymap = mymap


def _map_title(title=None, dims=False):
    """
    | An internal routine to draw a title on a map plot.
    |
    | title=None - title to put on map plot
    | dim=False - draw a set of dimension titles
    |
    """

    boundinglat = plotvars.boundinglat
    lon_0 = plotvars.lon_0
    lonmin = plotvars.lonmin
    lonmax = plotvars.lonmax
    latmin = plotvars.latmin
    latmax = plotvars.latmax
    polar_range = 90 - abs(boundinglat)

    myprojs = ["cyl", "robin", "moll", "merc"]
    if plotvars.proj in myprojs:
        lon_mid = lonmin + (lonmax - lonmin) / 2.0
        mylon = lon_mid
        if dims:
            mylon = lonmin
        projs = [
            ccrs.PlateCarree,
            ccrs.Robinson,
            ccrs.Mollweide,
            ccrs.Mercator,
        ]
        myind = myprojs.index(plotvars.proj)
        # if plotvars.proj == 'cyl':
        #    proj = ccrs.PlateCarree(central_longitude=lon_mid)
        proj = projs[myind](central_longitude=lon_mid)
        mylat = latmax
        xpt, ypt = proj.transform_point(mylon, mylat, ccrs.PlateCarree())
        ypt = ypt + (latmax - latmin) / 40.0

    if plotvars.proj == "npstere":
        mylon = lon_0 + 180
        mylat = boundinglat - polar_range / 15.0
        proj = ccrs.NorthPolarStereo(central_longitude=lon_0)
        xpt, ypt = proj.transform_point(mylon, mylat, ccrs.PlateCarree())
        if dims:
            mylon = lon_0 + 180
            mylat = boundinglat - polar_range / 15.0
            xpt_mid, ypt = proj.transform_point(
                mylon, mylat, ccrs.PlateCarree()
            )
            mylon = lon_0 - 90
            xpt, ypt_mid = proj.transform_point(
                mylon, mylat, ccrs.PlateCarree()
            )

    if plotvars.proj == "spstere":
        mylon = lon_0
        mylat = boundinglat + polar_range / 15.0
        proj = ccrs.SouthPolarStereo(central_longitude=lon_0)
        xpt, ypt = proj.transform_point(mylon, mylat, ccrs.PlateCarree())
        if dims:
            mylon = lon_0 + 0
            # mylat = boundinglat-polar_range/15.0
            mylat = boundinglat - polar_range / 15.0
            xpt_mid, ypt = proj.transform_point(
                mylon, mylat, ccrs.PlateCarree()
            )
            mylon = lon_0 - 90
            xpt, ypt_mid = proj.transform_point(
                mylon, mylat, ccrs.PlateCarree()
            )

    if plotvars.proj == "lcc":
        mylon = lonmin + (lonmax - lonmin) / 2.0
        if dims:
            mylon = lonmin
        lat_0 = 40
        if latmin <= 0 and latmax <= 0:
            lat_0 = 40
        proj = ccrs.LambertConformal(
            central_longitude=plotvars.lon_0,
            central_latitude=lat_0,
            cutoff=plotvars.latmin,
        )
        mylat = latmax
        xpt, ypt = proj.transform_point(mylon, mylat, ccrs.PlateCarree())

    fontsize = plotvars.title_fontsize

    if dims:
        halign = "left"
        fontsize = plotvars.axis_label_fontsize

        # Get plot position
        this_plot = plotvars.plot
        left, bottom, width, height = this_plot.get_position().bounds

        # Shift to left
        # if plotvars.plot_type == 1 and plotvars.proj !=cyl:
        left -= 0.1
        this_plot.set_position([left, bottom, width, height])

        left, bottom, width, height = this_plot.get_position().bounds

        plotvars.plot.text(
            left + width,
            bottom + height,
            title,
            va="bottom",
            ha=halign,
            rotation="horizontal",
            rotation_mode="anchor",
            fontsize=fontsize,
            fontweight=plotvars.title_fontweight,
        )

    else:
        halign = "center"
        plotvars.mymap.text(
            xpt,
            ypt,
            title,
            va="bottom",
            ha=halign,
            rotation="horizontal",
            rotation_mode="anchor",
            fontsize=fontsize,
            fontweight=plotvars.title_fontweight,
        )


def _plot_map_axes(
    axes=None,
    xaxis=None,
    yaxis=None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    user_xlabel=None,
    user_ylabel=None,
    verbose=None,
):
    """
    | An internal routine to draw the axes on a map plot.
    |
    | axes=None - drawing axes
    | xaxis=None - drawing x-axis
    | yaxis=None - drawing x-axis
    | xticks=None - user defined xticks
    | xticklabels=None - user defined xtick labels
    | yticks=None - user defined yticks
    | yticklabels=None - user defined ytick labels
    | user_xlabel=None - user defined xlabel
    | user_ylabel=None - user defined ylabel
    | verbose=None
    |
    """
    # Font definitions
    axis_label_fontsize = plotvars.axis_label_fontsize
    axis_label_fontweight = plotvars.axis_label_fontweight

    # Map parameters
    boundinglat = plotvars.boundinglat
    lon_0 = plotvars.lon_0
    lonmin = plotvars.lonmin
    lonmax = plotvars.lonmax
    latmin = plotvars.latmin
    latmax = plotvars.latmax

    # Cylindrical
    if plotvars.proj == "cyl":

        if verbose:
            print("con - adding cylindrical axes")
        lonticks, lonlabels = _mapaxis(
            min=plotvars.lonmin, max=plotvars.lonmax, type=1
        )
        latticks, latlabels = _mapaxis(
            min=plotvars.latmin, max=plotvars.latmax, type=2
        )

        if axes:
            if xaxis:
                if xticks is None:
                    axes_plot(xticks=lonticks, xticklabels=lonlabels)
                else:
                    if xticklabels is None:
                        axes_plot(xticks=xticks, xticklabels=xticks)
                    else:
                        axes_plot(xticks=xticks, xticklabels=xticklabels)
            if yaxis:
                if yticks is None:
                    axes_plot(yticks=latticks, yticklabels=latlabels)
                else:
                    if yticklabels is None:
                        axes_plot(yticks=yticks, yticklabels=yticks)
                    else:
                        axes_plot(yticks=yticks, yticklabels=yticklabels)

            if user_xlabel is not None:
                plot.text(
                    0.5,
                    -0.10,
                    user_xlabel,
                    va="bottom",
                    ha="center",
                    rotation="horizontal",
                    rotation_mode="anchor",
                    transform=plotvars.mymap.transAxes,
                    fontsize=axis_label_fontsize,
                    fontweight=axis_label_fontweight,
                )

            if user_ylabel is not None:
                plot.text(
                    -0.05,
                    0.50,
                    user_ylabel,
                    va="bottom",
                    ha="center",
                    rotation="vertical",
                    rotation_mode="anchor",
                    transform=plotvars.mymap.transAxes,
                    fontsize=axis_label_fontsize,
                    fontweight=axis_label_fontweight,
                )

    # Polar stereographic
    if plotvars.proj == "npstere" or plotvars.proj == "spstere":
        if verbose:
            print("con - adding stereographic axes")

        mymap = plotvars.mymap
        latrange = 90 - abs(boundinglat)
        proj = ccrs.Geodetic()

        # Add
        if axes:
            if xaxis:
                if yticks is None:
                    latvals = np.arange(5) * 30 - 60
                else:
                    latvals = np.array(yticks)

                if plotvars.proj == "npstere":
                    latvals = latvals[np.where(latvals >= boundinglat)]
                else:
                    latvals = latvals[np.where(latvals <= boundinglat)]

                for lat in latvals:
                    if abs(lat - boundinglat) > 1:
                        lons = np.arange(361)
                        lats = np.zeros(361) + lat
                        mymap.plot(
                            lons,
                            lats,
                            color=plotvars.grid_colour,
                            linewidth=plotvars.grid_thickness,
                            linestyle=plotvars.grid_linestyle,
                            transform=proj,
                        )

            if yaxis:
                if xticks is None:
                    lonvals = np.arange(7) * 60
                else:
                    lonvals = xticks

                for lon in lonvals:
                    label = _mapaxis(lon, lon, 1)[1][0]

                    if plotvars.proj == "npstere":
                        lats = np.arange(90 - boundinglat) + boundinglat
                    else:
                        lats = np.arange(boundinglat + 91) - 90
                    lons = np.zeros(np.size(lats)) + lon
                    mymap.plot(
                        lons,
                        lats,
                        color=plotvars.grid_colour,
                        linewidth=plotvars.grid_thickness,
                        linestyle=plotvars.grid_linestyle,
                        transform=proj,
                    )

            # Add longitude labels
            if plotvars.proj == "npstere":
                proj = ccrs.NorthPolarStereo(central_longitude=lon_0)
                pole = 90
                latpt = boundinglat - latrange / 40.0
            else:
                proj = ccrs.SouthPolarStereo(central_longitude=lon_0)
                pole = -90
                latpt = boundinglat + latrange / 40.0

            lon_mid, lat_mid = proj.transform_point(
                0, pole, ccrs.PlateCarree()
            )

            if xaxis and axis_label_fontsize > 0.0:
                for xtick in lonvals:
                    label = _mapaxis(xtick, xtick, 1)[1][0]
                    lonr, latr = proj.transform_point(
                        xtick, latpt, ccrs.PlateCarree()
                    )

                    v_align = "center"
                    if lonr < 1:
                        h_align = "right"
                    if lonr > 1:
                        h_align = "left"
                    if abs(lonr) <= 1:
                        h_align = "center"
                        if latr < 1:
                            v_align = "top"
                        if latr > 1:
                            v_align = "bottom"

                    mymap.text(
                        lonr,
                        latr,
                        label,
                        horizontalalignment=h_align,
                        verticalalignment=v_align,
                        fontsize=axis_label_fontsize,
                        fontweight=axis_label_fontweight,
                        zorder=101,
                    )

        # Make the plot circular by blanking off around the plot
        # Find min and max of plotting region in map coordinates
        lons = np.arange(360)
        lats = np.zeros(np.size(lons)) + boundinglat
        device_coords = proj.transform_points(ccrs.PlateCarree(), lons, lats)
        xmin = np.min(device_coords[:, 0])
        xmax = np.max(device_coords[:, 0])
        ymin = np.min(device_coords[:, 1])
        ymax = np.max(device_coords[:, 1])

        # blank off data past the bounding latitude
        pts = np.where(device_coords[:, 0] >= 0.0)
        xpts = np.append(
            device_coords[:, 0][pts], np.zeros(np.size(pts)) + xmax
        )
        ypts = np.append(
            device_coords[:, 1][pts], device_coords[:, 1][pts][::-1]
        )
        mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)

        xpts = np.append(
            np.zeros(np.size(pts)) + xmin, -1.0 * device_coords[:, 0][pts]
        )
        ypts = np.append(
            device_coords[:, 1][pts], device_coords[:, 1][pts][::-1]
        )
        mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)

        # Turn off map outside the cicular plot area
        # mymap.outline_patch.set_visible(False)
        mymap.set_frame_on(False)

        # Draw a line around the bounding latitude
        lons = np.arange(361)
        lats = np.zeros(np.size(lons)) + boundinglat
        device_coords = proj.transform_points(ccrs.PlateCarree(), lons, lats)
        mymap.plot(
            device_coords[:, 0],
            device_coords[:, 1],
            color="k",
            zorder=100,
            clip_on=False,
        )

        # Modify xlim and ylim values as the default values clip the
        # plot slightly
        xmax = np.max(np.abs(mymap.set_xlim(None)))
        mymap.set_xlim((-xmax, xmax), emit=False)
        ymax = np.max(np.abs(mymap.set_ylim(None)))
        mymap.set_ylim((-ymax, ymax), emit=False)

    # Lambert conformal
    if plotvars.proj == "lcc":
        lon_0 = plotvars.lonmin + (plotvars.lonmax - plotvars.lonmin) / 2.0
        lat_0 = plotvars.latmin + (plotvars.latmax - plotvars.latmin) / 2.0

        mymap = plotvars.mymap
        standard_parallels = [33, 45]
        if latmin <= 0 and latmax <= 0:
            standard_parallels = [-45, -33]
        proj = ccrs.LambertConformal(
            central_longitude=lon_0,
            central_latitude=lat_0,
            cutoff=40,
            standard_parallels=standard_parallels,
        )

        lonmin = plotvars.lonmin
        lonmax = plotvars.lonmax
        latmin = plotvars.latmin
        latmax = plotvars.latmax

        # Modify xlim and ylim values as the default values clip the
        # plot slightly
        xmin = mymap.set_xlim(None)[0]
        xmax = mymap.set_xlim(None)[1]
        ymin = mymap.set_ylim(None)[0]
        ymax = mymap.set_ylim(None)[1]

        mymap.set_ylim(ymin * 1.05, ymax, emit=False)
        mymap.set_ylim(None)

        lons = np.arange(lonmax - lonmin + 1) + lonmin
        lats = np.arange(latmax - latmin + 1) + latmin
        verts = []
        for lat in lats:
            verts.append([lonmin, lat])
        for lon in lons:
            verts.append([lon, latmax])
        for lat in lats[::-1]:
            verts.append([lonmax, lat])
        for lon in lons[::-1]:
            verts.append([lon, latmin])

        # Mask left and right of plot
        lats = np.arange(latmax - latmin + 1) + latmin
        lons = np.zeros(np.size(lats)) + lonmin
        device_coords = proj.transform_points(ccrs.PlateCarree(), lons, lats)
        xmin = np.min(device_coords[:, 0])
        xmax = np.max(device_coords[:, 0])
        if lat_0 > 0:
            ymin = np.min(device_coords[:, 1])
            ymax = np.max(device_coords[:, 1])
        else:
            ymin = np.max(device_coords[:, 1])
            ymax = np.min(device_coords[:, 1])

        # Left
        mymap.fill(
            [xmin, xmin, xmax, xmin],
            [ymin, ymax, ymax, ymin],
            alpha=1.0,
            color="w",
            zorder=100,
        )
        mymap.plot(
            [xmin, xmax], [ymin, ymax], color="k", zorder=101, clip_on=False
        )

        # Right
        mymap.fill(
            [-xmin, -xmin, -xmax, -xmin],
            [ymin, ymax, ymax, ymin],
            alpha=1.0,
            color="w",
            zorder=100,
        )
        mymap.plot(
            [-xmin, -xmax], [ymin, ymax], color="k", zorder=101, clip_on=False
        )

        # Upper
        lons = np.arange(lonmax - lonmin + 1) + lonmin
        lats = np.zeros(np.size(lons)) + latmax
        device_coords = proj.transform_points(ccrs.PlateCarree(), lons, lats)
        ymax = np.max(device_coords[:, 1])

        xpts = np.append(device_coords[:, 0], device_coords[:, 0][::-1])
        ypts = np.append(device_coords[:, 1], np.zeros(np.size(lons)) + ymax)

        mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)
        mymap.plot(
            device_coords[:, 0],
            device_coords[:, 1],
            color="k",
            zorder=101,
            clip_on=False,
        )

        # Lower
        lons = np.arange(lonmax - lonmin + 1) + lonmin
        lats = np.zeros(np.size(lons)) + latmin
        device_coords = proj.transform_points(ccrs.PlateCarree(), lons, lats)
        ymin = np.min(device_coords[:, 1]) * 1.05

        xpts = np.append(device_coords[:, 0], device_coords[:, 0][::-1])
        ypts = np.append(device_coords[:, 1], np.zeros(np.size(lons)) + ymin)

        mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)
        mymap.plot(
            device_coords[:, 0],
            device_coords[:, 1],
            color="k",
            zorder=101,
            clip_on=False,
        )

        # Turn off drawing of the rectangular box around the plot
        mymap.set_frame_on(False)

        if lat_0 < 0:
            lons = np.arange(lonmax - lonmin + 1) + lonmin
            lats = np.zeros(np.size(lons)) + latmax
            device_coords = proj.transform_points(
                ccrs.PlateCarree(), lons, lats
            )
            xmin = np.min(device_coords[:, 0])
            xmax = np.max(device_coords[:, 0])

            lons = np.arange(lonmax - lonmin + 1) + lonmin
            lats = np.zeros(np.size(lons)) + latmin
            device_coords = proj.transform_points(
                ccrs.PlateCarree(), lons, lats
            )
            ymax = np.min(device_coords[:, 1])
            ymin = ymax * 1.1

            xpts = [xmin, xmax, xmax, xmin, xmin]
            ypts = [ymin, ymin, ymax, ymax, ymin]
            mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)

        # Draw longitudes and latitudes if requested
        fs = plotvars.axis_label_fontsize
        fw = plotvars.axis_label_fontweight
        if axes and xaxis:
            if xticks is None:
                map_xticks, map_xticklabels = _mapaxis(
                    min=plotvars.lonmin, max=plotvars.lonmax, type=1
                )
            else:
                map_xticks = xticks
                if xticklabels is None:
                    map_xticklabels = xticks
                else:
                    map_xticklabels = xticklabels

            if axes and xaxis:
                lats = np.arange(latmax - latmin + 1) + latmin
                for tick in np.arange(np.size(map_xticks)):
                    lons = np.zeros(np.size(lats)) + map_xticks[tick]
                    device_coords = proj.transform_points(
                        ccrs.PlateCarree(), lons, lats
                    )
                    mymap.plot(
                        device_coords[:, 0],
                        device_coords[:, 1],
                        linewidth=plotvars.grid_thickness,
                        linestyle=plotvars.grid_linestyle,
                        color=plotvars.grid_colour,
                        zorder=101,
                    )

                    latpt = latmin - 3
                    if lat_0 < 0:
                        latpt = latmax + 1
                    device_coords = proj.transform_point(
                        map_xticks[tick], latpt, ccrs.PlateCarree()
                    )
                    mymap.text(
                        device_coords[0],
                        device_coords[1],
                        map_xticklabels[tick],
                        horizontalalignment="center",
                        fontsize=fs,
                        fontweight=fw,
                        zorder=101,
                    )

        if yticks is None:
            map_yticks, map_yticklabels = _mapaxis(
                min=plotvars.latmin, max=plotvars.latmax, type=2
            )
        else:
            map_yticks = yticks
            if yticklabels is None:
                map_yticklabels = yticks
            else:
                map_yticklabels = yticklabels

        if axes and yaxis:
            lons = np.arange(lonmax - lonmin + 1) + lonmin
            for tick in np.arange(np.size(map_yticks)):
                lats = np.zeros(np.size(lons)) + map_yticks[tick]
                device_coords = proj.transform_points(
                    ccrs.PlateCarree(), lons, lats
                )
                mymap.plot(
                    device_coords[:, 0],
                    device_coords[:, 1],
                    linewidth=plotvars.grid_thickness,
                    linestyle=plotvars.grid_linestyle,
                    color=plotvars.grid_colour,
                    zorder=101,
                )

                device_coords = proj.transform_point(
                    lonmin - 1, map_yticks[tick], ccrs.PlateCarree()
                )
                mymap.text(
                    device_coords[0],
                    device_coords[1],
                    map_yticklabels[tick],
                    horizontalalignment="right",
                    verticalalignment="center",
                    fontsize=fs,
                    fontweight=fw,
                    zorder=101,
                )

                device_coords = proj.transform_point(
                    lonmax + 1, map_yticks[tick], ccrs.PlateCarree()
                )
                mymap.text(
                    device_coords[0],
                    device_coords[1],
                    map_yticklabels[tick],
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=fs,
                    fontweight=fw,
                    zorder=101,
                )

    # UKCP grid
    if plotvars.proj == "UKCP" and plotvars.grid:
        # To plot grid, use xticks and yticks if provided for gridpoints else
        # plot all on defined grid with given grid spacing
        if xticks is None and yticks is None:
            map_grid()
        else:
            if plotvars.grid:
                plotvars.mymap.gridlines(
                    color=plotvars.grid_colour,
                    linewidth=plotvars.grid_thickness,
                    linestyle=plotvars.grid_linestyle,
                    xlocs=xticks,
                    ylocs=yticks,
                )
