from copy import deepcopy

import cartopy.crs as ccrs
import cartopy.util as cartopy_util
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plot
import numpy as np
from scipy.interpolate import griddata
import shapely.geometry as sgeom

import cf

from ..parameters import (
    gset,
    plotvars,
)
from ..mapping import (
    _mapaxis,
    map_grid,
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
