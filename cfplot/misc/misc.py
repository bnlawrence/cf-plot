from copy import deepcopy

import cartopy.crs as ccrs
import cartopy.util as cartopy_util
import matplotlib
import numpy as np
from scipy.interpolate import griddata

from ..parameters import (
    gset,
    plotvars,
)


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
