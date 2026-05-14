"""Utility functions for plotting.

Pure utility functions with no global state dependencies.
These are designed to be used by any plotting module.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def ndecs(data: np.ndarray | list) -> int:
    """Find the maximum number of decimal places in an array.
    
    Data with more decimal places will determine the result.
    Used to format colorbar and line labels consistently.
    
    Parameters
    ----------
    data : array-like
        Input array of numeric values
        
    Returns
    -------
    int
        Maximum number of decimal places found
    """
    maxdecs = 0
    for value in data:
        parts = str(value).split(".")
        if len(parts) == 2:
            number_decs = len(parts[1])
            if number_decs > maxdecs:
                maxdecs = number_decs
    return maxdecs


def gvals(
    dmin: float | None = None,
    dmax: float | None = None,
    mystep: float | None = None,
    mod: bool = True,
) -> tuple[np.ndarray, int]:
    """Generate sensible tick values between two limits.
    
    Works out appropriate step size and generates values,
    optionally scaling with a power-of-10 multiplier.
    Used for contour levels and axis labelling.
    
    Parameters
    ----------
    dmin : float
        Minimum value
    dmax : float
        Maximum value
    mystep : float, optional
        Use this step instead of auto-calculating
    mod : bool
        If True, apply multiplier for small/large ranges
        
    Returns
    -------
    vals : ndarray
        Array of tick values
    mult : int
        Multiplier exponent (10^mult) applied to values
    """
    # Copies of inputs as these might be changed
    dmin1 = deepcopy(dmin)
    dmax1 = deepcopy(dmax)

    # Swap if dmin1 > dmax1
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
            errstr = "\n\n cfp.gvals - no valid step values found \n\n"
            errstr += "cfp.gvals(" + str(dmin1) + "," + str(dmax1) + ")\n\n"
            raise ValueError(errstr)

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

        # Round off decimal numbers
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


def mapaxis(
    min_val: float | None = None,
    max_val: float | None = None,
    axis_type: int | None = None,
    degsym: bool = True,
) -> tuple[list, list]:
    """Generate longitude or latitude axis ticks and labels.
    
    Works out sensible tick marks and labels for geographic axes.
    
    Parameters
    ----------
    min_val : float
        Minimum axis value
    max_val : float
        Maximum axis value
    axis_type : int
        1 = longitude, 2 = latitude
    degsym : bool
        If True, use degree symbol in labels
        
    Returns
    -------
    ticks : list
        Tick positions
    labels : list
        Tick labels
    """
    degsym_str = r"$\degree$" if degsym else ""

    if axis_type == 1:
        # Longitude
        lonmin = min_val
        lonmax = max_val
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
                    lonlabels.append(str(abs(int(lon2))) + degsym_str + "W")
            if lon2 > 0 and lon2 <= 180:
                lonlabels.append(str(int(lon2)) + degsym_str + "E")
            if lon2 == 0:
                lonlabels.append("0" + degsym_str)
            if lon == 180 or lon == -180:
                lonlabels.append("180" + degsym_str)

        return (lonticks, lonlabels)

    if axis_type == 2:
        # Latitude
        latmin = min_val
        latmax = max_val
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
                latlabels.append(str(abs(int(lat))) + degsym_str + "S")
            if lat > 0:
                latlabels.append(str(int(lat)) + degsym_str + "N")
            if lat == 0:
                latlabels.append("0" + degsym_str)

        return (latticks, latlabels)

    return ([], [])


def fix_floats(data: list) -> list:
    """Fix numpy rounding issues where e.g. 0.4 becomes 0.3999999999.

    Returns data unchanged if any values contain an exponent ('e').
    """
    has_e = any("e" in str(val) for val in data)
    if has_e:
        return data

    data_ndecs = np.zeros(len(data))
    for i in np.arange(len(data)):
        data_ndecs[i] = len(str(float(data[i])).split(".")[1])

    if max(data_ndecs) >= 10:
        if min(data_ndecs) < 10:
            pts = np.where(data_ndecs >= 10)
            data_ndecs[pts] = 0
            ndecs_max = int(max(data_ndecs))
            for i in np.arange(len(data)):
                data[i] = round(data[i], ndecs_max)
        else:
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


def calculate_levels(
    field: np.ndarray,
    level_spacing: str = "linear",
    levels_step: Any | None = None,
    verbose: bool | None = None,
) -> tuple[np.ndarray, int, float]:
    """Calculate contour levels automatically from field data.

    Parameters
    ----------
    field : ndarray
        The data field to generate levels for.
    level_spacing : str
        One of 'linear', 'outlier', 'inspect', 'log', 'loglike'.
    levels_step : scalar or None
        If given, generate levels with this step size instead of auto.
    verbose : bool or None
        If True, print diagnostic messages.

    Returns
    -------
    clevs : ndarray
        Array of contour levels.
    mult : int
        Multiplier exponent applied (10 ** mult).
    fmult : float
        Inverse multiplier (10 ** -mult).
    """
    dmin = np.nanmin(field)
    dmax = np.nanmax(field)

    tight = True
    field2 = deepcopy(field)
    mult = 0
    fmult = 1.0
    clevs: Any = []

    if levels_step is None:
        if verbose:
            print("calculate_levels - generating automatic contour levels")

        if level_spacing in ("outlier", "inspect"):
            hist = np.histogram(field, 100)[0]
            pts_arr = np.size(field)
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

            clevs, mult = gvals(dmin=dmin, dmax=dmax)
            fmult = 10**-mult
            tight = False

        if level_spacing == "linear":
            if isinstance(np.ma.min(dmin), np.ma.core.MaskedConstant) or isinstance(
                np.ma.min(dmax), np.ma.core.MaskedConstant
            ):
                if verbose:
                    print(
                        "calculate_levels warning - data is entirely masked; "
                        "setting levels to 0 and 0.1"
                    )
                dmin = 0.0
                dmax = 0.1

            clevs, mult = gvals(dmin=dmin, dmax=dmax)
            fmult = 10**-mult
            tight = False

        if level_spacing in ("log", "loglike"):
            if dmin < 0.0 and dmax < 0.0:
                dmin1 = abs(dmax)
                dmax1 = abs(dmin)
            elif dmin > 0.0 and dmax > 0.0:
                dmin1 = abs(dmin)
                dmax1 = abs(dmax)
            else:
                dmax1 = max(abs(dmin), dmax)
                pts_neg = np.where(field < 0.0)
                close_below = np.max(field[pts_neg])
                pts_pos = np.where(field > 0.0)
                close_above = np.min(field[pts_pos])
                dmin1 = min(abs(close_below), close_above)

            if level_spacing == "log":
                clevs = []
                for i in np.arange(31):
                    val = 10 ** (i - 30.0)
                    clevs.append("{:.0e}".format(val))
            else:
                clevs = []
                for i in np.arange(61):
                    val = 10 ** (i - 30.0)
                    clevs.append("{:.0e}".format(val))
                    clevs.append("{:.0e}".format(val * 2))
                    clevs.append("{:.0e}".format(val * 5))

            clevs = np.float64(clevs)
            pts = np.where(np.logical_and(clevs >= abs(dmin1), clevs <= abs(dmax1)))
            clevs = clevs[pts]

            if dmin < 0.0 and dmax < 0.0:
                clevs = -1.0 * clevs[::-1]
            if dmin <= 0.0 and dmax >= 0.0:
                clevs = np.concatenate([-1.0 * clevs[::-1], [0.0], clevs])

    else:
        if verbose:
            print("calculate_levels - using specified step to generate contour levels")

        step = levels_step
        if isinstance(step, int):
            dmin = int(dmin)
            dmax = int(dmax)

        clevs_list = []
        if dmin < 0:
            clevs_list = list((np.arange(-1 * dmin / step + 1) * -step)[::-1])
        if dmax > 0:
            pos = list(np.arange(dmax / step + 1) * step)
            if len(clevs_list) > 0:
                clevs_list = list(clevs_list[:-1]) + pos
            else:
                clevs_list = pos
        clevs = np.array(clevs_list)
        if isinstance(step, int):
            clevs = clevs.astype(int)

    # Remove out-of-range values if tight mode
    if tight:
        pts = np.where(np.logical_and(clevs >= dmin, clevs <= dmax))
        clevs = clevs[pts]

    # Ensure at least two levels
    clevs = list(clevs)
    if len(clevs) < 2:
        clevs.append(clevs[0] + 0.001 if clevs else 0.001)

    # Fix floating-point rounding noise
    if isinstance(clevs[0], float):
        clevs = fix_floats(clevs)

    return (np.asarray(clevs), mult, fmult)


def timeaxis(
    dtimes: Any,
    user_gset: int = 0,
    xmin: Any = None,
    xmax: Any = None,
    ymin: Any = None,
    ymax: Any = None,
    tspace_year: int | None = None,
    tspace_hour: int | None = None,
    tspace_day: int | None = None,
) -> tuple[list, list, str]:
    """Calculate time axis ticks and labels for a CF time coordinate.

    Parameters
    ----------
    dtimes : cf time coordinate
        The time dimension of the CF field.
    user_gset : int
        Non-zero if the user has set axis limits via gset.
    xmin, xmax, ymin, ymax : scalar or str or None
        User-specified axis limits (possibly date strings).
    tspace_year, tspace_hour, tspace_day : int or None
        Override auto-calculated spacing for year/hour/day axes.

    Returns
    -------
    time_ticks : list
    time_labels : list
    axis_label : str
    """
    import cf as _cf

    time_units = dtimes.Units
    time_ticks: list = []
    time_labels: list = []
    axis_label = "Time"

    yearmin = min(dtimes.year.array)
    yearmax = max(dtimes.year.array)
    tmin = min(dtimes.dtarray)
    tmax = max(dtimes.dtarray)
    calendar = getattr(dtimes, "calendar", "standard")

    if user_gset != 0:
        if isinstance(xmin, str):
            t = _cf.Data(_cf.dt(xmin), units=time_units, calendar=calendar)
            yearmin = int(t.year)
            t = _cf.Data(_cf.dt(xmax), units=time_units, calendar=calendar)
            yearmax = int(t.year)
            tmin = _cf.dt(xmin, calendar=calendar)
            tmax = _cf.dt(xmax, calendar=calendar)
        if isinstance(ymin, str):
            t = _cf.Data(_cf.dt(ymin), units=time_units, calendar=calendar)
            yearmin = int(t.year)
            t = _cf.Data(_cf.dt(ymax), units=time_units, calendar=calendar)
            yearmax = int(t.year)
            tmin = _cf.dt(ymin, calendar=calendar)
            tmax = _cf.dt(ymax, calendar=calendar)

    # Years
    span = yearmax - yearmin
    if span > 4 and span < 3000:
        axis_label = "Time (year)"
        if span <= 15:
            step = 1
        elif span <= 30:
            step = 2
        elif span <= 60:
            step = 5
        elif span <= 160:
            step = 10
        elif span <= 300:
            step = 20
        elif span <= 600:
            step = 50
        elif span <= 1300:
            step = 100
        else:
            step = 200

        if tspace_year is not None:
            step = tspace_year

        years = np.arange(yearmax / step + 2) * step
        tvals = years[np.where((years >= yearmin) & (years <= yearmax))]

        if np.size(tvals) < 2:
            tvals = gvals(dmin=yearmin, dmax=yearmax)[0]

        for year in tvals:
            time_ticks.append(
                np.min(
                    _cf.Data(
                        _cf.dt(f"{int(year)}-01-01 00:00:00"),
                        units=time_units,
                        calendar=calendar,
                    ).array
                )
            )
            time_labels.append(str(int(year)))

    # Months
    if yearmax - yearmin <= 4:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        tsteps = 0
        for year in np.arange(yearmax - yearmin + 1) + yearmin:
            for month in np.arange(12):
                mytime = _cf.dt(f"{year}-{month + 1}-01 00:00:00", calendar=calendar)
                if mytime >= tmin and mytime <= tmax:
                    tsteps += 1

        mvals = np.arange(12) if tsteps < 17 else np.arange(4) * 3

        for year in np.arange(yearmax - yearmin + 1) + yearmin:
            for month in mvals:
                mytime = _cf.dt(f"{year}-{month + 1}-01 00:00:00", calendar=calendar)
                if mytime >= tmin and mytime <= tmax:
                    time_ticks.append(
                        np.min(
                            _cf.Data(mytime, units=time_units, calendar=calendar).array
                        )
                    )
                    time_labels.append(str(months[month]) + " " + str(int(year)))

    # Days and hours
    if np.size(time_ticks) <= 2:
        myday = _cf.dt(int(tmin.year), int(tmin.month), int(tmin.day), calendar=calendar)
        not_found = 0
        hour_counter = 0
        span = 0
        while not_found <= 48:
            mydate = _cf.Data(myday, dtimes.Units) + _cf.Data(hour_counter, "hour")
            if mydate >= tmin and mydate <= tmax:
                span += 1
            else:
                not_found += 1
            hour_counter += 1

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
        if tspace_hour is not None:
            step = tspace_hour
        if tspace_day is not None:
            step = tspace_day * 24

        not_found = 0
        hour_counter = 0
        axis_label = "Time (hour)"
        if span >= 24:
            axis_label = "Time"
        time_ticks = []
        time_labels = []

        while not_found <= 48:
            mytime = _cf.Data(myday, dtimes.Units) + _cf.Data(hour_counter, "hour")
            if mytime >= tmin and mytime <= tmax:
                time_ticks.append(np.min(mytime.array))
                label = f"{mytime.year}-{mytime.month}-{mytime.day}"
                if hour_counter / 24 != int(hour_counter / 24):
                    label += f" {mytime.hour}:00:00"
                time_labels.append(label)
            else:
                not_found += 1
            hour_counter += step

    return (time_ticks, time_labels, axis_label)
