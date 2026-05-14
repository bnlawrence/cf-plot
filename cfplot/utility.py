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
