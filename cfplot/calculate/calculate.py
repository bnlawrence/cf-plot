from copy import deepcopy

import numpy as np

from ..parameters import plotvars
from ..utils import _gvals, fix_floats


def calculate_levels(field=None, level_spacing=None, verbose=None):
    """Calculate contour levels."""

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
