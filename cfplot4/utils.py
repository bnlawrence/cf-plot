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


def shffl