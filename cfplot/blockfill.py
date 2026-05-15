"""Block-fill rendering helpers for contour workflows.

This module hosts contour-specific block-fill logic used by the refactored
contour path so it does not depend on cfplot.py internals.
"""

from __future__ import annotations

from copy import deepcopy

import cartopy.crs as ccrs
import cartopy.util as cartopy_util
import cf
import matplotlib.colors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PolyCollection

from .state import get_colour_scale_map, plotvars


def _max_ndecs_data(data: np.ndarray) -> int:
    """Return the maximum decimal places in the provided array."""
    ndecs_max = 1
    data_ndecs = np.zeros(len(data))
    for i in np.arange(len(data)):
        data_ndecs[i] = len(str(data[i]).split(".")[1])

    if max(data_ndecs) >= ndecs_max:
        if min(data_ndecs) < 10:
            pts = np.where(data_ndecs >= 10)
            data_ndecs[pts] = 0
            ndecs_max = int(max(data_ndecs))

    return ndecs_max


def _add_cyclic(field: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add cyclic point handling floating precision edge cases."""
    try:
        field, lons = cartopy_util.add_cyclic_point(field, lons)
    except Exception:
        ndecs_max = _max_ndecs_data(lons)
        lons = np.float64(lons).round(ndecs_max)
        field, lons = cartopy_util.add_cyclic_point(field, lons)

    return field, lons


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
    """Block-fill a field with coloured rectangles."""
    _ = (orca,)

    lonlat = False
    if plotvars.plot_type == 1:
        lonlat = True

    if single_fill_color is not None:
        white = False

    two_d = False
    if not isinstance(f, cf.Field):
        if np.ndim(x) == 2 and np.ndim(x) == 2:
            two_d = True

    plotargs = {}
    if lonlat:
        plotargs = {"transform": ccrs.PlateCarree()}

    if isinstance(f, cf.Field):
        field = f.array
    else:
        field = f

    levels = np.array(deepcopy(clevs)).astype("float")

    if single_fill_color is None:
        colmap = get_colour_scale_map()
        cmap = matplotlib.colors.ListedColormap(colmap)
        fill_colors = list(colmap)
        if plotvars.levels_extend in ["min", "both"]:
            cmap.set_under(plotvars.cs[0])
        if plotvars.levels_extend in ["max", "both"]:
            cmap.set_over(plotvars.cs[-1])

        under_index = None
        over_index = None
        if plotvars.levels_extend in ["min", "both"]:
            under_index = len(fill_colors)
            fill_colors.append(plotvars.cs[0])
        if plotvars.levels_extend in ["max", "both"]:
            over_index = len(fill_colors)
            fill_colors.append(plotvars.cs[-1])
    else:
        cols = single_fill_color
        cmap = matplotlib.colors.ListedColormap(cols)
        fill_colors = None
        under_index = None
        over_index = None

    colarr = np.zeros([np.shape(field)[0], np.shape(field)[1]]) - 1
    for i in np.arange(np.size(levels) - 1):
        lev = levels[i]
        pts = np.where(np.logical_and(field >= lev, field < levels[i + 1]))
        colarr[pts] = int(i)

    # Keep out-of-range values away from "white" so it is reserved for
    # actual missing/masked values only.
    if np.size(levels) >= 2:
        pts = np.where(field < levels[0])
        if np.size(pts) > 0:
            if under_index is not None:
                colarr[pts] = under_index
            else:
                colarr[pts] = 0

        pts = np.where(field >= levels[-1])
        if np.size(pts) > 0:
            if over_index is not None:
                colarr[pts] = over_index
            else:
                colarr[pts] = np.size(levels) - 2

    if isinstance(field, np.ma.MaskedArray):
        pts = np.ma.where(field.mask)
        if np.size(pts) > 0:
            colarr[pts] = -1

    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

    if isinstance(f, cf.Field):
        if f.ref("grid_mapping_name:transverse_mercator", default=False):
            lonlat = True

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
                xpts = x[0] - (x[1] - x[0]) / 2.0
                for ix in np.arange(np.size(x) - 1):
                    xpts = np.append(xpts, x[ix] + (x[ix + 1] - x[ix]) / 2.0)
                xpts = np.append(xpts, x[ix + 1] + (x[ix + 1] - x[ix]) / 2.0)

                ypts = y[0] - (y[1] - y[0]) / 2.0
                for iy in np.arange(np.size(y) - 1):
                    ypts = np.append(ypts, y[iy] + (y[iy + 1] - y[iy]) / 2.0)
                ypts = np.append(ypts, y[iy + 1] + (y[iy + 1] - y[iy]) / 2.0)

            if lonlat:
                upper_bound = ypts[-1]

                xpts = xpts[0:-1]
                ypts = ypts[0:-1]

                if plotvars.lonmin < np.nanmin(xpts):
                    xpts = xpts - 360
                if plotvars.lonmin > np.nanmax(xpts):
                    xpts = xpts + 360

                lonrange = np.nanmax(xpts) - np.nanmin(xpts)
                if lonrange < 360 and lonrange > 350:
                    field, xpts = _add_cyclic(field, xpts)

                right_bound = xpts[-1] + (xpts[-1] - xpts[-2])

                xpts = np.append(xpts, right_bound)
                ypts = np.append(ypts, upper_bound)

        if two_d:
            if fast:
                xpts = x
                ypts = y
            else:
                nx = np.shape(x)[1]
                ny = np.shape(x)[0]

                for ix in np.arange(nx):
                    for iy in np.arange(ny):
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

                        plotvars.mymap.add_patch(
                            mpatches.Polygon(
                                [
                                    [xpts[0], ypts[0]],
                                    [xpts[1], ypts[1]],
                                    [xpts[2], ypts[2]],
                                    [xpts[3], ypts[3]],
                                    [xpts[4], ypts[4]],
                                ],
                                facecolor=fill_colors[int(colarr[iy, ix])],
                                zorder=zorder,
                                transform=ccrs.PlateCarree(),
                            )
                        )

                return

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

    if transform:
        lonlat = True
    else:
        transform = ccrs.PlateCarree()

    if fast:
        if isinstance(clevs, int):
            norm = False

        if two_d:
            fixed_x = x.copy()
            for i, start in enumerate(np.argmax(np.abs(np.diff(x)) > 180, axis=1)):
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
                    plotvars.image = plotvars.plot.pcolormesh(xpts, ypts, field, cmap=cmap)
                else:
                    plotvars.image = plotvars.plot.pcolormesh(
                        xpts, ypts, field, cmap=cmap, norm=norm
                    )

    else:
        if plotvars.plot_type == 1 and plotvars.proj != "cyl":
            for i in np.unique(colarr[colarr >= 0]).astype(int):
                allverts = []
                xy_stack = np.column_stack(np.where(colarr == i))

                for pt in np.arange(np.shape(xy_stack)[0]):
                    ix = xy_stack[pt][1]
                    iy = xy_stack[pt][0]
                    lons = [xpts[ix], xpts[ix + 1], xpts[ix + 1], xpts[ix], xpts[ix]]
                    lats = [ypts[iy], ypts[iy], ypts[iy + 1], ypts[iy + 1], ypts[iy]]

                    verts = [
                        (lons[0], lats[0]),
                        (lons[1], lats[1]),
                        (lons[2], lats[2]),
                        (lons[3], lats[3]),
                        (lons[4], lats[4]),
                    ]

                    allverts.append(verts)

                if single_fill_color is None:
                    color = fill_colors[i]
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
            for i in np.unique(colarr[colarr >= 0]).astype(int):
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

                if single_fill_color is None:
                    color = fill_colors[i]
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
