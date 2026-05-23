from copy import deepcopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cf
import numpy as np

from .colour import cscale
from .layout_runtime import gclose, gopen, gpos, gset
from .map_runtime import MapSet, _apply_map_axes, _apply_map_title
from .map_runtime import mapset
from .state import plotvars
from . import utility
from .validate import _check_data


def _set_map():
    MapSet(plotvars).ensure_map_axes()


def _plot_map_axes(
    *,
    axes=True,
    xaxis=True,
    yaxis=True,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    user_xlabel=None,
    user_ylabel=None,
    verbose=None,
):
    del verbose

    xlabel = user_xlabel
    ylabel = user_ylabel
    map_xticks = xticks
    map_yticks = yticks
    map_xticklabels = xticklabels
    map_yticklabels = yticklabels

    if not axes:
        map_xticks = []
        map_yticks = []
        xlabel = ""
        ylabel = ""
    else:
        if not xaxis:
            map_xticks = []
            map_xticklabels = []
            xlabel = ""
        if not yaxis:
            map_yticks = []
            map_yticklabels = []
            ylabel = ""

    _apply_map_axes(
        xticks=map_xticks,
        yticks=map_yticks,
        xlabel=xlabel,
        ylabel=ylabel,
        xticklabels=map_xticklabels,
        yticklabels=map_yticklabels,
    )

    if plotvars.proj in ("npstere", "spstere"):
        MapSet(plotvars).draw_polar_axes()


def _map_title(title):
    if plotvars.mymap is None:
        return

    _apply_map_title(
        mymap=plotvars.mymap,
        title=title,
        proj=plotvars.proj,
        boundinglat=plotvars.boundinglat,
        lon_0=plotvars.lon_0,
        lonmin=plotvars.lonmin,
        lonmax=plotvars.lonmax,
        latmin=plotvars.latmin,
        latmax=plotvars.latmax,
        title_fontsize=plotvars.title_fontsize,
        title_fontweight=plotvars.title_fontweight,
    )


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
    """Plot a streamplot to show fluid flow and 2D field gradients."""
    del zorder

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

    user_xlabel = xlabel
    user_ylabel = ylabel

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

    if isinstance(u, cf.Field):
        ndims = np.squeeze(u.data).ndim
        if ndims != 2:
            errstr = (
                "\n\ncfp.vect error need a 2 dimensonal u field to make vectors\n"
                f"received {ndims}"
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
        ) = utility.cf_data_assign(
            u, colorbar_title, proj=("rotated" if rotated_vect else plotvars.proj)
        )
        del xpole, ypole
    elif isinstance(u, cf.FieldList):
        raise TypeError("Can't plot a field list")
    else:
        _check_data(u, x, y)
        u_data = deepcopy(u)
        u_x = deepcopy(x)
        u_y = deepcopy(y)
        xlabel = ""
        ylabel = ""

    if isinstance(v, cf.Field):
        ndims = np.squeeze(v.data).ndim
        if ndims != 2:
            errstr = (
                "\n\ncfp.vect error need a 2 dimensonal v field to make vectors\n"
                f"received {ndims}"
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
        ) = utility.cf_data_assign(
            v, colorbar_title, proj=("rotated" if rotated_vect else plotvars.proj)
        )
        del v_x, v_y, xpole, ypole
    elif isinstance(v, cf.FieldList):
        raise TypeError("Can't plot a field list")
    else:
        _check_data(v, x, y)
        v_data = deepcopy(v)
        xlabel = ""
        ylabel = ""

    if user_xlabel is not None:
        xlabel = user_xlabel
    if user_ylabel is not None:
        ylabel = user_ylabel

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

    if plotvars.user_plot == 0:
        gopen(user_plot=0)

    if plotvars.rows > 1 or plotvars.columns > 1:
        if plotvars.gpos_called is False:
            gpos(1)

    if ptype is not None:
        plotvars.plot_type = ptype

    lonrange = np.nanmax(u_x) - np.nanmin(u_x)
    latrange = np.nanmax(u_y) - np.nanmin(u_y)

    if plotvars.plot_type == 1:
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

        mymap.streamplot(
            u_x,
            u_y,
            u_data,
            v_data,
            transform=ccrs.PlateCarree(),
            **plotargs,
        )

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

        if title is not None:
            _map_title(title)

    if plotvars.user_plot == 0:
        gset()
        cscale()
        gclose()

    if plotvars.user_mapset == 0:
        mapset()
        mapset(resolution=resolution_orig)
