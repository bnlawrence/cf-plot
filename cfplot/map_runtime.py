"""Contour-owned map setup/runtime helpers.

This module contains the map-state and map-axes operations used by the
refactored contour renderer so it no longer needs map setup from cfplot.py.
"""

from __future__ import annotations

import cartopy.crs as ccrs
import numpy as np

from .state import plotvars


class MapSet:
    """Stateful map setup for contour rendering."""

    def __init__(self, pvars=plotvars):
        self.plotvars = pvars

    def configure(
        self,
        *,
        lonmin: float | None = None,
        lonmax: float | None = None,
        latmin: float | None = None,
        latmax: float | None = None,
        proj: str = "cyl",
        boundinglat: float = 0,
        lon_0: float = 0,
        lat_0: float = 40,
        resolution: str = "110m",
        user_mapset: int = 1,
        aspect: str | float | None = None,
    ) -> None:
        """Set map plotting parameters in shared state."""
        pv = self.plotvars
        pv.resolution = resolution

        if (
            all(val is None for val in [lonmin, lonmax, latmin, latmax, aspect])
            and proj == "cyl"
        ):
            pv.lonmin = -180
            pv.lonmax = 180
            pv.latmin = -90
            pv.latmax = 90
            pv.proj = "cyl"
            pv.user_mapset = 0
            pv.aspect = "equal"
            pv.plot_xmin = None
            pv.plot_xmax = None
            pv.plot_ymin = None
            pv.plot_ymax = None
            return

        if aspect is None:
            aspect = "equal"
        pv.aspect = aspect

        if lonmin is None:
            lonmin = -180
        if lonmax is None:
            lonmax = 180
        if latmin is None:
            latmin = -80 if proj == "merc" else -90
        if latmax is None:
            latmax = 80 if proj == "merc" else 90

        if proj == "moll":
            lonmin = lon_0 - 180
            lonmax = lon_0 + 180

        pv.lonmin = lonmin
        pv.lonmax = lonmax
        pv.latmin = latmin
        pv.latmax = latmax
        pv.proj = proj
        pv.boundinglat = boundinglat
        pv.lon_0 = lon_0
        pv.lat_0 = lat_0
        pv.user_mapset = user_mapset

    def ensure_map_axes(self) -> None:
        """Create map axes in plotvars.mymap when not already present."""
        pv = self.plotvars
        if pv.mymap is not None:
            return

        if pv.plot is None:
            from .layout_runtime import ensure_map_viewport

            ensure_map_viewport()

        extent = True
        lonmin = pv.lonmin
        lonmax = pv.lonmax
        latmin = pv.latmin
        latmax = pv.latmax
        lon_diff = lonmax - lonmin
        lon_mid = lonmin + lon_diff / 2.0

        vproj = pv.proj

        if vproj in ("cyl", "merc") and lon_diff == 360.0:
            lonmax += 0.01

        if vproj == "cyl":
            proj = ccrs.PlateCarree(central_longitude=lon_mid)
        elif vproj == "merc":
            min_latitude = -80.0
            if lonmin > min_latitude:
                min_latitude = pv.lonmin
            max_latitude = 84.0
            if lonmax < max_latitude:
                max_latitude = pv.lonmax
            proj = ccrs.Mercator(
                central_longitude=pv.lon_0,
                min_latitude=min_latitude,
                max_latitude=max_latitude,
            )
        elif vproj == "npstere":
            proj = ccrs.NorthPolarStereo(central_longitude=pv.lon_0)
            lonmin = pv.lon_0 - 180
            lonmax = pv.lon_0 + 180.01
            latmin = pv.boundinglat
            latmax = 90
            extent = False
        elif vproj == "spstere":
            proj = ccrs.SouthPolarStereo(central_longitude=pv.lon_0)
            lonmin = pv.lon_0 - 180
            lonmax = pv.lonmax + 180.01
            latmin = -90
            latmax = pv.boundinglat
            extent = False
        elif vproj == "ortho":
            proj = ccrs.Orthographic(
                central_longitude=pv.lon_0, central_latitude=pv.lat_0
            )
            lonmin = pv.lon_0 - 180.0
            lonmax = pv.lon_0 + 180.01
            extent = False
        elif vproj == "moll":
            proj = ccrs.Mollweide(central_longitude=pv.lon_0)
            lonmin = pv.lon_0 - 180.0
            lonmax = pv.lon_0 + 180.01
            extent = False
        elif vproj == "robin":
            proj = ccrs.Robinson(central_longitude=pv.lon_0)
        elif vproj == "lcc":
            lon_0 = lonmin + (lonmax - lonmin) / 2.0
            lat_0 = latmin + (latmax - latmin) / 2.0
            cutoff = -40 if lat_0 > 0 else 40
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
            proj = ccrs.TransverseMercator()
        elif vproj == "TransverseMercator":
            proj = ccrs.TransverseMercator()
            lonmin = pv.lon_0 - 180.0
            lonmax = pv.lon_0 + 180.01
            extent = False
        elif vproj == "LambertCylindrical":
            proj = ccrs.LambertCylindrical()
        else:
            proj = ccrs.PlateCarree(central_longitude=lon_mid)

        if pv.plot_xmin:
            delta_x = pv.plot_xmax - pv.plot_xmin
            delta_y = pv.plot_ymax - pv.plot_ymin
            mymap = pv.master_plot.add_axes(
                [pv.plot_xmin, pv.plot_ymin, delta_x, delta_y], projection=proj
            )
        else:
            mymap = pv.master_plot.add_subplot(
                pv.rows, pv.columns, pv.pos, projection=proj
            )

        set_extent = True
        if vproj in ["OSGB", "EuroPP", "UKCP", "robin", "lcc"]:
            set_extent = False
        if extent and set_extent:
            mymap.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

        if vproj == "cyl":
            mymap.set_aspect(pv.aspect)
        elif vproj == "lcc":
            mymap.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
        elif vproj == "UKCP":
            mymap.set_extent([-11, 3, 49, 61], crs=ccrs.PlateCarree())
        elif vproj == "EuroPP":
            mymap.set_extent([-12, 25, 30, 75], crs=ccrs.PlateCarree())

        if pv.plot is not None:
            pv.plot.set_frame_on(False)
            pv.plot.set_xticks([])
            pv.plot.set_yticks([])

        pv.mymap = mymap

    def draw_grid(self) -> None:
        """Plot a graticule on the active map axes."""
        pv = self.plotvars
        if pv.mymap is None:
            return

        lons = np.arange((360 / pv.grid_x_spacing) + 1) * pv.grid_x_spacing
        lons = np.concatenate([lons - 360, lons])
        lats = np.arange((180 / pv.grid_y_spacing) + 1) * pv.grid_y_spacing - 90

        pv.mymap.gridlines(
            color=pv.grid_colour,
            linewidth=pv.grid_thickness,
            linestyle=pv.grid_linestyle,
            xlocs=lons,
            ylocs=lats,
            zorder=pv.grid_zorder,
        )
