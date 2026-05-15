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
        elif vproj == "spstere":
            proj = ccrs.SouthPolarStereo(central_longitude=pv.lon_0)
            lonmin = pv.lon_0 - 180
            lonmax = pv.lon_0 + 180.01
            latmin = -90
            latmax = pv.boundinglat
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

    def draw_polar_axes(self) -> None:
        """Draw graticule lines and longitude labels for polar stereographic plots.

        Replicates the legacy behaviour: latitude circles, longitude spokes,
        and longitude text labels placed just outside the bounding latitude.
        No latitude labels are drawn (matching legacy output).
        """
        pv = self.plotvars
        if pv.mymap is None:
            return
        vproj = pv.proj
        if vproj not in ("npstere", "spstere"):
            return

        mymap = pv.mymap
        boundinglat = pv.boundinglat
        lon_0 = pv.lon_0
        geodetic = ccrs.Geodetic()

        # --- latitude circles ---
        latvals = np.arange(5) * 30 - 60  # [-60, -30, 0, 30, 60]
        if vproj == "npstere":
            latvals = latvals[latvals >= boundinglat]
        else:
            latvals = latvals[latvals <= boundinglat]

        for lat in latvals:
            if abs(lat - boundinglat) > 1:
                lons_line = np.arange(361, dtype=float)
                lats_line = np.full(361, lat)
                mymap.plot(
                    lons_line, lats_line,
                    color=pv.grid_colour,
                    linewidth=pv.grid_thickness,
                    linestyle=pv.grid_linestyle,
                    transform=geodetic,
                )

        # --- longitude spokes ---
        lonvals = np.arange(7) * 60  # [0, 60, 120, 180, 240, 300, 360]
        for lon in lonvals:
            if vproj == "npstere":
                lats_line = np.arange(90 - boundinglat) + boundinglat
            else:
                lats_line = np.arange(boundinglat + 91) - 90
            lons_line = np.full(lats_line.size, float(lon))
            mymap.plot(
                lons_line, lats_line,
                color=pv.grid_colour,
                linewidth=pv.grid_thickness,
                linestyle=pv.grid_linestyle,
                transform=geodetic,
            )

        # --- longitude labels ---
        if vproj == "npstere":
            polar_proj = ccrs.NorthPolarStereo(central_longitude=lon_0)
            pole = 90
            latrange = 90 - abs(boundinglat)
            latpt = boundinglat - latrange / 40.0
        else:
            polar_proj = ccrs.SouthPolarStereo(central_longitude=lon_0)
            pole = -90
            latrange = 90 - abs(boundinglat)
            latpt = boundinglat + latrange / 40.0

        axis_label_fontsize = getattr(pv, "axis_label_fontsize", 11)
        axis_label_fontweight = getattr(pv, "axis_label_fontweight", "normal")

        if axis_label_fontsize > 0:
            for lon in lonvals:
                # Build label the same way as legacy _mapaxis
                lon2 = np.mod(lon + 180, 360) - 180
                degsym = r"$\degree$" if getattr(pv, "degsym", False) else ""
                if lon2 < 0 and lon2 > -180:
                    label = str(abs(int(lon2))) + degsym + "W"
                elif lon2 > 0 and lon2 <= 180:
                    label = str(int(lon2)) + degsym + "E"
                elif lon2 == 0:
                    label = "0" + degsym
                else:  # 180
                    label = "180" + degsym

                lonr, latr = polar_proj.transform_point(
                    lon, latpt, ccrs.PlateCarree()
                )

                v_align = "center"
                if lonr < -1:
                    h_align = "right"
                elif lonr > 1:
                    h_align = "left"
                else:
                    h_align = "center"
                    if latr < 0:
                        v_align = "top"
                    else:
                        v_align = "bottom"

                mymap.text(
                    lonr, latr, label,
                    horizontalalignment=h_align,
                    verticalalignment=v_align,
                    fontsize=axis_label_fontsize,
                    fontweight=axis_label_fontweight,
                    zorder=101,
                )

        # Blank off corners to make the plot circular, then draw the bounding
        # latitude circle and adjust axes limits (mirrors legacy behaviour).
        lons_b = np.arange(360, dtype=float)
        lats_b = np.full(360, float(boundinglat))
        device_coords = polar_proj.transform_points(
            ccrs.PlateCarree(), lons_b, lats_b
        )
        xmax_b = np.max(device_coords[:, 0])
        xmin_b = np.min(device_coords[:, 0])

        pts = np.where(device_coords[:, 0] >= 0.0)
        xpts = np.append(
            device_coords[:, 0][pts], np.zeros(np.size(pts)) + xmax_b
        )
        ypts = np.append(
            device_coords[:, 1][pts], device_coords[:, 1][pts][::-1]
        )
        mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)

        xpts = np.append(
            np.zeros(np.size(pts)) + xmin_b, -1.0 * device_coords[:, 0][pts]
        )
        ypts = np.append(
            device_coords[:, 1][pts], device_coords[:, 1][pts][::-1]
        )
        mymap.fill(xpts, ypts, alpha=1.0, color="w", zorder=100)

        mymap.set_frame_on(False)

        # Draw the bounding latitude circle
        lons_circ = np.arange(361, dtype=float)
        lats_circ = np.full(361, float(boundinglat))
        circle_coords = polar_proj.transform_points(
            ccrs.PlateCarree(), lons_circ, lats_circ
        )
        mymap.plot(
            circle_coords[:, 0], circle_coords[:, 1],
            color="k", zorder=100, clip_on=False,
        )

        # Expand axes limits slightly so labels are not clipped
        xmax_ax = np.max(np.abs(mymap.set_xlim(None)))
        mymap.set_xlim((-xmax_ax, xmax_ax), emit=False)
        ymax_ax = np.max(np.abs(mymap.set_ylim(None)))
        mymap.set_ylim((-ymax_ax, ymax_ax), emit=False)
