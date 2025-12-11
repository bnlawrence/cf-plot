import os
import sys

import matplotlib
import numpy as np
from scipy import interpolate


# TODO remove these and use the matploltib versions! Why on earth would
# they be hard-coded in cf-plot???
#
# Default colour scales
# cscale1 is a differential data scale - blue to red
cscale1 = [
    "#0a3278",
    "#0f4ba5",
    "#1e6ec8",
    "#3ca0f0",
    "#50b4fa",
    "#82d2ff",
    "#a0f0ff",
    "#c8faff",
    "#e6ffff",
    "#fffadc",
    "#ffe878",
    "#ffc03c",
    "#ffa000",
    "#ff6000",
    "#ff3200",
    "#e11400",
    "#c00000",
    "#a50000",
]
# viridis is a continuous data scale - blue, green, yellow
viridis = [
    "#440154",
    "#440255",
    "#440357",
    "#450558",
    "#45065a",
    "#45085b",
    "#46095c",
    "#460b5e",
    "#460c5f",
    "#460e61",
    "#470f62",
    "#471163",
    "#471265",
    "#471466",
    "#471567",
    "#471669",
    "#47186a",
    "#48196b",
    "#481a6c",
    "#481c6e",
    "#481d6f",
    "#481e70",
    "#482071",
    "#482172",
    "#482273",
    "#482374",
    "#472575",
    "#472676",
    "#472777",
    "#472878",
    "#472a79",
    "#472b7a",
    "#472c7b",
    "#462d7c",
    "#462f7c",
    "#46307d",
    "#46317e",
    "#45327f",
    "#45347f",
    "#453580",
    "#453681",
    "#443781",
    "#443982",
    "#433a83",
    "#433b83",
    "#433c84",
    "#423d84",
    "#423e85",
    "#424085",
    "#414186",
    "#414286",
    "#404387",
    "#404487",
    "#3f4587",
    "#3f4788",
    "#3e4888",
    "#3e4989",
    "#3d4a89",
    "#3d4b89",
    "#3d4c89",
    "#3c4d8a",
    "#3c4e8a",
    "#3b508a",
    "#3b518a",
    "#3a528b",
    "#3a538b",
    "#39548b",
    "#39558b",
    "#38568b",
    "#38578c",
    "#37588c",
    "#37598c",
    "#365a8c",
    "#365b8c",
    "#355c8c",
    "#355d8c",
    "#345e8d",
    "#345f8d",
    "#33608d",
    "#33618d",
    "#32628d",
    "#32638d",
    "#31648d",
    "#31658d",
    "#31668d",
    "#30678d",
    "#30688d",
    "#2f698d",
    "#2f6a8d",
    "#2e6b8e",
    "#2e6c8e",
    "#2e6d8e",
    "#2d6e8e",
    "#2d6f8e",
    "#2c708e",
    "#2c718e",
    "#2c728e",
    "#2b738e",
    "#2b748e",
    "#2a758e",
    "#2a768e",
    "#2a778e",
    "#29788e",
    "#29798e",
    "#287a8e",
    "#287a8e",
    "#287b8e",
    "#277c8e",
    "#277d8e",
    "#277e8e",
    "#267f8e",
    "#26808e",
    "#26818e",
    "#25828e",
    "#25838d",
    "#24848d",
    "#24858d",
    "#24868d",
    "#23878d",
    "#23888d",
    "#23898d",
    "#22898d",
    "#228a8d",
    "#228b8d",
    "#218c8d",
    "#218d8c",
    "#218e8c",
    "#208f8c",
    "#20908c",
    "#20918c",
    "#1f928c",
    "#1f938b",
    "#1f948b",
    "#1f958b",
    "#1f968b",
    "#1e978a",
    "#1e988a",
    "#1e998a",
    "#1e998a",
    "#1e9a89",
    "#1e9b89",
    "#1e9c89",
    "#1e9d88",
    "#1e9e88",
    "#1e9f88",
    "#1ea087",
    "#1fa187",
    "#1fa286",
    "#1fa386",
    "#20a485",
    "#20a585",
    "#21a685",
    "#21a784",
    "#22a784",
    "#23a883",
    "#23a982",
    "#24aa82",
    "#25ab81",
    "#26ac81",
    "#27ad80",
    "#28ae7f",
    "#29af7f",
    "#2ab07e",
    "#2bb17d",
    "#2cb17d",
    "#2eb27c",
    "#2fb37b",
    "#30b47a",
    "#32b57a",
    "#33b679",
    "#35b778",
    "#36b877",
    "#38b976",
    "#39b976",
    "#3bba75",
    "#3dbb74",
    "#3ebc73",
    "#40bd72",
    "#42be71",
    "#44be70",
    "#45bf6f",
    "#47c06e",
    "#49c16d",
    "#4bc26c",
    "#4dc26b",
    "#4fc369",
    "#51c468",
    "#53c567",
    "#55c666",
    "#57c665",
    "#59c764",
    "#5bc862",
    "#5ec961",
    "#60c960",
    "#62ca5f",
    "#64cb5d",
    "#67cc5c",
    "#69cc5b",
    "#6bcd59",
    "#6dce58",
    "#70ce56",
    "#72cf55",
    "#74d054",
    "#77d052",
    "#79d151",
    "#7cd24f",
    "#7ed24e",
    "#81d34c",
    "#83d34b",
    "#86d449",
    "#88d547",
    "#8bd546",
    "#8dd644",
    "#90d643",
    "#92d741",
    "#95d73f",
    "#97d83e",
    "#9ad83c",
    "#9dd93a",
    "#9fd938",
    "#a2da37",
    "#a5da35",
    "#a7db33",
    "#aadb32",
    "#addc30",
    "#afdc2e",
    "#b2dd2c",
    "#b5dd2b",
    "#b7dd29",
    "#bade27",
    "#bdde26",
    "#bfdf24",
    "#c2df22",
    "#c5df21",
    "#c7e01f",
    "#cae01e",
    "#cde01d",
    "#cfe11c",
    "#d2e11b",
    "#d4e11a",
    "#d7e219",
    "#dae218",
    "#dce218",
    "#dfe318",
    "#e1e318",
    "#e4e318",
    "#e7e419",
    "#e9e419",
    "#ece41a",
    "#eee51b",
    "#f1e51c",
    "#f3e51e",
    "#f6e61f",
    "#f8e621",
    "#fae622",
    "#fde724",
]


# Read in defaults if they exist and overlay
# for contour options of fill, blockfill and lines
global_fill = True
global_lines = True
global_blockfill = False
global_degsym = False
global_viewer = "display"
defaults_file = os.path.expanduser("~") + "/.cfplot_defaults"
if os.path.exists(defaults_file):
    with open(defaults_file) as file:
        for line in file:
            vals = line.split(" ")
            com, val = vals
            v = False
            if val.strip() == "True":
                v = True
            if com == "blockfill":
                global_blockfill = v
            if com == "lines":
                global_lines = v
            if com == "fill":
                global_fill = v
            if com == "degsym":
                global_degsym = v
            if com == "viewer":
                global_viewer = val.strip()


"""Global plotting variables."""
# These are plotting variables that 'setvars' can adjust
setvars_defaults = {
    # TODO check docstring defaults are correct to match these
    #
    # Output graphics file saving and viewing
    "viewer": global_viewer,
    "file": None,
    # Output graphics file general config.
    "dpi": None,
    "tight": False,
    # 'tspace' related
    # TODO clarify what tspace is
    "tspace_year": None,
    "tspace_month": None,
    "tspace_day": None,
    "tspace_hour": None,
    # 2. Tick labelling
    "xtick_label_rotation": 0,
    "xtick_label_align": "center",
    "ytick_label_rotation": 0,
    "ytick_label_align": "right",
    # Font sizes
    "text_fontsize": 11,
    "axis_label_fontsize": 11,
    "colorbar_fontsize": 11,
    "title_fontsize": 15,
    "master_title_fontsize": 30,
    # TODO change for consistent name with the above, text_size -> fontsize
    "legend_text_size": 11,
    # Font weights
    "fontweight": "normal",
    "text_fontweight": "normal",
    "axis_label_fontweight": "normal",
    "colorbar_fontweight": "normal",
    "title_fontweight": "normal",
    "master_title_fontweight": "normal",
    # TODO change for consistent name with the above, text_weight -> fontweight
    "legend_text_weight": "normal",
    # Master title
    "master_title": None,
    "master_title_location": [0.5, 0.95],
    # Legend frame related
    "legend_frame": True,
    "legend_frame_edge_color": "k",
    "legend_frame_face_color": None,
    # Grid related
    "grid": True,
    "grid_x_spacing": 60,
    "grid_y_spacing": 30,
    "grid_zorder": 100,
    "grid_colour": "k",  # TODO API -> American spelling for consistency
    "grid_linestyle": "--",
    "grid_thickness": 1.0,
    # Rotated grid related
    # TODO SB why does rotated grid have separate config. like this?
    "rotated_grid_spacing": 10,
    "rotated_deg_spacing": 0.75,
    "rotated_continents": True,
    "rotated_grid": True,
    "rotated_grid_thickness": 1.0,
    "rotated_labels": True,
    # Feature additions
    "feature_zorder": 999,
    "land_color": None,
    "ocean_color": None,
    "lake_color": None,
    "continent_color": None,
    # Continent feature related
    "continent_thickness": None,
    "continent_linestyle": None,
    # Axis related
    "axis_width": None,
    "degsym": global_degsym,
    # Contour plot only
    "level_spacing": None,
    # Misc.
    "cs_uniform": True,  # make a uniform differential colour scale
}
# These are further plotting variables that are set globally and cannot
# be set by 'setvars'.
plotvars_defaults = {
    # 0. Top-level plotting config. eg. type, projection,
    "global_viewer": global_viewer,
    "plot_type": 1,
    "master_plot": None,
    "plot": None,
    "mymap": None,
    "proj": "cyl",
    "resolution": "110m",
    "norm": None,
    # 1. Maxima and minima
    # 1a) For lat and lon
    "lonmin": -180,
    "lonmax": 180,
    "latmin": -90,
    "latmax": 90,
    # 1b) for x and y
    "xmin": None,
    "xmax": None,
    "ymin": None,
    "ymax": None,
    # 1c) for the plot x and y boundaries
    "plot_xmin": None,
    "plot_xmax": None,
    "plot_ymin": None,
    "plot_ymax": None,
    # 1d) for the graph x and y boundaries
    "graph_xmin": None,
    "graph_xmax": None,
    "graph_ymin": None,
    "graph_ymax": None,
    # 2. Levels related
    "levels": None,
    "levels_min": None,
    "levels_max": None,
    "levels_step": None,
    "levels_extend": "both",
    # 3. Ticks and labels related
    "xticks": None,
    "xticklabels": None,
    "xlabel": None,
    "yticks": None,
    "yticklabels": None,
    "ylabel": None,
    # 4. Colour scale related
    "cs": cscale1,
    "cscale_flag": 0,
    # 5. Position related
    "pos": 1,
    "gpos_called": False,
    # 6. Lat and lon boundaries and centering
    "boundinglat": 0,
    "lon_0": 0,
    # 7. Log scale related
    "xlog": None,
    "ylog": None,
    # 8. Twin related
    "twinx": False,
    "twiny": False,
    # 9. Step related
    "xstep": None,
    "ystep": None,
    # 10. 'User' settings
    "user_mapset": 0,
    "user_gset": 0,
    "user_levs": 0,
    "user_plot": 0,
    "cs_user": "cscale1",  # TODO rename to 'user_cs' for consistency?
    # 11. Rows and columns
    "rows": 1,
    "columns": 1,
    # 12. Titles
    "title": None,
    "titles_con_called": False,
    # 13. General alignment
    "orientation": "landscape",
    "aspect": "equal",
}


class pvars:
    """Stores plotting variables in `cfp.plotvars`."""

    def __init__(self, **kwargs):
        """Initialize a new Pvars instance."""
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def __str__(self):
        """`x.__str__() <==> str(x)`"""
        a = None
        v = None
        out = [f"{a} = {repr(v)}"]
        for a, v in self.__dict__.items():
            return "\n".join(out)


allvars_defaults = {**setvars_defaults, **plotvars_defaults}
plotvars = pvars(**allvars_defaults)

# Check for iPython notebook inline
# and set the viewer to None if found
is_inline = "inline" in matplotlib.get_backend()
if is_inline:
    plotvars.viewer = None

# Check for OSX and if so use matplotlib for for the viewer
# Not all users will have ImageMagick installed / XQuartz running
# Users can still select this with cfp.setvars(viewer='display')
if sys.platform == "darwin":
    plotvars.global_viewer = "matplotlib"
    plotvars.viewer = "matplotlib"


def setvars(**kwargs):
    """
    | Set plotting variables and their defaults.
    |
    | file=None - output file name
    | title_fontsize=None - title fontsize, default=15
    | title_fontweight='normal' - title fontweight
    | text_fontsize='normal' - text font size, default=11
    | text_fontweight='normal' - text font weight
    | axis_label_fontsize=None - axis label fontsize, default=11
    | axis_label_fontweight='normal' - axis font weight
    | legend_text_size='11' - legend text size
    | legend_text_weight='normal' - legend text weight
    | colorbar_fontsize='11' - colorbar text size
    | colorbar_fontweight='normal' - colorbar font weight
    | legend_text_weight='normal' - legend text weight
    | master_title_fontsize=30 - master title font size
    | master_title_fontweight='normal' - master title font weight
    | continent_thickness=1.5 - default=1.5
    | continent_color='k' - default='k' (black)
    | continent_linestyle='solid' - default='k' (black)
    | viewer='display' - use ImageMagick display program
    |                    'matplotlib' to use image widget to view the picture
    | tspace_year=None - time axis spacing in years
    | tspace_month=None - time axis spacing in months
    | tspace_day=None - time axis spacing in days
    | tspace_hour=None - time axis spacing in hours
    | xtick_label_rotation=0 - rotation of xtick labels
    | xtick_label_align='center' - alignment of xtick labels
    | ytick_label_rotation=0 - rotation of ytick labels
    | ytick_label_align='right' - alignment of ytick labels

    | cs_uniform=True - make a uniform differential colour scale
    | master_title=None - master title text
    | master_title_location=[0.5,0.95] - master title location
    | dpi=None - dots per inch setting
    | land_color=None - land colour
    | ocean_color=None - ocean colour
    | lake_color=None - lake colour
    | feature_zorder - plotting zorder for above three features, default=999
    | rotated_grid_spacing=10 - rotated grid spacing in degrees
    | rotated_deg_spacing=0.75 - rotated grid spacing between graticule dots
    | rotated_deg_tkickness=1.0 - rotated grid thickness for longitude and
    |                             latitude lines
    | rotated_continents=True - draw rotated continents
    | rotated_grid=True - draw rotated grid
    | rotated_grid=1.0 - TODO, default 1.0
    | rotated_labels=True - draw rotated grid labels
    | legend_frame=True - draw a frame around a lineplot legend
    | legend_frame_edge_color='k' - color for the legend frame
    | legend_frame_face_color=None - color for the legend background
    | degsym=True - add degree symbol to longitude and latitude axis labels
    | axis_width=None - width of line for the axes
    | grid=True - draw grid
    | grid_x_spacing=60 - grid longitude spacing in degrees
    | grid_x_spacing=30 - grid latitude spacing in degrees
    | grid_colour='k' - grid colour
    | grid_linestyle='--' - grid line style
    | grid_zorder=100 - plotting order for the grid lines
    | grid_thickness=1.0 - grid thickness
    | tight=False - remove whitespace around the plot
    | level_spacing=None - default contour level spacing - takes 'linear',
    |                      'log', 'loglike', 'outlier' and 'inspect'
    |
    | Use setvars() to reset to the defaults
    |
    :Returns:
     name
    """
    # Set defaults first to ensure everything is assigned a valid value
    for def_var, def_value in setvars_defaults.items():
        setattr(plotvars, def_var, def_value)
    # Now override with anything the user specifies, which is unlikely to
    # be a large listing relative to the amount set as defaults above
    if kwargs:
        # TODO eventually add kwarg value validation e.g. type checking?
        for set_var, set_value in kwargs.items():

            # First ensure a given kwarg is a valid cf-plot setvars input i.e.
            # OK and meaningful to set on plotvars
            if set_var not in setvars_defaults:
                raise ValueError(
                    f"Unrecognised keyword argument for setvars: {set_var}"
                )

            setattr(plotvars, set_var, set_value)

    matplotlib.pyplot.ioff()


def mapset(
    lonmin=None,
    lonmax=None,
    latmin=None,
    latmax=None,
    proj="cyl",
    boundinglat=0,
    lon_0=0,
    lat_0=40,
    resolution="110m",
    user_mapset=1,
    aspect=None,
):
    """
    | Sets the mapping parameters.
    |
    | lonmin=lonmin - minimum longitude
    | lonmax=lonmax - maximum longitude
    | latmin=latmin - minimum latitude
    | latmax=latmax - maximum latitude
    | proj=proj - 'cyl' for cylindrical projection. 'npstere' or 'spstere'
    |      for northern hemisphere or southern hemisphere polar stereographic.
    |      ortho, merc, moll, robin and lcc are abreviations for orthographic,
    |      mercator, mollweide, robinson and lambert conformal projections
    |      'rotated' for contour plots on the native rotated grid.
    |
    | boundinglat=boundinglat - edge of the viewable latitudes in a
    |      stereographic plot
    | lon_0=0 - longitude centre of desired map domain in polar
    |           stereographic and orthogrphic plots
    | lat_0=40 - latitude centre of desired map domain in orthogrphic plots
    | resolution='110m' - the map resolution - can be one of '110m',
    | '50m' or '10m'.  '50m' means 1:50,000,000 and not 50 metre.
    | user_mapset=user_mapset - variable to indicate whether a user call
    |      to mapset has been made.
    |
    | The default map plotting projection is the cyclindrical equidistant
    | projection from -180 to 180 in longitude and -90 to 90 in latitude.
    | To change the map view in this projection to over the United Kingdom,
    | for example, you would use
    | mapset(lonmin=-6, lonmax=3, latmin=50, latmax=60)
    | or
    | mapset(-6, 3, 50, 60)
    |
    | The limits are -360 to 720 in longitude so to look at the equatorial
    | Pacific you could use
    | mapset(lonmin=90, lonmax=300, latmin=-30, latmax=30)
    | or
    | mapset(lonmin=-270, lonmax=-60, latmin=-30, latmax=30)
    |
    | The default setting for the cylindrical projection is for 1 degree of
    | longitude to have the same size as one degree of latitude.  When plotting
    | a smaller map setting aspect='auto' turns this off and the map fills the
    | plot area. Setting aspect to a number a circle will be stretched such
    | that the height is num times the width. aspect=1 is the same as
    | aspect='equal'.
    |
    | The proj parameter accepts 'npstere' and 'spstere' for northern
    | hemisphere or southern hemisphere polar stereographic projections.
    | In addition to these the boundinglat parameter sets the edge of the
    | viewable latitudes and lon_0 sets the centre of desired map domain.
    |
    |
    |
    | Map settings are persistent until a new call to mapset is made. To
    | reset to the default map settings use mapset().

    :Returns:
     None
    """

    # Set the continent resolution
    plotvars.resolution = resolution

    if (
        all(val is None for val in [lonmin, lonmax, latmin, latmax, aspect])
        and proj == "cyl"
    ):
        plotvars.lonmin = -180
        plotvars.lonmax = 180
        plotvars.latmin = -90
        plotvars.latmax = 90
        plotvars.proj = "cyl"
        plotvars.user_mapset = 0
        plotvars.aspect = "equal"

        plotvars.plot_xmin = None
        plotvars.plot_xmax = None
        plotvars.plot_ymin = None
        plotvars.plot_ymax = None

        return

    # Set the aspect ratio
    if aspect is None:
        aspect = "equal"
    plotvars.aspect = aspect

    if lonmin is None:
        lonmin = -180
    if lonmax is None:
        lonmax = 180
    if latmin is None:
        latmin = -90
        if proj == "merc":
            latmin = -80
    if latmax is None:
        latmax = 90
        if proj == "merc":
            latmax = 80

    if proj == "moll":
        lonmin = lon_0 - 180
        lonmax = lon_0 + 180

    plotvars.lonmin = lonmin
    plotvars.lonmax = lonmax
    plotvars.latmin = latmin
    plotvars.latmax = latmax
    plotvars.proj = proj
    plotvars.boundinglat = boundinglat
    plotvars.lon_0 = lon_0
    plotvars.lat_0 = lat_0
    plotvars.user_mapset = user_mapset


def levs(min=None, max=None, step=None, manual=None, extend="both"):
    """
    | Manually sets the contour levels.

    | min=min - minimum level
    | max=max - maximum level
    | step=step - step between levels
    | manual= manual - set levels manually
    | extend='neither', 'both', 'min', or 'max' - colour bar limit extensions

    | Use the levs command when a predefined set of levels is required. The
    | min, max and step parameters can be used to define a set of levels.
    | These can take integer or floating point numbers. If the min and max
    | are specified then a step also needs to be specified.

    | If just the step is specified then cf-plot will internally try to
    | define a reasonable set of levels.

    | If colour filled contours are plotted then the default is to extend
    | the minimum and maximum contours coloured for out of range values
    | - extend='both'.

    | Once a user call is made to levs the levels are persistent.
    | i.e. the next plot will use the same set of levels.
    | Use levs() to reset to undefined levels.

    :Returns:
     None

    """

    if all(val is not None for val in [min, max]) and step is None:
        print(
            "\ncfp.levs error: when the min and max are specified "
            "a step also needs to be specified\n"
        )
        return

    if all(val is None for val in [min, max, step, manual]):
        plotvars.levels = None
        plotvars.levels_min = None
        plotvars.levels_max = None
        plotvars.levels_step = None
        plotvars.levels_extend = "both"
        plotvars.norm = None
        plotvars.user_levs = 0
        return

    if manual is not None:
        plotvars.levels = np.array(manual)
        plotvars.levels_min = None
        plotvars.levels_max = None
        plotvars.levels_step = None
        # Set the normalization object as we are using potentially unevenly
        # spaced levels
        ncolors = np.size(plotvars.levels)
        if extend == "both" or extend == "max":
            ncolors = ncolors - 1
        plotvars.norm = matplotlib.colors.BoundaryNorm(
            boundaries=plotvars.levels, ncolors=ncolors
        )
        plotvars.user_levs = 1
    else:
        if all(val is not None for val in [min, max, step]):
            plotvars.levels_min = min
            plotvars.levels_max = max
            plotvars.levels_step = step
            plotvars.norm = None
            if all(isinstance(item, int) for item in [min, max, step]):
                lstep = step * 1e-10
                levs = np.arange(min, max + lstep, step, dtype=np.float64)
                levs = ((levs * 1e10).astype(np.int64)).astype(np.float64)
                levs = (levs / 1e10).astype(np.int64)
                plotvars.levels = levs
            else:
                lstep = step * 1e-10
                levs = np.arange(min, max + lstep, step, dtype=np.float64)
                levs = (levs * 1e10).astype(np.int64).astype(np.float64)
                levs = levs / 1e10
                plotvars.levels = levs
            plotvars.user_levs = 1

            # Check for spurious decimal places due to numeric representation
            # and fix if found
            for pt in np.arange(np.size(plotvars.levels)):
                ndecs = str(plotvars.levels[pt])[::-1].find(".")
                if ndecs > 7:
                    plotvars.levels[pt] = round(plotvars.levels[pt], 7)

    # If step only is set then reset user_levs to zero
    if step is not None and all(val is None for val in [min, max]):
        plotvars.user_levs = 0
        plotvars.levels = None
        plotvars.levels_step = step

    # Check extend has a proper value
    if extend not in ["neither", "min", "max", "both"]:
        errstr = "\n\n extend must be one of 'neither', 'min', 'max', 'both'\n"
        raise TypeError(errstr)
    plotvars.levels_extend = extend


def gset(
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xlog=False,
    ylog=False,
    user_gset=1,
    twinx=None,
    twiny=None,
):
    """
    | Set plot limits for all non-longitude-latitide plots.
    | xmin, xmax, ymin, ymax are all needed to set the plot limits.
    | Set xlog/ylog to True or 1 to get a log axis.
    |
    | xmin=None - x minimum
    | xmax=None - x maximum
    | ymin=None - y minimum
    | ymax=None - y maximum
    | xlog=False - log x
    | ylog=False - log y
    | twinx=None - set to True to make a twin y axis plot
    | twiny=None - set to True to make a twin x axis plot
    |
    | Once a user call is made to gset the plot limits are persistent.
    | i.e. the next plot will use the same set of plot limits.
    | Use gset() to reset to undefined plot limits i.e. the full range
    | of the data.
    |
    | To set date axes use date strings i.e.
    | cfp.gset(xmin = '1970-1-1', xmax = '1999-12-31', ymin = 285,
    |          ymax = 295)
    |
    | Note the correct date format is 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
    | anything else will give unexpected results.
    :Returns:
     None

    """

    plotvars.user_gset = user_gset

    if all(val is None for val in [xmin, xmax, ymin, ymax]):
        plotvars.xmin = None
        plotvars.xmax = None
        plotvars.ymin = None
        plotvars.ymax = None
        plotvars.xlog = False
        plotvars.ylog = False
        plotvars.twinx = False
        plotvars.twiny = False
        plotvars.user_gset = 0
        return

    bcount = 0
    for val in [xmin, xmax, ymin, ymax]:
        if val is None:
            bcount = bcount + 1

    if bcount != 0 and bcount != 4:
        errstr = (
            "gset error\n"
            "xmin, xmax, ymin, ymax all need to be passed to gset\n"
            "to set the plot limits\n"
        )
        raise Warning(errstr)

    plotvars.xmin = xmin
    plotvars.xmax = xmax
    plotvars.ymin = ymin
    plotvars.ymax = ymax
    plotvars.xlog = xlog
    plotvars.ylog = ylog

    # Check if any axes are time strings
    time_xstr = False
    time_ystr = False
    try:
        float(xmin)
    except Exception:
        time_xstr = True
    try:
        float(ymin)
    except Exception:
        time_ystr = True

    # Set plot limits
    if plotvars.plot is not None and twinx is None and twiny is None:
        if not time_xstr and not time_ystr:
            plotvars.plot.axis(
                [plotvars.xmin, plotvars.xmax, plotvars.ymin, plotvars.ymax]
            )

        if plotvars.xlog:
            plotvars.plot.set_xscale("log")
        if plotvars.ylog:
            plotvars.plot.set_yscale("log")

    # Set twinx or twiny if requested
    if twinx is not None:
        plotvars.twinx = twinx
    if twiny is not None:
        plotvars.twiny = twiny


# TODO move this to 'colour' module once set up. For now put here since
# it is called by 'reset' method.
def cscale(
    scale=None,
    ncols=None,
    white=None,
    below=None,
    above=None,
    reverse=False,
    uniform=False,
):
    """
    | Choose and manipulate colour maps. Around 200 colour scales are
    | available (see the gallery section for more details).
    |
    | scale=None - name of colour map
    | ncols=None - number of colours for colour map
    | white=None - change these colours to be white
    | below=None - change the number of colours below the mid point of
    |               the colour scale to be this
    | above=None - change the number of colours above the mid point of
    |               the colour scale to be this
    | reverse=False - reverse the colour scale
    | uniform=False - produce a uniform colour scale.
    |                 For example: if below=3 and above=10 are specified
    |                 then initially below=10 and above=10 are used.  The
    |                 colour scale is then cropped to use scale colours
    |                 6 to 19.  This produces a more uniform intensity colour
    |                 scale than one where all the blues are compressed into
    |                 3 colours.
    |
    |
    | Personal colour maps are available by saving the map as red green blue
    | to a file with a set of values on each line.
    |
    |
    | Use cscale() To reset to the default settings.
    |
    :Returns:
        None
    """

    # If no map requested reset to default
    if scale is None:
        scale = "scale1"
        plotvars.cscale_flag = 0
        return
    else:
        plotvars.cs_user = scale
        plotvars.cscale_flag = 1
        vals = [ncols, white, below, above]
        if any(val is not None for val in vals):
            plotvars.cscale_flag = 2
        if reverse is not False or uniform is not False:
            plotvars.cscale_flag = 2

    if scale == "scale1" or scale == "":
        if scale == "scale1":
            myscale = cscale1
        if scale == "viridis":
            myscale = viridis
        # convert cscale1 or viridis from hex to rgb
        r = []
        g = []
        b = []
        for myhex in myscale:
            myhex = myhex.lstrip("#")
            mylen = len(myhex)
            rgb = tuple(
                int(myhex[i : i + mylen // 3], 16)
                for i in range(0, mylen, mylen // 3)
            )
            r.append(rgb[0])
            g.append(rgb[1])
            b.append(rgb[2])
    else:
        package_path = os.path.dirname(__file__)
        # TODO improve path processing - eventually move to Pathlib
        file = os.path.join(
            package_path, "../colour/colourmaps/" + scale + ".rgb"
        )
        if os.path.isfile(file) is False:
            if os.path.isfile(scale) is False:
                errstr = (
                    "\ncscale error - colour scale not found:\n"
                    f"File {file} not found\n"
                    f"Scale {scale} not found\n"
                )
                raise Warning(errstr)
            else:
                file = scale

        # Read in rgb values and convert to hex
        with open(file, "r") as f:
            lines = f.read()
            lines = lines.splitlines()
            r = []
            g = []
            b = []
            for line in lines:
                vals = line.split()
                r.append(int(vals[0]))
                g.append(int(vals[1]))
                b.append(int(vals[2]))

    # Reverse the colour scale if requested
    if reverse:
        r = r[::-1]
        g = g[::-1]
        b = b[::-1]

    # Interpolate to a new number of colours if requested
    if ncols is not None:
        x = np.arange(np.size(r))
        xnew = np.linspace(0, np.size(r) - 1, num=ncols, endpoint=True)
        f_red = interpolate.interp1d(x, r)
        f_green = interpolate.interp1d(x, g)
        f_blue = interpolate.interp1d(x, b)
        r = f_red(xnew)
        g = f_green(xnew)
        b = f_blue(xnew)

    # Change the number of colours below and above the mid-point if requested
    if below is not None or above is not None:

        # Mid-point of colour scale
        npoints = np.size(r) // 2

        # Below mid point x locations
        x_below = []
        lower = 0
        if below == 1:
            x_below = 0
        if below is not None:
            lower = below
        if below is None:
            lower = npoints
        if below is not None and uniform:
            lower = max(above, below)
        if lower > 1:
            x_below = ((npoints - 1) / float(lower - 1)) * np.arange(lower)

        # Above mid point x locations
        x_above = []
        upper = 0
        if above == 1:
            x_above = npoints * 2 - 1
        if above is not None:
            upper = above
        if above is None:
            upper = npoints
        if above is not None and uniform:
            upper = max(above, below)
        if upper > 1:
            x_above = ((npoints - 1) / float(upper - 1)) * np.arange(
                upper
            ) + npoints

        # Append new colour positions
        xnew = np.append(x_below, x_above)

        # Interpolate to new colour scale
        xpts = np.arange(np.size(r))
        f_red = interpolate.interp1d(xpts, r)
        f_green = interpolate.interp1d(xpts, g)
        f_blue = interpolate.interp1d(xpts, b)
        r = f_red(xnew)
        g = f_green(xnew)
        b = f_blue(xnew)

        # Reset colours if uniform is set
        if uniform:
            mid_pt = max(below, above)
            r = r[mid_pt - below : mid_pt + above]
            g = g[mid_pt - below : mid_pt + above]
            b = b[mid_pt - below : mid_pt + above]

    # Convert to hex
    hexarr = []
    for col in np.arange(np.size(r)):
        hexarr.append(f"#{int(r[col]):02x}{int(g[col]):02x}{int(b[col]):02x}")

    # White requested colour positions
    if white is not None:
        if np.size(white) == 1:
            hexarr[white] = "#ffffff"
        else:
            for col in white:
                hexarr[col] = "#ffffff"

    # Set colour scale
    plotvars.cs = hexarr


def axes(
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xstep=None,
    ystep=None,
    xlabel=None,
    ylabel=None,
    title=None,
):
    """
    | Set axes plotting parameters. The xstep and ystep
    | parameters are used to label the axes starting at the left hand side and
    | bottom of the plot respectively. For tighter control over labelling use
    | xticks, yticks to specify the tick positions and xticklabels,
    | yticklabels to specify the associated labels.

    | xstep=xstep - x axis step
    | ystep=ystep - y axis step
    | xlabel=xlabel - label for the x-axis
    | ylabel=ylabel - label for the y-axis
    | xticks=xticks - values for x ticks
    | xticklabels=xticklabels - labels for x tick marks
    | yticks=yticks - values for y ticks
    | yticklabels=yticklabels - labels for y tick marks
    | title=None - set title
    |
    | Use axes() to reset all the axes plotting attributes to the default.

    :Returns:
     None
    """

    if all(
        val is None
        for val in [
            xticks,
            yticks,
            xticklabels,
            yticklabels,
            xstep,
            ystep,
            xlabel,
            ylabel,
            title,
        ]
    ):
        plotvars.xticks = None
        plotvars.yticks = None
        plotvars.xticklabels = None
        plotvars.yticklabels = None
        plotvars.xstep = None
        plotvars.ystep = None
        plotvars.xlabel = None
        plotvars.ylabel = None
        plotvars.title = None
        return

    plotvars.xticks = xticks
    plotvars.yticks = yticks
    plotvars.xticklabels = xticklabels
    plotvars.yticklabels = yticklabels
    plotvars.xstep = xstep
    plotvars.ystep = ystep
    plotvars.xlabel = xlabel
    plotvars.ylabel = ylabel
    plotvars.title = title


def reset():
    """
    | Reset all plotting variables.
    |
    :Returns:
     name
    """
    axes()
    cscale()
    levs()
    gset()
    mapset()
    setvars()
