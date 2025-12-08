import matplotlib
import numpy as np

from ..parameters import (
    plotvars,
)
from ..internal import (
    _cscale_get_map,
)


def cbar(
    labels=None,
    orientation=None,
    position=None,
    shrink=None,
    fraction=None,
    title=None,
    fontsize=None,
    fontweight=None,
    text_up_down=None,
    text_down_up=None,
    drawedges=None,
    levs=None,
    thick=None,
    anchor=None,
    extend=None,
    mid=None,
    verbose=None,
):
    """
    | The cf-plot interface to the Matplotlib colorbar routine.
    |
    | labels - colorbar labels
    | orientation - orientation 'horizontal' or 'vertical'
    | position - user specified colorbar position in normalised
    |            plot coordinates [left, bottom, width, height]
    | shrink - default=1.0 - scale colorbar along length
    | fraction - default = 0.12 - space for the colorbar in
    |            normalised plot coordinates
    | title - title for the colorbar
    | fontsize - font size for the colorbar text
    | fontweight - font weight for the colorbar text
    | text_up_down - label division text up and down starting with up
    | text_down_up - label division text down and up starting with down
    | drawedges - Draw internal delimeter lines in colorbar
    | levs - colorbar levels
    | thick - set height of colorbar - default = 0.015,
    |         in normalised plot coordinates
    | anchor - default=0.3 - anchor point of colorbar within the fraction
    |                        space. 0.0 = close to plot, 1.0 = further away
    | extend = None - extensions for colorbar. The default is for
    |                 extensions at both ends.
    | mid = False - label mid points of colours rather than the boundaries
    | verbose = None
    |
    """

    if verbose:
        print("con - adding a colour bar")

    if fontsize is None:
        fontsize = plotvars.colorbar_fontsize
    if fontweight is None:
        fontweight = plotvars.colorbar_fontweight
    if thick is None:
        thick = 0.012
        if plotvars.rows == 2:
            thick = 0.008
        if plotvars.rows == 3:
            thick = 0.005
        if plotvars.rows >= 4:
            thick = 0.003
    if drawedges is None:
        drawedges = True
    if orientation is None:
        orientation = "horizontal"
    if fraction is None:
        fraction = 0.12
        if plotvars.rows == 2:
            fraction = 0.08
        if plotvars.rows == 3:
            fraction = 0.06
        if plotvars.rows >= 4:
            fraction = 0.04
    if shrink is None:
        shrink = 1.0
    if anchor is None:
        anchor = 0.3
        if plotvars.plot_type > 1:
            anchor = 0.5

    # Code for when the user specifies nlevs to the contour command rather than
    # letting cf-plot work out some levels
    if isinstance(levs, int):
        if plotvars.plot_type == 0:
            myplot = plotvars.mymap
        else:
            myplot = plotvars.plot

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(myplot)
        if orientation == "horizontal":
            if plotvars.plot_type == 1:
                cax = divider.append_axes(
                    "bottom", size="2%", pad=0.3, title=title
                )
            else:
                cax = divider.append_axes(
                    "bottom", size="2%", pad=1.0, title=title
                )
        else:
            cax = divider.append_axes("right", size="2%", pad=0.5, title=title)

        plotvars.master_plot.colorbar(
            plotvars.image, cax=cax, orientation=orientation
        )

        return

    # Change plot position based on colorbar location
    ax1 = None  # define next
    if position is None:
        # Work out whether the plot is a map plot or normal plot
        if plotvars.plot_type == 1 or plotvars.plot_type == 6:
            this_plot = plotvars.mymap
        else:
            this_plot = plotvars.plot

        if plotvars.plot_type == 6 and (
            plotvars.proj == "rotated" or plotvars.proj == "UKCP"
        ):
            this_plot = plotvars.plot

        left, bottom, width, height = this_plot.get_position().bounds

        if orientation == "horizontal":
            if plotvars.plot_type == 1:
                # If the plot is too square in terms of aspect ratio, it
                # will push the colour bar down and it can be cut off, so
                # move the plot up in these cases. Use height and width
                # ratio since this represents the matplotlib object
                # position differences, so is the best way to measure.
                left, bottom, width, height = (
                    this_plot.get_position().bounds
                )
                # Empirical sweet spot: roughly where plot area is taller
                # than it is wide, including when plot area is roughly square
                if height / width >= 0.9:
                    this_plot.set_position(
                        [left, bottom + fraction, width, height - fraction]
                    )
                    left, bottom, width, height = (
                        this_plot.get_position().bounds
                    )

                ax1 = plotvars.master_plot.add_axes(
                    [
                        left + width * (1.0 - shrink) / 2.0,
                        bottom + fraction * (anchor - 1.0),
                        width * shrink,
                        thick,
                    ]
                )
            else:
                this_plot.set_position(
                    [left, bottom + fraction, width, height - fraction]
                )
                ax1 = plotvars.master_plot.add_axes(
                    [
                        left + width * (1.0 - shrink) / 2.0,
                        bottom,
                        width * shrink,
                        thick,
                    ]
                )
        else:
            ax1 = plotvars.master_plot.add_axes(
                [
                    left + width + fraction * (anchor - 1),
                    bottom + height * (1.0 - shrink) / 2.0,
                    thick,
                    height * shrink,
                ]
            )
            this_plot.set_position([left, bottom, width - fraction, height])

    else:
        # Set axes position on coords supplied by the user
        ax1 = plotvars.master_plot.add_axes(position)

    if levs is None:
        if plotvars.levels is not None:
            levs = np.array(plotvars.levels)
        else:
            if labels is None:
                errstr = "\n\ncbar error - No levels or labels supplied \n\n"
                raise TypeError(errstr)
            else:
                levs = np.arange(len(labels))

    if labels is None:
        labels = levs

    # Work out colour bar labeling
    lbot = levs
    if text_up_down:
        lbot = levs[1:][::2]
        ltop = levs[::2]
    if text_down_up:
        lbot = levs[::2]
        ltop = levs[1:][::2]

    # Get the colour map
    colmap = _cscale_get_map()
    cmap = matplotlib.colors.ListedColormap(colmap)
    if extend is None:
        extend = plotvars.levels_extend

    ncolors = np.size(levs)

    if extend in ("both", "max"):
        ncolors -= 1

    if mid is not None:
        lbot = [
            (lbot[i + 1] - lbot[i]) / 2.0 + lbot[i]
            for i in np.arange(len(labels))
        ]

    if not isinstance(levs, int):
        plotvars.norm = matplotlib.colors.BoundaryNorm(
            boundaries=levs, ncolors=ncolors
        )

        # Change boundaries to floats
        boundaries = levs.astype(float)

        # Add colorbar extensions if definded by levs.  Using boundaries[0]-1
        # for the lower and boundaries[-1]+1 is just for the colorbar and
        # has no meaning for the plot.
        if extend in ("both", "min"):
            cmap.set_under(plotvars.cs[0])
            boundaries = np.insert(boundaries, 0, boundaries[0] - 1)
        if extend in ("both", "max"):
            cmap.set_over(plotvars.cs[-1])
            boundaries = np.insert(
                boundaries, len(boundaries), boundaries[-1] + 1
            )

        if not isinstance(levs, list):
            lbot = None

        colorbar = matplotlib.colorbar.ColorbarBase(
            ax1,
            cmap=cmap,
            norm=plotvars.norm,
            extend=extend,
            extendfrac="auto",
            boundaries=boundaries,
            ticks=lbot,
            spacing="uniform",
            orientation=orientation,
            drawedges=drawedges,
        )

    else:
        ax1 = plotvars.master_plot.add_axes(position)
        colorbar = matplotlib.colorbar.ColorbarBase(
            ax1,
            cmap=cmap,
            norm=plotvars.norm,
            extend=extend,
            extendfrac="auto",
            boundaries=boundaries,
            ticks=lbot,
            spacing="uniform",
            orientation=orientation,
            drawedges=drawedges,
        )

    colorbar.set_label(title, fontsize=fontsize, fontweight=fontweight)

    # Bug in Matplotlib colorbar labelling
    # With clevs=[-1, 1, 10000, 20000, 30000, 40000, 50000, 60000]
    # Labels are [0, 2, 10001, 20001, 30001, 40001, 50001, 60001]
    # With a +1 near to the colorbar label

    # Check for an extraneous level compared to the levs
    if len(labels) > len(levs):
        labels = labels[: len(levs)]

    colorbar.set_ticklabels([str(i) for i in labels])
    if orientation == "horizontal":
        for tick in colorbar.ax.xaxis.get_ticklines():
            tick.set_visible(False)
        for t in colorbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)
            t.set_fontweight(fontweight)
    else:
        for tick in colorbar.ax.yaxis.get_ticklines():
            tick.set_visible(False)
        for t in colorbar.ax.get_yticklabels():
            t.set_fontsize(fontsize)
            t.set_fontweight(fontweight)

    # Alternate text top and bottom on a horizontal colorbar if requested
    # Use method described at:
    # https://stackoverflow.com/questions/37161022/matplotlib-colorbar-
    # alternating-top-bottom-labels
    if text_up_down or text_down_up:

        vmin = colorbar.norm.vmin
        vmax = colorbar.norm.vmax

        cbar_extend = colorbar.extend
        if cbar_extend == "min":
            shift_l = 0.05
            scaling = 0.95
        elif cbar_extend == "max":
            shift_l = 0.0
            scaling = 0.95
        elif cbar_extend == "both":
            shift_l = 0.05
            scaling = 0.9
        else:
            shift_l = 0.0
            scaling = 1.0

        # Print bottom tick labels
        colorbar.ax.set_xticklabels(lbot)

        # Print top tick labels
        for ii in ltop:
            colorbar.ax.text(
                shift_l + scaling * (ii - vmin) / (vmax - vmin),
                1.5,
                str(ii),
                transform=colorbar.ax.transAxes,
                va="bottom",
                ha="center",
                fontsize=fontsize,
                fontweight=fontweight,
            )

        for t in colorbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)
            t.set_fontweight(fontweight)
