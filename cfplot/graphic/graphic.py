import matplotlib
import matplotlib.pyplot as plot

from ..parameters import (
    levs,
    plotvars,
)


def gopen(
    rows=1,
    columns=1,
    user_plot=1,
    file="cfplot.png",
    orientation="landscape",
    figsize=[11.7, 8.3],
    left=None,
    right=None,
    top=None,
    bottom=None,
    wspace=None,
    hspace=None,
    dpi=None,
    user_position=False,
):
    """
    | Open a graphic file.
    |
    | rows=1 - number of plot rows on the page
    | columns=1 - number of plot columns on the page
    | user_plot=1 - internal plot variable - do not use.
    | file='cfplot.png' - default file name
    | orientation='landscape' - orientation - also takes 'portrait'
    | figsize=[11.7, 8.3]  - figure size in inches
    | left=None - left margin in normalised coordinates - default=0.12
    | right=None - right margin in normalised coordinates - default=0.92
    | top=None - top margin in normalised coordinates - default=0.08
    | bottom=None - bottom margin in normalised coordinates - default=0.08
    | wspace=None - width reserved for blank space between
    |               subplots - default=0.2
    | hspace=None - height reserved for white space between
    |               subplots - default=0.2
    | dpi=None - resolution in dots per inch
    | user_position=False - set to True to supply plot position via gpos
    |               xmin, xmax, ymin, ymax values
    :Returns:
     None

    """

    # Set values in globals
    plotvars.rows = rows
    plotvars.columns = columns
    if file != "cfplot.png":
        plotvars.file = file
    plotvars.orientation = orientation
    plotvars.user_plot = user_plot
    plotvars.gpos_called = False

    # Set user defined plot area to None
    plotvars.plot_xmin = None
    plotvars.plot_xmax = None
    plotvars.plot_ymin = None
    plotvars.plot_ymax = None

    if left is None:
        left = 0.12
    if right is None:
        right = 0.92
    if top is None:
        top = 0.95
    if bottom is None:
        bottom = 0.08
        if rows >= 3:
            bottom = 0.1
    if wspace is None:
        wspace = 0.2
    if hspace is None:
        hspace = 0.2
        if rows >= 3:
            hspace = 0.5

    if orientation != "landscape":
        if orientation != "portrait":
            errstr = (
                "gopen error\n"
                "orientation incorrectly set\n"
                f"input value was {orientation}\n"
                "Valid options are portrait or landscape\n"
            )
            raise Warning(errstr)

    # Set master plot size
    if orientation == "landscape":
        plotvars.master_plot = plot.figure(figsize=(figsize[0], figsize[1]))
    else:
        plotvars.master_plot = plot.figure(figsize=(figsize[1], figsize[0]))

    # Set margins
    plotvars.master_plot.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )

    # Set initial subplot
    if user_position is False and rows == 1 and columns == 1:
        gpos(pos=1)

    # Change tick length for plots > 2x2
    if columns > 2 or rows > 2:
        matplotlib.rcParams["xtick.major.size"] = 2
        matplotlib.rcParams["ytick.major.size"] = 2

    # Set image resolution
    if dpi is not None:
        plotvars.dpi = dpi


def gclose(view=True):
    """
    | Saves a graphics file. The default is to view the file as well
    | but `view=False` can be used to turn this off.

    | view = True - view graphics file
    :Returns:
     None

    """

    # Reset the user_plot variable to off
    plotvars.user_plot = 0

    # Test for python or ipython
    interactive = False
    try:
        __IPYTHON__
        interactive = True
    except NameError:
        interactive = False

    if matplotlib.is_interactive():
        interactive = True

    # Remove whitespace if requested
    saveargs = {}
    if plotvars.tight:
        saveargs = {"bbox_inches": "tight"}

    file = plotvars.file
    if file is not None:
        # Save a file
        type = 1
        if file[-3:] == ".ps":
            type = 1
        if file[-4:] == ".eps":
            type = 1
        if file[-4:] == ".png":
            type = 1
        if file[-4:] == ".pdf":
            type = 1
        if type is None:
            file = file + ".png"
        plotvars.master_plot.savefig(
            file,
            orientation=plotvars.orientation,
            dpi=plotvars.dpi,
            **saveargs,
        )
        plot.close()
    else:
        if plotvars.viewer == "display" and interactive is False:
            # Use Imagemagick display command if this exists
            disp = _which("display")
            if disp is not None:
                tfile = "cfplot.png"
                plotvars.master_plot.savefig(
                    tfile,
                    orientation=plotvars.orientation,
                    dpi=plotvars.dpi,
                    **saveargs,
                )
                matplotlib.pyplot.ioff()
                subprocess.Popen([disp, tfile])
            else:
                plotvars.viewer = "matplotlib"
        if plotvars.viewer == "matplotlib" or interactive:
            # Use Matplotlib viewer
            matplotlib.pyplot.ion()
            plot.show()

    # Reset plotting
    plotvars.plot = None
    plotvars.twinx = None
    plotvars.twiny = None
    plotvars.plot_xmin = None
    plotvars.plot_xmax = None
    plotvars.plot_ymin = None
    plotvars.plot_ymax = None
    plotvars.graph_xmin = None
    plotvars.graph_xmax = None
    plotvars.graph_ymin = None
    plotvars.graph_ymax = None
    plotvars.gpos_called = False
    plotvars.mymap = None
    plotvars.titles_con_called = False


def gpos(pos=1, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    | Set plot position. Plots start at top left and increase by one each plot
    | to the right. When the end of the row has been reached then the next
    | plot will be the leftmost plot on the next row down.

    | pos=pos - plot position
    |
    | The following four parameters are used to get full user control
    | over the plot position.  In addition to these cfp.gopen
    | must have the user_position=True parameter set.
    | xmin=None xmin in normalised coordinates
    | xmax=None xmax in normalised coordinates
    | ymin=None ymin in normalised coordinates
    | ymax=None ymax in normalised coordinates
    |
    :Returns:
     None

    """

    # Reset mymap
    plotvars.mymap = None

    # Check inputs are okay
    if pos < 1 or pos > plotvars.rows * plotvars.columns:
        errstr = (
            "pos error - pos out of range:\n range = 1 - "
            f"{plotvars.rows * plotvars.columns}"
            f"\n input pos was {pos}\n"
        )
        raise Warning(errstr)

    user_pos = False
    if all(val is not None for val in [xmin, xmax, ymin, ymax]):
        user_pos = True
        plotvars.plot_xmin = xmin
        plotvars.plot_xmax = xmax
        plotvars.plot_ymin = ymin
        plotvars.plot_ymax = ymax

    # Reset any accumulated muliple graph limits
    plotvars.graph_xmin = None
    plotvars.graph_xmax = None
    plotvars.graph_ymin = None
    plotvars.graph_ymax = None

    # Set gpos_called
    plotvars.gpos_called = True

    # Reset titles_con_called
    plotvars.titles_con_called = False

    if user_pos is False:
        plotvars.plot = plotvars.master_plot.add_subplot(
            plotvars.rows, plotvars.columns, pos
        )
    else:
        delta_x = plotvars.plot_xmax - plotvars.plot_xmin
        delta_y = plotvars.plot_ymax - plotvars.plot_ymin

        plotvars.plot = plotvars.master_plot.add_axes(
            [plotvars.plot_xmin, plotvars.plot_ymin, delta_x, delta_y]
        )

    plotvars.plot.tick_params(
        which="both", direction="out", right=True, top=True
    )

    # Set position in global variables
    plotvars.pos = pos

    # Reset contour levels if they are not defined by the user
    if plotvars.user_levs == 0:
        if plotvars.levels_step is None:
            levs()
        else:
            levs(step=plotvars.levels_step)
