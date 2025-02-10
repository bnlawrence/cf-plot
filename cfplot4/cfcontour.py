import cf
from cfplot4.core import CFP
from cfplot4.cfdata import get_cfdata
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import cartopy
import cartopy.crs as ccrs
import cartopy.util as cartopy_util
import cartopy.feature as cfeature
import numpy as np

class ContourFactory:
    """ 
    The task of this routine is to analyse the data and plotting request
    and return either a ContourData instance or a ContourMap instance. (ContourMap
    is a subclass of ContourData which is set up to use mapping projections
    on top of the normal contouring behaviour). 
    """
    @classmethod
    def create(cls, field, proj=None, radius=60, bbox=[],
                        blockfill=False, fill=True, lines=False, ax=None):
        """
        Create with a two dimensional and a plot-type requirement.
        Plot-types (xy,xz,xt ...) are inferred from the data provided.
        If the data is xy, supported projections are:
            proj = 'npstereo': north polar stereographic (out to radius degrees)
            proj = 'spstereo': south polar stereographic (out to radius degrees)
            proj = 'cyl' : (Default) Normal "Mercator-like" projection
            proj = 'native' : use any native coordinates provided directly (e.g. rotated pole)
        Where proj is one of 'cyl' or 'native', the optional bbox argument
        allows selection into the data.
        """
       
        data = field.data.squeeze()
        ndims = data.ndim
        if ndims > 2:
            raise ValueError(f'ContourFactory: Can only contour 2D fields (got {ndims})')
        ptype, xx, yy, xb, yb = get_cfdata(field)
        cfp = CFP(
            blockfill=blockfill,
            lines=lines,
            ptype=ptype,
            fill=fill,
            xb=xb,
            yb=yb
            )
        if ptype == 'xy':
            return ContourMap(cfp, field, data, xx, yy, proj, radius, bbox, ax)
        else:
            return ContourData(cfp, field, data, xx, yy, ax)
        
class ContourData:
    """ 
    This class is responsible for actually handling the 
    drawing of a contour map 
    """

    def __init__(self, cfp, field, data, xx, yy, ax=None):
        self.cfp = cfp
        self.field = field
        self.data = data
        self.xx = xx
        self.yy = yy
        self.transform = None
        self.zorder = 1
        self.set_levels()
        self.nx = len(xx)
        self.ny = len(yy)
        self.ax = ax

        # data specific
        self.irregular = False
        self.tripolar = False
        self.ugrid = False


    def set_levels(self, levels=0, level_spacing=None):
        """
        Set the colour/contour levels.
        """
        if level_spacing is not None:
            raise NotImplementedError
        if isinstance(levels, int):
            if levels == 0:
                levels = 10
            locator = MaxNLocator(nbins=levels)
            self.levels = locator.tick_values(self.data.min(), self.data.max())
        else:
            self.levels = levels

    def block_fill(self, zorder=1):
        """ 
        Always assume we have CF bounds, as we will have 
        created them before this point if we don't.
        """

        # make a coloured copy of the data quantized to the levels.
        colarr = np.zeros(self.data.shape)-1
        for i in range(np.size(self.levels)-1):
            lev = self.levels[i]
            levp1 = self.levels[i+1]
            pts = np.where(np.logical_and(self.data >= lev, self.data < levp1))
            colarr[pts] = int(i)

        for ix in range(self.nx):
            for iy in range(self.ny):

                xb = self.cfp.xb[ix]
                yb = self.cfp.yb[iy]
                value = self.cfp.cs[int(colarr[ix,iy])]

                # Plot the box
                self.ax.add_patch(
                    mpatches.Polygon(
                        [[xb[0], yb[0]], [xb[1],yb[0]], [xb[1], yb[1]],
                         [xb[0], yb[1]], [xb[0],yb[0]] ],
                        facecolor=value, 
                        zorder=zorder,
                        transform=self.transform
                        )
                    )     
                                                
                return

    def contour(self, zorder=2, alpha=None):
        """ 
        This method actually does the contouring. There are several control
        variables which effect which actual contour method is used.
        """
        
        if self.cfp.fill and not self.cfp.blockfill:

            colmap = self._cscale()
            cmap = ListedColormap(colmap)

            if not self.irregular or self.tripolar is True:               
                cs = self.ax.contourf(self.xx, self.yy, self.data.array, self.levels,
                               extend=self.cfp.levels_extend,
                               cmap=cmap, 
                               norm=self.cfp.norm,
                               alpha=alpha, 
                               transform=self.transform,
                               zorder=zorder, 
                               transform_first=self.cfp.transform_first)
                
            else:
                if np.size(self.data) > 0: 
                    cs = self.ax.tricontourf(self.xx, self.yy, self.data.array, self.levels, 
                                extend=self.cfp.levels_extend,
                                cmap=cmap, 
                                norm=self.cfp.norm,
                                alpha=alpha,   
                                transform=self.transform,
                                zorder=zorder)

        if self.cfp.lines:

            if not self.irregular or self.tripolar:
                    
                cs = self.ax.contour(self.xx, self.yy, self.data.array, self.levels, 
                                    colors=colors,
                                    linewidths=linewidths, 
                                    linestyles=linestyles, 
                                    alpha=alpha,
                                    transform=ccrs.transform, 
                                    zorder=zorder)
            else:
                cs = self.ax.tricontour(self.xx, self.yy, self.data.array,self.levels, 
                                    colors=colors,
                                    linewidths=linewidths, 
                                    linestyles=linestyles, 
                                    alpha=alpha,
                                    transform=self.transform, 
                                    zorder=zorder)

    def _cscale(self):
        """
        return colour map for use in contour plots.
        depends on the colour bar extensions
        """
        cscale_ncols = np.size(self.cfp.cs)
        if (self.cfp.levels_extend == 'both'):
            colmap = self.cfp.cs[1:cscale_ncols - 1]
        if (self.cfp.levels_extend == 'min'):
            colmap = self.cfp.cs[1:]
        if (self.cfp.levels_extend == 'max'):
            colmap = self.cfp.cs[:cscale_ncols - 1]
        if (self.cfp.levels_extend == 'neither'):
            colmap = self.cfp.cs
        return (colmap)

    def label_axes(self):
        pass


class ContourMap(ContourData):
    """ 
    This class is responsible for all the activities necessary
    for getting a georeferenced contour map 
    """


    def __init__(self, cfp, field, data, xx, yy, proj, radius, bbox, ax=None):

        super().__init__(cfp, field, data, xx, yy, ax=ax)
        self.proj = proj
        self.radius = radius
        self.bbox = bbox
    
        # Subset the data if a user map is set
        # This is use to speed up the plotting
            
        if proj == 'npstereo':
            self.cfp.boundinglat = 90-radius
            f = f.subspace(Y = cf.wi(self.cfp.boundinglat, 90.0))
            
        if proj == 'spstereo':
            self.cfp.boundlinglat = -90+radius
            f = f.subspace(Y = cf.wi(-90.0, self.cfp.boundinglat))

        if bbox:
            self.cfp.lonmin = bbox[0]
            self.cfp.lonmax = bbox[1]
            self.cfp.latmin = bbox[2]
            self.cfp.latmax = bbox[3]
        else:
            self.cfp.lonmin=self.xx[0]
            self.cfp.lonmax=self.xx[-1]
            self.cfp.latmin=self.yy[0]
            self.cfp.latmax=self.yy[-1]

        # Cartopy treats values as cyclic on the globe, so in order to avoid
        # errors where it thinks we are inputting the same value, e.g:
        #     UserWarning: Attempting to set identical low and high ylims makes
        #     transformation singular; automatically expanding.
        lon_diff = self.cfp.lonmax - self.cfp.lonmin
        lon_mid = self.cfp.lonmin + lon_diff / 2.0
        if proj in ("cyl", "merc") and lon_diff == 360.0:
            self.cfp.lonmax += 0.01  # ask to plot a tiny extra increment to distinguish

        self.ax = self._get_map_attributes(ax, proj, lon_mid)
        print('X=',self.cfp.lonmin,self.cfp.lonmax,lon_mid)

        # For fast map contours add transform_first=True to contourf command
        # and make lons and lats 2D.
        # Cartopy should transform the points before calling the contouring algorithm, 
        # which can have a significant impact on speed (it is much faster to transform
        # points than it is to transform patches) If this is unset and the number of points
        # in the x direction is > 400 then it is set to True.

        if self.cfp.transform_first is None and np.ndim(self.xx) == 1 and np.ndim(self.yy) == 1:
            if np.size(self.xx) >= 400:
                transform_first = True
                
        # Fast map contours are also needed when clevs is a integer
        #if type(clevs) == int and plotvars.plot_type == 1 and plotvars.proj == 'cyl':
        #    transform_first=True
        # FIXME: need to work out what is going on in the statement above given I have reordered level stuff 

        if self.cfp.transform_first:
            if np.ndim(self.xx) == 1 and np.ndim(self.yy) == 1:
                self.xx, self.yy = np.meshgrid(self.xx, self.yy)

        self.set_levels()
        
        if self.cfp.blockfill:
            self.block_fill()

        self.contour()

        self.decorate_axes()


    def decorate_axes(self, coastlines=True):

        self.label_axes()

        if coastlines:
            self.ax.coastlines()

    def _setup_axis(self, ax):
        """ Ensure the axis is in the right place with the right projection  """

        if ax is not None:
            # we don't really want it, but we like knowing where it is
            fig = ax.figure  # Get the existing figure
            new_ax = fig.add_axes(ax.get_position(), projection=self.transform)
            fig.delaxes(ax)  # Remove the old axis
            ax = new_ax  # Assign new axis with projection
        else:
            ax = plt.axes(projection=self.transform)  # Create new axis if none provided
        return ax  # Return the updated axis

    def _get_map_attributes(self, ax, proj, lon_mid):
        """ handles all the projection specific logic """
        if proj is None:
            proj = 'cylc'
        match proj:
            case "cylc":
                self.transform = ccrs.PlateCarree(central_longitude=lon_mid)
            case "merc":
                min_latitude = -80.0
                if self.cfp.lonmin > min_latitude:
                    min_latitude = self.cfp.lonmin
                max_latitude = 84.0
                if self.cfp.lonmax < max_latitude:
                    max_latitude = self.cfp.lonmax
                self.transform = ccrs.Mercator(
                    central_longitude=self.cfp.lon_0,
                    min_latitude=min_latitude,
                    max_latitude=max_latitude,
                )
            case "npstere":
                self.transform = ccrs.NorthPolarStereo(central_longitude=self.cfp.lon_0)
                # **cartopy 0.16 fix
                # Here we add in 0.01 to the longitude extent as this helps with
                # plotting lines and line labels
                self.cfp.lonmin = self.cfp.lon_0 - 180
                self.cfp.lonmax = self.cfp.lon_0 + 180.01
                self.cfp.latmin = self.cfp.boundinglat
                self.cfp.latmax = 90
            case "spstere":
                self.transform = ccrs.SouthPolarStereo(central_longitude=self.cfp.lon_0)
                # **cartopy 0.16 fix
                # Here we add in 0.01 to the longitude extent as this helps with
                # plotting lines and line labels
                self.cfp.lonmin = self.cfp.lon_0 - 180
                self.cfp.lonmax = self.cfp.lonmax + 180.01
                self.cfp.latmin = -90
                self.cfp.latmax = self.cfp.boundinglat
            case "ortho":
                self.transform = ccrs.Orthographic(
                    central_longitude=self.cfp.lon_0, 
                    central_latitude=self.cp.lat_0
                )
                self.cfp.lonmin = self.cfp.lon_0 - 180.0
                self.cfp.lonmax = self.cfp.lon_0 + 180.01
                self.extent = False
            case "moll": 
                self.transform = ccrs.Mollweide(central_longitude=self.cfp.lon_0)
                self.cfp.lonmin = self.cfp.lon_0 - 180.0
                self.cfp.lonmax = self.cfp.lon_0 + 180.01
                self.cfp.extent = False
            case "robin":
                self.transform = ccrs.Robinson(central_longitude=self.cfp.lon_0)
            case "lcc":
                self.cfp.lon_0 = self.cfp.lonmin + (self.cfp.lonmax - self.cfp.lonmin) / 2.0
                self.cfp.lat_0 = self.cfp.latmin + (self.cfp.latmax - self.cfp.latmin) / 2.0
                cutoff = -40
                if self.cfp.lat_0 <= 0:
                    cutoff = 40
                self.cfp.standard_parallels = [33, 45]
                if latmin <= 0 and latmax <= 0:
                    self.cfp.standard_parallels = [-45, -33]
                self.transform = ccrs.LambertConformal(
                    central_longitude=self.cfp.lon_0,
                    central_latitude=self.cfp.lat_0,
                    cutoff=cutoff,
                    standard_parallels=self.cfp.standard_parallels,
                )
            case "rotated":
                self.transform = ccrs.PlateCarree(central_longitude=self.cfp.lon_mid)
            case 'OSGB':
                self.transform = ccrs.OSGB()
            case "EuroPP":
                self.transform = ccrs.EuroPP()
            case "UKCP":
                # Special case of TransverseMercator for UKCP
                self.transform = ccrs.TransverseMercator()
            case "TransverseMercator":
                self.transform = ccrs.TransverseMercator()
                self.cfp.lonmin = self.cfp.lon_0 - 180.0
                self.cfp.lonmax = self.cfp.lon_0 + 180.01
                self.cfp.extent = False
            case "LambertCylindrical":
                self.transform = ccrs.LambertCylindrical()
        return self._setup_axis(ax)
 


