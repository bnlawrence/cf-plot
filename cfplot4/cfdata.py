import numpy as np

def cf_data_assign(f=None, colorbar_title=None, verbose=None, rotated_vect=False):
    """
     | Check cf input data is okay and return data for contour plot.
     | This is an internal routine not used by the user.
     | f=None - input cf field
     | colorbar_title=None - input colour bar title
     | rotated vect=False - return 1D x and y for rotated plot vectors
     | verbose=None - set to 1 to get a verbose idea of what the
     |          cf_data_assign is doing

     :Returns:
      | f - data for contouring
      | x - x coordinates of data (optional)
      | y - y coordinates of data (optional)
      | ptype - plot type
      | colorbar_title - colour bar title
      | xlabel - x label for plot
      | ylabel - y label for plot
      |
      |
      |
      |
      |
    """

    # Check input data has the correct number of dimensions
    # Take into account rotated pole fields having extra dimensions
    ndim = len(f.domain_axes().filter_by_size(cf.gt(1)))
    if f.ref('grid_mapping_name:rotated_latitude_longitude', default=False) is False:
        if (ndim > 2 or ndim < 1):
            print('')
            if (ndim > 2):
                errstr = 'cf_data_assign error - data has too many dimensions'
            if (ndim < 1):
                errstr = 'cf_data_assign error - data has too few dimensions'
            errstr += '\n cf-plot requires one or two dimensional data\n'
            for mydim in list(f.dimension_coordinates()):
                sn = getattr(f.construct(mydim), 'standard_name', False)
                ln = getattr(f.construct(mydim), 'long_name', False)
                if sn:
                    errstr = errstr + \
                        str(mydim) + ',' + str(sn) + ',' + \
                        str(f.construct(mydim).size) + '\n'
                else:
                    if ln:
                        errstr = errstr + \
                            str(mydim) + ',' + str(ln) + ',' + \
                            str(f.construct(mydim).size) + '\n'
            raise Warning(errstr)

    # Set up data arrays and variables
    lons = None
    lats = None
    height = None
    time = None
    xlabel = ''
    ylabel = ''
    has_lons = False
    has_lats = False
    has_height = False
    has_time = False
    xpole = None
    ypole = None
    ptype = None
    field = None
    x = None
    y = None
    
    
    # Check for multiple Z coordinates
    myz = find_z(f)

    
                    

    # assign field data
    field = np.squeeze(f.array)

    # Change Boolean data to integer
    if str(f.dtype) == 'bool':
        warnstr = '\n\n\n Warning - boolean data found - converting to integers\n\n\n'
        print(warnstr)
        g = deepcopy(f)
        g.dtype = int
        field = np.squeeze(g.array)

    # Check what plot type is required.
    # 0=simple contour plot, 1=map plot, 2=latitude-height plot,
    # 3=longitude-time plot, 4=latitude-time plot.
    if has_lons and has_lats:
        ptype = 1
        x = lons
        y = lats

    if has_lats and has_height:
        ptype = 2
        x = lats
        y = height
        
        xname = cf_var_name(field=f, dim='Y')
        xunits = str(getattr(f.construct('Y'), 'Units', ''))
        if xunits == 'degrees_north':
            xunits = 'degrees'
        if xunits != '':
            xlabel = xname + ' (' + xunits + ')'
        else:
            xlabel = xname
            
        yname = cf_var_name(field=f, dim=myz)
        yunits = str(getattr(f.construct(myz), 'Units', ''))
        if yunits != '':
            ylabel = yname + ' (' + yunits + ')'
        else:
            ylabel = yname 
                         
    if has_lons and has_height:
        ptype = 3
        x = lons
        y = height
        
        xname = cf_var_name(field=f, dim='X')
        xunits = str(getattr(f.construct('X'), 'Units', ''))
        if xunits == 'degrees_east':
            xunits = 'degrees'
        if xunits != '':
            xlabel = xname + ' (' + xunits + ')'
        else:
            xlabel = xname
            
        yname = cf_var_name(field=f, dim=myz)
        yunits = str(getattr(f.construct(myz), 'Units', ''))
        if yunits != '':
            ylabel = yname + ' (' + yunits + ')'
        else:
            ylabel = yname 
        
    if has_lons and has_time:
        ptype = 4
        x = lons
        y = time
        
        xname = cf_var_name(field=f, dim='X')
        xunits = str(getattr(f.construct('X'), 'Units', ''))
        if xunits == 'degrees_east':
            xunits = 'degrees'
        if xunits != '':
            xlabel = xname + ' (' + xunits + ')'
        else:
            xlabel = xname
            
        yname = cf_var_name(field=f, dim='T')
        yunits = str(getattr(f.construct('T'), 'Units', ''))
        if yunits != '':
            ylabel = yname + ' (' + yunits + ')'
        else:
            ylabel = yname 

    if has_lats and has_time:
        ptype = 5
        x = lats
        y = time
        
        xname = cf_var_name(field=f, dim='Y')
        xunits = str(getattr(f.construct('Y'), 'Units', ''))
        if xunits == 'degrees_north':
            xunits = 'degrees'
        if xunits != '':
            xlabel = xname + ' (' + xunits + ')'
        else:
            xlabel = xname
            
        yname = cf_var_name(field=f, dim='T')
        yunits = str(getattr(f.construct('T'), 'Units', ''))
        if yunits != '':
            ylabel = yname + ' (' + yunits + ')'
        else:
            ylabel = yname 

    # time height plot
    if has_height and has_time:
        ptype = 7
        x = time
        y = height
        
        xname = cf_var_name(field=f, dim='T')
        xunits = str(getattr(f.construct('T'), 'Units', ''))
        if xunits != '':
            xlabel = xname + ' (' + xunits + ')'
        else:
            xlabel = xname
            
        yname = cf_var_name(field=f, dim='Z')
        yunits = str(getattr(f.construct('Z'), 'Units', ''))
        if yunits != '':
            ylabel = yname + ' (' + yunits + ')'
        else:
            ylabel = yname 

        # Rotate array to get it as time vs height
        field = np.rot90(field)
        field = np.flipud(field)

    # Rotated pole
    if f.ref('grid_mapping_name:rotated_latitude_longitude', default=False):
        ptype = 6

        rotated_pole = f.ref('grid_mapping_name:rotated_latitude_longitude')
        xpole = rotated_pole['grid_north_pole_longitude']
        ypole = rotated_pole['grid_north_pole_latitude']

        # Extract grid x and y coordinates
        for mydim in list(f.dimension_coordinates()):
            name = cf_var_name(field=f, dim=mydim)

            if name in ['grid_longitude', 'longitude', 'x']:
                x = np.squeeze(f.construct(mydim).array)
                xunits = str(getattr(f.construct(mydim), 'units', ''))
                xlabel = cf_var_name(field=f, dim=mydim)

            if name in ['grid_latitude', 'latitude', 'y']:
                y = np.squeeze(f.construct(mydim).array)
                # Flip y and data if reversed
                if y[0] > y[-1]:
                    y = y[::-1]
                    field = np.flipud(field)
                yunits = str(getattr(f.construct(mydim), 'Units', ''))
                ylabel = cf_var_name(field=f, dim=mydim) + yunits


    # Extract auxiliary lons and lats if they exist
    if ptype == 1 or ptype is None:
        if plotvars.proj != 'rotated' and not rotated_vect:
            aux_lons = False
            aux_lats = False
            for mydim in list(f.auxiliary_coordinates()):
                name = cf_var_name(field=f, dim=mydim)
                if name in ['longitude']:
                    xpts = np.squeeze(f.construct(mydim).array)
                    aux_lons = True
                if name in ['latitude']:
                    ypts = np.squeeze(f.construct(mydim).array)
                    aux_lats = True

            if aux_lons and aux_lats:
                x = xpts
                y = ypts
                ptype = 1


    # UKCP grid
    if f.ref('grid_mapping_name:transverse_mercator', default=False):
        ptype = 1
        field = np.squeeze(f.array)

        # Find the auxiliary lons and lats if provided
        has_lons = False
        has_lats = False
        for mydim in list(f.auxiliary_coordinates()):
            name = cf_var_name(field=f, dim=mydim)
            if name in ['longitude']:
                x = np.squeeze(f.construct(mydim).array)
                has_lons = True
            if name in ['latitude']:
                y = np.squeeze(f.construct(mydim).array)
                has_lats = True

        # Calculate lons and lats if no auxiliary data for these
        if not has_lons or not has_lats:
            xpts = f.construct('X').array
            ypts = f.construct('Y').array
            field = np.squeeze(f.array)

            ref = f.ref('grid_mapping_name:transverse_mercator')
            false_easting = ref['false_easting']
            false_northing = ref['false_northing']
            central_longitude = ref['longitude_of_central_meridian']
            central_latitude = ref['latitude_of_projection_origin']
            scale_factor = ref['scale_factor_at_central_meridian']

            # Set the transform
            transform = ccrs.TransverseMercator(false_easting=false_easting,
                                                false_northing=false_northing,
                                                central_longitude=central_longitude,
                                                central_latitude=central_latitude,
                                                scale_factor=scale_factor)

            # Calculate the longitude and latitude points
            xvals, yvals = np.meshgrid(xpts, ypts)
            points = ccrs.PlateCarree().transform_points(transform, xvals, yvals)
            x = np.array(points)[:, :, 0]
            y = np.array(points)[:, :, 1]


    # None of the above
    if ptype is None:
        ptype = 0

        data_axes = f.get_data_axes()
        count = 1
        for d in data_axes:
            try:
                c = f.coordinate(filter_by_axis  = [d])
                if np.size(c.array) > 1:
                    if count == 1:
                        
                        y = c
                        mycoord = 'dimensioncoordinate'+str(d[-1])
                        yunits = str(getattr(f.coord(mycoord), 'Units', ''))
                        if yunits != '':
                            yunits = '(' + yunits + ')'
                        ylabel = cf_var_name(field=f, dim=mycoord) + yunits                         
                    elif count == 2:
                        x = c
                        mycoord = 'dimensioncoordinate'+str(d[-1])
                        xunits = str(getattr(f.coord(mycoord), 'units', ''))
                        if xunits != '':
                            xunits = '(' + xunits + ')'
                        xlabel = cf_var_name(field=f, dim=mycoord) + xunits
                    count += 1
            except ValueError:
                errstr = "\n\ncf_data_assign - cannot find data to return\n\n" 
                errstr += str(f.constructs.domain_axis_identity(d)) + "\n\n"
                raise Warning(errstr)



    # Assign colorbar_title
    if (colorbar_title is None):
        colorbar_title = 'No Name'
        if hasattr(f, 'id'):
            colorbar_title = f.id
        nc = f.nc_get_variable(None)
        if nc:
            colorbar_title = f.nc_get_variable()
        if hasattr(f, 'short_name'):
            colorbar_title = f.short_name
        if hasattr(f, 'long_name'):
            colorbar_title = f.long_name
        if hasattr(f, 'standard_name'):
            colorbar_title = f.standard_name

        if hasattr(f, 'Units'):
            if str(f.Units) == '':
                colorbar_title = colorbar_title + ''
            else:
                colorbar_title = colorbar_title + \
                    '(' + supscr(str(f.Units)) + ')'

        
    # Return data
    return(field, x, y, ptype, colorbar_title, xlabel, ylabel, xpole, ypole)


def get_cfdims(f, verbose=False):
    """ 
    Analyse the coordinates of a field and return the data
    arrays and coordinate type.
    """
    r = ''
    indexed = {}
    bounds = {}
    for mycoord in f.coords():
        c = f.coord(mycoord)
        if c.X:
            if verbose:
                print('lons -', mydim)
            lons = f.construct(mycoord).array.squeeze()
            if np.size(lons) > 1:
                r+='x'
                indexed['x']=lons
                if c.has_bounds():
                    bounds['x']=c.bounds.array
                else:
                    raise NotImplementedError('Need to make x bounds')

        if c.Y:
            if verbose:
                print('lats -', mydim)
            lats = f.construct(mycoord).array.squeeze()
            if np.size(lats) > 1:
                r+='y'
                indexed['y']=lats
                if c.has_bounds():
                    bounds['y']=c.bounds.array
                else:
                    raise NotImplementedError('Need to make y bounds')

        if c.Z:
            if verbose:
                print('height -', mydim)
            height = f.construct(mycoord).array.squeeze()
            if np.size(height) > 1:
                r+='z'
                indexed['z']=height
                if c.has_bounds():
                    bounds['z']=c.bounds.array
                else:
                    raise NotImplementedError('Need to make z bounds')

        if c.T:
            if verbose:
                print('time -', mydim)
            time = f.construct(mycoord).array.squeeze()
            if np.size(time) > 1:
                r+='t'
                indexed['z']=time
                if c.has_bounds():
                    bounds['t']=c.bounds.array
                else:
                    raise NotImplementedError('Need to make t bounds')


    if len(r) != 2:
        raise ValueError(f'Could not find two CF dimensions to plot (got "{r}")')

    if 'x' in r:
        xx = indexed.pop('x')
        other = [(k,v) for k,v in indexed.items()][0]
        r = 'x'+other[0]
        yy = other[1]
        xb = bounds['x']
        yb = bounds[other[0]]

    elif 't' in r:
        xx = indexed.pop('t')
        other = [(k,v) for k,v in indexed.items()][0]
        yy = other[1]
        r = other[0]+'t'
        xb = bounds['t']
        yb = bounds[other[0]]
    elif 'z' in r:
        yy = indexed.pop('z')
        other = [(k,v) for k,v in indexed.items()][0]
        r = other[0]+'z'
        xx = other[1]
        yb = bounds['z']
        xb = bounds[other[1]]

    return r, xx, yy, xb, yb