import numpy as np

def get_cfdata(f, verbose=False):
    """ 
    Analyse the coordinates of a field and return the data
    arrays and coordinate type. If data includes a dateline,
    reorganise appropriately.
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