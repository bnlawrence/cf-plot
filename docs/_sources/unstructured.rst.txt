.. _unstructured:

Unstructured grids and UGRID
****************************

*Unstructured* grids have data points in non-regular locations. Examples of
these are the LFRic model grid, the ORCA ocean grid and weather station data.

The `UGRID Conventions <https://ugrid-conventions.github.io/ugrid-conventions>`_
are conventions for storing unstructured (or flexible mesh) model data in
netCDF. As of CF-1.11, version 1.0 of UGRID is
`partially included within the CF Conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#ugrid-conventions>`_.

This page demonstrates how to plot data in the form of both UGRID-compliant
netCDF and NumPy arrays of unstructured data.


.. include:: examples/example_24a.rst


Here we identify the fields in the data that have the longitudes and
latitudes for the corner points for the field and pass them to ``cfp.con``.
Once UGRID is in the CF metadata conventions the face plotting commands
will be simplified as the face connectivity, associated longitudes and
latitudes will all be described within the data field. The plotted data
is a test field of potential temperature and isn't realistic in
regards to the actual values.


.. include:: examples/example_24b.rst


Here the projection was changed to show the north pole.


.. include:: examples/example_24c.rst


The data in the field has auxiliary longitudes and latitudes that can
be contoured as normal. Internally in cf-plot this is made using the
matplotlib ``tricontourf`` function as the data points are spatially irregular.


.. include:: examples/example_25.rst


The ORCA2 grid is an ocean grid with missing values over the land points.
The data in this file is from before the UGRID convention was started
and has no face connectivity or corner coordinates. In this case we can
only plot a normal contour plot.


Here we read in temperature data in a text file from meteorological
stations around the British Isles and make a contour plot.


.. include:: examples/example_26a.rst


To see if this plot is correct we can add some extra code to that above
to plot the station locations and values at that point, as below.
The decimal point is roughly where the data point is located.


.. include:: examples/example_26b.rst
