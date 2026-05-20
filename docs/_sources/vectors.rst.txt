.. _vector:

Vector and stream plots
***********************

.. include:: examples/example_13.rst


.. include:: examples/example_14.rst


.. include:: examples/example_15.rst

Here we see the difference between plotting the vectors on the data grid
and on a interpolated grid. The supplied grid gives a bullseye effect
making the wind direction difficult to see near the pole.

.. include:: examples/example_16a.rst

Here we make a zonal mean vector plot with different vector keys and
scaling factors for the X and Y directions.

.. include:: examples/example_16b.rst

A streamplot is used to show fluid flow and 2D field gradients. In this
first example the data goes from 0 to 358.875 in longitude. The
cartopy / matplotlib interface seems to need the data to be inside the
data window in longitude so we anchor the data in cf-python using the
anchor method to start at -180 in longitude. If we didn't do this any
longitudes less than zero would have no streams drawn.


.. include:: examples/example_16c.rst

In the second streamplot example a colorbar showing the intensity of the
wind is drawn.

