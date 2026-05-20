.. _trajectories:

Trajectories
************

Data stored as discrete sampling geometries, such as contiguous ragged array
format, can be plotted using cf-plot. Multiple trajectories represented by
multidimensional-arrays with a trajectory dimension can be plotted, as below.

.. include:: examples/example_39.rst


Single trajectories can also be plotted, including those in one-dimensional
array form, as below.

.. include:: examples/example_39b.rst

This a plot of relative vorticity tracks is made in the cylindrical projection.

.. include:: examples/example_40.rst

.. include:: examples/example_41.rst

.. include:: examples/example_42a.rst

In this plot the tracks between 1979-12-01 and 1979-12-30 are selected and
labelled according intensity with a colour bar.


.. include:: examples/example_42b.rst

Selecting `legend_lines=True` plots lines only and colours them according
to the sum of the start and end point divided by two. This can be a
useful option when there are lots of trajectories.
