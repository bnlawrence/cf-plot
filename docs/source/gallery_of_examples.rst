.. _examplesgallery:

Gallery of examples
===================

.. _allexamples:

Listing of all examples
-----------------------

.. note::
   All example code assumes the following imports have been made,
   where ``numpy`` is usually not required but it is for some
   of the examples:

   .. code-block:: python
      :caption: Imports required as setup for the examples, noting that
                cfplot is always aliased to ``cfp`` by convention

      import cfplot as cfp
      import cf
      import numpy as np  # only required for some examples


.. toctree::
   :maxdepth: 2

   examples/example_1.rst
   examples/example_2.rst
   examples/example_3.rst
   examples/example_4.rst
   examples/example_5.rst
   examples/example_6.rst
   examples/example_7.rst
   examples/example_8.rst
   examples/example_9.rst
   examples/example_10.rst
   examples/example_11.rst
   examples/example_12.rst
   examples/example_13.rst
   examples/example_14.rst
   examples/example_15.rst
   examples/example_16a.rst
   examples/example_16b.rst
   examples/example_16c.rst
   examples/example_17.rst
   examples/example_18.rst
   examples/example_19a.rst
   examples/example_19b.rst
   examples/example_19c.rst
   examples/example_20.rst
   examples/example_21a.rst
   examples/example_21b.rst
   examples/example_22.rst
   examples/example_23a.rst
   examples/example_23b.rst
   examples/example_24a.rst
   examples/example_24b.rst
   examples/example_25.rst
   examples/example_26a.rst
   examples/example_26b.rst
   examples/example_27.rst
   examples/example_28.rst
   examples/example_29.rst
   examples/example_30.rst
   examples/example_31.rst
   examples/example_32.rst
   examples/example_33.rst
   examples/example_34.rst
   examples/example_35.rst
   examples/example_36.rst
   examples/example_37.rst
   examples/example_38.rst
   examples/example_39.rst
   examples/example_39b.rst
   examples/example_40.rst
   examples/example_41.rst
   examples/example_42a.rst
   examples/example_42b.rst
   examples/example_43.rst


Example Datasets
~~~~~~~~~~~~~~~~

The examples make use of a small number of sample datasets. These are
available to view and download from a
`dedicated directory in the codebase repository <https://github.com/NCAS-CMS/cf-plot/tree/main/docs/source/data>`_.
Alternatively, they may be downloaded together as a zip file
``example-datasets.zip`` (total size on the order of 100 MB)
from :download:`here <_downloads/example-datasets.zip>`:

.. code-block:: console
   :caption: *Accessing all of the datasets for the cf-plot examples.*

   $ unzip -q example-datasets.zip
   $ ls -1 example-datasets                                                                       ─╯
   20160601-05T0000Z_INCOMPASS_km4p4_uv_RH_500.nc
   curvi_salinity.nc
   dsg_trajectory.nc
   ff_trs_pos.nc
   Geostropic_Adjustment.nc
   ggap.nc
   lfric_initial.nc
   orca2.nc
   rgp.nc
   synop_data.txt
   tas_A1.nc
   ukcp_rcm_test.nc
   vaAMIPlcd_DJF.nc
   wapAMIPlcd_DJF.nc


Gallery
-------

.. grid:: 1 3 3 3
   :padding: 0
   :margin: 0
   :gutter: 1

   .. grid-item-card:: Contour plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_1.png
      :link: cylindrical.html#example-1-basic-contour-plot-in-default-projection
      :text-align: center

   .. grid-item-card:: Blockfill plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_2.png
      :link: cylindrical.html#example-2-basic-blockfill-plot-in-default-projection
      :text-align: center

   .. grid-item-card:: Vector plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_13.png
      :link: vectors.html#example-13-basic-vector-plot
      :text-align: center

   .. grid-item-card:: Stream plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_16c.png
      :link: vectors.html#example-16c-stream-plot-in-a-colour-scale
      :text-align: center

   .. grid-item-card:: Stipple plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_17.png
      :link: stipple_plots.html#example-17-basic-stipple-plot
      :text-align: center

   .. grid-item-card:: Trajectories
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_42b.png
      :link: trajectories.html#example-42b-tracks-displayed-in-a-colour-scale
      :text-align: center

   .. grid-item-card:: Hovmöller plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_11.png
      :link: hovmuller.html#example-11-latitude-time-subset-view-hovmoller-plot
      :text-align: center

   .. grid-item-card:: Vertical (pressure or height) plots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_8.png
      :link: pressure.html#example-8-latitude-against-log-of-pressure-over-longitude-zonal-mean
      :text-align: center

   .. grid-item-card:: Lineplots
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_27.png
      :link: graphs.html#example-27-basic-line-plot
      :text-align: center

   .. grid-item-card:: Support for multiple projections
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_34.png
      :link: projections.html#example-34-cropping-the-lambert-conformal-conic-lcc-projection
      :text-align: center

   .. grid-item-card:: Polar projection views
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_40.png
      :link: trajectories.html#example-40-tracks-in-the-polar-stereographic-projection
      :text-align: center

   .. grid-item-card:: Unstructured grids (UGRID) support
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_24a.png
      :link: unstructured.html#example-24a-ugrid-blockfill-plot-with-lfric-cubed-sphere-mesh-output
      :text-align: center

   .. grid-item-card:: Rotated pole support
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_22.png
      :link: rotated_pole.html#example-22-plot-of-rotated-pole-data-on-its-native-grid
      :text-align: center

   .. grid-item-card:: Multiple plots on one figure
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_19b.png
      :link: multiple_plots.html#example-19b-multiple-plots-with-user-specified-plot-positions
      :text-align: center

   .. grid-item-card:: Flexible customisation e.g. user-defined axes
      :img-top: /../../cfplot/test/reference-example-images/ref_fig_21a.png
      :link: user_defined.html#example-21a-user-defined-axes
      :text-align: center
