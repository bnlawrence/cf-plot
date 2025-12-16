API reference
*************


API
===


.. toctree::
   :maxdepth: 2




Plotting routines
-----------------

.. toctree::
   :maxdepth: 2

   api/con
   api/lineplot
   api/stipple
   api/stream
   api/traj
   api/vect


Graphics related
----------------

.. toctree::
   :maxdepth: 2

   api/gopen
   api/gclose
   api/gpos


Parameter related
-----------------

.. toctree::
   :maxdepth: 2

   api/axes
   api/cscale
   api/gset
   api/levs
   api/mapset
   api/plotvars
   api/reset
   api/setvars


Mapping related
---------------

.. toctree::
   :maxdepth: 2

   api/axes_plot
   api/map_grid


Colour related
--------------

.. toctree::
   :maxdepth: 2

   api/cbar


Calculations
------------

.. toctree::
   :maxdepth: 2

   api/calculate_levels


Utility Routines
----------------

.. toctree::
   :maxdepth: 2

   api/add_cyclic
   api/cf_var_name
   api/cf_var_name_titles
   api/find_dim_names
   api/find_pos_in_array
   api/find_z
   api/fix_floats
   api/generate_titles
   api/irregular_window
   api/max_ndecs_data
   api/ndecs
   api/pcon
   api/polar_regular_grid
   api/rgaxes
   api/stipple_points
   api/vloc


Validation
----------

.. toctree::
   :maxdepth: 2

   api/check_well_formed
   api/orca_check


Deprecated and Obsolete API
===========================

Deprecations
------------

These functions *are not* intended for user use and will be removed in the next
version. In most cases these have no use in the codebase as of v3.3.0.

.. toctree::
   :maxdepth: 2

   regrid

Note the functions ``compare_arrays`` and ``regression_tests`` (version 3.3.0
and earlier) related to testing only so have been moved out of the
functional codebase.


Obsolete Internal API
---------------------

These routines were exposed in the user-facing API at versions 3.3.0 and
below but were marked as not being intended for user use. They have
therefore been removed except for under-the-hood processing at version 3.5.0.
They are listed here for completeness and posterity.

If you made use of any of the functionality provided by these internal-use
routines, and wish to have this in future, please
:ref:`get in touch <ways-to-contact-us>` to let
us know and we may be able to support them again.

* ``_bfill``
* ``_bfill_ugrid``
* ``_cf_data_assign``
* ``_check_data``
* ``_cscale_get_map``
* ``_dim_titles``
* ``_gvals``
* ``_map_title``
* ``_mapaxis``
* ``_plot_map_axes``
* ``_process_color_scales``
* ``_set_map``
* ``_supscr``
* ``_timeaxis``
