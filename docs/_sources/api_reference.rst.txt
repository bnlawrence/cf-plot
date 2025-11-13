API reference
*************


User-facing API
---------------

These functions *are* intended for user use.


.. toctree::
   :maxdepth: 2

   api/add_cyclic
   api/axes
   api/axes_plot
   api/calculate_levels
   api/cbar
   api/cf_var_name
   api/cf_var_name_titles
   api/check_well_formed
   api/con
   api/cscale
   api/find_dim_names
   api/find_pos_in_array
   api/find_z
   api/fix_floats
   api/gclose
   api/generate_titles
   api/gopen
   api/gpos
   api/gset
   api/irregular_window
   api/levs
   api/lineplot
   api/map_grid
   api/mapset
   api/max_ndecs_data
   api/ndecs
   api/orca_check
   api/pcon
   api/polar_regular_grid
   api/reset
   api/rgaxes
   api/setvars
   api/stipple
   api/stipple_points
   api/stream
   api/traj
   api/vect
   api/vloc

   
Internal API
------------

These functions *are not* intended for user use.

.. toctree::
   :maxdepth: 2

   api/_bfill
   api/_bfill_ugrid
   api/_cf_data_assign
   api/_check_data
   api/_cscale_get_map
   api/_dim_titles
   api/_gvals
   api/_map_title
   api/_mapaxis
   api/_plot_map_axes
   api/_process_color_scales
   api/_set_map
   api/_supscr
   api/_timeaxis


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
