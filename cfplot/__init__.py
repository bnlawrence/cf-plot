"""
cf-plot: code-light plotting for earth science and aligned research

Documentation is hosted and found at: https://ncas-cms.github.io/cf-plot/
"""

__author__ = "Andy Heaps"
__maintainer__ ="Sadie Bartholomew"
__date__ = "28th April 2025"
__version__ = "3.4.0"


from .cfplot import *  # noqa: F403, F401

# Check for the minimum cf-python version
cf_version_min = "3.17.0"
errstr = (
    f"cf-python >= {cf_version_min} needs to be installed to use cf-plot"
)
if StrictVersion(cf.__version__) < StrictVersion(cf_version_min):
    raise Warning(errstr)
# TODO add these checks for all other dependencies too?

# Check for a display and use the Agg backing store if none is present
# This is for batch mode processing
try:
    disp = os.environ["DISPLAY"]
except Exception:
    matplotlib.use("Agg")

# Check for user setting of pre_existing_data_dir pointing to central
# cartopy setup
# This is used in the cfview simple setup process
try:
    pre_existing_data_dir = os.environ["pre_existing_data_dir"]
    cartopy.config["pre_existing_data_dir"] = pre_existing_data_dir
except KeyError:
    pass
