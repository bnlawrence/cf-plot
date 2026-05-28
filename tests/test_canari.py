from pathlib import Path

import cf
import pytest

import cfplot as cfp
from tests.integration.test_contour_plot_examples import _configure_example_output


# This is a specific test for a bug report
DATA_DIR = Path(__file__).parent / "data"
DF = DATA_DIR / "bnl_tmp_NAEW.nc"


@pytest.mark.integration
def test_canari_1():
    if not DF.exists():
        pytest.skip(f"Missing integration dataset: {DF}")
    _configure_example_output("canari_1")
    flds = cf.read(str(DF))
    f = flds[0]
    field = f[0, 3, :, :]
    cfp.mapset(proj="rotated")
    cfp.con(field, lines=False)






