from pathlib import Path

import cf
import pytest

import cfplot as cfp


# Use available test data from docs directory
DATA_DIR = Path(__file__).parent.parent / "docs" / "source" / "data"
DF = DATA_DIR / "tas_A1.nc"


@pytest.mark.integration
@pytest.mark.xfail(reason="Test data handling in progress")
def test_simple():
    if not DF.exists():
        pytest.skip(f"Missing integration dataset: {DF}")

    flds = cf.read(str(DF))
    f = flds[0]
    field = f[0, 0, :, :]
    cfp.con(field)







