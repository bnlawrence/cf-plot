from pathlib import Path

import cf
import pytest

import cfplot as cfp


DF = Path.home() / "data" / "bnl_tmp_NAEW.nc"


@pytest.mark.integration
def test_simple():
   if not DF.exists():
      pytest.skip(f"Missing integration dataset: {DF}")

   flds = cf.read(str(DF))
   f = flds[0]
   field = f[0, 0, :, :]
   cfp.con(field)







