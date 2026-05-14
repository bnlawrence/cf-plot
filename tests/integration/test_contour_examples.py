"""Integration tests for contour plotting with reference data."""

from pathlib import Path

import cf
import pytest

import cfplot as cfp


# Path to test data
DATA_DIR = Path(__file__).parent.parent.parent / "docs" / "source" / "data"
REF_IMAGE_DIR = Path(__file__).parent.parent.parent / "cfplot" / "test" / "reference-example-images"


@pytest.mark.integration
def test_contour_basic_tas():
    """Test basic contour plot with temperature data."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")
    
    flds = cf.read(str(DATA_DIR / "tas_A1.nc"))
    f = flds[0]
    
    # Extract a 2D slice for contouring
    f_2d = f.subspace(time=0)
    
    # This should not raise
    cfp.con(f_2d)


@pytest.mark.integration
def test_contour_ggap_data():
    """Test contour plot with GGAP dataset."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")
    
    flds = cf.read(str(DATA_DIR / "ggap.nc"))
    
    # Test multiple fields from GGAP
    for idx in [0, 1]:
        if idx < len(flds):
            f = flds[idx]
            f_2d = f.subspace(time=0) if f.has_construct("T") else f
            cfp.con(f_2d)


@pytest.mark.integration
def test_contour_orca_grid():
    """Test contour plot with ORCA grid (irregular/tripolar)."""
    if not (DATA_DIR / "orca2.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'orca2.nc'}")
    
    flds = cf.read(str(DATA_DIR / "orca2.nc"))
    if len(flds) > 0:
        f = flds[0]
        # This is an irregular grid, should be detected automatically
        cfp.con(f)


@pytest.mark.integration
def test_contour_rotated_pole():
    """Test contour plot with rotated pole grid."""
    if not (DATA_DIR / "Geostropic_Adjustment.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'Geostropic_Adjustment.nc'}")
    
    flds = cf.read(str(DATA_DIR / "Geostropic_Adjustment.nc"))
    if len(flds) > 0:
        f = flds[0]
        cfp.con(f)