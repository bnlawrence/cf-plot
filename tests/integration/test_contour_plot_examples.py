"""Integration tests for contour plot examples.

Migrated from cfplot/test/test_examples.py::ExamplesTest
These tests verify that con() plotting functions work with real data.
"""

from pathlib import Path

import cf
import numpy as np
import pytest

import cfplot as cfp


# Path to test data
DATA_DIR = Path(__file__).parent.parent.parent / "docs" / "source" / "data"
TEST_GEN_DIR = Path(__file__).parent.parent.parent / "generated-example-images"
TEST_GEN_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def setup_cfplot():
    """Set up cfplot for each test."""
    cfp.reset()
    yield
    cfp.reset()


@pytest.mark.integration
def test_example_1_basic_cylindrical():
    """Test Example 1: a basic cylindrical projection."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_1.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    # This should not raise
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_2_blockfill():
    """Test Example 2: cylindrical projection with blockfill."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_2.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    # This should not raise (blockfill with filled colors)
    cfp.con(f.subspace(time=15), blockfill=True, lines=False)


@pytest.mark.integration
def test_example_3_map_limits_and_levels():
    """Test Example 3: altering the map limits and contour levels."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_3.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(lonmin=-15, lonmax=3, latmin=48, latmax=60)
    cfp.levs(min=265, max=285, step=1)
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_4_north_pole_stereographic():
    """Test Example 4: north pole polar stereographic projection."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")

    f = cf.read(str(DATA_DIR / "ggap.nc"))[1]
    fname = str(TEST_GEN_DIR / "gen_fig_4.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="npstere")
    cfp.con(f.subspace(pressure=500))


@pytest.mark.integration
def test_example_5_south_pole_with_boundary():
    """Test Example 5: south pole with bounding latitude."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")

    f = cf.read(str(DATA_DIR / "ggap.nc"))[1]
    fname = str(TEST_GEN_DIR / "gen_fig_5.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="spstere", boundinglat=-30, lon_0=180)
    cfp.con(f.subspace(pressure=500))


@pytest.mark.integration
def test_example_6_latitude_pressure_plot():
    """Test Example 6: latitude-pressure plot."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")

    f = cf.read(str(DATA_DIR / "ggap.nc"))[3]
    fname = str(TEST_GEN_DIR / "gen_fig_6.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.con(f.subspace(longitude=0))


@pytest.mark.integration
def test_example_7_lat_pressure_zonal_mean():
    """Test Example 7: latitude-pressure plot of a zonal mean."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")

    f = cf.read(str(DATA_DIR / "ggap.nc"))[1]
    fname = str(TEST_GEN_DIR / "gen_fig_7.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.con(f.collapse("mean", "longitude"))


@pytest.mark.integration
def test_example_8_log_scale_pressure():
    """Test Example 8: plot showing latitude against log-scale pressure."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")

    f = cf.read(str(DATA_DIR / "ggap.nc"))[1]
    fname = str(TEST_GEN_DIR / "gen_fig_8.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.con(f.collapse("mean", "longitude"), ylog=1)


@pytest.mark.integration
@pytest.mark.xfail(reason="cf-python issue #799")
def test_example_9_longitude_pressure_plot():
    """Test Example 9: longitude-pressure plot."""
    if not (DATA_DIR / "ggap.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'ggap.nc'}")

    f = cf.read(str(DATA_DIR / "ggap.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_9.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.con(f.collapse("mean", "latitude"))


@pytest.mark.integration
def test_example_10_latitude_time_hovmuller():
    """Test Example 10: latitude-time Hovmuller plot."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_10.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.cscale("plasma")
    cfp.con(f.subspace(longitude=0), lines=0)


@pytest.mark.integration
def test_example_11_latitude_time_subset_hovmuller():
    """Test Example 11: latitude-time subset Hovmuller plot."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_11.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.gset(-30, 30, "1960-1-1", "1980-1-1")
    cfp.levs(min=280, max=305, step=1)
    cfp.cscale("plasma")
    cfp.con(f.subspace(longitude=0), lines=0)


@pytest.mark.integration
def test_example_12_longitude_time_hovmuller():
    """Test Example 12: longitude-time Hovmuller plot."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_12.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.cscale("plasma")
    cfp.con(f.subspace(latitude=0), lines=0)


@pytest.mark.integration
@pytest.mark.xfail(reason="Rotated pole with wrong field index")
def test_example_20_rotated_pole_data():
    """Test Example 20: user labelling of axes with rotated pole data."""
    if not (DATA_DIR / "Geostropic_Adjustment.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'Geostropic_Adjustment.nc'}")

    f = cf.read(str(DATA_DIR / "Geostropic_Adjustment.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_20.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    # Using first 2D slice from field  
    cfp.con(f)


@pytest.mark.integration
@pytest.mark.xfail(reason="Rotated pole with wrong field index")
def test_example_21_rotated_pole_custom_ticks():
    """Test Example 21: rotated pole data plot with custom ticks."""
    if not (DATA_DIR / "Geostropic_Adjustment.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'Geostropic_Adjustment.nc'}")

    f = cf.read(str(DATA_DIR / "Geostropic_Adjustment.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_21.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.con(
        f,
        title="test data",
        xticks=np.arange(5) * 100000 + 100000,
        yticks=np.arange(7) * 2000 + 2000,
        xlabel="x-axis",
        ylabel="z-axis",
    )


@pytest.mark.integration
def test_example_21b_rgp_plasma():
    """Test Example 21b: RGP data with plasma colorscale."""
    if not (DATA_DIR / "rgp.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'rgp.nc'}")

    f = cf.read(str(DATA_DIR / "rgp.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_21b.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.cscale("plasma")
    cfp.con(f)


@pytest.mark.integration
def test_example_22_rgp_gray():
    """Test Example 22: RGP data with gray colorscale."""
    if not (DATA_DIR / "rgp.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'rgp.nc'}")

    f = cf.read(str(DATA_DIR / "rgp.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_22.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.cscale("gray")
    cfp.con(f)


@pytest.mark.integration
@pytest.mark.xfail(reason="rgaxes vloc error with empty array")
def test_example_22b_rgp_rotated_projection():
    """Test Example 22b: RGP data with rotated projection."""
    if not (DATA_DIR / "rgp.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'rgp.nc'}")

    f = cf.read(str(DATA_DIR / "rgp.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_22b.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.cscale("plasma")
    cfp.mapset(proj="rotated")
    cfp.con(f)


@pytest.mark.integration
def test_example_34_lambert_conformal():
    """Test Example 34: Cropped Lambert conformal projection."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_34.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="lcc", lonmin=-50, lonmax=50, latmin=20, latmax=85)
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_35_mollweide_projection():
    """Test Example 35: Mollweide projection."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_35.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="moll")
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_36_mercator_projection():
    """Test Example 36: Mercator projection."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_36.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="merc")
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_37_orthographic_projection():
    """Test Example 37: Orthographic projection."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_37.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="ortho")
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_38_robinson_projection():
    """Test Example 38: Robinson projection."""
    if not (DATA_DIR / "tas_A1.nc").exists():
        pytest.skip(f"Missing test data: {DATA_DIR / 'tas_A1.nc'}")

    f = cf.read(str(DATA_DIR / "tas_A1.nc"))[0]
    fname = str(TEST_GEN_DIR / "gen_fig_38.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    cfp.mapset(proj="robin")
    cfp.con(f.subspace(time=15))


@pytest.mark.integration
def test_example_numpy_arrays():
    """Test contour with raw numpy arrays."""
    # Create synthetic 2D data
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    field = np.sin(X) * np.cos(Y)

    fname = str(TEST_GEN_DIR / "gen_fig_numpy.png")
    cfp.setvars(file=fname, viewer="matplotlib")

    # Should accept raw arrays
    cfp.con(f=field, x=x, y=y)
