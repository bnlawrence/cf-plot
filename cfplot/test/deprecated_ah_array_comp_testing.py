import faulthandler
import functools
import hashlib
import os
import unittest
from pprint import pformat

import cartopy.crs as ccrs
import cf
import coverage
import matplotlib.testing.compare as mpl_compare
import numpy as np
from netCDF4 import Dataset as ncfile
from scipy.interpolate import griddata

import cfplot as cfp


def compare_arrays(
    ref=None,
    levs_test=None,
    gvals_test=None,
    mapaxis_test=None,
    min=None,
    max=None,
    step=None,
    mult=None,
    type=None,
):
    """
    Compare arrays and return an error string if they don't match.
    """
    plotvar_levs = cfp.plotvars.levels

    anom = 0
    if levs_test:
        cfp.levs(min, max, step)
        if np.size(ref) != np.size(plotvar_levs):
            anom = 1
        else:
            for val in np.arange(np.size(ref)):
                if abs(ref[val] - plotvar_levs[val]) >= 1e-6:
                    anom = 1

        if anom == 1:
            print(
                "***cfp.levs failure***\n"
                f"min, max, step are {min}, {max}, {step}\n"
                "generated levels are:\n"
                f"{plotvar_levs}\n"
                f"expected levels:\n{ref}"
            )
        else:
            pass_str = f"Passed cfp.levs(min={min}, max={max}, step={step})"
            print(pass_str)

    anom = 0
    if gvals_test:
        vals, testmult = cfp._gvals(min, max)
        if np.size(ref) != np.size(vals):
            anom = 1
        else:
            for val in np.arange(np.size(ref)):
                if abs(ref[val] - vals[val]) >= 1e-6:
                    anom = 1
        if mult != testmult:
            anom = 1

        if anom == 1:
            print(
                "***gvals failure***\n"
                f"cfp._gvals({min}, {max})\n\n"
                f"generated values are:{vals}\n"
                f"with a  multiplier of {testmult}\n\n"
                f"expected values:{ref}\n"
                f"with a multiplier of {mult}\n"
            )
        else:
            pass_str = f"Passed cfp._gvals({min}, {max})"
            print(pass_str)

    anom = 0
    if mapaxis_test:
        ref_ticks = ref[0]
        ref_labels = ref[1]
        test_ticks, test_labels = cfp._mapaxis(min=min, max=max, type=type)
        if np.size(test_ticks) != np.size(ref_ticks):
            anom = 1
        else:
            for val in np.arange(np.size(ref_ticks)):
                if abs(ref_ticks[val] - test_ticks[val]) >= 1e-6:
                    anom = 1
                if ref_labels[val] != test_labels[val]:
                    anom = 1

        if anom == 1:
            print(
                "***mapaxis failure***\n\n"
                f"cfp._mapaxis(min={min}, max={max}, type={type})\n"
                f"generated values are:{test_ticks}\n"
                f"with labels:{test_labels}\n\n"
                f"expected ticks:{ref_ticks}\n"
                f"with labels:{ref_labels}\n"
            )
        else:
            pass_str = (
                f"Passed cfp._mapaxis<(min={min}, max={max}, type={type})"
            )
            print(pass_str)


class BasicArrayTest(unittest.TestCase):
    """Contour levels `levs` array comparison testing."""

    def setup(self):
        """Preparations called immediately before each test method."""
        print(
            "---------------------------------\n"
            "Testing for `levs` contour levels\n"
            "---------------------------------\n"
        )

    def test_arrays_1(self):
        """Test 1 for `levs` contour levels array comparison."""
        ref_answer = [
            -35,
            -30,
            -25,
            -20,
            -15,
            -10,
            -5,
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
        ]
        compare_arrays(ref=ref_answer, levs_test=True, min=-35, max=65, step=5)

        ref_answer = [
            -6.0,
            -4.8,
            -3.6,
            -2.4,
            -1.2,
            0.0,
            1.2,
            2.4,
            3.6,
            4.8,
            6.0,
        ]
        compare_arrays(ref=ref_answer, levs_test=True, min=-6, max=6, step=1.2)

    def test_arrays_2(self):
        """Test 2 for `levs` contour levels array comparison."""
        ref_answer = [
            50000,
            51000,
            52000,
            53000,
            54000,
            55000,
            56000,
            57000,
            58000,
            59000,
            60000,
        ]
        compare_arrays(
            ref=ref_answer, levs_test=True, min=50000, max=60000, step=1000
        )

    def test_arrays_3(self):
        """Test 3 for `levs` contour levels array comparison."""
        ref_answer = [
            -7000,
            -6500,
            -6000,
            -5500,
            -5000,
            -4500,
            -4000,
            -3500,
            -3000,
            -2500,
            -2000,
            -1500,
            -1000,
            -500,
        ]
        compare_arrays(
            ref=ref_answer, levs_test=True, min=-7000, max=-300, step=500
        )


class GvalsArrayTest(unittest.TestCase):
    """Contour levels `gvals` array comparison testing."""

    def setup(self):
        """Preparations called immediately before each test method."""
        print(
            "----------------------------------\n"
            "Testing for `gvals` contour levels\n"
            "----------------------------------\n"
        )

    def test_gvals_1(self):
        """Test 1 for `gvals` contour levels array comparison."""
        ref_answer = [
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
            290,
            291,
            292,
            293,
        ]
        compare_arrays(
            ref=ref_answer,
            min=280.50619506835938,
            max=293.48431396484375,
            mult=0,
            gvals_test=True,
        )

    def test_gvals_2(self):
        """Test 2 for `gvals` contour levels array comparison."""
        ref_answer = [
            0.356,
            0.385,
            0.414,
            0.443,
            0.472,
            0.501,
            0.53,
            0.559,
            0.588,
            0.617,
            0.646,
            0.675,
        ]
        compare_arrays(
            ref=ref_answer, min=0.356, max=0.675, mult=0, gvals_test=True
        )

    def test_gvals_3(self):
        """Test 3 for `gvals` contour levels array comparison."""
        ref_answer = [
            -45,
            -40,
            -35,
            -30,
            -25,
            -20,
            -15,
            -10,
            -5,
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
        ]
        compare_arrays(
            ref=ref_answer,
            min=-49.510975,
            max=53.206604,
            mult=0,
            gvals_test=True,
        )

    def test_gvals_4(self):
        """Test 4 for `gvals` contour levels array comparison."""
        ref_answer = [
            47000,
            48000,
            49000,
            50000,
            51000,
            52000,
            53000,
            54000,
            55000,
            56000,
            57000,
            58000,
            59000,
            60000,
            61000,
            62000,
            63000,
            64000,
        ]
        compare_arrays(
            ref=ref_answer, min=46956, max=64538, mult=0, gvals_test=True
        )

    def test_gvals_5(self):
        """Test 5 for `gvals` contour levels array comparison."""
        ref_answer = [
            -1.0,
            -0.9,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
        ]
        compare_arrays(
            ref=ref_answer, min=-1.0, max=0.1, mult=0, gvals_test=True
        )


class LonLatTest(unittest.TestCase):
    """Tests for `mapaxis` longitude-latitude labelling."""

    def setup(self):
        """Preparations called immediately before each test method."""
        print("--------------------------------------------------\n")
        print("Testing for `mapaxis` longitude-latitude labelling\n")
        print("--------------------------------------------------\n")

    def test_lonlat_1(self):
        """Test 1 for `mapaxis` longitude-latitude labelling."""
        ref_answer = (
            [-180, -120, -60, 0, 60, 120, 180],
            ["180", "120W", "60W", "0", "60E", "120E", "180"],
        )
        compare_arrays(
            ref=ref_answer, min=-180, max=180, type=1, mapaxis_test=True
        )

    def test_lonlat_2(self):
        """Test 2 for `mapaxis` longitude-latitude labelling."""
        ref_answer = (
            [150, 180, 210, 240, 270],
            ["150E", "180", "150W", "120W", "90W"],
        )
        compare_arrays(
            ref=ref_answer, min=135, max=280, type=1, mapaxis_test=True
        )

    def test_lonlat_3(self):
        """Test 3 for `mapaxis` longitude-latitude labelling."""
        ref_answer = (
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            [
                "0",
                "10E",
                "20E",
                "30E",
                "40E",
                "50E",
                "60E",
                "70E",
                "80E",
                "90E",
            ],
        )
        compare_arrays(
            ref=ref_answer, min=0, max=90, type=1, mapaxis_test=True
        )

    def test_lonlat_4(self):
        """Test 4 for `mapaxis` longitude-latitude labelling."""
        ref_answer = (
            [-90, -60, -30, 0, 30, 60, 90],
            ["90S", "60S", "30S", "0", "30N", "60N", "90N"],
        )
        compare_arrays(
            ref=ref_answer, min=-90, max=90, type=2, mapaxis_test=True
        )

    def test_lonlat_5(self):
        """Test 5 for `mapaxis` longitude-latitude labelling."""
        ref_answer = (
            [0, 5, 10, 15, 20, 25, 30],
            ["0", "5N", "10N", "15N", "20N", "25N", "30N"],
        )
        compare_arrays(
            ref=ref_answer, min=0, max=30, type=2, mapaxis_test=True
        )


if __name__ == "__main__":
    print("==================\n" "Regression testing\n" "==================\n")
    cov = coverage.Coverage()
    cov.start()
    unittest.main()

    cov.stop()
    cov.save()

    cov.report()
    print("================\n" "Testing complete\n" "================\n")
