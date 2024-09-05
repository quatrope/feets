from feets.extractors import ext_stetson

import numpy as np

import pytest


def test_StetsonJ_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonJ()

    # init the seed
    random = np.random.default_rng(42)

    # run the simulation
    error_loc, error_scale = 1, 0.008
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=[
                "aligned_magnitude",
                "aligned_magnitude2",
                "aligned_error",
                "aligned_error2",
            ],
            aligned_error_loc=error_loc,
            aligned_error2_loc=error_loc,
            aligned_error_scale=error_scale,
            aligned_error2_scale=error_scale,
        )
        values[idx] = extractor.extract(**lc)["StetsonJ"]

    np.testing.assert_allclose(values.mean(), 0.000389878261606318)


def test_StetsonK_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonK()

    # init the seed
    random = np.random.default_rng(42)

    # run the simulation
    error_loc, error_scale = 1, 0.008
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=["magnitude", "error"],
            error_loc=error_loc,
            error_scale=error_scale,
        )
        values[idx] = extractor.extract(**lc)["StetsonK"]

    np.testing.assert_allclose(values.mean(), 0.7978257009818837)


@pytest.mark.slow
def test_StetsonKAC_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonKAC()

    # init the seed
    random = np.random.default_rng(42)

    # run the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=10000, data=["magnitude"])
        lc["time"] = np.arange(10000)
        values[idx] = extractor.extract(**lc)["StetsonK_AC"]

    print(values.mean())
    np.testing.assert_allclose(values.mean(), 0.21042263044101692)


def test_StetsonL_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonL()

    # init the seed
    random = np.random.default_rng(42)

    # run the simulation
    error_loc, error_scale = 1, 0.008
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=[
                "aligned_magnitude",
                "aligned_magnitude2",
                "aligned_error",
                "aligned_error2",
            ],
            aligned_error_loc=error_loc,
            aligned_error2_loc=error_loc,
            aligned_error_scale=error_scale,
            aligned_error2_scale=error_scale,
        )
        values[idx] = extractor.extract(**lc)["StetsonL"]

    np.testing.assert_allclose(values.mean(), 0.00030183305778540346)
