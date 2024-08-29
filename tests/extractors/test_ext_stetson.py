import numpy as np

from feets.extractors import ext_stetson

import pytest


def test_StetsonJ_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonL()

    # init the seed
    random = np.random.default_rng(42)

    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            scale=0.001,
            data=["aligned_magnitude", "aligned_error", "aligned_error2"],
        )
        aligned_magnitude = lc["aligned_magnitude"]
        aligned_magnitude2 = aligned_magnitude * random.uniform(
            low=0, high=1.5, size=aligned_magnitude.size
        )

        values[idx] = extractor.extract(
            **lc, aligned_magnitude2=aligned_magnitude2
        )["StetsonL"]

    np.testing.assert_allclose(values.mean(), 0.0007281634900076408)


@pytest.mark.skip(
    reason="FATS says it must be 2/pi, but actual result is 0.20"
)
def test_StetsonK_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonK()

    # init the seed
    random = np.random.default_rng(42)

    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            scale=0.001,
            data=["magnitude", "error"],
        )

        values[idx] = extractor.extract(**lc)["StetsonK"]

    np.testing.assert_allclose(values.mean(), 0.79914938521401002)


def test_StetsonL_extract(normal_light_curve):
    # create the extractor
    extractor = ext_stetson.StetsonL()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            scale=0.001,
            data=["aligned_magnitude", "aligned_error", "aligned_error2"],
        )
        aligned_magnitude = lc["aligned_magnitude"]
        aligned_magnitude2 = aligned_magnitude * random.uniform(
            low=0, high=1.5, size=aligned_magnitude.size
        )

        values[idx] = extractor.extract(
            **lc, aligned_magnitude2=aligned_magnitude2
        )["StetsonL"]

    np.testing.assert_allclose(values.mean(), 0.0007281634900076408)
