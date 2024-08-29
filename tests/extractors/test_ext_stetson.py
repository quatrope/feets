import numpy as np

from feets.extractors import ext_stetson

import pytest


@pytest.mark.skip(reason="FATS says it must be 0, but actual result is 0.35")
def test_StetsonJ_extract():
    # create the extractor
    extractor = ext_stetson.StetsonL()

    # init the seed
    random = np.random.default_rng(seed=42)

    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        mags2 = mags * random.uniform(low=0, high=1.5, size=mags.size)

        errors = random.normal(scale=0.001, size=1000)
        errors2 = random.normal(scale=0.001, size=1000)

        values[idx] = extractor.extract(
            aligned_magnitude=mags,
            aligned_magnitude2=mags2,
            aligned_error=errors,
            aligned_error2=errors2,
        )["StetsonL"]

    np.testing.assert_allclose(values.mean(), 0)


@pytest.mark.skip(
    reason="FATS says it must be 2/pi, but actual result is 0.20"
)
def test_StetsonK_extract():
    # create the extractor
    extractor = ext_stetson.StetsonK()

    # init the seed
    random = np.random.default_rng(seed=42)

    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        errors = random.normal(scale=0.001, size=1000)

        values[idx] = extractor.extract(
            magnitude=mags,
            error=errors,
        )["StetsonK"]

    np.testing.assert_allclose(values.mean(), 0.79914938521401002)


@pytest.mark.skip(reason="FATS says it must be 0, but actual result is 0.35")
def test_StetsonL_extract():
    # create the extractor
    extractor = ext_stetson.StetsonL()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        mags2 = mags * random.uniform(low=0, high=1.5, size=mags.size)

        errors = random.normal(scale=0.001, size=1000)
        errors2 = random.normal(scale=0.001, size=1000)

        values[idx] = extractor.extract(
            aligned_magnitude=mags,
            aligned_magnitude2=mags2,
            aligned_error=errors,
            aligned_error2=errors2,
        )["StetsonL"]

    np.testing.assert_allclose(values.mean(), 0.0085957106316273714)
