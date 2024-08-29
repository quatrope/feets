import numpy as np

from feets.extractors import ext_small_kurtosis


def test_Skew_extract():
    # create the extractor
    extractor = ext_small_kurtosis.SmallKurtosis()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = extractor.extract(magnitude=mags)["SmallKurtosis"]

    np.testing.assert_allclose(values.mean(), 0.01583864399587241)
