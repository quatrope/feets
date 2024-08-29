import numpy as np

from feets.extractors import ext_skew


def test_Skew_extract():
    # create the extractor
    extractor = ext_skew.Skew()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = extractor.extract(magnitude=mags)["Skew"]

    np.testing.assert_allclose(values.mean(), -0.0012018544388092045)
