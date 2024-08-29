import numpy as np

from feets.extractors import ext_median_abs_dev


def test_MedianAbsDev_extract():
    # create the extractor
    extractor = ext_median_abs_dev.MedianAbsDev()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = extractor.extract(magnitude=mags)["MedianAbsDev"]

    np.testing.assert_allclose(values.mean(), 0.6735277130207087)
