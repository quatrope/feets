import numpy as np

from feets.extractors import ext_median_abs_dev


def test_MedianAbsDev_extract(normal_light_curve):
    # create the extractor
    extractor = ext_median_abs_dev.MedianAbsDev()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=["magnitude"],
        )
        values[idx] = extractor.extract(**lc)["MedianAbsDev"]

    np.testing.assert_allclose(values.mean(), 0.6735277130207087)
