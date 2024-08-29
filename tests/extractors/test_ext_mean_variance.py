from feets.extractors import ext_mean_variance

import numpy as np


def test_MeanVariance_extract():
    # create the extractor
    extractor = ext_mean_variance.MeanVariance()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        magnitude = random.uniform(size=1000)
        values[idx] = extractor.extract(magnitude=magnitude)["Meanvariance"]

    np.testing.assert_allclose(values.mean(), 0.5770949)
