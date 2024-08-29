import numpy as np

from feets.extractors import ext_rcs


def test_RCS_extract():
    # create the extractor
    extractor = ext_rcs.RCS()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.uniform(size=1000)
        values[idx] = extractor.extract(magnitude=mags)["Rcs"]

    np.testing.assert_allclose(values.mean(), 0.038746172489149244)
