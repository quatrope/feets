from feets.extractors import ext_beyond1_std

import numpy as np


def test_Beyond1Std_extract(normal_light_curve):
    # create the extractor
    ext = ext_beyond1_std.Beyond1Std()

    # init the seed
    seed = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            seed=seed,
            size=1000,
            error_scale=0.001,
            data=["magnitude", "error"],
        )
        values[idx] = ext.extract(**lc)["Beyond1Std"]

    # test!
    np.testing.assert_allclose(values.mean(), 0.329713)
