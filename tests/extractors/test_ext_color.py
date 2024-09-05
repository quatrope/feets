from feets.extractors import ext_color

import numpy as np


def test_Color_extract(normal_light_curve):
    # create the extractor
    extractor = ext_color.Color()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random, size=1000, data=["magnitude", "magnitude2"]
        )
        values[idx] = extractor.extract(**lc)["Color"]

    np.testing.assert_allclose(values.mean(), -0.0013145675264525064)