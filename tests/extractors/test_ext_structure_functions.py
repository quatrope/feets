from feets.extractors import ext_structure_functions

import numpy as np


def test_StructureFunctions_extract(normal_light_curve):
    # create the extractor
    extractor = ext_structure_functions.StructureFunctions()
    features = [
        "StructureFunction_index_21",
        "StructureFunction_index_31",
        "StructureFunction_index_32",
    ]

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    sims = 1000
    size = 1000

    time = np.arange(size)
    values = np.empty([sims, 3])
    for idx in range(sims):
        lc = normal_light_curve(random=random, size=size, data=["magnitude"])
        lc["time"] = time

        results = extractor.extract(**lc)
        for index, feature in enumerate(features):
            values[idx, index] = results[feature]

    # test
    expected = [
        1.8438983006429244,
        2.637129298119476,
        1.525586514233299,
    ]
    np.testing.assert_allclose(values.mean(axis=0), expected)
