import numpy as np

from feets.extractors import ext_dmdt


def test_DeltamDeltat_extract():
    # create the extractor
    extractor = ext_dmdt.DeltamDeltat()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    time = np.arange(0, 1000)
    values = np.empty(50)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        feats = extractor.extract(magnitude=mags, time=time)
        values[idx] = np.sum(list(feats.values()))

    print(values.mean())
    np.testing.assert_allclose(values.mean(), 425.86)
