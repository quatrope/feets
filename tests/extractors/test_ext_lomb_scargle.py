from feets.extractors import ext_lomb_scargle

import numpy as np

import pytest


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_LombScargle_extract(periodic_light_curve):
    # create the extractor
    lscargle_kwds = {
        "autopower_kwds": {
            "normalization": "standard",
            "nyquist_factor": 1,
        }
    }
    extractor = ext_lomb_scargle.LombScargle(lscargle_kwds=lscargle_kwds)
    features = [
        "PeriodLS",
        "Period_fit",
        "Psi_CS",
        "Psi_eta",
    ]

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    sims = 100
    size = 100

    time = np.arange(size)
    values = np.empty([sims, 4])
    for idx in range(sims):
        lc = periodic_light_curve(
            random=random, size=size, data=["magnitude"], magnitude_period=20
        )
        lc["time"] = time

        results = extractor.extract(**lc)
        for index, feature in enumerate(features):
            np.testing.assert_(len(results[feature]) == 3)
            values[idx, index] = results[feature][0]

    # test
    expected = [
        20.262508344699437,
        1.4306433603192423e-11,
        0.23181927251239132,
        0.9003366875414928,
    ]
    np.testing.assert_allclose(values.mean(axis=0), expected)
