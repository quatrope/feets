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

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    iters = 100
    size = 100

    time = np.arange(size)

    PeriodLS = np.empty(iters)
    Period_fit = np.empty(iters)
    Psi_CS = np.empty(iters)
    Psi_eta = np.empty(iters)

    for idx in range(iters):
        lc = periodic_light_curve(
            random=random, size=size, data=["magnitude"], magnitude_period=20
        )
        lc["time"] = time

        results = extractor.extract(**lc)
        for value in results.values():
            assert len(value) == 3

        # save the results related to the best period
        PeriodLS[idx], Period_fit[idx], Psi_CS[idx], Psi_eta[idx] = (
            results["PeriodLS"][0],
            results["Period_fit"][0],
            results["Psi_CS"][0],
            results["Psi_eta"][0],
        )

    # check the results
    np.testing.assert_allclose(PeriodLS.mean(), 20.262508344699437)
    np.testing.assert_allclose(Period_fit.mean(), 1.4306433603192423e-11)
    np.testing.assert_allclose(Psi_CS.mean(), 0.23181927251239132)
    np.testing.assert_allclose(Psi_eta.mean(), 0.9003366875414928)
