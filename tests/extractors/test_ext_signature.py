#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_amplitude, ext_lomb_scargle, ext_signature

import numpy as np

import pytest


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_Signature_extract(periodic_light_curve):
    # create the extractors
    phase_bins = 6
    mag_bins = 6

    extractor_sig = ext_signature.Signature(
        phase_bins=phase_bins, mag_bins=mag_bins
    )
    extractor_ls = ext_lomb_scargle.LombScargle()
    extractor_amp = ext_amplitude.Amplitude()

    feature_attrs = []
    for i in range(mag_bins):
        for j in range(phase_bins):
            feature_attrs.append(f"ph_{j}_mag_{i}")

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    sims = 100
    size = 100

    time = np.arange(size)
    values = np.empty([sims, phase_bins * mag_bins])
    for idx in range(sims):
        lc = periodic_light_curve(
            random=random, size=size, data=["magnitude"], magnitude_period=20
        )

        Amplitude = extractor_amp.extract(**lc)["Amplitude"]
        PeriodLS = extractor_ls.extract(**lc, time=time)["PeriodLS"]

        results = extractor_sig.extract(
            **lc, time=time, PeriodLS=PeriodLS, Amplitude=Amplitude
        )["Signature"]

        np.testing.assert_(len(results) == 3)
        for index, key in enumerate(feature_attrs):
            values[idx, index] = results[0][key]

    expected = [
        2.071238109392945,
        0.605183273905034,
        0.24333596730297213,
        0.09589404540593081,
        0.010818366220838163,
        0.0018183670960752788,
        0.2290716162792312,
        0.6062864999869148,
        0.6722235824671547,
        0.7046220542488407,
        0.5048057372524497,
        0.4013777017040398,
        0.06727958113501273,
        0.11815881097067889,
        0.23420706778547973,
        0.423337128676334,
        0.6285534452762233,
        1.40232316611256,
        0.03454897450202003,
        0.10337866297630369,
        0.18895026706657636,
        0.2506092723165407,
        0.7175840089781574,
        1.7167746306123846,
        0.20852227534234732,
        0.486706490984589,
        0.5465103970930273,
        0.5118751458287931,
        0.5524339468804103,
        0.7996196334751763,
        1.9041271155713761,
        0.6685079850958521,
        0.28502781354816714,
        0.12883394218816935,
        0.029090165858995598,
        0.010910202140423953,
    ]
    np.testing.assert_allclose(values.mean(axis=0), expected)
