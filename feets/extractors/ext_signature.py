#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Signature(Extractor):
    features = ["Signature"]

    def __init__(self, phase_bins=18, mag_bins=12):
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins

    def extract(self, magnitude, time, PeriodLS, Amplitude):
        phase_bins, mag_bins = self.phase_bins, self.mag_bins
        bins = (phase_bins, mag_bins)
        labels = tuple(
            f"ph_{j}_mag_{i}"
            for i in range(mag_bins)
            for j in range(phase_bins)
        )

        lc_yaxis = (magnitude - np.min(magnitude)) / np.float64(Amplitude)

        # SHIFT TO BEGIN AT MINIMUM
        loc = np.argmin(lc_yaxis)

        signatures = np.full(len(PeriodLS), None, dtype=object)
        for idx, period_ls in enumerate(PeriodLS):
            lc_phases = np.remainder(time - time[loc], period_ls) / period_ls

            count = np.histogram2d(
                lc_phases, lc_yaxis, bins=bins, density=True
            )[0]

            signature = zip(labels, count.reshape(phase_bins * mag_bins))
            signatures[idx] = dict(signature)

        return {"Signature": signatures}
