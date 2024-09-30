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

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class AMP(Extractor):
    r"""Calculate the AMP feature.

    .. math::

        AMP = \log_{10}(\sqrt{N_{obs}})

    """

    data = ["time", "magnitude"]
    features = ["AMP"]

    def fit(self, time, magnitude):
        sort_mask = np.argsort(time)
        mags = magnitude[sort_mask]
        fluxs = 10 ** (mags / -2.5)
        count, std_flux, mean_flux = len(fluxs), np.std(fluxs), np.mean(fluxs)
        amp = np.log10(np.sqrt(count) * std_flux / mean_flux)
        return {"AMP": amp}
