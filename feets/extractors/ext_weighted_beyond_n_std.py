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

"""Beyond-one-standard-deviation extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class WeightedBeyondNStd(Extractor):
    """Beyond-one-standard-deviation extractor.

    **Beyond1Std**

    Percentage of points beyond one standard deviation from the weighted mean.
    For a normal distribution, it should take a value close to :math:`0.32`.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['Beyond1Std'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Beyond1Std': 0.327}

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["WeightedBeyondNStd"]

    def __init__(self, nstd=1):
        if nstd <= 0:
            raise ValueError("nstd should be positive")

        self.nstd = nstd

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, error):
        n = len(magnitude)

        weighted_mean = np.average(magnitude, weights=1 / error**2)

        # Standard deviation with respect to the weighted mean

        var = sum((magnitude - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(
            np.logical_or(
                magnitude > weighted_mean + self.nstd * std,
                magnitude < weighted_mean - self.nstd * std,
            )
        )

        return {"WeightedBeyondNStd": float(count) / n}

    @doctools.doc_inherit(Extractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "WeightedBeyondNStd":
            N = self.nstd
            return {f"WeightedBeyond{N}Std": value}
        return super().flatten_feature(feature, value)
