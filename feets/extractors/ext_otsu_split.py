#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# DOC
# =============================================================================

"""Otsu split extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import OtsuSplit as _OtsuSplit

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class OtsuSplit(LightCurveExtractor):
    """Otsu threshholding algorithm.

    Difference of subset means, standard deviation of the lower subset,
    standard deviation of the upper subset and lower-to-all observation count
    ratio for two subsets of magnitudes obtained by Otsu's method split.

    Otsu's method is used to perform automatic thresholding. The algorithm
    returns a single threshold that separate values into two classes. This
    threshold is determined by minimizing intra-class intensity variance, or
    equivalently, by maximizing inter-class variance.

    The algorithm returns the minimum threshold which corresponds to the
    absolute maximum of the inter-class variance.

    References
    ----------
    .. [otsu1979glh] Otsu, N. (1979).
       A Threshold Selection Method from Gray-Level Histograms.
       IEEE Transactions on Systems, Man and Cybernetics, 9, 62--66.
       doi: 10.1109/TSMC.1979.4310076
    """

    features = [
        "OtsuMeanDiff",
        "OtsuStdLower",
        "OtsuStdUpper",
        "OtsuLowerToAllRatio",
    ]

    def __init__(self):
        self._extract = _OtsuSplit()

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [
            otsu_mean_diff,
            otsu_std_lower,
            otsu_std_upper,
            otsu_lower_to_all_ratio,
        ] = self._extract(time, magnitude, error)

        return {
            "OtsuMeanDiff": otsu_mean_diff,
            "OtsuStdLower": otsu_std_lower,
            "OtsuStdUpper": otsu_std_upper,
            "OtsuLowerToAllRatio": otsu_lower_to_all_ratio,
        }
