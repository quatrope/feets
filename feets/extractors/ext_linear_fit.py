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

"""Linear fit extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import LinearFit as _LinearFit

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearFit(LightCurveExtractor):
    r"""Linear fit extractor.

    The slope, its error and reduced chi-squared of the light curve in the
    linear fit.

    Least squares fit of the linear stochastic model with Gaussian noise
    described by observation errors :math:`\{\delta_i\}`:

    .. math::

        m_i = c + \mathrm{slope} t_i + \delta_i \varepsilon_i

    where :math:`c` is a constant, :math:`\{\varepsilon_i\}` are standard
    distributed random variables.

    Feature values are :math:`\mathrm{slope}`, :math:`\sigma_\mathrm{slope}`
    and
    :math:`\frac{\sum{((m_i - c - \mathrm{slope} t_i) / \delta_i)^2}}{N - 2}`.

    Parameters
    ----------
    transform : str or bool or None, optional
        Transformer to apply to the feature values. If str, must be one of:

        - 'default' - use default transformer for the feature, it same as
          giving True. The default for this feature is 'identity'
        - 'arcsinh' - Hyperbolic arcsine feature transformer
        - 'clipped_lg' - Decimal logarithm of a value clipped to a minimum
          value
        - 'identity' - Identity feature transformer
        - 'lg' - Decimal logarithm feature transformer
        - 'ln1p' - :math:`ln(1+x)` feature transformer
        - 'sqrt' - Square root feature transformer

        If bool, must be True to use default transformer or False to disable.
        If None, no transformation is applied.
    """

    features = [
        "LinearFit_Slope",
        "LinearFit_Sigma",
        "LinearFit_ReducedChi2",
    ]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _LinearFit(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error):
        """
        Parameters
        ----------
        time : array-like
        magnitude : array-like
        error : array-like
        """
        [slope, slope_sigma, reduced_chi2] = self._extract(
            time, magnitude, error
        )
        return {
            "LinearFit_Slope": slope,
            "LinearFit_Sigma": slope_sigma,
            "LinearFit_ReducedChi2": reduced_chi2,
        }
