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

"""Stetson variability index extractors."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .ext_slotted_a_length import start_conditions
from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StetsonJ(Extractor):
    r"""Stetson J variability index extractor.

    **StetsonJ**

    Stetson J is a robust version of the variability index. It is calculated
    based on two simultaneous light curves of a same star and is defined as:

    .. math::

        J =  \sum_{k=1}^n  sgn(P_k) \sqrt{|P_k|}

    with :math:`P_k = \delta_{i_k} \delta_{j_k}`

    For a Gaussian magnitude distribution, :math:`J` should take a value close
    to zero.

    Notes
    -----
    This feature is based on the Welch/Stetson variability index :math:`I`
    (Stetson, 1996) defined by the equation:

    .. math::

        I = \sqrt{\frac{1}{n(n-1)}} \sum_{i=1}^n {
            (\frac{b_i-\hat{b}}{\sigma_{b,i}})
            (\frac{v_i - \hat{v}}{\sigma_{v,i}})}

    where :math:`b_i` and :math:`v_i` are the apparent magnitudes obtained for
    the candidate star in two observations closely spaced in time on some
    occasion :math:`i`, :math:`\sigma_{b, i}` and :math:`\sigma_{v, i}` are the
    standard errors of those magnitudes, :math:`\hat{b}` and \hat{v} are the
    weighted mean magnitudes in the two filters, and :math:`n` is the number of
    observation pairs.

    Since a given frame pair may include data from two filters which did not
    have equal numbers of observations overall, the "relative error" is
    calculated as follows:

    .. math::

        \delta = \sqrt{\frac{n}{n-1}} \frac{v-\hat{v}}{\sigma_v}

    allowing all residuals to be compared on an equal basis.

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['StetsonJ'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'StetsonJ': np.float64(0.01823276018663087)}
    """

    features = ["StetsonJ"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(
        self,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ):
        """
        Parameters
        ----------
        aligned_magnitude : array-like
        aligned_magnitude2 : array-like
        aligned_error : array-like
        aligned_error2 : array-like
        """

        N = len(aligned_magnitude)

        mean_mag = np.sum(
            aligned_magnitude / (aligned_error * aligned_error)
        ) / np.sum(1.0 / (aligned_error * aligned_error))

        mean_mag2 = np.sum(
            aligned_magnitude2 / (aligned_error2 * aligned_error2)
        ) / np.sum(1.0 / (aligned_error2 * aligned_error2))

        sigmap = (
            np.sqrt(N * 1.0 / (N - 1))
            * (aligned_magnitude[:N] - mean_mag)
            / aligned_error
        )
        sigmaq = (
            np.sqrt(N * 1.0 / (N - 1))
            * (aligned_magnitude2[:N] - mean_mag2)
            / aligned_error2
        )
        sigma_i = sigmap * sigmaq

        J = (
            1.0
            / len(sigma_i)
            * np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i)))
        )

        return {"StetsonJ": J}


@doctools.doc_inherit(StetsonJ, warn_class=False)
class StetsonKAC(Extractor):
    r"""Stetson K to slotted autocorrelation extractor.

    **StetsonK_AC**

    Stetson K applied to the slotted autocorrelation function of the
    light-curve.

    Parameters
    ----------
    T : int, optional, default: `1`
        :math:`tau` - slot size in days.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['SlottedALength','StetsonK_AC'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'SlottedALength': np.int64(1),
     'StetsonK_AC': np.float64(0.6440898442951952)}
    """

    features = ["StetsonK_AC"]

    def __init__(self, T=1):
        self.T = T

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, time):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like
        """
        autocor_vector = start_conditions(magnitude, time, self.T)[-1]

        N_autocor = len(autocor_vector)
        sigmap = (
            np.sqrt(N_autocor * 1.0 / (N_autocor - 1))
            * (autocor_vector - np.mean(autocor_vector))
            / np.std(autocor_vector)
        )

        K = (
            1
            / np.sqrt(N_autocor * 1.0)
            * np.sum(np.abs(sigmap))
            / np.sqrt(np.sum(sigmap**2))
        )

        return {"StetsonK_AC": K}


@doctools.doc_inherit(StetsonJ, warn_class=False)
class StetsonL(Extractor):
    r"""Stetson L variability index extractor.

    **StetsonL**

    Stetson L variability index describes the synchronous variability of
    different bands and is defined as:

    .. math::

        L = \frac{JK}{0.798}

    Again, for a Gaussian magnitude distribution, :math:`L` should take a value
    close to zero.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['StetsonL'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'StetsonL': np.float64(0.0015499030048823923)}
    """

    features = ["StetsonL"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(
        self,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ):
        """
        Parameters
        ----------
        aligned_magnitude : array-like
        aligned_magnitude2 : array-like
        aligned_error : array-like
        aligned_error2 : array-like
        """
        magnitude, magnitude2 = aligned_magnitude, aligned_magnitude2
        error, error2 = aligned_error, aligned_error2

        N = len(magnitude)

        mean_mag = np.sum(magnitude / (error * error)) / np.sum(
            1.0 / (error * error)
        )
        mean_mag2 = np.sum(magnitude2 / (error2 * error2)) / np.sum(
            1.0 / (error2 * error2)
        )

        sigmap = (
            np.sqrt(N * 1.0 / (N - 1)) * (magnitude[:N] - mean_mag) / error
        )

        sigmaq = (
            np.sqrt(N * 1.0 / (N - 1)) * (magnitude2[:N] - mean_mag2) / error2
        )
        sigma_i = sigmap * sigmaq

        J = (
            1.0
            / len(sigma_i)
            * np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i)))
        )

        K = (
            1
            / np.sqrt(N * 1.0)
            * np.sum(np.abs(sigma_i))
            / np.sqrt(np.sum(sigma_i**2))
        )

        return {"StetsonL": J * K / 0.798}
