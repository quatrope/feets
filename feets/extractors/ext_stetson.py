#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# DOC
# =============================================================================

r"""These three features are based on the Welch/Stetson variability
index :math:`I` (Stetson, 1996) defined by the equation:

.. math::

    I = \sqrt{\frac{1}{n(n-1)}} \sum_{i=1}^n {
        (\frac{b_i-\hat{b}}{\sigma_{b,i}})
        (\frac{v_i - \hat{v}}{\sigma_{v,i}})}

where \:math:`b_i` and :math:`v_i` are the apparent magnitudes obtained for
the candidate star in two observations closely spaced in time on some occasion
:math:`i`, :math:`\sigma_{b, i}` and :math:`\sigma_{v, i}` are the standard
errors of those magnitudes, :math:`\hat{b}` and \hat{v} are the weighted mean
magnitudes in the two filters, and :math:`n` is the number of observation
pairs.

Since a given frame pair may include data from two filters which did not have
equal numbers of observations overall, the "relative error" is calculated as
follows:

.. math::

    \delta = \sqrt{\frac{n}{n-1}} \frac{v-\hat{v}}{\sigma_v}

allowing all residuals to be compared on an equal basis.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..utils import indent
from .core import Extractor
from .ext_slotted_a_length import SlottedA_length


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StetsonJ(Extractor):
    __doc__ = indent(__doc__) + r"""

    **StetsonJ**

    Stetson J is a robust version of the variability index. It is calculated
    based on two simultaneous light curves of a same star and is defined as:

    .. math::

        J =  \sum_{k=1}^n  sgn(P_k) \sqrt{|P_k|}

    with :math:`P_k = \delta_{i_k} \delta_{j_k}`

    For a Gaussian magnitude distribution, J should take a value close to zero:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['StetsonJ'])
        >>> rs = fs.extract(**lc_normal)
        >>> rs.as_dict()
        {'StetsonJ': 0.010765631555204736}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    data = ['aligned_magnitude', 'aligned_magnitude2',
            'aligned_error', 'aligned_error2']
    features = ["StetsonJ"]
    warnings = [
        ("The original FATS documentation says that the result of StetsonJ "
         "must be ~0 for gaussian distribution but the result is ~-0.41")]

    def fit(self, aligned_magnitude, aligned_magnitude2,
            aligned_error, aligned_error2):

        N = len(aligned_magnitude)

        mean_mag = (
            np.sum(aligned_magnitude / (aligned_error * aligned_error)) /
            np.sum(1.0 / (aligned_error * aligned_error)))

        mean_mag2 = (
            np.sum(aligned_magnitude2 / (aligned_error2 * aligned_error2)) /
            np.sum(1.0 / (aligned_error2 * aligned_error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude[:N] - mean_mag) /
                  aligned_error)
        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude2[:N] - mean_mag2) /
                  aligned_error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) * np.sum(np.sign(sigma_i) *
             np.sqrt(np.abs(sigma_i))))

        return {"StetsonJ": J}


class StetsonK(Extractor):
    __doc__ = indent(__doc__) + r"""

    **StetsonK**

    Stetson K is a robust kurtosis measure:

    .. math::

        \frac{1/N \sum_{i=1}^N |\delta_i|}{\sqrt{1/N \sum_{i=1}^N \delta_i^2}}

    where the index :math:`i` runs over all :math:`N` observations available
    for the star without regard to pairing. For a Gaussian magnitude
    distribution K should take a value close to :math:`\sqrt{2/\pi} = 0.798`:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['StetsonK'])
        >>> rs = fs.extract(**lc_normal)
        >>> rs.as_dict()
        {'StetsonK': 0.79914938521401002}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    data = ['magnitude', 'error']
    features = ['StetsonK']
    warnings = [
        ("The original FATS documentation says that the result of StetsonK "
         "must be 2/pi=0.798 for gaussian distribution but the "
         "result is ~0.2")]

    def fit(self, magnitude, error):
        mean_mag = (np.sum(magnitude / (error * error)) /
                    np.sum(1.0 / (error * error)))

        N = len(magnitude)
        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude - mean_mag) / error)

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        return {"StetsonK": K}


class StetsonKAC(Extractor):
    __doc__ = indent(__doc__) + r"""

    **StetsonK_AC**

    Stetson K applied to the slotted autocorrelation function of the
    light-curve.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['SlottedA_length','StetsonK_AC'])
        >>> rs = fs.extract(**lc_normal)
        >>> rs.as_dict()
        {'SlottedA_length': 1.0, 'StetsonK_AC': 0.20917402545294403}

    **Parameters**

    - ``T``: tau - slot size in days (default=1).

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    data = ['magnitude', 'time', 'error']
    features = ["StetsonK_AC"]
    params = {"T": 1}

    def fit(self, magnitude, time, error, T):
        sal = SlottedA_length(T=T)
        autocor_vector = sal.start_conditions(
            magnitude, time, **sal.params)[-1]

        N_autocor = len(autocor_vector)
        sigmap = (np.sqrt(N_autocor * 1.0 / (N_autocor - 1)) *
                  (autocor_vector - np.mean(autocor_vector)) /
                  np.std(autocor_vector))

        K = (1 / np.sqrt(N_autocor * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        return {"StetsonK_AC": K}


class StetsonL(Extractor):
    __doc__ = indent(__doc__) + r"""

    **StetsonL**

    Stetson L variability index describes the synchronous variability of
    different bands and is defined as:

    .. math::

        L = \frac{JK}{0.798}

    Again, for a Gaussian magnitude distribution, L should take a value close
    to zero:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['SlottedL'])
        >>> rs = fs.extract(**lc_normal)
        >>> rs.as_dict()
        {'StetsonL': 0.0085957106316273714}

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    data = ['aligned_magnitude', 'aligned_magnitude2',
            'aligned_error', 'aligned_error2']
    features = ["StetsonL"]

    def fit(self, aligned_magnitude, aligned_magnitude2,
            aligned_error, aligned_error2):
        magnitude, magnitude2 = aligned_magnitude, aligned_magnitude2
        error, error2 = aligned_error, aligned_error2

        N = len(magnitude)

        mean_mag = (np.sum(magnitude / (error * error)) /
                    np.sum(1.0 / (error * error)))
        mean_mag2 = (np.sum(magnitude2 / (error2 * error2)) /
                     np.sum(1.0 / (error2 * error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude[:N] - mean_mag) /
                  error)

        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude2[:N] - mean_mag2) /
                  error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) *
             np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i))))

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i ** 2)))

        L = J * K / 0.798

        return {"StetsonL": L}
