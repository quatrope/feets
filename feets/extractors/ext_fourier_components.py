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

import copy

import numpy as np

from scipy.optimize import curve_fit

from .ext_lomb_scargle import DEFAULT_LSCARGLE_KWDS, lscargle
from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class FourierComponents(Extractor):
    r"""
    **Periodic features extracted from light-curves using Lomb–Scargle**
    **(Richards et al., 2011)**

    Here, we adopt a model where the time series of the photometric magnitudes
    of variable stars is modeled as a superposition of sines and cosines:

    .. math::

        y_i(t|f_i) = a_i\sin(2\pi f_i t) + b_i\cos(2\pi f_i t) + b_{i,\circ}

    where :math:`a` and :math:`b` are normalization constants for the
    sinusoids of frequency :math:`f_i` and :math:`b_{i,\circ}` is the
    magnitude offset.

    To find periodic variations in the data, we fit the equation above by
    minimizing the sum of squares, which we denote :math:`\chi^2`:

    .. math::

        \chi^2 = \sum_k \frac{(d_k - y_i(t_k))^2}{\sigma_k^2}

    where :math:`\sigma_k` is the measurement uncertainty in data point
    :math:`d_k`. We allow the mean to float, leading to more robust period
    estimates in the case where the periodic phase is not uniformly sampled;
    in these cases, the model light curve has a non-zero mean. This can be
    important when searching for periods on the order of the data span
    :math:`T_{tot}`. Now, define

    .. math::

        \chi^2_{\circ} = \sum_k \frac{(d_k - \mu)^2}{\sigma_k^2}

    where :math:`\mu` is the weighted mean

    .. math::

        \mu = \frac{\sum_k d_k / \sigma_k^2}{\sum_k 1/\sigma_k^2}

    Then, the generalized Lomb-Scargle periodogram is:

    .. math::

        P_f(f) = \frac{(N-1)}{2} \frac{\chi_{\circ}^2 - \chi_m^2(f)}
                                      {\chi_{\circ}^2}

    where :math:`\chi_m^2(f)` is :math:`\chi^2` minimized with respect to
    :math:`a, b` and :math:`b_{\circ}`.

    Following Debosscher et al. (2007), we fit each light curve with a linear
    term plus a harmonic sum of sinusoids:

    .. math::

        y(t) = ct + \sum_{i=1}^{3}\sum_{j=1}^{4} y_i(t|jf_i)

    where each of the three test frequencies :math:`f_i` is allowed to have
    four harmonics at frequencies :math:`f_{i,j} = jf_i`. The three test
    frequencies :math:`f_i` are found iteratively, by successfully finding and
    removing periodic signal producing a peak in :math:`P_f(f)` , where
    :math:`P_f(f)` is the Lomb-Scargle periodogram as defined above.

    Given a peak in :math:`P_f(f)`, we whiten the data with respect to that
    frequency by fitting away a model containing that frequency as well as
    components with frequencies at 2, 3, and 4 times that fundamental
    frequency (harmonics). Then, we subtract that model from the data, update
    :math:`\chi_{\circ}^2`, and recalculate :math:`P_f(f)` to find more
    periodic components.

    **Algorithm:**

    1. For :math:`i = {1, 2, 3}`
    2. Calculate Lomb-Scargle periodogram :math:`P_f(f)` for light curve.
    3. Find peak in :math:`P_f(f)`, subtract that model from data.
    4. Update :math:`\chi_{\circ}^2`, return to *Step 1*.

    Then, the features extracted are given as an amplitude and a phase:

    .. math::

        A_{i,j} = \sqrt{a_{i,j}^2 + b_{i,j}^2}\\
        \textrm{PH}_{i,j} = \arctan(\frac{b_{i,j}}{a_{i,j}})

    where :math:`A_{i,j}` is the amplitude of the :math:`j-th` harmonic of the
    :math:`i-th` frequency component and :math:`\textrm{PH}_{i,j}` is the
    phase component, which we then correct to a relative phase with respect
    to the phase of the first component:

    .. math::

        \textrm{PH}'_{i,j} = \textrm{PH}_{i,j} - \textrm{PH}_{00}

    and remapped to :math:`|-\pi, +\pi|`

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.


    """

    features = [
        "Freq1_harmonics_amplitude_0",
        "Freq1_harmonics_amplitude_1",
        "Freq1_harmonics_amplitude_2",
        "Freq1_harmonics_amplitude_3",
        "Freq2_harmonics_amplitude_0",
        "Freq2_harmonics_amplitude_1",
        "Freq2_harmonics_amplitude_2",
        "Freq2_harmonics_amplitude_3",
        "Freq3_harmonics_amplitude_0",
        "Freq3_harmonics_amplitude_1",
        "Freq3_harmonics_amplitude_2",
        "Freq3_harmonics_amplitude_3",
        "Freq1_harmonics_rel_phase_0",
        "Freq1_harmonics_rel_phase_1",
        "Freq1_harmonics_rel_phase_2",
        "Freq1_harmonics_rel_phase_3",
        "Freq2_harmonics_rel_phase_0",
        "Freq2_harmonics_rel_phase_1",
        "Freq2_harmonics_rel_phase_2",
        "Freq2_harmonics_rel_phase_3",
        "Freq3_harmonics_rel_phase_0",
        "Freq3_harmonics_rel_phase_1",
        "Freq3_harmonics_rel_phase_2",
        "Freq3_harmonics_rel_phase_3",
    ]

    def __init__(self, lscargle_kwds=None):
        self.lscargle_kwds = (
            copy.deepcopy(DEFAULT_LSCARGLE_KWDS)
            if lscargle_kwds is None
            else dict(lscargle_kwds)
        )

    def _model(self, x, a, b, c, Freq):
        return (
            a * np.sin(2 * np.pi * Freq * x)
            + b * np.cos(2 * np.pi * Freq * x)
            + c
        )

    def _yfunc_maker(self, Freq):
        def func(x, a, b, c):
            return (
                a * np.sin(2 * np.pi * Freq * x)
                + b * np.cos(2 * np.pi * Freq * x)
                + c
            )

        return func

    def _components(self, magnitude, time, lscargle_kwds):
        time = time - np.min(time)
        A, PH = [], []

        frequency, fmax, fap = lscargle(
            time, magnitude, nfrequencies=3, **lscargle_kwds
        )
        fundamental_Freq = frequency[fmax]

        for i in range(3):
            Atemp, PHtemp = [], []
            omagnitude = magnitude

            for j in range(4):
                function_to_fit = self._yfunc_maker(
                    (j + 1) * fundamental_Freq[i]
                )
                popt0, popt1, popt2 = curve_fit(
                    function_to_fit, time, omagnitude
                )[0][:3]

                Atemp.append(np.sqrt(popt0**2 + popt1**2))
                PHtemp.append(np.arctan(popt1 / popt0))

                model = self._model(
                    time, popt0, popt1, popt2, (j + 1) * fundamental_Freq[i]
                )
                magnitude = np.array(magnitude) - model

            A.append(Atemp)
            PH.append(PHtemp)

        PH = np.asarray(PH)
        scaledPH = PH - PH[:, 0].reshape((len(PH), 1))

        return A, scaledPH

    def extract(self, magnitude, time):
        lscargle_kwds = self.lscargle_kwds
        A, sPH = self._components(magnitude, time, lscargle_kwds)
        result = {
            "Freq1_harmonics_amplitude_0": A[0][0],
            "Freq1_harmonics_amplitude_1": A[0][1],
            "Freq1_harmonics_amplitude_2": A[0][2],
            "Freq1_harmonics_amplitude_3": A[0][3],
            "Freq2_harmonics_amplitude_0": A[1][0],
            "Freq2_harmonics_amplitude_1": A[1][1],
            "Freq2_harmonics_amplitude_2": A[1][2],
            "Freq2_harmonics_amplitude_3": A[1][3],
            "Freq3_harmonics_amplitude_0": A[2][0],
            "Freq3_harmonics_amplitude_1": A[2][1],
            "Freq3_harmonics_amplitude_2": A[2][2],
            "Freq3_harmonics_amplitude_3": A[2][3],
            "Freq1_harmonics_rel_phase_0": sPH[0][0],
            "Freq1_harmonics_rel_phase_1": sPH[0][1],
            "Freq1_harmonics_rel_phase_2": sPH[0][2],
            "Freq1_harmonics_rel_phase_3": sPH[0][3],
            "Freq2_harmonics_rel_phase_0": sPH[1][0],
            "Freq2_harmonics_rel_phase_1": sPH[1][1],
            "Freq2_harmonics_rel_phase_2": sPH[1][2],
            "Freq2_harmonics_rel_phase_3": sPH[1][3],
            "Freq3_harmonics_rel_phase_0": sPH[2][0],
            "Freq3_harmonics_rel_phase_1": sPH[2][1],
            "Freq3_harmonics_rel_phase_2": sPH[2][2],
            "Freq3_harmonics_rel_phase_3": sPH[2][3],
        }
        return result
