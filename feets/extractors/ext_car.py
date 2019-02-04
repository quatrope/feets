#!/usr/bin/env python

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

""""""


# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

from scipy.optimize import minimize

from .core import Extractor, FeatureExtractionWarning


# =============================================================================
# CONST
# =============================================================================

EPSILON = 1e-300

CTE_NEG = -np.infty


# =============================================================================
# FUNCTIONS
# =============================================================================

def _car_like(parameters, t, x, error_vars):
    sigma, tau = parameters

    t, x, error_vars = t.flatten(), x.flatten(), error_vars.flatten()

    b = np.mean(x) / tau
    num_datos = np.size(x)

    Omega = [(tau * (sigma ** 2)) / 2.]
    x_hat = [0.]
    x_ast = [x[0] - b * tau]

    loglik = 0.

    for i in range(1, num_datos):

        a_new = np.exp(-(t[i] - t[i - 1]) / tau)

        x_ast.append(x[i] - b * tau)

        x_hat.append(
            a_new * x_hat[i - 1] +
            (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
            (x_ast[i - 1] - x_hat[i - 1]))

        Omega.append(
            Omega[0] * (1 - (a_new ** 2)) + ((a_new ** 2)) * Omega[i - 1] *
            (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1]))))

        loglik_inter = np.log(
            ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
            (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
             (Omega[i] + error_vars[i]))) + EPSILON))

        loglik = loglik + loglik_inter

        if loglik <= CTE_NEG:
            warnings.warn(
                "CAR log-likelihood to inf", FeatureExtractionWarning)
            return -np.infty

    # the minus one is to perfor maximization using the minimize function
    return -loglik


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class CAR(Extractor):
    r"""In order to model the irregular sampled times series we use CAR
    (Brockwell and Davis, 2002), a continious time auto regressive model.

    CAR process has three parameters, it provides a natural and consistent way
    of estimating a characteristic time scale and variance of light-curves.
    CAR process is described by the following stochastic differential equation:

    .. math::

        dX(t) = - \frac{1}{\tau} X(t)dt +
            \sigma_C \sqrt{dt} \epsilon(t) + bdt, \\
        for \: \tau, \sigma_C, t \geq 0

    where the mean value of the lightcurve :math:`X(t)` is :math:`b\tau`
    and the variance is :math:`\frac{\tau\sigma_C^2}{2}`.
    :math:`\tau`  is the relaxation time of the process :math:`X(t)`, it can
    be interpreted as describing the variability amplitude of the time series.
    :math:`\sigma_C` can be interpreted as describing the variability of the
    time series on time scales shorter than :math:`\tau`.
    :math:`\epsilon(t)` is a white noise process with zero mean and variance
    equal to one.

    The likelihood function of a CAR model for a light-curve with observations
    :math:`x - \{x_1, \dots, x_n\}` observed at times
    :math:`\{t_1, \dots, t_n\}` with measurements error variances
    :math:`\{\delta_1^2, \dots, \delta_n^2\}` is:

    .. math::

        p (x|b,\sigma_C,\tau) = \prod_{i=1}^n \frac{1}{
            [2 \pi (\Omega_i + \delta_i^2 )]^{1/2}} exp \{-\frac{1}{2}
            \frac{(\hat{x}_i - x^*_i )^2}{\Omega_i + \delta^2_i}\} \\

        x_i^* = x_i - b\tau \\

        \hat{x}_0 = 0 \\

        \Omega_0 = \frac{\tau \sigma^2_C}{2} \\

        \hat{x}_i = a_i\hat{x}_{i-1} + \frac{a_i \Omega_{i-1}}{\Omega_{i-1} +
            \delta^2_{i-1}} (x^*_{i-1} + \hat{x}_{i-1}) \\

        \Omega_i = \Omega_0 (1- a_i^2 ) + a_i^2 \Omega_{i-1}
            (1 - \frac{\Omega_{i-1}}{\Omega_{i-1} + \delta^2_{i-1}} )

    To find the optimal parameters we maximize the likelihood with respect to
    :math:`\sigma_C` and :math:`\tau` and calculate :math:`b` as the mean
    magnitude of the light-curve divided by :math:`\tau`.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(
        ...     only=['CAR_sigma', 'CAR_tau','CAR_mean'])
        >>> features, values = fs.extract(**lc_periodic)
        >>> dict(zip(features, values))
        {'CAR_mean': -9.230698873903961,
         'CAR_sigma': -0.21928049298842511,
         'CAR_tau': 0.64112037377348619}

    References
    ----------

    .. [brockwell2002introduction] Brockwell, P. J., & Davis, R. A. (2002).
       Introduction toTime Seriesand Forecasting.

    .. [pichara2012improved] Pichara, K., Protopapas, P., Kim, D. W.,
       Marquette, J. B., & Tisserand, P. (2012). An improved quasar detection
       method in EROS-2 and MACHO LMC data sets. Monthly Notices of the Royal
       Astronomical Society, 427(2), 1284-1297.
       Doi:10.1111/j.1365-2966.2012.22061.x.

    """
    data = ['magnitude', 'time', 'error']
    features = ["CAR_sigma", "CAR_tau", "CAR_mean"]
    params = {"minimize_method": "nelder-mead"}

    def _calculate_CAR(self, time, magnitude, error, minimize_method):
        magnitude = magnitude.copy()
        time = time.copy()
        error = error.copy() ** 2

        x0 = [10, 0.5]
        bnds = ((0, 100), (0, 100))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            res = minimize(_car_like, x0,
                           args=(time, magnitude, error),
                           method=minimize_method, bounds=bnds)
        sigma, tau = res.x[0], res.x[1]
        return sigma, tau

    def fit(self, magnitude, time, error, minimize_method):
        sigma, tau = self._calculate_CAR(
            time, magnitude, error, minimize_method)
        mean = np.mean(magnitude) / tau

        return {"CAR_sigma": sigma, "CAR_tau": tau, "CAR_mean": mean}
