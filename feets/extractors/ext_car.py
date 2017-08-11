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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from scipy.optimize import minimize

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class CAR(Extractor):

    data = ['magnitude', 'time', 'error']
    features = ["CAR_sigma", "CAR_tau", "CAR_mean"]

    def _CAR_Like(self, parameters, t, x, error_vars):

        sigma = parameters[0]
        tau = parameters[1]

        b = np.mean(x) / tau
        epsilon = 1e-300
        cte_neg = -np.infty
        num_datos = np.size(x)

        Omega, x_hat, a, x_ast = [], [], [], []

        Omega.append((tau * (sigma ** 2)) / 2.)
        x_hat.append(0.)
        a.append(0.)
        x_ast.append(x[0] - b * tau)

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
                 (Omega[i] + error_vars[i]))) + epsilon))

            loglik = loglik + loglik_inter

            if(loglik <= cte_neg):
                print('CAR loglikelihood to inf')
                return None

        # the minus one is to perfor maximization using the minimize function
        return -loglik

    def _calculate_CAR(self, time, magnitude, error):
        N = len(magnitude)
        magnitude = magnitude.reshape((N, 1))
        time = time.reshape((N, 1))
        error = error.reshape((N, 1)) ** 2

        x0 = [10, 0.5]
        bnds = ((0, 100), (0, 100))
        res = minimize(self._CAR_Like, x0, args=(time, magnitude, error),
                       method='nelder-mead', bounds=bnds)
        sigma, tau = res.x[0], res.x[1]
        return sigma, tau

    def fit(self, magnitude, time, error):
        sigma, tau = self._calculate_CAR(time, magnitude, error)
        mean = np.mean(magnitude) / tau

        return {"CAR_sigma": sigma, "CAR_tau": tau, "CAR_mean": mean}
