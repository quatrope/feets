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

from scipy.optimize import curve_fit

from ..libs import lomb

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class FourierComponents(Extractor):

    data = ['magnitude', 'time']
    features = ['Freq1_harmonics_amplitude_0',
                'Freq1_harmonics_amplitude_1',
                'Freq1_harmonics_amplitude_2',
                'Freq1_harmonics_amplitude_3',
                'Freq2_harmonics_amplitude_0',
                'Freq2_harmonics_amplitude_1',
                'Freq2_harmonics_amplitude_2',
                'Freq2_harmonics_amplitude_3',
                'Freq3_harmonics_amplitude_0',
                'Freq3_harmonics_amplitude_1',
                'Freq3_harmonics_amplitude_2',
                'Freq3_harmonics_amplitude_3',
                'Freq1_harmonics_rel_phase_0',
                'Freq1_harmonics_rel_phase_1',
                'Freq1_harmonics_rel_phase_2',
                'Freq1_harmonics_rel_phase_3',
                'Freq2_harmonics_rel_phase_0',
                'Freq2_harmonics_rel_phase_1',
                'Freq2_harmonics_rel_phase_2',
                'Freq2_harmonics_rel_phase_3',
                'Freq3_harmonics_rel_phase_0',
                'Freq3_harmonics_rel_phase_1',
                'Freq3_harmonics_rel_phase_2',
                'Freq3_harmonics_rel_phase_3']

    def _model(self, x, a, b, c, Freq):
        return (a * np.sin(2 * np.pi * Freq * x) +
                b * np.cos(2 * np.pi * Freq * x) + c)

    def _yfunc_maker(self, Freq):
        def func(x, a, b, c):
            return (a * np.sin(2 * np.pi * Freq * x) +
                    b * np.cos(2 * np.pi * Freq * x) + c)
        return func

    def _compoenents(self, magnitude, time):
        time = time - np.min(time)
        A, PH, scaledPH = [], [], []
        for i in range(3):

            wk1, wk2, nout, jmax, prob = lomb.fasper(time, magnitude, 6., 100.)

            fundamental_Freq = wk1[jmax]
            Atemp, PHtemp, popts = [], [], []

            for j in range(4):
                function_to_fit = self._yfunc_maker((j + 1) * fundamental_Freq)
                popt, pcov = curve_fit(function_to_fit, time, magnitude)
                Atemp.append(np.sqrt(popt[0] ** 2 + popt[1] ** 2))
                PHtemp.append(np.arctan(popt[1] / popt[0]))
                popts.append(popt)

            A.append(Atemp)
            PH.append(PHtemp)

            for j in range(4):
                model = self._model(
                    time, popts[j][0], popts[j][1],
                    popts[j][2], (j+1) * fundamental_Freq)
                magnitude = np.array(magnitude) - model

        for ph in PH:
            scaledPH.append(np.array(ph) - ph[0])

    def fit(self, magnitude, time):
        A, PH, sPH = self._components(magnitude, time)
        freq1_aplitudes = [A[0][0], A[0][1], A[0][2], A[0][3]]
        freq2_aplitudes = [A[1][0], A[1][1], A[1][2], A[1][3]]
        freq3_aplitudes = [A[2][0], A[2][1], A[2][2], A[2][3]]

        freq1_phases = [sPH[0][0], sPH[0][1], sPH[0][2], sPH[0][3]]
        freq2_phases = [sPH[1][0], sPH[1][1], sPH[1][2], sPH[1][3]]
        freq3_phases = [sPH[2][0], sPH[2][1], sPH[2][2], sPH[2][3]]

        return (freq1_aplitudes + freq2_aplitudes + freq3_aplitudes +
                freq1_phases + freq2_phases + freq3_phases)
