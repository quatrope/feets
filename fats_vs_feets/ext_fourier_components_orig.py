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

from scipy.optimize import curve_fit

import lomb

from feets import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class FourierComponents(Extractor):

    data = ["magnitude", "time"]
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
    params = {"ofac": 6.0}

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

    def _components(self, magnitude, time, ofac):
        time = time - np.min(time)
        A, PH = [], []
        for i in range(3):

            wk1, wk2, nout, jmax, prob = lomb.fasper(
                time, magnitude, ofac, 100.0
            )

            fundamental_Freq = wk1[jmax]
            Atemp, PHtemp = [], []
            omagnitude = magnitude

            for j in range(4):
                function_to_fit = self._yfunc_maker((j + 1) * fundamental_Freq)
                popt0, popt1, popt2 = curve_fit(
                    function_to_fit, time, omagnitude
                )[0][:3]

                Atemp.append(np.sqrt(popt0**2 + popt1**2))
                PHtemp.append(np.arctan(popt1 / popt0))

                model = self._model(
                    time, popt0, popt1, popt2, (j + 1) * fundamental_Freq
                )
                magnitude = np.array(magnitude) - model

            A.append(Atemp)
            PH.append(PHtemp)

        PH = np.asarray(PH)
        scaledPH = PH - PH[:, 0].reshape((len(PH), 1))

        return A, scaledPH

    def fit(self, magnitude, time, ofac):
        A, sPH = self._components(magnitude, time, ofac)
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
