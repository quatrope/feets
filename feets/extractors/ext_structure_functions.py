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

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

from scipy.interpolate import interp1d

from .core import Extractor, FeatureExtractionWarning


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StructureFunctions(Extractor):
    r"""The structure function of rotation measures (RMs) contains information
    on electron density and magnetic field fluctuations.

    References
    ----------

    .. [simonetti1984small] Simonetti, J. H., Cordes, J. M., & Spangler, S. R.
       (1984). Small-scale variations in the galactic magnetic field-The
       rotation measure structure function and birefringence in interstellar
       scintillations. The Astrophysical Journal, 284, 126-134.

    """

    features = [
        "StructureFunction_index_21",
        "StructureFunction_index_31",
        "StructureFunction_index_32",
    ]

    def __init__(self):
        pass

    def extract(self, magnitude, time):
        Nsf, Np = 100, 100
        sf1, sf2, sf3 = np.zeros(Nsf), np.zeros(Nsf), np.zeros(Nsf)
        f = interp1d(time, magnitude)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
            sf1[tau - 1] = np.mean(
                np.power(np.abs(mag_int[0 : Np - tau] - mag_int[tau:Np]), 1.0)
            )
            sf2[tau - 1] = np.mean(
                np.abs(
                    np.power(
                        np.abs(mag_int[0 : Np - tau] - mag_int[tau:Np]), 2.0
                    )
                )
            )
            sf3[tau - 1] = np.mean(
                np.abs(
                    np.power(
                        np.abs(mag_int[0 : Np - tau] - mag_int[tau:Np]), 3.0
                    )
                )
            )
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        if len(sf1_log) and len(sf2_log):
            m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        else:
            warnings.warn(
                "Can't compute StructureFunction_index_21",
                FeatureExtractionWarning,
            )
            m_21 = np.nan

        if len(sf1_log) and len(sf3_log):
            m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        else:
            warnings.warn(
                "Can't compute StructureFunction_index_31",
                FeatureExtractionWarning,
            )
            m_31 = np.nan

        if len(sf2_log) and len(sf3_log):
            m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)
        else:
            warnings.warn(
                "Can't compute StructureFunction_index_32",
                FeatureExtractionWarning,
            )
            m_32 = np.nan

        return {
            "StructureFunction_index_21": m_21,
            "StructureFunction_index_31": m_31,
            "StructureFunction_index_32": m_32,
        }
