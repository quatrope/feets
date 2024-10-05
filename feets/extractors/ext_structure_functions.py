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

from scipy.interpolate import interp1d

from .extractor import Extractor


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
                np.power(np.abs(mag_int[: Np - tau] - mag_int[tau:Np]), 1.0)
            )
            sf2[tau - 1] = np.mean(
                np.abs(
                    np.power(
                        np.abs(mag_int[: Np - tau] - mag_int[tau:Np]), 2.0
                    )
                )
            )
            sf3[tau - 1] = np.mean(
                np.abs(
                    np.power(
                        np.abs(mag_int[: Np - tau] - mag_int[tau:Np]), 3.0
                    )
                )
            )
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        if len(sf1_log) and len(sf2_log):
            m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        else:
            self.feature_warning("Can't compute StructureFunction_index_21")
            m_21 = np.nan

        if len(sf1_log) and len(sf3_log):
            m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        else:
            self.feature_warning("Can't compute StructureFunction_index_31")
            m_31 = np.nan

        if len(sf2_log) and len(sf3_log):
            m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)
        else:
            self.feature_warning("Can't compute StructureFunction_index_32")
            m_32 = np.nan

        return {
            "StructureFunction_index_21": m_21,
            "StructureFunction_index_31": m_31,
            "StructureFunction_index_32": m_32,
        }
