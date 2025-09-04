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

"""Structure functions extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from scipy.interpolate import interp1d

from .extractor import Extractor, feature_warning
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StructureFunctions(Extractor):
    r"""TStructure functions extractor.

    The structure function of rotation measures (RMs) contains information
    on electron density and magnetic field fluctuations.

    References
    ----------
    .. [simonetti1984small] Simonetti, J. H., Cordes, J. M., & Spangler, S. R.
       (1984). Small-scale variations in the galactic magnetic field-The
       rotation measure structure function and birefringence in interstellar
       scintillations. The Astrophysical Journal, 284, 126-134.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=[
    ...     'StructureFunction_index_21',
    ...     'StructureFunction_index_31',
    ...     'StructureFunction_index_32',
    ... ])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'StructureFunction_index_21': np.float64(1.6029987396657115),
     'StructureFunction_index_31': np.float64(2.050072565193364),
     'StructureFunction_index_32': np.float64(1.4137753817054497)}
    """

    features = [
        "StructureFunction_index_21",
        "StructureFunction_index_31",
        "StructureFunction_index_32",
    ]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, time):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like
        """
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
            feature_warning("Can't compute StructureFunction_index_21")
            m_21 = np.nan

        if len(sf1_log) and len(sf3_log):
            m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        else:
            feature_warning("Can't compute StructureFunction_index_31")
            m_31 = np.nan

        if len(sf2_log) and len(sf3_log):
            m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)
        else:
            feature_warning("Can't compute StructureFunction_index_32")
            m_32 = np.nan

        return {
            "StructureFunction_index_21": m_21,
            "StructureFunction_index_31": m_31,
            "StructureFunction_index_32": m_32,
        }
