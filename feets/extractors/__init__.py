#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Features extractors classes and register utilities."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import registry
from .ext_amplitude import Amplitude
from .ext_anderson_darling import AndersonDarling
from .ext_astropy_lomb_scargle import AstropyLombScargle
from .ext_autocor_length import AutocorLength
from .ext_bazin_fit import BazinFit
from .ext_beyond_n_std import BeyondNStd
from .ext_car import CAR
from .ext_color import Color
from .ext_con import Con
from .ext_cusum import Cusum
from .ext_dmdt import DeltamDeltat
from .ext_duration import Duration
from .ext_eta import Eta
from .ext_eta_color import EtaColor
from .ext_eta_e import EtaE
from .ext_excess_variance import ExcessVariance
from .ext_fourier_components import FourierComponents
from .ext_gskew import Gskew
from .ext_inter_percentile_range import InterPercentileRange
from .ext_light_curve_lomb_scargle import LightCurveLombScargle
from .ext_linear_fit import LinearFit
from .ext_linear_trend import LinearTrend
from .ext_linexp_fit import LinexpFit
from .ext_max_slope import MaxSlope
from .ext_max_time_interval import MaxTimeInterval
from .ext_mean import Mean
from .ext_mean_variance import MeanVariance
from .ext_median_abs_dev import MedianAbsDev
from .ext_median_amplitude import MedianAmplitude
from .ext_median_brp import MedianBRP
from .ext_min_time_interval import MinTimeInterval
from .ext_otsu_split import OtsuSplit
from .ext_pair_slope_trend import PairSlopeTrend
from .ext_percent_amplitude import PercentAmplitude
from .ext_percent_diff_percentile import PercentDiffPercentile
from .ext_percentage_ratio import PercentageRatio
from .ext_q31 import Q31, Q31Color
from .ext_rcs import RCS
from .ext_reduced_chi2 import ReducedChi2
from .ext_roms import Roms
from .ext_signature import Signature
from .ext_skew import Skew
from .ext_slotted_a_length import SlottedALength
from .ext_small_kurtosis import SmallKurtosis
from .ext_std import Std
from .ext_stetson import StetsonJ, StetsonKAC, StetsonL
from .ext_stetson_k import StetsonK
from .ext_structure_functions import StructureFunctions
from .ext_time_mean import TimeMean
from .ext_time_std import TimeStd
from .ext_villar_fit import VillarFit
from .ext_weighted_beyond_n_std import WeightedBeyondNStd
from .ext_weighted_mean import WeightedMean
from .extractor import (
    DATAS,
    Extractor,
    ExtractorBadDefinedError,
    ExtractorValidationError,
    ExtractorWarning,
)


__all__ = [
    "DATAS",
    "Extractor",
    "ExtractorBadDefinedError",
    "ExtractorValidationError",
    "ExtractorWarning",
    "registry",
    "extractor_registry",
]


# =============================================================================
# REGISTERS
# =============================================================================

extractors = [
    Amplitude,
    AndersonDarling,
    AstropyLombScargle,
    AutocorLength,
    BazinFit,
    BeyondNStd,
    CAR,
    Color,
    Con,
    Cusum,
    DeltamDeltat,
    Duration,
    Eta,
    EtaColor,
    EtaE,
    ExcessVariance,
    FourierComponents,
    Gskew,
    InterPercentileRange,
    LightCurveLombScargle,
    LinearFit,
    LinearTrend,
    LinexpFit,
    MaxSlope,
    MaxTimeInterval,
    Mean,
    MeanVariance,
    MedianAbsDev,
    MedianAmplitude,
    MedianBRP,
    MinTimeInterval,
    OtsuSplit,
    PairSlopeTrend,
    PercentageRatio,
    PercentAmplitude,
    PercentDiffPercentile,
    Q31,
    Q31Color,
    RCS,
    ReducedChi2,
    Roms,
    Signature,
    Skew,
    SlottedALength,
    SmallKurtosis,
    Std,
    StetsonJ,
    StetsonK,
    StetsonKAC,
    StetsonL,
    StructureFunctions,
    TimeMean,
    TimeStd,
    VillarFit,
    WeightedBeyondNStd,
    WeightedMean,
]


extractor_registry = registry.ExtractorRegistry()

for cls in extractors:
    extractor_registry.register_extractor(cls)
del cls
