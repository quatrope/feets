#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
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
from .extractor import (
    DATAS,
    Extractor,
    ExtractorBadDefinedError,
    ExtractorValidationError,
    ExtractorWarning,
)

__all__ = [
    "DATAS",
    "ExtractorBadDefinedError",
    "ExtractorValidationError",
    "ExtractorWarning",
    "Extractor",
    "registry",
]


# =============================================================================
# REGISTERS
# =============================================================================

from .ext_astropy_lomb_scargle import AstropyLombScargle
from .ext_autocor_length import AutocorLength
from .ext_car import CAR
from .ext_color import Color
from .ext_con import Con
from .ext_dmdt import DeltamDeltat
from .ext_eta_color import EtaColor
from .ext_fourier_components import FourierComponents
from .ext_gskew import Gskew
from .ext_median_amplitude import MedianAmplitude
from .ext_pair_slope_trend import PairSlopeTrend
from .ext_q31 import Q31, Q31Color
from .ext_rcs import RCS
from .ext_signature import Signature
from .ext_slotted_a_length import SlottedALength
from .ext_stetson import StetsonJ, StetsonKAC, StetsonL
from .ext_structure_functions import StructureFunctions
from .ext_weighted_beyond_N_std import WeightedBeyondNStd

from .light_curve.ext_amplitude import Amplitude
from .light_curve.ext_anderson_darling import AndersonDarling
from .light_curve.ext_bazin_fit import BazinFit
from .light_curve.ext_beyond_n_std import BeyondNStd
from .light_curve.ext_cusum import Cusum
from .light_curve.ext_duration import Duration
from .light_curve.ext_eta import Eta
from .light_curve.ext_eta_e import EtaE
from .light_curve.ext_excess_variance import ExcessVariance
from .light_curve.ext_inter_percentile_range import InterPercentileRange
from .light_curve.ext_light_curve_lomb_scargle import LightCurveLombScargle
from .light_curve.ext_linear_fit import LinearFit
from .light_curve.ext_linear_trend import LinearTrend
from .light_curve.ext_linexp_fit import LinexpFit
from .light_curve.ext_max_slope import MaxSlope
from .light_curve.ext_max_time_interval import MaxTimeInterval
from .light_curve.ext_mean import Mean
from .light_curve.ext_mean_variance import MeanVariance
from .light_curve.ext_median_abs_dev import MedianAbsDev
from .light_curve.ext_median_brp import MedianBRP
from .light_curve.ext_min_time_interval import MinTimeInterval
from .light_curve.ext_otsu_split import OtsuSplit
from .light_curve.ext_percent_amplitude import PercentAmplitude
from .light_curve.ext_percent_diff_percentile import PercentDiffPercentile
from .light_curve.ext_percentage_ratio import PercentageRatio
from .light_curve.ext_reduced_chi2 import ReducedChi2
from .light_curve.ext_roms import Roms
from .light_curve.ext_skew import Skew
from .light_curve.ext_small_kurtosis import SmallKurtosis
from .light_curve.ext_std import Std
from .light_curve.ext_stetson_k import StetsonK
from .light_curve.ext_time_mean import TimeMean
from .light_curve.ext_time_std import TimeStd
from .light_curve.ext_villar_fit import VillarFit
from .light_curve.ext_weighted_mean import WeightedMean


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
