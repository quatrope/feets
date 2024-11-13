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

"""Flux percentile ratio extractors."""


# =============================================================================
# IMPORTS
# =============================================================================

import math

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class FluxPercentileRatioMid20(Extractor):
    r"""Flux percentile ratio mid 20 extractor.

    Notes
    -----
    In order to caracterize the sorted magnitudes distribution we use
    percentiles. If :math:`F_{5, 95}` is the difference between :math:`95%%`
    and :math:`5%%` magnitude values, we calculate the following:

    - FluxPercentileRatioMid20: ratio :math:`F_{40, 60}/F_{5, 95}`
    - FluxPercentileRatioMid35: ratio :math:`F_{32.5, 67.5}/F_{5, 95}`
    - FluxPercentileRatioMid50: ratio :math:`F_{25, 75}/F_{5, 95}`
    - FluxPercentileRatioMid65: ratio :math:`F_{17.5, 82.5}/F_{5, 95}`
    - FluxPercentileRatioMid80: ratio :math:`F_{10, 90}/F_{5, 95}`

    For the first feature for example, in the case of a normal distribution,
    this is equivalente to calculate:

    .. math::

        \frac{erf^{-1}(2 \cdot 0.6-1)-erf^{-1}(2 \cdot 0.4-1)}
                {erf^{-1}(2 \cdot 0.95-1)-erf^{-1}(2 \cdot 0.05-1)}


    So, the expected values for each of the flux percentile features are:

    - FluxPercentileRatioMid20 = :math:`0.154`
    - FluxPercentileRatioMid35 = :math:`0.275`
    - FluxPercentileRatioMid50 = :math:`0.410`
    - FluxPercentileRatioMid65 = :math:`0.568`
    - FluxPercentileRatioMid80 = :math:`0.779`

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["FluxPercentileRatioMid20"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'FluxPercentileRatioMid20': np.float64(0.14882100252933414)}

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
        Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
        Rischard, M. (2011). On machine-learned classification of variable stars
        with sparse and noisy time-series data.
        The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["FluxPercentileRatioMid20"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_60_index = int(math.ceil(0.60 * lc_length))
        F_40_index = int(math.ceil(0.40 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        return {"FluxPercentileRatioMid20": F_mid20}


@doctools.doc_inherit(FluxPercentileRatioMid20, warn_class=False)
class FluxPercentileRatioMid35(Extractor,):
    """Flux percentile ratio mid 35 extractor.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["FluxPercentileRatioMid35"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'FluxPercentileRatioMid35': np.float64(0.27423232011430465)}
    """

    features = ["FluxPercentileRatioMid35"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_325_index = int(math.ceil(0.325 * lc_length))
        F_675_index = int(math.ceil(0.675 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        return {"FluxPercentileRatioMid35": F_mid35}


@doctools.doc_inherit(FluxPercentileRatioMid20, warn_class=False)
class FluxPercentileRatioMid50(Extractor):
    """Flux percentile ratio mid 50 extractor.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["FluxPercentileRatioMid50"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'FluxPercentileRatioMid50': np.float64(0.4020921304774109)}
    """

    features = ["FluxPercentileRatioMid50"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_25_index = int(math.ceil(0.25 * lc_length))
        F_75_index = int(math.ceil(0.75 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        return {"FluxPercentileRatioMid50": F_mid50}


@doctools.doc_inherit(FluxPercentileRatioMid20, warn_class=False)
class FluxPercentileRatioMid65(Extractor):
    """Flux percentile ratio mid 65 extractor.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["FluxPercentileRatioMid65"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'FluxPercentileRatioMid65': np.float64(0.5808781429992802)}
    """

    features = ["FluxPercentileRatioMid65"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_175_index = int(math.ceil(0.175 * lc_length))
        F_825_index = int(math.ceil(0.825 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        return {"FluxPercentileRatioMid65": F_mid65}


@doctools.doc_inherit(FluxPercentileRatioMid20, warn_class=False)
class FluxPercentileRatioMid80(Extractor):
    """Flux percentile ratio mid 80 extractor.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["FluxPercentileRatioMid80"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'FluxPercentileRatioMid80': np.float64(0.7878789796839074)}
    """

    features = ["FluxPercentileRatioMid80"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_10_index = int(math.ceil(0.10 * lc_length))
        F_90_index = int(math.ceil(0.90 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        return {"FluxPercentileRatioMid80": F_mid80}
