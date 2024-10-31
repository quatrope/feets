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

"""Standard deviation extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Std(Extractor):
    r"""Standard deviation extractor.

    **Std**

    The standard deviation of the magnitudes. The standard deviation
    :math:`\sigma` of the sample is defined as:

    .. math::

        \sigma=\frac{1}{N-1}\sum_{i} (y_{i}-\hat{y})^2

    For example, a white noise time serie should have :math:`\sigma=1`

    .. code-block:: pycon

    Examples
    --------
    Standard deviation of a white noise time series:

    >>> fs = feets.FeatureSpace(only=['Std'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Std': np.float64(1.0140342446921735)}

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["Std"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        std = np.std(magnitude)
        return {"Std": std}
