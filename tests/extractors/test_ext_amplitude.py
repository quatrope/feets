#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_amplitude

import numpy as np


def test_Amplitude_extract():
    extractor = ext_amplitude.Amplitude()
    magnitude = np.arange(0, 1000)
    value = extractor.extract(magnitude=magnitude)["Amplitude"]
    np.testing.assert_allclose(value, 475)
