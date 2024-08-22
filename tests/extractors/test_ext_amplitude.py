import numpy as np

from feets.extractors import ext_amplitude


def test_Amplitude_extract():
    extractor = ext_amplitude.Amplitude()
    value = extractor.extract(np.arange(0, 1001))["Amplitude"]
    np.testing.assert_allclose(value, 475)
