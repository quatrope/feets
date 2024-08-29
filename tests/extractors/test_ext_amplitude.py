import numpy as np

from feets.extractors import ext_amplitude


def test_Amplitude_extract():
    extractor = ext_amplitude.Amplitude()
    magnitude = np.arange(0, 1000)
    value = extractor.extract(magnitude=magnitude)["Amplitude"]
    np.testing.assert_allclose(value, 475)
