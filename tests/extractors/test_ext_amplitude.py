import numpy as np

from feets.extractors import ext_amplitude


def test_Amplitude_extract():
    extractor = ext_amplitude.Amplitude()
    mags = np.arange(0, 1001)
    value = extractor.extract(magnitude=mags)["Amplitude"]
    np.testing.assert_allclose(value, 475)
