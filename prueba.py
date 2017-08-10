
import numpy as np
import FATS
import feets


with np.load("lc_1.3444.614.B_R.npz") as npz:
    lc = (
        npz['mag'],
        npz['time'],
        npz['error'],
        npz['mag2'],
        npz['aligned_mag'],
        npz['aligned_mag2'],
        npz['aligned_time'],
        npz['aligned_error'],
        npz['aligned_error2'])

fats = FATS.FeatureSpace(
        Data="all")
values = fats.calculateFeature(lc)
import ipdb; ipdb.set_trace()
