

import sys
import time as tmod
import warnings

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

import pandas as pd

warnings.simplefilter("ignore")

sys.path.insert(0, "../FATS/")
import FATS

iterations = 100000
lc_size = 1000


random = np.random.RandomState(42)

results = {
    "StetsonK": np.empty(iterations),
    "StetsonJ": np.empty(iterations),
    "AndersonDarling": np.empty(iterations)}

for it in range(iterations):
    fs = FATS.FeatureSpace(featureList=list(results.keys()))

    # a simple time array from 0 to 99 with steps of 0.01
    time = np.arange(0, 100, 100./lc_size).shape

    # create 1000 magnitudes with mu 0 and std 1
    mags = random.normal(size=lc_size)

    # create 1000 magnitudes with difference <= 0.1% than mags
    mags2 = mags * random.uniform(0, 0.01, mags.size)

    # create two errors for the magnitudes equivalent to the 0.001%
    # of the magnitudes
    errors = random.normal(scale=0.00001, size=lc_size)
    errors2 = random.normal(scale=0.00001, size=lc_size)

    lc = np.array([
            mags,  # magnitude
            time,  # time
            errors, # error
            mags,  # magnitude2
            mags,  # aligned_magnitude
            mags,  # aligned_magnitude2
            time,  # aligned_time
            errors, # aligned_error
            errors  # aligned_error2
    ])

    fs.calculateFeature(lc)
    for k, v in fs.result("dict").items():
        results[k][it] = v


df = pd.DataFrame(results).describe()
print df
df.to_latex("features_montecarlo.tex", float_format='%.4f')
