import os
import pathlib
import pickle
import sys

import hashlib

import joblib

from timeit import default_timer as timer

import numpy as np

import feets


PATH = pathlib.Path(os.getcwd())
RESULTS_PATH = PATH / "cache"

LIGHT_CURVE_FEATURES = [
    "Amplitude",
    "BazinFit_Amplitude",
    "BazinFit_Baseline",
    "BazinFit_ReferenceTime",
    "BazinFit_RiseTime",
    "BazinFit_FallTime",
    "BazinFit_ReducedChi2",
    "BeyondNStd",
    "Cusum",
    "Duration",
    "Eta",
    "ExcessVariance",
    "LightCurve_PeriodLS",
    "Period_s_to_n",
    "LinearFit_Slope",
    "LinearFit_Sigma",
    "LinearFit_ReducedChi2",
    "LinexpFit_Amplitude",
    "LinexpFit_Baseline",
    "LinexpFit_ReferenceTime",
    "LinexpFit_FallTime",
    "LinexpFit_ReducedChi2",
    "MaxTimeInterval",
    "MinTimeInterval",
    "OtsuMeanDiff",
    "OtsuStdLower",
    "OtsuStdUpper",
    "OtsuLowerToAllRatio",
    "ReducedChi2",
    "Roms",
    "TimeMean",
    "TimeStd",
    "VillarFit_Amplitude",
    "VillarFit_Baseline",
    "VillarFit_ReferenceTime",
    "VillarFit_RiseTime",
    "VillarFit_FallTime",
    "VillarFit_PlateauRelAmplitude",
    "VillarFit_PlateauDuration",
    "VillarFit_ReducedChi2",
    "WeightedMean",
]

DASK_SCHEDULERS = {"synchronous", "processes", "threads"}

DASK_OPTIONS = {
    # "num_workers": 4,
}

LC_GROUP_SIZES = [1, 50, 100, 500, 1000]
LC_LENGTHS = [30, 100, 300, 1000]
MAX_ITERS = 10


def get_filepath(result, preffix="result"):
    bresult = pickle.dumps(result)
    hresult = hashlib.md5(bresult).hexdigest()
    fname = f"{preffix}_{hresult}.jpkl"
    return RESULTS_PATH / fname


def periodic_light_curve(*, length=100, random=None, period=10):
    random = np.random.default_rng(random)
    cov = np.exp(
        -np.sin(
            (np.pi / period)
            * np.subtract.outer(np.arange(length), np.arange(length))
        )
        ** 2
    )

    time = np.arange(length)
    error = random.uniform(0, 0.08, length)
    magnitude = random.multivariate_normal(mean=np.zeros(length), cov=cov)
    flux = np.exp(-magnitude)
    flux_error = random.uniform(0, 0.08, length)

    return {
        "time": time,
        "magnitude": magnitude,
        "error": error,
        "magnitude2": magnitude,
        "error2": error,
        "aligned_time": time,
        "aligned_magnitude": magnitude,
        "aligned_error": error,
        "aligned_magnitude2": magnitude,
        "aligned_error2": error,
        "flux": flux,
        "flux_error": flux_error,
    }


def periodic_light_curve_group(*, lc_group_size, lc_length):
    return [
        periodic_light_curve(length=lc_length) for _ in range(lc_group_size)
    ]


def run(*, lc_group, dask_options, exclude):
    fs = feets.FeatureSpace(dask_options=dask_options, exclude=exclude)

    start = timer()
    features = fs.extract(*lc_group)
    end = timer()

    time = end - start

    return time, features


def test_single(*, lc_group_size, lc_length, scheduler, it, exclude, lc_group):
    result = {
        "group_size": lc_group_size,
        "lc_length": lc_length,
        "scheduler": scheduler,
        "it": it,
        "exclude": exclude,
    }

    filepath = get_filepath(result, preffix=scheduler)
    if filepath.exists():
        return

    try:
        time, features = run(
            lc_group=lc_group,
            dask_options={"scheduler": scheduler, **DASK_OPTIONS},
            exclude=[] if exclude else LIGHT_CURVE_FEATURES,
        )
        result["lc_group"] = lc_group
        result["times"] = time
        result["features"] = features.as_frame()
        joblib.dump(result, filepath)
    except Exception as e:
        filepath = get_filepath(result, preffix=f"error_{scheduler}")
        joblib.dump(e, filepath)


def test(scheduler):
    for lc_group_size in LC_GROUP_SIZES:
        for lc_length in LC_LENGTHS:
            for it in range(MAX_ITERS):
                lc_group = periodic_light_curve_group(
                    lc_group_size=lc_group_size, lc_length=lc_length
                )

                test_single(
                    lc_group_size=lc_group_size,
                    lc_length=lc_length,
                    scheduler=scheduler,
                    it=it,
                    exclude=False,
                    lc_group=lc_group,
                )

                test_single(
                    lc_group_size=lc_group_size,
                    lc_length=lc_length,
                    scheduler=scheduler,
                    it=it,
                    exclude=True,
                    lc_group=lc_group,
                )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sequential.py <scheduler>")
    else:
        scheduler = sys.argv[1]
        if scheduler not in DASK_SCHEDULERS:
            print(f"Scheduler must be one of {DASK_SCHEDULERS}")
            sys.exit(1)

        test(scheduler)
