import numpy as np


def preprocess_data(
    *, time=None, magnitude=None, error=None, dtype=np.float64
):
    if time is None and magnitude is None:
        raise ValueError(
            "At least one of 'time' or 'magnitude' must be provided"
        )
    shape = len(time) if time is not None else len(magnitude)

    time = (
        np.arange(shape, dtype=dtype)
        if time is None
        else np.array(time, dtype)
    )
    magnitude = (
        np.zeros(shape, dtype)
        if magnitude is None
        else np.array(magnitude, dtype)
    )
    sigma = (
        np.ones(shape, dtype)
        if error is None
        else np.array(1 / error**2, dtype)
    )
    return time, magnitude, sigma


def mag_to_flux(magnitude, zero_point):
    return 10 ** (-0.4 * (magnitude - zero_point))
