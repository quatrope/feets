#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

__all__ = ["remove_noise", "align"]


# =============================================================================
# FUNCTIONS
# =============================================================================


def remove_noise(time, magnitude, error, error_limit=3, std_limit=5):
    """Points within 'std_limit' standard deviations from the mean and with
    errors greater than 'error_limit' times the error mean are
    considered as noise and thus are eliminated.

    """
    data, mjd = magnitude, time

    data_len = len(mjd)
    error_mean = np.mean(error)
    error_tolerance = error_limit * (error_mean or 1)
    data_mean = np.mean(data)
    data_std = np.std(data)

    mjd_out, data_out, error_out = [], [], []
    for i in range(data_len):
        is_not_noise = (
            error[i] < error_tolerance
            and (np.absolute(data[i] - data_mean) / data_std) < std_limit
        )

        if is_not_noise:
            mjd_out.append(mjd[i])
            data_out.append(data[i])
            error_out.append(error[i])

    data_out = np.asarray(data_out)
    mjd_out = np.asarray(mjd_out)
    error_out = np.asarray(error_out)

    return mjd_out, data_out, error_out


def align(time, time2, magnitude, magnitude2, error, error2):
    """Synchronizes the light-curves in the two different bands.

    Returns
    -------

    aligned_time
    aligned_magnitude
    aligned_magnitude2
    aligned_error
    aligned_error2

    """

    error = np.zeros(time.shape) if error is None else error
    error2 = np.zeros(time2.shape) if error2 is None else error2

    # this asume that the first series is the short one
    sserie = pd.DataFrame({"mag": magnitude, "error": error}, index=time)
    lserie = pd.DataFrame({"mag": magnitude2, "error": error2}, index=time2)

    # if the second serie is logest then revert
    if len(time) > len(time2):
        sserie, lserie = lserie, sserie

    # make the merge
    merged = sserie.join(lserie, how="inner", rsuffix="2")

    # recreate columns
    new_time = merged.index.values
    new_mag, new_mag2 = merged.mag.values, merged.mag2.values
    new_error, new_error2 = merged.error.values, merged.error2.values

    if len(time) > len(time2):
        new_mag, new_mag2 = new_mag2, new_mag
        new_error, new_error2 = new_error2, new_error

    return new_time, new_mag, new_mag2, new_error, new_error2
