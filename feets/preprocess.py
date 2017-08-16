#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from six.moves import range

__all__ = [
    "remove_noise",
    "align"]


# =============================================================================
# FUNCTIONS
# =============================================================================

def remove_noise(mag, time, error, error_limit=3, std_limit=5):
    """Points within 'std_limit' standard deviations from the mean and with
    errors greater than 'error_limit' times the error mean are
    considered as noise and thus are eliminated.

    """
    data, mjd = mag, time

    data_len = len(mjd)
    error_mean = np.mean(error)
    error_tolerance = error_limit * (error_mean or 1)
    data_mean = np.mean(data)
    data_std = np.std(data)

    mjd_out, data_out, error_out = [], [], []
    for i in range(data_len):
        is_not_noise = (
            error[i] < error_tolerance and
            (np.absolute(data[i] - data_mean) / data_std) < std_limit)

        if is_not_noise:
            mjd_out.append(mjd[i])
            data_out.append(data[i])
            error_out.append(error[i])

    data_out = np.asarray(data_out)
    mjd_out = np.asarray(mjd_out)
    error_out = np.asarray(error_out)

    return data_out, mjd_out, error_out


def align(mag, mag2, time, time2, error, error2):
    """Synchronizes the light-curves in the two different bands.

    Returns
    -------

    aligned_time
    aligned_mag
    aligned_mag2
    aligned_error
    aligned_error2

    """

    mjd, mjd2 = time, time2  # TODO: use time instead of mjd
    data, data2 = mag, mag2  # TODO: use mag instead of data

    if len(data2) > len(data):

        new_data2 = []
        new_error2 = []
        new_mjd2 = []
        new_mjd = np.copy(mjd)
        new_error = np.copy(error)
        new_data = np.copy(data)
        count = 0

        for index in range(len(data)):

            where = np.where(mjd2 == mjd[index])

            if np.array_equal(where[0], []) is False:

                new_data2.append(data2[where])
                new_error2.append(error2[where])
                new_mjd2.append(mjd2[where])
            else:
                new_mjd = np.delete(new_mjd, index - count)
                new_error = np.delete(new_error, index - count)
                new_data = np.delete(new_data, index - count)
                count = count + 1

        new_data2 = np.asarray(new_data2).flatten()
        new_error2 = np.asarray(new_error2).flatten()

    else:

        new_data = []
        new_error = []
        new_mjd = []
        new_mjd2 = np.copy(mjd2)
        new_error2 = np.copy(error2)
        new_data2 = np.copy(data2)
        count = 0
        for index in range(len(data2)):
            where = np.where(mjd == mjd2[index])

            if np.array_equal(where[0], []) is False:
                new_data.append(data[where])
                new_error.append(error[where])
                new_mjd.append(mjd[where])
            else:
                new_mjd2 = np.delete(new_mjd2, (index - count))
                new_error2 = np.delete(new_error2, (index - count))
                new_data2 = np.delete(new_data2, (index - count))
                count = count + 1

        new_data = np.asarray(new_data).flatten()
        new_mjd = np.asarray(new_mjd).flatten()
        new_error = np.asarray(new_error).flatten()

    return new_mjd, new_data, new_data2, new_error, new_error2
