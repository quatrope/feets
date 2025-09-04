#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Utilities for accessing MACHO light curves.

The files are gathered from the original FATS project tutorial:
https://github.com/isadoranun/tsfeat
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib
import tarfile

import numpy as np

from .base import LightCurveDataset
from ..extractors.extractor import DATA_ERROR, DATA_MAGNITUDE, DATA_TIME
from ..libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

DATA_PATH = PATH / "data" / "macho"

DATASET_NAME = "MACHO"

DATASET_DESCRIPTION = (
    "The files are gathered from the original FATS project "
    "tutorial: https://github.com/isadoranun/tsfeat"
)

MACHO_EXAMPLE_ID = "lc_1.3444.614"

# =============================================================================
# FUNCTIONS
# =============================================================================


def available_MACHO_lc():
    """List the available MACHO light curves.

    Returns
    -------
    list
        The list of available MACHO light curves.
    """
    return [fp.rsplit(".", 2)[0] for fp in os.listdir(DATA_PATH)]


def load_MACHO_example():
    """Retrieve a light curve from the MACHO survey.

    The returned light curve contains data from 2 bands: R, B.

    Returns
    -------
    LightCurveDataset
        Dataset with the retrieved light curve data vectors.

    Notes
    -----
    The files are gathered from the original FATS project tutorial:
    https://github.com/isadoranun/tsfeat

    See Also
    --------
    available_MACHO_lc

    Examples
    --------
    >>> ds = load_MACHO_example()
    >>> ds
    LightCurveDataset(
        _id='lc_1.3444.614', name='MACHO', bands=('R', 'B')
    )
    >>> ds.bands
    ('R', 'B')
    >>> ds.data.B
    <LightCurve time[1235], magnitude[1235], error[1235]>
    >>> ds.data.B.magnitude
    array([-6.081, -6.041, -6.046, ..., -6.009, -5.985, -5.997], shape=(1235,))
    """
    return load_MACHO(MACHO_EXAMPLE_ID)


@doctools.doc_inherit(load_MACHO_example)
def load_MACHO(macho_id):
    """
    Parameters
    ----------
    macho_id : str
        The ID of the MACHO light curve to retrieve.

    Examples
    --------
    >>> ds = load_MACHO('lc_1.3444.614')
    >>> ds
    LightCurveDataset(
        _id='lc_1.3444.614', name='MACHO', bands=('R', 'B')
    )
    >>> ds.bands
    ('R', 'B')
    >>> ds.data.B
    <LightCurve time[1235], magnitude[1235], error[1235]>
    >>> ds.data.B.magnitude
    array([-6.081, -6.041, -6.046, ..., -6.009, -5.985, -5.997], shape=(1235,))
    """
    # Read the data
    tarpath = DATA_PATH / f"{macho_id}.tar.bz2"

    members = {
        "R": {"path": f"{macho_id}.R.mjd"},
        "B": {"path": f"{macho_id}.B.mjd"},
    }

    with tarfile.open(tarpath, mode="r:bz2") as tf:
        for band, member in members.items():
            members[band]["lc"] = np.loadtxt(tf.extractfile(member["path"]))

    bands = []
    data = {}
    for band, member in members.items():
        lc = member["lc"]
        data[band] = {
            DATA_TIME: lc[:, 0],
            DATA_MAGNITUDE: lc[:, 1],
            DATA_ERROR: lc[:, 2],
        }
        bands.append(band)

    return LightCurveDataset(
        id=macho_id,
        name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
        bands=bands,
        data=data,
    )
