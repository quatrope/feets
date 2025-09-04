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

"""Base code for IO dataset retrieval."""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib
import shutil
from collections.abc import Mapping

import attr

import numpy as np

import requests

from ..extractors.extractor import DATAS, DATA_TIME
from ..libs.bunch import Bunch


# =============================================================================
# CONSTANTS
# =============================================================================

HOME_PATH = pathlib.Path.home()

FEETS_DATA_DIR = "feets_data"

FEETS_DATA_DIR_ENV_VAR = "feets_DATA"

# =============================================================================
# FUNCTIONS
# =============================================================================


def get_data_home(data_home=None):
    """Return the path of the feets data directory.

    This directory is used by some large dataset loaders to avoid downloading
    the same data several times.

    By default, this is a directory named 'feets_data' in the user's home
    directory.

    Alternatively, it can be programatically set with the `data_home` variable,
    or by the 'feets_DATA' environment variable. The '~' prefix will be
    expanded to the user's home directory.

    If the directory does not already exist, it will be automatically created.

    Parameters
    ----------
    data_home : str, pathlib.Path or None, optional
        The path to the feets data directory.

    Returns
    -------
    pathlib.Path
        The path to the feets data directory.
    """
    if data_home is None:
        data_home = HOME_PATH / os.environ.get(
            FEETS_DATA_DIR_ENV_VAR, FEETS_DATA_DIR
        )
    data_home = pathlib.Path(data_home).expanduser()

    data_home.mkdir(parents=True, exist_ok=True)

    return data_home


def clear_data_home(data_home=None):
    """Delete all cached files from the feets data directory.

    Parameters
    ----------
    data_home : str, pathlib.Path or None, optional
        The path to the feets data directory.

    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def fetch(url, dest, force=False):
    """Retrieve data from `url` and store it into `dest`.

    Parameters
    ----------
    url: str
        Link to the remote data
    dest: str or pathlib.Path
        Path where the file must be stored
    force: bool, default=False
        Overwrite if the file already exists

    Returns
    -------
    cached: bool
        True if the file already exists
    dest: pathlib.Path
        The path to the downloaded file
    """
    cached = True
    dest = pathlib.Path(dest)
    if force or not dest.exists():
        cached = False
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest, "wb") as fp:
                for chunk in response.iter_content(1024):
                    fp.write(chunk)
    return cached, dest


# =============================================================================
# CLASSES
# =============================================================================

# This ugly code creates a LightCurve object based on the extractor constants
# and ad som validations and a custom repr, as
_LightCurveBase = attr.make_class(
    "LightCurveBase",
    {
        data: attr.ib(
            default=attr.NOTHING if data == DATA_TIME else None,
            converter=attr.converters.optional(np.asarray),
        )
        for data in DATAS
    },
    frozen=True,
)


class LightCurve(_LightCurveBase, Mapping):
    """Time series data representation.

    This class holds the time series data for a single photometric band.
    This may include the time, magnitude, and associated errors.

    Each of the available data vectors (time, magnitude, error) can be accessed
    as attributes of the class.
    """

    def __getitem__(self, key):
        """Get a data vector by name."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __len__(self):
        """Get the number of available data vectors."""
        return len(
            [
                value
                for value in attr.asdict(self).values()
                if value is not None
            ]
        )

    def __iter__(self):
        """Iterate over the names of the available data vectors."""
        return iter(
            key
            for key, value in attr.asdict(self).items()
            if value is not None
        )

    def __repr__(self):
        """String representation of the `LightCurve` object."""
        fields = [
            f"{key}[{len(value)}]"
            for key, value in attr.asdict(self).items()
            if value is not None
        ]
        fields_str = ", ".join(fields)
        return f"<LightCurve {fields_str}>"


@attr.s(frozen=True)
class LightCurveDataset(Mapping):
    """An immutable container for a single light curve dataset.

    This object encapsulates the time series data for one astronomical object,
    potentially across multiple photometric bands, along with its corresponding
    metadata. It behaves like a dictionary, allowing access to its main
    attributes (`_id`, `name`, `description`, etc.) via key-based lookup.

    The actual time series data is stored in the `data` attribute as a
    `feets.libs.bunch.Bunch` of `LightCurve` objects, one for each band.

    Attributes
    ----------
    _id : str or None
        A unique identifier for the light curve dataset object.
    name : str
        The human-readable name of the dataset from which this object
        originates.
    description : str
        A detailed description of the dataset's origin, content, or purpose.
    bands : tuple of str
        An ordered collection of the band names for which time series data is
        available. These names correspond to the keys in the `data` attribute.
    data : dict-like
        A dictionary-like object mapping band names (from `bands`) to their
        corresponding time series data. This is internally converted to a
        `feets.libs.bunch.Bunch` of `LightCurve` objects for convenient
        attribute-style access.
    metadata : dict-like or None
        A dictionary-like object containing arbitrary metadata about the light
        curve object (e.g., celestial coordinates, redshift). It is internally
        converted to a `feets.libs.bunch.Bunch` for convenient attribute-style
        access.

    See Also
    --------
    feets.libs.bunch.Bunch : Container object exposing keys as attributes.
    LightCurve
    """

    _id: str | None = attr.ib(converter=str)
    name: str = attr.ib(converter=str)
    description: str = attr.ib(converter=str, repr=False)
    bands: tuple = attr.ib(converter=tuple)
    data: Bunch = attr.ib(
        converter=lambda data: Bunch(
            "LightCurve", {band: LightCurve(**ts) for band, ts in data.items()}
        ),
        repr=False,
    )
    metadata: Bunch | None = attr.ib(
        converter=lambda metadata: (
            Bunch("Metadata", metadata) if metadata else None
        ),
        repr=False,
        default=None,
    )

    def __getitem__(self, key):
        """Get an attribute by name."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __len__(self):
        """Get the number of available attributes."""
        return len(
            [
                value
                for value in attr.asdict(self).values()
                if value is not None
            ]
        )

    def __iter__(self):
        """Iterate over the available attributes."""
        return iter(
            key
            for key, value in attr.asdict(self).items()
            if value is not None
        )
