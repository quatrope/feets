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
# DOCS
# =============================================================================

"""Base code for IO dataset retrieval


"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import shutil
from collections import Mapping

import numpy as np

import requests

import attr

from ..extractors.core import DATAS


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_data_home(data_home=None):
    """Return the path of the feets data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'feets_data' in the
    user home folder.

    Alternatively, it can be set by the 'feets_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to feets data dir.
    """
    if data_home is None:
        data_home = os.environ.get(
            'feets_DATA', os.path.join('~', 'feets_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    Parameters
    ----------
    data_home : str | None
        The path to feets data dir.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def fetch(url, dest, force=False):
    """Retrieve data from an url and store it into dest.

    Parameters
    ----------
    url: str
        Link to the remote data
    dest: str
        Path where the file must be stored
    force: bool (default=False)
        Overwrite if the file exists

    Returns
    -------
    cached: bool
        True if the file already exists
    dest: str
        The same string of the parameter


    """

    cached = True
    if force or not os.path.exists(dest):
        cached = False
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
    return cached, dest


# =============================================================================
# CLASSES
# =============================================================================

class Bunch(Mapping):  # THANKS SKLEARN
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes.

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, data=None, **kwargs):
        if data and kwargs:
            raise ValueError(
                "If 'data' is not none keywords aguments are not allowed")
        self._data = dict(data) if data else kwargs

    def __repr__(self):
        keys_str = ", ".join(self._data.keys())
        return "Bunch({})".format(keys_str)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __dir__(self):
        return self._data.keys()

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


# This ugly code creates a LightCurve object based on the extractor constants
# and ad som validations and a custom repr, as

LightCurveBase = attr.make_class(
    'LightCurveBase', {
        k: attr.ib(default=attr.NOTHING if k in DATAS[:2] else None,
                   converter=attr.converters.optional(np.asarray))
        for k in DATAS}, frozen=True)


class LightCurve(LightCurveBase, Mapping):

    def __repr__(self):
        fields = []
        for a in attr.fields(LightCurveBase):
            v = getattr(self, a.name)
            if v is not None:
                fields.append("{}[{}]".format(a.name, len(v)))
        fields_str = ", ".join(fields)
        return "LightCurve({})".format(fields_str)

    def __getitem__(self, k):
        try:
            return getattr(self, k)
        except AttributeError:
            raise KeyError(k)

    def __iter__(self):
        return iter(k for k, v in attr.asdict(self).items() if v is not None)

    def __len__(self):
        return len(attr.fields(LightCurveBase))


# The real dataset object

@attr.s(frozen=True)
class Dataset(Mapping):
    """This object encapsulates a full dataset with their metadata.

    Attributes
    ----------

    id : any object or None
        the id of the lightcurve or None
    ds_name : str
        The name of the dataset
    description : str
        description about the dataser
    bands : tuple
        the names of the attributes inside data
    metadata : dict-like
        arbitrary data.
    data : dict-like
        lightcurves collection in a dint-like object

    """
    id = attr.ib()
    ds_name = attr.ib(converter=str)
    description = attr.ib(converter=str, repr=False)
    bands = attr.ib(converter=tuple)
    metadata = attr.ib(
        repr=False, converter=attr.converters.optional(Bunch))
    data = attr.ib(
        repr=False, converter=lambda value: Bunch({
            k: LightCurve(**v) for k, v in value.items()}))

    def __getitem__(self, k):
        try:
            return getattr(self, k)
        except AttributeError:
            raise KeyError(k)

    def __iter__(self):
        return iter(k for k, v in attr.asdict(self).items() if v is not None)

    def __len__(self):
        return len(attr.fields(Dataset))
