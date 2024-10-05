#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# This code was ripped of from scikit-criteria on 05-oct-2024.
# https://github.com/quatrope/scikit-criteria/blob/5f829f7/tests/utils/test_bunch.py
# Until this point the copyright is
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.bunch

"""


# =============================================================================
# IMPORTS
# =============================================================================

import copy

import pytest

from feets.libs import bunch


# =============================================================================
# TEST Bunch
# =============================================================================


def test_Bunch_creation():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    assert len(md) == 1


def test_Bunch_creation_empty():
    md = bunch.Bunch("foo", {})
    assert len(md) == 0


def test_Bunch_key_notfound():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(KeyError):
        md["bravo"]


def test_Bunch_attribute_notfound():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(AttributeError):
        md.bravo


def test_Bunch_iter():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert list(iter(md)) == ["alfa"]


def test_Bunch_repr():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert repr(md) == "<foo {'alfa'}>"


def test_Bunch_dir():
    md = bunch.Bunch("foo", {"alfa": 1})
    assert "alfa" in dir(md)


def test_Bunch_deepcopy():
    md = bunch.Bunch("foo", {"alfa": 1})
    md_c = copy.deepcopy(md)

    assert md is not md_c
    assert md._name == md_c._name  # string are inmutable never deep copy
    assert md._data == md_c._data and md._data is not md_c._data


def test_Bunch_copy():
    md = bunch.Bunch("foo", {"alfa": 1})
    md_c = copy.copy(md)

    assert md is not md_c
    assert md._name == md_c._name
    assert md._data == md_c._data and md._data is md_c._data


def test_Bunch_data_is_not_a_mapping():
    with pytest.raises(TypeError, match="Data must be some kind of mapping"):
        bunch.Bunch("foo", None)