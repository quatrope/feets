#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# This code was ripped of from scikit-criteria on 05-oct-2024.
# https://github.com/quatrope/scikit-criteria/blob/48ab420/tests/utils/test_doctools.py
# Until this point the copyright is
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.decorator

"""


# =============================================================================
# IMPORTS
# =============================================================================

import string
import warnings

import numpy as np

import pytest

from feets.libs import doctools

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_doc_inherit():
    chars = tuple(string.ascii_letters + string.digits)
    random = np.random.default_rng(seed=42)

    doc = "".join(random.choice(chars, 1000))

    def func_a():
        ...

    func_a.__doc__ = doc

    @doctools.doc_inherit(func_a)
    def func_b():
        ...

    @doctools.doc_inherit(doc)
    def func_c():
        ...

    assert doc == func_a.__doc__ == func_b.__doc__ == func_c.__doc__

    # test warnings
    with pytest.warns(UserWarning):

        @doctools.doc_inherit(doc, warn_class=True)
        class A:  # noqa
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @doctools.doc_inherit(doc, warn_class=False)
        class A:  # noqa
            pass