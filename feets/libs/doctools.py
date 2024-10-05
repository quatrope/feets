#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# This code was ripped of from scikit-criteria on 05-oct-2024.
# https://github.com/quatrope/scikit-criteria/blob/48ab420/skcriteria/utils/doctools.py
# Until this point the copyright is
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Multiple decorator to use inside scikit-criteria."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from inspect import isclass

from custom_inherit import doc_inherit as _doc_inherit

# =============================================================================
# DOC INHERITANCE
# =============================================================================


def doc_inherit(parent, warn_class=True):
    """Inherit the 'parent' docstring.

    Returns a function/method decorator that, given parent, updates
    the docstring of the decorated function/method based on the `numpy`
    style and the corresponding attribute of parent.

    Parameters
    ----------
    parent : Union[str, Any]
        The docstring, or object of which the docstring is utilized as the
        parent docstring during the docstring merge.
    warn_class : bool
        If it is true, and the decorated is a class, it throws a warning
        since there are some issues with inheritance of documentation in
        classes.

    Notes
    -----
    This decorator is a thin layer over
    :py:func:`custom_inherit.doc_inherit decorator`.

    Check: <github `https://github.com/rsokl/custom_inherit`>__


    """

    def _wrapper(obj):
        if isclass(obj) and warn_class:
            warnings.warn(
                f"{obj} is a class, check if the "
                "documentation was inherited properly "
            )
        dec = _doc_inherit(parent, style="numpy")
        return dec(obj)

    return _wrapper