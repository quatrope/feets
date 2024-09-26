#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """feets utilities"""


# =============================================================================
# FUNCTIONS
# =============================================================================


def indent(s, c=" ", n=4):
    """Indent the string 's' with the character 'c', 'n' times.

    Parameters
    ----------

    s : str
        String to indent
    c : str, default space
        String to use as indentation
    n : int, default 4
        Number of chars to indent

    """
    indentation = c * n
    return "\n".join([indentation + line for line in s.splitlines()])
