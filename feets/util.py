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
# FUTURE
# =============================================================================

from __future__ import unicode_literals, print_function


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """Utilities"""


# =============================================================================
# IMPORTS
# =============================================================================

from collections import namedtuple


# =============================================================================
# FUNCTIONS
# =============================================================================

def dict2nt(name, d):
    "Convet a dict to a named tuple"
    nt = namedtuple(name, list(d.keys()))
    return nt(**d)


def fero(extractors):
    """Calculate the Feature Extractor Resolution Order of bases using the
    C3 algorithm.

    Based on:
        http://code.activestate.com/recipes/577748-calculate-the-mro-of-a-class

    """
    # map every feature to their extractor
    f2e = {}
    for e in extractors:
        f2e.update([(f, e) for f in e._conf.features])

    # map all the extractors dependencies
    seqs = [
        f2e[f] for f in e._conf.features] for e in extractors
    ] + [list(extractors)]

    # C3
    res = []
    while True:
      non_empty = list(filter(None, seqs))
      if not non_empty:
          # Nothing left to process, we're done.
          return tuple(res)
      for seq in non_empty:  # Find merge candidates among seq heads.
          candidate = seq[0]
          not_head = [s for s in non_empty if candidate in s[1:]]
          if not_head:
              # Reject the candidate.
              candidate = None
          else:
              break
      if not candidate:
          raise TypeError("inconsistent hierarchy, no C3 FERO is possible")
      res.append(candidate)
      for seq in non_empty:
          # Remove candidate.
          if seq[0] == candidate:
              del seq[0]

    return res
