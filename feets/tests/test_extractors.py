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

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """Extractors Tests"""


# =============================================================================
# IMPORTS
# =============================================================================

from .. import FeatureSpace, Extractor, register_extractor, extractors

import mock

from .core import FeetsTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class SortByFependenciesTest(FeetsTestCase):

    def test_sort_by_dependencies(self):
        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class B1(Extractor):
            data = ["magnitude"]
            features = ["test_b1"]
            dependencies = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class B2(Extractor):
            data = ["magnitude"]
            features = ["test_b2"]
            dependencies = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class C(Extractor):
            data = ["magnitude"]
            features = ["test_c"]
            dependencies = ["test_b1", "test_b2", "test_a"]

            def fit(self, *args):
                pass

        space = mock.MagicMock()

        a, b1, b2, c = A(space), B1(space), B2(space), C(space)
        exts = [c, b1, a, b2]
        plan = extractors.sort_by_dependencies(exts)
        for idx, ext in enumerate(plan):
            if idx == 0:
                self.assertIs(ext, a)
            elif idx in (1, 2):
                self.assertIn(ext, (b1, b2))
            elif idx == 3:
                self.assertIs(ext, c)
            else:
                self.fail("to many extractors in plan: {}".format(idx))
