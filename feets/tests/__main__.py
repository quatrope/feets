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

from __future__ import unicode_literals, absolute_import


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import argparse
import unittest
import importlib

from . import core, utils


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))


# =============================================================================
# FUNCTIONS
# =============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="Run the core test cases for feets")

    parser.add_argument(
        "-f", "--failfast", dest='failfast', default=False,
        help='Stop on first fail or error', action='store_true')

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-v", "--verbose", dest='verbosity',  default=0, const=1,
        help='Verbose output', action='store_const')
    group.add_argument(
        "-vv", "--vverbose", dest='verbosity', const=2,
        help='Verbose output', action='store_const')

    return parser


def load_test_modules():
    base_pkg, test_modules_names = ".".join(["feets", "tests"]), []
    for dirpath, dirnames, filenames in os.walk(PATH):
        pkg = [] if dirpath == PATH else [os.path.basename(dirpath)]
        for fname in filenames:
            name, ext = os.path.splitext(fname)
            if name.startswith("test_") and ext == ".py":
                modname = ".".join(pkg + [name])
                test_modules_names.append((base_pkg, modname))

    test_modules = set()
    for pkg, modname in test_modules_names:
        dot_modname = ".{}".format(modname)
        module = importlib.import_module(dot_modname, pkg)
        test_modules.add(module)
    return tuple(test_modules)


def collect_subclasses(cls):
    def collect(basecls):
        collected = set()
        for subcls in basecls.__subclasses__():
            collected.add(subcls)
            collected.update(collect(subcls))
        return collected
    return tuple(collect(cls))


def run_tests(verbosity=1, failfast=False):
    """Run test of feets"""

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    runner = unittest.runner.TextTestRunner(
        verbosity=verbosity, failfast=failfast)

    load_test_modules()

    for testcase in collect_subclasses(core.FeetsTestCase):
        tests = loader.loadTestsFromTestCase(testcase)
        if tests.countTestCases():
                suite.addTests(tests)
    return runner.run(suite)


def main(argv):
    parser = create_parser()
    arguments = parser.parse_args(argv)

    # RUN THE TESTS
    result = run_tests(
        verbosity=arguments.verbosity, failfast=arguments.failfast)

    # EXIT WITH CORRECT STATUS
    sys.exit(not result.wasSuccessful())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])
