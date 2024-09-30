#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

__doc__ = """All ogle3 access tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import tarfile

import numpy as np

import pandas as pd

import mock

from ...datasets import ogle3

from ..core import FeetsTestCase, DATA_PATH


# =============================================================================
# BASE CLASS
# =============================================================================


class LoadOGLE3CatalogTestCase(FeetsTestCase):

    def test_load_OGLE3_catalog(self):
        df = pd.DataFrame({"# ID": [1, 2, 3]})
        with (
            mock.patch("pandas.read_table", return_value=df),
            mock.patch("bz2.BZ2File") as BZ2File,
        ):
            self.assertNotIn("ID", df.columns)
            self.assertIn("# ID", df.columns)
            ogle3.load_OGLE3_catalog()
            self.assertIn("ID", df.columns)
            self.assertNotIn("# ID", df.columns)
            BZ2File.assert_called_with(ogle3.CATALOG_PATH)


class LoadFetchOGLE3(FeetsTestCase):

    def test_fetch_OGLE3(self):
        store_path = ogle3._get_OGLE3_data_home(None)
        cat = ogle3.load_OGLE3_catalog()
        oid = np.random.choice(cat.ID)
        url = ogle3.URL.format(oid)
        file_path = os.path.join(store_path, "{}.tar".format(oid))

        with (
            mock.patch("feets.datasets.base.fetch") as fetch,
            mock.patch("tarfile.TarFile"),
        ):
            data = ogle3.fetch_OGLE3(oid)
            self.assertEquals(data["id"], oid)
            fetch.assert_called_with(url, file_path)

    def test_fetch_OGLE3_real_TAR(self):
        file_path = os.path.join(DATA_PATH, "OGLE-BLG-LPV-232406.tar")
        with tarfile.TarFile(file_path) as tfp:
            with (
                mock.patch("feets.datasets.base.fetch"),
                mock.patch("tarfile.TarFile", return_value=tfp),
            ):
                ogle3.fetch_OGLE3("OGLE-BLG-LPV-232406")
