#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# DOC
# =============================================================================

"""Utilities for accessing the OGLE-III On-line Catalog of Variable Stars.

The main goal of this catalog is to record all variable sources located in the
OGLE-III fields in the Magellanic Clouds and Galactic bulge. The data
currently available include:

- classical Cepheids in the Galactic Bulge, LMC and SMC,
- type II Cepheids in the Galactic Bulge, LMC and SMC,
- anomalous Cepheids in LMC and SMC,
- RR Lyrae stars in the Galactic Bulge, LMC and SMC,
- Long Period Variables in the Galactic Bulge, LMC and SMC,
- Double Period Variables in LMC,
- R CrB stars in LMC,
- Delta Sct stars in LMC.

The catalog data include basic parameters of the stars (coordinates, periods,
mean magnitudes, amplitudes, parameters of the Fourier light curve
decompositions), VI multi-epoch photometry collected since 2001, and for
some stars supplemented with the OGLE-II photometry obtained between
1997 and 2000, finding charts and cross-identifications with previously
published catalogs.

**Note to the user:** If you use or refer to the data obtained from this
catalog in your scientific work, please cite the appropriate papers:

- Udalski, Szymanski, Soszynski and Poleski, 2008, Acta Astron., 58, 69
  (OGLE-III photometry)
- Soszynski et al., 2008a, Acta Astron., 58, 163
  (Classical Cepheids in the LMC)
- Soszynski et al., 2008b, Acta Astron., 58, 293
  (Type II and Anomalous Cepheids in the LMC)
- Soszynski et al., 2009a, Acta Astron., 59, 1
  (RR Lyrae Stars in the LMC)
- Soszynski et al., 2009b, Acta Astron., 59, 239
  (Long Period Variables in the LMC)
- Soszynski et al., 2009c, Acta Astron., 59, 335
  (R CrB Variables in the LMC)
- Poleski et al., 2010a, Acta Astron., 60, 1
  (δ Scuti Variables in the LMC)
- Poleski et al., 2010b, Acta Astron., 60, 179
  (Double Period Variables in the LMC)
- Soszynski et al., 2010a, Acta Astron., 60, 17
  (Classical Cepheids in the SMC)
- Soszynski et al., 2010b, Acta Astron., 60, 91
  (Type II Cepheids in the SMC)
- Soszynski et al., 2010c, Acta Astron., 60, 165
  (RR Lyrae Stars in the SMC)
- Soszynski et al., 2011a, Acta Astron., 61, 1
  (RR Lyrae Stars in the Galactic Bulge)
- Soszynski et al., 2011b, Acta Astron., 61, 217
  (Long-Period Variables in the Small Magellanic Cloud)
- Soszynski et al., 2011c, Acta Astron., 61, 285;   2013b,
  Acta Astron., 63, 37;  (Classical and Type II Cepheids in the Galactic Bulge)
- Soszynski et al., 2013a, Acta Astron., 63, 21
  (Long-Period Variables in the Galactic Bulge)

More Info: http://ogledb.astrouw.edu.pl/~ogle/CVS/
"""

# =============================================================================
# IMPORTS
# =============================================================================

import bz2
import os
import pathlib
import warnings

import numpy as np

import pandas as pd

from . import base
from .base import LightCurveDataset
from ..extractors.extractor import DATA_ERROR, DATA_MAGNITUDE, DATA_TIME


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
OGLE_CATALOG_PATH = PATH / "data" / "ogle3.txt.bz2"

OGLE_CATALOG_BASE_URL = "http://ogledb.astrouw.edu.pl/~ogle/CVS/data"

OGLE_DATA_DIRECTORY = "ogle3"

DATASET_DESCRIPTION = """Light curve data retrieved from OGLE-3

The main goal of this catalog is to record all variable sources located in the
OGLE-III fields in the Magellanic Clouds and Galactic bulge. The data
currently available include:

- classical Cepheids in the Galactic Bulge, LMC and SMC,
- type II Cepheids in the Galactic Bulge, LMC and SMC,
- anomalous Cepheids in LMC and SMC,
- RR Lyrae stars in the Galactic Bulge, LMC and SMC,
- Long Period Variables in the Galactic Bulge, LMC and SMC,
- Double Period Variables in LMC,
- R CrB stars in LMC,
- Delta Sct stars in LMC.

The catalog data include basic parameters of the stars (coordinates, periods,
mean magnitudes, amplitudes, parameters of the Fourier light curve
decompositions), VI multi-epoch photometry collected since 2001, and for
some stars supplemented with the OGLE-II photometry obtained between
1997 and 2000, finding charts and cross-identifications with previously
published catalogs.

**Note to the user:** If you use or refer to the data obtained from this
catalog in your scientific work, please cite the appropriate papers:

- Udalski, Szymanski, Soszynski and Poleski, 2008, Acta Astron., 58, 69
  (OGLE-III photometry)
- Soszynski et al., 2008a, Acta Astron., 58, 163
  (Classical Cepheids in the LMC)
- Soszynski et al., 2008b, Acta Astron., 58, 293
  (Type II and Anomalous Cepheids in the LMC)
- Soszynski et al., 2009a, Acta Astron., 59, 1
  (RR Lyrae Stars in the LMC)
- Soszynski et al., 2009b, Acta Astron., 59, 239
  (Long Period Variables in the LMC)
- Soszynski et al., 2009c, Acta Astron., 59, 335
  (R CrB Variables in the LMC)
- Poleski et al., 2010a, Acta Astron., 60, 1
  (δ Scuti Variables in the LMC)
- Poleski et al., 2010b, Acta Astron., 60, 179
  (Double Period Variables in the LMC)
- Soszynski et al., 2010a, Acta Astron., 60, 17
  (Classical Cepheids in the SMC)
- Soszynski et al., 2010b, Acta Astron., 60, 91
  (Type II Cepheids in the SMC)
- Soszynski et al., 2010c, Acta Astron., 60, 165
  (RR Lyrae Stars in the SMC)
- Soszynski et al., 2011a, Acta Astron., 61, 1
  (RR Lyrae Stars in the Galactic Bulge)
- Soszynski et al., 2011b, Acta Astron., 61, 217
  (Long-Period Variables in the Small Magellanic Cloud)
- Soszynski et al., 2011c, Acta Astron., 61, 285;   2013b,
  Acta Astron., 63, 37;  (Classical and Type II Cepheids in the Galactic Bulge)
- Soszynski et al., 2013a, Acta Astron., 63, 21
  (Long-Period Variables in the Galactic Bulge)

More Info: http://ogledb.astrouw.edu.pl/~ogle/CVS/
"""

# =============================================================================
# FUNCTIONS
# =============================================================================


def _get_OGLE3_data_home(data_home_path):
    data_home_path = base.get_data_home(data_home=data_home_path)
    o3_data_path = data_home_path / OGLE_DATA_DIRECTORY
    o3_data_path.mkdir(parents=True, exist_ok=True)
    return o3_data_path


def _check_dim(lc):
    if lc.ndim == 1:
        # lc consists of a single observation, reshape it to a 2D array
        lc.shape = 1, 3
    return lc


def _get_path_by_band(ogle3_id, band, store_path):
    return store_path / f"{ogle3_id}.{band}.dat"


def _get_url_by_band(ogle3_id, band, store_path):
    return f"{OGLE_CATALOG_BASE_URL}/{band}/{ogle3_id[-2:]}/{ogle3_id}.dat"


def load_OGLE3_catalog():
    """List the OGLE-III catalog of variable stars as a `pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
        The full OGLE-III catalog of variable stars.
    """
    with bz2.BZ2File(OGLE_CATALOG_PATH) as bz2fp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_table(bz2fp, skiprows=6)
    df.rename(columns={"# ID": "ID"}, inplace=True)
    return df


def fetch_OGLE3(
    ogle3_id, data_home=None, metadata=True, download_if_missing=True
):
    """Retrieve a light curve from the OGLE-3 catalog.

    Parameters
    ----------
    ogle3_id : str
        The OGLE-III ID of the light curve as seen on the `load_OGLE3_catalog`
        dataframe.
    data_home : str or pathlib.Path, optional
        Cache directory for the downloaded datasets.
        See `datasets.base.get_data_home` for more info.
    metadata : bool, default=True
        If True, the row from the `load_OGLE3_catalog` dataframe corresponding
        to the given light curve will be included in the resulting dataset as
        metadata.
    download_if_missing : bool, default=True
        If True, try to download the data from the source site when it's not
        locally available. If False, it will raise a `FileNotFoundError`
        instead.

    Raises
    ------
    ValueError
        If the provided OGLE-III ID is invalid.
    FileNotFoundError
        If the data is not locally available and `download_if_missing` is set
        to False.

    Returns
    -------
    LightCurveDataset
        Dataset with the retrieved light curve data vectors and metadata (if
        `metadata` is True)

    See Also
    --------
    datasets.base.get_data_home : Return the path of the feets data directory.
    load_OGLE3_catalog

    Examples
    --------
    >>> ds = fetch_OGLE3("OGLE-BLG-LPV-232377")
    >>> ds
    LightCurveDataset(
        _id='OGLE-BLG-LPV-232377', name='OGLE-III', bands=('I', 'V')
    )
    >>> ds.bands
    ('I', 'V')
    >>> ds.data.I
    <LightCurve time[100], magnitude[100], error[100]>
    >>> ds.data.I.magnitude
    array([13.816, 13.826, 13.818, 13.812, 13.8  , 13.827, 13.797, 13.82 ,
       13.804, 13.783, 13.823, 13.8  , 13.84 , 13.817, 13.802, 13.824,
       13.822, 13.81 , 13.844, 13.848, 13.813, 13.836, 13.83 , 13.83 ,
       13.837, 13.811, 13.814, 13.82 , 13.826, 13.822, 13.821, 13.817,
       13.813, 13.809, 13.817, 13.836, 13.804, 13.801, 13.813, 13.823,
       13.818, 13.831, 13.833, 13.814, 13.814, 13.812, 13.822, 13.814,
       13.818, 13.817, 13.8  , 13.804, 13.799, 13.809, 13.815, 13.846,
       13.796, 13.791, 13.804, 13.853, 13.839, 13.816, 13.825, 13.81 ,
       13.8  , 13.807, 13.819, 13.829, 13.844, 13.84 , 13.842, 13.818,
       13.801, 13.804, 13.814, 13.821, 13.821, 13.822, 13.82 , 13.803,
       13.813, 13.826, 13.855, 13.865, 13.854, 13.828, 13.809, 13.828,
       13.833, 13.829, 13.816, 13.82 , 13.827, 13.834, 13.811, 13.817,
       13.808, 13.834, 13.814, 13.829])
    """
    # check if it's a valid ID
    cat = load_OGLE3_catalog()

    if ogle3_id not in cat.ID.values:
        raise ValueError(f"Invalid OGLE-3 ID: {ogle3_id}")
    if metadata:
        cat = load_OGLE3_catalog()
        metadata = cat[cat.ID == ogle3_id].iloc[0].to_dict()
    del cat

    # retrieve the data dir for ogle
    store_path = _get_OGLE3_data_home(data_home)

    # the two bands of ogle3
    bands = {"I", "V"}

    # download all necessary files
    if download_if_missing:
        for band in bands:
            base.fetch(
                _get_url_by_band(ogle3_id, band, store_path),
                _get_path_by_band(ogle3_id, band, store_path),
            )

    data = {}
    for band in bands:
        src = _get_path_by_band(ogle3_id, band, store_path)
        lc = _check_dim(np.loadtxt(src))

        data[band] = {
            DATA_TIME: lc[:, 0],
            DATA_MAGNITUDE: lc[:, 1],
            DATA_ERROR: lc[:, 2],
        }

    return LightCurveDataset(
        id=ogle3_id,
        metadata=metadata,
        name="OGLE-III",
        description=DATASET_DESCRIPTION,
        bands=bands,
        data=data,
    )
