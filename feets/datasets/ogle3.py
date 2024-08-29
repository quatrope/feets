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
# IMPORTS
# =============================================================================

import bz2
import os
import tarfile
import warnings

import numpy as np

import pandas as pd

from . import base
from .base import Data

# =============================================================================
# DOCS
# =============================================================================

"""Code for acces the OGLE-III On-line Catalog of Variable Stars.

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

# This is for add as descr in every Data instance
DESCR = "LightCurve from OGLE-3\n\n{}".format(
    "\n".join(__doc__.splitlines()[2:])
)


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

CATALOG_PATH = os.path.join(PATH, "data", "ogle3.txt.bz2")

DATA_DIR = "ogle3"

URL = "http://ogledb.astrouw.edu.pl/~ogle/CVS/sendobj.php?starcat={}"


# =============================================================================
# FUNCTIONS
# =============================================================================


def _get_OGLE3_data_home(data_home):
    # retrieve the data home
    data_home = base.get_data_home(data_home=data_home)
    o3_dh = os.path.join(data_home, DATA_DIR)
    if not os.path.exists(o3_dh):
        os.makedirs(o3_dh)
    return o3_dh


def _check_dim(lc):
    if lc.ndim == 1:
        lc.shape = 1, 3
    return lc


def load_OGLE3_catalog():
    """Return the full list of variables stars of OGLE-3 as a DataFrame"""
    with bz2.BZ2File(CATALOG_PATH) as bz2fp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_table(bz2fp, skiprows=6)
    df.rename(columns={"# ID": "ID"}, inplace=True)
    return df


def fetch_OGLE3(
    ogle3_id, data_home=None, metadata=None, download_if_missing=True
):
    """Retrieve a lighte curve from OGLE-3 database

    Parameters
    ----------
    ogle3_id : str
        The id of the source (see: ``load_OGLE3_catalog()`` for
        available sources.
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all feets data is stored in '~/feets' subfolders.
    metadata : bool | None
        If it's True, the row of the dataframe from ``load_OGLE3_catalog()``
        with the metadata of the source are added to the result.
    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------

    A Data object.


    Examples
    --------

    .. code-block:: pycon

        >>> ds = fetch_OGLE3("OGLE-BLG-LPV-232377")
        >>> ds
        Data(_id='OGLE-BLG-LPV-232377', ds_name='OGLE-III', bands=('I', 'V'))
        >>> ds.bands
        ('I', 'V')
        >>> ds.data.I
        LightCurve(time[100], magnitude[100], error[100])
        >>> ds.data.I.magnitude
        array([ 13.816,  13.826,  13.818,  13.812,  13.8  ,  13.827,  13.797,
                13.82 ,  13.804,  13.783,  13.823,  13.8  ,  13.84 ,  13.817,
                13.802,  13.824,  13.822,  13.81 ,  13.844,  13.848,  13.813,
                13.836,  13.83 ,  13.83 ,  13.837,  13.811,  13.814,  13.82 ,
                13.826,  13.822,  13.821,  13.817,  13.813,  13.809,  13.817,
                13.836,  13.804,  13.801,  13.813,  13.823,  13.818,  13.831,
                13.833,  13.814,  13.814,  13.812,  13.822,  13.814,  13.818,
                13.817,  13.8  ,  13.804,  13.799,  13.809,  13.815,  13.846,
                13.796,  13.791,  13.804,  13.853,  13.839,  13.816,  13.825,
                13.81 ,  13.8  ,  13.807,  13.819,  13.829,  13.844,  13.84 ,
                13.842,  13.818,  13.801,  13.804,  13.814,  13.821,  13.821,
                13.822,  13.82 ,  13.803,  13.813,  13.826,  13.855,  13.865,
                13.854,  13.828,  13.809,  13.828,  13.833,  13.829,  13.816,
                13.82 ,  13.827,  13.834,  13.811,  13.817,  13.808,  13.834,
                13.814,  13.829])

    """

    # retrieve the data dir for ogle
    store_path = _get_OGLE3_data_home(data_home)

    # the data dir for this lightcurve
    file_path = os.path.join(store_path, "{}.tar".format(ogle3_id))

    # members of the two bands of ogle3
    members = {
        "I": "./{}.I.dat".format(ogle3_id),
        "V": "./{}.V.dat".format(ogle3_id),
    }

    # the url of the lightcurve
    if download_if_missing:
        url = URL.format(ogle3_id)
        base.fetch(url, file_path)

    bands = []
    data = {}
    with tarfile.TarFile(file_path) as tfp:
        members_names = tfp.getnames()
        for band_name, member_name in members.items():
            if member_name in members_names:
                member = tfp.getmember(member_name)
                src = tfp.extractfile(member)
                lc = _check_dim(np.loadtxt(src))
                data[band_name] = {
                    "time": lc[:, 0],
                    "magnitude": lc[:, 1],
                    "error": lc[:, 2],
                }
                bands.append(band_name)
    if metadata:
        cat = load_OGLE3_catalog()
        metadata = cat[cat.ID == ogle3_id].iloc[0].to_dict()
        del cat

    return Data(
        _id=ogle3_id,
        metadata=metadata,
        ds_name="OGLE-III",
        description=DESCR,
        bands=bands,
        data=data,
    )
