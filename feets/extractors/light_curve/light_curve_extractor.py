#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..extractor import (
    _ExtractorConf,
    _is_abstract_method,
    Extractor,
    ExtractorBadDefinedError,
)
from ...libs import doctools

# =============================================================================
# LIGHT CURVE EXTRACTOR CLASS
# =============================================================================


class LightCurveExtractor(Extractor):
    __abstractclass__ = True

    def __init_subclass__(cls):
        cls.__abstractclass__ = False
        cls_name = cls.__qualname__

        if cls.is_abstract():
            return

        if _is_abstract_method(cls.extract):
            raise ExtractorBadDefinedError(
                f"'{cls_name}.extract()' method must be redefined"
            )

        cls._conf = _ExtractorConf.from_extractor_class(cls)
        del cls.features

    # PERSISTENCE =============================================================

    def to_dict(self):
        """Convert the LightCurveExtractor to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing the parameters of the extractor instance.
        """
        cls_name = type(self).__name__
        state = vars(self)
        if "lightcurve_ext" in state:
            del state["lightcurve_ext"]
        return {cls_name: state}

    # MAGIC ===================================================================

    def __repr__(self):
        """Return a string representation of the LightCurveExtractor object."""
        cls_name = type(self).__name__
        state = {}
        for aname, avalue in vars(self).items():
            if aname == "lightcurve_ext":
                continue
            if len(repr(avalue)) > 20:
                avalue = "<MANY CONFIGURATIONS>"
            state[aname] = avalue
        return f"<{cls_name} {state}>" if state else f"<{cls_name}>"

    # API =====================================================================

    @doctools.doc_inherit(Extractor.prepare_extract)
    def prepare_extract(self, data, dependencies):
        time, magnitude, flux, error, flux_error = (
            data.get("time"),
            data.get("magnitude"),
            data.get("flux"),
            data.get("error"),
            data.get("flux_error"),
        )

        shape = (
            len(time)
            if time is not None
            else len(magnitude) if magnitude is not None else len(flux)
        )

        time = (
            np.arange(shape, dtype=np.float64)
            if time is None
            else np.array(time, dtype=np.float64)
        )
        magnitude = (
            np.zeros(shape, dtype=np.float64)
            if magnitude is None
            else np.array(magnitude, dtype=np.float64)
        )
        flux = (
            np.zeros(shape, dtype=np.float64)
            if flux is None
            else np.array(flux, dtype=np.float64)
        )
        error = (
            np.ones(shape, dtype=np.float64)
            if error is None
            else np.array(1 / error**2, dtype=np.float64)
        )
        flux_error = (
            np.ones(shape, dtype=np.float64)
            if flux_error is None
            else np.array(1 / flux_error**2, dtype=np.float64)
        )

        data.update(
            {
                "time": time,
                "magnitude": magnitude,
                "flux": flux,
                "error": error,
                "flux_error": flux_error,
            }
        )

        kwargs = super().prepare_extract(data, dependencies)
        return kwargs
