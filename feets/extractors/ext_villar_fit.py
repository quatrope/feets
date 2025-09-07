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

"""Villar fit extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import VillarFit as _VillarFit

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class VillarFit(LightCurveExtractor):
    r"""Villar function fit.

    Seven fit parameters and goodness of fit (reduced :math:`\chi^2`) of
    the Villar function developed for supernovae classification:

    .. math::

        f(t) = c + \frac{A}{1 + \exp{\frac{-(t-t_0)}{\tau_\mathrm{rise}}}}
        \left\{
          \begin{array}{ll}
            1 - \frac{\nu (t - t_0)}{\gamma}, &t < t_0 + \gamma \\
            (1 - \nu) \exp{\frac{-(t-t_0-\gamma)}{\tau_\mathrm{fall} }},
            &t \geq t_0 + \gamma
          \end{array}
        \right.

    where :math:`A, \gamma, \tau_\mathrm{rise}, \tau_\mathrm{fall} > 0`,
    :math:`\nu \in [0; 1)`.

    Here we introduce a new dimensionless parameter :math:`\nu` instead of
    the plateau slope :math:`\beta` from the original paper:

    .. math::

        \nu \equiv -\beta \gamma / A

    Note, that the Villar function is developed to be used with fluxes,
    not magnitudes.

    **VillarFit_Amplitude** (:math:`A`)

    Half amplitude of the Villar function

    **VillarFit_Baseline** (:math:`c`)

    Baseline of the Villar function

    **VillarFit_ReferenceTime** (:math:`t_0`)

    Reference time of the Villar function

    **VillarFit_RiseTime** (:math:`\tau_\mathrm{rise}`)

    Rise time of the Villar function

    **VillarFit_FallTime** (:math:`\tau_\mathrm{fall}`)

    Decline time of the Villar function

    **VillarFit_PlateauRelAmplitude** (:math:`\nu = -\beta \gamma / A`)

    Relative plateau amplitude of the Villar function

    **VillarFit_PlateauDuration** (:math:`\gamma`)

    Plateau duration of the Villar function

    **VillarFit_ReducedChi2** (reduced :math:`\chi^2`)

    Villar fit quality

    Parameters
    ----------
    algorithm : str
        Non-linear least-square algorithm, supported values are:
        'mcmc', 'ceres', 'mcmc-ceres', 'lmsder', 'mcmc-lmsder'.
    mcmc_niter : int, default=128
        Number of MCMC iterations.
    ceres_niter : int, default=10
        Number of Ceres iterations.
    ceres_loss_reg : float, optional
        Ceres loss regularization, default is to use square norm as is, if set
        to a number, the loss function is regularized to discriminate outlier
        residuals larger than this value.
        Default is None which means no regularization.
    lmsder_niter : int, default=10
        Number of LMSDER iterations.
    init : list or None, optional
        Initial conditions, must be `None` or a `list` of `float`s or `None`s.
        The length of the list must be 7, `None` values will be replaced
        with some default values. It is supported by MCMC only
    bounds : list of tuples or None, optional
        Boundary conditions, must be `None` or a `list` of `tuple`s of `float`s
        or `None`s. The length of the list must be 7, boundary conditions must
        include initial conditions, `None` values will be replaced with some
        broad defaults. It is supported by MCMC only
    ln_prior : str or list of ln_prior.LnPrior1D or None, optional
        Prior for MCMC, None means no prior. It is specified by a string
        literal or a list of 7 `light_curve.ln_prior.LnPrior1D` objects, see
        `light_curve.ln_prior` submodule for corresponding functions. Available
        string literals are:

        - 'no': no prior
        - 'hosseinzadeh2020': prior adopted from Hosseinzadeh et al. 2020, it
          assumes that `t` is in days

    transform : str or bool or None, optional
        If `False` or `None` (default) output is not transformed. If `True` output
        is transformed as following:

        - Half-amplitude A is transformed as :math:`zp - 2.5 lg(2*A)`,
          :math:`zp = 8.9`, so that the amplitude is assumed to be the object
          peak flux in Jy.
        - baseline flux is normalised by :math:`A: baseline -> baseline / A`
        - reference time is removed
        - goodness of fit is transformed as :math:`ln(reduced chi^2 + 1)` to
          reduce its spread
        - other parameters are not transformed

    """

    features = [
        "VillarFit_Amplitude",
        "VillarFit_Baseline",
        "VillarFit_ReferenceTime",
        "VillarFit_RiseTime",
        "VillarFit_FallTime",
        "VillarFit_PlateauRelAmplitude",
        "VillarFit_PlateauDuration",
        "VillarFit_ReducedChi2",
    ]

    def __init__(
        self,
        algorithm="mcmc",
        mcmc_niter=128,
        lmsder_niter=10,
        ceres_niter=10,
        ceres_loss_reg=None,
        init=None,
        bounds=None,
        ln_prior=None,
        transform=None,
    ):
        self.algorithm = algorithm
        self.mcmc_niter = mcmc_niter
        self.lmsder_niter = lmsder_niter
        self.ceres_niter = ceres_niter
        self.ceres_loss_reg = ceres_loss_reg
        self.init = init
        self.bounds = bounds
        self.ln_prior = ln_prior
        self.transform = transform

        self._extract = _VillarFit(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, flux, flux_error):
        """
        Parameters
        ----------
        time : array-like
        flux : array-like
        flux_error : array-like
        """
        [
            amplitude,
            baseline,
            reference_time,
            rise_time,
            fall_time,
            plateau_rel_amplitude,
            plateau_duration,
            reduced_chi2,
        ] = self._extract(time, flux, flux_error)

        return {
            "VillarFit_Amplitude": amplitude,
            "VillarFit_Baseline": baseline,
            "VillarFit_ReferenceTime": reference_time,
            "VillarFit_RiseTime": rise_time,
            "VillarFit_FallTime": fall_time,
            "VillarFit_PlateauRelAmplitude": plateau_rel_amplitude,
            "VillarFit_PlateauDuration": plateau_duration,
            "VillarFit_ReducedChi2": reduced_chi2,
        }
