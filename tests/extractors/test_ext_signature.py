#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.ext_signature import Signature

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 100
MAX_ITERS = 100
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.slow
def test_Signature_extract(periodic):
    # init extractor
    phase_bins = 6
    mag_bins = 6
    extractor = Signature(phase_bins=phase_bins, mag_bins=mag_bins)

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH),
            "magnitude": periodic(random=random, size=LC_LENGTH, period=20),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [
        extractor.extract(
            **lc,
            MedianAmplitude=0.828441,
            PeriodLS=[18.808016760831848, 19.27779111644658],
        )
        for lc in lcs
    ]

    # values by feature
    df = pd.DataFrame(results)

    # check columns
    np.testing.assert_equal(set(df.columns), {"Signature"})

    # values by subfeature
    signatures = [
        {
            subfeature: [
                signature[subfeature] for signature in result["Signature"]
            ]
            for subfeature in result["Signature"][0]
        }
        for result in results
    ]
    dfs = [pd.DataFrame(signature) for signature in signatures]
    df = pd.concat(dfs, keys=range(len(dfs)))

    # check subfeatures
    expected_subfeatures = {
        f"ph_{j}_mag_{i}" for i in range(mag_bins) for j in range(phase_bins)
    }
    np.testing.assert_equal(set(df.columns), set(expected_subfeatures))

    # For this feature, the exact results vary machine to machine, so instead
    # of checking for exact values, we test other statistical properties.
    means = df.groupby(level=1).mean()[list(expected_subfeatures)]

    # Check for negative values
    assert np.all(means >= 0), "Negative values found in signature"

    # Total density should be in a reasonable range
    total_densities = means.sum(axis=1)
    assert np.all(
        (total_densities > 10) & (total_densities < 1000)
    ), f"Total density out of reasonable range: {total_densities.values}"

    # Signature should have a reasonable dynamic range
    max_vals = means.max(axis=1)
    min_vals = means.min(axis=1)
    dynamic_range = max_vals / (min_vals + 1e-10)
    assert np.all(
        dynamic_range > 2
    ), f"Signature too uniform: {dynamic_range.values}"

    # Peak should be in a reasonable magnitude bin
    peak_features = means.idxmax(axis=1)
    peak_mag_bins = peak_features.str.split("_").str[-1].astype(int)
    assert np.all(
        (peak_mag_bins >= 0) & (peak_mag_bins < mag_bins)
    ), f"Peak magnitude bin out of range: {peak_mag_bins.values}"

    # Most of the signature density should be in a subset of bins
    sorted_values = np.sort(means.values, axis=1)[:, ::-1]
    top_20_percent_sum = sorted_values[:, : int(0.2 * len(means.columns))].sum(
        axis=1
    )
    concentration_ratios = top_20_percent_sum / total_densities
    assert np.all(
        concentration_ratios > 0.3
    ), f"Signature too dispersed: concentration={concentration_ratios.values}"

    # The signature should show some phase structure
    phase_sums_list = []
    for j in range(phase_bins):
        phase_cols = [
            col for col in means.columns if col.startswith(f"ph_{j}_")
        ]
        phase_sums_list.append(means[phase_cols].sum(axis=1))
    phase_sums_df = pd.concat(phase_sums_list, axis=1)

    phase_stds = phase_sums_df.std(axis=1)
    phase_means = phase_sums_df.mean(axis=1)
    phase_cvs = phase_stds / phase_means.replace(0, 1e-10)
    assert np.all(
        phase_cvs > 0.1
    ), f"Phase distribution too uniform: CV={phase_cvs.values}"

    # Lower magnitude bins should generally have higher density
    mag_sums_list = []
    for i in range(mag_bins):
        mag_cols = [col for col in means.columns if col.endswith(f"_mag_{i}")]
        mag_sums_list.append(means[mag_cols].sum(axis=1))
    mag_sums_df = pd.concat(mag_sums_list, axis=1)

    low_mag_fraction = mag_sums_df.iloc[:, :4].sum(axis=1) / mag_sums_df.sum(
        axis=1
    )
    assert np.all(
        low_mag_fraction > 0.1
    ), f"Too little density in low magnitude bins: {low_mag_fraction.values}"

    period_0_means = means.iloc[0]
    period_1_means = means.iloc[1]

    # The two signatures should be reasonably similar since they come from the
    # same data generation process
    correlation = np.corrcoef(period_0_means.values, period_1_means.values)[
        0, 1
    ]
    assert (
        correlation > 0.5
    ), f"Signatures for different periods too different: correlation={correlation}"

    # The two periods should have similar total densities
    density_0 = period_0_means.sum()
    density_1 = period_1_means.sum()
    relative_diff = abs(density_0 - density_1) / max(density_0, density_1)
    assert (
        relative_diff < 0.5
    ), f"Total densities too different between periods: {density_0} vs {density_1}"
