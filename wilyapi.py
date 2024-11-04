# wily_api.py

import attrs
import contextlib
import io
import pathlib
from unittest import mock

import joblib

import pandas as pd

from wily import config as wconfig
from wily.commands import report, list_metrics, rank

# =============================================================================
# CONSTANTS
# =============================================================================


def _get_metrics() -> pd.DataFrame:
    rows = []
    for name, op in list_metrics.ALL_OPERATORS.items():
        for metric in op.operator_cls.metrics:
            rows.append(
                {
                    "Family": name,
                    "Metric": metric.name,
                    "Description": metric.description,
                }
            )
    return pd.DataFrame(rows)


def _get_metrics_names(metrics) -> frozenset:
    def join_metric(row):
        return f"{row.Family}.{row.Metric}"

    return frozenset(metrics.apply(join_metric, axis="columns").tolist())


METRICS = _get_metrics()
METRICS_NAMES = _get_metrics_names(METRICS)

del _get_metrics, _get_metrics_names

# =============================================================================
# API
# =============================================================================


@attrs.define(frozen=True)
class WilyAPI:
    """Class to handle Wily metrics analysis on Python code."""

    config: wconfig.WilyConfig = attrs.field(
        default=wconfig.DEFAULT_CONFIG,
        repr=True,
        validator=attrs.validators.instance_of(wconfig.WilyConfig),
    )

    @classmethod
    def from_dir(cls, path):
        path = pathlib.Path(path)
        print(path)
        if path.is_dir():
            path = path / "wily.cfg"
        config = wconfig.load(path)
        return WilyAPI(config)

    def _run_capture(self, function, kwargs):
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff):
            function(**kwargs)
        buff.seek(0)
        return buff

    def file_report(
        self,
        file_path: str,
        metrics: list | tuple = None,
        add_filename: bool = False,
        add_revision_author: bool = False,
    ) -> pd.DataFrame:
        metrics = METRICS_NAMES if metrics is None else tuple(metrics)

        params = {
            "config": self.config,
            "path": str(file_path),
            "metrics": metrics,
            "n": None,
            "output": "/dev/null",
            "console_format": "tsv",
        }

        # Capture the output
        buff = self._run_capture(report.report, params)

        import ipdb

        ipdb.set_trace()

        # Read results
        df = pd.read_table(buff)

        # Clean metric values
        def clean_metric(v):
            v = v.split()[0].strip()
            try:
                v = float(v)
            except ValueError:
                v = pd.NA
            return v

        df.columns = df.columns.str.strip()
        for column in df.columns[3:]:
            df[column] = df[column].apply(clean_metric)

        df = df.set_index("Revision", drop=True)
        df.Date = pd.to_datetime(df.Date)

        if not add_revision_author:
            df.drop(columns="Author", inplace=True)

        if add_filename:
            df["File"] = file_path

            columns_order = ["File"] + df.columns[:-1].tolist()
            df = df[columns_order]

        return df

    def files_rank(self, *, metric="mi"):
        params = {
            "config": self.config,
            "path": None,
            "metric": metric,
            "revision_index": None,
            "limit": None,
            "threshold": None,
            "descending": False,
            "wrap": True,
        }

        # Capture the output
        with mock.patch.object(rank, "get_style", new=lambda: "tsv"):
            buff = self._run_capture(rank.rank, params)

        df = pd.read_table(buff)

        df.columns = df.columns.str.strip()
        df.File = df.File.str.strip()
        df = df.iloc[:-1]

        return df

    def list_files(self, *, subdir=None):
        df = self.files_rank()
        files = sorted(df.File[df.File.str.endswith(".py")])

        if subdir:
            subdir = pathlib.Path(subdir).resolve()
            files = [
                filename
                for filename in files
                if pathlib.Path(filename).resolve().is_relative_to(subdir)
            ]

        return files

    def metric_report(self, metric, *, subdir=None, n_jobs=-1):
        files = self.list_files(subdir=subdir)
        with joblib.Parallel(n_jobs=n_jobs) as P:
            file_report = joblib.delayed(self.file_report)
            results = P(
                file_report(file, metrics=[metric], add_filename=True)
                for file in files
            )

        return pd.concat(results)

    def plot(self, metric, *, subdir=None):
        df = self.metric_report(metric, subdir=subdir)
        df = df.pivot(
            index=["Revision", "Date"],
            columns="File",
            values="Maintainability Index",
        )
