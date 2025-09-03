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

import pathlib
import shutil
import tempfile

from feets.datasets.base import (
    FEETS_DATA_DIR,
    FEETS_DATA_DIR_ENV_VAR,
    LightCurve,
    LightCurveDataset,
    clear_data_home,
    fetch,
    get_data_home,
)

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = "foo/bar/"
URL = "http://fake.url/data.txt"
DEST_FILE = "data.txt"

# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_get_data_home_default(mocker):
    mocker.patch.dict("feets.datasets.base.os.environ", {}, clear=True)

    with tempfile.TemporaryDirectory() as tmpdirpath:
        mocker.patch("feets.datasets.base.HOME_PATH", pathlib.Path(tmpdirpath))

        path = pathlib.Path(tmpdirpath) / FEETS_DATA_DIR
        assert not path.exists()

        data_home = get_data_home()

        np.testing.assert_equal(data_home, path)
        assert path.exists()


def test_get_data_home_env(mocker):
    mocker.patch.dict(
        "feets.datasets.base.os.environ",
        {FEETS_DATA_DIR_ENV_VAR: DATA_DIR},
        clear=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirpath:
        mocker.patch("feets.datasets.base.HOME_PATH", pathlib.Path(tmpdirpath))

        path = pathlib.Path(tmpdirpath) / DATA_DIR
        assert not path.exists()

        data_home = get_data_home()

        np.testing.assert_equal(data_home, path)
        assert path.exists()


def test_get_data_home_arg(mocker):
    with tempfile.TemporaryDirectory() as tmpdirpath:
        mocker.patch("feets.datasets.base.HOME_PATH", pathlib.Path(tmpdirpath))

        path = pathlib.Path(tmpdirpath) / DATA_DIR
        assert not path.exists()

        data_home = get_data_home(path)

        np.testing.assert_equal(data_home, path)
        assert path.exists()


def test_clear_data_home(mocker):
    tmpdirname = tempfile.mkdtemp()

    mocker.patch("feets.datasets.base.HOME_PATH", pathlib.Path(tmpdirname))

    path = pathlib.Path(tmpdirname) / DATA_DIR
    data_home = get_data_home(path)

    file_path_1 = data_home / "file_1.txt"
    file_path_2 = data_home / "file_2.txt"
    file_path_3 = data_home / "file_3.txt"

    file_path_1.touch()
    file_path_2.touch()
    file_path_3.touch()

    assert data_home.exists()
    assert file_path_1.exists()
    assert file_path_2.exists()
    assert file_path_3.exists()

    clear_data_home(data_home)

    assert not file_path_1.exists()
    assert not file_path_2.exists()
    assert not file_path_3.exists()
    assert not data_home.exists()

    shutil.rmtree(tmpdirname)


def test_fetch_download(mocker):
    content = b"some data"

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [content]
    mock_get = mocker.patch(
        "feets.datasets.base.requests.get", return_value=mock_response
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = pathlib.Path(tmpdir) / DEST_FILE
        assert not dest.exists()

        cached, path = fetch(URL, dest)

        assert not cached
        np.testing.assert_equal(path, dest)
        assert dest.exists()
        np.testing.assert_equal(dest.read_bytes(), content)
        mock_get.assert_called_once_with(URL, stream=True)


def test_fetch_cached(mocker):
    initial_content = b"existing data"
    mock_get = mocker.patch("feets.datasets.base.requests.get")

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = pathlib.Path(tmpdir) / DEST_FILE
        dest.write_bytes(initial_content)

        cached, path = fetch(URL, dest, force=False)

        assert cached
        np.testing.assert_equal(path, dest)
        np.testing.assert_equal(dest.read_bytes(), initial_content)
        mock_get.assert_not_called()


def test_fetch_force(mocker):
    initial_content = b"old data"
    new_content = b"new data"

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [new_content]
    mock_get = mocker.patch(
        "feets.datasets.base.requests.get", return_value=mock_response
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = pathlib.Path(tmpdir) / DEST_FILE
        dest.write_bytes(initial_content)

        cached, path = fetch(URL, dest, force=True)

        assert not cached
        np.testing.assert_equal(path, dest)
        np.testing.assert_equal(dest.read_bytes(), new_content)
        mock_get.assert_called_once_with(URL, stream=True)


def test_fetch_http_error(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 404
    mock_get = mocker.patch(
        "feets.datasets.base.requests.get", return_value=mock_response
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = pathlib.Path(tmpdir) / "data.txt"
        assert not dest.exists()

        cached, path = fetch(URL, dest)

        assert not cached
        np.testing.assert_equal(path, dest)
        assert not dest.exists()
        mock_get.assert_called_once_with(URL, stream=True)


# =============================================================================
# TEST LIGHT CURVE
# =============================================================================


def test_LightCurve_init():
    time = [1, 2, 3]
    magnitude = [10, 20, 30]
    error = [0.1, 0.2, 0.3]

    lc = LightCurve(time=time, magnitude=magnitude, error=error)

    np.testing.assert_array_equal(lc.time, time)
    np.testing.assert_array_equal(lc.magnitude, magnitude)
    np.testing.assert_array_equal(lc.error, error)


def test_LightCurve_init_minimal():
    time = [1, 2, 3]
    lc = LightCurve(time=time)

    np.testing.assert_array_equal(lc.time, time)
    np.testing.assert_array_equal(lc.magnitude, None)
    np.testing.assert_array_equal(lc.error, None)


def test_LightCurve_init_missing_time():
    with pytest.raises(TypeError):
        LightCurve(magnitude=[10, 20, 30], error=[0.1, 0.2, 0.3])


def test_LightCurve_conversion():
    time = [1, 2, 3]
    magnitude = [10, 20, 30]
    error = [0.1, 0.2, 0.3]

    lc = LightCurve(time=time, magnitude=magnitude, error=error)

    assert isinstance(lc.time, np.ndarray)
    assert isinstance(lc.magnitude, np.ndarray)
    assert isinstance(lc.error, np.ndarray)


def test_LightCurve_getitem():
    time = [1, 2, 3]
    magnitude = [10, 20, 30]
    lc = LightCurve(time=time, magnitude=magnitude)

    np.testing.assert_array_equal(lc["time"], time)
    np.testing.assert_array_equal(lc["magnitude"], magnitude)
    assert lc["error"] is None

    with np.testing.assert_raises(KeyError):
        _ = lc["invalid_data"]


def test_LightCurve_len():
    lc_minimal = LightCurve(time=[1, 2, 3])
    np.testing.assert_equal(len(lc_minimal), 1)

    lc = LightCurve(
        time=[1, 2, 3],
        magnitude=[10, 20, 30],
        error=[0.1, 0.2, 0.3],
    )
    np.testing.assert_equal(len(lc), 3)


def test_LightCurve_iter():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])
    keys = list(lc)
    assert "time" in keys
    assert "magnitude" in keys
    assert "error" not in keys
    np.testing.assert_equal(len(keys), 2)


def test_LightCurve_repr():
    lc_minimal = LightCurve(time=[1, 2, 3])
    np.testing.assert_equal(repr(lc_minimal), "<LightCurve time[3]>")

    lc = LightCurve(
        time=[1, 2, 3],
        magnitude=[10, 20, 30],
        error=[0.1, 0.2, 0.3],
    )
    np.testing.assert_equal(
        repr(lc), "<LightCurve time[3], magnitude[3], error[3]>"
    )


# =============================================================================
# TEST LIGHT CURVE DATASET
# =============================================================================


def test_LightCurveDataset_init():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])

    _id = "test_dataset"
    name = "Test Dataset"
    description = "A test dataset."
    bands = ("N",)
    data = {"N": lc}

    ds = LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
    )

    np.testing.assert_array_equal(ds._id, _id)
    np.testing.assert_array_equal(ds.name, name)
    np.testing.assert_array_equal(ds.description, description)
    np.testing.assert_array_equal(ds.bands, bands)
    np.testing.assert_array_equal(ds.data.N, data["N"])
    assert ds.metadata is None


def test_LightCurveDataset_init_full():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])

    _id = "test_dataset"
    name = "Test Dataset"
    description = "A test dataset."
    bands = ("N",)
    data = {"N": lc}
    metadata = {"foo": "test"}

    ds = LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
        metadata=metadata,
    )

    np.testing.assert_array_equal(ds._id, _id)
    np.testing.assert_array_equal(ds.name, name)
    np.testing.assert_array_equal(ds.description, description)
    np.testing.assert_array_equal(ds.bands, bands)
    np.testing.assert_array_equal(ds.data.N, lc)
    np.testing.assert_array_equal(ds.metadata.foo, "test")


def test_LightCurveDataset_init_missing_required():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])

    _id = "test_dataset"
    name = "Test Dataset"
    description = "A test dataset."
    bands = ("N",)
    data = {"N": lc}

    with pytest.raises(TypeError):
        LightCurveDataset(
            name=name, description=description, bands=bands, data=data
        )

    with pytest.raises(TypeError):
        LightCurveDataset(
            id=_id, description=description, bands=bands, data=data
        )

    with pytest.raises(TypeError):
        LightCurveDataset(id=_id, name=name, bands=bands, data=data)

    with pytest.raises(TypeError):
        LightCurveDataset(
            id=_id, name=name, description=description, data=data
        )

    with pytest.raises(TypeError):
        LightCurveDataset(
            id=_id, name=name, description=description, bands=bands
        )


def test_LightCurveDataset_getitem():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])

    _id = "test_dataset"
    name = "Test Dataset"
    description = "A test dataset."
    bands = ("N",)
    data = {"N": lc}

    ds = LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
    )

    np.testing.assert_array_equal(ds["_id"], _id)
    np.testing.assert_array_equal(ds["name"], name)
    np.testing.assert_array_equal(ds["description"], description)
    np.testing.assert_array_equal(ds["bands"], bands)
    np.testing.assert_array_equal(ds["data"]["N"], lc)
    assert ds["metadata"] is None


def test_LightCurveDataset_len():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])

    _id = "test_dataset"
    name = "Test Dataset"
    description = "A test dataset."
    bands = ("N",)
    data = {"N": lc}
    metadata = {"foo": "test"}

    ds = LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
    )

    ds_full = LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
        metadata=metadata,
    )

    np.testing.assert_array_equal(len(ds), 5)
    np.testing.assert_array_equal(len(ds_full), 6)


def test_LightCurveDataset_iter():
    lc = LightCurve(time=[1, 2, 3], magnitude=[10, 20, 30])

    _id = "test_dataset"
    name = "Test Dataset"
    description = "A test dataset."
    bands = ("N",)
    data = {"N": lc}

    ds = LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
    )

    keys = list(ds)
    assert "_id" in keys
    assert "name" in keys
    assert "description" in keys
    assert "bands" in keys
    assert "data" in keys
    assert "metadata" not in keys
    np.testing.assert_equal(len(keys), 5)
