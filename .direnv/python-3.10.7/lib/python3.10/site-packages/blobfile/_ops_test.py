# https://github.com/tensorflow/tensorflow/issues/27023
import datetime
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re

import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np

import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common

GCS_TEST_BUCKET = os.getenv("GCS_TEST_BUCKET", "csh-test-3")
AS_TEST_ACCOUNT = os.getenv("AS_TEST_ACCOUNT", "cshteststorage2")
AS_TEST_ACCOUNT2 = os.getenv("AS_TEST_ACCOUNT2", "cshteststorage3")
AS_TEST_CONTAINER = os.getenv("AS_TEST_CONTAINER", "testcontainer2")
AS_TEST_CONTAINER2 = os.getenv("AS_TEST_CONTAINER2", "testcontainer3")
AS_INVALID_ACCOUNT = f"{AS_TEST_ACCOUNT}-does-not-exist"
AS_EXTERNAL_ACCOUNT = "cshteststorage4"

AZURE_VALID_CONTAINER = f"az://{AS_TEST_ACCOUNT}/{AS_TEST_CONTAINER}"
AZURE_INVALID_CONTAINER = f"az://{AS_TEST_ACCOUNT}/{AS_TEST_CONTAINER}-does-not-exist"
AZURE_INVALID_CONTAINER_NO_ACCOUNT = f"az://{AS_INVALID_ACCOUNT}/{AS_TEST_CONTAINER}"
GCS_VALID_BUCKET = f"gs://{GCS_TEST_BUCKET}"
GCS_INVALID_BUCKET = f"gs://{GCS_TEST_BUCKET}-does-not-exist"

AZURE_PUBLIC_URL = (
    f"https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat.png"
)
AZURE_PUBLIC_URL_HEADER = b"\x89PNG"


@pytest.fixture(scope="session", autouse=True)
def setup_gcloud_auth():
    # only run this for our docker tests, this tells gcloud to use the credentials supplied by the
    # test running script
    if platform.system() == "Linux" and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        sp.run(
            [
                "gcloud",
                "auth",
                "activate-service-account",
                f"--key-file={os.environ['GOOGLE_APPLICATION_CREDENTIALS']}",
            ]
        )
    yield


@contextlib.contextmanager
def chdir(path):
    original_path = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(original_path)


@contextlib.contextmanager
def _get_temp_local_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert isinstance(tmpdir, str)
        path = os.path.join(tmpdir, "file.name")
        yield path


@contextlib.contextmanager
def _get_temp_gcs_path():
    path = f"gs://{GCS_TEST_BUCKET}/" + "".join(
        random.choice(string.ascii_lowercase) for i in range(16)
    )
    gfile.mkdir(path)
    yield path + "/file.name"
    gfile.rmtree(path)


@contextlib.contextmanager
def _get_temp_as_path(account=AS_TEST_ACCOUNT, container=AS_TEST_CONTAINER):
    random_id = "".join(random.choice(string.ascii_lowercase) for i in range(16))
    path = f"https://{account}.blob.core.windows.net/{container}/" + random_id
    yield path + "/file.name"
    sp.run(
        [
            "az",
            "storage",
            "blob",
            "delete-batch",
            "--account-name",
            account,
            "--source",
            container,
            "--pattern",
            f"{random_id}/*",
        ],
        check=True,
        shell=platform.system() == "Windows",
    )


def _write_contents(path, contents):
    if ".blob.core.windows.net" in path or path.startswith("az://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert isinstance(tmpdir, str)
            account, container, blob = azure.split_path(path)
            filepath = os.path.join(tmpdir, "tmp")
            with open(filepath, "wb") as f:
                f.write(contents)
            sp.run(
                [
                    "az",
                    "storage",
                    "blob",
                    "upload",
                    "--overwrite",
                    "--account-name",
                    account,
                    "--container-name",
                    container,
                    "--name",
                    blob,
                    "--file",
                    filepath,
                ],
                check=True,
                shell=platform.system() == "Windows",
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
    else:
        with gfile.GFile(path, "wb") as f:
            f.write(contents)


def _read_contents(path):
    if ".blob.core.windows.net" in path:
        with tempfile.TemporaryDirectory() as tmpdir:
            assert isinstance(tmpdir, str)
            account, container, blob = azure.split_path(path)
            filepath = os.path.join(tmpdir, "tmp")
            sp.run(
                [
                    "az",
                    "storage",
                    "blob",
                    "download",
                    "--account-name",
                    account,
                    "--container-name",
                    container,
                    "--name",
                    blob,
                    "--file",
                    filepath,
                ],
                check=True,
                shell=platform.system() == "Windows",
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
            with open(filepath, "rb") as f:
                return f.read()
    else:
        with gfile.GFile(path, "rb") as f:
            return f.read()


def test_basename():
    testcases = [
        ("/", ""),
        ("a/", ""),
        ("a", "a"),
        ("a/b", "b"),
        ("", ""),
        ("gs://a", ""),
        ("gs://a/", ""),
        ("gs://a/b/", ""),
        ("gs://a/b", "b"),
        ("gs://a/b/c/test.filename", "test.filename"),
        ("https://a.blob.core.windows.net/b", ""),
        ("https://a.blob.core.windows.net/b/", ""),
        ("https://a.blob.core.windows.net/b/c/", ""),
        ("https://a.blob.core.windows.net/b/c", "c"),
        ("https://a.blob.core.windows.net/b/c/test.filename", "test.filename"),
    ]
    for input_, desired_output in testcases:
        actual_output = bf.basename(input_)
        assert desired_output == actual_output


def test_dirname():
    b = bf.create_context(output_az_paths=False)
    testcases = [
        ("a", ""),
        ("a/b", "a"),
        ("a/b/c", "a/b"),
        ("a/b/c/", "a/b/c"),
        ("a/b/c/////", "a/b/c"),
        ("", ""),
        ("gs://a", "gs://a"),
        ("gs://a/", "gs://a"),
        ("gs://a/////", "gs://a"),
        ("gs://a/b", "gs://a"),
        ("gs://a/b/c/test.filename", "gs://a/b/c"),
        ("gs://a/b/c/", "gs://a/b"),
        ("gs://a/b/c/////", "gs://a/b"),
        (
            "https://a.blob.core.windows.net/container",
            "https://a.blob.core.windows.net/container",
        ),
        (
            "https://a.blob.core.windows.net/container/",
            "https://a.blob.core.windows.net/container",
        ),
        (
            "https://a.blob.core.windows.net/container/////",
            "https://a.blob.core.windows.net/container",
        ),
        (
            "https://a.blob.core.windows.net/container/b",
            "https://a.blob.core.windows.net/container",
        ),
        (
            "https://a.blob.core.windows.net/container/b/c/test.filename",
            "https://a.blob.core.windows.net/container/b/c",
        ),
        (
            "https://a.blob.core.windows.net/container/b/c/",
            "https://a.blob.core.windows.net/container/b",
        ),
        (
            "https://a.blob.core.windows.net/container/b/c//////",
            "https://a.blob.core.windows.net/container/b",
        ),
    ]
    for input_, desired_output in testcases:
        actual_output = b.dirname(input_)
        assert desired_output == actual_output, f"{input_}"


def test_join():
    b = bf.create_context(output_az_paths=False)
    testcases = [
        ("a", "b", "a/b"),
        ("a/b", "c", "a/b/c"),
        ("a/b/", "c", "a/b/c"),
        ("a/b/", "c/", "a/b/c/"),
        ("a/b/", "/c/", "/c/"),
        ("", "", ""),
        (
            "gs://a/b/c",
            "d0123456789-._~!$&'()*+,:=@;",
            "gs://a/b/c/d0123456789-._~!$&'()*+,:=@;",
        ),
        ("gs://a", "b", "gs://a/b"),
        ("gs://a/b", "c", "gs://a/b/c"),
        ("gs://a/b/", "c", "gs://a/b/c"),
        ("gs://a/b/", "c/", "gs://a/b/c/"),
        ("gs://a/b/", "/c/", "gs://a/c/"),
        ("gs://a/b/", "../c", "gs://a/c"),
        ("gs://a/b/", "../c/", "gs://a/c/"),
        ("gs://a/b/", "../../c/", "gs://a/c/"),
        (
            "https://a.blob.core.windows.net/container",
            "b",
            "https://a.blob.core.windows.net/container/b",
        ),
        (
            "https://a.blob.core.windows.net/container/b",
            "c",
            "https://a.blob.core.windows.net/container/b/c",
        ),
        (
            "https://a.blob.core.windows.net/container/b/",
            "c",
            "https://a.blob.core.windows.net/container/b/c",
        ),
        (
            "https://a.blob.core.windows.net/container/b/",
            "c/",
            "https://a.blob.core.windows.net/container/b/c/",
        ),
        (
            "https://a.blob.core.windows.net/container/b/",
            "/c/",
            "https://a.blob.core.windows.net/container/c/",
        ),
        (
            "https://a.blob.core.windows.net/container/b/",
            "../c",
            "https://a.blob.core.windows.net/container/c",
        ),
        (
            "https://a.blob.core.windows.net/container/b/",
            "../c/",
            "https://a.blob.core.windows.net/container/c/",
        ),
        (
            "https://a.blob.core.windows.net/container/b/",
            "../../c/",
            "https://a.blob.core.windows.net/container/c/",
        ),
        ("gs://test/a/b", "c:d", "gs://test/a/b/c:d"),
        ("gs://test/a/b", "c:d;", "gs://test/a/b/c:d;"),
    ]
    for input_a, input_b, desired_output in testcases:
        actual_output = b.join(input_a, input_b)
        assert desired_output == actual_output, f"{input_a} {input_b}"
        # also make sure az:// urls work
        if "blob.core.windows.net" in input_a:
            az_input_a = _convert_https_to_az(input_a)
            actual_output = b.join(az_input_a, input_b)
            assert desired_output == actual_output, f"{az_input_a} {input_b}"


def _convert_https_to_az(path):
    return path.replace("https://", "az://").replace(".blob.core.windows.net", "")


def _convert_az_to_https(path):
    return re.sub(r"^az://(.*?)/", r"https://\1.blob.core.windows.net/", path)


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_get_url(ctx):
    contents = b"meow!"
    with ctx() as path:
        _write_contents(path, contents)
        url, _ = bf.get_url(path)
        assert urllib.request.urlopen(url).read() == contents


def test_azure_public_get_url():
    contents = urllib.request.urlopen(AZURE_PUBLIC_URL).read()
    assert contents.startswith(AZURE_PUBLIC_URL_HEADER)
    url, _ = bf.get_url(AZURE_PUBLIC_URL)
    assert urllib.request.urlopen(url).read() == contents


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
@pytest.mark.parametrize("streaming", [True, False])
def test_read_write(ctx, streaming):
    contents = b"meow!\npurr\n"
    with ctx() as path:
        path = bf.join(path, "a folder", "a.file")
        bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, "wb", streaming=streaming) as w:
            w.write(contents)
            assert w.tell() == len(contents)
        with bf.BlobFile(path, "rb", streaming=streaming) as r:
            assert r.read() == contents
            assert r.tell() == len(contents)
        with bf.BlobFile(path, "rb", streaming=streaming) as r:
            lines = list(r)
            assert b"".join(lines) == contents


@pytest.mark.parametrize("ctx", [_get_temp_gcs_path, _get_temp_as_path])
def test_zipfile(ctx):
    contents = b"meow!\npurr\n"
    with ctx() as path:
        with bf.BlobFile(path, "wb", streaming=True) as f:
            with zipfile.ZipFile(f, "w") as zf:
                with zf.open("eggs.txt", "w") as myfile:
                    myfile.write(contents)

        with bf.BlobFile(path, "rb", streaming=True) as f:
            with zipfile.ZipFile(f, "r") as zf:
                with zf.open("eggs.txt", "r") as myfile:
                    assert myfile.read() == contents


def test_az_path():
    contents = b"meow!\npurr\n"
    with _get_temp_as_path() as path:
        path = _convert_https_to_az(path)
        path = bf.join(path, "a folder", "a.file")
        path = _convert_https_to_az(path)
        bf.makedirs(_convert_https_to_az(bf.dirname(path)))
        with bf.BlobFile(path, "wb") as w:
            w.write(contents)
        with bf.BlobFile(path, "rb") as r:
            assert r.read() == contents
        with bf.BlobFile(path, "rb") as r:
            lines = list(r)
            assert b"".join(lines) == contents


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_append(ctx):
    contents = b"meow!\n"
    additional_contents = b"purr\n"
    with ctx() as path:
        with bf.BlobFile(path, "ab", streaming=False) as w:
            w.write(contents)
        with bf.BlobFile(path, "ab", streaming=False) as w:
            w.write(additional_contents)
        with bf.BlobFile(path, "rb") as r:
            assert r.read() == contents + additional_contents


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_stat(ctx):
    contents = b"meow!"
    with ctx() as path:
        _write_contents(path, contents)
        s = bf.stat(path)
        assert s.size == len(contents)
        assert abs(time.time() - s.mtime) <= 30


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_set_mtime(ctx):
    contents = b"meow!"
    with ctx() as path:
        _write_contents(path, contents)
        s = bf.stat(path)
        assert abs(time.time() - s.mtime) <= 30
        new_mtime = 1
        assert bf.set_mtime(path, new_mtime)
        assert bf.stat(path).mtime == new_mtime


@pytest.mark.parametrize("ctx", [_get_temp_as_path])
def test_azure_metadata(ctx):
    # make sure metadata is preserved when opening a file for writing
    # which clears uncommitted blocks
    contents = b"meow!"

    with ctx() as path:
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)

        bf.set_mtime(path, 1)
        time.sleep(5)
        with bf.BlobFile(path, "wb", streaming=True) as f:
            st = bf.stat(path)
        assert st.mtime == 1


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_remove(ctx):
    contents = b"meow!"
    with ctx() as path:
        _write_contents(path, contents)
        assert bf.exists(path)
        bf.remove(path)
        assert not bf.exists(path)


@pytest.mark.parametrize(
    # don't test local path because that has slightly different behavior
    "ctx",
    [_get_temp_gcs_path, _get_temp_as_path],
)
def test_rmdir(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        # this is an error for a local path but not for a blob path
        bf.rmdir(bf.join(dirpath, "fakedirname"))
        new_dirpath = bf.join(dirpath, "dirname")
        bf.makedirs(new_dirpath)
        assert bf.exists(new_dirpath)
        bf.rmdir(new_dirpath)
        assert not bf.exists(new_dirpath)

        # double delete is fine
        bf.rmdir(new_dirpath)

        # implicit dir
        new_filepath = bf.join(dirpath, "dirname", "name")
        _write_contents(new_filepath, contents)
        with pytest.raises(OSError):
            # not empty dir
            bf.rmdir(new_dirpath)
        bf.remove(new_filepath)
        bf.rmdir(new_dirpath)


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_makedirs(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.join(path, "x", "x", "x")
        bf.makedirs(dirpath)
        assert bf.exists(dirpath)
        _write_contents(bf.join(dirpath, "testfile"), contents)


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_isdir(ctx):
    contents = b"meow!"
    with ctx() as path:
        assert not bf.isdir(path)
        _write_contents(path, contents)
        assert not bf.isdir(path)

        dirpath = path + ".dir"
        bf.makedirs(dirpath)
        assert bf.isdir(dirpath)
        assert not bf.isdir(dirpath[:-1])

        filepath = bf.join(path + ".otherdir", "subdir", "file.name")
        if "://" not in path:
            # implicit directory
            bf.makedirs(bf.dirname(filepath))
        dirpath = bf.dirname(bf.dirname(filepath))
        _write_contents(filepath, contents)
        assert bf.isdir(dirpath)
        assert not bf.isdir(dirpath[:-1])


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_listdir(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        bf.makedirs(dirpath)
        a_path = bf.join(dirpath, "a")
        with bf.BlobFile(a_path, "wb") as w:
            w.write(contents)
        b_path = bf.join(dirpath, "b")
        with bf.BlobFile(b_path, "wb") as w:
            w.write(contents)
        bf.makedirs(bf.join(dirpath, "c"))
        expected = ["a", "b", "c"]
        assert sorted(list(bf.listdir(dirpath))) == expected
        dirpath = _convert_https_to_az(dirpath)
        assert sorted(list(bf.listdir(dirpath))) == expected


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_scandir(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, "a")
        with bf.BlobFile(a_path, "wb") as w:
            w.write(contents)
        b_path = bf.join(dirpath, "b")
        with bf.BlobFile(b_path, "wb") as w:
            w.write(contents)
        bf.makedirs(bf.join(dirpath, "c"))
        entries = sorted(list(bf.scandir(dirpath)))
        assert [e.name for e in entries] == ["a", "b", "c"]
        assert [e.path for e in entries] == [
            bf.join(dirpath, name) for name in ["a", "b", "c"]
        ]
        assert [e.is_dir for e in entries] == [False, False, True]
        assert [e.is_file for e in entries] == [True, True, False]
        assert entries[0].stat.size == len(contents)
        assert entries[1].stat.size == len(contents)
        assert entries[2].stat is None


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_listdir_sharded(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        with bf.BlobFile(bf.join(dirpath, "a"), "wb") as w:
            w.write(contents)
        with bf.BlobFile(bf.join(dirpath, "aa"), "wb") as w:
            w.write(contents)
        with bf.BlobFile(bf.join(dirpath, "b"), "wb") as w:
            w.write(contents)
        with bf.BlobFile(bf.join(dirpath, "ca"), "wb") as w:
            w.write(contents)
        bf.makedirs(bf.join(dirpath, "c"))
        with bf.BlobFile(bf.join(dirpath, "c/a"), "wb") as w:
            w.write(contents)
        # this should also test shard_prefix_length=2 but that takes too long
        assert sorted(list(bf.listdir(dirpath, shard_prefix_length=1))) == [
            "a",
            "aa",
            "b",
            "c",
            "ca",
        ]


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
@pytest.mark.parametrize("topdown", [False, True])
def test_walk(ctx, topdown):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, "a")
        with bf.BlobFile(a_path, "wb") as w:
            w.write(contents)
        bf.makedirs(bf.join(dirpath, "c/d"))
        b_path = bf.join(dirpath, "c/d/b")
        with bf.BlobFile(b_path, "wb") as w:
            w.write(contents)
        expected = [
            (dirpath, ["c"], ["a"]),
            (bf.join(dirpath, "c"), ["d"], []),
            (bf.join(dirpath, "c", "d"), [], ["b"]),
        ]
        if not topdown:
            expected = list(reversed(expected))
        assert list(bf.walk(dirpath, topdown=topdown)) == expected
        dirpath = _convert_https_to_az(dirpath)
        assert list(bf.walk(dirpath, topdown=topdown)) == expected


@pytest.mark.parametrize("ctx", [_get_temp_local_path])
def test_local_glob(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, "ab")
        with bf.BlobFile(a_path, "wb") as w:
            w.write(contents)
        b_path = bf.join(dirpath, "bb")
        with bf.BlobFile(b_path, "wb") as w:
            w.write(contents)

        def assert_listing_equal(path, desired):
            desired = sorted([bf.join(dirpath, p) for p in desired])
            actual = sorted(list(bf.glob(path)))
            assert actual == desired, f"{actual} != {desired}"

        # example patterns (didn't add tests for all these)
        # /a/b*c
        # /a/*b
        # /a/c*
        # /a/*
        # /a/*/
        # /a/b*c/d
        # a/b*c
        # a/*b
        # a/c*
        # a/b*c/d
        # b*c
        # *c
        # b*
        # *
        # */
        # */a
        with chdir(dirpath):
            assert_listing_equal("ab", ["ab"])
            assert_listing_equal("*b", ["ab", "bb"])
            assert_listing_equal("a*", ["ab"])
            assert_listing_equal("ab*", ["ab"])
            assert_listing_equal("*", ["ab", "bb"])
            assert_listing_equal("bb", ["bb"])


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
@pytest.mark.parametrize("parallel", [False, True])
def test_glob(ctx, parallel):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, "ab")
        with bf.BlobFile(a_path, "wb") as w:
            w.write(contents)
        b_path = bf.join(dirpath, "bb")
        with bf.BlobFile(b_path, "wb") as w:
            w.write(contents)

        def assert_listing_equal(path, desired):
            desired = sorted([bf.join(dirpath, p) for p in desired])
            actual = sorted(list(bf.glob(path, parallel=parallel)))
            assert actual == desired, f"{actual} != {desired}"

        assert_listing_equal(bf.join(dirpath, "*b"), ["ab", "bb"])
        assert_listing_equal(bf.join(dirpath, "a*"), ["ab"])
        assert_listing_equal(bf.join(dirpath, "ab*"), ["ab"])
        assert_listing_equal(bf.join(dirpath, "*"), ["ab", "bb"])
        assert_listing_equal(bf.join(dirpath, "bb"), ["bb"])

        path = bf.join(dirpath, "test.txt")
        with bf.BlobFile(path, "wb") as w:
            w.write(contents)
        path = bf.join(dirpath, "subdir", "test.txt")
        bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)
        path = bf.join(dirpath, "subdir", "subsubdir", "test.txt")
        if "://" not in path:
            # implicit directory
            bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)

        assert_listing_equal(bf.join(dirpath, "*/test.txt"), ["subdir/test.txt"])
        assert_listing_equal(bf.join(dirpath, "*/*.txt"), ["subdir/test.txt"])
        assert_listing_equal(
            bf.join(dirpath, "**.txt"),
            ["test.txt", "subdir/test.txt", "subdir/subsubdir/test.txt"],
        )
        assert_listing_equal(
            bf.join(dirpath, "sub**.txt"),
            ["subdir/test.txt", "subdir/subsubdir/test.txt"],
        )
        assert_listing_equal(
            bf.join(dirpath, "subdir/**.txt"),
            ["subdir/test.txt", "subdir/subsubdir/test.txt"],
        )
        assert_listing_equal(
            bf.join(dirpath, "**/*.txt"),
            ["subdir/test.txt", "subdir/subsubdir/test.txt"],
        )

        assert_listing_equal(bf.join(dirpath, "*/test"), [])
        assert_listing_equal(bf.join(dirpath, "subdir/test.txt"), ["subdir/test.txt"])

        # directories
        assert_listing_equal(bf.join(dirpath, "*"), ["ab", "bb", "subdir", "test.txt"])
        assert_listing_equal(bf.join(dirpath, "subdir"), ["subdir"])
        assert_listing_equal(bf.join(dirpath, "subdir/"), ["subdir"])
        assert_listing_equal(bf.join(dirpath, "*/"), ["subdir"])
        assert_listing_equal(bf.join(dirpath, "*dir"), ["subdir"])
        assert_listing_equal(bf.join(dirpath, "subdir/*dir"), ["subdir/subsubdir"])
        assert_listing_equal(bf.join(dirpath, "subdir/*dir/"), ["subdir/subsubdir"])
        assert_listing_equal(bf.join(dirpath, "su*ir/*dir/"), ["subdir/subsubdir"])


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_scanglob(ctx):
    contents = b"meow!"
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, "ab")
        with bf.BlobFile(a_path, "wb") as w:
            w.write(contents)
        b_path = bf.join(dirpath, "bb")
        with bf.BlobFile(b_path, "wb") as w:
            w.write(contents)
        path = bf.join(dirpath, "test.txt")
        with bf.BlobFile(path, "wb") as w:
            w.write(contents)
        path = bf.join(dirpath, "subdir", "test.txt")
        bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)

        entries = sorted(list(bf.scanglob(bf.join(dirpath, "*b*"))))
        assert entries[0].name == "ab" and entries[0].is_file
        assert entries[1].name == "bb" and entries[1].is_file
        assert entries[2].name == "subdir" and entries[2].is_dir

        for shard_prefix_length in [0, 1]:
            for pattern in ["*b", "b*", "**", "b**", "**t", "*b*"]:
                normal_entries = sorted(list(bf.scanglob(bf.join(dirpath, pattern))))
                parallel_entries = sorted(
                    list(
                        bf.scanglob(
                            bf.join(dirpath, pattern),
                            parallel=True,
                            shard_prefix_length=shard_prefix_length,
                        )
                    )
                )
                assert parallel_entries == normal_entries

        if "://" in path:
            # ** behaves a bit differently on local paths, so don't check those
            entries = sorted(list(bf.scanglob(bf.join(dirpath, "**"))))
            assert entries[0].name == "ab" and entries[0].is_file
            assert entries[1].name == "bb" and entries[1].is_file
            assert entries[2].name == "subdir" and entries[2].is_dir


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
@pytest.mark.parametrize("parallel", [False, True])
def test_rmtree(ctx, parallel):
    contents = b"meow!"
    with ctx() as path:
        root = bf.dirname(path)
        destroy_path = bf.join(root, "destroy")
        bf.makedirs(destroy_path)
        save_path = bf.join(root, "save")
        bf.makedirs(save_path)

        # implicit dir
        if not "://" in path:
            bf.makedirs(bf.join(destroy_path, "adir"))
        with bf.BlobFile(bf.join(destroy_path, "adir/b"), "wb") as w:
            w.write(contents)

        # explicit dir
        bf.makedirs(bf.join(destroy_path, "bdir"))
        with bf.BlobFile(bf.join(destroy_path, "bdir/b"), "wb") as w:
            w.write(contents)

        bf.makedirs(bf.join(save_path, "somedir"))
        with bf.BlobFile(bf.join(save_path, "somefile"), "wb") as w:
            w.write(contents)

        def assert_listing_equal(path, desired):
            actual = list(bf.walk(path))
            # ordering of os walk is weird, only compare sorted order
            assert sorted(actual) == sorted(desired), f"{actual} != {desired}"

        assert_listing_equal(
            root,
            [
                (root, ["destroy", "save"], []),
                (destroy_path, ["adir", "bdir"], []),
                (bf.join(destroy_path, "adir"), [], ["b"]),
                (bf.join(destroy_path, "bdir"), [], ["b"]),
                (save_path, ["somedir"], ["somefile"]),
                (bf.join(save_path, "somedir"), [], []),
            ],
        )

        bf.rmtree(destroy_path, parallel=parallel)

        assert_listing_equal(
            root,
            [
                (root, ["save"], []),
                (save_path, ["somedir"], ["somefile"]),
                (bf.join(save_path, "somedir"), [], []),
            ],
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_copy(parallel):
    for contents in [b"", b"meow!", b"meow!" * (2 * 2 ** 20)]:
        with _get_temp_local_path() as local_path1, _get_temp_local_path() as local_path2, _get_temp_local_path() as local_path3, _get_temp_gcs_path() as gcs_path1, _get_temp_gcs_path() as gcs_path2, _get_temp_as_path() as as_path1, _get_temp_as_path() as as_path2, _get_temp_as_path(
            account=AS_TEST_ACCOUNT2, container=AS_TEST_CONTAINER2
        ) as as_path3, _get_temp_as_path() as as_path4:
            with pytest.raises(FileNotFoundError):
                bf.copy(gcs_path1, gcs_path2, parallel=parallel)
            with pytest.raises(FileNotFoundError):
                bf.copy(as_path1, as_path2, parallel=parallel)

            _write_contents(local_path1, contents)

            testcases = [
                (local_path1, local_path2),
                (local_path1, gcs_path1),
                (gcs_path1, gcs_path2),
                (gcs_path2, as_path1),
                (as_path1, as_path2),
                (as_path2, as_path3),
                (as_path3, local_path3),
                (local_path3, as_path4),
            ]

            for src, dst in testcases:
                h = bf.copy(src, dst, return_md5=True, parallel=parallel)
                assert h == hashlib.md5(contents).hexdigest()
                assert _read_contents(dst) == contents
                with pytest.raises(FileExistsError):
                    bf.copy(src, dst, parallel=parallel)
                bf.copy(src, dst, overwrite=True, parallel=parallel)
                assert _read_contents(dst) == contents


@pytest.mark.parametrize("parallel", [False, True])
def test_copy_invalid(parallel):
    for contents in [b"", b"meow!", b"meow!" * (2 * 2 ** 20)]:
        with _get_temp_local_path() as local_path, _get_temp_as_path() as as_path1, _get_temp_as_path() as as_path2:
            invalid_container_as_path = _convert_az_to_https(
                bf.join(AZURE_INVALID_CONTAINER, "file.bin")
            )
            invalid_account_as_path = _convert_az_to_https(
                bf.join(AZURE_INVALID_CONTAINER_NO_ACCOUNT, "file.bin")
            )

            _write_contents(local_path, contents)
            bf.copy(local_path, as_path1, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_container_as_path):
                bf.copy(local_path, invalid_container_as_path, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_account_as_path):
                bf.copy(local_path, invalid_account_as_path, parallel=parallel)

            with pytest.raises(FileNotFoundError, match=invalid_container_as_path):
                bf.copy(as_path1, invalid_container_as_path, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_account_as_path):
                bf.copy(as_path1, invalid_account_as_path, parallel=parallel)

            with pytest.raises(FileNotFoundError, match=invalid_container_as_path):
                bf.copy(invalid_container_as_path, as_path2, parallel=parallel)

            # this fails with a RequestFailure to get the sas token
            # with pytest.raises(FileNotFoundError, match=invalid_account_as_path):
            # bf.copy(invalid_account_as_path, as_path2, parallel=parallel)


# the tests already take awhile and this adds like a minute
@pytest.mark.slow
@pytest.mark.parametrize("ctx", [_get_temp_gcs_path, _get_temp_as_path])
def test_parallel_copy_large_file(ctx):
    contents = b"meow!" * common.PARALLEL_COPY_MINIMUM_PART_SIZE + b"meow???"
    with ctx() as remote_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "test.txt")
            with open(local_path, "wb") as f:
                f.write(contents)
            bf.copy(local_path, remote_path, parallel=True)

        assert _read_contents(remote_path) == contents

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "test.txt")
            bf.copy(remote_path, local_path, parallel=True)
            assert _read_contents(local_path) == contents


def test_copy_azure_public():
    with _get_temp_as_path() as dst:
        bf.copy(AZURE_PUBLIC_URL, dst)
        assert _read_contents(dst)[:4] == AZURE_PUBLIC_URL_HEADER


def test_gcs_public():
    filepath = "gs://tfds-data/datasets/mnist/3.0.1/dataset_info.json"
    assert bf.exists(filepath)
    assert len(bf.BlobFile(filepath, "rb").read()) > 0


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_exists(ctx):
    contents = b"meow!"
    with ctx() as path:
        assert not bf.exists(path)
        _write_contents(path, contents)
        assert bf.exists(path)


def test_concurrent_write_gcs():
    with _get_temp_gcs_path() as path:
        outer_contents = b"miso" * (2 ** 20 + 1)
        inner_contents = b"momo" * (2 ** 20 + 1)
        with bf.BlobFile(path, "wb", streaming=True) as f:
            f.write(outer_contents)
            with bf.BlobFile(path, "wb", streaming=True) as f:
                f.write(inner_contents)

        # the outer write will finish last and overwrite the inner one
        # the last writer to finish wins with this setup
        with bf.BlobFile(path, "rb") as f:
            assert f.read() == outer_contents


def test_concurrent_write_as():
    with _get_temp_as_path() as path:
        b = bf.create_context(azure_write_chunk_size=2 ** 20)
        outer_contents = b"miso" * (2 ** 20 + 1)
        inner_contents = b"momo" * (2 ** 20 + 1)
        # the inner write will invalidate the outer one, the last writer
        # to start wins with this setup
        with pytest.raises(bf.ConcurrentWriteFailure):
            with b.BlobFile(path, "wb", streaming=True) as f:
                f.write(outer_contents)
                with b.BlobFile(path, "wb", streaming=True) as f:
                    f.write(inner_contents)

        # the outer write will finish last and overwrite the inner one
        with b.BlobFile(path, "rb") as f:
            assert f.read() == inner_contents


@contextlib.contextmanager
def environ_context():
    env = os.environ.copy()
    yield
    os.environ = env


def test_more_exists():
    testcases = [
        (AZURE_INVALID_CONTAINER, False),
        (AZURE_INVALID_CONTAINER + "/", False),
        (AZURE_INVALID_CONTAINER + "//", False),
        (AZURE_INVALID_CONTAINER + "/invalid.file", False),
        (GCS_INVALID_BUCKET, False),
        (GCS_INVALID_BUCKET + "/", False),
        (GCS_INVALID_BUCKET + "//", False),
        (GCS_INVALID_BUCKET + "/invalid.file", False),
        (AZURE_INVALID_CONTAINER_NO_ACCOUNT, False),
        (AZURE_INVALID_CONTAINER_NO_ACCOUNT + "/", False),
        (AZURE_INVALID_CONTAINER_NO_ACCOUNT + "//", False),
        (AZURE_INVALID_CONTAINER_NO_ACCOUNT + "/invalid.file", False),
        (AZURE_VALID_CONTAINER, True),
        (AZURE_VALID_CONTAINER + "/", True),
        (AZURE_VALID_CONTAINER + "//", False),
        (AZURE_VALID_CONTAINER + "/invalid.file", False),
        (GCS_VALID_BUCKET, True),
        (GCS_VALID_BUCKET + "/", True),
        (GCS_VALID_BUCKET + "//", False),
        (GCS_VALID_BUCKET + "/invalid.file", False),
        (f"/does-not-exist", False),
        (f"/", True),
    ]
    for path, should_exist in testcases:
        assert bf.exists(path) == should_exist


@pytest.mark.parametrize(
    "base_path",
    [AZURE_INVALID_CONTAINER_NO_ACCOUNT, AZURE_INVALID_CONTAINER, GCS_INVALID_BUCKET],
)
def test_invalid_paths(base_path):
    for suffix in ["", "/", "//", "/invalid.file", "/invalid/dir/"]:
        path = base_path + suffix
        print(path)
        if path.endswith("/"):
            expected_error = IsADirectoryError
        else:
            expected_error = FileNotFoundError
        list(bf.glob(path))
        if suffix == "":
            for pattern in ["*", "**"]:
                try:
                    list(bf.glob(path + pattern))
                except bf.Error as e:
                    assert "Wildcards cannot be used" in e.message
        else:
            for pattern in ["*", "**"]:
                list(bf.glob(path + pattern))
        with pytest.raises(FileNotFoundError):
            list(bf.listdir(path))
        assert not bf.exists(path)
        assert not bf.isdir(path)
        with pytest.raises(expected_error):
            bf.remove(path)
        if suffix in ("", "/"):
            try:
                bf.rmdir(path)
            except bf.Error as e:
                assert "Cannot delete bucket" in e.message
        else:
            bf.rmdir(path)
        with pytest.raises(NotADirectoryError):
            bf.rmtree(path)
        with pytest.raises(FileNotFoundError):
            bf.stat(path)

        if base_path == AZURE_INVALID_CONTAINER_NO_ACCOUNT:
            with pytest.raises(bf.Error):
                bf.get_url(path)
        else:
            bf.get_url(path)

        with pytest.raises(FileNotFoundError):
            bf.md5(path)
        with pytest.raises(bf.Error):
            bf.makedirs(path)
        list(bf.walk(path))
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "test.txt")
            with pytest.raises(expected_error):
                bf.copy(path, local_path)
            with open(local_path, "w") as f:
                f.write("meow")
            with pytest.raises(expected_error):
                bf.copy(local_path, path)
        for streaming in [False, True]:
            with pytest.raises(expected_error):
                with bf.BlobFile(path, "rb", streaming=streaming) as f:
                    f.read()
            with pytest.raises(expected_error):
                with bf.BlobFile(path, "wb", streaming=streaming) as f:
                    f.write(b"meow")


@pytest.mark.parametrize("buffer_size", [1, 100])
@pytest.mark.parametrize("ctx", [_get_temp_gcs_path, _get_temp_as_path])
def test_read_stats(buffer_size, ctx):
    with ctx() as path:
        contents = b"meow!"

        with bf.BlobFile(path, "wb") as w:
            w.write(contents)

        with bf.BlobFile(path, "rb", buffer_size=buffer_size) as r:
            r.read(1)

        if buffer_size == 1:
            assert r.raw.bytes_read == 1  # type: ignore
        else:
            assert r.raw.bytes_read == len(contents)  # type: ignore

        with bf.BlobFile(path, "rb", buffer_size=buffer_size) as r:
            r.read(1)
            r.seek(4)
            r.read(1)
            r.seek(1000000)
            assert r.read(1) == b""

        if buffer_size == 1:
            assert r.raw.requests == 2  # type: ignore
            assert r.raw.bytes_read == 2  # type: ignore
        else:
            assert r.raw.requests == 1  # type: ignore
            assert r.raw.bytes_read == len(contents)  # type: ignore


@pytest.mark.parametrize("ctx", [_get_temp_gcs_path, _get_temp_as_path])
def test_cache_dir(ctx):
    cache_dir = tempfile.mkdtemp()
    contents = b"meow!"
    alternative_contents = b"purr!"
    with ctx() as path:
        with bf.BlobFile(path, mode="wb") as f:
            f.write(contents)
        with bf.BlobFile(path, mode="rb", streaming=False, cache_dir=cache_dir) as f:
            assert f.read() == contents
        content_hash = hashlib.md5(contents).hexdigest()
        cache_path = bf.join(cache_dir, content_hash, bf.basename(path))
        with open(cache_path, "rb") as f:
            assert f.read() == contents
        # alter the cached file to make sure we are not re-reading the remote file
        with open(cache_path, "wb") as f:
            f.write(alternative_contents)
        with bf.BlobFile(path, mode="rb", streaming=False, cache_dir=cache_dir) as f:
            assert f.read() == alternative_contents
        with bf.BlobFile(
            f"https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat_no_md5.png",
            mode="rb",
            streaming=False,
            cache_dir=cache_dir,
        ) as f:
            assert len(f.read()) > 0


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
@pytest.mark.parametrize("use_random", [False, True])
def test_change_file_size(ctx, use_random):
    chunk_size = 8 * 2 ** 20
    long_contents = b"\x00" * chunk_size * 3
    short_contents = b"\xFF" * chunk_size * 2
    if use_random:
        long_contents = os.urandom(len(long_contents))
        short_contents = os.urandom(len(short_contents))
    with ctx() as path:
        # make file shorter
        with bf.BlobFile(path, "wb") as f:
            f.write(long_contents)
        with bf.BlobFile(path, "rb") as f:
            read_contents = f.read(chunk_size)
            with bf.BlobFile(path, "wb") as f2:
                f2.write(short_contents)
            # close underlying connection
            f.raw._f = None  # type: ignore
            read_contents += f.read()
            assert len(f.read()) == 0
            assert (
                read_contents
                == long_contents[:chunk_size] + short_contents[chunk_size:]
            )

        # make file longer
        with bf.BlobFile(path, "wb") as f:
            f.write(short_contents)
        with bf.BlobFile(path, "rb") as f:
            read_contents = f.read(chunk_size)
            with bf.BlobFile(path, "wb") as f2:
                f2.write(long_contents)
            # close underlying connection
            f.raw._f = None  # type: ignore
            read_contents += f.read()
            assert len(f.read()) == 0
            expected = (
                short_contents[:chunk_size] + long_contents[chunk_size : chunk_size * 2]
            )
            # local files behave differently and read the new contents until the
            # end of the new file size
            if not path.startswith("gs://") and not path.startswith("https://"):
                expected = short_contents[:chunk_size] + long_contents[chunk_size:]
            assert read_contents == expected


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_overwrite_while_reading(ctx):
    chunk_size = 8 * 2 ** 20
    contents = b"\x00" * chunk_size * 2
    alternative_contents = b"\xFF" * chunk_size * 4
    with ctx() as path:
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)
        with bf.BlobFile(path, "rb") as f:
            read_contents = f.read(chunk_size)
            with bf.BlobFile(path, "wb") as f2:
                f2.write(alternative_contents)
            # close underlying connection
            f.raw._f = None  # type: ignore
            read_contents += f.read(chunk_size)
            assert (
                read_contents
                == contents[:chunk_size]
                + alternative_contents[chunk_size : chunk_size * 2]
            )


def test_create_local_intermediate_dirs():
    contents = b"meow"
    with _get_temp_local_path() as path:
        dirpath = bf.dirname(path)
        with chdir(dirpath):
            for filepath in [
                bf.join(dirpath, "dirname", "file.name"),
                bf.join("..", bf.basename(dirpath), "file.name"),
                "./file.name",
                "file.name",
            ]:
                with bf.BlobFile(filepath, "wb") as f:
                    f.write(contents)


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_more_read_write(binary, streaming, ctx):
    rng = np.random.RandomState(0)

    with ctx() as path:
        if binary:
            read_mode = "rb"
            write_mode = "wb"
        else:
            read_mode = "r"
            write_mode = "w"

        with bf.BlobFile(path, write_mode, streaming=streaming) as w:
            pass

        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert len(r.read()) == 0

        contents = b"meow!"
        if not binary:
            contents = contents.decode("utf8")

        with bf.BlobFile(path, write_mode, streaming=streaming) as w:
            w.write(contents)

        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert r.read(1) == contents[:1]
            assert r.read() == contents[1:]
            assert len(r.read()) == 0

        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            for i in range(len(contents)):
                assert r.read(1) == contents[i : i + 1]
            assert len(r.read()) == 0
            assert len(r.read()) == 0

        contents = b"meow!\n\nmew!\n"
        lines = [b"meow!\n", b"\n", b"mew!\n"]
        if not binary:
            contents = contents.decode("utf8")
            lines = [line.decode("utf8") for line in lines]

        with bf.BlobFile(path, write_mode, streaming=streaming) as w:
            w.write(contents)

        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert r.readlines() == lines

        with bf.BlobFile(path, read_mode, streaming=streaming) as r:
            assert [line for line in r] == lines

        if binary:
            for size in [2 * 2 ** 20, 12_345_678]:
                contents = rng.randint(0, 256, size=size, dtype=np.uint8).tobytes()

                with bf.BlobFile(path, write_mode, streaming=streaming) as w:
                    w.write(contents)

                with bf.BlobFile(path, read_mode, streaming=streaming) as r:
                    size = rng.randint(0, 1_000_000)
                    buf = b""
                    while True:
                        b = r.read(size)
                        if b == b"":
                            break
                        buf += b
                    assert buf == contents
        else:
            obj = {"a": 1}

            with bf.BlobFile(path, write_mode, streaming=streaming) as w:
                json.dump(obj, w)

            with bf.BlobFile(path, read_mode, streaming=streaming) as r:
                assert json.load(r) == obj


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_video(streaming, ctx):
    rng = np.random.RandomState(0)
    shape = (256, 64, 64, 3)
    video_data = rng.randint(0, 256, size=np.prod(shape), dtype=np.uint8).reshape(shape)

    with ctx() as path:
        with bf.BlobFile(path, mode="wb", streaming=streaming) as wf:
            with imageio.get_writer(
                wf,
                format="ffmpeg",
                quality=None,
                codec="libx264rgb",
                pixelformat="bgr24",
                output_params=["-f", "mp4", "-crf", "0"],
            ) as w:
                for frame in video_data:
                    w.append_data(frame)

        with bf.BlobFile(path, mode="rb", streaming=streaming) as rf:
            with imageio.get_reader(
                rf, format="ffmpeg", input_params=["-f", "mp4"]
            ) as r:
                for idx, frame in enumerate(r):
                    assert np.array_equal(frame, video_data[idx])

        with bf.BlobFile(path, mode="rb", streaming=streaming) as rf:
            container = av.open(rf)
            stream = container.streams.video[0]
            for idx, frame in enumerate(container.decode(stream)):
                assert np.array_equal(frame.to_image(), video_data[idx])


# this is pretty slow and docker will often run out of memory
@pytest.mark.slow
@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_large_file(ctx):
    contents = b"0" * 2 ** 32
    with ctx() as path:
        with bf.BlobFile(path, "wb", streaming=True) as f:
            f.write(contents)
        with bf.BlobFile(path, "rb", streaming=True) as f:
            assert contents == f.read()


def test_composite_objects():
    with _get_temp_gcs_path() as remote_path:
        with _get_temp_local_path() as local_path:
            contents = b"0" * 2 * 2 ** 20
            with open(local_path, "wb") as f:
                f.write(contents)

            def create_composite_file():
                sp.run(
                    [
                        "gsutil",
                        "-o",
                        "GSUtil:parallel_composite_upload_threshold=1M",
                        "cp",
                        local_path,
                        remote_path,
                    ],
                    check=True,
                )

            local_md5 = hashlib.md5(contents).hexdigest()
            create_composite_file()
            assert bf.stat(remote_path).md5 is None
            assert local_md5 == bf.md5(remote_path)
            assert bf.stat(remote_path).md5 == local_md5
            assert local_md5 == bf.md5(remote_path)

            bf.remove(remote_path)
            create_composite_file()
            assert bf.stat(remote_path).md5 is None

            with tempfile.TemporaryDirectory() as tmpdir:
                with bf.BlobFile(
                    remote_path, "rb", cache_dir=tmpdir, streaming=False
                ) as f:
                    assert f.read() == contents
            assert bf.stat(remote_path).md5 == local_md5


@pytest.mark.parametrize(
    "ctx", [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path]
)
def test_md5(ctx):
    contents = b"meow!"
    meow_hash = hashlib.md5(contents).hexdigest()

    with ctx() as path:
        _write_contents(path, contents)
        assert bf.md5(path) == meow_hash
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)
        assert bf.md5(path) == meow_hash
        with bf.BlobFile(path, "wb") as f:
            f.write(contents)
        assert bf.md5(path) == meow_hash


@pytest.mark.parametrize("ctx", [_get_temp_as_path])
def test_azure_maybe_update_md5(ctx):
    contents = b"meow!"
    meow_hash = hashlib.md5(contents).hexdigest()
    alternative_contents = b"purr"
    purr_hash = hashlib.md5(alternative_contents).hexdigest()

    with ctx() as path:
        _write_contents(path, contents)
        st = azure.maybe_stat(ops.default_context._conf, path)
        assert azure.maybe_update_md5(
            ops.default_context._conf, path, st.version, meow_hash
        )
        _write_contents(path, alternative_contents)
        assert not azure.maybe_update_md5(
            ops.default_context._conf, path, st.version, meow_hash
        )
        st = azure.maybe_stat(ops.default_context._conf, path)
        assert st.md5 == purr_hash
        bf.remove(path)
        assert not azure.maybe_update_md5(
            ops.default_context._conf, path, st.version, meow_hash
        )


def test_azure_timestamp_parsing():
    timestamp = "Sun, 27 Sep 2009 18:41:57 GMT"

    def ref_parse_timestamp(text: str) -> float:
        return datetime.datetime.strptime(
            text.replace("GMT", "Z"), "%a, %d %b %Y %H:%M:%S %z"
        ).timestamp()

    assert ref_parse_timestamp(timestamp) == azure._parse_timestamp(timestamp)


def _get_http_pool_id(q):
    q.put(id(ops.default_context._conf.get_http_pool()))


def test_fork():
    q = mp.Queue()
    # this reference should keep the old http client alive in the child process
    # to ensure that a new one does not recycle the memory address
    http1 = ops.default_context._conf.get_http_pool()
    parent1 = id(http1)
    p = mp.Process(target=_get_http_pool_id, args=(q,))
    p.start()
    p.join()
    http2 = ops.default_context._conf.get_http_pool()
    parent2 = id(http2)

    child = q.get()
    assert parent1 == parent2
    assert child != parent1


def test_azure_public_container():
    for error, path in [
        (
            None,
            f"https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat.png",
        ),
        (
            bf.Error,
            f"https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/private/test_cat.png",
        ),  # an account that exists but with a non-public container
        (
            FileNotFoundError,
            f"https://{AS_INVALID_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat.png",
        ),  # account that does not exist
    ]:
        ctx = contextlib.nullcontext()
        if error is not None:
            ctx = pytest.raises(error)
        with ctx:
            with bf.BlobFile(path, "rb") as f:
                contents = f.read()
                assert contents.startswith(AZURE_PUBLIC_URL_HEADER)


def test_scandir_error():
    for error, path in [
        (None, AZURE_VALID_CONTAINER),
        (FileNotFoundError, AZURE_INVALID_CONTAINER),
        (FileNotFoundError, AZURE_INVALID_CONTAINER_NO_ACCOUNT),
        (bf.Error, f"https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/private"),
    ]:
        ctx = contextlib.nullcontext()
        if error is not None:
            ctx = pytest.raises(error)
        with ctx:
            print(path)
            list(bf.scandir(path))


def test_windowed_file():
    with _get_temp_local_path() as path:
        with open(path, "wb") as f:
            f.write(b"meow")

        with open(path, "rb") as f:
            f2 = common.WindowedFile(f, start=1, end=3)
            assert f2.read() == b"eo"

            f2.seek(0)
            assert f2.read(1) + f2.read(1) + f2.read(1) == b"eo"

            with pytest.raises(AssertionError):
                f2.seek(-1)

            with pytest.raises(AssertionError):
                f2.seek(2)


class Deadline:
    def __init__(self):
        self._deadline = None

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def get_deadline(self) -> float:
        return self._deadline


@pytest.mark.parametrize("ctx", [_get_temp_gcs_path, _get_temp_as_path])
def test_deadline(ctx):
    contents = b"meow"
    with ctx() as path:
        _write_contents(path, contents)
        deadline = Deadline()
        bf_ctx = bf.create_context(get_deadline=deadline.get_deadline)
        deadline.set_deadline(time.time() + 5)
        with bf_ctx.BlobFile(path, "rb") as f:
            f.read()
        time.sleep(5)
        with pytest.raises(bf.DeadlineExceeded):
            with bf_ctx.BlobFile(path, "rb") as f:
                f.read()
        deadline.set_deadline(None)
        with bf_ctx.BlobFile(path, "rb") as f:
            f.read()


def test_pickle_config():
    ctx = ops.create_context()
    c = ctx._conf
    pickle.dumps(c)
    c.get_http_pool()
    c2 = pickle.loads(pickle.dumps(c))
    c2.get_http_pool()


@pytest.mark.parametrize("ctx", [_get_temp_gcs_path, _get_temp_as_path])
def test_read_with_size(ctx):
    contents = b"meow!\npurr\n"
    with ctx() as path:
        path = bf.join(path, "a folder", "a.file")
        bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, "wb") as w:
            w.write(contents)
        with bf.BlobFile(path, "rb", file_size=1) as r:
            assert r.read() == contents[:1]
            assert r.tell() == 1
