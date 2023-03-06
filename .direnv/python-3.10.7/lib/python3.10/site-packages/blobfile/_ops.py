# https://mypy.readthedocs.io/en/stable/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
from __future__ import annotations

import concurrent.futures
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    Callable,
    Iterator,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    overload,
)

import urllib3

if TYPE_CHECKING:
    # Literal is only in the stdlib in Python 3.8+
    # this works without having a runtime installation of typing_extensions because
    # a) we postponed evaluation of type annotations with PEP 563,
    # b) we don't use Literal as a base class or for casting,
    # c) type checkers always know what typing_extensions is
    from typing_extensions import Literal

from blobfile._common import DirEntry, Stat
from blobfile._context import (
    DEFAULT_AZURE_WRITE_CHUNK_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_CONNECTION_POOL_MAX_SIZE,
    DEFAULT_GOOGLE_WRITE_CHUNK_SIZE,
    DEFAULT_MAX_CONNECTION_POOL_COUNT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_RETRY_COMMON_LOG_THRESHOLD,
    DEFAULT_RETRY_LOG_THRESHOLD,
    create_context,
    default_log_fn,
)

default_context = create_context()


def configure(
    *,
    log_callback: Callable[[str], None] = default_log_fn,
    connection_pool_max_size: int = DEFAULT_CONNECTION_POOL_MAX_SIZE,
    max_connection_pool_count: int = DEFAULT_MAX_CONNECTION_POOL_COUNT,
    # https://docs.microsoft.com/en-us/rest/api/storageservices/understanding-block-blobs--append-blobs--and-page-blobs#about-block-blobs
    # the chunk size determines the maximum size of an individual blob
    azure_write_chunk_size: int = DEFAULT_AZURE_WRITE_CHUNK_SIZE,
    google_write_chunk_size: int = DEFAULT_GOOGLE_WRITE_CHUNK_SIZE,
    retry_log_threshold: int = DEFAULT_RETRY_LOG_THRESHOLD,
    retry_common_log_threshold: int = DEFAULT_RETRY_COMMON_LOG_THRESHOLD,
    retry_limit: Optional[int] = None,
    connect_timeout: Optional[int] = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: Optional[int] = DEFAULT_READ_TIMEOUT,
    output_az_paths: bool = True,
    use_azure_storage_account_key_fallback: bool = False,
    get_http_pool: Optional[Callable[[], urllib3.PoolManager]] = None,
    use_streaming_read: bool = False,
    default_buffer_size: int = DEFAULT_BUFFER_SIZE,
    save_access_token_to_disk: bool = True,
    multiprocessing_start_method: str = "spawn",
) -> None:
    """
    log_callback: a log callback function `log(msg: string)` to use instead of printing to stdout
    connection_pool_max_size: the max size for each per-host connection pool
    max_connection_pool_count: the maximum count of per-host connection pools
    azure_write_chunk_size: the size of blocks to write to Azure Storage blobs, can be set to a maximum of 100MB
    google_write_chunk_size: the size of blocks to write to Google Cloud Storage blobs in bytes, this only determines the unit of request retries
    retry_log_threshold: set a retry count threshold above which to log failures to the log callback function
    connect_timeout: the maximum amount of time (in seconds) to wait for a connection attempt to a server to succeed, set to None to wait forever
    read_timeout: the maximum amount of time (in seconds) to wait between consecutive read operations for a response from the server, set to None to wait forever
    output_az_paths: output `az://` paths instead of using the `https://` for azure
    use_azure_storage_account_key_fallback: fallback to storage account keys for azure containers, having this enabled requires listing your subscriptions and may run into 429 errors if you hit the low azure quotas for subscription listing
    get_http_pool: a function that returns a `urllib3.PoolManager` to be used for requests
    use_streaming_read: if set to `True`, use a single read per file instead of reading a chunk at a time (not recommended for azure)
    default_buffer_size: the default buffer size to use for reading files (and writing local files)
    save_access_token_to_disk: set to `True` to save access tokens to disk so that other processes can read the access tokens to avoid the small amount of time it usually takes to get a token (if the token is still valid).
    multiprocessing_start_method: the start method to use when creating processes for parallel work
    """
    global default_context
    default_context = create_context(
        log_callback=log_callback,
        connection_pool_max_size=connection_pool_max_size,
        max_connection_pool_count=max_connection_pool_count,
        azure_write_chunk_size=azure_write_chunk_size,
        retry_log_threshold=retry_log_threshold,
        retry_common_log_threshold=retry_common_log_threshold,
        retry_limit=retry_limit,
        google_write_chunk_size=google_write_chunk_size,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        output_az_paths=output_az_paths,
        use_azure_storage_account_key_fallback=use_azure_storage_account_key_fallback,
        get_http_pool=get_http_pool,
        use_streaming_read=use_streaming_read,
        default_buffer_size=default_buffer_size,
        save_access_token_to_disk=save_access_token_to_disk,
        multiprocessing_start_method=multiprocessing_start_method,
    )


def copy(
    src: str,
    dst: str,
    overwrite: bool = False,
    parallel: bool = False,
    parallel_executor: Optional[concurrent.futures.Executor] = None,
    return_md5: bool = False,
) -> Optional[str]:
    """
    Copy a file from one path to another

    If both paths are on the same blob storage, this will perform a remote copy operation without downloading
    the contents locally.

    If `overwrite` is `False` (the default), an exception will be raised if the destination
    path exists.

    If `parallel` is `True`, use multiple processes to dowload or upload the file.  For this to work, one path must be on blob storage and the other path must be local.  This can be faster on cloud machines but is not in general guaranteed to be faster than using serial copy.  The default is `False`.

    If `parallel_executor` is set to a `concurrent.futures.Executor` and `parallel` is set to `True`, the provided executor will be used instead of creating a new one for each call to `copy()`.

    If `return_md5` is set to `True`, an md5 will be calculated during the copy and returned if available,
    or else None will be returned.
    """
    return default_context.copy(
        src=src,
        dst=dst,
        overwrite=overwrite,
        parallel=parallel,
        parallel_executor=parallel_executor,
        return_md5=return_md5,
    )


def exists(path: str) -> bool:
    """
    Return true if that path exists (either as a file or a directory)
    """
    return default_context.exists(path=path)


def basename(path: str) -> str:
    """
    Get the filename component of the path

    For GCS, this is the part after the bucket
    """
    return default_context.basename(path=path)


def glob(pattern: str, parallel: bool = False) -> Iterator[str]:
    """
    Find files and directories matching a pattern. Supports * and **

    For local paths, this function uses glob.glob() which has special handling for * and **
    that is not quite the same as remote paths.  See https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames#different-behavior-for-dot-files-in-local-file-system_1 for more information.

    Globs can have confusing performance, see https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames#efficiency-consideration:-using-wildcards-over-many-objects for more information.

    You can set `parallel=True` to use multiple processes to perform the glob.  It's likely
    that the results will no longer be in order.
    """
    return default_context.glob(pattern=pattern, parallel=parallel)


def scanglob(
    pattern: str, parallel: bool = False, shard_prefix_length: int = 0
) -> Iterator[DirEntry]:
    """
    Same as `glob`, but returns `DirEntry` objects instead of strings
    """
    return default_context.scanglob(
        pattern=pattern, parallel=parallel, shard_prefix_length=shard_prefix_length
    )


def isdir(path: str) -> bool:
    """
    Return true if a path is an existing directory
    """
    return default_context.isdir(path=path)


def listdir(path: str, shard_prefix_length: int = 0) -> Iterator[str]:
    """
    Returns an iterator of the contents of the directory at `path`

    If your filenames are uniformly distributed (like hashes) then you can use `shard_prefix_length`
    to query them more quickly.  `shard_prefix_length` will do multiple queries in parallel,
    querying each possible prefix independently.

    Using `shard_prefix_length` will only consider prefixes that are not unusual characters
    (mostly these are ascii values < 0x20) some of these could technically show up in a path.
    """
    return default_context.listdir(path=path, shard_prefix_length=shard_prefix_length)


def scandir(path: str, shard_prefix_length: int = 0) -> Iterator[DirEntry]:
    """
    Same as `listdir`, but returns `DirEntry` objects instead of strings
    """
    return default_context.scandir(path=path, shard_prefix_length=shard_prefix_length)


def makedirs(path: str) -> None:
    """
    Make any directories necessary to ensure that path is a directory
    """
    return default_context.makedirs(path=path)


def remove(path: str) -> None:
    """
    Remove a file at the given path
    """
    return default_context.remove(path=path)


def rmdir(path: str) -> None:
    """
    Remove an empty directory at the given path
    """
    return default_context.rmdir(path=path)


def stat(path: str) -> Stat:
    """
    Stat a file or object representing a directory, returns a Stat object
    """
    return default_context.stat(path=path)


def set_mtime(path: str, mtime: float, version: Optional[str] = None) -> bool:
    """
    Set the mtime for a path, returns True on success

    A version can be specified (as returned by `stat()`) to only update the mtime if the
    version matches
    """
    return default_context.set_mtime(path=path, mtime=mtime, version=version)


def rmtree(
    path: str,
    parallel: bool = False,
    parallel_executor: Optional[concurrent.futures.Executor] = None,
) -> None:
    """
    Delete a directory tree
    """
    return default_context.rmtree(
        path=path, parallel=parallel, parallel_executor=parallel_executor
    )


def walk(
    top: str, topdown: bool = True, onerror: Optional[Callable[[OSError], None]] = None
) -> Iterator[Tuple[str, Sequence[str], Sequence[str]]]:
    """
    Walk a directory tree in a similar manner to os.walk
    """
    return default_context.walk(top=top, topdown=topdown, onerror=onerror)


def dirname(path: str) -> str:
    """
    Get the directory name of the path

    On GCS, the root directory is gs://<bucket name>/
    On Azure Storage, the root directory is https://<account>.blob.core.windows.net/<container>/
    """
    return default_context.dirname(path=path)


def join(a: str, *args: str) -> str:
    """
    Join file paths, if a path is an absolute path, it will replace the entire path component of previous paths
    """
    return default_context.join(a, *args)


def get_url(path: str) -> Tuple[str, Optional[float]]:
    """
    Get a URL for the given path that a browser could open
    """
    return default_context.get_url(path=path)


def md5(path: str) -> str:
    """
    Get the MD5 hash for a file in hexdigest format.

    For GCS this will look up the MD5 in the blob's metadata, unless it's a composite object, in which case
    it must be calculated by downloading the file.
    For Azure this can look up the MD5 if it's available, otherwise it must calculate it.
    For local paths, this must always calculate the MD5.
    """
    return default_context.md5(path=path)


@overload
def BlobFile(
    path: str,
    mode: Literal["rb", "wb", "ab"],
    streaming: Optional[bool] = ...,
    buffer_size: int = ...,
    cache_dir: Optional[str] = ...,
    file_size: Optional[int] = None,
) -> BinaryIO:
    ...


@overload
def BlobFile(
    path: str,
    mode: Literal["r", "w", "a"] = ...,
    streaming: Optional[bool] = ...,
    buffer_size: int = ...,
    cache_dir: Optional[str] = ...,
    file_size: Optional[int] = None,
) -> TextIO:
    ...


def BlobFile(
    path: str,
    mode: Literal["r", "rb", "w", "wb", "a", "ab"] = "r",
    streaming: Optional[bool] = None,
    buffer_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    file_size: Optional[int] = None,
):
    """
    Open a local or remote file for reading or writing

    Args:
        path local or remote path
        mode: one of "r", "rb", "w", "wb", "a", "ab" indicating the mode to open the file in
        streaming: the default for `streaming` is `True` when `mode` is in `"r", "rb"` and `False` when `mode` is in `"w", "wb", "a", "ab"`.
            * `streaming=True`:
                * Reading is done without downloading the entire remote file.
                * Writing is done to the remote file directly, but only in chunks of a few MB in size.  `flush()` will not cause an early write.
                * Appending is not implemented.
            * `streaming=False`:
                * Reading is done by downloading the remote file to a local file during the constructor.
                * Writing is done by uploading the file on `close()` or during destruction.
                * Appending is done by downloading the file during construction and uploading on `close()` or during destruction.
        buffer_size: number of bytes to buffer, this can potentially make reading more efficient.
        cache_dir: a directory in which to cache files for reading, only valid if `streaming=False` and `mode` is in `"r", "rb"`.   You are reponsible for cleaning up the cache directory.

    Returns:
        A file-like object
    """
    return default_context.BlobFile(
        path=path,
        mode=mode,
        streaming=streaming,
        buffer_size=buffer_size,
        cache_dir=cache_dir,
        file_size=file_size,
    )
