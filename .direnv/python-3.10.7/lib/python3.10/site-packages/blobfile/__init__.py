import os

# all the modules are under _internal to indicate that they are not exported
# the A as A syntax seems to cause pycharm to list these
# symbols as coming from blobfile instead of blobfile._internal
from blobfile._ops import (
    copy as copy,
    exists as exists,
    glob as glob,
    scanglob as scanglob,
    isdir as isdir,
    listdir as listdir,
    scandir as scandir,
    makedirs as makedirs,
    remove as remove,
    rmdir as rmdir,
    rmtree as rmtree,
    stat as stat,
    walk as walk,
    basename as basename,
    dirname as dirname,
    join as join,
    get_url as get_url,
    md5 as md5,
    set_mtime as set_mtime,
    configure as configure,
    BlobFile as BlobFile,
)
from blobfile._context import Context as Context, create_context as create_context
from blobfile._common import (
    Request as Request,
    Error as Error,
    RequestFailure as RequestFailure,
    RestartableStreamingWriteFailure as RestartableStreamingWriteFailure,
    ConcurrentWriteFailure as ConcurrentWriteFailure,
    DeadlineExceeded as DeadlineExceeded,
    Stat as Stat,
    DirEntry as DirEntry,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, "VERSION")) as _version_file:
    __version__ = _version_file.read().strip()

__all__ = [
    "copy",
    "exists",
    "glob",
    "scanglob",
    "isdir",
    "listdir",
    "scandir",
    "makedirs",
    "remove",
    "rmdir",
    "rmtree",
    "stat",
    "walk",
    "basename",
    "dirname",
    "join",
    "get_url",
    "md5",
    "set_mtime",
    "configure",
    "BlobFile",
    "Request",
    "Error",
    "RequestFailure",
    "RestartableStreamingWriteFailure",
    "ConcurrentWriteFailure",
    "DeadlineExceeded",
    "Stat",
    "DirEntry",
    "Context",
    "create_context",
]
