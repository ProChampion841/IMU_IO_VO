"""Loading a checkpoint written on the other operating system.

A run trained on Ubuntu and evaluated on Windows - which is this project's
normal split, training on the servers and reading the numbers on a laptop -
fails at ``torch.load`` before a single tensor is read::

    pathlib._abc.UnsupportedOperation:
        cannot instantiate 'PosixPath' on your system

The cause is not the weights. The argument record saved beside them holds
``pathlib.Path`` objects - ``--dataset``, ``--run-dir``, ``--calibration`` -
and a ``Path`` pickles as its concrete flavour. ``PosixPath.__new__`` refuses
to run on Windows, and the unpickler calls it while rebuilding the arguments.

Those paths are provenance: they record where the run was trained, and nothing
downstream reopens them - the dataset to read is named on the command line.
So binding the foreign flavour to the native one for the duration of the load
costs nothing real and makes checkpoints portable in both directions.
"""

from __future__ import annotations

import os
import pathlib
import sys
from contextlib import contextmanager
from typing import Any

import torch

#: Both the public module and, on 3.13+, the private one the classes actually
#: live in. Whichever module the pickle names has to be patched, and a pickle
#: written by another Python version may name either.
_PATHLIB_MODULES = ("pathlib", "pathlib._local")

#: Pure* included because a checkpoint may hold either.
_FLAVOURS = (("PosixPath", "WindowsPath"), ("PurePosixPath", "PureWindowsPath"))


@contextmanager
def portable_paths():
    """Bind the foreign path flavour to the native one while unpickling."""

    windows = os.name == "nt"
    patched = []
    for name in _PATHLIB_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for posix_name, windows_name in _FLAVOURS:
            foreign = posix_name if windows else windows_name
            native = windows_name if windows else posix_name
            replacement = getattr(module, native, None)
            if replacement is None or not hasattr(module, foreign):
                continue
            patched.append((module, foreign, getattr(module, foreign)))
            setattr(module, foreign, replacement)
    try:
        yield
    finally:
        for module, name, original in reversed(patched):
            setattr(module, name, original)


def load_checkpoint(path: Any, map_location: Any = "cpu", **kwargs: Any) -> Any:
    """``torch.load`` that survives a checkpoint written on the other OS.

    The native load is tried first and used unchanged when it works, so a
    same-platform checkpoint takes no different path through this than it did
    before. Only the path-flavour failure falls back.
    """

    kwargs.setdefault("weights_only", False)
    try:
        return torch.load(path, map_location=map_location, **kwargs)
    except NotImplementedError:
        # Python <= 3.12 raises NotImplementedError from Path.__new__.
        pass
    except Exception as error:  # pragma: no cover - narrowed by the check below
        # 3.13 raises pathlib UnsupportedOperation, which subclasses
        # NotImplementedError there but is not importable by that name on
        # older versions, so match on what the message says instead.
        if "cannot instantiate" not in str(error):
            raise
    _ = pathlib  # the module must be imported for the patch below to find it
    with portable_paths():
        return torch.load(path, map_location=map_location, **kwargs)


__all__ = ["load_checkpoint", "portable_paths"]
