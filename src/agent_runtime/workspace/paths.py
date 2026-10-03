"""``resolve(root, user_path)``.

Before it touches the disk it rejects: an empty path, a NUL character, a drive letter, a UNC prefix,
a device-namespace prefix (``\\\\?\\``, ``\\\\.\\``), an absolute path, a leading ``~``, a component
that is a reserved device name (``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``-``COM9``, ``LPT1``-``LPT9``,
with or without an extension), a component with a colon (alternate data streams), and a component
that ends with a dot or a space. It then joins the path to the root, resolves symbolic links and
junctions (``os.path.realpath``; for a path that does not exist yet the nearest existing ancestor is
resolved and the rest appended), and requires the result to lie under the real path of the root
(case-insensitively on Windows). This is a check at the time of the call: a link that another
process swaps in between the check and the use defeats it.
"""

from __future__ import annotations

import os
import re

RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_DRIVE = re.compile(r"^[A-Za-z]:")


class PathRejected(ValueError):
    def __init__(self, reason: str, path: str) -> None:
        super().__init__(f"{reason}: {path!r}")
        self.reason = reason


def check_syntax(user_path: str) -> list[str]:
    """Return the components of a relative path, or raise ``PathRejected`` (no disk access)."""
    if not isinstance(user_path, str) or user_path == "":
        raise PathRejected("empty", str(user_path))
    if "\x00" in user_path:
        raise PathRejected("nul_character", user_path)
    if user_path.startswith(("\\\\?\\", "\\\\.\\", "//?/", "//./")):
        raise PathRejected("device_namespace", user_path)
    if user_path.startswith(("\\\\", "//")):
        raise PathRejected("unc_path", user_path)
    if _DRIVE.match(user_path):
        raise PathRejected("drive_letter", user_path)
    if user_path.startswith(("/", "\\")):
        raise PathRejected("absolute_path", user_path)
    if user_path.startswith("~"):
        raise PathRejected("home_prefix", user_path)
    parts = [p for p in re.split(r"[\\/]", user_path)]
    for comp in parts:
        if comp in ("", ".", ".."):
            continue
        if ":" in comp:
            raise PathRejected("alternate_data_stream", user_path)
        if comp.endswith((".", " ")):
            raise PathRejected("trailing_dot_or_space", user_path)
        if comp.split(".", 1)[0].upper().rstrip() in RESERVED:
            raise PathRejected("reserved_device_name", user_path)
    return [p for p in parts if p not in ("", ".")]


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def is_under(path: str, root: str) -> bool:
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def resolve(root: str, user_path: str) -> str:
    """The real absolute path of ``user_path`` inside ``root``, or ``PathRejected``."""
    parts = check_syntax(user_path)
    root_real = os.path.realpath(root)
    real = os.path.realpath(os.path.join(root_real, *parts)) if parts else root_real
    if not is_under(real, root_real):
        raise PathRejected("outside_workspace", user_path)
    return real


def relative(root: str, real: str) -> str:
    """The workspace-relative POSIX form of a resolved path (what spans record)."""
    rel = os.path.relpath(real, os.path.realpath(root))
    return "." if rel == "." else rel.replace(os.sep, "/")
