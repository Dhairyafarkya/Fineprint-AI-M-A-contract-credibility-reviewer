"""Unpack uploaded datasets: loose .txt files, folders, and .zip archives (e.g. MAUD's data.zip).

Only text documents are kept. Other files (CSVs, images, code) are skipped and
reported back, so the user knows what was ignored and why.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

TEXT_EXTENSIONS = (".txt", ".text")
MAX_ARCHIVE_TEXT_BYTES = 500 * 1024 * 1024  # stop unpacking past 500 MB of text (zip-bomb guard)


@dataclass
class UnpackResult:
    files: list[tuple[str, bytes]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _is_hidden(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return any(p.startswith(".") or p == "__MACOSX" for p in parts)


def unpack(uploads: list[tuple[str, bytes]]) -> UnpackResult:
    """Turn uploaded (name, bytes) pairs into text documents, expanding any .zip files."""
    out = UnpackResult()
    total = 0
    for name, data in uploads:
        base = PurePosixPath(name.replace("\\", "/")).name
        low = base.lower()
        if _is_hidden(name):
            continue
        if low.endswith(TEXT_EXTENSIONS):
            out.files.append((base, data))
        elif low.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        inner = info.filename
                        if info.is_dir() or _is_hidden(inner):
                            continue
                        if not inner.lower().endswith(TEXT_EXTENSIONS):
                            out.skipped.append(f"{base}/{inner}")
                            continue
                        total += info.file_size
                        if total > MAX_ARCHIVE_TEXT_BYTES:
                            out.skipped.append(f"{base}: stopped after 500 MB of text")
                            break
                        out.files.append((PurePosixPath(inner).name, zf.read(info)))
            except zipfile.BadZipFile:
                out.skipped.append(f"{base} (not a valid zip file)")
        else:
            out.skipped.append(base)
    return out


def natural_key(text: str) -> list:
    """Sort 'contract_2' before 'contract_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]
