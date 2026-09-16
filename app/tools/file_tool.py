"""File read/write tool for the AI Coding Agent CLI.

Provides :class:`FileTool`, a thin, cross-platform wrapper around
``pathlib.Path`` for reading and writing text files, together with
structured dataclass results and a small exception hierarchy so that
callers can distinguish read failures from write failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class FileToolError(Exception):
    """Base exception for all file tool errors."""


class FileReadError(FileToolError):
    """Raised when a file cannot be read (missing, unreadable, bad encoding)."""


class FileWriteError(FileToolError):
    """Raised when a file cannot be written."""


@dataclass(frozen=True)
class FileContent:
    """Result of reading a file.

    Attributes:
        path: Absolute path of the file that was read.
        content: The file's text content.
        encoding: The encoding used to decode the file.
        size: File size in bytes.
        line_count: Number of lines in the content.
    """

    path: Path
    content: str
    encoding: str
    size: int
    line_count: int


@dataclass(frozen=True)
class WriteResult:
    """Result of writing a file.

    Attributes:
        path: Absolute path of the file that was written.
        bytes_written: Number of bytes written.
        created: ``True`` if the file was newly created, ``False`` if an
            existing file was overwritten.
    """

    path: Path
    bytes_written: int
    created: bool


class FileTool:
    """Cross-platform file read/write helper.

    All paths are handled via :class:`pathlib.Path`, so both POSIX and
    Windows-style inputs are accepted. Errors are wrapped in
    :class:`FileReadError` / :class:`FileWriteError` (both subclasses of
    :class:`FileToolError`).
    """

    def read(self, path: str | Path, encoding: str = "utf-8") -> FileContent:
        """Read a text file and return its content.

        Args:
            path: Path to the file, as a string or ``Path``.
            encoding: Text encoding used to decode the file.

        Returns:
            A :class:`FileContent` with the content and file metadata.

        Raises:
            FileReadError: If the file is missing, unreadable, or cannot
                be decoded with the given encoding.
        """
        file_path = Path(path).expanduser()
        try:
            resolved = file_path.resolve(strict=True)
            content = resolved.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            raise FileReadError(
                f"Failed to decode {file_path} with encoding {encoding!r}"
            ) from exc
        except OSError as exc:
            raise FileReadError(
                f"Failed to read {file_path}: {exc}"
            ) from exc
        return FileContent(
            path=resolved,
            content=content,
            encoding=encoding,
            size=len(content.encode(encoding)),
            line_count=len(content.splitlines()),
        )

    def write(
        self,
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = True,
    ) -> WriteResult:
        """Write text content to a file.

        Args:
            path: Path to the file, as a string or ``Path``.
            content: Text content to write.
            encoding: Text encoding used to encode the content.
            overwrite: If ``True``, overwrite an existing file; if
                ``False``, raise :class:`FileWriteError` when the file
                already exists.

        Returns:
            A :class:`WriteResult` with the path, byte count, and whether
            the file was newly created.

        Raises:
            FileWriteError: If the write fails or the file exists and
                ``overwrite`` is ``False``.
        """
        file_path = Path(path).expanduser()
        created = not file_path.exists()
        if not created and not overwrite:
            raise FileWriteError(
                f"File already exists: {file_path} (overwrite=False)"
            )
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode(encoding)
            file_path.write_bytes(data)
            resolved = file_path.resolve()
        except OSError as exc:
            raise FileWriteError(
                f"Failed to write {file_path}: {exc}"
            ) from exc
        return WriteResult(
            path=resolved,
            bytes_written=len(data),
            created=created,
        )
