"""File read/write tool for the AI Coding Agent CLI.

Provides :class:`FileTool`, a thin, cross-platform wrapper around
``pathlib.Path`` for reading and writing text and binary files, together
with structured dataclass results and a small exception hierarchy so
that callers can distinguish read failures from write failures.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


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


@dataclass(frozen=True)
class DirEntry:
    """Structured description of a single directory entry.

    Attributes:
        name: Entry name (file name including extension).
        path: Full path of the entry.
        is_dir: ``True`` for directories, ``False`` for files.
        size: Size in bytes (``0`` for directories).
    """

    name: str
    path: Path
    is_dir: bool
    size: int


class FileTool:
    """Cross-platform file read/write helper.

    All paths are handled via :class:`pathlib.Path`, so both POSIX and
    Windows-style inputs are accepted. Errors are wrapped in
    :class:`FileReadError` / :class:`FileWriteError` (both subclasses of
    :class:`FileToolError`).

    The tool is stateless: the :meth:`batch` context manager only emits
    log lines around a group of operations and tracks nothing between
    calls.
    """

    #: Minimum confidence below which a chardet guess is discarded.
    _DETECT_MIN_CONFIDENCE: float = 0.5

    def read(
        self,
        path: str | Path,
        encoding: str | None = "utf-8",
    ) -> FileContent:
        """Read a text file and return its content.

        Args:
            path: Path to the file, as a string or ``Path``.
            encoding: Text encoding used to decode the file. Pass
                ``None`` to auto-detect the encoding with ``chardet``
                (falls back to UTF-8 when detection is inconclusive or
                the detected encoding fails to decode).

        Returns:
            A :class:`FileContent` with the content and file metadata.

        Raises:
            FileReadError: If the file is missing, unreadable, or cannot
                be decoded with the given (or detected) encoding.
        """
        file_path = Path(path).expanduser()
        try:
            resolved = file_path.resolve(strict=True)
        except OSError as exc:
            raise FileReadError(
                f"Failed to read {file_path}: {exc}"
            ) from exc

        if encoding is None:
            content, encoding = self._read_text_auto(resolved)
        else:
            try:
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

    def read_lines(
        self,
        path: str | Path,
        encoding: str | None = "utf-8",
    ) -> list[str]:
        """Read a text file and return its lines without trailing newlines.

        Args:
            path: Path to the file, as a string or ``Path``.
            encoding: Text encoding used to decode the file (``None``
                enables auto-detection, see :meth:`read`).

        Returns:
            A list of lines with newline characters stripped.

        Raises:
            FileReadError: If the file cannot be read (same conditions
                as :meth:`read`).
        """
        return self.read(path, encoding=encoding).content.splitlines()

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

    def append(
        self,
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> WriteResult:
        """Append text content to the end of a file.

        Args:
            path: Path to the file, as a string or ``Path``.
            content: Text content to append.
            encoding: Text encoding used to encode the content.

        Returns:
            A :class:`WriteResult` with the path, byte count of the
            appended content, and whether the file was newly created.

        Raises:
            FileWriteError: If the append fails.
        """
        file_path = Path(path).expanduser()
        created = not file_path.exists()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode(encoding)
            with open(file_path, "ab") as f:
                f.write(data)
            resolved = file_path.resolve()
        except OSError as exc:
            raise FileWriteError(
                f"Failed to append to {file_path}: {exc}"
            ) from exc
        return WriteResult(
            path=resolved,
            bytes_written=len(data),
            created=created,
        )

    def list_dir(
        self,
        path: str | Path,
        recursive: bool = False,
        pattern: str = "*",
    ) -> list[DirEntry]:
        """List the entries of a directory as structured objects.

        Args:
            path: Path to the directory, as a string or ``Path``.
            recursive: If ``True``, enumerate recursively using
                :meth:`Path.rglob` with ``pattern``; otherwise list only
                direct children with :meth:`Path.iterdir`.
            pattern: Glob pattern matched against entry names when
                ``recursive`` is ``True`` (ignored otherwise).

        Returns:
            A sorted list of :class:`DirEntry` (stable order for
            reproducible output and tests). Directory sizes are ``0``.

        Raises:
            FileReadError: If the path does not exist, is not a
                directory, or cannot be enumerated.
        """
        dir_path = Path(path).expanduser()
        if not dir_path.exists():
            raise FileReadError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise FileReadError(f"Not a directory: {dir_path}")
        try:
            children = (
                list(dir_path.rglob(pattern))
                if recursive
                else list(dir_path.iterdir())
            )
        except OSError as exc:
            raise FileReadError(
                f"Failed to list {dir_path}: {exc}"
            ) from exc

        entries: list[DirEntry] = []
        for child in sorted(children):
            try:
                size = child.stat().st_size if child.is_file() else 0
            except OSError:
                size = 0
            entries.append(
                DirEntry(
                    name=child.name,
                    path=child,
                    is_dir=child.is_dir(),
                    size=size,
                )
            )
        return entries

    def read_bytes(self, path: str | Path) -> bytes:
        """Read a file as raw bytes (no encoding applied).

        Args:
            path: Path to the file, as a string or ``Path``.

        Returns:
            The file's raw content as ``bytes``.

        Raises:
            FileReadError: If the file cannot be read.
        """
        file_path = Path(path).expanduser()
        try:
            return file_path.read_bytes()
        except OSError as exc:
            raise FileReadError(
                f"Failed to read {file_path}: {exc}"
            ) from exc

    def write_bytes(
        self,
        path: str | Path,
        data: bytes,
    ) -> WriteResult:
        """Write raw bytes to a file (no encoding applied).

        Args:
            path: Path to the file, as a string or ``Path``.
            data: Raw bytes to write.

        Returns:
            A :class:`WriteResult` with the path, byte count, and whether
            the file was newly created.

        Raises:
            FileWriteError: If the write fails.
        """
        file_path = Path(path).expanduser()
        created = not file_path.exists()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
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

    @contextmanager
    def batch(self) -> Iterator[FileTool]:
        """Context manager marking a group of file operations.

        Prints a start line on entry and an end line on exit (including
        exceptional exits), which helps an Agent delimit multi-file
        operations in logs. Yields the tool itself; no state is kept
        between calls.

        Yields:
            The :class:`FileTool` instance.
        """
        print("批量开始")
        try:
            yield self
        finally:
            print("批量结束")

    def _read_text_auto(self, file_path: Path) -> tuple[str, str]:
        """Decode a file whose encoding is unknown.

        Reads the raw bytes, asks ``chardet`` for a best guess, and
        tries the guess first and UTF-8 as a fallback. Low-confidence
        guesses are discarded, since short samples routinely produce
        misleading detections.

        Args:
            file_path: Resolved path to the file.

        Returns:
            A ``(content, encoding)`` pair with the decoded text and the
            encoding that succeeded.

        Raises:
            FileReadError: If the file cannot be read or no candidate
                encoding can decode it.
        """
        try:
            import chardet
        except ImportError as exc:
            raise FileReadError(
                "Auto-detect requires the 'chardet' package: "
                "install it with 'pip install chardet'"
            ) from exc

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            raise FileReadError(
                f"Failed to read {file_path}: {exc}"
            ) from exc

        try:
            guess = chardet.detect(raw)
        except Exception as exc:  # detection must never crash the tool
            guess = None
        detected = (
            (guess or {}).get("encoding")
            if (guess or {}).get("confidence", 0.0) >= self._DETECT_MIN_CONFIDENCE
            else None
        )

        candidates: list[str] = []
        if detected:
            candidates.append(detected)
        if "utf-8" not in candidates:
            candidates.append("utf-8")

        for candidate in candidates:
            try:
                return raw.decode(candidate), candidate
            except (UnicodeDecodeError, LookupError):
                continue
        raise FileReadError(
            f"Failed to decode {file_path}: tried {candidates}"
        )

    def _detect_encoding(self, raw: bytes) -> str | None:
        """Guess the encoding of a byte string with ``chardet``.

        Args:
            raw: Raw bytes to inspect.

        Returns:
            The detected encoding name, or ``None`` when detection is
            unavailable or below the confidence threshold.
        """
        try:
            import chardet
        except ImportError:
            return None
        try:
            guess = chardet.detect(raw)
        except Exception:
            return None
        if (guess or {}).get("confidence", 0.0) < self._DETECT_MIN_CONFIDENCE:
            return None
        return (guess or {}).get("encoding")
