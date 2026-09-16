"""Unit tests for app.tools.file_tool.FileTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.file_tool import (
    FileContent,
    FileReadError,
    FileTool,
    FileToolError,
    FileWriteError,
    WriteResult,
)


@pytest.fixture()
def tool() -> FileTool:
    """Provide a fresh FileTool instance for each test."""
    return FileTool()


class TestRead:
    """Tests for FileTool.read."""

    def test_read_returns_file_content(self, tool: FileTool, tmp_path: Path) -> None:
        file_path = tmp_path / "sample.txt"
        file_path.write_text("hello\nworld\n", encoding="utf-8")

        result = tool.read(file_path)

        assert isinstance(result, FileContent)
        assert result.path == file_path.resolve()
        assert result.content == "hello\nworld\n"
        assert result.encoding == "utf-8"
        assert result.size == 12
        assert result.line_count == 2

    def test_read_missing_file_raises_file_read_error(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        with pytest.raises(FileReadError) as exc_info:
            tool.read(tmp_path / "does_not_exist.txt")

        assert isinstance(exc_info.value, FileToolError)

    def test_read_directory_raises_file_read_error(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        with pytest.raises(FileReadError):
            tool.read(tmp_path)

    def test_read_bad_encoding_raises_file_read_error(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "utf8.txt"
        file_path.write_text("中文", encoding="utf-8")

        with pytest.raises(FileReadError):
            tool.read(file_path, encoding="ascii")


class TestWrite:
    """Tests for FileTool.write."""

    def test_write_creates_new_file(self, tool: FileTool, tmp_path: Path) -> None:
        file_path = tmp_path / "new.txt"

        result = tool.write(file_path, "content")

        assert isinstance(result, WriteResult)
        assert result.path == file_path.resolve()
        assert result.bytes_written == len("content".encode("utf-8"))
        assert result.created is True
        assert file_path.read_text(encoding="utf-8") == "content"

    def test_write_overwrite_existing(self, tool: FileTool, tmp_path: Path) -> None:
        file_path = tmp_path / "existing.txt"
        file_path.write_text("old", encoding="utf-8")

        result = tool.write(file_path, "new content")

        assert result.created is False
        assert file_path.read_text(encoding="utf-8") == "new content"

    def test_write_no_overwrite_raises_file_write_error(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "existing.txt"
        file_path.write_text("old", encoding="utf-8")

        with pytest.raises(FileWriteError) as exc_info:
            tool.write(file_path, "new", overwrite=False)

        assert isinstance(exc_info.value, FileToolError)
        # Original content must be untouched.
        assert file_path.read_text(encoding="utf-8") == "old"

    def test_write_creates_missing_parent_dirs(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "a" / "b" / "c" / "nested.txt"

        result = tool.write(file_path, "nested")

        assert result.created is True
        assert file_path.read_text(encoding="utf-8") == "nested"


class TestRoundTrip:
    """Tests for read-after-write consistency."""

    @pytest.mark.parametrize(
        "content",
        [
            "simple text",
            "multi\nline\ntext\n",
            "with unicode 中文 and émojis",
            "",
        ],
    )
    def test_write_then_read_round_trip(
        self, tool: FileTool, tmp_path: Path, content: str
    ) -> None:
        file_path = tmp_path / "roundtrip.txt"

        tool.write(file_path, content)
        result = tool.read(file_path)

        assert result.content == content
        assert result.encoding == "utf-8"
        assert result.size == len(content.encode("utf-8"))


class TestLineCount:
    """Tests for FileContent.line_count."""

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("", 0),
            ("no trailing newline", 1),
            ("one line\n", 1),
            ("a\nb\nc", 3),
            ("a\nb\nc\n", 3),
            ("\n", 1),
            ("\n\n\n", 3),
        ],
    )
    def test_line_count(
        self, tool: FileTool, tmp_path: Path, content: str, expected: int
    ) -> None:
        file_path = tmp_path / "lines.txt"
        tool.write(file_path, content)

        result = tool.read(file_path)

        assert result.line_count == expected
