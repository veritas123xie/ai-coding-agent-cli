"""Unit tests for app.tools.file_tool.FileTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.file_tool import (
    DirEntry,
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


class TestListDir:
    """Tests for FileTool.list_dir (Step 10)."""

    def test_list_dir_returns_sorted_entries(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        (tmp_path / "b.txt").write_text("hi", encoding="utf-8")
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "sub").mkdir()

        entries = tool.list_dir(tmp_path)

        assert all(isinstance(e, DirEntry) for e in entries)
        assert [e.name for e in entries] == ["a.txt", "b.txt", "sub"]
        sub = next(e for e in entries if e.name == "sub")
        assert sub.is_dir is True
        assert sub.size == 0
        txt = next(e for e in entries if e.name == "a.txt")
        assert txt.is_dir is False
        assert txt.size == 5

    def test_list_dir_recursive_with_pattern(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c.py").write_text("x", encoding="utf-8")
        (tmp_path / "a" / "b" / "c.txt").write_text("y", encoding="utf-8")

        entries = tool.list_dir(tmp_path, recursive=True, pattern="*.py")

        assert [e.name for e in entries] == ["c.py"]
        assert all(not e.is_dir for e in entries)

    def test_list_dir_missing_raises(self, tool: FileTool, tmp_path: Path) -> None:
        with pytest.raises(FileReadError):
            tool.list_dir(tmp_path / "nope")

    def test_list_dir_on_file_raises(self, tool: FileTool, tmp_path: Path) -> None:
        file_path = tmp_path / "f.txt"
        file_path.write_text("x", encoding="utf-8")

        with pytest.raises(FileReadError):
            tool.list_dir(file_path)


class TestAppendAndReadLines:
    """Tests for FileTool.append and FileTool.read_lines (Step 11)."""

    def test_append_creates_file(self, tool: FileTool, tmp_path: Path) -> None:
        file_path = tmp_path / "log.txt"

        result = tool.append(file_path, "first\n")

        assert result.created is True
        assert result.bytes_written == len("first\n".encode("utf-8"))
        assert file_path.read_text(encoding="utf-8") == "first\n"

    def test_append_twice_concatenates(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "log.txt"

        first = tool.append(file_path, "one\n")
        second = tool.append(file_path, "two\n")

        assert first.created is True
        assert second.created is False
        assert file_path.read_text(encoding="utf-8") == "one\ntwo\n"

    def test_read_lines_strips_newlines(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "lines.txt"
        tool.write(file_path, "a\nb\nc\n")

        assert tool.read_lines(file_path) == ["a", "b", "c"]

    def test_read_lines_missing_raises(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        with pytest.raises(FileReadError):
            tool.read_lines(tmp_path / "nope.txt")


class TestDetectEncoding:
    """Tests for encoding auto-detection in FileTool.read (Step 12)."""

    def test_read_auto_detects_gbk(self, tool: FileTool, tmp_path: Path) -> None:
        # 需要足够长的样本，chardet 对短文本的猜测置信度很低。
        text = "这是一段用于测试编码自动检测的中文文本，内容需要足够长。"
        file_path = tmp_path / "gbk.txt"
        file_path.write_bytes(text.encode("gbk"))

        result = tool.read(file_path, encoding=None)

        assert result.content == text
        # chardet 版本不同可能返回 GB2312 / GBK / GB18030，
        # 三者互为兼容超集，内容正确才是本质断言。
        assert result.encoding in ("GB2312", "GBK", "GB18030")

    def test_read_auto_detects_utf8(self, tool: FileTool, tmp_path: Path) -> None:
        file_path = tmp_path / "utf8.txt"
        tool.write(file_path, "plain utf-8 text")

        result = tool.read(file_path, encoding=None)

        assert result.content == "plain utf-8 text"


class TestBinary:
    """Tests for binary read/write (Step 13)."""

    def test_write_then_read_bytes_round_trip(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "img" / "pixel.bin"
        data = bytes(range(256)) + b"\x00\xff"

        result = tool.write_bytes(file_path, data)

        assert result.created is True
        assert result.bytes_written == len(data)
        assert tool.read_bytes(file_path) == data

    def test_read_bytes_missing_raises(
        self, tool: FileTool, tmp_path: Path
    ) -> None:
        with pytest.raises(FileReadError):
            tool.read_bytes(tmp_path / "nope.bin")


class TestBatch:
    """Tests for the FileTool.batch context manager (Step 13)."""

    def test_batch_prints_start_and_end(
        self, tool: FileTool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with tool.batch() as batched:
            assert batched is tool
            batched.write(tmp_path / "a.txt", "a")

        out = capsys.readouterr().out
        assert "批量开始" in out
        assert "批量结束" in out

    def test_batch_prints_end_on_exception(
        self, tool: FileTool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(FileReadError):
            with tool.batch():
                tool.read(tmp_path / "missing.txt")

        out = capsys.readouterr().out
        assert "批量开始" in out
        assert "批量结束" in out
