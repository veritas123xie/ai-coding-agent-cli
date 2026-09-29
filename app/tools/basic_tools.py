"""Lab03 BasicTools：四种可调用工具 + Schema + 显式分发。

工具（直接调用方法，输入非法时抛错）：
- ``read_file(path)``          -> {"ok": True, "content": ...}
- ``write_file(path, content, overwrite=False)``
                               -> {"ok": True, "path", "bytes_written"}
- ``list_dir(path)``           -> {"ok": True, "entries", "truncated"}
- ``execute_command(command)`` -> {"ok": ..., "stdout", "stderr",
                                   "returncode", "timed_out", "truncated"}

``dispatch(name, arguments)`` 是模型调用的入口：名称白名单 + 参数 JSON
解码 + 对象/字段/类型校验，任何输入或执行错误都转换为
``{"ok": False, "error": ...}`` 的 Observation，而不是抛给 Agent。

边界：所有路径必须位于教学根目录内（拒绝绝对路径与 ``..`` 逃逸）；
命令只允许固定白名单（Schema 与 Python 双重检查），不使用 eval/exec。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from app.tools.file_tool import FileTool, FileToolError

MAX_LIST_ENTRIES = 50
MAX_OUTPUT_CHARS = 2000
COMMAND_TIMEOUT = 5

WHITELISTED_COMMANDS = {"echo lab03-shell-ok"}
WHITELISTED_COMMANDS.add("cd" if os.name == "nt" else "pwd")


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class BasicTools:
    """教学目录内的四种基础工具。"""

    def __init__(self, root: str | Path, file_tool: FileTool | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_tool = file_tool or FileTool()

    # ---------- 路径边界 ----------

    def _resolve(self, path: Any, *, must_be_dir: bool = False) -> Path:
        """把模型给的相对路径约束到教学根目录内。"""
        if not isinstance(path, str) or not path:
            raise ValueError("path 必须是非空字符串")
        candidate = Path(path)
        if candidate.is_absolute():
            raise ValueError(f"不允许绝对路径: {path!r}")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"路径越出教学目录: {path!r}")
        if must_be_dir and not resolved.is_dir():
            raise ValueError(f"不是目录: {path!r}")
        return resolved

    # ---------- 四个工具（直接调用：非法输入抛错） ----------

    def read_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        content = self.file_tool.read(target, encoding="utf-8").content
        return {"ok": True, "content": content}

    def write_file(
        self, path: str, content: str, overwrite: bool = False
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("content 必须是字符串")
        target = self._resolve(path)
        result = self.file_tool.write(
            target, content, encoding="utf-8", overwrite=overwrite
        )
        return {
            "ok": True,
            "path": result.path.name,
            "bytes_written": result.bytes_written,
        }

    def list_dir(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path, must_be_dir=True)
        entries = sorted(p.name for p in target.iterdir())
        truncated = len(entries) > MAX_LIST_ENTRIES
        return {
            "ok": True,
            "entries": entries[:MAX_LIST_ENTRIES],
            "truncated": truncated,
        }

    def execute_command(self, command: str) -> dict[str, Any]:
        if not isinstance(command, str) or not command:
            raise ValueError("command 必须是非空字符串")
        if command not in WHITELISTED_COMMANDS:
            raise ValueError(f"命令不在白名单: {command!r}")
        if os.name == "nt":
            argv = [os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"), "/d", "/c", command]
        else:
            argv = ["/bin/sh", "-c", command]
        try:
            completed = subprocess.run(
                argv,
                cwd=self.root,
                capture_output=True,
                timeout=COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            data: dict[str, Any] = {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": None,
                "timed_out": True,
            }
        else:
            data = {
                "stdout": completed.stdout.decode("utf-8", errors="replace"),
                "stderr": completed.stderr.decode("utf-8", errors="replace"),
                "returncode": completed.returncode,
                "timed_out": False,
            }
        data["stdout"], out_trunc = _truncate(data["stdout"])
        data["stderr"], err_trunc = _truncate(data["stderr"])
        data["truncated"] = out_trunc or err_trunc
        # 非零退出码保留失败状态（ok=False 覆盖前面的默认值）
        return {"ok": data["returncode"] == 0 and not data["timed_out"], **data}

    # ---------- Schema ----------

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        """四工具的 Function Calling Schema（描述只是说明，执行仍要校验）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取教学目录中的 UTF-8 文本文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "相对文件路径",
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "在教学目录中创建 UTF-8 文本文件（默认不覆盖已有文件）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "相对文件路径"},
                            "content": {"type": "string", "description": "文件内容"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "列出教学目录某一层的条目（不递归）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "相对目录路径"}
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "执行白名单内的固定 Shell 命令并返回输出与退出码",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "完整命令字符串，必须在白名单内",
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    # ---------- 显式分发 ----------

    _HANDLERS = ("read_file", "write_file", "list_dir", "execute_command")

    def dispatch(self, name: Any, arguments: Any) -> dict[str, Any]:
        """模型调用的入口：校验后执行，错误转换为 Observation。"""
        if name not in self._HANDLERS:
            return {"ok": False, "error": f"未知工具: {name!r}"}
        try:
            args = self._parse_arguments(arguments)
            self._validate_args(name, args)
            handler = getattr(self, name)
            return handler(**args)
        except (ValueError, FileToolError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, str):
            raise ValueError("arguments 必须是 JSON 字符串")
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError:
            raise ValueError("工具参数不是合法 JSON") from None
        if not isinstance(value, dict):
            raise ValueError("工具参数必须是对象")
        return value

    def _validate_args(self, name: str, args: dict[str, Any]) -> None:
        spec: dict[str, dict[str, Any]] = {
            "read_file": {"path": str},
            "write_file": {"path": str, "content": str},
            "list_dir": {"path": str},
            "execute_command": {"command": str},
        }
        required = set(spec[name])
        keys = set(args)
        if keys != required:
            missing = required - keys
            extra = keys - required
            parts = []
            if missing:
                parts.append(f"缺少字段: {sorted(missing)}")
            if extra:
                parts.append(f"多余字段: {sorted(extra)}")
            raise ValueError("；".join(parts))
        wrong = [k for k, t in spec[name].items() if not isinstance(args[k], t)]
        if wrong:
            raise ValueError(f"字段类型错误: {wrong}")
