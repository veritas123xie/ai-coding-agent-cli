# AI Coding Agent CLI

PKU Course Project — 面向金融的 Python

从零复现 AI Coding Agent：Python + LLM + Tool Calling + ReAct

## 项目结构

```text
.
├── main.py            # CLI 入口（Typer + Rich）
├── init_project.py    # 脚手架脚本：重建目录结构
├── environment.yml    # Conda 环境定义
├── requirements.txt   # pip 依赖清单（pytest / chardet）
├── app/               # 核心包（agent / cli / core / llm / tools）
│   └── tools/
│       └── file_tool.py   # FileTool：Agent 的文件读写工具（Lab02）
├── tests/             # pytest 单元测试
└── docs/              # 文档
```

## 安装

```powershell
conda create -n .venv python=3.13 -y
conda activate .venv
conda env update -f environment.yml
```

## 运行

```powershell
python main.py hello               # 默认问候 Developer
python main.py hello --name 张三   # 个性化问候
python main.py version             # 查看版本号
python main.py help                # 列出所有子命令
python main.py --help              # Typer 内置帮助
```

`hello` 命令输出示例：

```text
+------------------------- AI Coding Agent CLI v0.1 -------------------------+
| 你好, 张三！欢迎使用 AI Coding Agent CLI。                                 |
| 输入 --help 查看可用命令。                                                 |
+----------------------------------------------------------------------------+
```

## 工具（Tools）

`app/tools/` 下的工具为 Agent 提供可复用的原子能力，返回值均为带类型的
dataclass，失败时抛出专属异常（便于 Agent 捕获错误并自我修复）。

### FileTool（Lab02）

跨平台文件读写工具，`pathlib.Path` 处理路径，统一 UTF-8。

| 方法 | 说明 | 返回 |
|---|---|---|
| `read(path, encoding="utf-8")` | 读取文本文件（`encoding=None` 时用 chardet 自动检测） | `FileContent` |
| `read_lines(path, encoding="utf-8")` | 按行读取，返回去换行的行列表 | `list[str]` |
| `write(path, content, encoding="utf-8", overwrite=True)` | 写入文本，自动创建父目录 | `WriteResult` |
| `append(path, content, encoding="utf-8")` | 追加写 | `WriteResult` |
| `list_dir(path, recursive=False, pattern="*")` | 列举目录（可递归 + glob 过滤） | `list[DirEntry]` |
| `read_bytes(path)` / `write_bytes(path, data)` | 二进制读写 | `bytes` / `WriteResult` |
| `batch()` | 上下文管理器，标记批量操作的开始/结束 | `FileTool` |

异常体系：`FileToolError`（基类）→ `FileReadError` / `FileWriteError`。

使用示例：

```python
from app.tools import FileTool

tool = FileTool()
result = tool.write("demo.txt", "你好\n")
print(result.bytes_written)          # 7
fc = tool.read("demo.txt")
print(fc.line_count)                 # 1
```

## 测试

```powershell
pytest -v                        # 运行全部测试（33 个用例）
pytest tests/test_file_tool.py   # 只运行 FileTool 测试
```

测试使用 pytest 的 `tmp_path` fixture，全部在临时目录中进行，不污染真实文件系统。

## 脚手架脚本

删除 `app/` 后可一键重建目录结构与空的 `__init__.py`：

```powershell
python init_project.py
```
