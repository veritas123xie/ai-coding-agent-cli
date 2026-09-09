# AI Coding Agent CLI

PKU Course Project — 面向金融的 Python

从零复现 AI Coding Agent：Python + LLM + Tool Calling + ReAct

## 项目结构

```text
.
├── main.py            # CLI 入口（Typer + Rich）
├── init_project.py    # 脚手架脚本：重建目录结构
├── environment.yml    # Conda 环境定义
├── app/               # 核心包（agent / cli / core / llm / tools）
├── tests/             # 测试
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

## 脚手架脚本

删除 `app/` 后可一键重建目录结构与空的 `__init__.py`：

```powershell
python init_project.py
```
