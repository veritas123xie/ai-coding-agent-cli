"""初始化脚本：自动创建 AI Coding Agent CLI 的目录结构与空 __init__.py。

用法: python init_project.py
"""

from pathlib import Path

# 需要创建的包目录（相对于项目根目录）
PACKAGES = [
    "app",
    "app/agent",
    "app/cli",
    "app/core",
    "app/llm",
    "app/tools",
    "tests",
]

# 其他普通目录（不需要 __init__.py）
DIRS = [
    "docs",
]


def init_project() -> None:
    """创建目录结构，并在包目录下生成空 __init__.py。"""
    for package in PACKAGES:
        # parents=True 确保父目录一并创建；exist_ok=True 允许重复运行
        Path(package).mkdir(parents=True, exist_ok=True)
        # 创建空 __init__.py，标记该目录为 Python 包
        Path(package, "__init__.py").touch(exist_ok=True)
    for directory in DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("项目结构初始化完成。")


if __name__ == "__main__":
    init_project()
