"""AI Coding Agent CLI — 命令行入口。

使用 Typer 构建子命令，Rich 负责美化终端输出。
"""

import typer
from rich.console import Console
from rich.panel import Panel

# 应用元信息
APP_NAME = "AI Coding Agent CLI"
VERSION = "0.1"

# 全局 Rich 控制台与 Typer 应用实例
console = Console()
app = typer.Typer(
    name="ai-coding-agent",
    help="AI Coding Agent CLI — 你的智能编程助手",
    add_completion=False,
)


@app.command()
def hello(name: str = typer.Option("Developer", "--name", "-n", help="要问候的用户名")) -> None:
    """输出欢迎信息：显示版本号并个性化问候 --name 指定的用户。"""
    # 面板标题：应用名 + 版本号
    title = f"{APP_NAME} v{VERSION}"
    # 个性化欢迎语
    message = f"你好, [bold cyan]{name}[/bold cyan]！欢迎使用 AI Coding Agent CLI。\n" \
              f"输入 [green]--help[/green] 查看可用命令。"
    console.print(Panel(message, title=title, border_style="green"))


@app.command()
def version() -> None:
    """输出版本号。"""
    console.print(f"{APP_NAME} v{VERSION}")


def main() -> None:
    """CLI 主入口：委托给 Typer 的命令分发。"""
    app()


if __name__ == "__main__":
    main()
