"""AI Coding Agent CLI — 命令行入口。

使用 Typer 构建子命令，Rich 负责美化终端输出。
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
def help(ctx: typer.Context) -> None:
    """列出所有可用子命令及说明（等价于 --help，但不含 help 本身）。"""
    # Rich 表格：左列命令名，右列说明
    table = Table(title=f"{APP_NAME} v{VERSION} — 可用子命令", border_style="cyan")
    table.add_column("命令", style="bold green", no_wrap=True)
    table.add_column("说明", style="white")
    # 遍历 Typer 注册的命令，跳过 help 自身
    for name, command in ctx.parent.command.commands.items():
        if name == "help":
            continue
        # 取函数 docstring 首行作为说明
        doc = (command.callback.__doc__ or "").strip().splitlines()
        table.add_row(name, doc[0] if doc else "（无说明）")
    console.print(table)
    console.print("提示: 使用 [green]python main.py <命令> --help[/green] 查看命令的详细参数。")


@app.command()
def version() -> None:
    """输出版本号。"""
    console.print(f"{APP_NAME} v{VERSION}")


def main() -> None:
    """CLI 主入口：委托给 Typer 的命令分发。"""
    app()


if __name__ == "__main__":
    main()
