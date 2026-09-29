"""Lab03 CLI：Fake / Live 两种模式的演示入口。

用法（项目根目录）：

    python -m app.cli.lab03_demo --fake --trace          # 离线 Fake 四工具轨迹
    python -m app.cli.lab03_demo --live --chat-only      # 真实 GLM 普通对话
    python -m app.cli.lab03_demo --live --trace          # 真实 GLM 四工具任务 + 追问

GLM_API_KEY 未设置时用 getpass 隐藏输入；密钥只进请求头，不写进历史。
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from app.agent.react_agent import Agent, AgentResult
from app.llm.glm_client import GLMClient, LLMError, ProtocolError
from app.llm.fake_llm import FakeLLMClient
from app.tools.basic_tools import BasicTools

DEMO_ROOT = Path("demo_lab03")

CHAT_ONLY_QUESTION = "用一句话介绍 Python。"
TASK_PROMPT = (
    "请在当前目录完成以下任务：1) 列出目录内容；2) 读取 notes.txt；"
    "3) 根据说明写一份简短报告，保存为新文件 report.txt"
    "（若已存在请换一个不重名的文件名）；"
    "4) 执行命令 echo lab03-shell-ok 验证 Shell 可用。"
    "完成后汇报报告文件路径与命令输出。"
)
FOLLOWUP_PROMPT = "报告文件叫什么？请读取它并复核内容是否与 notes.txt 一致。"


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


def build_fake_script() -> list[dict[str, Any]]:
    """Fake 决策：列目录 -> 读文件 -> 写报告 -> 执行命令 -> 预设最终回答。

    报告文件名带随机后缀，重跑不会覆盖已有报告。
    """
    report_name = f"report-{secrets.token_hex(4)}.txt"
    return [
        _tool_call("call_list_1", "list_dir", {"path": "."}),
        _tool_call("call_read_1", "read_file", {"path": "notes.txt"}),
        _tool_call(
            "call_write_1",
            "write_file",
            {
                "path": report_name,
                "content": "【Fake 预设报告】notes.txt 包含两行说明：\n"
                "1. Lab03 演示目录：练习 Reasoning / Acting / Observing。\n"
                "2. 请读取本文件并写一份简短报告。\n"
                "本报告由 FakeLLMClient 预设内容生成，用于离线验证 ReAct 闭环。",
            },
        ),
        _tool_call("call_cmd_1", "execute_command", {"command": "echo lab03-shell-ok"}),
        {
            "role": "assistant",
            "content": (
                "【Fake 预设回答】任务完成（预设文本，非真实模型输出）："
                f"报告已写入 {report_name}，命令 echo lab03-shell-ok 返回 0。"
            ),
        },
    ]


def _trace(result: AgentResult, agent: Agent, fake: bool) -> None:
    print("=" * 60)
    print(f"模式: {'FAKE（离线，预设脚本）' if fake else 'LIVE（真实模型）'}")
    print(f"结果: {result.status} / iterations={result.iterations}")
    print("-" * 60)
    for index, message in enumerate(agent.messages):
        role = message["role"]
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                names = [c["function"]["name"] for c in calls]
                print(f"[{index}] assistant -> 申请工具调用: {names}")
            else:
                print(f"[{index}] assistant -> 最终回答: {message.get('content')}")
        elif role == "tool":
            print(
                f"[{index}] tool (id={message['tool_call_id']}): "
                f"{message['content'][:120]}"
            )
        else:
            print(f"[{index}] {role}: {str(message.get('content'))[:120]}")
    print("=" * 60)
    print(f"最终输出: {result.content}")


def run_fake(trace: bool) -> int:
    tools = BasicTools(DEMO_ROOT)
    fake_client = FakeLLMClient(build_fake_script())
    agent = Agent(fake_client, tools, max_iterations=6)
    result = agent.run(TASK_PROMPT)
    if trace:
        _trace(result, agent, fake=True)
    else:
        print(f"{result.status} / iterations={result.iterations}")
        print(result.content)
    return 0 if result.status == "completed" else 1


def run_live(chat_only: bool, trace: bool) -> int:
    model = os.environ.get("GLM_MODEL", "").strip()
    if not model or model.startswith("替换"):
        print("错误: 请先用 GLM_MODEL 环境变量设置模型标识", file=sys.stderr)
        return 2
    api_key = os.environ.get("GLM_API_KEY") or getpass.getpass("GLM API Key: ")
    client = GLMClient(model=model, api_key=api_key)
    try:
        if chat_only:
            messages = [{"role": "user", "content": CHAT_ONLY_QUESTION}]
            message = client.chat(messages)
            print(f"[模型 {model}] {message['content']}")
            return 0

        tools = BasicTools(DEMO_ROOT)
        agent = Agent(client, tools, max_iterations=8)
        print(f"用户目标: {TASK_PROMPT}")
        result = agent.run(TASK_PROMPT)
        if trace:
            _trace(result, agent, fake=False)
        else:
            print(f"{result.status} / iterations={result.iterations}")
            print(result.content)

        while result.status == "completed":
            follow = input("追问（/exit 退出）: ").strip()
            if follow in ("/exit", "/quit"):
                break
            if not follow:
                continue
            result = agent.run(follow)
            print(f"{result.status} / iterations={result.iterations}")
            print(result.content)
        return 0 if result.status == "completed" else 1
    except LLMError as exc:
        print(f"HTTP/网络错误，停止: {exc}", file=sys.stderr)
        return 1
    except ProtocolError as exc:
        print(f"协议错误，停止: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lab03 GLMClient 与 ReAct 演示")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="离线 Fake 轨迹")
    mode.add_argument("--live", action="store_true", help="真实 GLM 调用")
    parser.add_argument("--chat-only", action="store_true", help="只做普通对话（仅 live）")
    parser.add_argument("--trace", action="store_true", help="打印消息轨迹")
    args = parser.parse_args(argv)
    if args.fake:
        return run_fake(args.trace)
    return run_live(args.chat_only, args.trace)


if __name__ == "__main__":
    raise SystemExit(main())
