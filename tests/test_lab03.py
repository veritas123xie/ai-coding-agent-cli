"""Lab03 测试：GLMClient（HTTP Mock）、BasicTools、Fake ReAct 闭环。

分层验证：
- MockTransport 检查 GLMClient 构造的请求与解析的响应；
- Fake + 真实临时文件工具检查 Agent 分支、顺序、回灌、次数；
- 故障注入（如缺失回灌）时相关断言必须失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.agent.react_agent import Agent, AgentResult
from app.llm.glm_client import API_URL, GLMClient, LLMError, ProtocolError
from app.llm.fake_llm import FakeLLMClient, FakeExhaustedError
from app.tools.basic_tools import MAX_OUTPUT_CHARS, BasicTools
from app.tools.file_tool import FileToolError

MODEL = "glm-4-test"
KEY = "test-key"


def make_client(handler) -> GLMClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GLMClient(model=MODEL, api_key=KEY, http_client=http)


def ok_body(message: dict, finish: str = "stop") -> dict:
    return {"choices": [{"finish_reason": finish, "message": message}]}


def tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


@pytest.fixture()
def demo(tmp_path: Path) -> BasicTools:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "notes.txt").write_text("第一行说明\n第二行说明\n", encoding="utf-8")
    return BasicTools(root)


# ---------- A. JSON / Observation 基础 ----------


def test_observation_preserves_chinese() -> None:
    result = {"ok": True, "content": "中文说明"}
    wire_text = json.dumps(result, ensure_ascii=False)
    assert json.loads(wire_text) == result
    assert "中文" in wire_text  # ensure_ascii=False 不产生 \uXXXX


def test_arguments_need_second_decode() -> None:
    """function.arguments 是字符串，外层 json 解析后仍需单独解码。"""
    outer = {"tool_calls": [tool_call("c1", "read_file", {"path": "notes.txt"})]}
    raw = outer["tool_calls"][0]["function"]["arguments"]
    assert isinstance(raw, str)
    assert json.loads(raw) == {"path": "notes.txt"}


# ---------- B. GLMClient（HTTP Mock） ----------


def test_glm_request_construction() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body({"role": "assistant", "content": "你好"}))

    client = make_client(handler)
    message = client.chat([{"role": "user", "content": "hi"}])
    assert captured["url"] == API_URL
    assert captured["auth"] == f"Bearer {KEY}"
    payload = captured["payload"]
    assert payload["model"] == MODEL
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload  # 普通对话不带 tools
    assert message["content"] == "你好"


def test_glm_sends_tools_when_given() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body({"role": "assistant", "content": None, "tool_calls": None}))

    client = make_client(handler)
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    client.chat([{"role": "user", "content": "hi"}], tools=tools)
    assert captured["payload"]["tools"] == tools


def test_glm_parses_tool_calls() -> None:
    message_in = {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call("call_1", "read_file", {"path": "notes.txt"})],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(message_in, finish="tool_calls"))

    client = make_client(handler)
    message = client.chat([{"role": "user", "content": "读文件"}])
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["name"] == "read_file"


def test_glm_http_error_raises_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    client = make_client(handler)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_glm_invalid_json_raises_llmerror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = make_client(handler)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_glm_unsupported_finish_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body({"role": "assistant", "content": "截断"}, finish="length"))

    client = make_client(handler)
    with pytest.raises(ProtocolError):
        client.chat([{"role": "user", "content": "hi"}])


def test_glm_missing_tool_call_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        bad = {"role": "assistant", "content": None,
               "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}
        return httpx.Response(200, json=ok_body(bad, finish="tool_calls"))

    client = make_client(handler)
    with pytest.raises(ProtocolError):
        client.chat([{"role": "user", "content": "hi"}])


def test_glm_rejects_bad_message_structure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body({"role": "assistant", "content": "x"}))

    client = make_client(handler)
    with pytest.raises(ProtocolError):
        client.chat([{"role": "hacker", "content": "hi"}])  # 未知角色


# ---------- C. BasicTools ----------


def test_read_file_returns_content(demo: BasicTools) -> None:
    result = demo.read_file("notes.txt")
    assert result["ok"] is True
    assert "第一行说明" in result["content"]


def test_dispatch_read_missing_file_returns_ok_false(demo: BasicTools) -> None:
    result = demo.dispatch("read_file", json.dumps({"path": "nope.txt"}))
    assert result["ok"] is False
    assert "error" in result


def test_write_file_creates_and_counts_bytes(demo: BasicTools) -> None:
    result = demo.write_file("报告.txt", "中文内容")  # 2 字符 != 字节数
    assert result["ok"] is True
    assert result["bytes_written"] == len("中文内容".encode("utf-8"))
    assert (demo.root / "报告.txt").exists()


def test_write_file_no_overwrite_conflict(demo: BasicTools) -> None:
    first = demo.dispatch("write_file", json.dumps({"path": "a.txt", "content": "1"}))
    assert first["ok"] is True
    second = demo.dispatch("write_file", json.dumps({"path": "a.txt", "content": "2"}))
    assert second["ok"] is False  # overwrite=False 默认，重复写入返回错误
    assert (demo.root / "a.txt").read_text(encoding="utf-8") == "1"


def test_list_dir_sorted_and_entries(demo: BasicTools) -> None:
    (demo.root / "b.txt").write_text("b", encoding="utf-8")
    (demo.root / "a.txt").write_text("a", encoding="utf-8")
    result = demo.list_dir(".")
    assert result == {"ok": True, "entries": ["a.txt", "b.txt", "notes.txt"], "truncated": False}


def test_dispatch_list_dir_missing_dir(demo: BasicTools) -> None:
    result = demo.dispatch("list_dir", json.dumps({"path": "no-such-dir"}))
    assert result["ok"] is False


def test_dispatch_rejects_path_traversal(demo: BasicTools) -> None:
    result = demo.dispatch("read_file", json.dumps({"path": "../outside.txt"}))
    assert result["ok"] is False
    assert "教学目录" in result["error"]


def test_dispatch_rejects_absolute_path(demo: BasicTools) -> None:
    result = demo.dispatch("read_file", json.dumps({"path": "C:/Windows/system.ini"}))
    assert result["ok"] is False
    assert "绝对路径" in result["error"]


def test_execute_command_whitelisted(demo: BasicTools) -> None:
    result = demo.execute_command("echo lab03-shell-ok")
    assert result["ok"] is True
    assert "lab03-shell-ok" in result["stdout"]
    assert result["returncode"] == 0
    assert result["timed_out"] is False


def test_execute_command_non_whitelisted(demo: BasicTools) -> None:
    result = demo.dispatch(
        "execute_command", json.dumps({"command": "echo lab03-shell-ok && del notes.txt"})
    )
    assert result["ok"] is False  # 组合命令不属于白名单
    assert (demo.root / "notes.txt").exists()


def test_execute_command_timeout(demo: BasicTools, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.basic_tools as bt

    def fake_run(*args, **kwargs):
        raise bt.subprocess.TimeoutExpired(cmd="sleep", timeout=5)

    monkeypatch.setattr(bt.subprocess, "run", fake_run)
    result = demo.execute_command("echo lab03-shell-ok")
    assert result["timed_out"] is True
    assert result["ok"] is False  # 超时不能当作正常完成


def test_dispatch_unknown_tool(demo: BasicTools) -> None:
    assert demo.dispatch("delete_file", "{}")["ok"] is False


def test_dispatch_invalid_json_arguments(demo: BasicTools) -> None:
    assert demo.dispatch("read_file", "not-json")["ok"] is False


def test_dispatch_non_object_arguments(demo: BasicTools) -> None:
    assert demo.dispatch("read_file", "[1, 2]")["ok"] is False


def test_dispatch_missing_field(demo: BasicTools) -> None:
    result = demo.dispatch("read_file", json.dumps({}))
    assert result["ok"] is False
    assert "缺少字段" in result["error"]


def test_dispatch_extra_field(demo: BasicTools) -> None:
    result = demo.dispatch("read_file", json.dumps({"path": "notes.txt", "force": True}))
    assert result["ok"] is False
    assert "多余字段" in result["error"]


def test_dispatch_wrong_type_field(demo: BasicTools) -> None:
    result = demo.dispatch("read_file", json.dumps({"path": 17}))
    assert result["ok"] is False
    assert "类型错误" in result["error"]


# ---------- D. Agent + Fake ReAct 闭环 ----------


def four_tool_script(tmp_path: Path) -> tuple[list[dict], Path]:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "notes.txt").write_text("工具结果必须回灌\n", encoding="utf-8")
    script = [
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c_list", "list_dir", {"path": "."})]},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c_read", "read_file", {"path": "notes.txt"})]},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c_write", "write_file", {"path": "report.txt", "content": "回灌报告"})]},
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c_cmd", "execute_command", {"command": "echo lab03-shell-ok"})]},
        {"role": "assistant", "content": "全部完成（Fake 预设）"},
    ]
    return script, root


def test_fake_four_tool_trajectory(tmp_path: Path) -> None:
    script, root = four_tool_script(tmp_path)
    fake = FakeLLMClient(script)
    agent = Agent(fake, BasicTools(root))

    result = agent.run("列目录、读说明、写报告、执行 echo 命令")

    assert result.status == "completed"
    assert result.iterations == 5          # 4 次工具 + 1 次最终回答
    assert fake.call_count == 5
    # 真实文件产物：Fake 声称完成不算数，文件必须落盘
    assert (root / "report.txt").read_text(encoding="utf-8") == "回灌报告"
    # 消息链：system user assistant(tool) tool ×4 assistant
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool",
                     "assistant", "tool", "assistant", "tool", "assistant"]
    # 第 2~5 次请求都携带了前一轮的 Observation
    for i in range(1, 5):
        prior_roles = [m["role"] for m in fake.requests[i]["messages"]]
        assert "tool" in prior_roles
    # assistant/tool 配对：每条 tool 消息的 id 与上一条 assistant 申请一致
    for i, message in enumerate(agent.messages):
        if message["role"] == "tool":
            assert message["tool_call_id"] == agent.messages[i - 1]["tool_calls"][0]["id"]
    # 第 3 次请求（写文件前）应已包含读取到的真实内容
    third = fake.requests[2]["messages"]
    tool_msgs = [m for m in third if m["role"] == "tool"]
    assert any("工具结果必须回灌" in m["content"] for m in tool_msgs)


def test_fake_snapshot_not_polluted(tmp_path: Path) -> None:
    """deepcopy 快照：第 1 次请求不应包含后来追加的 Observation。"""
    script, root = four_tool_script(tmp_path)
    fake = FakeLLMClient(script)
    agent = Agent(fake, BasicTools(root))
    agent.run("任务")
    first_roles = [m["role"] for m in fake.requests[0]["messages"]]
    assert first_roles == ["system", "user"]  # 只有初始任务


def test_iteration_limit_and_unconsumed_script(tmp_path: Path) -> None:
    script, root = four_tool_script(tmp_path)
    # 预设 5 个响应但上限 3：多余响应不允许被消费
    fake = FakeLLMClient(script)
    agent = Agent(fake, BasicTools(root), max_iterations=3)
    result = agent.run("任务")
    assert result.status == "iteration_limit"
    assert result.iterations == 3
    assert fake.call_count == 3
    assert len(fake._scripted) == 2  # 剩余脚本未被消费


def test_followup_keeps_history_and_resets_iterations(tmp_path: Path) -> None:
    script, root = four_tool_script(tmp_path)
    script.append({"role": "assistant", "content": "报告在 report.txt（Fake 预设）"})
    fake = FakeLLMClient(script)
    agent = Agent(fake, BasicTools(root))

    first = agent.run("四工具任务")
    assert first.iterations == 5
    follow = agent.run("报告文件叫什么？")
    assert follow.status == "completed"
    assert follow.iterations == 1  # 追问重新计数
    # 追问请求携带完整先前历史
    last = fake.requests[-1]["messages"]
    roles = [m["role"] for m in last]
    assert roles[0] == "system"
    assert "tool" in roles
    assert last[-1] == {"role": "user", "content": "报告文件叫什么？"}


def test_mixed_content_and_tool_calls_not_final(tmp_path: Path) -> None:
    """content 非空但存在 tool_calls：仍须执行工具，不能提前结束。"""
    root = tmp_path / "demo"
    root.mkdir()
    script = [
        {"role": "assistant", "content": "我先列一下目录", "tool_calls": [tool_call("c1", "list_dir", {"path": "."})]},
        {"role": "assistant", "content": "完成"},
    ]
    fake = FakeLLMClient(script)
    agent = Agent(fake, BasicTools(root))
    result = agent.run("任务")
    assert result.status == "completed"
    assert result.iterations == 2  # 若误把第一轮当最终回答则 iterations==1


def test_multiple_tool_calls_execute_in_order(tmp_path: Path) -> None:
    """一次响应申请先写后读：必须按顺序执行、按 ID 配对两条 Observation。"""
    root = tmp_path / "demo"
    root.mkdir()
    script = [
        {"role": "assistant", "content": None, "tool_calls": [
            tool_call("c_write", "write_file", {"path": "x.txt", "content": "数据"}),
            tool_call("c_read", "read_file", {"path": "x.txt"}),
        ]},
        {"role": "assistant", "content": "完成"},
    ]
    fake = FakeLLMClient(script)
    agent = Agent(fake, BasicTools(root))
    result = agent.run("先写后读")
    assert result.status == "completed"
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c_write", "c_read"]
    # 后写先读会失败：第二个 Observation 必须包含真实读取到的内容
    assert "数据" in tool_msgs[1]["content"]


def test_empty_final_content_raises_protocol_error(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    fake = FakeLLMClient([{"role": "assistant", "content": None}])
    agent = Agent(fake, BasicTools(root))
    with pytest.raises(ProtocolError):
        agent.run("任务")


def test_missing_tool_call_id_raises_protocol_error(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    bad = {"role": "assistant", "content": None,
           "tool_calls": [{"type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]}
    fake = FakeLLMClient([bad])
    agent = Agent(fake, BasicTools(root))
    with pytest.raises(ProtocolError):
        agent.run("任务")


def test_fake_exhausted_is_visible(tmp_path: Path) -> None:
    """脚本被消费完后继续调用必须显式失败，而不是默默返回成功。"""
    root = tmp_path / "demo"
    root.mkdir()
    fake = FakeLLMClient([{"role": "assistant", "content": "唯一回答"}])
    agent = Agent(fake, BasicTools(root), max_iterations=5)
    agent.run("任务")
    with pytest.raises(FakeExhaustedError):
        agent.run("再来一次")  # 没有预设响应了


def test_empty_user_text_rejected(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    agent = Agent(FakeLLMClient([]), BasicTools(root))
    with pytest.raises(ValueError):
        agent.run("   ")
