"""Lab03 LLM 客户端包。

- :class:`GLMClient` — 真实 HTTP 客户端（手写 httpx 请求/解析）。
- :class:`FakeLLMClient` — 脚本化假客户端，离线测试控制流。
- :class:`LLMClient` — Protocol 契约，真实与 Fake 可互换。
"""

from app.llm.base import LLMClient
from app.llm.fake_llm import FakeExhaustedError, FakeLLMClient
from app.llm.glm_client import API_URL, GLMClient, LLMError, ProtocolError

__all__ = [
    "API_URL",
    "GLMClient",
    "LLMClient",
    "LLMError",
    "ProtocolError",
    "FakeLLMClient",
    "FakeExhaustedError",
]
