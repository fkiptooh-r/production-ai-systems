"""Tests for the Ollama client using a fake HTTP transport, so no server or network is needed."""

import json

import httpx
import pytest

from foundations.llm import LLMError, stream_chat


def ndjson(*chunks: dict) -> str:
    return "\n".join(json.dumps(chunk) for chunk in chunks) + "\n"


HELLO_REPLY = ndjson(
    {"message": {"content": "Hel"}, "done": False},
    {"message": {"content": "lo"}, "done": False},
    {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 2},
)


def fake_client(*outcomes: httpx.Response | Exception) -> httpx.Client:
    """A client whose requests get these outcomes in order: a response, or a raised error."""
    remaining = list(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.Client(transport=httpx.MockTransport(handler))


def ok() -> httpx.Response:
    return httpx.Response(200, text=HELLO_REPLY)


class SleepRecorder:
    """Stands in for time.sleep: records the delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def test_streams_tokens_in_order_and_returns_counts():
    tokens: list[str] = []
    result = stream_chat("hi", client=fake_client(ok()), on_token=tokens.append)

    assert tokens == ["Hel", "lo"]
    assert result.text == "Hello"
    assert result.input_tokens == 12
    assert result.output_tokens == 2


def test_retries_after_connection_error_then_succeeds():
    sleep = SleepRecorder()
    client = fake_client(httpx.ConnectError("refused"), ok())

    result = stream_chat("hi", client=client, sleep=sleep)

    assert result.text == "Hello"
    assert sleep.delays == [1]


def test_backoff_doubles_each_attempt():
    sleep = SleepRecorder()
    client = fake_client(
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.Response(503),
        ok(),
    )

    stream_chat("hi", client=client, sleep=sleep, max_retries=3)

    assert sleep.delays == [1, 2, 4]


def test_gives_up_after_max_retries():
    sleep = SleepRecorder()
    client = fake_client(*[httpx.ConnectError("refused")] * 3)

    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        stream_chat("hi", client=client, sleep=sleep, max_retries=2)
    assert sleep.delays == [1, 2]


def test_client_errors_are_not_retried():
    sleep = SleepRecorder()
    client = fake_client(httpx.Response(404, json={"error": "model 'nope' not found"}))

    with pytest.raises(LLMError, match="404"):
        stream_chat("hi", model="nope", client=client, sleep=sleep)
    assert sleep.delays == []


def test_error_inside_stream_raises():
    client = fake_client(httpx.Response(200, text=ndjson({"error": "out of memory"})))

    with pytest.raises(LLMError, match="out of memory"):
        stream_chat("hi", client=client)


def test_sends_messages_format_and_options_to_ollama():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return ok()

    client = httpx.Client(transport=httpx.MockTransport(handler))
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    schema = {"type": "object"}

    stream_chat(messages, client=client, response_format=schema, options={"temperature": 0})

    assert sent[0]["messages"] == messages
    assert sent[0]["format"] == schema
    assert sent[0]["options"] == {"temperature": 0}


def test_collects_tool_calls_and_sends_tool_definitions():
    sent: list[dict] = []
    tool_call = {"function": {"name": "get_listing", "arguments": {"listing_id": "L007"}}}
    body = ndjson(
        {"message": {"content": "", "tool_calls": [tool_call]}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 5, "eval_count": 3},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, text=body)

    tools = [{"type": "function", "function": {"name": "get_listing"}}]
    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = stream_chat("details of L007", client=client, tools=tools)

    assert sent[0]["tools"] == tools
    assert result.tool_calls == [tool_call]
    assert result.text == ""


def test_stream_without_done_message_raises():
    partial = ndjson({"message": {"content": "Hel"}, "done": False})
    client = fake_client(httpx.Response(200, text=partial))

    with pytest.raises(LLMError, match="without a final"):
        stream_chat("hi", client=client)
