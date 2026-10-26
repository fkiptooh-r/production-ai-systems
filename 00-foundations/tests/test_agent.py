import json

import pytest

from foundations.agent import AgentError, run_agent
from foundations.llm import ChatResult
from foundations.tool_eval import grade
from foundations.tools import load_listings

LISTINGS = load_listings()


def call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def reply(text: str = "", *tool_calls: dict) -> ChatResult:
    return ChatResult(text, 0, 0, 0, tool_calls=list(tool_calls))


class ScriptedModel:
    """Stands in for stream_chat: returns scripted replies and records each conversation."""

    def __init__(self, *replies: ChatResult) -> None:
        self.replies = list(replies)
        self.conversations: list[list[dict]] = []

    def __call__(self, messages: list[dict], **kwargs) -> ChatResult:
        self.conversations.append(list(messages))
        return self.replies.pop(0)


def test_answers_directly_when_no_tool_is_needed():
    model = ScriptedModel(reply("A bedsitter is a single room with a kitchenette."))

    result = run_agent("What is a bedsitter?", chat=model, listings=LISTINGS)

    assert result.answer.startswith("A bedsitter")
    assert result.steps == []


def test_runs_tool_and_sends_result_back_before_answering():
    model = ScriptedModel(
        reply("", call("search_listings", bedrooms=2, max_price=40000, parking=True)),
        reply("L010 in Ruaka for 35,000 and L011 in Kilimani for 38,000."),
    )

    result = run_agent("2BR under 40k with parking", chat=model, listings=LISTINGS)

    assert [step.name for step in result.steps] == ["search_listings"]
    tool_message = model.conversations[1][-1]
    assert tool_message["role"] == "tool"
    assert [listing["id"] for listing in json.loads(tool_message["content"])] == ["L010", "L011"]
    assert "L010" in result.answer


def test_tool_errors_go_back_to_the_model_so_it_can_recover():
    model = ScriptedModel(
        reply("", call("search_listings", max_price=-5)),
        reply("", call("search_listings", max_price=5000)),
        reply("L005 in Githurai 45 for 4,000."),
    )

    result = run_agent("cheap places", chat=model, listings=LISTINGS)

    assert "invalid arguments" in result.steps[0].output
    assert result.answer.startswith("L005")


def test_stops_a_model_that_never_stops_calling_tools():
    model = ScriptedModel(*[reply("", call("get_listing", listing_id="L001"))] * 3)

    with pytest.raises(AgentError, match="3 steps"):
        run_agent("loop forever", chat=model, listings=LISTINGS, max_steps=3)


@pytest.mark.parametrize(
    ("tool_calls", "expected_grade"),
    [
        ([call("search_listings", max_price="40000", bedrooms=2)], (True, True)),
        ([call("search_listings", max_price=40000)], (True, False)),
        ([call("get_listing", listing_id="L001")], (False, False)),
        ([], (False, False)),
    ],
)
def test_grade_first_call(tool_calls, expected_grade):
    expected = {"tool": "search_listings", "arguments": {"max_price": 40000, "bedrooms": 2}}
    assert grade(expected, tool_calls) == expected_grade


def test_grade_rewards_not_calling_a_tool_when_none_is_needed():
    expected = {"tool": None, "arguments": None}
    assert grade(expected, []) == (True, True)
    assert grade(expected, [call("search_listings")]) == (False, False)
