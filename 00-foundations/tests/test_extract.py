import json

import pytest
from pydantic import ValidationError
from shared.llm import ChatResult

from foundations.extract import ExtractionError, Listing, extract_listing
from foundations.extract_eval import score

VALID_JSON = json.dumps(
    {
        "property_type": "apartment",
        "bedrooms": 2,
        "price_per_month": 45000,
        "location": "Kilimani",
        "parking": True,
        "furnished": None,
        "amenities": ["water 24/7"],
    }
)


class FakeChat:
    """Stands in for stream_chat: returns scripted replies and records what it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict], **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(text=self.replies.pop(0), input_tokens=0, output_tokens=0, seconds=0)


def test_valid_reply_becomes_a_listing():
    listing = extract_listing("2BR in Kilimani, 45k", chat=FakeChat(VALID_JSON))

    assert listing.bedrooms == 2
    assert listing.price_per_month == 45000
    assert listing.parking is True
    assert listing.furnished is None


def test_invalid_reply_is_repaired_with_the_error_message():
    bad = json.dumps({"property_type": "castle", "bedrooms": -1})
    chat = FakeChat(bad, VALID_JSON)

    listing = extract_listing("2BR in Kilimani, 45k", chat=chat)

    assert listing.location == "Kilimani"
    repair_request = chat.calls[1][-1]["content"]
    assert "invalid" in repair_request
    assert "property_type" in repair_request


def test_gives_up_after_max_repairs():
    chat = FakeChat("not json", "still not json", "nope")

    with pytest.raises(ExtractionError, match="3 attempts"):
        extract_listing("anything", chat=chat, max_repairs=2)


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"property_type": "castle"},
        {"price_per_month": -100},
        {"bedrooms": 50},
    ],
)
def test_listing_rejects_impossible_values(bad_fields):
    with pytest.raises(ValidationError):
        Listing(**{**json.loads(VALID_JSON), **bad_fields})


def test_every_field_is_required_so_the_model_cannot_skip_it():
    # Regression test: optional fields let the model omit keys (bedrooms was 10% accurate).
    schema = Listing.model_json_schema()
    assert set(schema["required"]) == set(Listing.model_fields)

    missing_bedrooms = {k: v for k, v in json.loads(VALID_JSON).items() if k != "bedrooms"}
    with pytest.raises(ValidationError):
        Listing(**missing_bedrooms)


def test_score_compares_text_case_insensitively_and_fails_missing_listings():
    expected = {
        "property_type": "apartment",
        "bedrooms": 2,
        "price_per_month": 45000,
        "location": "Kilimani",
        "parking": True,
        "furnished": None,
    }
    listing = Listing.model_validate_json(VALID_JSON.replace("Kilimani", "kilimani "))

    assert all(score(expected, listing).values())
    assert not any(score(expected, None).values())
