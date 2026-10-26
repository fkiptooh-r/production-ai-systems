"""Extract structured house-listing data from free text with an LLM.

Three layers of protection against bad model output:
1. Ollama's "format" constrains the reply to our JSON schema.
2. Pydantic validates types and ranges (e.g. no negative prices).
3. If validation still fails, the error is sent back to the model to fix (a "repair" turn).
"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from foundations.llm import DEFAULT_MODEL, ChatResult, stream_chat

PropertyType = Literal["apartment", "house", "bedsitter", "studio", "single_room", "other"]


class Listing(BaseModel):
    # Every field is required but may be null. Fields with defaults are left out of the
    # schema's "required" list, which lets the model skip them silently instead of
    # answering, and that dropped bedrooms accuracy to 10% in the first eval.
    property_type: PropertyType
    bedrooms: int | None = Field(ge=0, le=20)
    price_per_month: int | None = Field(ge=0)
    location: str | None
    parking: bool | None
    furnished: bool | None
    amenities: list[str]


SYSTEM_PROMPT = """You extract facts from rental listings into JSON.

Rules:
- Use null for anything the listing does not state. Never guess.
- price_per_month: a whole number with no currency or commas. "45k" means 45000.
  If the price is negotiable or not given, use null.
- bedrooms: a number. A bedsitter, studio or single room has 0 bedrooms.
- location: the neighbourhood or area name only, e.g. "Kilimani".
- parking / furnished: true or false only if the listing says so, otherwise null.
- amenities: short lowercase phrases such as "borehole", "gym", "security"."""


class ExtractionError(Exception):
    """The model never produced a valid Listing."""


ChatFn = Callable[..., ChatResult]


def extract_listing(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    max_repairs: int = 2,
    chat: ChatFn = stream_chat,
) -> Listing:
    """Turn listing text into a validated Listing, asking the model to repair invalid JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    for attempt in range(max_repairs + 1):
        result = chat(
            messages,
            model=model,
            response_format=Listing.model_json_schema(),
            options={"temperature": 0},  # extraction should be repeatable, not creative
        )
        try:
            return Listing.model_validate_json(result.text)
        except ValidationError as exc:
            if attempt == max_repairs:
                raise ExtractionError(
                    f"no valid listing after {attempt + 1} attempts: {exc}"
                ) from exc
            messages += [
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": f"That JSON is invalid:\n{exc}\nReply with corrected JSON only.",
                },
            ]
    raise AssertionError("unreachable")
