"""Tools the model can call, and the code that runs them safely.

Each tool is a Pydantic model for its arguments plus a plain Python function.
The model never runs code itself: it only *asks* for a call, and execute_tool
validates the arguments before anything runs.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

LISTINGS_FILE = Path(__file__).resolve().parents[2] / "data" / "listings.json"

# Small models often write the *string* "null" for a filter they don't want.
# Taken literally, that would search for a neighbourhood called "null".
NULL_STRINGS = {"", "null", "none"}


def load_listings(path: Path = LISTINGS_FILE) -> list[dict]:
    return json.loads(path.read_text())


class ToolArgs(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def null_strings_mean_not_set(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
            return None
        return value


class SearchListingsArgs(ToolArgs):
    location: str | None = Field(None, description="Neighbourhood name, e.g. Kilimani")
    max_price: int | None = Field(None, ge=0, description="Highest monthly rent, a whole number")
    bedrooms: int | None = Field(None, ge=0, le=20, description="Exact number of bedrooms")
    parking: bool | None = Field(None, description="true if the user requires parking")


class GetListingArgs(ToolArgs):
    listing_id: str = Field(description="Listing id such as L007")


def search_listings(args: SearchListingsArgs, listings: list[dict]) -> list[dict]:
    """Listings matching every filter that was given, cheapest first, at most 5."""
    matches = [
        listing
        for listing in listings
        if (args.location is None or listing["location"].lower() == args.location.lower())
        and (args.max_price is None or listing["price_per_month"] <= args.max_price)
        and (args.bedrooms is None or listing["bedrooms"] == args.bedrooms)
        and (args.parking is not True or listing["parking"])
    ]
    return sorted(matches, key=lambda listing: listing["price_per_month"])[:5]


def get_listing(args: GetListingArgs, listings: list[dict]) -> dict:
    for listing in listings:
        if listing["id"].lower() == args.listing_id.strip().lower():
            return listing
    return {"error": f"no listing with id {args.listing_id}"}


@dataclass
class Tool:
    description: str
    args_model: type[BaseModel]
    run: Callable[[BaseModel, list[dict]], object]


TOOLS: dict[str, Tool] = {
    "search_listings": Tool(
        "Search rental listings. Only set the filters the user actually asked for.",
        SearchListingsArgs,
        search_listings,
    ),
    "get_listing": Tool(
        "Get full details of one listing by its id.",
        GetListingArgs,
        get_listing,
    ),
}


def tool_schemas() -> list[dict]:
    """Tool definitions in the format the model expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": tool.args_model.model_json_schema(),
            },
        }
        for name, tool in TOOLS.items()
    ]


def execute_tool(name: str, arguments: dict, listings: list[dict]) -> str:
    """Run a tool call and return JSON for the model.

    Mistakes (unknown tool, bad arguments) come back as an error message instead of
    raising, so the model can read what went wrong and try again.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name!r}, available: {sorted(TOOLS)}"})
    try:
        args = tool.args_model.model_validate(arguments)
    except ValidationError as exc:
        return json.dumps({"error": f"invalid arguments for {name}: {exc}"})
    return json.dumps(tool.run(args, listings))
