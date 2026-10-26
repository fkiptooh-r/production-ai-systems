import json

from foundations.tools import (
    SearchListingsArgs,
    execute_tool,
    load_listings,
    search_listings,
    tool_schemas,
)

LISTINGS = load_listings()


def ids(results: list[dict]) -> list[str]:
    return [listing["id"] for listing in results]


def test_search_applies_every_filter_and_sorts_cheapest_first():
    args = SearchListingsArgs(bedrooms=2, max_price=40000, parking=True)
    assert ids(search_listings(args, LISTINGS)) == ["L010", "L011"]


def test_location_match_ignores_case():
    args = SearchListingsArgs(location="kilimani")
    assert ids(search_listings(args, LISTINGS)) == ["L011", "L001"]


def test_search_returns_at_most_five():
    assert len(search_listings(SearchListingsArgs(), LISTINGS)) == 5


def test_execute_runs_a_valid_call():
    output = json.loads(execute_tool("get_listing", {"listing_id": "L007"}, LISTINGS))
    assert output["location"] == "Ngong"


def test_execute_reports_unknown_tool_instead_of_crashing():
    output = json.loads(execute_tool("delete_everything", {}, LISTINGS))
    assert "unknown tool" in output["error"]


def test_execute_reports_invalid_arguments_instead_of_crashing():
    output = json.loads(execute_tool("search_listings", {"max_price": -5}, LISTINGS))
    assert "invalid arguments" in output["error"]


def test_execute_coerces_numeric_strings():
    # Small models often send "40000" instead of 40000; Pydantic converts it.
    output = json.loads(execute_tool("search_listings", {"max_price": "10000"}, LISTINGS))
    assert {listing["id"] for listing in output} == {"L002", "L005", "L015"}


def test_string_null_means_filter_not_set():
    # Regression: llama3.2:3b sent {"location": "null", "bedrooms": 3}, which searched
    # for a neighbourhood literally called "null" and found nothing.
    args = SearchListingsArgs.model_validate(
        {"location": "null", "max_price": "None", "parking": "", "bedrooms": 3}
    )
    assert args == SearchListingsArgs(bedrooms=3)
    assert ids(search_listings(args, LISTINGS)) == ["L007", "L012"]


def test_get_listing_with_null_id_is_an_error_not_a_lookup():
    output = json.loads(execute_tool("get_listing", {"listing_id": "null"}, LISTINGS))
    assert "invalid arguments" in output["error"]


def test_schemas_describe_every_tool():
    names = {schema["function"]["name"] for schema in tool_schemas()}
    assert names == {"search_listings", "get_listing"}
