"""Answer generation with a fake model and a fake search: no Ollama, no database."""

from shared.llm import ChatResult

from rag.answer_eval import Judged, summarize
from rag.chunking import Chunk
from rag.evaluate import Question
from rag.generate import agency_name, answer, build_messages, cited_numbers, is_abstention

PETS = Chunk(
    "ruaka-rentals",
    "tenancy-policy.md",
    "Ruaka Rentals Tenancy Terms > Pets",
    5,
    "No pets are allowed in any Ruaka Rentals unit. This includes cats, dogs and birds.",
)
RENT = Chunk(
    "ruaka-rentals",
    "tenancy-policy.md",
    "Ruaka Rentals Tenancy Terms > Rent payment",
    2,
    "Rent is due on the 1st of every month.",
)


def fake_search(question, tenant_id, k):
    return [PETS, RENT][:k]


def fake_chat(reply: str, seen: list | None = None):
    def chat(messages, **kwargs):
        if seen is not None:
            seen.append((messages, kwargs))
        for token in reply.split(" "):
            kwargs.get("on_token", lambda _: None)(token)
        return ChatResult(text=reply, input_tokens=100, output_tokens=12, seconds=1.5)

    return chat


def question(answerable=True) -> Question:
    if answerable:
        return Question(
            "rr-07",
            "ruaka-rentals",
            "paraphrase",
            "Can I keep a cat?",
            "tenancy-policy.md",
            "cats, dogs and birds",
            None,
        )
    return Question("rr-20", "ruaka-rentals", "not_in_data", "Is there a gym?", None, None, "gym")


def test_agency_name_from_tenant_id():
    assert agency_name("ruaka-rentals") == "Ruaka Rentals"


def test_prompt_numbers_the_sources_and_names_the_agency():
    system, user = build_messages("Can I keep a cat?", "ruaka-rentals", [PETS, RENT])

    assert "Ruaka Rentals" in system["content"]
    assert "I don't know" in system["content"]
    assert "[1] (tenancy-policy.md > Pets)\nNo pets are allowed" in user["content"]
    assert "[2] (tenancy-policy.md > Rent payment)" in user["content"]
    assert user["content"].endswith("Question: Can I keep a cat?")


def test_citations_are_parsed_in_order_without_duplicates_or_invalid_numbers():
    assert cited_numbers("No [2]. Also [1][2]. See [9] and [0].", available=2) == [2, 1]
    assert cited_numbers("No citations here.", available=2) == []


def test_abstention_detection_handles_curly_apostrophes():
    assert is_abstention("I don't know.")
    assert is_abstention("I don’t know.")
    assert not is_abstention("No pets are allowed [1].")


def test_answer_maps_citations_back_to_chunks():
    seen = []
    tokens = []
    result = answer(
        "Can I keep a cat?",
        "ruaka-rentals",
        fake_search,
        chat=fake_chat("No, no pets are allowed [1].", seen),
        on_token=tokens.append,
    )

    assert result.cited == [PETS]
    assert result.retrieved == [PETS, RENT]
    assert not result.abstained
    assert tokens  # streamed to the caller
    assert seen[0][1]["options"] == {"temperature": 0}  # deterministic, for fair evals


def test_judging_a_grounded_answer_and_a_correct_abstention():
    good = answer(
        "Can I keep a cat?",
        "ruaka-rentals",
        fake_search,
        chat=fake_chat("No: cats, dogs and birds are not allowed [1]."),
    )
    abstain = answer(
        "Is there a gym?", "ruaka-rentals", fake_search, chat=fake_chat("I don't know.")
    )

    report = summarize([Judged(question(), good), Judged(question(False), abstain)])

    assert (report.grounded, report.phrase_in_answer, report.false_abstentions) == (1, 1, 0)
    assert report.correct_abstentions == 1
    assert report.failures == []


def test_judging_catches_wrong_citations_and_made_up_answers():
    wrong_cite = answer(
        "Can I keep a cat?",
        "ruaka-rentals",
        fake_search,
        chat=fake_chat("Yes, rent is due on the 1st [2]."),
    )
    invented = answer(
        "Is there a gym?",
        "ruaka-rentals",
        fake_search,
        chat=fake_chat("Yes, there is a gym on the ground floor."),
    )

    report = summarize([Judged(question(), wrong_cite), Judged(question(False), invented)])

    assert report.grounded == 0  # cited the rent chunk, which doesn't contain the answer
    assert report.correct_abstentions == 0
    assert len(report.failures) == 2
