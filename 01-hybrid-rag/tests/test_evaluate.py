import pytest

from rag.chunking import Chunk
from rag.evaluate import Question, evaluate, first_relevant_rank, is_relevant


def chunk(tenant="t1", source="policy.md", text="The deposit is one month's rent.", index=0):
    return Chunk(
        tenant_id=tenant, source=source, section="Policy > Deposit", index=index, text=text
    )


def question(qid="q1", kind="direct", tenant="t1", source="policy.md", answer="one month's rent"):
    return Question(
        id=qid,
        tenant=tenant,
        kind=kind,
        question="How much is the deposit?",
        source=source,
        answer=answer,
        absent=None,
    )


RIGHT = chunk()
WRONG_TENANT = chunk(tenant="t2")  # same words, another agency: a confidently wrong answer
WRONG_SOURCE = chunk(source="viewings.md")
NO_ANSWER = chunk(text="Pets are not allowed.")


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(RIGHT, True), (WRONG_TENANT, False), (WRONG_SOURCE, False), (NO_ANSWER, False)],
)
def test_relevance_needs_tenant_source_and_phrase(candidate, expected):
    assert is_relevant(candidate, question()) is expected


def test_phrase_match_ignores_case_and_whitespace():
    messy = chunk(text="The deposit is ONE   month's\nrent.")
    assert is_relevant(messy, question())


def test_phrase_can_match_in_the_heading_trail():
    assert is_relevant(chunk(text="Two months."), question(answer="policy > deposit"))


def test_rank_is_one_based_and_none_when_missing():
    q = question()
    assert first_relevant_rank([NO_ANSWER, WRONG_TENANT, RIGHT], q) == 3
    assert first_relevant_rank([NO_ANSWER, WRONG_TENANT], q) is None


def test_recall_and_mrr():
    questions = [question("a"), question("b"), question("c", kind="exact"), question("d")]
    results = {
        "a": [RIGHT],  # rank 1
        "b": [NO_ANSWER, RIGHT],  # rank 2
        "c": [WRONG_TENANT] * 5 + [RIGHT],  # rank 6: outside the top 5 → a miss
        "d": [NO_ANSWER],  # miss
    }
    order = iter("abcd")

    report = evaluate(lambda text, tenant, k: results[next(order)], questions, k=5)

    assert report.recall == pytest.approx(2 / 4)
    assert report.mrr == pytest.approx((1 + 1 / 2) / 4)
    assert report.by_kind["direct"] == (3, pytest.approx(2 / 3), pytest.approx(1.5 / 3))
    assert report.by_kind["exact"] == (1, 0.0, 0.0)
    assert report.misses == ["c", "d"]


def test_not_in_data_questions_are_left_out_of_retrieval_scores():
    unanswerable = Question(
        id="x",
        tenant="t1",
        kind="not_in_data",
        question="Gym?",
        source=None,
        answer=None,
        absent="gym",
    )
    report = evaluate(lambda text, tenant, k: [RIGHT], [question(), unanswerable])
    assert report.questions == 1
    assert report.recall == 1.0
