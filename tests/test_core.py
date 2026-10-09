"""Offline tests (no network, no LLM):   pip install pytest && python -m pytest tests"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402
from check_citations import check  # noqa: E402
from research import slugify  # noqa: E402


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.time, "sleep", calls.append)
    return calls


def flaky(failures, retry_after=None):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise tools.RetryableError("temporary", retry_after=retry_after)
        return "ok"

    return fn, state


def test_retry_succeeds_with_capped_exponential_backoff(sleeps):
    fn, state = flaky(3)
    assert tools.with_retry(fn, attempts=5, base=1.0, cap=3.0) == "ok"
    assert state["calls"] == 4
    assert 1.0 <= sleeps[0] <= 2.0 and 2.0 <= sleeps[1] <= 3.0 and sleeps[2] == 3.0


def test_retry_honours_retry_after(sleeps):
    fn, _ = flaky(1, retry_after=7)
    assert tools.with_retry(fn, attempts=3, cap=30.0) == "ok"
    assert sleeps == [7]


def test_retry_gives_up_without_sleeping_after_last_attempt(sleeps):
    fn, state = flaky(10)
    with pytest.raises(tools.RetryableError):
        tools.with_retry(fn, attempts=3)
    assert state["calls"] == 3 and len(sleeps) == 2


def test_retry_does_not_retry_other_errors(sleeps):
    def boom():
        raise ValueError("bug")

    with pytest.raises(ValueError):
        tools.with_retry(boom)
    assert sleeps == []


def test_arxiv_empty_query_does_not_call_network(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: pytest.fail("network called"))
    assert tools.arxiv_search.invoke({"query": "  ' : AND OR  "}) == "NO RESULTS"


def test_exa_key_is_redacted(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")

    def fail(*args, **kwargs):
        raise RuntimeError("bad url https://mcp.exa.ai/mcp?exaApiKey=secret-key-123")

    monkeypatch.setattr(tools.httpx, "post", fail)
    result = tools.web_search.invoke({"query": "world models"})
    assert result.startswith("ERROR:") and "secret-key-123" not in result


@pytest.mark.parametrize("topic, expected", [
    ("survey about world model", "survey-about-world-model"),
    ("../../x", "x"),
    ("", "topic"),
    ("   ", "topic"),
])
def test_slugify(topic, expected):
    assert slugify(topic) == expected


def test_slugify_truncates():
    assert len(slugify("a" * 200)) == 60


SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "", "source": "arxiv"},
    {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002", "title": "B", "date": "",
     "source": "hf-search"},
    {"n": 3, "id": "page", "url": "https://example.org/page", "title": "C", "date": "", "source": "web"},
]
REFS = ("## References\n[1] A. arxiv. https://arxiv.org/abs/2501.00001\n"
        "[2] B. hf-search. https://huggingface.co/papers/2501.00002\n[3] C. web. https://example.org/page\n")


def test_valid_report_passes():
    assert check("# T\nClaim [1], more [2-3].\n\n" + REFS, SOURCES) == []


def test_grouped_citations_are_expanded():
    assert check("# T\nClaim [1, 2, 3].\n\n" + REFS, SOURCES) == []


def test_missing_and_uncited_sources():
    problems = check("# T\nClaim [1][4].\n\n" + REFS, SOURCES)
    assert "[4] cited but missing from sources.json" in problems
    assert "source [2] never cited" in problems


def test_bundled_reference_line_is_rejected():
    refs = REFS.replace("https://example.org/page", "https://example.org/page; https://example.org/other")
    assert any("exactly one URL" in p for p in check("# T\n[1][2][3]\n\n" + refs, SOURCES))


def test_missing_references_heading():
    assert "report is missing the ## References section" in check("# T\n[1][2][3]\n", SOURCES)


def test_family_must_match_url():
    sources = [dict(SOURCES[0], url="https://arxiv.org/html/2501.00001v1"), *SOURCES[1:]]
    refs = REFS.replace("https://arxiv.org/abs/2501.00001", "https://arxiv.org/html/2501.00001v1")
    assert any("does not match that family" in p for p in check("# T\n[1][2][3]\n\n" + refs, sources))


def test_url_with_brackets_is_rejected():
    sources = [*SOURCES[:2], dict(SOURCES[2], url="https://example.org/a_(b)")]
    refs = REFS.replace("https://example.org/page", "https://example.org/a_(b)")
    assert any("brackets" in p for p in check("# T\n[1][2][3]\n\n" + refs, sources))


def test_needs_three_families():
    sources = [dict(s, source="web") if s["n"] == 2 else s for s in SOURCES]
    assert any("source families" in p for p in check("# T\n[1][2][3]\n\n" + REFS, sources))


def test_citations_in_code_and_links_are_ignored():
    problems = check("# T\n[1][2] `[3]` and [3](https://example.org/page)\n\n" + REFS, SOURCES)
    assert problems == ["source [3] never cited"]


def test_resolver_then_finalizer_produce_a_valid_report():
    from finalize_citations import finalize
    from resolve_citations import resolve

    draft = "# T\nA [@2501.00002] then B [@2501.00001, @page] and `[@not-a-citation]`.\n"
    report, problems = resolve(draft, SOURCES)
    assert problems == [] and "A [2] then B [1][3]" in report and "`[@not-a-citation]`" in report
    report, sources, problems = finalize(report, SOURCES)
    assert problems == [] and check(report, sources) == []
    assert sources[0]["url"] == "https://huggingface.co/papers/2501.00002"


def test_resolver_rejects_unknown_ids_and_numeric_citations():
    from resolve_citations import resolve

    _, problems = resolve("# T\n[@nope] and [2]\n", SOURCES)
    assert any("[@nope]" in p for p in problems) and any("numeric citation [2]" in p for p in problems)
