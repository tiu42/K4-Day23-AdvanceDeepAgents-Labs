"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree  # arXiv answers with Atom XML

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_ARXIV_MIN_INTERVAL = 3.0
_arxiv_last_call = 0.0
_arxiv_call_lock = threading.Lock()
_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _http_get(url, params, *, source, attempts=5, base=1.0, cap=30.0,
              before_request=None):
    def request():
        if before_request is not None:
            before_request()
        try:
            response = httpx.get(url, params=params, timeout=30.0)
        except httpx.TransportError as exc:
            raise RetryableError(str(exc)) from exc
        if response.status_code in _RETRYABLE_STATUS_CODES:
            retry_after = response.headers.get("Retry-After")
            try:
                retry_after = (
                    float(retry_after) if retry_after is not None else None
                )
            except ValueError:
                retry_after = None
            raise RetryableError(
                f"{source} returned HTTP {response.status_code}",
                retry_after=retry_after,
            )
        response.raise_for_status()
        return response

    return with_retry(request, attempts=attempts, base=base, cap=cap)


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again (other exceptions propagate).

    Wait Retry-After seconds when the server sent it, else base * 2**attempt plus random jitter; never more than cap.
    After the last attempt the error is re-raised without sleeping.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None:
                delay = min(e.retry_after, cap)
            else:
                delay = min(cap, base * 2 ** attempt + random.uniform(0, base))
            time.sleep(delay)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few keywords (all must match), newest first. Use 2-5 plain keywords.

    Returns a JSON list of {id, url, published, title, summary} records (url = https://arxiv.org/abs/<id>),
    or NO RESULTS / ERROR: ... when arXiv has nothing or keeps failing.
    """
    try:
        terms = re.findall(r"[\w-]+", query)
        terms = [t for t in terms if t.upper() not in ("AND", "OR", "ANDNOT")]
        if not terms:
            return "NO RESULTS"  # nothing left to search: do not call the network
        search_query = " AND ".join(f"all:{t}" for t in terms)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max(1, min(int(max_results), 30)),
        }

        def respect_arxiv_interval():
            global _arxiv_last_call
            with _arxiv_call_lock:
                elapsed = time.monotonic() - _arxiv_last_call
                if elapsed < _ARXIV_MIN_INTERVAL:
                    time.sleep(_ARXIV_MIN_INTERVAL - elapsed)
                _arxiv_last_call = time.monotonic()

        response = _http_get(
            ARXIV_URL,
            params,
            source="arXiv",
            attempts=7,
            cap=60.0,
            before_request=respect_arxiv_interval,
        )
        root = xml.etree.ElementTree.fromstring(response.text)
        namespace = "{http://www.w3.org/2005/Atom}"
        records = []
        for entry in root.findall(f"{namespace}entry"):
            raw_id = (entry.findtext(f"{namespace}id") or "").strip()
            if "/abs/" not in raw_id:
                continue
            paper_id = raw_id.split("/abs/", 1)[1].rsplit("/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            title = re.sub(
                r"\s+", " ", entry.findtext(f"{namespace}title") or ""
            ).strip()
            summary = re.sub(
                r"\s+", " ", entry.findtext(f"{namespace}summary") or ""
            ).strip()[:600]
            published = (entry.findtext(f"{namespace}published") or "").strip()
            records.append(
                {
                    "id": paper_id,
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "published": published[:10],
                    "title": title,
                    "summary": summary,
                }
            )
        return "NO RESULTS" if not records else json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def _hf_record(paper, *, prefer_ai_summary=False):
    paper_id = paper.get("id")
    if not paper_id:
        return None
    summary = paper.get("summary") or ""
    if prefer_ai_summary:
        summary = paper.get("ai_summary") or summary
    summary = re.sub(r"\s+", " ", str(summary)).strip()[:600]
    title = re.sub(r"\s+", " ", str(paper.get("title") or "")).strip()
    published = str(paper.get("publishedAt") or "")[:10]
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": published,
        "title": title,
        "summary": summary,
        "upvotes": paper.get("upvotes", 0),
        "github": paper.get("githubRepo"),
        "stars": paper.get("githubStars", 0),
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""

    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        response = _http_get(HF_DAILY_URL, params, source="Hugging Face")
        items = response.json()
        records = []
        for item in items:
            paper = item.get("paper") or {}
            record = _hf_record(paper)
            if record is not None:
                records.append(record)
        if keyword:
            keyword = keyword.casefold()
            records = [
                record
                for record in records
                if keyword in f"{record['title']} {record['summary']}".casefold()
            ]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return "NO RESULTS" if not records else json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face Papers by topic (relevance-ranked). Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} (url = https://huggingface.co/papers/<id>),
    or NO RESULTS / ERROR: ...."""
    try:
        params = {"q": query, "limit": max(1, min(int(limit), 50))}
        response = _http_get(HF_SEARCH_URL, params, source="Hugging Face")
        items = response.json()
        records = []
        for item in items:
            paper = item.get("paper") or {}
            record = _hf_record(paper, prefer_ai_summary=True)
            if record is not None:
                records.append(record)
        return "NO RESULTS" if not records else json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def _redact_api_key(value):
    api_key = os.getenv("EXA_API_KEY")
    return value.replace(api_key, "[REDACTED]") if api_key else value


def _exa_payloads(response):
    payloads = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                payloads.append(json.loads(data))
    if not payloads and response.text.strip():
        payloads.append(response.json())
    return payloads


def _exa_is_rate_limited(meta):
    if not isinstance(meta, dict):
        return False
    return any(
        "ratelimit" in re.sub(r"[^a-z]", "", str(key).casefold()) and value
        for key, value in meta.items()
    )


def _exa_call(name, arguments):
    api_key = os.getenv("EXA_API_KEY")
    params = {"exaApiKey": api_key} if api_key else None
    request_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }

    def request():
        try:
            response = httpx.post(
                EXA_URL,
                params=params,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=60.0,
            )
        except httpx.TransportError as exc:
            raise RetryableError(_redact_api_key(str(exc))) from exc
        if response.status_code in _RETRYABLE_STATUS_CODES:
            retry_after = response.headers.get("Retry-After")
            try:
                retry_after = (
                    float(retry_after) if retry_after is not None else None
                )
            except ValueError:
                retry_after = None
            raise RetryableError(
                f"Exa returned HTTP {response.status_code}",
                retry_after=retry_after,
            )
        response.raise_for_status()
        payloads = _exa_payloads(response)
        if not payloads:
            raise ValueError("Exa returned an empty response")
        for payload in payloads:
            if payload.get("error") is not None:
                raise ValueError(f"Exa JSON-RPC error: {payload['error']}")
            result = payload.get("result")
            if isinstance(result, dict):
                if _exa_is_rate_limited(result.get("_meta")):
                    raise RetryableError("Exa rate limit reached")
                return result
        raise ValueError("Exa response did not contain a result")

    return with_retry(request, attempts=7, cap=60.0)


def _exa_text(result):
    return "\n\n".join(
        item["text"]
        for item in result.get("content", [])
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ).strip()


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa) for surveys, blogs and project pages. `query` is the search text, `objective` describes the
    ideal page in natural language. Returns the clean text of the top results with their URLs, or NO RESULTS / ERROR: ...."""
    try:
        objective = objective.strip() or f"Find authoritative and relevant pages about {query}."
        result = _exa_call(
            "web_search_exa",
            {
                "query": query,
                "objective": objective,
                "numResults": max(1, min(int(num_results), 10)),
            },
        )
        text = _exa_text(result)
        return text or "NO RESULTS"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return f"ERROR: {_redact_api_key(message)}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown, truncated to ~12000 characters.
    Returns NO RESULTS / ERROR: ... when the page cannot be read."""
    try:
        result = _exa_call("web_fetch_exa", {"urls": [url]})
        text = _exa_text(result)
        return text[:12000] if text else "NO RESULTS"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return f"ERROR: {_redact_api_key(message)}"


# the researcher subagent gets exactly these
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
        except KeyboardInterrupt:
            print(f"== {name}\nERROR: retry interrupted by user\n")
            break
