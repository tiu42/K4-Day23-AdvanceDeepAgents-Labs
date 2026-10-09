"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    RESOLVER_PATH,
    NOTES_DIR,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from check_citations import SOURCE_FAMILIES
from check_citations import check as check_citations
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
RESOLVER_SOURCE = ROOT / "resolve_citations.py"     # ours: turns the draft's [@id] citations into [n]


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars, never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    value = re.sub(r"[^\w]+", "-", str(topic).strip().lower(), flags=re.UNICODE)
    value = value.strip("-_")[:60].strip("-_")
    return value or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Conduct a deep, citation-checked research survey about this topic:

{topic}

Complete the workflow in your system prompt. Write the final report to {REPORT_PATH} and the
source registry to {SOURCES_PATH}. Use {NOTES_DIR} for researcher notes. The uploaded scripts
are {FINALIZER_PATH} and {VALIDATOR_PATH}. Write the report in English and never invent facts,
sources, URLs, dates, or numbers."""


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
        for call in calls or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                name = call.get("name") or function.get("name")
            else:
                name = getattr(call, "name", None)
                function = getattr(call, "function", None)
                name = name or getattr(function, "name", None)
            if name:
                tool_calls[name] += 1
        usage = message.get("usage_metadata") if isinstance(message, dict) else getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(sorted(tool_calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("sandbox did not produce a non-empty report")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sandbox did not produce sources.json")
    try:
        report_text = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"sandbox output is invalid: {exc}") from exc
    if not isinstance(sources, list) or not all(isinstance(source, dict) for source in sources):
        raise RuntimeError("sources.json must contain a JSON array of objects")

    summary = summarize(messages, elapsed, model_name)
    if summary["subagent_calls"] < 3:
        raise RuntimeError(
            "lead used fewer than three researcher subagent calls: "
            f"{summary['subagent_calls']}; need at least 3"
        )

    # The host only checks, it never rewrites: finalize_citations.py and the validator loop run inside the
    # sandbox, and the files saved below are byte-for-byte the ones downloaded from it (RUBRIC, general rules).
    # check_citations also rejects fewer than 3 source families and family labels that do not match the URL.
    citation_problems = check_citations(report_text, sources)
    if citation_problems:
        raise RuntimeError(
            "report failed citation validation: " + "; ".join(citation_problems)
        )

    slug = slugify(topic)
    metadata = {
        "topic": topic,
        **summary,
        "n_sources": len(sources),
        "source_families": sorted({source.get("source") for source in sources} & SOURCE_FAMILIES),
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "report": reports_dir / f"{slug}.md",
        "sources": reports_dir / f"{slug}.sources.json",
        "meta": reports_dir / f"{slug}.meta.json",
    }
    try:
        output_paths["report"].write_bytes(report_bytes)
        output_paths["sources"].write_bytes(sources_bytes)
        output_paths["meta"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        for path in output_paths.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise RuntimeError(f"could not save report outputs: {exc}") from exc
    return output_paths["report"]


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = str(topic).strip()
    if not topic:
        print('Usage: python research.py "survey topic"', file=sys.stderr)
        return 2

    start = time.monotonic()
    try:
        model = make_model()
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or type(model).__name__
        )
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(
                backend,
                {
                    VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                    FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                    RESOLVER_PATH: RESOLVER_SOURCE.read_bytes(),
                },
            )
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            messages = (
                result.get("messages", [])
                if isinstance(result, dict)
                else getattr(result, "messages", [])
            )
            report_path = save_outputs(
                backend,
                topic,
                messages,
                time.monotonic() - start,
                model_name,
            )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Saved report to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
