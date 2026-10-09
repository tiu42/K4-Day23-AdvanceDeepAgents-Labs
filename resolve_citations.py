"""Runs INSIDE the sandbox (standard library only), right before finalize_citations.py:

    python3 resolve_citations.py [draft.md] [sources.draft.json] [report.md] [sources.json]

The lead cites sources by their registry id, e.g. "DeepSeekMath [@2402.03300]" or "[@2402.03300, @react-blog]",
instead of by number: an LLM keeps ids (which sit next to the facts in the notes) straight far better than numbers.
This script replaces every [@id] with the [n] of that id in the draft registry and writes report.md + sources.json,
which finalize_citations.py then renumbers. The drafts are never modified.
Unknown ids, duplicate ids and bare numeric citations [n] in the draft are reported and nothing is written.
"""
import json
import re
import sys

DRAFT = "/tmp/work/report/draft.md"
SOURCES_DRAFT = "/tmp/work/research/sources.draft.json"
REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_KEYS = re.compile(r"\[(@[^\[\]\s,;]+(?:\s*[,;]\s*@[^\[\]\s,;]+)*)\](?!\()")
_NUMERIC = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def resolve(draft_text, sources):
    """Return (report_text, problems)."""
    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        return draft_text, ["sources.draft.json must be a JSON list of objects"]
    number_of, problems = {}, []
    for source in sources:
        key = str(source.get("id") or "").strip()
        if not key:
            problems.append(f"source n={source.get('n')!r} has no id")
        elif key in number_of:
            problems.append(f"id {key!r} is used by two sources: give each source a unique id")
        else:
            number_of[key] = source.get("n")
    segments = _CODE.split(draft_text)  # odd indexes are code: never touched

    def replace(match):
        out = []
        for key in re.split(r"\s*[,;]\s*", match.group(1)):
            key = key.lstrip("@")
            if key not in number_of:
                message = f"[@{key}] does not match the id of any source in sources.draft.json"
                if message not in problems:
                    problems.append(message)
                out.append(f"[@{key}]")
            else:
                out.append(f"[{number_of[key]}]")
        return "".join(out)

    for i, segment in enumerate(segments):
        if i % 2:
            continue
        for match in _NUMERIC.finditer(segment):
            problems.append(f"numeric citation {match.group(0)} in the draft: cite by id, e.g. [@<id>]")
        segments[i] = _KEYS.sub(replace, segment)
    return "".join(segments), problems


def main(argv):
    paths = argv[1:5] + [DRAFT, SOURCES_DRAFT, REPORT, SOURCES][len(argv[1:5]):]
    draft_path, sources_draft_path, report_path, sources_path = paths
    try:
        with open(draft_path, encoding="utf-8") as f:
            draft = f.read()
        with open(sources_draft_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    report, problems = resolve(draft, sources)
    if problems:
        print("NOT resolved (fix these in the drafts, then run again):")
        print("\n".join(problems))
        return 1
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(sources_path, "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)
    print(f"RESOLVED: {len(sources)} sources; citations converted to [n]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
