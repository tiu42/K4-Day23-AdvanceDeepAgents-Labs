"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
SOURCE_FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
MIN_FAMILIES = 3
# The family must match the URL (RUBRIC 2.2): an arXiv page found by web_search is "web", not "arxiv".
FAMILY_URL = {
    "arxiv": re.compile(r"^https://arxiv\.org/abs/(?!\S*v\d+$)\S+$"),
    "hf-daily": re.compile(r"^https://huggingface\.co/papers/[^\s/]+$"),
    "hf-search": re.compile(r"^https://huggingface\.co/papers/[^\s/]+$"),
}


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    source_by_n = {}
    urls = {}
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} is not an object")
            continue

        number = source.get("n")
        if isinstance(number, bool) or not isinstance(number, int):
            problems.append(f"source n={number!r} is not an int")
        elif number in source_by_n:
            problems.append(f"source number [{number}] is duplicated")
        else:
            source_by_n[number] = source

        url = source.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source n={number!r} has an invalid URL")
        elif not re.fullmatch(r"https?://[^\s<>()\[\]\"']+", url):
            # such characters make the URL ambiguous inside a reference line: percent-encode them or use another URL
            problems.append(f"source n={number!r} URL contains spaces, brackets or quotes: {url}")
        elif url in urls:
            problems.append(f"source URL is duplicated: {url}")
        else:
            urls[url] = number

        family = source.get("source")
        if family not in SOURCE_FAMILIES:
            problems.append(f"source n={number!r} has source={family!r}, expected one of {sorted(SOURCE_FAMILIES)}")
        elif family in FAMILY_URL and isinstance(url, str) and not FAMILY_URL[family].match(url):
            problems.append(f"source n={number!r} is labelled {family} but its url {url} does not match that family")

    families = {source.get("source") for source in sources if isinstance(source, dict)} & SOURCE_FAMILIES
    if len(families) < MIN_FAMILIES:
        problems.append(f"only {len(families)} source families {sorted(families)}; need at least {MIN_FAMILIES}")

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if heading is None:
        problems.append("report is missing the ## References section")
        body = report_text
        references = ""
    else:
        body = report_text[:heading.start()]
        references = report_text[heading.end():]

    # Mask code so citation-like text in examples is not treated as evidence.
    masked_body = re.sub(r"```.*?```|`[^`\n]*`", "", body, flags=re.DOTALL)
    citation_pattern = re.compile(r"\[((?:\d+(?:\s*,\s*\d+|\s*-\s*\d+)*))\](?!\()")

    def expand(match):
        numbers = []
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part.strip())
            if span:
                start, end = map(int, span.groups())
                if end >= start and end - start <= 200:
                    numbers.extend(range(start, end + 1))
                else:
                    numbers.extend((start, end))
            else:
                numbers.append(int(part.strip()))
        return numbers

    cited = set()
    for match in citation_pattern.finditer(masked_body):
        cited.update(expand(match))

    known_numbers = set(source_by_n)
    for number in sorted(cited - known_numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(known_numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = re.findall(r"(?m)^\[(\d+)\][^\n]*$", references)
    reference_numbers = [int(number) for number in reference_lines]
    for number in sorted(known_numbers - set(reference_numbers)):
        problems.append(f"reference for source [{number}] is missing")
    for number in sorted(set(reference_numbers) - known_numbers):
        problems.append(f"reference [{number}] has no matching source")
    for number in sorted(set(reference_numbers)):
        if reference_numbers.count(number) > 1:
            problems.append(f"reference [{number}] appears more than once")

    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\](.*)$", line)
        if not match:
            continue
        number = int(match.group(1))
        found_urls = re.findall(r"https?://[^\s<>()]+", line)
        if len(found_urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in source_by_n and found_urls[0] != source_by_n[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
