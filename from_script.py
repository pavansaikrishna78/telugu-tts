#!/usr/bin/env python
"""Render a lecture-script markdown file, one output file per "### ..." section.

  python from_script.py script.md [--limit 3] [--voice pavan] [-o output] [--dry-run]

Only text from [NARRATION]: up to [DURATION]: is spoken. --dry-run shows the prepared
text without using the GPU.
"""
import argparse
import re
from pathlib import Path

HEADING = re.compile(r"^###(?!#)\s*(.+?)\s*$", re.MULTILINE)
_DECOR = r"[*_\s]*"
NARRATION = re.compile(rf"{_DECOR}\[\s*NARRATION\s*\]{_DECOR}:?{_DECOR}", re.IGNORECASE)
DURATION = re.compile(rf"{_DECOR}\[\s*DURATION\b[^\]]*\]", re.IGNORECASE)


def slugify(heading: str) -> str:
    text = re.sub(r"[\[\]*_`]", " ", heading)
    text = re.sub(r"[^A-Za-z0-9ఀ-౿]+", "_", text).strip("_")
    return text[:60].rstrip("_").lower() or "section"


def narration_of(body: str) -> str:
    """All [NARRATION]: ... [DURATION]: spans in a section. A span with no [DURATION] runs to the end."""
    spans = []
    position = 0
    while True:
        start = NARRATION.search(body, position)
        if not start:
            break
        end = DURATION.search(body, start.end())
        spans.append(body[start.end(): end.start() if end else len(body)])
        position = end.end() if end else len(body)
    return "\n\n".join(s.strip() for s in spans if s.strip())


def parse_script(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns ([(heading, narration)], headings without narration)."""
    matches = list(HEADING.finditer(text))
    sections, skipped = [], []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        narration = narration_of(text[match.end():end])
        if narration:
            sections.append((match.group(1), narration))
        else:
            skipped.append(match.group(1))
    return sections, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--voice", default="pavan")
    parser.add_argument("--limit", type=int, help="only the first N sections")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--refs-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print prepared text, no audio")
    args = parser.parse_args()

    sections, skipped = parse_script(args.script.read_text(encoding="utf-8"))
    for heading in skipped:
        print(f"WARNING: no [NARRATION] in section, skipped: {heading}")
    if not sections:
        raise SystemExit("No sections with [NARRATION] found.")
    if args.limit:
        sections = sections[: args.limit]
    items = [(f"Voice_{i:03d}_{slugify(h)}", body) for i, (h, body) in enumerate(sections, 1)]

    if args.dry_run:
        from textprep import Silence, TextPrep
        prep = TextPrep()
        for stem, body in items:
            print(f"\n=== {stem}")
            for seg in prep.prepare(body):
                print(f"  [silence {seg.seconds} s]" if isinstance(seg, Silence)
                      else f"  ({seg.est_seconds:.0f} s) {seg.text}")
        return

    from batch import render_all
    from tts_engine import DEFAULT_OUTPUT, DEFAULT_REFS, DEFAULT_SEED, Engine
    engine = Engine(refs_dir=args.refs_dir or DEFAULT_REFS)
    engine.voice(args.voice)
    render_all(engine, items, args.output_dir or DEFAULT_OUTPUT, args.voice,
               DEFAULT_SEED if args.seed is None else args.seed)


if __name__ == "__main__":
    main()
