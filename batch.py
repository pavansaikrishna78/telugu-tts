#!/usr/bin/env python
"""Render a folder of slide text files in natural order.

  python batch.py input_dir/ [--voice pavan] [-o output]

slide_01.txt, slide_02.txt, slide_10.txt ... -> output/Voice_001.wav/.mp3 ... + output/manifest.csv
"""
import argparse
import csv
import re
import time
from pathlib import Path

from tts_engine import DEFAULT_OUTPUT, DEFAULT_REFS, DEFAULT_SEED, Engine

MANIFEST_FIELDS = ["file", "characters", "chunks", "audio_seconds", "generation_seconds"]


def natural_key(path: Path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def write_manifest(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_all(engine: Engine, items: list[tuple[str, str]], output_dir: Path, voice: str, seed: int) -> list[dict]:
    """items: (output stem, text). Keeps going if an item has nothing to speak."""
    rows = []
    started = time.perf_counter()
    for number, (stem, text) in enumerate(items, 1):
        print(f"[{number}/{len(items)}] {stem}")
        try:
            result = engine.render(text, output_dir / f"{stem}.wav", voice=voice, seed=seed)
        except ValueError as error:
            print(f"  skipped: {error}")
            continue
        rows.append({"file": result.wav.name, "characters": result.characters, "chunks": result.chunks,
                     "audio_seconds": round(result.audio_seconds, 2),
                     "generation_seconds": round(result.generation_seconds, 2)})
        write_manifest(rows, output_dir)  # after every item, so a crash loses nothing
    audio = sum(r["audio_seconds"] for r in rows)
    print(f"Done: {len(rows)} file(s), {audio / 60:.1f} min of audio in {(time.perf_counter() - started) / 60:.1f} min")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voice", default="pavan")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS)
    args = parser.parse_args()

    files = sorted((p for p in args.input_dir.glob("*.txt") if p.is_file()), key=natural_key)
    if not files:
        raise SystemExit(f"No .txt files in {args.input_dir}")
    items = [(f"Voice_{i:03d}", p.read_text(encoding="utf-8")) for i, p in enumerate(files, 1)]
    engine = Engine(refs_dir=args.refs_dir)
    engine.voice(args.voice)
    render_all(engine, items, args.output_dir, args.voice, args.seed)


if __name__ == "__main__":
    main()
