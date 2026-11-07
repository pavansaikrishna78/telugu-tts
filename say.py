#!/usr/bin/env python
"""Speak one piece of Telugu text.

  python say.py "తెలుగు వాక్యం" -o out.wav [--voice pavan]
"""
import argparse
from pathlib import Path

from tts_engine import DEFAULT_REFS, DEFAULT_SEED, Engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("-o", "--output", type=Path, default=Path("out.wav"))
    parser.add_argument("--voice", default="pavan")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS)
    args = parser.parse_args()

    engine = Engine(refs_dir=args.refs_dir)
    engine.voice(args.voice)
    result = engine.render(args.text, args.output, voice=args.voice, seed=args.seed)
    print(f"Wrote {result.wav}" + (f" and {result.mp3}" if result.mp3 else ""))
    print(f"{result.audio_seconds:.1f} s of audio in {result.generation_seconds:.1f} s, {result.chunks} chunk(s)")


if __name__ == "__main__":
    main()
