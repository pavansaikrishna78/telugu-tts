#!/usr/bin/env python
"""Automatic quality check: back-transcribe every chunk and compare with the input.

  python qc.py [output_dir] [--voice-refs refs] [--no-regen]

Telugu ASR is imperfect, so CER is only relative: the worst 10% of chunks and any
duration anomaly are flagged, regenerated once with a new seed, and the better take
is kept. Writes logs/qc_report.csv and rebuilds any file whose chunk changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))

FIELDS = ["file", "chunk", "seconds", "est_seconds", "duration_ratio", "seed", "cer", "flagged", "reason",
          "regen_seed", "regen_seconds", "regen_cer", "kept", "text", "transcript"]
QC_SEED_OFFSET = 2000
WORST_FRACTION = 0.10
WHISPER_RATE = 16000
WINDOW_SECONDS = 8.0


def normalize_for_cer(text: str) -> str:
    """Whisper writes numbers as digits, so they become words before scoring."""
    from textprep import number_to_telugu
    text = re.sub(r"\d[\d,]*", lambda m: number_to_telugu(int(m.group(0).replace(",", ""))), text)
    return re.sub(r"[\s‌‍.,?!।॥:;'\"“”‘’()\[\]–—-]+", "", text)


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    return edit_distance(ref, hyp) / len(ref) if ref else 0.0


def select_flagged(rows: list[dict]) -> None:
    """Marks the worst 10% by CER (at least one) and every duration anomaly."""
    from tts_engine import is_anomalous
    worst = max(1, math.ceil(len(rows) * WORST_FRACTION)) if rows else 0
    ranked = sorted(rows, key=lambda r: r["cer"], reverse=True)
    for row in ranked[:worst]:
        row["reason"] = "worst_cer"
    for row in rows:
        if is_anomalous(row["seconds"], row["est_seconds"]):
            row["reason"] = f"{row['reason']}+duration".lstrip("+") if row["reason"] else "duration"
        row["flagged"] = bool(row["reason"])


def better_take(old_cer: float, old_bad: bool, new_cer: float, new_bad: bool) -> bool:
    """True when the new take should replace the old one. A sound length beats a lower CER."""
    if old_bad != new_bad:
        return old_bad
    return new_cer < old_cer


def _split_long(audio, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Cuts an interval longer than the limit at its quietest 50 ms in the allowed range."""
    import numpy as np
    pieces = []
    frame = WHISPER_RATE // 20
    while end - start > limit:
        search_from = start + limit // 2
        search = audio[search_from:start + limit]
        usable = len(search) // frame * frame
        energy = np.square(search[:usable]).reshape(-1, frame).sum(axis=1)
        cut = search_from + int(np.argmin(energy)) * frame + frame // 2
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def speech_windows(audio, max_seconds: float = WINDOW_SECONDS) -> list[tuple[int, int]]:
    """Groups speech into windows of at most max_seconds, cutting at quiet points."""
    import librosa
    limit = int(max_seconds * WHISPER_RATE)
    intervals = []
    for start, end in librosa.effects.split(audio, top_db=30):
        intervals.extend(_split_long(audio, int(start), int(end), limit))
    windows: list[tuple[int, int]] = []
    for start, end in intervals:
        if windows and end - windows[-1][0] <= limit:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    return windows


class Transcriber:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    def __call__(self, path: Path) -> str:
        """Short windows only: Whisper hits its token limit on long Telugu passages and
        stops mid-word, which would look like dropped speech."""
        import librosa
        audio, _ = librosa.load(path, sr=WHISPER_RATE, mono=True)
        texts = []
        for start, end in speech_windows(audio):
            segments, _ = self.model.transcribe(audio[start:end], language="te", beam_size=5,
                                                condition_on_previous_text=False, vad_filter=False)
            texts.extend(s.text.strip() for s in segments)
        return " ".join(texts)


def run_qc(output_dir: Path, refs_dir: Path, regen: bool = True, report: Path | None = None) -> list[dict]:
    from tts_engine import SAMPLE_RATE, Engine, assemble, is_anomalous
    import soundfile as sf

    report = report or ROOT / "logs" / "qc_report.csv"
    transcribe = Transcriber()
    rows: list[dict] = []
    for meta_path in sorted((output_dir / "chunks").glob("*/meta.json")):
        chunk_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for seg in meta["segments"]:
            if seg["type"] != "speech":
                continue
            hypothesis = transcribe(chunk_dir / seg["file"])
            rows.append({"file": chunk_dir.name, "chunk": seg["file"], "seconds": round(seg["seconds"], 2),
                         "est_seconds": round(seg["est_seconds"], 2),
                         "duration_ratio": round(seg["seconds"] / max(seg["est_seconds"], 0.1), 2),
                         "seed": seg["seed"], "cer": round(cer(seg["text"], hypothesis), 4), "flagged": False,
                         "reason": "", "regen_seed": "", "regen_seconds": "", "regen_cer": "", "kept": "original",
                         "text": seg["text"], "transcript": hypothesis})
            print(f"  {chunk_dir.name}/{seg['file']}: CER {rows[-1]['cer']:.3f}, ratio {rows[-1]['duration_ratio']}")
    select_flagged(rows)

    flagged = [r for r in rows if r["flagged"]]
    if regen and flagged:
        engine = Engine(refs_dir=refs_dir)
        changed: set[str] = set()
        for row in flagged:
            chunk_dir = output_dir / "chunks" / row["file"]
            meta_path = chunk_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            seg = next(s for s in meta["segments"] if s.get("file") == row["chunk"])
            new_seed = int(row["seed"]) + QC_SEED_OFFSET
            audio, gen_seconds = engine.generate(seg["text"], seg.get("voice", meta["voice"]), new_seed)
            candidate = chunk_dir / f"regen_{row['chunk']}"
            sf.write(candidate, audio, SAMPLE_RATE, subtype="FLOAT")
            new_cer = round(cer(seg["text"], transcribe(candidate)), 4)
            new_seconds = audio.size / SAMPLE_RATE
            row.update(regen_seed=new_seed, regen_seconds=round(new_seconds, 2), regen_cer=new_cer)
            if better_take(row["cer"], is_anomalous(row["seconds"], row["est_seconds"]),
                           new_cer, is_anomalous(new_seconds, row["est_seconds"])):
                candidate.replace(chunk_dir / row["chunk"])
                seg.update(seed=new_seed, seconds=new_seconds, gen_seconds=seg["gen_seconds"] + gen_seconds,
                           anomaly=is_anomalous(new_seconds, row["est_seconds"]))
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                row["kept"] = "regen"
                changed.add(row["file"])
            else:
                candidate.unlink()
            print(f"  regen {row['file']}/{row['chunk']}: CER {row['cer']} -> {new_cer}, kept {row['kept']}")
        for name in sorted(changed):
            chunk_dir = output_dir / "chunks" / name
            meta = json.loads((chunk_dir / "meta.json").read_text(encoding="utf-8"))
            assemble(chunk_dir, Path(meta["wav"]))
        update_manifest(output_dir, changed)

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def update_manifest(output_dir: Path, changed: set[str]) -> None:
    """Keeps audio_seconds in manifest.csv correct for rebuilt files."""
    import soundfile as sf
    path = output_dir / "manifest.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if Path(row["file"]).stem in changed:
            row["audio_seconds"] = round(sf.info(output_dir / row["file"]).duration, 2)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from tts_engine import DEFAULT_REFS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", nargs="?", type=Path, default=ROOT / "output")
    parser.add_argument("--voice-refs", type=Path, default=DEFAULT_REFS)
    parser.add_argument("--no-regen", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = run_qc(args.output_dir, args.voice_refs, regen=not args.no_regen, report=args.report)
    flagged = [r for r in rows if r["flagged"]]
    print(f"{len(rows)} chunks checked, {len(flagged)} flagged, "
          f"{sum(r['kept'] == 'regen' for r in rows)} replaced.")


if __name__ == "__main__":
    main()
