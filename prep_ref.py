#!/usr/bin/env python
"""Prepare a reference voice clip.

  python prep_ref.py refs/raw/recording.m4a --text "exact transcript"
  python prep_ref.py recording.mp3 --name pavan --text-file transcript.txt

Output: refs/<name>.wav (mono, 24 kHz, 16-bit, about -20 LUFS) and refs/<name>.txt.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 24000
TARGET_LUFS = -20.0
MIN_SECONDS, MAX_SECONDS = 8.0, 15.0
SUPPORTED = {".mp3", ".m4a", ".wav", ".ogg"}


def decode(path: Path, out: Path, highpass: bool) -> None:
    filters = ["-af", "highpass=f=80"] if highpass else []
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
         *filters, "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(out)],
        check=True)


def prepare(source: Path, name: str, transcript: str, refs_dir: Path) -> list[str]:
    """Writes the clip and transcript. Returns warnings."""
    if source.suffix.lower() not in SUPPORTED:
        raise SystemExit(f"Unsupported format {source.suffix}. Use one of: {sorted(SUPPORTED)}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not installed.")
    if not transcript.strip():
        raise SystemExit("The transcript is empty. Give the exact words spoken in the clip.")
    warnings = []

    with tempfile.TemporaryDirectory() as tmp:
        raw_path, filtered_path = Path(tmp) / "raw.wav", Path(tmp) / "filtered.wav"
        decode(source, raw_path, highpass=False)
        decode(source, filtered_path, highpass=True)
        raw, _ = sf.read(raw_path, dtype="float32")
        audio, _ = sf.read(filtered_path, dtype="float32")

    clipped = int(np.sum(np.abs(raw) >= 0.999))
    if clipped > 3:
        warnings.append(f"The recording clips ({clipped} samples at full scale). Re-record a little quieter.")

    audio, _ = librosa.effects.trim(audio, top_db=35, frame_length=1024, hop_length=256)
    seconds = len(audio) / SAMPLE_RATE
    if seconds < MIN_SECONDS:
        warnings.append(f"Clip is {seconds:.1f} s after trimming, shorter than {MIN_SECONDS:.0f} s.")
    if seconds > MAX_SECONDS:
        warnings.append(f"Clip is {seconds:.1f} s after trimming, longer than {MAX_SECONDS:.0f} s.")

    loudness = pyln.Meter(SAMPLE_RATE).integrated_loudness(audio)
    gain = 10 ** ((TARGET_LUFS - loudness) / 20)
    peak = float(np.max(np.abs(audio))) * gain
    if peak > 0.98:  # never clip in order to reach the target
        gain *= 0.98 / peak
        warnings.append("Loudness stayed below the target to avoid clipping. The clip has sharp peaks.")
    audio = audio * gain

    refs_dir.mkdir(parents=True, exist_ok=True)
    wav_path, txt_path = refs_dir / f"{name}.wav", refs_dir / f"{name}.txt"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    txt_path.write_text(transcript, encoding="utf-8")
    print(f"Saved {wav_path} ({seconds:.1f} s, {pyln.Meter(SAMPLE_RATE).integrated_loudness(audio):.1f} LUFS)")
    print(f"Saved {txt_path}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", default="pavan", help="voice name (default: pavan)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="exact transcript, word for word")
    group.add_argument("--text-file", type=Path, help="UTF-8 file with the exact transcript")
    parser.add_argument("--refs-dir", type=Path, default=ROOT / "refs")
    args = parser.parse_args()

    transcript = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8").rstrip("\n")
    warnings = prepare(args.source, args.name, transcript, args.refs_dir)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
