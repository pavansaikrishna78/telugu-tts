#!/usr/bin/env python
"""Clone a voice from a short recording and speak Telugu text with IndicF5.

  python clone_tts.py "తెలుగు వాక్యం" --ref recording.m4a --ref-text "exact words in the recording" -o out.wav
  python clone_tts.py --text-file script.txt --ref recording.m4a --ref-text-file transcript.txt -o out.wav
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))  # where the model weights are stored

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

REPO = "ai4bharat/IndicF5"
REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"  # the tested model version
SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 200  # long single generations drift and drop words
PAUSE_SECONDS = 0.25


def split_text(text: str) -> list[str]:
    """Group sentences into chunks of up to MAX_CHUNK_CHARS. A longer sentence stays whole."""
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?।])\s+", " ".join(text.split())):
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return chunks + [current] if current else chunks


def to_float(audio) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="Telugu text to speak")
    parser.add_argument("--text-file", type=Path, help="UTF-8 file holding the text to speak")
    parser.add_argument("--ref", type=Path, required=True, help="10-15 s recording of the voice to clone")
    parser.add_argument("--ref-text", help="exact words spoken in the recording")
    parser.add_argument("--ref-text-file", type=Path, help="UTF-8 file holding those exact words")
    parser.add_argument("-o", "--output", type=Path, default=Path("out.wav"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
    ref_text = args.ref_text_file.read_text(encoding="utf-8").strip() if args.ref_text_file else args.ref_text
    if not text or not text.strip():
        parser.error("give the text to speak, or --text-file")
    if not ref_text:
        parser.error("give --ref-text or --ref-text-file")
    if not torch.cuda.is_available():
        raise SystemExit("No Nvidia GPU found. IndicF5 is too slow to use on a CPU.")

    # FP32 only: FP16 produces silent output with this model.
    model = AutoModel.from_pretrained(REPO, revision=REVISION, code_revision=REVISION, trust_remote_code=True)
    model = model.to("cuda")

    chunks = split_text(text)
    pause = np.zeros(int(PAUSE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    pieces = []

    with tempfile.TemporaryDirectory() as tmp:
        # The model expects a clean mono WAV, whatever format the recording is in.
        ref_wav = Path(tmp) / "ref.wav"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(args.ref),
                        "-ac", "1", "-ar", str(SAMPLE_RATE), str(ref_wav)], check=True)

        for number, chunk in enumerate(chunks, 1):
            print(f"Chunk {number}/{len(chunks)}: {len(chunk)} characters", flush=True)
            torch.manual_seed(args.seed)
            pieces.append(to_float(model(chunk, ref_audio_path=str(ref_wav), ref_text=ref_text)))
            if number < len(chunks):
                pieces.append(pause)

    audio = np.concatenate(pieces)
    if not np.isfinite(audio).all() or np.abs(audio).max() < 1e-4:
        raise SystemExit("The model produced silent or invalid audio, so nothing was saved.")
    audio = audio * min(1.0, 0.98 / np.abs(audio).max())  # turn it down if the peaks would clip

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"Wrote {args.output} ({len(audio) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
    main()
