"""IndicF5 synthesis engine: chunk-by-chunk generation, joins, loudness, WAV + MP3.

Every rendered item keeps its chunks in output/chunks/<stem>/ with a meta.json, so
qc.py can replace one chunk and rebuild the final file.
"""
from __future__ import annotations

import atexit
import io
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from pydub import AudioSegment, silence

from textprep import Silence, Speech, TextPrep

REPO = "ai4bharat/IndicF5"
REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"
SAMPLE_RATE = 24000
MP3_SAMPLE_RATE = 44100  # MP3 at 24 kHz cannot go above 160 kbps
DEFAULT_SEED = 42
RETRY_SEED_OFFSET = 1000
CHUNK_GAP_SECONDS = 0.25
FADE_SECONDS = 0.010
TARGET_LUFS = -16.0
CHUNK_RMS = 0.1
PEAK_CEILING = 0.944  # about -0.5 dBFS
MIN_RATIO, MAX_RATIO = 0.40, 2.50
DEFAULT_REFS = ROOT / "refs"
DEFAULT_OUTPUT = ROOT / "output"
# [STYLE excited] uses the voice <voice>_excited until the next tag. [STYLE normal] goes back.
STYLE_TAG = re.compile(r"\[\s*STYLE\s+([^\]\s]+)\s*\]\s*:?", re.IGNORECASE)
NORMAL_STYLE = "normal"


def styled_parts(text: str, voice: str) -> list[tuple[str, str]]:
    """Splits text at [STYLE ...] tags. Returns [(voice name, text)]. Every call starts at normal."""
    pieces = STYLE_TAG.split(text)  # odd indexes hold style names
    parts = [(voice, pieces[0])]
    for i in range(1, len(pieces), 2):
        style = pieces[i].lower()
        parts.append((voice if style == NORMAL_STYLE else f"{voice}_{style}", pieces[i + 1]))
    return parts


def voices_used(text: str, voice: str) -> set[str]:
    return {name for name, part in styled_parts(text, voice) if part.strip()}


def plan_segments(prep, text: str, voice: str) -> list:
    """[(Speech or Silence, voice name)] in speaking order. Chunks never cross a style change."""
    return [(segment, name) for name, part in styled_parts(text, voice) for segment in prep.prepare(part)]


@dataclass
class Rendered:
    wav: Path
    mp3: Path | None
    characters: int
    chunks: int
    audio_seconds: float
    generation_seconds: float
    retried: int


def is_anomalous(seconds: float, expected: float) -> bool:
    return seconds <= 0 or not (MIN_RATIO <= seconds / max(expected, 0.1) <= MAX_RATIO)


def list_voices(refs_dir=DEFAULT_REFS) -> list[str]:
    refs_dir = Path(refs_dir)
    return sorted(p.stem for p in refs_dir.glob("*.wav") if p.with_suffix(".txt").exists())


def tidy_chunk(wave: np.ndarray) -> np.ndarray:
    """Shortens silences over 1 s, trims the edges, levels the chunk. Stays in float.

    Silent or non-finite output comes back empty, which triggers the retry.
    """
    if wave.size == 0 or not np.isfinite(wave).all() or not np.any(wave):
        return np.zeros(0, dtype=np.float32)
    scale = max(1.0, float(np.max(np.abs(wave))))
    pcm = (wave / scale * 32767).astype(np.int16)
    segment = AudioSegment(pcm.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
    parts = silence.split_on_silence(segment, min_silence_len=1000, silence_thresh=-50,
                                     keep_silence=500, seek_step=10)
    if parts:
        segment = sum(parts, AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE))
    audio = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio, _ = librosa.effects.trim(audio, top_db=40, frame_length=1024, hop_length=256)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return audio * (CHUNK_RMS / rms) if rms > 0 else audio


def fade_edges(audio: np.ndarray) -> np.ndarray:
    n = int(FADE_SECONDS * SAMPLE_RATE)
    if audio.size < 2 * n:
        return audio
    audio = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=audio.dtype)
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def limit_peaks(audio: np.ndarray, ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Look-ahead limiter, so loudness gain never clips."""
    needed = np.minimum(1.0, ceiling / np.maximum(np.abs(audio), 1e-9))
    if needed.min() >= 1.0:
        return audio
    window = int(0.005 * SAMPLE_RATE) | 1
    pad = window // 2
    # Moving minimum, then a smooth curve so the gain change is inaudible.
    from scipy.ndimage import minimum_filter1d
    envelope = minimum_filter1d(needed, size=window * 2 + 1, mode="nearest")
    kernel = np.hanning(window + 2)[1:-1]
    kernel /= kernel.sum()
    smooth = np.convolve(np.pad(envelope, pad, mode="edge"), kernel, mode="valid")
    return np.clip(audio * smooth[: audio.size], -ceiling, ceiling)


def normalize_loudness(audio: np.ndarray, target: float = TARGET_LUFS) -> np.ndarray:
    if audio.size < SAMPLE_RATE // 2:  # the meter needs at least 0.4 s
        return limit_peaks(audio)
    meter = pyln.Meter(SAMPLE_RATE)
    # Limiting lowers loudness a little, so correct once more after it.
    for _ in range(2):
        loudness = meter.integrated_loudness(audio)
        if not np.isfinite(loudness):
            return audio
        audio = limit_peaks(audio * 10 ** ((target - loudness) / 20))
    return audio


def write_outputs(audio: np.ndarray, wav_path: Path) -> Path | None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg is not installed, so no MP3 was written.")
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
                    "-ar", str(MP3_SAMPLE_RATE), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)], check=True)
    return mp3_path


def assemble(chunk_dir: Path, wav_path: Path) -> tuple[Path | None, float]:
    """Builds the final file from chunk_dir/meta.json. Returns (mp3 path, seconds)."""
    meta = json.loads((chunk_dir / "meta.json").read_text(encoding="utf-8"))
    pieces: list[np.ndarray] = []
    gap = np.zeros(int(CHUNK_GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    segments = meta["segments"]
    for i, seg in enumerate(segments):
        if seg["type"] == "silence":
            pieces.append(np.zeros(int(seg["seconds"] * SAMPLE_RATE), dtype=np.float32))
            continue
        audio, _ = sf.read(chunk_dir / seg["file"], dtype="float32")
        pieces.append(fade_edges(audio))
        if i + 1 < len(segments):
            pieces.append(gap)
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    audio = normalize_loudness(audio)
    return write_outputs(audio, wav_path), audio.size / SAMPLE_RATE


class Engine:
    """Load once, generate many. FP32 only: FP16 autocast produces NaN audio with this model."""

    def __init__(self, refs_dir=DEFAULT_REFS, prep: TextPrep | None = None):
        from transformers import AutoModel
        from f5_tts.infer import utils_infer

        self._utils = utils_infer
        self.refs_dir = Path(refs_dir)
        self.prep = prep or TextPrep()
        self.device = "cuda"
        self.peak_vram_gib = 0.0
        self.model = AutoModel.from_pretrained(
            REPO, revision=REVISION, code_revision=REVISION, trust_remote_code=True).to(self.device)
        self.vocab = self._load_vocab()
        self._voices: dict[str, tuple[str, str]] = {}
        self._warned_chars: set[str] = set()
        atexit.register(self._cleanup)

    def _load_vocab(self) -> set[str]:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, "checkpoints/vocab.txt", revision=REVISION)
        return set(Path(path).read_text(encoding="utf-8").split("\n"))

    def _cleanup(self) -> None:
        for path, _ in self._voices.values():
            Path(path).unlink(missing_ok=True)

    def voice(self, name: str) -> tuple[str, str]:
        if name not in self._voices:
            wav, txt = self.refs_dir / f"{name}.wav", self.refs_dir / f"{name}.txt"
            if not (wav.exists() and txt.exists()):
                raise SystemExit(f"Voice '{name}' not found. Expected {wav} and {txt}. Run prep_ref.py first.")
            self._voices[name] = self._utils.preprocess_ref_audio_text(
                str(wav), txt.read_text(encoding="utf-8"), show_info=lambda *_: None)
        return self._voices[name]

    def for_model(self, text: str) -> str:
        """The en dash is not in the model's vocabulary, so it becomes a comma pause."""
        text = text.replace(" – ", ", ").replace("–", ", ")
        unknown = {c for c in text if c not in self.vocab and not c.isspace()} - self._warned_chars
        if unknown:
            self._warned_chars |= unknown
            print(f"WARNING: characters not in the model vocabulary, read as a space: {sorted(unknown)}")
        return text

    def generate(self, text: str, voice: str, seed: int) -> tuple[np.ndarray, float]:
        """One chunk. Returns (audio, generation seconds)."""
        ref_audio, ref_text = self.voice(voice)
        torch.manual_seed(seed)
        start = time.perf_counter()
        wave, _, _ = self._utils.infer_process(
            ref_audio, ref_text, self.for_model(text), self.model.ema_model, self.model.vocoder,
            mel_spec_type="vocos", speed=self.model.config.speed, device=self.device,
            show_info=lambda *_: None)
        elapsed = time.perf_counter() - start
        self.peak_vram_gib = max(self.peak_vram_gib, torch.cuda.max_memory_allocated() / 2**30)
        return tidy_chunk(np.asarray(wave, dtype=np.float32)), elapsed

    def generate_checked(self, text: str, expected: float, voice: str, seed: int):
        """Retries once with a new seed when the output is empty or the wrong length."""
        audio, elapsed = self.generate(text, voice, seed)
        seconds = audio.size / SAMPLE_RATE
        if not is_anomalous(seconds, expected):
            return audio, seed, elapsed, False
        retry_seed = seed + RETRY_SEED_OFFSET
        print(f"  retry: got {seconds:.1f} s, expected about {expected:.1f} s. New seed {retry_seed}.")
        retry, retry_elapsed = self.generate(text, voice, retry_seed)
        elapsed += retry_elapsed
        retry_seconds = retry.size / SAMPLE_RATE
        if abs(np.log(max(retry_seconds, 0.01) / expected)) < abs(np.log(max(seconds, 0.01) / expected)):
            return retry, retry_seed, elapsed, True
        return audio, seed, elapsed, True

    def render(self, text: str, wav_path, voice: str = "pavan", seed: int = DEFAULT_SEED,
               chunk_root=None, on_chunk=None) -> Rendered:
        """on_chunk(est_seconds), if given, is called after each finished speech chunk."""
        wav_path = Path(wav_path)
        segments = plan_segments(self.prep, text, voice)
        if not any(isinstance(s, Speech) for s, _ in segments):
            raise ValueError("Nothing to speak after removing markers.")
        chunk_root = Path(chunk_root) if chunk_root else wav_path.parent / "chunks"
        chunk_dir = chunk_root / wav_path.stem
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True)

        meta, total_gen, characters, retried, index = [], 0.0, 0, 0, 0
        speech_total = sum(isinstance(s, Speech) for s, _ in segments)
        for seg, seg_voice in segments:
            if isinstance(seg, Silence):
                meta.append({"type": "silence", "seconds": seg.seconds})
                continue
            index += 1
            print(f"  chunk {index}/{speech_total}: about {seg.est_seconds:.0f} s, {len(seg.text)} characters")
            audio, used_seed, gen_seconds, was_retried = self.generate_checked(
                seg.text, seg.est_seconds, seg_voice, seed)
            filename = f"chunk_{index:03d}.wav"
            sf.write(chunk_dir / filename, audio, SAMPLE_RATE, subtype="FLOAT")
            seconds = audio.size / SAMPLE_RATE
            meta.append({"type": "speech", "file": filename, "text": seg.text, "voice": seg_voice, "seed": used_seed,
                         "seconds": seconds, "est_seconds": seg.est_seconds, "gen_seconds": gen_seconds,
                         "retried": was_retried, "anomaly": is_anomalous(seconds, seg.est_seconds)})
            total_gen += gen_seconds
            characters += len(seg.text)
            retried += was_retried
            if on_chunk:
                on_chunk(seg.est_seconds)
        (chunk_dir / "meta.json").write_text(
            json.dumps({"voice": voice, "wav": str(wav_path.resolve()), "segments": meta},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        mp3, audio_seconds = assemble(chunk_dir, wav_path)
        return Rendered(wav_path, mp3, characters, speech_total, audio_seconds, total_gen, retried)
