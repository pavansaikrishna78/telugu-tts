"""Tests for script parsing and audio logic. These need no GPU work and no model."""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batch import natural_key  # noqa: E402
from from_script import narration_of, parse_script, slugify  # noqa: E402
from qc import WHISPER_RATE, better_take, cer, normalize_for_cer, select_flagged, speech_windows  # noqa: E402
from textprep import TextPrep  # noqa: E402
from tts_engine import (PEAK_CEILING, SAMPLE_RATE, assemble, fade_edges, is_anomalous,  # noqa: E402
                        limit_peaks, normalize_loudness, tidy_chunk)

SCRIPT = """# Lecture

### INTRO
[ON SCREEN]: ఇది చెప్పకూడదు
[NARRATION]: నమస్తే. [TRAP]: ఇది ముఖ్యం.
[VERIFY rate] [EXAM POINT]: పరీక్షలో వస్తుంది.
[DURATION]: 30s
తరువాత వచ్చేది చెప్పకూడదు.

### Slide 3 – Residential Status
**[NARRATION]:** రెండవ భాగం.
**[DURATION]:** 20s

### Notes only
[ON SCREEN]: ఇక్కడ మాటలు లేవు

### [ICAI EXTRA – no slide] Extra topic
[NARRATION]: [ICAI EXTRA – no slide] అదనపు విషయం.
[DURATION]: 10s
"""


def test_sections_and_order():
    sections, skipped = parse_script(SCRIPT)
    assert [h for h, _ in sections] == ["INTRO", "Slide 3 – Residential Status", "[ICAI EXTRA – no slide] Extra topic"]
    assert skipped == ["Notes only"]


def test_only_narration_is_spoken():
    sections, _ = parse_script(SCRIPT)
    body = sections[0][1]
    assert "నమస్తే" in body and "పరీక్షలో" in body
    assert "చెప్పకూడదు" not in body


def test_prepared_script_drops_labels_and_notes(tmp_path):
    prep = TextPrep(unknown_log=tmp_path / "u.txt")
    sections, _ = parse_script(SCRIPT)
    assert " ".join(s.text for s in prep.prepare(sections[0][1])) == "నమస్తే. ఇది ముఖ్యం. పరీక్షలో వస్తుంది."
    assert [s.text for s in prep.prepare(sections[2][1])] == ["అదనపు విషయం."]


def test_narration_without_duration_runs_to_end():
    assert narration_of("[NARRATION]: ఒకటి.") == "ఒకటి."


def test_four_hashes_do_not_start_a_section():
    sections, _ = parse_script("### A\n[NARRATION]: ఒకటి.\n#### sub\nరెండు.\n[DURATION]: 1s\n")
    assert len(sections) == 1


@pytest.mark.parametrize("heading,slug", [
    ("INTRO", "intro"),
    ("Slide 3 – Residential Status", "slide_3_residential_status"),
    ("[ICAI EXTRA – no slide] Extra", "icai_extra_no_slide_extra"),
])
def test_slug(heading, slug):
    assert slugify(heading) == slug


def test_natural_sort():
    names = ["slide_10.txt", "slide_2.txt", "slide_01.txt"]
    assert sorted(map(Path, names), key=natural_key) == [Path("slide_01.txt"), Path("slide_2.txt"), Path("slide_10.txt")]


@pytest.mark.parametrize("seconds,expected,bad", [(0, 10, True), (3.9, 10, True), (4, 10, False), (25, 10, False),
                                                  (25.1, 10, True)])
def test_duration_anomaly_limits(seconds, expected, bad):
    assert is_anomalous(seconds, expected) is bad


def tone(seconds=3.0, amplitude=0.2):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_limiter_never_exceeds_ceiling():
    audio = tone(amplitude=2.0)
    limited = limit_peaks(audio)
    assert limited.size == audio.size
    assert np.max(np.abs(limited)) <= PEAK_CEILING + 1e-6


def test_limiter_leaves_quiet_audio_alone():
    audio = tone(amplitude=0.1)
    assert np.array_equal(limit_peaks(audio), audio)


def test_loudness_reaches_target():
    import pyloudnorm as pyln
    audio = normalize_loudness(tone(seconds=5, amplitude=0.05))
    assert abs(pyln.Meter(SAMPLE_RATE).integrated_loudness(audio) + 16) < 0.5
    assert np.max(np.abs(audio)) <= PEAK_CEILING + 1e-6


def test_fade_edges_start_and_end_at_zero():
    faded = fade_edges(np.ones(SAMPLE_RATE, dtype=np.float32))
    assert faded[0] == 0 and faded[-1] == 0 and faded[SAMPLE_RATE // 2] == 1


def test_tidy_chunk_handles_silence():
    assert tidy_chunk(np.zeros(1000, dtype=np.float32)).size == 0


def test_tidy_chunk_treats_nan_as_empty_so_retry_fires():
    wave = np.full(1000, np.nan, dtype=np.float32)
    assert tidy_chunk(wave).size == 0
    assert is_anomalous(0, 10) is True


def test_assemble_adds_gaps_and_pauses(tmp_path):
    import json
    chunk_dir = tmp_path / "chunks" / "x"
    chunk_dir.mkdir(parents=True)
    sf.write(chunk_dir / "a.wav", tone(1.0), SAMPLE_RATE, subtype="FLOAT")
    sf.write(chunk_dir / "b.wav", tone(1.0), SAMPLE_RATE, subtype="FLOAT")
    segments = [{"type": "speech", "file": "a.wav"}, {"type": "speech", "file": "b.wav"},
                {"type": "silence", "seconds": 2.0}]
    (chunk_dir / "meta.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    _, seconds = assemble(chunk_dir, tmp_path / "x.wav")
    assert seconds == pytest.approx(1 + 0.25 + 1 + 0.25 + 2, abs=0.01)
    assert sf.info(tmp_path / "x.wav").samplerate == SAMPLE_RATE


def test_cer_ignores_punctuation_and_spaces():
    assert cer("నమస్తే, బాగున్నారా?", "నమస్తే బాగున్నారా") == 0
    assert normalize_for_cer("a b.") == "ab"


def test_cer_counts_errors():
    assert cer("abcd", "abxd") == pytest.approx(0.25)
    assert cer("abcd", "") == 1


def test_select_flagged_takes_worst_ten_percent_and_anomalies():
    rows = [{"cer": i / 100, "seconds": 10, "est_seconds": 10, "reason": ""} for i in range(20)]
    rows[0].update(seconds=1)
    select_flagged(rows)
    flagged = [i for i, r in enumerate(rows) if r["flagged"]]
    assert flagged == [0, 18, 19]
    assert rows[0]["reason"] == "duration"


def test_better_take_prefers_sound_duration_over_cer():
    assert better_take(0.2, True, 0.5, False) is True
    assert better_take(0.2, False, 0.1, True) is False
    assert better_take(0.2, False, 0.1, False) is True
    assert better_take(0.2, False, 0.3, False) is False


def test_cer_reads_whisper_digits_as_words():
    assert cer("తొమ్మిది లక్షల రూపాయలు", "9 లక్షల రూపాయలు") == 0
    assert cer("పది శాతం", "10 శాతం") == 0


def test_speech_windows_cover_long_audio_within_limit():
    t = np.arange(25 * WHISPER_RATE) / WHISPER_RATE
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # no silence at all
    windows = speech_windows(audio)
    assert all(end - start <= 8 * WHISPER_RATE for start, end in windows)
    assert windows[0][0] == 0 and windows[-1][1] == audio.size
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))


def test_speech_windows_merge_short_phrases():
    gap = np.zeros(WHISPER_RATE // 2, dtype=np.float32)
    t = np.arange(WHISPER_RATE) / WHISPER_RATE
    word = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    audio = np.concatenate([word, gap, word, gap, word])
    assert len(speech_windows(audio)) == 1
