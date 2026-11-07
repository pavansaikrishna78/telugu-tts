#!/usr/bin/env python
"""Phase 2: proves the GPU path with a prompt bundled in the model repo.

The bundled prompt is not your voice. This only measures speed and memory.
Reports load time, peak VRAM, real-time factor and the weight dtype.
"""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

REPO = "ai4bharat/IndicF5"
REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"
# Transcript as given in the upstream README.
REF_TEXT = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
TEXT = "నమస్కారం. ఈ రోజు మనం ఆదాయపు పన్ను గురించి నేర్చుకుందాం. ఇది ఒక పరీక్ష మాత్రమే."
OUT = ROOT / "logs" / "smoke_test.wav"


def main():
    snapshot = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True))
    ref_path = snapshot / "prompts" / "PAN_F_HAPPY_00001.wav"

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = AutoModel.from_pretrained(REPO, revision=REVISION, code_revision=REVISION, trust_remote_code=True)
    model = model.to("cuda")
    load_seconds = time.perf_counter() - start

    dtypes = sorted({str(p.dtype) for p in model.ema_model.parameters()})
    assert not any("bfloat16" in d for d in dtypes), f"bf16 is not allowed: {dtypes}"

    # First call includes compile and warm-up. The second is the steady-state speed.
    runs = []
    for seed in (42, 42):
        torch.manual_seed(seed)
        start = time.perf_counter()
        audio = model(TEXT, ref_audio_path=str(ref_path), ref_text=REF_TEXT)
        runs.append((time.perf_counter() - start, len(audio) / 24000))

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    OUT.parent.mkdir(exist_ok=True)
    sf.write(OUT, np.asarray(audio, dtype=np.float32), 24000)

    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Weight dtypes   : {dtypes}")
    print(f"Load time       : {load_seconds:.1f} s")
    print(f"Peak VRAM       : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved")
    for label, (gen, dur) in zip(("first run", "second run"), runs):
        print(f"{label:<16}: {gen:.1f} s to generate {dur:.1f} s of audio, RTF {gen / dur:.2f}")
    print(f"Wrote           : {OUT}")


if __name__ == "__main__":
    main()
