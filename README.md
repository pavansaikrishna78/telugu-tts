# Telugu TTS

Local Telugu voice cloning and text-to-speech with [IndicF5](https://github.com/AI4Bharat/IndicF5).
Everything lives in `~/telugu-tts`. Output is 24 kHz WAV plus 192 kbps MP3 at -16 LUFS.

Clone only your own voice, or a voice you have permission to use.

## Start a session

`tmux` keeps long jobs running if your connection drops.

```bash
tmux new -A -s tts
cd ~/telugu-tts
source .venv/bin/activate
```

Detach with `Ctrl+B`, then `D`. Reattach with the first command above.

To rebuild the environment, run `./setup.sh`. It is safe to re-run.

## Add a reference voice

1. Record 10-15 seconds in a quiet room, in your normal lecture voice. No music or fan noise.
2. Upload the recording (mp3, m4a, wav or ogg) to `refs/`. From Windows PowerShell:
   ```powershell
   scp "C:\path\to\recording.m4a" pavan@<server-ip>:~/telugu-tts/refs/
   ```
3. Prepare it, giving the exact words spoken. A wrong transcript is the most common cause of bad output.
   ```bash
   python prep_ref.py refs/recording.m4a --name pavan --text "మీ రికార్డింగ్‌లోని ఖచ్చితమైన మాటలు"
   ```
   This writes `refs/pavan.wav` and `refs/pavan.txt`. For another voice, change `--name`.

## Generate speech

| Task | Command |
|---|---|
| One sentence | `python say.py "తెలుగు వాక్యం" -o out.wav --voice pavan` |
| Folder of `slide_01.txt`, `slide_02.txt`... | `python batch.py input_dir/` |
| Lecture script | `python from_script.py script.md` |
| First 3 sections only | `python from_script.py script.md --limit 3` |
| See the prepared text, no audio | `python from_script.py script.md --dry-run` |
| Quality check | `python qc.py output/` |

`batch.py` and `from_script.py` write numbered `Voice_###` files and `manifest.csv` to `output/`.
Use `-o other_dir` to send them elsewhere. Re-using a folder overwrites files with the same name.

### Web UI

```bash
python app.py
```

It listens on `127.0.0.1:7860` only. Open a tunnel from Windows PowerShell, then browse to http://localhost:7860:

```powershell
ssh -L 7860:localhost:7860 pavan@<server-ip>
```

## Fix pronunciation

- **`lexicon.tsv`**: one `english<TAB>telugu pronunciation` per line. Longest match wins.
  All-caps entries of up to 4 letters match only in capitals, so `IT` is not "it".
- **`logs/unknown_terms.txt`**: English words not in the lexicon, with the guess that was used.
  Copy the ones that sound wrong into `lexicon.tsv` and correct them.
- **`legal_rules.yaml`**: how section references are read. `three_digit_style` is `hundred`
  (202 = టూ హండ్రెడ్ టూ) or `pair` (202 = టూ ఓ టూ).

After editing, run `python -m pytest tests/ -q`. Some tests check the default wording, so a deliberate
change can fail one of them.

## Script format

- Each `### heading` starts a new file, named from the heading.
- Only text from `[NARRATION]:` up to `[DURATION]:` is spoken.
- `[PAUSE 2s]` becomes 2 seconds of silence.
- `[STYLE excited]` speaks with the voice `<voice>_excited` until the next tag. `[STYLE normal]` goes back, and each section starts normal.
- Labels such as `[TRAP]:` and `[EXAM POINT]:` are removed and their text is spoken.
- `[VERIFY ...]`, `[CONFLICT ...]` and `[ON SCREEN]...` are never spoken.

## Quality check

`qc.py` transcribes every chunk with Whisper large-v3 and compares it with the input. Telugu speech
recognition is imperfect, so the error rate is only useful for comparing chunks. The worst 10% and any
chunk with an odd length are regenerated once, and the better take is kept. The results are in
`logs/qc_report.csv`. It does not replace listening.

## Speed and limits

- About 2.1 minutes of GPU time per minute of audio, using about 1.4 GiB of VRAM on the Tesla T4.
- FP32 only. FP16 produces silent output (NaN) with this model. BF16 is not supported by the T4.
- Very long sentences stay in one chunk, but IndicF5 may split them internally at commas with a short crossfade.

## Copy outputs to Windows

- **WinSCP**: connect to the server over SFTP as `pavan`, open `/home/pavan/telugu-tts/output`, and drag the files across.
- **PowerShell**, copying into `C:\Voice`:
  ```powershell
  scp "pavan@<server-ip>:~/telugu-tts/output/Voice_*" C:\Voice\
  ```

## When you finish

**Power off the node from the E2E Networks dashboard to stop billing.** Shutting down from inside
the server may not stop the charges, so check the dashboard.
