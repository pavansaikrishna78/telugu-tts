#!/usr/bin/env python
"""Local web UI, bound to 127.0.0.1 only.

  python app.py
From Windows: ssh -L 7860:localhost:7860 pavan@<server-ip>, then open http://localhost:7860
"""
import time

import gradio as gr

from tts_engine import DEFAULT_OUTPUT, Engine, list_voices

HOST = "127.0.0.1"
PORT = 7860
UI_DIR = DEFAULT_OUTPUT / "ui"
engine = Engine()


def speak(text: str, voice: str):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    if not voice:
        raise gr.Error("No voice available. Run prep_ref.py first.")
    stem = time.strftime("say_%Y%m%d_%H%M%S")
    try:
        result = engine.render(text, UI_DIR / f"{stem}.wav", voice=voice, chunk_root=UI_DIR / "chunks")
    except ValueError as error:
        raise gr.Error(str(error)) from error
    files = [str(result.wav)] + ([str(result.mp3)] if result.mp3 else [])
    return str(result.wav), files, (f"{result.audio_seconds:.1f} s of audio, {result.chunks} chunk(s), "
                                    f"{result.generation_seconds:.1f} s to generate")


def build() -> gr.Blocks:
    voices = list_voices()
    with gr.Blocks(title="Telugu TTS") as demo:
        gr.Markdown("# Telugu TTS")
        text = gr.Textbox(label="Telugu text", lines=6)
        voice = gr.Dropdown(voices, value=("pavan" if "pavan" in voices else (voices[0] if voices else None)),
                            label="Voice")
        button = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Result", type="filepath")
        files = gr.File(label="Download", file_count="multiple")
        info = gr.Markdown()
        button.click(speak, [text, voice], [audio, files, info])
    return demo


if __name__ == "__main__":
    UI_DIR.mkdir(parents=True, exist_ok=True)
    build().launch(server_name=HOST, server_port=PORT, share=False, allowed_paths=[str(UI_DIR)])
