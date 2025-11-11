"""
FastAPI server that exposes REST endpoints for text-to-speech generation.

Routes:
- POST /tts        → returns a WAV file once generation finishes
- POST /stream-tts → streams PCM audio chunks while the model is generating
- GET  /health     → health check
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from nemo.utils.nemo_logging import Logger
from pydantic import BaseModel
import torch
from pathlib import Path

from kani_vie.streaming_inference import (
    SAMPLE_RATE,
    STREAM_CHUNK_SIZE,
    STREAM_LOOKBACK_FRAMES,
    StreamingAudioWriter,
    StreamingKaniGenerator,
)
from kani_vie.tts_core import Config, KaniModel, NemoAudioPlayer
from utils.normalize_text import VietnameseTTSNormalizer

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

app = FastAPI(title="Kani TTS Vie API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    normalize: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    chunk_size: Optional[int] = None
    lookback_frames: Optional[int] = None


normalizer = VietnameseTTSNormalizer()

generator: Optional[StreamingKaniGenerator] = None
player: Optional[NemoAudioPlayer] = None
kani: Optional[KaniModel] = None


@app.on_event("startup")
async def startup_event():
    global generator, player, kani
    if generator is not None:
        return

    print("🚀 Initialising Kani TTS Vie models...")
    config = Config()
    player = NemoAudioPlayer(config)
    kani = KaniModel(config, player)
    generator = StreamingKaniGenerator(kani)
    print("✅ Models ready!")


def _ensure_ready():
    if generator is None or player is None or kani is None:
        raise HTTPException(status_code=503, detail="TTS models not initialised yet.")


def _prepare_prompt(request: TTSRequest) -> str:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    if request.normalize:
        text = normalizer.normalize(text)
    return text


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "tts_initialised": generator is not None,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    _ensure_ready()
    prompt = _prepare_prompt(request)

    audio_writer = StreamingAudioWriter(
        player,
        output_file=None,
        chunk_size=request.chunk_size or STREAM_CHUNK_SIZE,
        lookback_frames=request.lookback_frames or STREAM_LOOKBACK_FRAMES,
    )
    audio_writer.start()

    try:
        generator.generate(
            prompt,
            audio_writer,
            speaker_id=request.speaker_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        audio = audio_writer.finalize()
    except Exception as exc:  # noqa: BLE001
        audio_writer.finalize()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if audio is None or len(audio) == 0:
        raise HTTPException(status_code=500, detail="No audio generated.")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="speech.wav"'}
    return Response(content=wav_buffer.read(), media_type="audio/wav", headers=headers)


@app.post("/stream-tts")
async def stream_tts(request: TTSRequest):
    _ensure_ready()
    prompt = _prepare_prompt(request)

    loop = asyncio.get_running_loop()
    chunk_queue: asyncio.Queue[tuple[str, Optional[np.ndarray | str]]] = asyncio.Queue()

    def on_chunk(chunk: np.ndarray):
        loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", chunk))

    audio_writer = StreamingAudioWriter(
        player,
        output_file=None,
        chunk_size=request.chunk_size or STREAM_CHUNK_SIZE,
        lookback_frames=request.lookback_frames or STREAM_LOOKBACK_FRAMES,
        chunk_callback=on_chunk,
    )

    def run_generation():
        try:
            audio_writer.start()
            generator.generate(
                prompt,
                audio_writer,
                speaker_id=request.speaker_id,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            )
            audio_writer.finalize()
            loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None))
        except Exception as exc:  # noqa: BLE001
            audio_writer.finalize()
            loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", str(exc)))

    loop.run_in_executor(None, run_generation)

    async def chunk_stream():
        try:
            while True:
                msg_type, payload = await chunk_queue.get()
                if msg_type == "chunk" and isinstance(payload, np.ndarray):
                    pcm = np.clip(payload, -1.0, 1.0)
                    pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
                    length_prefix = struct.pack("<I", len(pcm_bytes))
                    yield length_prefix + pcm_bytes
                elif msg_type == "done":
                    yield struct.pack("<I", 0)
                    break
                elif msg_type == "error":
                    yield struct.pack("<I", 0xFFFFFFFF)
                    break
        finally:
            # Ensure any background processing settles even if the client disconnects.
            audio_writer.running = False

    headers = {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Channels": "1",
        "X-Bit-Depth": "16",
    }
    return StreamingResponse(chunk_stream(), media_type="application/octet-stream", headers=headers)


@app.get("/")
async def root():
    index_path = Path(__file__).resolve().parent / "client" / "index.html"
    if not index_path.exists():
        return {
            "name": "Kani TTS Vie API",
            "version": "1.0.0",
            "endpoints": {
                "/tts": "POST - Generate complete audio (WAV)",
                "/stream-tts": "POST - Stream PCM chunks",
                "/health": "GET - Health check",
            },
        }
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn

    print("🎤 Starting Kani TTS Vie server on http://0.0.0.0:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

