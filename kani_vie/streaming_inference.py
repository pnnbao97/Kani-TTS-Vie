"""
Streaming inference helpers inspired by the official KaniTTS examples.
"""

from __future__ import annotations

import os
import time
from collections import deque
from threading import Thread
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch
from nemo.utils.nemo_logging import Logger
from transformers.generation.streamers import BaseStreamer

STREAM_CHUNK_SIZE = 25  # frames ≈2.0 seconds of NEW audio
STREAM_LOOKBACK_FRAMES = 15  # frames ≈1.2 seconds of context
SAMPLE_RATE = 22_050

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()


class TokenIDStreamer(BaseStreamer):
    """Minimal streamer that forwards generated token ids to a callback."""

    def __init__(self, callback: Callable[[int], None]):
        super().__init__()
        self.callback = callback

    def put(self, value):
        token_ids = value[0].tolist() if len(value.shape) > 1 else value.tolist()
        for token_id in token_ids:
            self.callback(token_id)

    def end(self):
        pass


class StreamingAudioWriter:
    """Sliding window decoder with optional per-chunk callback."""

    def __init__(
        self,
        player,
        output_file: Optional[str],
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = STREAM_CHUNK_SIZE,
        lookback_frames: int = STREAM_LOOKBACK_FRAMES,
        chunk_callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.player = player
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.lookback_frames = lookback_frames
        self.chunk_callback = chunk_callback

        self.audio_chunks: list[np.ndarray] = []
        self.all_tokens: list[int] = []

        self.running = False
        self.inside_speech = False
        self.frames_decoded = 0
        self._queue: deque[int] = deque()

    def add_token(self, token: int):
        self._queue.append(token)

    def _process_queue(self):
        while self.running or self._queue:
            if not self._queue:
                time.sleep(0.01)
                continue

            token_id = self._queue.popleft()

            if token_id == self.player.start_of_speech:
                self.inside_speech = True
                continue

            if token_id == self.player.end_of_speech:
                self._flush_remaining()
                self.inside_speech = False
                continue

            if self.inside_speech:
                self.all_tokens.append(token_id)
                self._decode_if_ready()

    def _decode_if_ready(self):
        total_frames = len(self.all_tokens) // 4
        new_frames = total_frames - self.frames_decoded
        if new_frames < self.chunk_size:
            return

        start_frame = max(0, self.frames_decoded - self.lookback_frames)
        start_token = start_frame * 4
        tokens_to_decode = self.all_tokens[start_token:]
        num_frames = len(tokens_to_decode) // 4
        if num_frames == 0:
            return

        codes = np.array(tokens_to_decode[: num_frames * 4]).reshape(-1, 4)
        audio_chunk = self.player.decode_audio_chunk(codes)
        if audio_chunk is None:
            return

        samples_per_frame = len(audio_chunk) // num_frames
        lookback_skip = min(self.frames_decoded, self.lookback_frames)
        skip_samples = lookback_skip * samples_per_frame
        new_samples = self.chunk_size * samples_per_frame
        new_audio = audio_chunk[skip_samples : skip_samples + new_samples]

        self.audio_chunks.append(new_audio)
        if self.chunk_callback:
            self.chunk_callback(new_audio)
        self.frames_decoded += self.chunk_size

    def _flush_remaining(self):
        total_frames = len(self.all_tokens) // 4
        remaining_frames = total_frames - self.frames_decoded
        if remaining_frames <= 0:
            return

        start_frame = max(0, self.frames_decoded - self.lookback_frames)
        start_token = start_frame * 4
        tokens_to_decode = self.all_tokens[start_token:]
        num_frames = len(tokens_to_decode) // 4
        if num_frames == 0:
            return

        codes = np.array(tokens_to_decode[: num_frames * 4]).reshape(-1, 4)
        audio_chunk = self.player.decode_audio_chunk(codes)
        if audio_chunk is None:
            return

        samples_per_frame = len(audio_chunk) // num_frames
        lookback_skip = min(self.frames_decoded, self.lookback_frames)
        skip_samples = min(len(audio_chunk), lookback_skip * samples_per_frame)
        new_audio = audio_chunk[skip_samples:]

        if new_audio.size:
            self.audio_chunks.append(new_audio)
            if self.chunk_callback:
                self.chunk_callback(new_audio)
        self.frames_decoded = total_frames

    def start(self):
        self.running = True
        self._worker = Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def finalize(self):
        self.running = False
        if hasattr(self, "_worker"):
            self._worker.join()

        if not self.audio_chunks:
            return None
        full_audio = np.concatenate(self.audio_chunks)
        if self.output_file:
            sf.write(self.output_file, full_audio, samplerate=self.sample_rate)
        return full_audio


class StreamingKaniGenerator:
    """Wrapper around the causal LM with streaming token support."""

    def __init__(self, kani_model):
        self.kani = kani_model
        self.player = kani_model.player

    def generate(
        self,
        prompt: str,
        audio_writer: StreamingAudioWriter,
        *,
        speaker_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ):
        input_ids, attention_mask = self.kani.get_input_ids(prompt, speaker_id)
        input_ids = input_ids.to(self.kani.device)
        attention_mask = attention_mask.to(self.kani.device)

        token_ids: list[int] = []

        def on_token(token_id: int):
            token_ids.append(token_id)
            audio_writer.add_token(token_id)

        streamer = TokenIDStreamer(on_token)

        cfg = self.kani.config
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens or cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else cfg.repetition_penalty
            ),
            num_return_sequences=1,
            eos_token_id=self.player.end_of_speech,
            streamer=streamer,
        )

        start_time = time.time()
        with torch.no_grad():
            self.kani.model.generate(**generation_kwargs)
        generation_end = time.time()

        generated_text = self.kani.tokenizer.decode(token_ids, skip_special_tokens=True)

        return {
            "generated_text": generated_text,
            "token_ids": token_ids,
            "point_1": start_time,
            "point_2": generation_end,
            "generation_time": generation_end - start_time,
        }
