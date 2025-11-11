![Kani TTS Vie](public/logo.png)

# 😻 Kani TTS Vie

Fast and expressive Vietnamese text-to-speech built on top of the Kani 370M family.  
This repository powers both local inference scripts and the UI/API demos that accompany the
[pnnbao-ump/kani-tts-370m-vie](https://huggingface.co/pnnbao-ump/kani-tts-370m-vie) release on Hugging Face.

## Highlights

- 🚀 **Fast inference** – ~3 s for short paragraphs on a single GPU, real-time factor around 0.1–0.3×.
- 🎭 **Multi-speaker** – 18 curated voices spanning Vietnamese, English, Korean, German, Spanish, Chinese, and Arabic.
- 🧩 **Composable components** – `gradio_app.py` for a sleek non-streaming demo, `server.py` for FastAPI (streaming + batch), and `client/index.html` for a lightweight web UI.
- 📓 **Notebooks included** – End-to-end inference, dataset preparation, and LoRA fine-tuning workflows inside `finetune/`.

## Supported Voices

| Locale | Voices |
| ------ | ------ |
| Vietnamese | Khoa (north male), Hùng (south male), Trinh (south female) |
| English | David (British), Puck (Gemini), Kore (Gemini), Andrew, Jenny (Irish), Simon, Katie |
| Korean | Seulgi |
| German | Bert, Thorsten (Hessisch) |
| Spanish | Maria |
| Chinese | Mei (Cantonese), Ming (Shanghai) |
| Arabic | Karim, Nur |
| Neutral | No speaker ID (`None`) |

> Streaming is not exposed inside the Gradio demo.  
> For a full streaming experience use the reference implementation at [pnnbao97/Kani-TTS-Vie](https://github.com/pnnbao97/Kani-TTS-Vie).

## Repository Layout

- `main.py` – simple CLI inference script (batch mode).
- `gradio_app.py` – polished Gradio Blocks demo with animated loader + multi-language voices.
- `server.py` – FastAPI service exposing `/tts` and `/stream-tts`.
- `client/index.html` – static frontend that talks to the FastAPI server.
- `kani_vie/` – core model orchestration, streaming helpers, and audio player utilities.
- `finetune/` – notebooks for LoRA training and dataset preparation.
- `requirements.txt` / `pyproject.toml` – dependency manifests (pip or uv).

## Prerequisites

1. **Python 3.12** (or the version pinned in `.python-version`).
2. **GPU drivers + CUDA** compatible with your PyTorch install.
3. **ffmpeg** (optional but recommended for audio tooling).
4. **Hugging Face access token** with rights to download the base checkpoints.

Install dependencies using either `uv` (recommended) or `pip`:

```bash
# Using uv
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### 1. Command-line inference

```bash
uv run python main.py \
  --text "Xin chào! Tôi là Kani TTS." \
  --speaker_id "nam-mien-nam"
```

This writes WAV files to disk for each requested speaker.

### 2. Gradio demo (non-streaming)

```bash
uv run python gradio_app.py
```

Open the reported URL (default `http://127.0.0.1:7860`).  
The app auto-normalises text, estimates run time, and previews progress with a custom equaliser animation.

### 3. FastAPI server + static web client

Run the API:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Serve the vanilla frontend (for example):

```bash
python -m http.server 3000 --directory client
```

The client exposes both `/tts` (batch) and `/stream-tts` (chunked PCM) flows backed by the FastAPI service.

### 4. Notebooks

- `kani-tts-inference.ipynb` – detailed walkthrough of token layout, sampling parameters, and speaker mixing.
- `prepare_dataset.ipynb` – data cleaning, number normalisation, shard building.
- `finetune/kani-tts-vi-finetune.ipynb` – LoRA-based fine-tuning recipe.

Launch them with your favourite Jupyter environment after activating the virtual environment.

## Tips & Troubleshooting

- **Slow streaming?** Try decreasing `chunk_size` or running on a faster disk/GPU.
- **Non-Vietnamese inference** still works; simply pick the relevant speaker (e.g., `Seulgi` for Korean).
- **Environment warnings** about `gradio` or `soundfile` usually mean the virtual environment is missing those packages—run `pip install -r requirements.txt`.

## Contributing

Contributions are welcome!  

1. Fork the repository.
2. Create a feature branch.
3. Run linting/tests relevant to your changes.
4. Open a pull request describing the improvement.

## License

This project is released under the [Apache License 2.0](LICENSE) unless noted otherwise.  
Please review third-party model and dataset licenses before redistribution.

