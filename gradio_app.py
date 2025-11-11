import os
import tempfile
import time
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf

from kani_vie.tts_core import Config, KaniModel, NemoAudioPlayer
from utils.normalize_text import VietnameseTTSNormalizer

SPEAKER_CHOICES = [
    ("Khoa – Nam miền Bắc", "nam-mien-bac"),
    ("Hùng – Nam miền Nam", "nam-mien-nam"),
    ("Trinh – Nữ miền Nam", "nu-mien-nam"),
    ("David – English (British)", "david"),
    ("Puck – English (Gemini)", "puck"),
    ("Kore – English (Gemini)", "kore"),
    ("Andrew – English", "andrew"),
    ("Jenny – English (Irish)", "jenny"),
    ("Simon – English", "simon"),
    ("Katie – English", "katie"),
    ("Seulgi – Korean", "seulgi"),
    ("Bert – German", "bert"),
    ("Thorsten – German (Hessisch)", "thorsten"),
    ("Maria – Spanish", "maria"),
    ("Mei – Chinese (Cantonese)", "mei"),
    ("Ming – Chinese (Shanghai OpenAI)", "ming"),
    ("Karim – Arabic", "karim"),
    ("Nur – Arabic", "nur"),
    ("Không chỉ định", None),
]

def _init_models():
    config = Config()
    player = NemoAudioPlayer(config)
    kani = KaniModel(config, player)
    return config, player, kani

CONFIG, PLAYER, KANI_MODEL = _init_models()
NORMALIZER = VietnameseTTSNormalizer()
SAMPLE_RATE = 22050

def _save_audio(audio: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, audio.astype(np.float32), SAMPLE_RATE)
    return path

def _run_standard(text: str, speaker_id: Optional[str]) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    audio, _ = KANI_MODEL.run_model(text, speaker_id=speaker_id)
    elapsed = time.perf_counter() - start
    return audio, elapsed

def synthesize(
    text: str, speaker_label: str, normalize: bool = True
):
    text = (text or "").strip()
    if not text:
        yield (
            None, 
            "⚠️ Vui lòng nhập nội dung cần đọc.", 
            None, 
            gr.update(visible=False),
            gr.update(interactive=True)  # Re-enable button
        )
        return

    speaker_id = dict(SPEAKER_CHOICES).get(speaker_label, None)
    
    # Estimate processing time
    char_count = len(text)
    estimated_time = max(2, char_count / 40)
    
    # Show loading state
    yield (
        None,
        f"⏳ Đang xử lý văn bản ({char_count} ký tự)...",
        None,
        gr.update(visible=True),
        gr.update(interactive=False)  # Disable button
    )
    
    processed_text = NORMALIZER.normalize(text)
    
    yield (
        None,
        f"🎵 Đang tạo giọng nói (ước tính ~{estimated_time:.0f}s)...",
        None,
        gr.update(visible=True),
        gr.update(interactive=False)
    )
    
    try:
        audio, elapsed = _run_standard(processed_text, speaker_id)
    except Exception as exc:
        yield (
            None,
            f"❌ Lỗi khi suy luận: {exc}",
            None,
            gr.update(visible=False),
            gr.update(interactive=True)  # Re-enable on error
        )
        return

    if audio is None or len(audio) == 0:
        yield (
            None,
            "⚠️ Không tạo được audio đầu ra.",
            None,
            gr.update(visible=False),
            gr.update(interactive=True)  # Re-enable on error
        )
        return

    wav_path = _save_audio(audio)
    duration = len(audio) / SAMPLE_RATE
    yield (
        wav_path,
        f"✅ Hoàn tất sau {elapsed:.2f}s | Độ dài audio: {duration:.1f}s | RTF: {elapsed/duration:.2f}x",
        wav_path,
        gr.update(visible=False),
        gr.update(interactive=True)  # Re-enable after success
    )

def build_interface():
    css = """
    /* Thay đổi màu chính của Gradio sang xanh ngọc bích */
    .primary {
        background: linear-gradient(90deg, #14b8a6 0%, #06b6d4 100%) !important;
        border: none !important;
    }
    
    button.primary, .primary button {
        background: linear-gradient(90deg, #14b8a6 0%, #06b6d4 100%) !important;
        border: none !important;
    }
    
    .gradio-button.primary {
        background: linear-gradient(90deg, #14b8a6 0%, #06b6d4 100%) !important;
    }
    
    .eq-wrapper {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 18px 28px 20px 28px;
        margin: 16px auto 0 auto;
        width: min(260px, 90%);
        border-radius: 18px;
        background: radial-gradient(circle at top, rgba(20, 184, 166, 0.32), rgba(6, 182, 212, 0.12));
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 24px 40px rgba(6, 182, 212, 0.18);
        backdrop-filter: blur(12px);
        overflow: hidden;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .eq-wrapper::after {
        content: "";
        position: absolute;
        inset: -30%;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.14), transparent 55%);
        filter: blur(18px);
        z-index: 0;
        animation: rotate 8s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .eq-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #d1fae5;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        z-index: 1;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .eq-loader {
        display: flex;
        gap: 8px;
        align-items: flex-end;
        justify-content: center;
        height: 64px;
        width: 170px;
        z-index: 1;
    }
    
    .eq-bar {
        width: 10px;
        background: linear-gradient(180deg, rgba(20, 184, 166, 0.9) 0%, rgba(6, 182, 212, 0.9) 100%);
        border-radius: 12px;
        animation: kani-bounce 1.1s ease-in-out infinite;
        box-shadow: 
            0 0 18px rgba(6, 182, 212, 0.55),
            0 0 36px rgba(20, 184, 166, 0.3),
            inset 0 0 12px rgba(255, 255, 255, 0.2);
    }
    
    .eq-bar:nth-child(2) { animation-delay: 0.15s; }
    .eq-bar:nth-child(3) { animation-delay: 0.3s; }
    .eq-bar:nth-child(4) { animation-delay: 0.45s; }
    .eq-bar:nth-child(5) { animation-delay: 0.6s; }
    
    @keyframes kani-bounce {
        0%, 100% { height: 20%; opacity: 0.4; transform: scaleX(1); }
        50% { height: 100%; opacity: 1; transform: scaleX(1.1); }
    }
    
    .eq-hint {
        font-size: 0.82rem;
        color: rgba(209, 250, 229, 0.75);
        letter-spacing: 0.015em;
        z-index: 1;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.75; }
        50% { opacity: 1; }
    }
    
    /* Style cho button khi disabled */
    button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
    
    /* Style cho status messages */
    .status-processing {
        color: #14b8a6;
        animation: statusPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes statusPulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    """

    with gr.Blocks(title="Kani TTS Vie Demo", css=css) as demo:
        gr.Markdown(
            "# 😻 Kani TTS Vie – Fast and Expressive Speech Generation Model that supports Vietnamese\n\n"
            "🚫 Bản demo Gradio hiện không hỗ trợ streaming trực tiếp.\n"
            "🔗 Để trải nghiệm streaming thực sự, vui lòng sử dụng mã nguồn tại [pnnbao97/Kani-TTS-Vie](https://github.com/pnnbao97/Kani-TTS-Vie).\n\n"
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="📝 Nội dung",
                    lines=6,
                    placeholder="Nhập văn bản cần chuyển đổi thành giọng nói...",
                    value="Khi bạn kề vai sát cánh cùng đồng đội của mình, bạn có thể làm nên những điều phi thường."
                )
                
                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        choices=[label for label, _ in SPEAKER_CHOICES],
                        value="Hùng – Nam miền Nam",
                        label="🎤 Giọng đọc",
                    )
                run_button = gr.Button(
                    "🎵 Tạo giọng nói", 
                    variant="primary",
                    size="lg"
                )
        
        # Loading indicator
        loader_output = gr.HTML(
            value=(
                "<div class='eq-wrapper'>"
                "<span class='eq-title'>Đang chuẩn bị audio</span>"
                "<div class='eq-loader'>"
                "<span class='eq-bar'></span>"
                "<span class='eq-bar'></span>"
                "<span class='eq-bar'></span>"
                "<span class='eq-bar'></span>"
                "<span class='eq-bar'></span>"
                "</div>"
                "<span class='eq-hint'>Vui lòng đợi một chút...</span>"
                "</div>"
            ),
            visible=False,
        )
        
        status_output = gr.Markdown()
        
        with gr.Row():
            audio_output = gr.Audio(
                label="🔊 Âm thanh",
                autoplay=True,
                streaming=False,
            )
        
        download_output = gr.File(
            label="💾 Tải WAV", 
            interactive=False
        )

        # Event handler with button state management
        run_button.click(
            fn=synthesize,
            inputs=[text_input, speaker_dropdown],
            outputs=[audio_output, status_output, download_output, loader_output, run_button],
        )

    demo.queue()
    return demo

demo = build_interface()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))