import soundfile as sf
import torch
from nemo.utils.nemo_logging import Logger
from tts_core import Config, NemoAudioPlayer, KaniModel
from utils.normalize_text import VietnameseTTSNormalizer

normalizer = VietnameseTTSNormalizer()

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

print(f"CUDA available: {torch.cuda.is_available()}")

config = Config()
player = NemoAudioPlayer(config)
kani = KaniModel(config, player)
prompt = "Hôm qua tôi thức dậy từ sáng sớm lúc 6 giờ, sau khi ăn sáng xong thì mở máy tính ra kiểm tra thư điện tử và trả lời một số tin nhắn quan trọng từ nhóm làm việc."
prompt = normalizer.normalize(prompt)

speaker_profiles = {
    "nam-mien-bac": "Nam miền Bắc",
    "nu-mien-nam": "Nữ miền Nam",
    "nam-mien-nam": "Nam miền Nam",
    "nu-mien-bac": "Nữ miền Bắc",
}

for speaker_id, description in speaker_profiles.items():
    print(f"\n=== Synthesizing voice: {speaker_id} — {description} ===")
    audio, text = kani.run_model(prompt, speaker_id=speaker_id)
    print(f"TEXT: {text}")
    output_path = f"audio-{speaker_id}.wav"
    sf.write(output_path, audio, samplerate=22050)
    print(f"Audio saved to {output_path}")