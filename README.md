# 😻 Kani TTS Vie

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/Kani-TTS-Vie)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/kani-tts-370m-vie)

<img width="500" height="500" alt="s83RYjg6BCrSPTNNXMu4u" src="https://github.com/user-attachments/assets/11384214-379c-4872-b464-c417d3b57458" />

Mô hình chuyển văn bản thành giọng nói tiếng Việt nhanh, rõ ràng và giàu biểu cảm, với điểm mạnh là inference cực nhanh và streaming mượt mà, xây dựng trên nền tảng **Kani 370M**.
Kho lưu trữ này hỗ trợ cả **script chạy cục bộ** và các **demo UI/API** đi kèm với bản phát hành [pnnbao-ump/kani-tts-370m-vie](https://huggingface.co/pnnbao-ump/kani-tts-370m-vie) trên Hugging Face.

## Điểm nổi bật

* 🚀 **Inference nhanh** – khoảng 3 giây cho đoạn văn ngắn trên GPU đơn, hệ số thời gian thực ~0.1–0.3×.
* 🎭 **Đa giọng** – 18 giọng đọc, bao gồm Tiếng Việt, Tiếng Anh, Hàn, Đức, Tây Ban Nha, Trung và Ả Rập. Lưu ý: bạn vẫn có thể dùng các giọng nước ngoài để đọc văn bản tiếng Việt.
* 📓 **Notebooks đi kèm** – Hướng dẫn chi tiết inference, chuẩn bị dataset, và fine-tuning LoRA trong thư mục `finetune/`.

## Giọng đọc hỗ trợ

Nghe mẫu sample tại đây: **[Mẫu Giọng Demo (Inferred)]**(https://huggingface.co/pnnbao-ump/kani-tts-370m-vie)

| Ngôn ngữ          | Giọng đọc                                                                          |
| ----------------- | ---------------------------------------------------------------------------------- |
| Tiếng Việt        | Khoa (Nam Bắc), Hùng (Nam Nam), Trinh (Nữ Nam)                                     |
| Tiếng Anh         | David (British), Puck (Gemini), Kore (Gemini), Andrew, Jenny (Irish), Simon, Katie |
| Tiếng Hàn         | Seulgi                                                                             |
| Tiếng Đức         | Bert, Thorsten (Hessisch)                                                          |
| Tiếng Tây Ban Nha | Maria                                                                              |
| Tiếng Trung       | Mei (Cantonese), Ming (Shanghai)                                                   |
| Tiếng Ả Rập       | Karim, Nur                                                                         |

---

## Yêu cầu cài đặt

1.  **Python 3.12**
2.  **GPU**
    * **VRAM:** Cần **tối thiểu 3 GB VRAM** trở lên để inference (sinh giọng nói) được mượt mà và ổn định.
    * **Driver & CUDA:** Cần có Driver GPU và CUDA Toolkit tương thích với phiên bản PyTorch đang được sử dụng.
3.  **Hệ điều hành**
    * **Linux (Khuyến nghị):** Được hỗ trợ nguyên bản.
    * **Windows:** Bắt buộc phải sử dụng môi trường **WSL 2 (Windows Subsystem for Linux)**. Xem cách cài đặt tại đây: https://learn.microsoft.com/en-us/windows/wsl/install
4.  **ffmpeg**

Cài đặt dependencies:

```bash
# Dùng uv (khuyến nghị)
uv sync

# Hoặc dùng pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách sử dụng

### Local Installation

Đây là cách thiết lập môi trường cục bộ, khuyến nghị sử dụng **uv** để cài đặt và chạy:

```bash
# Clone repository
git clone https://github.com/pnnbao97/Kani-TTS-Vie
cd Kani-TTS-Vie

# Cài đặt dependencies (Sử dụng uv)
uv sync

# Chạy ứng dụng Gradio/FastAPI (ví dụ: FastAPI)
uv run uvicorn server:app
```

### Notebooks

* `kani-tts-inference.ipynb` – walkthrough chi tiết token layout, sampling, trộn giọng.
* `prepare_dataset.ipynb` – dọn dữ liệu, chuẩn hóa số, xây dựng shards.
* `finetune/kani-tts-vi-finetune.ipynb` – công thức fine-tuning LoRA.

Mở chúng bằng môi trường Jupyter sau khi kích hoạt virtual environment.

## Đóng góp

Chào đón mọi đóng góp!

1. Fork repository.
2. Tạo branch mới cho tính năng.
3. Chạy lint/tests liên quan.
4. Mở pull request mô tả cải tiến.

## Giấy phép

Dự án này phát hành theo [Apache License 2.0](LICENSE).
Kiểm tra giấy phép các mô hình/dataset bên thứ ba trước khi phân phối lại.

