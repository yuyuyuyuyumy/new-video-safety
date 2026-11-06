FROM python:3.10-slim

# ffmpeg と必要ライブラリのインストール
RUN apt-get update && apt-get install -y ffmpeg gcc python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# CPU専用の軽量版 PyTorch を指定
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# パッケージインストール
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    ffmpeg-python \
    openai-whisper \
    transformers \
    pillow

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
