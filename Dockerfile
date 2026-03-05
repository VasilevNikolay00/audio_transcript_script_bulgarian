FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperx.git

    
COPY transcribe_voice.py .
COPY .env .

ENTRYPOINT ["python", "transcribe_voice.py"]