FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperx.git

ARG HF_TOKEN
ARG WHISPER_MODEL
ARG WAV2VEC2_MODEL

RUN python -c "\
import whisperx; \
whisperx.load_model('${WHISPER_MODEL}', 'cpu', compute_type='int8'); \
print('Whisper model cached.')"

RUN python -c "\
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
Wav2Vec2ForCTC.from_pretrained('${WAV2VEC2_MODEL}'); \
Wav2Vec2Processor.from_pretrained('${WAV2VEC2_MODEL}'); \
print('Alignment model cached.')"

RUN python -c "\
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token='${HF_TOKEN}'); \
print('Diarization model cached.')"

COPY transcribe_voice.py .
COPY .env .

ENTRYPOINT ["python", "transcribe_voice.py"]
