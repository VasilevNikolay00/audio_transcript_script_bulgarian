Python 3.12 


docker build --no-cache -t voice_transcriber:latest .

docker run --rm --gpus all \
  --env-file .env \
  -v $(pwd)/audio_data:/app/audio_data \
  -v $(pwd)/output:/app/output \
  voice_transcriber:latest \
  audio_data/video.mp4

