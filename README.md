Python 3.12 

1. Create .env using the example template

2. Run the given build command.

docker build --no-cache \
  --build-arg HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2) \
  --build-arg WHISPER_MODEL=$(grep WHISPER_MODEL .env | cut -d '=' -f2) \
  --build-arg WAV2VEC2_MODEL=$(grep ALIGN_MODEL .env | cut -d '=' -f2) \
  -t voice_transcriber:latest .

3. Run the given run command. Please set the correct param for the audio

docker run --rm --gpus all \
  --env-file .env \
  -v $(pwd)/audio_data:/app/audio_data \
  -v $(pwd)/output:/app/output \
  voice_transcriber:latest \
  audio_data/{the audio/video file}
