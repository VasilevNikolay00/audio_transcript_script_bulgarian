import os
import warnings

# --- Must be before ANY other imports ---
os.environ["KMP_WARNINGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import gc
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import whisperx
import sys
from whisperx.diarize import DiarizationPipeline
from pathlib import Path


# ---------------------------------------------------------------------------
# .env loader (no external deps — works without python-dotenv installed)
# ---------------------------------------------------------------------------
def load_env_file(env_path: str = ".env"):
    """Load key=value pairs from a .env file into os.environ."""
    path = Path(env_path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Only set if not already present in the environment
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(".env")

# ---------------------------------------------------------------------------
# Config — read from environment (populated from .env above)
# ---------------------------------------------------------------------------
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
ALIGN_MODEL    = os.environ.get("ALIGN_MODEL", "infinitejoy/wav2vec2-large-xls-r-300m-bulgarian")
LANGUAGE       = os.environ.get("LANGUAGE", "bg") or "bg"   # guard against empty string
OUTPUT_FILE    = os.environ.get("OUTPUT_FILE", "transcript.txt")
MIN_SPEAKERS   = int(os.environ.get("MIN_SPEAKERS", "1"))
MAX_SPEAKERS   = int(os.environ.get("MAX_SPEAKERS", "10"))
BATCH_SIZE_GPU = int(os.environ.get("BATCH_SIZE_GPU", "16"))
BATCH_SIZE_CPU = int(os.environ.get("BATCH_SIZE_CPU", "4"))

_compute_type_env  = os.environ.get("COMPUTE_TYPE", "").strip()
_num_speakers_env  = os.environ.get("NUM_SPEAKERS", "").strip()
NUM_SPEAKERS_DEFAULT = int(_num_speakers_env) if _num_speakers_env else None

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN is not set.\n"
        "Add it to your .env file or run: export HF_TOKEN='hf_...'"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    if seconds is None:
        return "[00:00]"
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"


def resolve_compute_type(device: str) -> str:
    if _compute_type_env:
        return _compute_type_env
    return "float16" if device == "cuda" else "int8"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def transcribe_and_diarize(audio_file_path: str, num_speakers: int = None):
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = resolve_compute_type(device)
    batch_size   = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU

    if num_speakers is None:
        num_speakers = NUM_SPEAKERS_DEFAULT

    print(f"--- Config ---")
    print(f"  device       : {device} ({compute_type})")
    print(f"  whisper model: {WHISPER_MODEL}")
    print(f"  align model  : {ALIGN_MODEL}")
    print(f"  language     : {LANGUAGE}")
    print(f"  batch size   : {batch_size}")
    print(f"  num_speakers : {num_speakers if num_speakers else 'auto'}")
    print(f"  TF32         : enabled")
    print()

    # 1. Load audio
    audio = whisperx.load_audio(audio_file_path)

    # 2. Transcribe
    print(f"--- Step 1: Transcribing ({WHISPER_MODEL}) ---")
    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language=LANGUAGE)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Align (word-level timestamps)
    print(f"--- Step 2: Aligning ({ALIGN_MODEL}) ---")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=LANGUAGE,
            device=device,
            model_name=ALIGN_MODEL,
        )
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Alignment error (continuing without it): {e}")
        if "segments" not in result:
            result = {"segments": result}

    # 4. Diarize
    print("--- Step 3: Diarizing ---")
    try:
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)

        if num_speakers is not None:
            diarize_segments = diarize_model(
                audio,
                min_speakers=num_speakers,
                max_speakers=num_speakers,
            )
        else:
            diarize_segments = diarize_model(
                audio,
                min_speakers=MIN_SPEAKERS,
                max_speakers=MAX_SPEAKERS,
            )

        result = whisperx.assign_word_speakers(diarize_segments, result)
    except Exception as e:
        print(f"Diarization failed: {e}")

    return result.get("segments", result)


# ---------------------------------------------------------------------------
# Transcript formatter
# ---------------------------------------------------------------------------

def build_transcript(segments: list) -> str:
    final_output = ""
    current_speaker = None

    for seg in segments:
        if "words" in seg and seg["words"]:
            for word in seg["words"]:
                speaker    = word.get("speaker", "UNKNOWN")
                text       = word.get("word", "").strip()
                start_time = word.get("start")

                if not text:
                    continue

                if speaker != current_speaker:
                    time_str = format_timestamp(start_time)
                    final_output += f"\n{time_str} {speaker}: "
                    current_speaker = speaker

                final_output += f"{text} "
        else:
            speaker = seg.get("speaker", "UNKNOWN")
            text    = seg.get("text", "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                final_output += f"\n{format_timestamp(seg.get('start'))} {speaker}: "
                current_speaker = speaker
            final_output += f"{text} "

    return final_output.strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_voice.py <audio_file> [num_speakers]")
        print("  num_speakers (optional): overrides NUM_SPEAKERS in .env")
        return

    audio_path   = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return

    try:
        segments   = transcribe_and_diarize(audio_path, num_speakers=num_speakers)

        print("\n" + "=" * 20 + " FINAL TRANSCRIPT " + "=" * 20)
        transcript = build_transcript(segments)
        print(transcript)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\nTranscript saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()