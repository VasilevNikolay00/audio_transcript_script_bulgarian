import os
import gc
import sys
import torch
import warnings
import whisperx
from pathlib import Path
from whisperx.diarize import DiarizationPipeline

# --- 1. System & Performance Tweaks ---
os.environ["KMP_WARNINGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Enable Tensor Cores for NVIDIA GPUs (Optimizes speed on RTX cards)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Robust .env Loader
# ---------------------------------------------------------------------------
def load_env():
    """Manually load .env and clean values to prevent 'Empty string' errors."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        val = value.strip().strip('"').strip("'")
        os.environ[key] = val

load_env()

# ---------------------------------------------------------------------------
# Core Transcriber Class
# ---------------------------------------------------------------------------
class Transcriber:
    def __init__(self):
        # 1. Device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. Compute Type
        raw_compute = os.environ.get("COMPUTE_TYPE", "").strip()
        if not raw_compute:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = raw_compute

        # 3. Model & Language
        # .env uses WHISPER_MODEL=turbo → WhisperX expects "large-v3-turbo"
        _model_raw = os.environ.get("WHISPER_MODEL", "large-v3-turbo").strip()
        self.model_name = "large-v3-turbo" if _model_raw == "turbo" else (_model_raw or "large-v3-turbo")

        # ALIGN_MODEL: optional custom HuggingFace model for alignment
        self.align_model = os.environ.get("ALIGN_MODEL", "").strip() or None

        self.lang = os.environ.get("LANGUAGE", "bg").strip() or "bg"
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()

        # NUM_SPEAKERS: if set, fixes exact speaker count for diarization
        _num_s = os.environ.get("NUM_SPEAKERS", "").strip()
        self.num_speakers = int(_num_s) if _num_s else None

        # 4. Batch Sizes
        if self.device == "cuda":
            self.batch_size = int(os.environ.get("BATCH_SIZE_GPU", "16"))
        else:
            self.batch_size = int(os.environ.get("BATCH_SIZE_CPU", "4"))

        print(f"--- Initialization ---")
        print(f"  Device      : {self.device}")
        print(f"  Compute     : {self.compute_type}")
        print(f"  Model       : {self.model_name}")
        print(f"  Align Model : {self.align_model or '(auto)'}")
        print(f"  Language    : {self.lang}")
        print(f"  Batch Size  : {self.batch_size}")
        print(f"  Speakers    : {self.num_speakers or 'auto'}")
        print(f"----------------------\n")

    def _flush(self):
        """Force garbage collection to prevent VRAM overflow."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def load_wordlists(self, folder="wordlists"):
        """Reads all .txt files to build a vocabulary prompt for higher accuracy."""
        words = []
        path = Path(folder)
        if path.exists() and path.is_dir():
            for f in path.glob("*.txt"):
                with open(f, "r", encoding="utf-8") as file:
                    words.extend([line.strip() for line in file if line.strip()])
        
        if not words:
            return None
        
        prompt = "Това е разговор на български език. Ключови думи: " + ", ".join(list(set(words)))
        return prompt[:800]

    @torch.inference_mode()
    def run(self, audio_path, num_speakers=None):
        print(f"Processing: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        initial_prompt = None

        # --- STEP 1: TRANSCRIPTION ---
        print(f"-> Phase 1: Transcribing (Accurate Mode)...")

        # WhisperX requires initial_prompt and beam_size inside asr_options at
        # load_model time — passing them to transcribe() is silently ignored.
        asr_options = {"beam_size": 10}
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt

        model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            asr_options=asr_options,
        )

        result = model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.lang,
        )

        del model
        self._flush()

        # --- STEP 2: ALIGNMENT ---
        print("-> Phase 2: Aligning word timestamps...")
        try:
            # Use ALIGN_MODEL from .env if provided, otherwise let WhisperX auto-select
            align_kwargs = {"language_code": self.lang, "device": self.device}
            if self.align_model:
                align_kwargs["model_name"] = self.align_model

            model_a, metadata = whisperx.load_align_model(**align_kwargs)
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            del model_a
            self._flush()
        except Exception as e:
            print(f"   [!] Alignment skipped: {e}")

        # --- STEP 3: DIARIZATION ---
        print("-> Phase 3: Diarizing speakers...")
        if not self.hf_token:
            print("   [!] HF_TOKEN missing. Skipping diarization.")
            return result["segments"]

        try:
            diarize_model = DiarizationPipeline(token=self.hf_token, device=self.device)

            # CLI arg takes priority over .env NUM_SPEAKERS; fall back to min/max range
            fixed = num_speakers or self.num_speakers
            if fixed:
                min_s = max_s = fixed
            else:
                min_s = int(os.environ.get("MIN_SPEAKERS", "1"))
                max_s = int(os.environ.get("MAX_SPEAKERS", "10"))

            diarize_segments = diarize_model(audio, min_speakers=min_s, max_speakers=max_s)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            self._flush()
        except Exception as e:
            print(f"   [!] Diarization failed: {e}")

        return result["segments"]

    def format_transcript(self, segments):
        """Converts segment data into clean readable text."""
        output = []
        current_speaker = None

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            
            if speaker != current_speaker:
                start_t = seg.get("start", 0)
                m, s = divmod(int(start_t), 60)
                output.append(f"\n[{m:02d}:{s:02d}] {speaker}: ")
                current_speaker = speaker
            
            text = seg.get("text", "").strip()
            output.append(f"{text} ")

        return "".join(output).strip()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_voice.py <audio_file> [num_speakers]")
        return

    audio_file = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return

    engine = Transcriber()
    try:
        segments = engine.run(audio_file, num_speakers)
        transcript = engine.format_transcript(segments)
        
        print("\n" + "="*20 + " TRANSCRIPT " + "="*20)
        print(transcript)
        
        out_path = os.environ.get("OUTPUT_FILE", "transcript.txt")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\nSaved to: {out_path}")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
