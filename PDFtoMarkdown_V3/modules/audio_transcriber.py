import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

import ollama
from modules.ollama_utils import is_ollama_running

_MODEL_CACHE: Dict[str, Any] = {}


def get_whisper_model(model_size: str = "base", cpu_threads: int = 8) -> Any:
    """
    Get or load a cached faster-whisper model.
    Runs on CPU using int8 quantization for ultra-fast inference.
    """
    if WhisperModel is None:
        raise ImportError(
            "faster-whisper is not installed. "
            "Please install it using: pip install faster-whisper"
        )

    cache_key = f"{model_size}_{cpu_threads}"
    if cache_key not in _MODEL_CACHE:
        print(f"Loading Whisper model '{model_size}' (CPU int8, {cpu_threads} threads)...")
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
        )
    return _MODEL_CACHE[cache_key]


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    seconds = max(0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def summarize_transcript(transcript_text: str, model: str = "qwen2.5:3b") -> Optional[str]:
    """
    Generate an executive summary of the audio transcript using local Ollama.
    """
    if not transcript_text or len(transcript_text.strip()) < 50:
        return None

    if not is_ollama_running():
        return None

    prompt = f"""You are a professional meeting and audio summarizer.
Summarize the following audio transcript clearly and concisely.

Provide:
- **Overview**: 1-2 sentence core topic/context.
- **Key Takeaways & Points**: Bullet list of main discussion items or information.
- **Action Items / Decisions**: If any actions or decisions were discussed.

Transcript:
\"\"\"
{transcript_text[:4000]}
\"\"\"
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 300, "num_thread": 8},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Transcript summarization skipped: {e}")
        return None


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "base",
    cpu_threads: int = 8,
    generate_summary: bool = True,
    summary_model: str = "qwen2.5:3b",
) -> Dict[str, Any]:
    """
    Transcribe an audio file into structured text with timestamps and metadata.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file (.mp3, .wav, .m4a, .ogg, .flac, .aac, .wma)
    model_size : str
        Whisper model size ('tiny', 'base', 'small', 'medium')
    cpu_threads : int
        Number of CPU threads to allocate for CTranslate2
    generate_summary : bool
        Whether to generate an LLM summary via Ollama
    summary_model : str
        Ollama model for summarization

    Returns
    -------
    dict
        Structured transcription result including metadata, segments, and full text.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model(model_size=model_size, cpu_threads=cpu_threads)

    t0 = time.perf_counter()

    # Transcribe with beam search and vad filter for clean segment boundaries
    segments_raw, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments = []
    full_text_parts = []

    for seg in segments_raw:
        text = seg.text.strip()
        if not text:
            continue

        start_fmt = format_timestamp(seg.start)
        end_fmt = format_timestamp(seg.end)

        segments.append({
            "start": seg.start,
            "end": seg.end,
            "start_formatted": start_fmt,
            "end_formatted": end_fmt,
            "text": text,
        })
        full_text_parts.append(text)

    t1 = time.perf_counter()
    transcription_time = t1 - t0

    full_text = " ".join(full_text_parts)
    duration = getattr(info, "duration", 0.0)
    language = getattr(info, "language", "unknown")
    language_prob = getattr(info, "language_probability", 0.0)

    summary = None
    if generate_summary and full_text:
        summary = summarize_transcript(full_text, model=summary_model)

    return {
        "filename": audio_path.name,
        "filepath": str(audio_path),
        "duration": duration,
        "duration_formatted": format_timestamp(duration),
        "language": language.upper() if language else "UNKNOWN",
        "language_probability": round(language_prob * 100, 1),
        "segment_count": len(segments),
        "segments": segments,
        "full_text": full_text,
        "summary": summary,
        "transcription_time": round(transcription_time, 2),
        "model_size": model_size,
    }
