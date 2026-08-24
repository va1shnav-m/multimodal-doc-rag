import time
from pathlib import Path
from typing import Dict, Any, Optional

from modules.audio_transcriber import transcribe_audio
from modules.ui import ConsoleUI


def format_audio_markdown(result: Dict[str, Any]) -> str:
    """Format structured transcription dictionary into clean Markdown."""
    lines = []

    lines.append(f"# Audio Transcript: {result['filename']}")
    lines.append("")

    # ----------------------------------------------------
    # Metadata Section
    # ----------------------------------------------------
    lines.append("## Metadata")
    lines.append(f"- **Filename**: `{result['filename']}`")
    lines.append(f"- **Duration**: `{result['duration_formatted']}` ({result['duration']:.1f}s)")
    lines.append(f"- **Detected Language**: {result['language']} ({result['language_probability']}% confidence)")
    lines.append(f"- **Transcription Model**: Whisper `{result['model_size']}` (CPU int8)")
    lines.append(f"- **Processing Time**: `{result['transcription_time']}s`")
    lines.append("")

    # ----------------------------------------------------
    # Executive Summary (if available)
    # ----------------------------------------------------
    if result.get("summary"):
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(result["summary"].strip())
        lines.append("")

    # ----------------------------------------------------
    # Timestamped Segments
    # ----------------------------------------------------
    if result.get("segments"):
        lines.append("## Timestamped Transcript")
        lines.append("")
        for seg in result["segments"]:
            lines.append(
                f"- **`[{seg['start_formatted']} - {seg['end_formatted']}]`**: {seg['text']}"
            )
        lines.append("")

    # ----------------------------------------------------
    # Full Transcript
    # ----------------------------------------------------
    lines.append("## Full Transcript")
    lines.append("")
    if result.get("full_text"):
        lines.append(result["full_text"].strip())
    else:
        lines.append("*No speech detected.*")
    lines.append("")

    return "\n".join(lines)


def process_audio_pipeline(
    audio_path: str | Path,
    output_dir: str | Path,
    ui: Optional[ConsoleUI] = None,
    whisper_model: str = "base",
    generate_summary: bool = True,
    summary_model: str = "qwen2.5:3b",
) -> Dict[str, Any]:
    """
    End-to-end pipeline to convert an audio file to structured Markdown.

    Parameters
    ----------
    audio_path : str | Path
    output_dir : str | Path
    ui : ConsoleUI, optional
    whisper_model : str
    generate_summary : bool
    summary_model : str

    Returns
    -------
    dict
        Pipeline execution statistics and output paths.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ui is None:
        ui = ConsoleUI()

    start_total = time.perf_counter()

    ui.info(f"Transcribing Audio: {audio_path.name}")

    result = transcribe_audio(
        audio_path=audio_path,
        model_size=whisper_model,
        cpu_threads=8,
        generate_summary=generate_summary,
        summary_model=summary_model,
    )

    ui.success(
        f"Transcribed {result['duration_formatted']} in {result['transcription_time']}s "
        f"[{result['language']} - {result['segment_count']} segments]"
    )

    markdown_content = format_audio_markdown(result)

    output_file = output_dir / f"{audio_path.stem}.md"
    output_file.write_text(markdown_content, encoding="utf-8")

    end_total = time.perf_counter()
    total_time = end_total - start_total

    ui.success(f"Saved transcript to: {output_file.name}")

    return {
        "output_path": output_file,
        "filename": audio_path.name,
        "duration": result["duration"],
        "duration_formatted": result["duration_formatted"],
        "transcription_time": result["transcription_time"],
        "total_time": round(total_time, 2),
        "language": result["language"],
        "segment_count": result["segment_count"],
        "model": result["model_size"],
    }
