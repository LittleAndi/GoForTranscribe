"""Transcribe audio to timestamped segments.

Stage 2 of the pipeline. Knows nothing about speakers — `merge.py` attaches those.

    uv run transcribe.py --file episode.mp3 --output transcript.json

Defaults to KBLab's Swedish Whisper fine-tune, which KBLab report at roughly 47%
lower WER than openai/whisper-large-v3 on Swedish. Pass --model for other
languages; --language must match whatever the model expects.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from common import (
    fail,
    decode,
    format_timestamp,
    load_waveform,
    make_deterministic,
    neutralize_broken_torchcodec,
    resolve_token,
    select_device,
)

DEFAULT_MODEL = "KBLab/kb-whisper-large"


def transcribe(samples, rate, args, device) -> list[dict]:
    # Must happen before transformers is imported: its ASR pipeline imports
    # torchcodec unconditionally when the package is present, and a broken install
    # would otherwise abort the run. See common.neutralize_broken_torchcodec.
    neutralize_broken_torchcodec()

    import torch
    from transformers import pipeline

    # fp16 halves memory traffic and roughly doubles throughput; Whisper is trained
    # in fp16 so this costs no accuracy. CPU has no usable fp16 path, hence fp32.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        dtype=dtype,
        token=resolve_token(args.token),
    )

    generate_kwargs = {"task": "transcribe"}
    if args.language and args.language != "auto":
        generate_kwargs["language"] = args.language

    # The dict form states the sample rate explicitly rather than relying on the
    # pipeline assuming a bare array is already at the model's rate.
    result = asr(
        {"raw": samples[0], "sampling_rate": rate},
        chunk_length_s=args.chunk_length,
        batch_size=args.batch_size,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )

    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        text = chunk.get("text", "").strip()
        if start is None or not text:
            continue
        # The final chunk can come back without an end timestamp when the audio
        # stops mid-utterance; fall back to the chunk length rather than dropping
        # what is often a whole closing sentence.
        if end is None:
            end = min(start + args.chunk_length, len(samples[0]) / rate)
        segments.append(
            {
                "start": round(start + args.offset, 3),
                "end": round(end + args.offset, 3),
                "text": text,
            }
        )
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, type=Path, help="audio file to transcribe")
    parser.add_argument("--output", type=Path, help="write segments here (.json or .txt)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--language", default="sv", help="'auto' to let the model decide")
    parser.add_argument("--offset", type=float, default=0.0, help="skip this many seconds")
    parser.add_argument("--duration", type=float, help="only process this many seconds")
    parser.add_argument("--chunk-length", type=float, default=30.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="chunks decoded in parallel; raise to use more VRAM, lower if out of memory",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--token", help="Hugging Face token (default: stored login)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args()

    if not args.file.is_file():
        fail(f"no such file: {args.file}")

    if not args.nondeterministic:
        make_deterministic(args.seed)

    device = select_device(args.device)

    with tempfile.TemporaryDirectory() as workspace:
        decoded = Path(workspace) / "audio.wav"
        decode(args.file, decoded, args.offset, args.duration)
        samples, rate = load_waveform(decoded)
        seconds = len(samples[0]) / rate
        print(f"Transcribing {format_timestamp(seconds)} with {args.model}", file=sys.stderr)

        started = time.perf_counter()
        segments = transcribe(samples, rate, args, device)
        elapsed = time.perf_counter() - started

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".txt":
            args.output.write_text(
                "\n".join(segment["text"] for segment in segments), encoding="utf-8"
            )
        else:
            args.output.write_text(json.dumps(segments, indent=2, ensure_ascii=False), "utf-8")
        print(f"Wrote {len(segments)} segments to {args.output}", file=sys.stderr)
    else:
        for segment in segments:
            print(
                f"[{format_timestamp(segment['start'])} --> "
                f"{format_timestamp(segment['end'])}] {segment['text']}"
            )

    speed = seconds / elapsed if elapsed else 0
    print(
        f"\n{len(segments)} segments in {elapsed:.1f}s ({speed:.1f}x realtime)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
