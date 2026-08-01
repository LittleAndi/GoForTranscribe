"""Audio in, speaker-attributed transcript out.

    uv run pipeline.py --file episode.mp3 --output episode.txt

Runs the three stages in one process: diarize, transcribe, merge. The separate
tools remain the better choice while iterating, since each stage's output can be
inspected and the expensive stages need not be repeated. This is for when the
pipeline is just meant to run.

Diarization is released from the GPU before transcription loads, so peak memory
is whichever stage is larger rather than the sum of both.
"""

import argparse
import json
from pathlib import Path

import diarize
import transcribe as transcribe_stage
from common import fail, format_timestamp, make_deterministic, resolve_token, select_device
from merge import UNKNOWN, collapse, merge, render


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, type=Path, help="audio file to process")
    parser.add_argument("--output", type=Path, help="write the transcript here; default stdout")
    parser.add_argument("--speakers", type=int, help="exact count; see CLAUDE.md before using")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--language", default="sv", help="'auto' to let the model decide")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument(
        "--keep-intermediate",
        type=Path,
        help="directory to save turns.json and transcript.json into",
    )
    parser.add_argument("--diarization-model", default=diarize.DEFAULT_MODEL)
    parser.add_argument("--asr-model", default=transcribe_stage.DEFAULT_MODEL)
    parser.add_argument("--chunk-length", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args()

    if not args.file.is_file():
        fail(f"no such file: {args.file}")

    token = resolve_token(args.token)
    if not token:
        fail("no Hugging Face token", hint=diarize.TOKEN_HINT)

    if not args.nondeterministic:
        make_deterministic(args.seed)
    device = select_device(args.device)

    print("[1/3] Diarizing", flush=True)
    constraints = {
        key: value
        for key, value in (
            ("num_speakers", args.speakers),
            ("min_speakers", args.min_speakers),
            ("max_speakers", args.max_speakers),
        )
        if value
    }
    pipeline = diarize.load_pipeline(args.diarization_model, token, device)
    turns = diarize.run(
        pipeline,
        args.file,
        offset=args.offset,
        duration=args.duration,
        constraints=constraints,
    )
    share = diarize.shares(turns)
    print(f"      {len(share)} speaker(s), {len(turns)} turns")

    del pipeline
    free_gpu()

    print("[2/3] Transcribing", flush=True)
    import tempfile

    from common import decode, load_waveform

    with tempfile.TemporaryDirectory() as workspace:
        decoded = Path(workspace) / "audio.wav"
        decode(args.file, decoded, args.offset, args.duration)
        samples, rate = load_waveform(decoded)
        transcribe_args = argparse.Namespace(
            model=args.asr_model,
            language=args.language,
            offset=args.offset,
            chunk_length=args.chunk_length,
            batch_size=args.batch_size,
            token=args.token,
        )
        segments = transcribe_stage.transcribe(samples, rate, transcribe_args, device)
    print(f"      {len(segments)} segments")

    print("[3/3] Merging", flush=True)
    merged = merge(turns, segments)
    blocks = collapse(merged)

    if args.keep_intermediate:
        args.keep_intermediate.mkdir(parents=True, exist_ok=True)
        (args.keep_intermediate / "turns.json").write_text(
            json.dumps(turns, indent=2), encoding="utf-8"
        )
        (args.keep_intermediate / "transcript.json").write_text(
            json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"      intermediates in {args.keep_intermediate}")

    text = render(blocks, args.timestamps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {len(blocks)} blocks to {args.output}")
    else:
        print()
        print(text)

    spoken = sum(turn["end"] - turn["start"] for turn in turns)
    unattributed = sum(1 for segment in merged if segment["speaker"] == UNKNOWN)
    print(
        f"\n{len(share)} speaker(s), {format_timestamp(spoken)} of speech, "
        f"{len(blocks)} blocks, {unattributed} unattributed segments"
    )


if __name__ == "__main__":
    main()
