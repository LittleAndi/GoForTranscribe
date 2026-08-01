"""Audio in, speaker-attributed transcript out.

    uv run pipeline.py --file episode.mp3 --output episode.txt
    uv run pipeline.py --folder podcasts\ --timestamps

With --folder, every audio file in the directory is processed and written beside
it as <name>.txt, or into --output-dir. Already-transcribed files are skipped
unless --overwrite is given, so an interrupted batch resumes where it stopped.

A batch loads each model once rather than once per file, which is why it works
in two passes — diarize everything, then transcribe everything — instead of
running both stages per file. On a folder of ten episodes that is the difference
between two model loads and twenty.

The separate stage tools remain the better choice while iterating, since each
stage's output can be inspected and the expensive stages need not be repeated.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import diarize
import transcribe as asr_stage
from common import (
    check_vram,
    decode,
    fail,
    format_timestamp,
    load_waveform,
    make_deterministic,
    resolve_token,
    select_device,
)
from merge import UNKNOWN, collapse, merge, render

# Containers ffmpeg will decode. Video files are included because a recording is
# often delivered as mp4, and we only ever take the audio stream.
AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac",
    ".mp4", ".mkv", ".webm", ".mov", ".aiff", ".aif",
}


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def discover(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def destination(source: Path, output_dir: Path | None, root: Path) -> Path:
    if output_dir is None:
        return source.with_suffix(".txt")
    # Mirror any subdirectory structure rather than flattening it, so two files
    # with the same name in different folders cannot overwrite each other.
    relative = source.relative_to(root).with_suffix(".txt")
    return output_dir / relative


def write_result(turns, segments, target: Path, args) -> dict:
    merged = merge(turns, segments)
    blocks = collapse(merged)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(blocks, args.timestamps) + "\n", encoding="utf-8")

    if args.keep_intermediate:
        folder = args.keep_intermediate / target.stem
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "turns.json").write_text(json.dumps(turns, indent=2), encoding="utf-8")
        (folder / "transcript.json").write_text(
            json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "blocks": len(blocks),
        "unattributed": sum(1 for segment in merged if segment["speaker"] == UNKNOWN),
        "speakers": len(diarize.shares(turns)),
        "speech": sum(turn["end"] - turn["start"] for turn in turns),
    }


def read_audio(path: Path, offset: float, duration: float | None):
    with tempfile.TemporaryDirectory() as workspace:
        decoded = Path(workspace) / "audio.wav"
        decode(path, decoded, offset, duration)
        return load_waveform(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="a single audio file")
    source.add_argument("--folder", type=Path, help="a directory of audio files")
    parser.add_argument("--output", type=Path, help="output path; --file only")
    parser.add_argument("--output-dir", type=Path, help="where .txt files go; --folder only")
    parser.add_argument("--recursive", action="store_true", help="descend into subdirectories")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redo files that already have a .txt beside them",
    )
    parser.add_argument("--speakers", type=int, help="exact count; see CLAUDE.md before using")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--language", default="sv", help="'auto' to let the model decide")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--keep-intermediate", type=Path, help="also save per-stage JSON")
    parser.add_argument("--diarization-model", default=diarize.DEFAULT_MODEL)
    parser.add_argument("--asr-model", default=asr_stage.DEFAULT_MODEL)
    parser.add_argument(
        "--chunk-length",
        type=float,
        help="chunked long-form decoding: faster, less accurate, much more VRAM",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args()

    if args.file and args.output_dir:
        fail("--output-dir applies to --folder; use --output for a single file")
    if args.folder and args.output:
        fail("--output applies to --file; use --output-dir for a folder")

    if args.file:
        if not args.file.is_file():
            fail(f"no such file: {args.file}")
        root = args.file.parent
        files = [args.file]
    else:
        if not args.folder.is_dir():
            fail(f"no such directory: {args.folder}")
        root = args.folder
        files = discover(args.folder, args.recursive)
        if not files:
            fail(
                f"no audio files in {args.folder}",
                hint=f"Looked for: {', '.join(sorted(AUDIO_SUFFIXES))}",
            )

    targets = {
        path: (args.output if args.file and args.output else destination(path, args.output_dir, root))
        for path in files
    }

    if not args.overwrite:
        pending = [path for path in files if not targets[path].exists()]
        skipped = len(files) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already transcribed; --overwrite to redo them")
        files = pending
    if not files:
        print("Nothing to do.")
        return

    token = resolve_token(args.token)
    if not token:
        fail("no Hugging Face token", hint=diarize.TOKEN_HINT)

    if not args.nondeterministic:
        make_deterministic(args.seed)
    device = select_device(args.device)
    check_vram(device)

    constraints = {
        key: value
        for key, value in (
            ("num_speakers", args.speakers),
            ("min_speakers", args.min_speakers),
            ("max_speakers", args.max_speakers),
        )
        if value
    }

    started = time.perf_counter()
    failures: list[tuple[Path, str]] = []

    print(f"[1/3] Diarizing {len(files)} file(s)", flush=True)
    pipeline = diarize.load_pipeline(args.diarization_model, token, device)
    turns_by_file: dict[Path, list[dict]] = {}
    for index, path in enumerate(files, 1):
        try:
            turns_by_file[path] = diarize.run(
                pipeline,
                path,
                offset=args.offset,
                duration=args.duration,
                constraints=constraints,
                progress=len(files) == 1,
            )
            print(f"      {index}/{len(files)} {path.name}: {len(turns_by_file[path])} turns")
        except Exception as error:
            # One unreadable file should not cost the whole batch.
            failures.append((path, f"diarization: {error}"))
            print(f"      {index}/{len(files)} {path.name}: FAILED ({error})", file=sys.stderr)

    del pipeline
    free_gpu()

    print(f"[2/3] Transcribing {len(turns_by_file)} file(s)", flush=True)
    asr = asr_stage.load_asr(args.asr_model, device, args.token)
    segments_by_file: dict[Path, list[dict]] = {}
    for index, path in enumerate(turns_by_file, 1):
        try:
            samples, rate = read_audio(path, args.offset, args.duration)
            segments_by_file[path] = asr_stage.run(
                asr,
                samples,
                rate,
                language=args.language,
                chunk_length=args.chunk_length,
                batch_size=args.batch_size,
                offset=args.offset,
            )
            print(
                f"      {index}/{len(turns_by_file)} {path.name}: "
                f"{len(segments_by_file[path])} segments"
            )
        except Exception as error:
            failures.append((path, f"transcription: {error}"))
            print(f"      {index}/{len(turns_by_file)} {path.name}: FAILED ({error})", file=sys.stderr)

    del asr
    free_gpu()

    print("[3/3] Merging", flush=True)
    results = []
    for path, segments in segments_by_file.items():
        summary = write_result(turns_by_file[path], segments, targets[path], args)
        results.append((path, targets[path], summary))
        print(
            f"      {path.name} -> {targets[path].name}  "
            f"{summary['speakers']} speaker(s), {summary['blocks']} blocks"
            + (f", {summary['unattributed']} unattributed" if summary["unattributed"] else "")
        )

    elapsed = time.perf_counter() - started
    speech = sum(summary["speech"] for _, _, summary in results)
    print(
        f"\n{len(results)} file(s) in {format_timestamp(elapsed)}; "
        f"{format_timestamp(speech)} of speech"
    )
    if failures:
        print(f"\n{len(failures)} failed:", file=sys.stderr)
        for path, reason in failures:
            print(f"  {path.name}: {reason}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
