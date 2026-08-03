"""Audio in, speaker-attributed transcript out.

    uv run pipeline.py --file episode.mp3 --output episode.txt
    uv run pipeline.py --folder podcasts\ --timestamps

With --folder, every audio file in the directory is processed and written beside
it as <name>.txt, or into --output-dir. Already-transcribed files are skipped
unless --overwrite is given, so an interrupted batch resumes where it stopped.

Language is detected per file by default (a small Whisper model, see
detect_language.py) rather than assumed, and each file's transcription model
follows from what was detected — KBLab/kb-whisper-large only knows Swedish, so
anything else routes to a multilingual model instead of being forced through
it. Pass --language to force one language (and, with it, one model) for every
file instead, exactly as before this existed.

A file whose sampled windows disagree on the language is flagged
MIXED-LANGUAGE in the run's output rather than silently picked one way or the
other — it usually means the file opens in one language and switches to
another partway through (a real example lives in this project's own test
library; see CLAUDE.md). Rerun a flagged file on its own with --file and
--split-language to transcribe it region-by-region instead of forcing one
language over the whole thing. Not folded into a --folder run: it costs a
second ASR model load on top of the normal one, worth paying only for the
rare file that actually needs it.

A batch loads each model once rather than once per file, which is why it works
in two passes — diarize everything, then transcribe everything — instead of
running both stages per file. On a folder of ten episodes that is the difference
between two model loads and twenty.

Files are processed in groups of --batch-files (default 8), and each group is
written out before the next begins. Two model loads per group is the price;
what it buys is a bounded amount of held turns and segments, and results on
disk as the run proceeds rather than only at the very end.

The separate stage tools remain the better choice while iterating, since each
stage's output can be inspected and the expensive stages need not be repeated.
"""

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import detect_language
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

# KBLab's fine-tune (asr_stage.DEFAULT_MODEL) only knows Swedish. A file
# detected as anything else needs a multilingual model instead of being forced
# through it — see route_files().
FALLBACK_ASR_MODEL = "openai/whisper-large-v3"


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


def batches(files: list[Path], size: int | None) -> list[list[Path]]:
    """Split the work into groups, or one group of everything when size <= 0."""
    if not size or size <= 0:
        return [files]
    return [files[start : start + size] for start in range(0, len(files), size)]


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


def model_for(language: str, args) -> str:
    """Which ASR model speaks this language, honouring an explicit override."""
    if args.asr_model is not None:
        return args.asr_model
    return asr_stage.DEFAULT_MODEL if language == "sv" else FALLBACK_ASR_MODEL


def detect_languages(
    files: list[Path], args, device
) -> tuple[dict[Path, tuple[str, float]], list[Path]]:
    """Language + confidence per file, from a small model loaded once for all of them.

    Runs ahead of the (--file/--folder-wide) batch loop rather than per group:
    it is cheap enough — one whisper-tiny encoder pass per window — that
    loading it once for the whole run costs nothing worth optimising away.

    Also returns the files whose sampled windows didn't agree on one language
    — the whole-file (asr_model, language) routing this feeds into is a single
    choice, which is wrong for a file that genuinely switches languages
    partway through (see CLAUDE.md's "Language routing" section for how one
    such file in this project's own test library was found). Flagging it here
    means it shows up in every normal run rather than requiring a special scan
    to notice; --split-language then handles it as its own pass, one file at
    a time.
    """
    print(f"[0/3] Detecting language of {len(files)} file(s)", flush=True)
    whisper, extractor = detect_language.load_model(args.langid_model, device)
    results: dict[Path, tuple[str, float]] = {}
    mixed: list[Path] = []
    for index, path in enumerate(files, 1):
        language, confidence, readings = detect_language.detect_file(
            whisper, extractor, path, device, args.langid_samples, args.langid_window
        )
        results[path] = (language, confidence)

        if len(readings) > 1 and len({lang for lang, _ in readings}) > 1:
            mixed.append(path)
            detail = ", ".join(f"{lang} {conf:.0%}" for lang, conf in readings)
            print(f"      {index}/{len(files)} MIXED-LANGUAGE {path.name}: {detail}")
        else:
            print(f"      {index}/{len(files)} {path.name}: {language} ({confidence:.0%})")

    del whisper
    free_gpu()

    if mixed:
        print(
            f"\n{len(mixed)} file(s) flagged MIXED-LANGUAGE (see above) — the whole-file "
            "language below is a best guess for these. Rerun each individually with "
            "--file <path> --split-language to transcribe by region instead:"
        )
        for path in mixed:
            print(f"  {path}")
        print()

    return results, mixed


def route_files(files: list[Path], args, device) -> dict[Path, tuple[str, str]]:
    """Decide (asr_model, language) for every file.

    An explicit --language always wins, for full backward compatibility (a
    forced 'sv' or 'auto' behaves exactly as before, model included). Only
    when --language is omitted does this run the language-id pass and, unless
    --asr-model was also given explicitly, route each file's model by what
    was detected — because the default ASR model cannot speak for anything
    but Swedish. A low-confidence read (near-silence, a music-only intro)
    falls back to --fallback-language rather than being trusted blindly.
    """
    detect_lang = args.language is None
    detected, _mixed = detect_languages(files, args, device) if detect_lang else ({}, [])

    resolved: dict[Path, tuple[str, str]] = {}
    for path in files:
        if detect_lang:
            language, confidence = detected[path]
            if confidence < args.langid_min_confidence:
                print(
                    f"  {path.name}: {language} at only {confidence:.0%} confidence, "
                    f"falling back to --fallback-language {args.fallback_language}",
                    file=sys.stderr,
                )
                language = args.fallback_language
        else:
            language = args.language

        model = model_for(language, args) if detect_lang else (args.asr_model or asr_stage.DEFAULT_MODEL)
        resolved[path] = (model, language)

    if detect_lang:
        counts = Counter(model for model, _ in resolved.values())
        print("Language routing: " + ", ".join(f"{model}={n}" for model, n in counts.items()))

    return resolved


def bisect_boundary(
    whisper, extractor, path: Path, device, start: float, end: float, before_language: str,
    window: float, min_resolution: float, max_iterations: int = 8,
) -> float:
    """Narrow [start, end) down to the approximate point the language switches.

    `start` is known to read as `before_language`, `end` is known not to.
    Stops once the bracket is tighter than min_resolution — Whisper's own
    long-form decoding works in ~30s windows anyway, so resolving the switch
    point any finer buys nothing.
    """
    for _ in range(max_iterations):
        if end - start <= min_resolution:
            break
        midpoint = (start + end) / 2
        samples, rate = detect_language.sample_window(path, midpoint, window)
        language, _ = detect_language.detect(whisper, extractor, samples, rate, device)
        if language == before_language:
            start = midpoint
        else:
            end = midpoint
    return (start + end) / 2


def locate_regions(whisper, extractor, path: Path, device, readings, offsets, args):
    """Turn language-id readings into [(start, duration, language, model), ...] regions.

    Assumes at most one switch: every non-majority reading must sit at one end
    of the sampled sequence in time order — a clean "opens in X, switches to
    Y" pattern, which is what the one mixed-language file found in this
    project's own library actually looks like (see CLAUDE.md). Anything more
    tangled (language bouncing back and forth across the samples) is refused
    rather than guessed at, since a wrong guess here silently mistranscribes
    part of the file rather than failing loudly.
    """
    languages = [language for language, _ in readings]
    majority_language, majority_count = Counter(languages).most_common(1)[0]

    if majority_count == len(languages):
        return [(0.0, None, majority_language, model_for(majority_language, args))]

    minority_positions = [i for i, language in enumerate(languages) if language != majority_language]
    is_prefix = minority_positions == list(range(len(minority_positions)))
    is_suffix = minority_positions == list(
        range(len(languages) - len(minority_positions), len(languages))
    )
    if not (is_prefix or is_suffix):
        fail(
            f"{path.name}: language readings don't form a single clean switch ({languages})",
            hint="--split-language only handles one boundary. Inspect with "
            "'detect_language.py --file ... --samples 8' and transcribe this one manually.",
        )

    if is_prefix:
        before_language, after_language = languages[minority_positions[-1]], majority_language
        search_start, search_end = offsets[minority_positions[-1]], offsets[minority_positions[-1] + 1]
    else:
        before_language, after_language = majority_language, languages[minority_positions[0]]
        search_start, search_end = offsets[minority_positions[0] - 1], offsets[minority_positions[0]]

    boundary = bisect_boundary(
        whisper, extractor, path, device, search_start, search_end, before_language,
        args.langid_window, args.split_min_resolution,
    )

    return [
        (0.0, boundary, before_language, model_for(before_language, args)),
        (boundary, None, after_language, model_for(after_language, args)),
    ]


def run_split_language(path: Path, target: Path, args, device, token, constraints) -> dict:
    """Transcribe one file in language-homogeneous regions rather than one pass.

    For a MIXED-LANGUAGE file only (flagged by a normal run's detect_languages
    pass) — one whole-file --language/--asr-model choice cannot represent a
    file that opens in one language and switches to another partway through.
    Deliberately not folded into the batch path: locating the switch point
    costs a handful of extra langid calls and a second ASR model load, worth
    paying only for the rare file a normal run already flagged as needing it.

    Diarization runs once over the whole file — turns are language-agnostic.
    Each region is transcribed with its own model via the existing --offset
    machinery, and merge() combines the resulting segments with the turns by
    time overlap exactly as it does for a single-pass file; it has no notion
    of which ASR call produced a given segment.
    """
    print("[0/4] Detecting language regions", flush=True)
    whisper, extractor = detect_language.load_model(args.langid_model, device)
    total = detect_language.probe_duration(path)
    _, _, readings = detect_language.detect_file(
        whisper, extractor, path, device, args.langid_samples, args.langid_window
    )
    offsets = detect_language.sample_offsets(total, args.langid_samples, args.langid_window)
    regions = locate_regions(whisper, extractor, path, device, readings, offsets, args)
    del whisper
    free_gpu()

    if len(regions) == 1:
        print("      no clean split point found; transcribing as a single language")
    for start, duration, language, model in regions:
        span = f"{format_timestamp(start)}+" if duration is None else f"{format_timestamp(start)}-{format_timestamp(start + duration)}"
        print(f"      {span} [{language}] via {model}")

    print("[1/4] Diarizing", flush=True)
    pipeline = diarize.load_pipeline(args.diarization_model, token, device)
    turns = diarize.run(
        pipeline, path, offset=args.offset, duration=args.duration, constraints=constraints
    )
    del pipeline
    free_gpu()
    print(f"      {len(turns)} turns")

    print(f"[2/4] Transcribing {len(regions)} region(s)", flush=True)
    segments = []
    for index, (start, duration, language, model) in enumerate(regions, 1):
        asr = asr_stage.load_asr(model, device, args.token)
        samples, rate = read_audio(path, start, duration)
        region_segments = asr_stage.run(
            asr, samples, rate, language=language,
            chunk_length=args.chunk_length, batch_size=args.batch_size, offset=start,
        )
        segments.extend(region_segments)
        print(f"      region {index}/{len(regions)} [{model}]: {len(region_segments)} segments")
        del asr
        free_gpu()
    segments.sort(key=lambda segment: segment["start"])

    print("[3/4] Merging", flush=True)
    summary = write_result(turns, segments, target, args)
    print(
        f"      {target.name}  {summary['speakers']} speaker(s), {summary['blocks']} blocks"
        + (f", {summary['unattributed']} unattributed" if summary["unattributed"] else "")
    )
    return summary


def run_batch(files, targets, args, device, token, constraints, resolved_by_file, progress):
    """Take one group of files all the way to written .txt files.

    Each model is loaded once for the group and released before the next stage,
    so only one set of weights is resident at a time. Everything the group holds
    — turns, segments — is dropped when it returns.
    """
    results = []
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
                progress=progress,
            )
            print(f"      {index}/{len(files)} {path.name}: {len(turns_by_file[path])} turns")
        except Exception as error:
            # One unreadable file should not cost the whole batch.
            failures.append((path, f"diarization: {error}"))
            print(f"      {index}/{len(files)} {path.name}: FAILED ({error})", file=sys.stderr)

    del pipeline
    free_gpu()

    print(f"[2/3] Transcribing {len(turns_by_file)} file(s)", flush=True)
    # Files in one group can still need different ASR models (a Swedish episode
    # and an English one, say), so group by resolved model rather than assuming
    # one for the whole batch — each distinct model still loads only once.
    by_model: dict[str, list[Path]] = defaultdict(list)
    for path in turns_by_file:
        model, _ = resolved_by_file[path]
        by_model[model].append(path)

    segments_by_file: dict[Path, list[dict]] = {}
    done = 0
    for model, paths in by_model.items():
        asr = asr_stage.load_asr(model, device, args.token)
        for path in paths:
            done += 1
            try:
                samples, rate = read_audio(path, args.offset, args.duration)
                _, language = resolved_by_file[path]
                segments_by_file[path] = asr_stage.run(
                    asr,
                    samples,
                    rate,
                    language=language,
                    chunk_length=args.chunk_length,
                    batch_size=args.batch_size,
                    offset=args.offset,
                )
                print(
                    f"      {done}/{len(turns_by_file)} [{model}] {path.name}: "
                    f"{len(segments_by_file[path])} segments"
                )
            except Exception as error:
                failures.append((path, f"transcription: {error}"))
                print(
                    f"      {done}/{len(turns_by_file)} {path.name}: FAILED ({error})",
                    file=sys.stderr,
                )
        del asr
        free_gpu()

    print("[3/3] Merging", flush=True)
    for path, segments in segments_by_file.items():
        summary = write_result(turns_by_file[path], segments, targets[path], args)
        results.append((path, targets[path], summary))
        print(
            f"      {path.name} -> {targets[path].name}  "
            f"{summary['speakers']} speaker(s), {summary['blocks']} blocks"
            + (f", {summary['unattributed']} unattributed" if summary["unattributed"] else "")
        )

    return results, failures


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
    parser.add_argument(
        "--batch-files",
        type=int,
        default=8,
        metavar="N",
        help="files per group, each group written before the next starts; "
        "0 processes the whole folder in one group (--folder only, default: 8)",
    )
    parser.add_argument("--speakers", type=int, help="exact count; see CLAUDE.md before using")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--language",
        default=None,
        help="force one language for every file. Default: detect it per file "
        "(see --langid-* below); 'auto' lets the ASR model decide per chunk instead",
    )
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--keep-intermediate", type=Path, help="also save per-stage JSON")
    parser.add_argument("--diarization-model", default=diarize.DEFAULT_MODEL)
    parser.add_argument(
        "--asr-model",
        default=None,
        help="force one ASR model for every file. Default: route by detected "
        f"language ({asr_stage.DEFAULT_MODEL} for Swedish, {FALLBACK_ASR_MODEL} otherwise)",
    )
    parser.add_argument(
        "--langid-model",
        default=detect_language.DEFAULT_MODEL,
        help=f"language-id checkpoint; default: {detect_language.DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--langid-samples",
        type=int,
        default=detect_language.DEFAULT_SAMPLES,
        help="windows spread across each file for language id, majority vote decides; "
        f"a single non-representative stretch (e.g. a long intro in a different "
        f"language than the rest) can't out-vote the rest; default: {detect_language.DEFAULT_SAMPLES}",
    )
    parser.add_argument(
        "--langid-window",
        type=float,
        default=detect_language.DEFAULT_WINDOW,
        help=f"seconds per language-id window; default: {detect_language.DEFAULT_WINDOW}",
    )
    parser.add_argument(
        "--langid-min-confidence",
        type=float,
        default=0.5,
        help="below this, fall back to --fallback-language rather than trust a shaky read",
    )
    parser.add_argument(
        "--fallback-language",
        default="sv",
        help="used when a file's language isn't given and detection confidence is low",
    )
    parser.add_argument(
        "--split-language",
        action="store_true",
        help="for a MIXED-LANGUAGE file (see a normal run's flagged list): transcribe in "
        "language-homogeneous regions instead of one pass. --file only, not batchable",
    )
    parser.add_argument(
        "--split-min-resolution",
        type=float,
        default=15.0,
        help="stop narrowing down the switch point once the search window is this short",
    )
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
    if args.split_language and args.folder:
        fail("--split-language only works with --file, not --folder")
    if args.split_language and args.language:
        fail("--split-language contradicts a forced --language; drop one or the other")

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

    if args.split_language:
        summary = run_split_language(files[0], targets[files[0]], args, device, token, constraints)
        elapsed = time.perf_counter() - started
        print(
            f"\n1 file in {format_timestamp(elapsed)}; "
            f"{format_timestamp(summary['speech'])} of speech"
        )
        return

    resolved_by_file = route_files(files, args, device)
    results = []
    failures: list[tuple[Path, str]] = []

    groups = batches(files, args.batch_files)
    for number, group in enumerate(groups, 1):
        if len(groups) > 1:
            print(f"\n=== Batch {number}/{len(groups)} ({len(group)} file(s)) ===", flush=True)
        done, broken = run_batch(
            group, targets, args, device, token, constraints, resolved_by_file, progress=len(files) == 1
        )
        results.extend(done)
        failures.extend(broken)

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
