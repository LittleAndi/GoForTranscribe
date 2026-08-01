"""Cut per-speaker audio clips from a diarization result, to check it by ear.

Speech-time share catches cluster collapse but cannot tell a correct attribution
from a consistently wrong one. Listening can: if every clip in a speaker's folder
is the same voice, the clustering is doing its job.

    uv run diarize.py --file episode.mp3 --output turns.json
    uv run sample_turns.py --turns turns.json --file episode.mp3

Clips are spread across the whole recording rather than taken from one stretch,
so a speaker who is only confused during part of it still shows up.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def pick(turns: list[dict], count: int, min_seconds: float) -> list[dict]:
    """Choose clips: long enough to judge, spread evenly over the timeline."""
    usable = [turn for turn in turns if turn["end"] - turn["start"] >= min_seconds]
    if not usable:
        return []
    if len(usable) <= count:
        return usable
    step = len(usable) / count
    return [usable[int(index * step)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", required=True, type=Path, help="JSON from diarize.py")
    parser.add_argument("--file", required=True, type=Path, help="the audio it was made from")
    parser.add_argument("--out", type=Path, default=Path("out/clips"))
    parser.add_argument("--per-speaker", type=int, default=6)
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=3.0,
        help="ignore turns shorter than this; too brief to recognise a voice",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found on PATH")
    for path in (args.turns, args.file):
        if not path.is_file():
            fail(f"no such file: {path}")

    turns = json.loads(args.turns.read_text(encoding="utf-8"))
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for turn in turns:
        by_speaker[turn["speaker"]].append(turn)

    total = 0
    for speaker, speaker_turns in sorted(by_speaker.items()):
        chosen = pick(speaker_turns, args.per_speaker, args.min_seconds)
        if not chosen:
            print(
                f"{speaker}: no turn reaches {args.min_seconds}s "
                f"(longest {max(t['end'] - t['start'] for t in speaker_turns):.1f}s)",
                file=sys.stderr,
            )
            continue

        folder = args.out / speaker
        folder.mkdir(parents=True, exist_ok=True)
        for turn in chosen:
            start, end = turn["start"], turn["end"]
            destination = folder / f"{start:08.1f}-{end:08.1f}.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-ss", f"{start}", "-t", f"{end - start}",
                    "-i", str(args.file), "-ac", "1", "-ar", "16000", str(destination),
                ],
                check=True,
            )
            total += 1
        print(f"{speaker}: {len(chosen)} clips -> {folder}")

    print(f"\n{total} clips written. Every clip in a folder should be the same voice.")


if __name__ == "__main__":
    main()
