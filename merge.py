"""Attach speakers to transcript segments.

Stage 3, and the point of the whole project: `diarize.py` says who spoke when,
`transcribe.py` says what was said, and this joins them.

    uv run merge.py --turns turns.json --transcript transcript.json --timestamps

The two stages cut the audio in different places — Whisper on sentence
boundaries, the diarizer on turn boundaries — so their segment lists never align.
Each transcript segment is therefore given the speaker it shares the most time
with, which tolerates the boundaries disagreeing by a word or two. This is the
approach ported from GoForWhisper's SpeakerLabeler.cs, and it is why the
diarization stage defaults to turns with overlapping speech already removed.
"""

import argparse
import json
from bisect import bisect_left
from pathlib import Path

from common import fail, format_timestamp

UNKNOWN = "SPEAKER ?"


class SpeakerLabeler:
    """Resolves which speaker occupies a time range."""

    def __init__(self, turns: list[dict]) -> None:
        self.turns = sorted(turns, key=lambda turn: turn["start"])
        self.starts = [turn["start"] for turn in self.turns]

    def resolve(self, start: float, end: float) -> tuple[str | None, float]:
        """The speaker holding most of start..end, plus how much of it they hold.

        Returns (None, 0.0) when the range overlaps no turn at all, which happens
        when the transcriber produced text for something the segmenter judged to
        be non-speech — music, or a hallucination over silence.
        """
        best: str | None = None
        best_overlap = 0.0
        totals: dict[str, float] = {}

        # Turns are sorted by start, so scanning can begin at the last turn that
        # could still be running when this range opens, and stop as soon as one
        # starts after it closes.
        index = max(0, bisect_left(self.starts, start) - 1)
        for turn in self.turns[index:]:
            if turn["start"] >= end:
                break
            if turn["end"] <= start:
                continue
            overlap = min(turn["end"], end) - max(turn["start"], start)
            if overlap <= 0:
                continue
            speaker = turn["speaker"]
            totals[speaker] = totals.get(speaker, 0.0) + overlap
            if totals[speaker] > best_overlap:
                best_overlap = totals[speaker]
                best = speaker

        return best, best_overlap


def label(name: str | None) -> str:
    """Render a speaker as a stable label.

    Diarizer speaker ids are arbitrary cluster names — they carry no identity
    beyond "not the other one" — so they are renumbered to a readable form but
    still must not be read as identities across files or runs.
    """
    if name is None:
        return UNKNOWN
    digits = "".join(character for character in name if character.isdigit())
    return f"SPEAKER {int(digits) + 1}" if digits else name


def merge(turns: list[dict], segments: list[dict]) -> list[dict]:
    labeler = SpeakerLabeler(turns)
    merged = []
    for segment in segments:
        speaker, overlap = labeler.resolve(segment["start"], segment["end"])
        duration = segment["end"] - segment["start"]
        merged.append(
            {
                **segment,
                "speaker": label(speaker),
                # How much of the segment the winning speaker actually covers.
                # Low values flag places where the two stages disagree, which is
                # where attribution errors concentrate.
                "confidence": round(overlap / duration, 3) if duration > 0 else 0.0,
            }
        )
    return merged


def collapse(merged: list[dict]) -> list[dict]:
    """Join consecutive segments from one speaker into a single block."""
    blocks: list[dict] = []
    for segment in merged:
        if blocks and blocks[-1]["speaker"] == segment["speaker"]:
            blocks[-1]["end"] = segment["end"]
            blocks[-1]["text"] += " " + segment["text"]
        else:
            blocks.append(dict(segment))
    return blocks


def render(blocks: list[dict], timestamps: bool) -> str:
    lines = []
    for block in blocks:
        prefix = (
            f"[{format_timestamp(block['start'])} --> {format_timestamp(block['end'])}] "
            if timestamps
            else ""
        )
        lines.append(f"{prefix}{block['speaker']}: {block['text'].strip()}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", required=True, type=Path, help="JSON from diarize.py")
    parser.add_argument("--transcript", required=True, type=Path, help="JSON from transcribe.py")
    parser.add_argument("--output", type=Path, help="write here (.json or .txt); default stdout")
    parser.add_argument("--timestamps", action="store_true", help="prefix each line with times")
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="keep one line per transcript segment instead of joining a speaker's run",
    )
    args = parser.parse_args()

    for path in (args.turns, args.transcript):
        if not path.is_file():
            fail(f"no such file: {path}")

    turns = json.loads(args.turns.read_text(encoding="utf-8"))
    segments = json.loads(args.transcript.read_text(encoding="utf-8"))
    if not turns:
        fail(f"{args.turns} contains no turns")
    if not segments:
        fail(f"{args.transcript} contains no segments")

    merged = merge(turns, segments)
    blocks = merged if args.no_collapse else collapse(merged)

    if args.output and args.output.suffix == ".json":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(blocks)} blocks to {args.output}")
    else:
        text = render(blocks, args.timestamps)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote {len(blocks)} blocks to {args.output}")
        else:
            print(text)

    unattributed = [segment for segment in merged if segment["speaker"] == UNKNOWN]
    weak = [segment for segment in merged if 0 < segment["confidence"] < 0.5]
    print(
        f"\n{len(merged)} segments -> {len(blocks)} blocks; "
        f"{len(unattributed)} unattributed, {len(weak)} with under half their time "
        "on the assigned speaker"
    )


if __name__ == "__main__":
    main()
