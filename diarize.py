"""Answer "who spoke when" for an audio file, using pyannote.audio.

Stage 1 of the pipeline. Emits speaker turns; `transcribe.py` and `merge.py`
attach the words.

    uv run diarize.py --file interview.mp3

The --offset flag exists for the stability protocol described in CLAUDE.md: the
same file diarized at several small time offsets should give roughly the same
answer, and when it does not, the configuration is on a decision boundary and
any single run of it is meaningless. `stability.py` automates that sweep.
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
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

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

TOKEN_HINT = (
    "pyannote's models are gated. One-time setup:\n"
    f"  1. Accept the terms at https://hf.co/{DEFAULT_MODEL}\n"
    "  2. Create a read token at https://hf.co/settings/tokens\n"
    "  3. uv run hf auth login\n"
    "\nA stored login persists across shells. Setting $env:HF_TOKEN works too,\n"
    "but only for processes started afterwards. Never commit the token."
)


def load_pipeline(model: str, token: str, device):
    """Load a diarization pipeline onto a device.

    Separate from main() so `stability.py` can sweep many offsets against one
    loaded pipeline instead of paying the load cost per run.
    """
    # Ahead of the pyannote import: it survives a broken torchcodec but prints a
    # 40-line traceback while doing so, which buries the actual output. We never
    # let it decode anything anyway. See common.neutralize_broken_torchcodec.
    neutralize_broken_torchcodec()

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model, token=token)
    if pipeline is None:
        # from_pretrained returns None rather than raising when the checkpoint is
        # gated and the account has not accepted its terms.
        fail(
            f"could not load {model}",
            hint=(
                "Usually the token is valid but the model terms have not been accepted.\n"
                f"Visit https://hf.co/{model}, accept, then retry."
            ),
        )
    pipeline.to(device)
    return pipeline


def run(
    pipeline,
    file: Path,
    offset: float = 0.0,
    duration: float | None = None,
    constraints: dict | None = None,
    overlapping: bool = False,
    progress: bool = True,
) -> list[dict]:
    """Diarize one file (or window of it) and return turns."""
    import torch

    from pyannote.audio.pipelines.utils.hook import ProgressHook

    with tempfile.TemporaryDirectory() as workspace:
        decoded = Path(workspace) / "audio.wav"
        decode(file, decoded, offset, duration)
        samples, rate = load_waveform(decoded)
        audio = {"waveform": torch.from_numpy(samples), "sample_rate": rate}

        if progress:
            with ProgressHook() as hook:
                result = pipeline(audio, hook=hook, **(constraints or {}))
        else:
            result = pipeline(audio, **(constraints or {}))

    # 4.x returns a DiarizeOutput carrying both an overlap-aware annotation and one
    # with overlaps removed; older versions returned the annotation directly.
    if hasattr(result, "speaker_diarization"):
        annotation = (
            result.speaker_diarization if overlapping else result.exclusive_speaker_diarization
        )
    else:
        annotation = result

    return to_turns(annotation, offset)


def shares(turns: list[dict]) -> dict[str, float]:
    """Speech-time share per speaker, as percentages."""
    spoken: dict[str, float] = defaultdict(float)
    for turn in turns:
        spoken[turn["speaker"]] += turn["end"] - turn["start"]
    total = sum(spoken.values())
    return {speaker: 100 * seconds / total for speaker, seconds in spoken.items()} if total else {}


def to_turns(annotation, offset: float) -> list[dict]:
    """Flatten pyannote's annotation, shifting times back past any --offset trim."""
    return [
        {
            "start": round(segment.start + offset, 3),
            "end": round(segment.end + offset, 3),
            "speaker": speaker,
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def write_rttm(turns: list[dict], path: Path, name: str) -> None:
    lines = [
        f"SPEAKER {name} 1 {turn['start']:.3f} {turn['end'] - turn['start']:.3f} "
        f"<NA> <NA> {turn['speaker']} <NA> <NA>"
        for turn in turns
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_share(turns: list[dict]) -> None:
    """Speech-time share per speaker.

    This is a collapse detector, not an accuracy measure: it catches one cluster
    eating the whole recording, but a merely lopsided split may be perfectly real
    (see the excerpt-versus-full-episode finding in CLAUDE.md).
    """
    spoken: dict[str, float] = defaultdict(float)
    for turn in turns:
        spoken[turn["speaker"]] += turn["end"] - turn["start"]

    total = sum(spoken.values())
    print(f"\n{len(spoken)} speaker(s), {format_timestamp(total)} of speech", file=sys.stderr)
    for speaker, seconds in sorted(spoken.items(), key=lambda item: -item[1]):
        share = 100 * seconds / total if total else 0
        print(f"  {speaker:<12} {share:5.1f}%  {format_timestamp(seconds)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, type=Path, help="audio file to diarize")
    parser.add_argument(
        "--speakers",
        type=int,
        help="exact speaker count; unsafe when ads or intros add voices, see CLAUDE.md",
    )
    parser.add_argument("--min-speakers", type=int, help="lower bound when the count is open")
    parser.add_argument("--max-speakers", type=int, help="upper bound when the count is open")
    parser.add_argument("--output", type=Path, help="write turns here (.json or .rttm)")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="skip this many seconds of audio; for stability runs (see CLAUDE.md)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="only process this many seconds, so a window can exclude intros and ads",
    )
    parser.add_argument(
        "--overlapping",
        action="store_true",
        help="keep overlapping speech turns; default excludes them, for the merge stage",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--token", help="Hugging Face token (default: stored login)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="allow TF32 and cuDNN autotuning; faster, but results vary run to run",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        fail(f"no such file: {args.file}")
    if args.speakers and (args.min_speakers or args.max_speakers):
        fail("--speakers is exact; do not combine it with --min-speakers/--max-speakers")

    token = resolve_token(args.token)
    if not token:
        fail("no Hugging Face token", hint=TOKEN_HINT)

    if not args.nondeterministic:
        make_deterministic(args.seed)

    device = select_device(args.device)
    pipeline = load_pipeline(args.model, token, device)

    constraints = {
        key: value
        for key, value in (
            ("num_speakers", args.speakers),
            ("min_speakers", args.min_speakers),
            ("max_speakers", args.max_speakers),
        )
        if value
    }

    turns = run(
        pipeline,
        args.file,
        offset=args.offset,
        duration=args.duration,
        constraints=constraints,
        overlapping=args.overlapping,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".rttm":
            write_rttm(turns, args.output, args.file.stem)
        else:
            args.output.write_text(json.dumps(turns, indent=2), encoding="utf-8")
        print(f"Wrote {len(turns)} turns to {args.output}", file=sys.stderr)
    else:
        for turn in turns:
            print(
                f"[{format_timestamp(turn['start'])} --> {format_timestamp(turn['end'])}] "
                f"{turn['speaker']}"
            )

    report_share(turns)


if __name__ == "__main__":
    main()
