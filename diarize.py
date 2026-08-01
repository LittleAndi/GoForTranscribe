"""Answer "who spoke when" for an audio file, using pyannote.audio.

Stage 1 of the pipeline. Emits speaker turns; transcription and merging come later.

    uv run diarize.py --file interview.mp3 --speakers 2

The --offset flag exists for the stability protocol described in CLAUDE.md: the
same file diarized at several small time offsets should give roughly the same
answer, and when it does not, the configuration is on a decision boundary and
any single run of it is meaningless.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import NoReturn

SAMPLE_RATE = 16000
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def fail(message: str, *, hint: str | None = None) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(f"\n{hint}", file=sys.stderr)
    raise SystemExit(1)


def resolve_token(explicit: str | None) -> str:
    """Find a Hugging Face credential.

    Prefers a stored login over an environment variable, because `hf auth login`
    persists across shells and keeps the token off command lines and out of
    process listings. `huggingface_hub.get_token()` covers the login cache and the
    standard variables; the explicit ones are checked first only so --token wins.
    """
    from huggingface_hub import get_token

    token = (
        explicit
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or get_token()
    )
    if not token:
        fail(
            "no Hugging Face token",
            hint=(
                "pyannote's models are gated. One-time setup:\n"
                f"  1. Accept the terms at https://hf.co/{DEFAULT_MODEL}\n"
                "  2. Create a read token at https://hf.co/settings/tokens\n"
                "  3. uv run hf auth login\n"
                "\nA stored login persists across shells. Setting $env:HF_TOKEN works too,\n"
                "but only for processes started afterwards. Never commit the token."
            ),
        )
    return token


def decode(source: Path, dest: Path, offset: float) -> None:
    """Decode to 16 kHz mono 16-bit WAV, the form both pipeline stages want.

    Going through ffmpeg rather than letting pyannote read the file keeps input
    format support broad (mp3/m4a/flac/wma) and makes --offset exact.
    """
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found on PATH", hint="Install it, or add its bin directory to PATH.")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if offset:
        command += ["-ss", f"{offset}"]
    command += [
        "-i", str(source),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-vn",
        str(dest),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"ffmpeg failed to decode {source}\n{result.stderr.strip()}")


def load_waveform(path: Path) -> dict:
    """Read a 16-bit PCM WAV into the in-memory form pyannote accepts.

    Deliberately not handing pyannote the path: its built-in decoding goes through
    torchcodec, whose native libraries do not load on this Windows setup. Feeding
    a waveform dict bypasses that entirely, so the tool does not depend on a
    torchcodec install being repaired. Stdlib `wave` keeps it dependency-free.
    """
    import numpy
    import torch

    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            fail(f"expected 16-bit PCM from ffmpeg, got {8 * source.getsampwidth()}-bit")
        frames = source.readframes(source.getnframes())
        channels = source.getnchannels()
        rate = source.getframerate()

    samples = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32) / 32768.0
    waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy())
    return {"waveform": waveform, "sample_rate": rate}


def select_device(requested: str) -> "torch.device":  # type: ignore[name-defined] # noqa: F821
    import torch

    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if requested == "cuda":
            fail(
                "CUDA requested but unavailable",
                hint=(
                    "If this machine has an NVIDIA GPU, torch was probably installed from the\n"
                    "default index. See the GPU section of CLAUDE.md — Blackwell needs cu128."
                ),
            )
        print("No CUDA device; running on CPU (slower).", file=sys.stderr)
        return torch.device("cpu")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)

    # A torch without matching kernels reports the device happily and only fails on
    # the first real kernel launch, so check the arch list up front and say why.
    architectures = [arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")]
    if f"sm_{major}{minor}" not in architectures:
        fail(
            f"{name} is sm_{major}{minor}, which this torch build does not support "
            f"(has: {', '.join(architectures)})",
            hint="Reinstall torch from a CUDA 12.8+ index — see CLAUDE.md.",
        )

    print(f"Device: {name} (sm_{major}{minor})", file=sys.stderr)
    return torch.device("cuda")


def make_deterministic(seed: int) -> None:
    """Remove numerical variation between runs.

    Clustering on similar-sounding voices has been measured sitting on a decision
    boundary, so TF32's reduced precision is enough to flip a result. Without this,
    a config change and a coin flip look identical in the output.
    """
    import numpy
    import torch

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


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


def format_timestamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def write_rttm(turns: list[dict], path: Path, name: str) -> None:
    lines = [
        f"SPEAKER {name} 1 {turn['start']:.3f} {turn['end'] - turn['start']:.3f} "
        f"<NA> <NA> {turn['speaker']} <NA> <NA>"
        for turn in turns
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_share(turns: list[dict]) -> None:
    """Speech-time share per speaker — the headline metric from CLAUDE.md.

    A correct two-speaker result sits near 50/50. One speaker taking nearly
    everything is the collapse failure, and it looks perfectly confident.
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
    parser.add_argument("--speakers", type=int, help="exact speaker count, when known")
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
        "--overlapping",
        action="store_true",
        help="keep overlapping speech turns; default excludes them, for the merge stage",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--token", help="Hugging Face token (default: $HF_TOKEN)")
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

    # Importing torch takes seconds, so it happens after argument validation.
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    if not args.nondeterministic:
        make_deterministic(args.seed)

    device = select_device(args.device)

    pipeline = Pipeline.from_pretrained(args.model, token=token)
    if pipeline is None:
        # from_pretrained returns None rather than raising when the checkpoint is
        # gated and the account has not accepted its terms.
        fail(
            f"could not load {args.model}",
            hint=(
                "Usually the token is valid but the model terms have not been accepted.\n"
                f"Visit https://hf.co/{args.model}, accept, then retry."
            ),
        )
    pipeline.to(device)

    constraints = {
        key: value
        for key, value in (
            ("num_speakers", args.speakers),
            ("min_speakers", args.min_speakers),
            ("max_speakers", args.max_speakers),
        )
        if value
    }

    with tempfile.TemporaryDirectory() as workspace:
        decoded = Path(workspace) / "audio.wav"
        decode(args.file, decoded, args.offset)
        audio = load_waveform(decoded)
        with ProgressHook() as hook:
            result = pipeline(audio, hook=hook, **constraints)

    # 4.x returns a DiarizeOutput carrying both an overlap-aware annotation and one
    # with overlaps removed; older versions returned the annotation directly.
    if hasattr(result, "speaker_diarization"):
        annotation = (
            result.speaker_diarization
            if args.overlapping
            else result.exclusive_speaker_diarization
        )
    else:
        annotation = result

    turns = to_turns(annotation, args.offset)

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
