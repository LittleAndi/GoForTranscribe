"""Build a control recording with two obviously distinct voices.

    uv run make_control.py --output samples/control.wav

The control answers one question: can the pipeline separate speakers that are
easy? An approach that fails here is disqualified before hard audio is worth
trying. Passing it proves very little on its own — the previous sherpa-onnx stack
passed a control cleanly and still collapsed on real speech — so treat it as a
floor, not evidence of quality.

Because the audio is assembled from known pieces, the exact reference labels are
written alongside it. That turns the control into something `evaluate.py` can
score properly, rather than something judged by eye.

Windows only: it drives the SAPI voices through PowerShell. Elsewhere, supply a
control recording by other means — a real one with two dissimilar speakers is
strictly better if you have labels for it.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from common import SAMPLE_RATE, fail, load_waveform

# Alternating turns of unequal length, so a correct result is not simply 50/50 by
# construction — a diarizer that always splits evenly would still be wrong here.
SCRIPT = [
    ("A", "Welcome to the show. Today we are going to talk about how speaker separation works."),
    ("B", "Thanks for having me."),
    ("A", "Let us start with the basics."),
    ("B", "Sure. The first step is finding out where the speech actually is, and where it stops."),
    ("A", "And after that?"),
    ("B", "After that each region gets turned into a vector that describes the voice itself, "
          "not the words. Similar vectors are grouped together, and each group becomes a speaker."),
    ("A", "That sounds straightforward enough."),
    ("B", "It is, until two people happen to sound alike."),
    ("A", "Which is exactly the case we care about. Let us try a much longer sentence now, so "
          "that one speaker clearly holds more of the total time than the other one does."),
    ("B", "Agreed."),
]


def synthesize(voice: str, text: str, dest: Path) -> None:
    """Render one utterance with a SAPI voice."""
    escaped = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); "
        f"$s.SetOutputToWaveFile('{dest}'); "
        f"$s.Speak('{escaped}'); "
        "$s.Dispose()"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest.is_file():
        fail(f"speech synthesis failed for voice '{voice}'\n{result.stderr.strip()}")


def duration_of(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def speech_regions(
    path: Path,
    threshold: float = 0.02,
    min_silence: float = 0.3,
    min_speech: float = 0.15,
) -> list[tuple[float, float]]:
    """The stretches inside a clip that actually contain speech.

    SAPI pads each utterance with silence and pauses between its sentences.
    Labelling the whole clip as speech makes the reference claim speech where
    there is none, and any diarizer is then charged for "missing" silence: the
    first version of this control scored 22% DER while attributing every single
    speaker correctly. Splitting on real pauses is what makes the number mean
    something.
    """
    import numpy

    samples, rate = load_waveform(path)
    envelope = numpy.abs(samples[0])
    if envelope.size == 0 or envelope.max() == 0:
        return []

    # Judge loudness per 10 ms frame rather than per sample, so a single quiet
    # sample mid-word does not punch a hole in a region.
    frame = max(1, int(0.01 * rate))
    usable = (envelope.size // frame) * frame
    frames = envelope[:usable].reshape(-1, frame).max(axis=1)
    loud = frames > threshold * envelope.max()
    if not loud.any():
        return []

    # Contiguous runs of loud frames, then merge those separated by a short gap.
    changes = numpy.flatnonzero(numpy.diff(loud.astype(numpy.int8)))
    edges = numpy.concatenate(([0], changes + 1, [loud.size]))
    regions = [
        (edges[i] * frame / rate, edges[i + 1] * frame / rate)
        for i in range(len(edges) - 1)
        if loud[edges[i]]
    ]

    merged: list[list[float]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] < min_silence:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged if end - start >= min_speech]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("samples/control.wav"))
    parser.add_argument("--voice-a", default="Microsoft Zira Desktop", help="female by default")
    parser.add_argument("--voice-b", default="Microsoft David Desktop", help="male by default")
    parser.add_argument(
        "--gap",
        type=float,
        default=0.25,
        help="silence between turns; keep short so turn detection is still tested",
    )
    args = parser.parse_args()

    if platform.system() != "Windows":
        fail("make_control.py drives Windows SAPI voices", hint=__doc__.strip().splitlines()[-1])
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found on PATH")

    voices = {"A": args.voice_a, "B": args.voice_b}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        silence = workspace / "gap.wav"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                "-t", str(args.gap), "-c:a", "pcm_s16le", str(silence),
            ],
            check=True,
        )

        pieces: list[Path] = []
        reference: list[dict] = []
        position = 0.0

        for index, (speaker, text) in enumerate(SCRIPT):
            raw = workspace / f"{index:02d}-raw.wav"
            normalized = workspace / f"{index:02d}.wav"
            synthesize(voices[speaker], text, raw)
            # SAPI writes at its own rate; force the pipeline's format so the
            # concatenation is uniform and no resampling happens later.
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(normalized),
                ],
                check=True,
            )

            length = duration_of(normalized)
            for speech_start, speech_end in speech_regions(normalized):
                reference.append(
                    {
                        "start": round(position + speech_start, 3),
                        "end": round(position + speech_end, 3),
                        "speaker": f"SPEAKER_{0 if speaker == 'A' else 1:02d}",
                    }
                )
            position += length + args.gap
            pieces += [normalized, silence]

        listing = workspace / "pieces.txt"
        listing.write_text(
            "\n".join(f"file '{piece.as_posix()}'" for piece in pieces), encoding="utf-8"
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", str(args.output),
            ],
            check=True,
        )

    truth = args.output.with_name(args.output.stem + "-reference.json")
    truth.write_text(json.dumps(reference, indent=2), encoding="utf-8")

    spoken = sum(turn["end"] - turn["start"] for turn in reference)
    by_speaker: dict[str, float] = {}
    for turn in reference:
        by_speaker[turn["speaker"]] = by_speaker.get(turn["speaker"], 0.0) + (
            turn["end"] - turn["start"]
        )

    print(f"Wrote {args.output} ({position:.1f}s) and {truth}")
    print(f"Reference: {len(reference)} turns, {spoken:.1f}s of speech", file=sys.stderr)
    for speaker, seconds in sorted(by_speaker.items()):
        print(f"  {speaker}  {100 * seconds / spoken:5.1f}%  {seconds:6.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
