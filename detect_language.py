"""Guess the spoken language of an audio file with a small Whisper model.

    uv run detect_language.py --file episode.mp3

Exists so `pipeline.py` can route each file to the right transcription model
and language token without the caller having to know in advance — the default
ASR model is a Swedish-only fine-tune, so a non-Swedish file needs a different
model, not just a different --language flag.

`openai/whisper-tiny` is used rather than a dedicated language-id model: it is
already multilingual, transformers is already a dependency, and one encoder
pass over a short window costs a few hundred milliseconds even on CPU. No new
model family, no new dependency.
"""

import argparse
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from common import decode, fail, load_waveform, neutralize_broken_torchcodec, select_device

DEFAULT_MODEL = "openai/whisper-tiny"
DEFAULT_SAMPLES = 3
DEFAULT_WINDOW = 20.0

# Used only when a file is too short to hold DEFAULT_SAMPLES well-separated
# windows and detection falls back to a single one.
DEFAULT_WINDOW_OFFSET = 60.0


def load_model(model: str, device):
    """Load a Whisper checkpoint for language id only — never for transcription.

    Kept independent of whatever ASR model `pipeline.py` ends up choosing: the
    point of this pass is to decide *that* model, and a 39M-parameter
    checkpoint is plenty for language id even though it would be a poor
    transcriber.
    """
    # Ahead of the transformers import, same reason as transcribe.py: a broken
    # torchcodec install otherwise takes the process down with it.
    neutralize_broken_torchcodec()

    import torch
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    whisper = WhisperForConditionalGeneration.from_pretrained(model, dtype=dtype).to(device)
    whisper.eval()
    extractor = WhisperFeatureExtractor.from_pretrained(model)
    return whisper, extractor


def probe_duration(path: Path) -> float:
    """Total duration in seconds, or 0.0 if ffprobe can't tell (caller falls back)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def sample_offsets(total_duration: float, count: int, window: float) -> list[float]:
    """Spread `count` windows across the middle of a file.

    Skips the very start and end: podcasts often cold-open on a jingle and
    close on a fixed outro, neither representative of what the episode is
    actually spoken in. A file too short to hold `count` well-separated
    windows falls back to one, offset past a likely cold open.
    """
    if total_duration <= window * 1.5:
        return [0.0]
    if count <= 1:
        return [min(DEFAULT_WINDOW_OFFSET, max(total_duration - window, 0.0))]
    fractions = [0.15 + i * 0.6 / (count - 1) for i in range(count)]
    return [total_duration * fraction for fraction in fractions]


def sample_window(path: Path, offset: float, duration: float):
    """Decode a short window of audio rather than the whole file.

    Falls back to the start of the file when the requested window decodes to
    under a second of audio — an offset beyond the file's actual length would
    otherwise be judged on near-silence instead of falling back sensibly.
    """
    with tempfile.TemporaryDirectory() as workspace:
        decoded = Path(workspace) / "window.wav"
        decode(path, decoded, offset, duration)
        samples, rate = load_waveform(decoded)
        if samples.shape[1] < rate and offset:
            decode(path, decoded, 0.0, duration)
            samples, rate = load_waveform(decoded)
    return samples[0], rate


def detect(whisper, extractor, samples, rate, device) -> tuple[str, float]:
    """Return (ISO-639-1 language code, confidence) for one window of audio.

    Reimplements transformers' `WhisperForConditionalGeneration.detect_language`
    rather than calling it directly: that method only returns the winning
    token id, and the confidence is needed here to weigh disagreeing windows
    against each other in detect_file().
    """
    import torch

    inputs = extractor(samples, sampling_rate=rate, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=whisper.dtype)

    generation_config = whisper.generation_config
    decoder_input_ids = torch.full(
        (1, 1), generation_config.decoder_start_token_id, device=device, dtype=torch.long
    )

    with torch.no_grad():
        logits = whisper(
            input_features, decoder_input_ids=decoder_input_ids, use_cache=False
        ).logits[0, -1]

    lang_to_id = generation_config.lang_to_id
    lang_token_ids = torch.tensor(list(lang_to_id.values()), device=device)
    id_to_lang = {token_id: token.strip("<|>") for token, token_id in lang_to_id.items()}

    # Restrict the softmax to language tokens only, otherwise the confidence
    # figure would be diluted by every other token in Whisper's vocabulary.
    masked = torch.full_like(logits, float("-inf"))
    masked[lang_token_ids] = logits[lang_token_ids]
    probs = torch.softmax(masked, dim=-1)

    best_id = int(torch.argmax(probs))
    return id_to_lang[best_id], float(probs[best_id])


def detect_file(
    whisper,
    extractor,
    path: Path,
    device,
    samples: int = DEFAULT_SAMPLES,
    window: float = DEFAULT_WINDOW,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Detect a file's language from several windows spread across it, by majority vote.

    A single window can land inside a long stretch that isn't representative
    of the episode as a whole — a multi-minute Swedish intro ahead of an
    otherwise-English interview is a real example from this project's own test
    library, and it fooled a single-window read. Spreading windows across the
    middle of the file means one confidently-wrong window no longer decides
    the outcome. The confidence returned is the mean over only the windows
    that agreed with the winner, so a narrow majority reads as genuinely less
    certain rather than borrowing one window's confident number.

    Returns (language, confidence, per-window readings) — the readings are
    exposed for callers that want to explain a borderline verdict.
    """
    total = probe_duration(path)
    offsets = sample_offsets(total, samples, window)

    readings = []
    for offset in offsets:
        window_samples, rate = sample_window(path, offset, window)
        readings.append(detect(whisper, extractor, window_samples, rate, device))

    votes = Counter(language for language, _ in readings)
    winner, _ = votes.most_common(1)[0]
    agreeing = [confidence for language, confidence in readings if language == winner]
    return winner, sum(agreeing) / len(agreeing), readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, type=Path, help="audio file to identify")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"windows spread across the file, majority vote decides; default: {DEFAULT_SAMPLES}",
    )
    parser.add_argument(
        "--window", type=float, default=DEFAULT_WINDOW, help=f"seconds per window; default: {DEFAULT_WINDOW}"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if not args.file.is_file():
        fail(f"no such file: {args.file}")

    device = select_device(args.device)
    whisper, extractor = load_model(args.model, device)
    language, confidence, readings = detect_file(
        whisper, extractor, args.file, device, args.samples, args.window
    )

    if len(readings) > 1:
        detail = ", ".join(f"{lang} {conf:.0%}" for lang, conf in readings)
        print(f"windows: {detail}", file=sys.stderr)
    print(f"{language} ({confidence:.0%} confidence)")


if __name__ == "__main__":
    main()
