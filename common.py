"""Shared helpers for the pipeline stages.

Audio decoding, device selection and timestamp formatting are identical across
the stages, and the decoding in particular carries hard-won detail (see
`load_waveform`) that must not drift between them.
"""

import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import NoReturn

SAMPLE_RATE = 16000


def fail(message: str, *, hint: str | None = None) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(f"\n{hint}", file=sys.stderr)
    raise SystemExit(1)


def decode(source: Path, dest: Path, offset: float = 0.0, duration: float | None = None) -> None:
    """Decode to 16 kHz mono 16-bit WAV, the form every stage wants.

    Going through ffmpeg rather than letting a library read the file keeps input
    format support broad (mp3/m4a/flac/wma) and makes --offset exact.
    """
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found on PATH", hint="Install it, or add its bin directory to PATH.")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if offset:
        command += ["-ss", f"{offset}"]
    if duration:
        command += ["-t", f"{duration}"]
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


def load_waveform(path: Path):
    """Read a 16-bit PCM WAV as (samples, sample_rate).

    Deliberately not letting pyannote or transformers read the file themselves:
    their decoding goes through torchcodec, whose native libraries do not load on
    this Windows setup. Reading the samples here bypasses that entirely, so the
    tools work without a repaired torchcodec. Stdlib `wave` keeps it
    dependency-free. **Do not replace this with a file path handed to a library.**
    """
    import numpy

    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            fail(f"expected 16-bit PCM from ffmpeg, got {8 * source.getsampwidth()}-bit")
        frames = source.readframes(source.getnframes())
        channels = source.getnchannels()
        rate = source.getframerate()

    samples = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32) / 32768.0
    return samples.reshape(-1, channels).T, rate


def neutralize_broken_torchcodec() -> bool:
    """Stop a broken torchcodec install from taking the process down with it.

    torchcodec needs FFmpeg's *shared* libraries. A static ffmpeg build — the
    common Windows "essentials" build — satisfies the ffmpeg CLI we decode with
    but ships no DLLs, so `import torchcodec` raises RuntimeError.

    pyannote tolerates that and merely warns. transformers does not: its ASR
    pipeline asks `is_torchcodec_available()`, which answers yes because the
    package *is* installed, and then imports it and dies — even though we hand it
    a decoded numpy array and no decoding is needed at all.

    Registering a stub module makes the import succeed and the `isinstance` check
    against AudioDecoder simply return False, which is the correct answer for our
    inputs. A working torchcodec is always preferred; the stub is only installed
    when the real import fails. The permanent fix is a shared-libraries FFmpeg
    build on PATH, at which point this becomes a no-op.

    Returns True when the real module is usable, False when stubbed.
    """
    import sys
    import types

    if "torchcodec" in sys.modules:
        return True

    try:
        import torchcodec  # noqa: F401

        return True
    except Exception:
        pass

    import importlib.machinery

    module = types.ModuleType("torchcodec")
    decoders = types.ModuleType("torchcodec.decoders")

    class AudioDecoder:
        """Placeholder so isinstance() checks resolve to False."""

    decoders.AudioDecoder = AudioDecoder
    module.decoders = decoders

    # importlib raises "__spec__ is None" for a bare ModuleType, and treating the
    # stub as a package (__path__) lets `import torchcodec.decoders` resolve.
    module.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
    module.__path__ = []
    decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)

    sys.modules["torchcodec"] = module
    sys.modules["torchcodec.decoders"] = decoders
    return False


def select_device(requested: str):
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


def format_timestamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def resolve_token(explicit: str | None) -> str | None:
    """Find a Hugging Face credential, if one is configured.

    Prefers a stored login over an environment variable, because `hf auth login`
    persists across shells and keeps the token off command lines. Returns None
    when nothing is configured, which is fine for ungated models.
    """
    import os

    from huggingface_hub import get_token

    return (
        explicit
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or get_token()
    )
