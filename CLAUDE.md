# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Stage 1 (diarization) exists as a spike: `diarize.py`, on pyannote.audio. Transcription, merging,
and the stability harness are not written yet. There are no tests.

```powershell
uv sync                                    # create/refresh the venv from uv.lock
uv run diarize.py --file audio.mp3 --speakers 2
uv run diarize.py --file audio.mp3 --output turns.json      # or .rttm
uv run diarize.py --file audio.mp3 --offset 0.05            # one point of a stability sweep
uv run diarize.py --file audio.mp3 --device cpu             # force the CPU path
```

A Hugging Face token is required — see [Gated models](#gated-models).

## Goal

**Speaker diarization first, transcription second.** The sibling project
[GoForWhisper](https://github.com/LittleAndi/GoForWhisper) — checked out locally alongside this
one — already does plain transcription well; the point of this project is to find the *best* way to
answer "who spoke when", then attach transcribed text to each identified speaker.

Diarization quality is the deliverable. Transcription is a solved dependency, not the problem
being worked on.

Constraints from the project owner:

- **Ship CLI tools.** Not a library, not a service, not a notebook — command-line programs that
  take an audio file and produce a speaker-attributed transcript. More than one tool is fine and
  expected: the pipeline stages (preprocess, diarize, transcribe, merge) and the evaluation
  harness are each reasonable candidates for their own command, so intermediate results can be
  inspected and re-run without redoing the whole chain.
- Python and/or .NET are both acceptable — pick per component, not per dogma.
- Keep the structure **as simple as possible**. No premature layering, no solution-per-concern
  sprawl. Prefer one project/one script until there is a real reason to split.
- **Offload to the GPU wherever possible**, but never require it — see
  [GPU](#gpu--offload-by-default-degrade-gracefully). Code must still run on a machine without
  this hardware.

## Prior art: GoForWhisper — read this before choosing an approach

[GoForWhisper](https://github.com/LittleAndi/GoForWhisper) is a .NET 10 CLI (Whisper.net +
sherpa-onnx) that already implements a full diarization pipeline. **Its `README.md` contains
measured reliability data that should directly shape decisions here — read it rather than
re-discovering the same failures.** `CLAUDE.local.md` records where it sits on this machine.

What it established:

- **Whisper cannot diarize.** whisper.cpp's `--diarize` only compares stereo channel energy;
  `--tinydiarize` marks turn boundaries without identifying anyone. Diarization must be a
  separate pass over the same samples.
- **The sherpa-onnx stack (pyannote segmentation-3.0 + an embedding model + clustering) is
  unstable on similar-sounding voices.** On a Swedish two-host podcast with the speaker count
  fixed at 2, shifting the audio by a few *inaudible* milliseconds flipped the result between
  correct (~50/50 speech split) and total collapse (~3/97, both hosts in one cluster). A control
  recording of one male + one female TTS voice returned 49/51 at every shift, so the pipeline is
  sound — those two voices simply sit on the knife edge.
- **Therefore: never tune against a single run.** Re-run with a small time offset
  (`ffmpeg -ss 0.05`) before concluding a change helped. A good result on well-separated
  speakers predicts nothing about hard audio.
- **Bigger models are not better.** NeMo TitaNet-large was among the worst; the ~40 MB
  ERes2Net-base was the most stable. WeSpeaker VoxCeleb CAM++ fails even the trivial control.
- **Automatic speaker-count estimation is the weakest link** — threshold-based estimation
  produced 39 speakers on a 38-minute two-host episode. Passing a known count avoids it.
- Its own README concludes that **[pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.1
  is materially more robust** because it does overlap-aware clustering — rejected there only
  because that project deliberately avoided a Python + Hugging Face token dependency. **This
  project has no such constraint**, which makes pyannote.audio the obvious first thing to
  evaluate here.

Reusable design pieces worth carrying over rather than reinventing:

- **Merge transcript and diarization by greatest temporal overlap.** The two models cut audio at
  different places (Whisper on sentence boundaries, the segmenter on turn boundaries), so segment
  lists never align. See `GoForWhisper\Services\SpeakerLabeler.cs` — assign each transcript
  segment the speaker it shares the most time with; a segment overlapping no detected speech is
  unattributed, not force-assigned.
- **Speaker numbers are cluster ids, not identities.** `SPEAKER 1` means "same voice as the other
  SPEAKER 1 lines" — arbitrary, and not stable across runs or files.
- **Preprocess once, feed both passes.** GoForWhisper decodes to 16 kHz mono, level-normalises to
  −20 dBFS, and trims leading/trailing silence (the main source of hallucinated filler), then
  shifts timestamps back past the trim so they still refer to the original file.

## Evaluation discipline

Because a single confident-looking run proves nothing (see above), any comparison of approaches
needs a repeatable measurement, not eyeballing. At minimum:

- A control pair of clearly distinct voices — an approach that fails this is disqualified outright.
- A hard case with similar voices, run at several small time offsets, reporting the *spread* of
  results rather than the best one.
- Speech-time share per speaker as the headline metric (correct ≈ the true split; collapse shows
  as one speaker taking ~everything). Move to DER against reference labels once a reference exists.

## Environment (verified on this machine)

| Tool          | Version                                                               |
| ------------- | --------------------------------------------------------------------- |
| .NET          | 10.0.110                                                              |
| Python        | 3.11                                                                  |
| uv            | 0.6.9                                                                 |
| ffmpeg        | on `PATH`                                                             |
| gh            | installed                                                             |
| GPU 1         | NVIDIA RTX 5060 Ti, 16 GB, **compute capability 12.0**, driver 610.47 |
| GPU 0         | AMD Radeon integrated                                                 |
| CUDA toolkits | 12.9 and 13.3 installed; `nvcc` on `PATH` is 12.9                     |

Shell is PowerShell 7 on Windows 11. Use `uv` for Python dependency management rather than bare
`pip`/`venv`.

**Machine-specific paths do not belong in this file.** Absolute paths, sample-audio locations,
and anything else particular to one developer's machine go in `CLAUDE.local.md`, which is
gitignored. Read it if present; keep its contents out of anything committed. Secrets — the
Hugging Face token above all — live in the environment and never in a tracked file.

## GPU — offload by default, degrade gracefully

Prefer GPU execution everywhere it is available, but **always fall back to CPU** rather than
hard-failing. Nothing here may assume this specific rig.

### The Blackwell trap — read before installing PyTorch

The RTX 5060 Ti is **Blackwell, compute capability 12.0 (`sm_120`)**. A default
`uv add torch` / `pip install torch` pulls a build whose kernels stop at `sm_90`, and it will
fail at runtime with *"no kernel image is available for execution on the device"* — or silently
sit on the CPU. **PyTorch must come from a CUDA 12.8-or-newer index:**

```powershell
uv add torch torchaudio --index https://download.pytorch.org/whl/cu128
```

The installed CUDA *toolkits* are irrelevant to PyTorch — the wheel ships its own CUDA runtime,
and only the driver (610.47, new enough) matters. The toolkits matter only for native code that
links against them: Whisper.net 1.9.1 needs the **13.x** toolkit specifically, which is why
`v13.3` is installed alongside `v12.9`.

Always verify rather than assume, since the failure mode is a silent CPU fallback:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

### Device selection

The AMD part is an integrated display adapter, not a compute target — PyTorch on Windows has no
ROCm build, so **do not try to use it for diarization**. Its usefulness is that it can drive the
displays, leaving the full 16 GB free. Select the NVIDIA card explicitly (`cuda:0` is correct
once CUDA enumerates only the NVIDIA device; `CUDA_VISIBLE_DEVICES` pins it if that ever changes)
and expose a `--device` flag defaulting to auto-detect.

For the .NET side, GoForWhisper measured **Vulkan at ~9.4x realtime versus CUDA at ~8.0x on this
exact GPU** — ggml's Vulkan backend uses cooperative-matrix instructions on Blackwell. Do not
assume CUDA wins; measure.

### What the GPU actually buys, given the quality focus

Be honest about the mechanism: **for a fixed model and fixed settings, the GPU changes speed, not
accuracy.** It serves the quality goal indirectly, and those indirect routes are the point:

1. **It makes the stability protocol affordable.** The single most important finding carried over
   from GoForWhisper is that one run proves nothing — a config must be re-run at several time
   offsets to see whether it is stable. That is an N-times-the-work evaluation, and on CPU it is
   expensive enough that it quietly stops being done. On GPU it is cheap, so it actually happens.
2. **It affords heavier models.** Overlap-aware pyannote pipelines and large ASR models are the
   quality-relevant choices, and they are the ones CPU inference makes impractical.
3. **16 GB fits diarization and ASR resident at once**, so a single pass over long audio does not
   have to page models in and out.

### GPU non-determinism interacts badly with the knife-edge instability

TF32 matmuls and cuDNN algorithm selection make GPU results non-bit-identical to CPU, and
non-identical between runs. Normally irrelevant — but clustering on similar-sounding voices has
already been measured sitting on a decision boundary where inaudible perturbations flip the
outcome. So for **evaluation** runs, remove that variable: fix seeds, and consider disabling TF32
(`torch.backends.cuda.matmul.allow_tf32 = False`) so embeddings are computed in full fp32.
Otherwise a config change and a numerical coin flip are indistinguishable in the results.

## First measured result — pyannote is stable where sherpa-onnx was not

Measured on a 6-minute excerpt of the same Swedish two-host podcast that broke the sherpa-onnx
stack, speaker count fixed at 2, varying only the time offset. Share of speech time given to
each speaker:

| Offset          | 0 ms | 3 ms | 6 ms | 12 ms | 24 ms | 50 ms | 100 ms | 250 ms |
| --------------- | ---- | ---- | ---- | ----- | ----- | ----- | ------ | ------ |
| pyannote 4.0.7  | 76.5 | 76.0 | 76.1 | 76.1  | 76.2  | 76.2  | 76.1   | 76.5   |
| sherpa (before) | 58   | 3    | 3    | 55    | 58    | 58    | 58     | 62     |

**Spread of 0.5 points versus the old stack's coin flip between 3 and 62.** On this audio the
approach is no longer the source of variance, which is the single biggest reason to prefer it.
(The ~76 figure is an artefact of this short window — see
[below](#the-7624-was-the-excerpt-not-the-diarizer). What matters in this table is the spread
across a row, not its level.)

**Automatic speaker counting also works here.** Run without `--speakers`, pyannote finds exactly
2 and returns an identical split — where threshold-based estimation in the old stack produced 39
speakers on a comparable episode. The count no longer has to be supplied to get a sane result,
though supplying it is still free insurance.

Structure of the baseline run: 119 turns, 51 speaker changes, mean turn 2.1 s and 2.8 s for the
two speakers. That is a genuine back-and-forth, not the collapse signature (one speaker holding
nearly all the time with few changes).

### The 76/24 was the excerpt, not the diarizer

Run over the **full 38-minute episode** with automatic counting, the same pipeline returns
**51.7 / 48.3 across 819 turns** — an almost perfectly balanced two-host split, and still exactly
2 speakers. The lopsided 76/24 was a property of the 6-minute window, where one host happened to
be leading the topic; it is not a misattribution.

This is worth remembering as a methodology lesson: **speech-time share on a short excerpt is a
collapse detector, not an accuracy measure.** It catches the catastrophic failure (one cluster
eating everything) but says nothing about whether a merely-lopsided split is real. Only the full
recording, or a labelled reference, distinguishes those. Sanity-check a suspicious ratio against
a longer window before concluding anything from it.

Convergence to ~50/50 over 38 minutes is decent evidence the attribution is broadly right — a
systematic misassignment would be unlikely to average out that cleanly.

**Confirmed by ear.** Six clips per speaker, spread from 0:00 to 32:42 and cut with
`sample_turns.py`, were listened through: each speaker's folder is consistently one voice. So on
this episode pyannote is not merely stable, it is correct — which is exactly what the sherpa-onnx
stack could not manage on this same material.

**This is the reference result for the project.** Any future change to the diarization stage
should reproduce it: full episode, automatic counting, 2 speakers, ~52/48, and clips that hold up
by ear.

Throughput: the full episode diarized in **69 seconds on the GPU**, roughly 33x realtime.

### Intros and ads

Podcast intros and ad reads introduce voices that are not the hosts, and **ads can appear
anywhere in an episode**, not just at the start. That makes an exact `--speakers n` actively
dangerous on real material: pyannote does not discard the extra voice, it folds it into one of
the clusters you allowed, corrupting the split.

**Prefer automatic counting, or `--min-speakers`/`--max-speakers`, over an exact count** unless
the audio is known to be clean. On this episode automatic counting found 2 in every window
tested, including the intro — so the intro voice is not spawning a spurious cluster here. (This
particular episode carries no ads.)

Still missing: a **control** recording of two clearly distinct voices, and a scripted stability
harness (the sweep above was run by hand).

## Implementation notes worth knowing before editing `diarize.py`

Three things here are load-bearing and look like arbitrary choices:

- **pyannote installed is 4.0.7, not the 3.1 that most documentation describes.** The API moved:
  `Pipeline.from_pretrained` takes `token=`, not `use_auth_token=`, and the pipeline returns a
  `DiarizeOutput` rather than an `Annotation`. That object carries **two** annotations, and the
  distinction matters for this project: `exclusive_speaker_diarization` has overlapping speech
  removed and is documented as the one meant for downstream transcription, so it is the default
  here; `speaker_diarization` keeps overlaps and is behind `--overlapping`.
- **The default checkpoint is `pyannote/speaker-diarization-community-1`**, which is what 4.x
  recommends — `speaker-diarization-3.1` is the older generation.
- **Audio is decoded by ffmpeg and handed to the pipeline as an in-memory waveform dict, never
  as a path.** pyannote's own file reading goes through torchcodec, whose native libraries do not
  load on this Windows setup (`libtorchcodec_core*.dll` fails for every ffmpeg version it tries).
  Passing a waveform sidesteps the broken decoder entirely, which is why the tool works without
  repairing torchcodec. It also makes `--offset` exact and keeps input-format support broad.
  Reading the WAV uses stdlib `wave` plus numpy, so it adds no dependency. **Do not "simplify"
  this by passing the file path — it will break.**

### Gated models

pyannote's checkpoints require accepting terms on Hugging Face and a read token:

```powershell
$env:HF_TOKEN = "hf_..."
```

The token is read from `$HF_TOKEN`/`$HUGGINGFACE_TOKEN` or `--token`, and must never be written
to a tracked file. Note that `Pipeline.from_pretrained` returns `None` — rather than raising —
when the token is valid but the model's terms have not been accepted; `diarize.py` detects that
case and says so, because the bare `None` is otherwise a baffling failure.

## Repository hygiene

Model weights and audio are large and must stay out of git — GoForWhisper ignores `models/`,
`*.bin`, and `appsettings*.json` for exactly this reason. Download models on first run to a
`models/` directory (stream to a `.partial` sidecar first so an interrupted download cannot leave
a truncated file that later looks valid), and commit an `*.example.json` template instead of real
configuration. If a Hugging Face token is needed for pyannote, keep it in user secrets or an
environment variable, never in a committed file.
