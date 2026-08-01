# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

The repository is **empty** — no code, no `git init` yet. There is nothing to build, lint, or
test, so this file describes the goal and the prior art rather than commands that do not exist
yet. Add build/run/test commands here as soon as the first project is scaffolded.

Remote to push to (already created, empty): `https://github.com/LittleAndi/GoForTranscribe.git`

## Goal

**Speaker diarization first, transcription second.** The sibling project the sibling GoForWhisper checkout
already does plain transcription well; the point of this project is to find the *best* way to
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

the sibling GoForWhisper checkout is a .NET 10 CLI (Whisper.net + sherpa-onnx) that already implements a
full diarization pipeline. **Its `README.md` contains measured reliability data that should
directly shape decisions here — read it rather than re-discovering the same failures.**

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

| Tool          | Version / path                                            |
| ------------- | --------------------------------------------------------- |
| .NET          | 10.0.110                                                  |
| Python        | 3.11.9                                                    |
| uv            | 0.6.9 (`on PATH`)                        |
| ffmpeg        | `on PATH`                           |
| gh            | installed                                                 |
| GPU 1         | NVIDIA RTX 5060 Ti, 16 GB, **compute capability 12.0**, driver 610.47 |
| GPU 0         | AMD Radeon integrated                                     |
| CUDA toolkits | 12.9 and 13.3 installed; `nvcc` on `PATH` is 12.9         |

Shell is PowerShell 7 on Windows 11. Use `uv` for Python dependency management rather than bare
`pip`/`venv`.

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

## Repository hygiene

Model weights and audio are large and must stay out of git — GoForWhisper ignores `models/`,
`*.bin`, and `appsettings*.json` for exactly this reason. Download models on first run to a
`models/` directory (stream to a `.partial` sidecar first so an interrupted download cannot leave
a truncated file that later looks valid), and commit an `*.example.json` template instead of real
configuration. If a Hugging Face token is needed for pyannote, keep it in user secrets or an
environment variable, never in a committed file.
