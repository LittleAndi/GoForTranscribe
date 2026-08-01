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

- Python and/or .NET are both acceptable — pick per component, not per dogma.
- Keep the structure **as simple as possible**. No premature layering, no solution-per-concern
  sprawl. Prefer one project/one script until there is a real reason to split.

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

| Tool    | Version / path                    |
| ------- | --------------------------------- |
| .NET    | 10.0.110                          |
| Python  | 3.11.9                            |
| uv      | 0.6.9 (`on PATH`) |
| ffmpeg  | `on PATH`   |
| gh      | installed                         |

Shell is PowerShell 7 on Windows 11. Use `uv` for Python dependency management rather than bare
`pip`/`venv`.

## Repository hygiene

Model weights and audio are large and must stay out of git — GoForWhisper ignores `models/`,
`*.bin`, and `appsettings*.json` for exactly this reason. Download models on first run to a
`models/` directory (stream to a `.partial` sidecar first so an interrupted download cannot leave
a truncated file that later looks valid), and commit an `*.example.json` template instead of real
configuration. If a Hugging Face token is needed for pyannote, keep it in user secrets or an
environment variable, never in a committed file.
