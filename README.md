# GoForTranscribe

Speaker-attributed transcription: work out **who spoke when**, then transcribe each speaker's
turns so a conversation comes out split by person rather than as one undifferentiated wall of
text.

```text
[00:00:04.120 --> 00:00:08.640] SPEAKER 1: Välkommen till avsnitt 285.
[00:00:08.640 --> 00:00:12.310] SPEAKER 2: Tack, kul att vara här igen.
```

Everything runs locally on the GPU. Nothing is sent to a transcription API.

## Why a separate project

The sibling project [GoForWhisper](https://github.com/LittleAndi/GoForWhisper) already does
plain transcription well, and bolted diarization onto it via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). That works when voices are clearly
distinct and became **unreliable when they are not** — on a two-host podcast where both hosts
sound alike, shifting the audio by a few inaudible milliseconds flipped the result between a
correct speaker split and both hosts collapsing into a single cluster.

That project deliberately avoided a Python dependency, which ruled out
[pyannote.audio](https://github.com/pyannote/pyannote-audio) — the reference implementation,
whose overlap-aware clustering is materially more robust. This project has no such constraint,
and on the same audio pyannote holds where the old stack fell over. See
[Results](#results).

## Requirements

- Python 3.11 or 3.12, and [uv](https://docs.astral.sh/uv/)
- ffmpeg on `PATH`
- A Hugging Face account, for the gated pyannote models
- An NVIDIA GPU is strongly recommended but not required; everything falls back to CPU

```powershell
uv sync
uv run hf auth login
```

The pyannote models are gated: accept the terms at
[hf.co/pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
first, or loading fails with a message telling you so.

> **On an NVIDIA Blackwell card (RTX 50-series), install from the CUDA 12.8 index.** That is
> already configured in `pyproject.toml`. A default `pip install torch` has no kernels for these
> cards and will either error or silently drop to the CPU.

## Usage

The whole chain in one command:

```powershell
uv run pipeline.py --file episode.mp3 --output episode.txt --timestamps
```

Or a whole folder, writing `<name>.txt` beside each input:

```powershell
uv run pipeline.py --folder podcasts\ --timestamps
uv run pipeline.py --folder podcasts\ --output-dir transcripts\ --recursive
```

A batch loads each model **once** rather than once per file, which is why it runs in two passes —
diarize everything, then transcribe everything. Files that already have a `.txt` are skipped
unless `--overwrite` is passed, so an interrupted batch resumes where it stopped, and one
unreadable file is reported at the end rather than killing the run.

Work is done in groups of `--batch-files` (default 8), and **each group's transcripts are written
before the next group starts**. Both models are loaded per group, so a smaller number costs more
model loads — the default trades one extra pair of them per group for two things worth having on
a long run: turns and segments for at most `N` files held at once, and results on disk as it goes
rather than only at the very end. `--batch-files 0` processes the whole folder as one group.

`--recursive` descends into subdirectories, and the structure is mirrored in `--output-dir` so
two files with the same name in different folders cannot overwrite each other.

> **`--language` applies to every file in the batch.** It defaults to `sv`, and Whisper will
> quietly _translate_ rather than refuse: an English recording transcribed with `--language sv`
> comes back as fluent Swedish. For a mixed-language folder use `--language auto`.

Or one stage at a time, which is what you want while iterating — each stage's output can be
inspected, and the expensive stages need not be repeated:

```powershell
# 1. who spoke when
uv run diarize.py --file episode.mp3 --output turns.json

# 2. what was said
uv run transcribe.py --file episode.mp3 --output transcript.json

# 3. put them together
uv run merge.py --turns turns.json --transcript transcript.json --timestamps --output final.txt
```

Useful flags:

| Flag                                | Stage               | Effect                                                                                             |
| ----------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------- |
| `--speakers n`                      | diarize             | Fix the speaker count. **Avoid unless the audio is clean** — see [Ads and intros](#ads-and-intros) |
| `--min-speakers` / `--max-speakers` | diarize             | Bound the count without fixing it                                                                  |
| `--offset` / `--duration`           | diarize, transcribe | Work on a window rather than the whole file                                                        |
| `--language`                        | transcribe          | Defaults to `sv`; `auto` lets the model decide                                                     |
| `--chunk-length`                    | transcribe          | Switch to chunked decoding — see [Memory](#memory)                                                 |
| `--model`                           | both                | Any compatible checkpoint                                                                          |
| `--device`                          | both                | `auto`, `cuda`, or `cpu`                                                                           |
| `--timestamps`                      | merge               | Prefix each line with its time range                                                               |
| `--keep-intermediate`               | pipeline            | Also save the per-stage JSON                                                                       |
| `--batch-files n`                   | pipeline            | Files per group when processing a folder; `0` for all at once                                     |
| `--overwrite`                       | pipeline            | Redo folder entries that already have a `.txt`                                                    |

Diarization emits `.json` or `.rttm`; transcription `.json` or `.txt`; merge `.json` or plain
text.

### Memory

Transcription defaults to Whisper's **sequential** long-form decoding, which holds one 30-second
window at a time. `--chunk-length 30` switches to chunked decoding, which cuts fixed windows and
decodes `--batch-size` of them at once.

**Chunking is the faster option in theory and the riskier one in practice.** It holds several
windows of activations simultaneously, and on a 16 GB card that is also driving a desktop
session, that was enough to exhaust VRAM — at which point Windows spills into system memory and
the machine can hang rather than reporting an out-of-memory error. Sequential decoding measured
_faster_ here anyway (3.9x realtime against 2.5x), as well as more accurate, so reach for
`--chunk-length` only if you have measured a reason to.

Both stages print free VRAM before loading and warn when it looks tight. Closing browsers and
chat apps genuinely helps; so does letting the integrated GPU drive the displays.

## Is the result any good?

Speaker diarization fails in a way that is easy to miss: any single run produces confident,
plausible-looking output. Three tools exist to check it rather than trust it.

```powershell
# does the answer survive an inaudible time shift?
uv run stability.py --file episode.mp3

# cut clips per speaker and listen — each folder should be one voice
uv run sample_turns.py --turns turns.json --file episode.mp3

# score against known labels
uv run make_control.py                     # builds a control from two distinct voices
uv run diarize.py --file samples/control.wav --output out/control.json
uv run evaluate.py --reference samples/control-reference.json --hypothesis out/control.json
```

`stability.py` re-runs diarization at several small time offsets. The shifts are inaudible and
meaningless, so a sound configuration returns the same answer each time; one sitting on a
clustering decision boundary does not, and there any single run is worthless.

## Results

Measured on a 6-minute excerpt of a Swedish two-host podcast — the same material where the
sherpa-onnx stack collapsed both hosts into one cluster. Share of speech time given to the
dominant speaker, varying only an inaudible time offset:

| Offset               | 0 ms | 3 ms | 6 ms | 12 ms | 24 ms | 50 ms | 100 ms | 250 ms |
| -------------------- | ---- | ---- | ---- | ----- | ----- | ----- | ------ | ------ |
| **pyannote 4.0.7**   | 76.5 | 76.0 | 76.1 | 76.1  | 76.2  | 76.2  | 76.1   | 76.5   |
| sherpa-onnx (before) | 58   | 3    | 3    | 55    | 58    | 58    | 58     | 62     |

**0.5 points of spread, against a coin flip between 3 and 62.** What matters is the spread
across a row, not its level — the ~76 is a property of that particular 6-minute window, where one
host was leading the topic. Over the **full 38-minute episode the split is 51.7 / 48.3** across
819 turns.

Other results on the same material:

- **Automatic speaker counting finds exactly 2**, in every window tested. The old stack's
  threshold-based estimation produced 39 speakers on a comparable episode.
- **Attribution confirmed by ear**: six clips per speaker, spread across the episode, are each
  consistently one voice.
- **DER 1.29% on the control**, with zero missed speech and zero time attributed to the wrong
  speaker.
- **Throughput**: the full episode diarizes in 69 seconds on an RTX 5060 Ti, about 33x realtime.

### End to end

The full 38-minute episode through `pipeline.py`:

|                     |                                                          |
| ------------------- | -------------------------------------------------------- |
| Speakers found      | 2, automatically                                         |
| Diarization turns   | 819                                                      |
| Transcript segments | 633                                                      |
| Output blocks       | 127                                                      |
| Unattributed        | 1 segment — a one-second "Tack!" outside detected speech |
| Peak VRAM           | 6.5 GB                                                   |

The unattributed segment is the intended behaviour rather than a miss: a transcript segment
overlapping no detected speech is left unlabelled instead of being forced onto the nearest
speaker.

Note when reading the output that blocks always alternate between speakers — consecutive
segments from one speaker are merged into a single block, so alternation is a property of the
formatting, not evidence about the diarization.

### Ads and intros

Podcast intros and ad reads contain voices that are not the hosts, and ads can appear anywhere in
an episode. An exact `--speakers n` does not discard the extra voice — it folds it into one of
the clusters you allowed, corrupting the split. Prefer automatic counting, or a
`--min-speakers`/`--max-speakers` range, unless the audio is known to be clean.

## Models

| Stage         | Default                                    | Why                                                                                                               |
| ------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Diarization   | `pyannote/speaker-diarization-community-1` | Overlap-aware clustering; the reason this project exists                                                          |
| Transcription | `KBLab/kb-whisper-large`                   | The National Library of Sweden's Whisper fine-tune, reported at ~47% lower WER than `whisper-large-v3` on Swedish |

Both download on first use. For non-Swedish audio, pass `--model openai/whisper-large-v3` and a
matching `--language`.

## Licence

MIT — see [LICENSE](LICENSE).
