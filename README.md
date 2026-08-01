# GoForTranscribe

Speaker-attributed transcription: work out **who spoke when**, then transcribe each speaker's
turns so a conversation comes out split by person rather than as one undifferentiated wall of
text.

```text
[00:00:04.120 --> 00:00:08.640] SPEAKER 1: Välkommen till avsnitt 285.
[00:00:08.640 --> 00:00:12.310] SPEAKER 2: Tack, kul att vara här igen.
```

**Status: just started.** There is no implementation yet — this repository currently holds the
project's goal, prior findings, and licence. Usage instructions will land here alongside the
first working pipeline.

## Why a separate project

The sibling project [GoForWhisper](https://github.com/LittleAndi/GoForWhisper) already does
plain transcription well, and bolted diarization onto it via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). That works when voices are clearly
distinct and becomes **unreliable when they are not** — on a two-host podcast where both hosts
sound alike, shifting the audio by a few inaudible milliseconds flipped the result between a
correct speaker split and both hosts collapsing into a single cluster.

Here, diarization quality *is* the deliverable rather than a feature bolted on at the end.
Transcription is treated as a solved dependency.

## Approach

The distinguishing constraint: GoForWhisper deliberately avoided a Python dependency, which
ruled out [pyannote.audio](https://github.com/pyannote/pyannote-audio) — the reference
implementation, whose overlap-aware clustering is materially more robust. **This project has no
such constraint**, so pyannote.audio is the first thing to evaluate.

Python and .NET are both fair game, chosen per component. The structure stays as simple as the
problem allows.

Whichever stack wins, the shape of the pipeline is the same:

1. **Preprocess once** — decode to 16 kHz mono, normalise level, trim silence — and feed the
   same samples to both passes.
2. **Diarize** — segment speech into turns, embed each turn as a voice vector, cluster the
   vectors into speakers.
3. **Transcribe** — run speech-to-text over the same audio.
4. **Merge** — the two passes cut audio at different places (transcription on sentence
   boundaries, diarization on turn boundaries), so each transcript segment is assigned the
   speaker it shares the most time with.

Speaker numbers are cluster ids, not identities: `SPEAKER 1` means "the same voice as the other
`SPEAKER 1` lines", and the numbering is not stable across runs or files.

## Measuring, not eyeballing

Any single run produces confident-looking output, so a good-looking result proves very little.
Comparisons between approaches are made against:

- A **control** pair of clearly distinct voices — an approach that fails this is out.
- A **hard case** with similar voices, run at several small time offsets, reporting the spread
  of results rather than the best one.

## Licence

MIT — see [LICENSE](LICENSE).
