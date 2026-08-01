"""Check whether a diarization result is real or a coin flip.

    uv run stability.py --file episode.mp3

Re-runs diarization at several small time offsets. The shifts are inaudible and
semantically meaningless, so a sound configuration returns essentially the same
answer every time. A configuration sitting on a clustering decision boundary does
not, and any single run of it means nothing — which is exactly the failure that
made the previous sherpa-onnx stack unusable on similar-sounding voices.

The pipeline is loaded once and reused across offsets, so a sweep costs about as
much as N diarization runs with none of the model-loading overhead.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from common import fail, format_timestamp, make_deterministic, resolve_token, select_device
from diarize import DEFAULT_MODEL, TOKEN_HINT, load_pipeline, run, shares

# Milliseconds. Spaced roughly logarithmically: the earlier sherpa-onnx failures
# showed up at single-digit milliseconds, while genuine drift needs longer shifts.
DEFAULT_OFFSETS = [0, 3, 6, 12, 24, 50, 100, 250]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, type=Path, help="audio to sweep")
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=DEFAULT_OFFSETS,
        help=f"offsets in milliseconds (default: {' '.join(map(str, DEFAULT_OFFSETS))})",
    )
    parser.add_argument("--speakers", type=int, help="fix the speaker count")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--duration", type=float, help="only sweep this many seconds")
    parser.add_argument("--output", type=Path, help="write the full result table as JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--token")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.file.is_file():
        fail(f"no such file: {args.file}")

    token = resolve_token(args.token)
    if not token:
        fail("no Hugging Face token", hint=TOKEN_HINT)

    # Determinism is the point here: any variation left in the output should come
    # from the offset under test, not from TF32 or cuDNN autotuning.
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
    if not constraints:
        print("Speaker count is open — estimation is part of what is being tested.", file=sys.stderr)

    results = []
    started = time.perf_counter()
    for milliseconds in args.offsets:
        turns = run(
            pipeline,
            args.file,
            offset=milliseconds / 1000.0,
            duration=args.duration,
            constraints=constraints,
            progress=False,
        )
        share = shares(turns)
        ordered = sorted(share.values(), reverse=True)
        results.append(
            {
                "offset_ms": milliseconds,
                "speakers": len(share),
                "shares": [round(value, 1) for value in ordered],
                "turns": len(turns),
            }
        )
        print(
            f"  {milliseconds:>6.0f} ms  {len(share)} speaker(s)  "
            f"{' / '.join(f'{value:.1f}' for value in ordered)}",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - started
    print(f"\nSwept {len(results)} offsets in {format_timestamp(elapsed)}", file=sys.stderr)

    counts = {result["speakers"] for result in results}
    tops = [result["shares"][0] for result in results if result["shares"]]
    spread = max(tops) - min(tops) if tops else 0.0

    print(f"Speaker count: {'stable at ' + str(counts.pop()) if len(counts) == 1 else f'UNSTABLE {sorted(counts)}'}")
    print(f"Dominant-speaker share: {min(tops):.1f}–{max(tops):.1f}%  (spread {spread:.1f} points)")

    # A few points of movement is ordinary — the windows genuinely differ. Tens of
    # points means the clustering, not the audio, is deciding the answer.
    if len(counts) > 1:
        verdict = "UNSTABLE — the speaker count itself changes with an inaudible shift"
    elif spread > 10:
        verdict = "UNSTABLE — treat any single run as meaningless"
    elif spread > 3:
        verdict = "MARGINAL — re-run on more audio before trusting it"
    else:
        verdict = "STABLE"
    print(f"Verdict: {verdict}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"file": args.file.name, "spread": round(spread, 2), "runs": results}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
