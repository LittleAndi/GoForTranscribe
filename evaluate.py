"""Score a diarization result against reference labels.

    uv run evaluate.py --reference samples/control-reference.json --hypothesis out/control.json

Reports diarization error rate, the standard measure: the share of speech time
that is wrong, counting missed speech, speech invented where there was none, and
time given to the wrong speaker. Lower is better and 0 is perfect; it is a rate
rather than a percentage and can exceed 100%.

DER is what the project has been missing. Speech-time share only catches total
collapse, and listening does not scale — this compares against known labels and
handles the fact that cluster ids are arbitrary by finding the best possible
mapping between reference and hypothesis speakers before scoring.
"""

import argparse
import json
from pathlib import Path

from common import fail, format_timestamp


def to_annotation(turns: list[dict], name: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=name)
    for turn in turns:
        if turn["end"] > turn["start"]:
            annotation[Segment(turn["start"], turn["end"])] = turn["speaker"]
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, type=Path, help="known-correct labels")
    parser.add_argument("--hypothesis", required=True, type=Path, help="turns from diarize.py")
    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="seconds ignored either side of a boundary, the usual convention; 0 to score strictly",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="ignore regions where the reference has more than one speaker at once",
    )
    args = parser.parse_args()

    for path in (args.reference, args.hypothesis):
        if not path.is_file():
            fail(f"no such file: {path}")

    from pyannote.metrics.diarization import DiarizationErrorRate

    reference = to_annotation(json.loads(args.reference.read_text(encoding="utf-8")), "ref")
    hypothesis = to_annotation(json.loads(args.hypothesis.read_text(encoding="utf-8")), "hyp")

    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    rate = metric(reference, hypothesis, detailed=True)

    total = rate["total"]
    print(f"Reference: {len(reference.labels())} speakers, {format_timestamp(total)} of speech")
    print(f"Hypothesis: {len(hypothesis.labels())} speakers")
    print()
    print(f"  DER            {100 * rate['diarization error rate']:6.2f}%")
    for label, key in (
        ("missed speech", "missed detection"),
        ("false alarm", "false alarm"),
        ("wrong speaker", "confusion"),
    ):
        seconds = rate[key]
        print(f"  {label:<14} {100 * seconds / total if total else 0:6.2f}%  ({seconds:.1f}s)")

    # The mapping is what makes cluster ids comparable; showing it makes a
    # speaker-count mismatch obvious rather than hidden inside the number.
    mapping = metric.optimal_mapping(reference, hypothesis)
    print("\nBest speaker mapping (hypothesis -> reference):")
    for source, target in sorted(mapping.items()):
        print(f"  {source} -> {target}")
    unmapped = set(hypothesis.labels()) - set(mapping)
    if unmapped:
        print(f"  unmatched hypothesis speakers: {', '.join(sorted(unmapped))}")


if __name__ == "__main__":
    main()
