"""Post a structured summary to a Discord webhook.

    $env:DISCORD_WEBHOOK = "https://discord.com/api/webhooks/..."
    uv run notify.py --title "Run finished" --status ok `
        --field "Speakers=2" --field "DER=1.29%"

The webhook URL is a credential — anyone holding it can post to the channel — so
it is read from the environment and never written to a tracked file, the same
rule as the Hugging Face token.

Uses stdlib urllib rather than requests, so it adds no dependency.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

from common import fail

# Discord renders the left edge of an embed in this colour, which is the fastest
# way to see how a run went without reading any of it.
COLOURS = {
    "ok": 0x2ECC71,
    "warn": 0xF1C40F,
    "fail": 0xE74C3C,
    "info": 0x5865F2,
}


def post(webhook: str, payload: dict) -> None:
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "GoForTranscribe"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status not in (200, 204):
                fail(f"Discord returned {response.status}")
    except urllib.error.HTTPError as error:
        # Discord explains refusals in the body; the status alone is rarely enough.
        fail(f"Discord rejected the post ({error.code})\n{error.read().decode('utf-8', 'replace')}")
    except urllib.error.URLError as error:
        fail(f"could not reach Discord: {error.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="", help="markdown; use - for stdin")
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a name/value pair; repeatable. Prefix NAME with '!' to make it full-width",
    )
    parser.add_argument("--status", choices=sorted(COLOURS), default="info")
    parser.add_argument("--footer", default="GoForTranscribe")
    parser.add_argument("--webhook", help="default: $DISCORD_WEBHOOK")
    args = parser.parse_args()

    webhook = args.webhook or os.environ.get("DISCORD_WEBHOOK")
    if not webhook:
        fail(
            "no Discord webhook",
            hint='Set it in the environment:\n  $env:DISCORD_WEBHOOK = "https://discord.com/..."',
        )

    description = sys.stdin.read() if args.description == "-" else args.description

    fields = []
    for entry in args.field:
        if "=" not in entry:
            fail(f"--field needs NAME=VALUE, got: {entry}")
        name, value = entry.split("=", 1)
        # Discord lays out up to three inline fields per row; full-width is better
        # for anything long enough to wrap.
        inline = not name.startswith("!")
        fields.append({"name": name.lstrip("!"), "value": value or "-", "inline": inline})

    embed = {
        "title": args.title,
        "color": COLOURS[args.status],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": args.footer},
    }
    if description:
        embed["description"] = description[:4096]
    if fields:
        embed["fields"] = fields[:25]

    post(webhook, {"embeds": [embed]})
    print(f"Posted to Discord: {args.title}")


if __name__ == "__main__":
    main()
