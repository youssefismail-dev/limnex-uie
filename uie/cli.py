"""
uie/cli.py

The command line of the Limnex URL Intelligence Engine (D2 step 4).

    uie generate --weights limnex-uie-e16_b5.pt --known-urls limnex-uie-known-urls.bloom.gz --label phishing -n 10
    uie generate --weights ... --known-urls ... --label benign -n 50 --seed 7 \
                 --blocklist threat_feed_snapshot.txt
    uie score    --weights limnex-uie-e16_b5.pt example.com/login https://example.org/
    uie score    --weights ... --file urls.txt

`python -m uie ...` does the same. Generated URLs go to standard output, one per line; the gate's
report and the responsible-use notice go to standard error. Scores are JSON lines.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from uie.api import NOTICE, load, load_blocklist


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="uie", description="Limnex URL Intelligence Engine. " + NOTICE)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--weights", required=True, help="the published checkpoint (limnex-uie-e16_b5.pt)")
        p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
        p.add_argument("--no-verify", action="store_true",
                       help="skip the published-file checks (only for a checkpoint or filter of your own)")

    g = sub.add_parser("generate", help="generate synthetic URLs (gated)")
    common(g)
    g.add_argument("--label", required=True, choices=["benign", "phishing"])
    g.add_argument("-n", type=int, default=10, help="URLs to draw before the gate (default 10)")
    g.add_argument("--seed", type=int, default=None, help="for a reproducible draw")
    g.add_argument("--known-urls", default=None,
                   help="the published Bloom filter of training URLs (limnex-uie-known-urls.bloom.gz), "
                        "which removes copies of real URLs; required")
    g.add_argument("--blocklist", default=None,
                   help="a current threat-feed snapshot: one domain or URL per line")

    s = sub.add_parser("score", help="log P(url | class) and their difference, as JSON lines")
    common(s)
    s.add_argument("urls", nargs="*", help="URLs to score")
    s.add_argument("--file", default=None, help="a file with one URL per line")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        return _run(args)
    except ValueError as exc:            # a refused request: an unpublished file, phishing without the filter
        print(f"uie: {exc}", file=sys.stderr)
        return 2


def _run(args) -> int:
    if args.command == "generate":
        print(NOTICE, file=sys.stderr)
        engine = load(args.weights, device=args.device, verify=not args.no_verify, known_urls=args.known_urls)
        blocked = load_blocklist(args.blocklist) if args.blocklist else None
        urls = engine.generate(args.label, n=args.n, seed=args.seed, blocked_domains=blocked)
        for u in urls:
            print(u)
        print(engine.report_text(), file=sys.stderr)
        if not blocked:
            print("  -> pass --blocklist with a current threat-feed snapshot before sharing any generated URL",
                  file=sys.stderr)
        return 0
    urls = list(args.urls)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            urls += [line.strip() for line in fh if line.strip()]
    if not urls:
        print("nothing to score: give URLs or --file", file=sys.stderr)
        return 2
    engine = load(args.weights, device=args.device, verify=not args.no_verify)
    for row in engine.score(urls):
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
