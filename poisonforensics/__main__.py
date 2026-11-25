"""Command line entry point:  python -m poisonforensics <step> [options]"""
from __future__ import annotations

import argparse
import sys

from .config import Config, load_env


def main() -> int:
    p = argparse.ArgumentParser(prog="poisonforensics", description=__doc__)
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="step", required=True)
    sub.add_parser("collect", help="download Transfer logs from the RPC node")
    b = sub.add_parser("build", help="match lookalikes, label, compute features")
    b.add_argument("--reuse", action="store_true", help="keep the matches from the last build; only redo labels and incidents")
    sub.add_parser("train", help="train, evaluate, calibrate, explain")
    a = sub.add_parser("attribute", help="cluster lookalikes into campaigns")
    a.add_argument("--enrich", action="store_true", help="link by submitting account and contract (RPC calls)")
    a.add_argument("--funders", action="store_true", help="also link by first funder (Etherscan key)")
    r = sub.add_parser("report", help="write evidence reports")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--victim")
    r.add_argument("--lookalike")
    v = sub.add_parser("verify", help="re-check a report against an independent node")
    v.add_argument("report")
    v.add_argument("--rpc")
    v.add_argument("--chunks", action="store_true", help="also recompute the cited chunk hashes")
    sub.add_parser("summarize", help="write results/paper_numbers.md")
    sub.add_parser("synth", help="generate synthetic data for an offline test")
    args = p.parse_args()

    load_env()
    cfg = Config.load(args.config)
    if args.step == "collect":
        from . import collect; collect.run(cfg)
    elif args.step == "build":
        from . import dataset; dataset.run(cfg, reuse=args.reuse)
    elif args.step == "train":
        from . import model; model.run(cfg)
    elif args.step == "attribute":
        from . import attribute; attribute.run(cfg, enrich=args.enrich or args.funders, funders=args.funders)
    elif args.step == "report":
        from . import report; report.run(cfg, top=args.top, victim=args.victim, lookalike=args.lookalike)
    elif args.step == "verify":
        from . import report; return 0 if report.verify(cfg, args.report, args.rpc, args.chunks) else 1
    elif args.step == "summarize":
        from . import summarize; summarize.run(cfg)
    elif args.step == "synth":
        from . import synth; synth.run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())