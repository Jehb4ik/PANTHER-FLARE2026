"""Download the FLARE 2026 Task 1 validation subtree from FLARE-MedFM/PancancerCTSeg
(public 50 + labels, hidden 100, 172 healthy; ~20-25 GB).

Usage: python download_validation.py --root /some/path [--subset public|hidden|healthy]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="destination root")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subset", choices=["all", "public", "hidden", "healthy"],
                    default="all",
                    help="which validation subset to fetch (default: all)")
    args = ap.parse_args()

    if args.subset == "all":
        patterns = ["validation/**"]
    elif args.subset == "public":
        patterns = ["validation/Validation-Public-Images/**",
                    "validation/Validation-Public-Labels/**"]
    elif args.subset == "hidden":
        patterns = ["validation/Validation-Hidden-Images/**"]
    else:
        patterns = ["validation/HealthyImages-noLesion/**"]

    args.root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="FLARE-MedFM/PancancerCTSeg",
        repo_type="dataset",
        local_dir=str(args.root),
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    print(f"\nvalidation subset '{args.subset}' downloaded into {args.root}/validation/")


if __name__ == "__main__":
    main()
