"""Validate data manifests and summarize domains before starting a long run."""

import argparse
from collections import Counter
from pathlib import Path

from datasets.manifest import read_manifest


def main():
    parser = argparse.ArgumentParser(description="Check data and SAM checkpoint paths")
    parser.add_argument("--task", choices=("vessel", "odoc"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--labeled", action="store_true")
    args = parser.parse_args()
    records = read_manifest(args.manifest, args.task, require_labels=args.labeled)
    if args.sam_checkpoint and not Path(args.sam_checkpoint).is_file():
        raise FileNotFoundError(args.sam_checkpoint)
    print(f"samples: {len(records)}")
    for domain, count in sorted(Counter(item.get("domain", "unspecified") for item in records).items()):
        print(f"  {domain}: {count}")
    print(f"paired source images: {sum('source_image' in item for item in records)}")
    print("preflight: OK")


if __name__ == "__main__":
    main()
