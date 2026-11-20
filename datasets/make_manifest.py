"""Match image/mask files by stem and create a validated JSONL manifest."""

import argparse
import json
import os
from pathlib import Path

from datasets.manifest import read_manifest


SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def index_images(directory: str) -> dict[str, Path]:
    folder = Path(directory).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {folder}")
            result[path.stem] = path
    if not result:
        raise ValueError(f"No supported images in {folder}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Create a source training manifest")
    parser.add_argument("--task", choices=("vessel", "odoc"), required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", help="Vessel mask directory")
    parser.add_argument("--disc-masks", help="Optic disc mask directory")
    parser.add_argument("--cup-masks", help="Optic cup mask directory")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.task == "vessel" and not args.masks:
        parser.error("--task vessel requires --masks")
    if args.task == "odoc" and (not args.disc_masks or not args.cup_masks):
        parser.error("--task odoc requires --disc-masks and --cup-masks")

    images = index_images(args.images)
    mask_folders = {"mask": args.masks} if args.task == "vessel" else {
        "disc_mask": args.disc_masks, "cup_mask": args.cup_masks}
    masks = {key: index_images(folder) for key, folder in mask_folders.items()}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for stem, image in images.items():
        record = {"image": os.path.relpath(image, output.parent), "domain": args.domain}
        for key, indexed in masks.items():
            if stem not in indexed:
                raise ValueError(f"No {key} for image stem {stem!r}")
            record[key] = os.path.relpath(indexed[stem], output.parent)
        lines.append(json.dumps(record, ensure_ascii=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    read_manifest(str(output), args.task, require_labels=True)
    print(f"Wrote {len(lines)} samples to {output}")


if __name__ == "__main__":
    main()
