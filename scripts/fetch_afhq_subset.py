"""Fetch a small AFHQ subset via the HuggingFace Datasets Server REST API.

The dataset `huggan/AFHQ` exposes per-row image URLs through
`https://datasets-server.huggingface.co/rows`, so we can pull a handful of
samples per class without parquet/datasets dependencies.

Usage
-----
    python scripts/fetch_afhq_subset.py \
        --output data/afhq \
        --per-class 60 \
        --max-rows 4500

The script walks the `train` split in pages of 100 rows, and stops when each
of the three classes (`cat`, `dog`, `wild`) has reached the per-class cap.
Each downloaded JPEG is saved as
`<output>/<class_name>/<row_idx:06d>.jpg`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "huggan/AFHQ"
CONFIG = "default"
SPLIT = "train"
LABELS = ["cat", "dog", "wild"]
PAGE_SIZE = 100  # max length per /rows call
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET}&config={CONFIG}&split={SPLIT}"
)
USER_AGENT = "csdldptv2-seeder/1.0 (+https://example.local)"


def _http_get_json(url: str, *, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url: str, *, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_page(offset: int, length: int) -> list[dict]:
    url = f"{ROWS_URL}&offset={offset}&length={length}"
    payload = _http_get_json(url)
    return payload.get("rows", [])


def fetch_subset(
    output: Path, per_class: int, max_rows: int, polite_delay: float
) -> dict[str, int]:
    """Walk the dataset rows, downloading at most `per_class` per label."""
    counts = {name: 0 for name in LABELS}
    for name in LABELS:
        (output / name).mkdir(parents=True, exist_ok=True)

    offset = 0
    seen = 0
    while seen < max_rows and any(counts[c] < per_class for c in LABELS):
        try:
            rows = _fetch_page(offset, PAGE_SIZE)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"[fetch] page offset={offset} failed: {exc}", file=sys.stderr)
            time.sleep(2.0)
            offset += PAGE_SIZE
            continue
        if not rows:
            break

        for entry in rows:
            seen += 1
            row = entry.get("row", {})
            label_idx = row.get("label")
            if not isinstance(label_idx, int) or not 0 <= label_idx < len(LABELS):
                continue
            label = LABELS[label_idx]
            if counts[label] >= per_class:
                continue

            image_meta = row.get("image") or {}
            src = image_meta.get("src")
            if not src:
                continue

            filename = output / label / f"{entry.get('row_idx', seen):06d}.jpg"
            if filename.exists() and filename.stat().st_size > 0:
                counts[label] += 1
                continue

            try:
                blob = _http_get_bytes(src)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"[fetch] image {src[:80]}... failed: {exc}", file=sys.stderr)
                continue
            filename.write_bytes(blob)
            counts[label] += 1
            print(
                f"[fetch] {label:<5} row={entry['row_idx']:>5} "
                f"{counts[label]:>3}/{per_class} -> {filename.relative_to(output.parent)}"
            )
            if polite_delay > 0:
                time.sleep(polite_delay)

        offset += PAGE_SIZE

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/afhq"),
        help="Destination root. Per-class subdirs are created here.",
    )
    parser.add_argument(
        "--per-class", type=int, default=60,
        help="Number of images to keep per class (cat/dog/wild).",
    )
    parser.add_argument(
        "--max-rows", type=int, default=15000,
        help="Hard cap on rows scanned across all classes.",
    )
    parser.add_argument(
        "--polite-delay", type=float, default=0.0,
        help="Seconds to sleep between image downloads (politeness throttle).",
    )
    args = parser.parse_args(argv)

    output = args.output.resolve()
    print(f"[fetch] writing to {output}")
    counts = fetch_subset(
        output=output,
        per_class=args.per_class,
        max_rows=args.max_rows,
        polite_delay=args.polite_delay,
    )

    print("\n[fetch] done. counts per class:")
    for label, count in counts.items():
        print(f"  {label:<5} {count}")
    return 0 if all(c > 0 for c in counts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
