"""Extract a balanced AFHQ subset from a downloaded parquet shard.

Reads `<source>/data/train-00000-of-00002.parquet` (downloaded via
`huggingface-cli download huggan/AFHQ --repo-type dataset`) and writes JPEG
files into `<output>/<class_name>/<row_idx:06d>.jpg` for each label in
{cat, dog, wild}, capped at `--per-class` images per class.

The script uses pyarrow streaming row-by-row so peak memory stays bounded.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

LABELS = ["cat", "dog", "wild"]


def extract(sources: list[Path], output: Path, per_class: int) -> dict[str, int]:
    import pyarrow.parquet as pq
    from PIL import Image as PILImage

    counts = {name: 0 for name in LABELS}
    for name in LABELS:
        (output / name).mkdir(parents=True, exist_ok=True)

    global_row_idx = 0
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if all(counts[c] >= per_class for c in LABELS):
            break
        pf = pq.ParquetFile(source)
        print(
            f"[extract] {source.name} fields={[f.name for f in pf.schema_arrow]} "
            f"row_groups={pf.num_row_groups} rows={pf.metadata.num_rows}"
        )

        for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
            if all(counts[c] >= per_class for c in LABELS):
                break
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for img_struct, label_idx in zip(images, labels, strict=False):
                global_row_idx += 1
                if not isinstance(label_idx, int) or not 0 <= label_idx < len(LABELS):
                    continue
                label = LABELS[label_idx]
                if counts[label] >= per_class:
                    continue
                payload = img_struct.get("bytes")
                if not payload:
                    continue
                try:
                    img = PILImage.open(io.BytesIO(payload))
                    img.verify()
                except (OSError, ValueError) as exc:
                    print(
                        f"[extract] skip bad row {global_row_idx}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                target = output / label / f"{global_row_idx:06d}.jpg"
                target.write_bytes(payload)
                counts[label] += 1
                if sum(counts.values()) % 25 == 0:
                    print(f"[extract] progress {counts}")

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, action="append", default=None,
        help="Parquet shard path (repeatable). Defaults to both AFHQ shards.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/afhq"),
        help="Destination root; per-class subdirs will be created here.",
    )
    parser.add_argument(
        "--per-class", type=int, default=80,
        help="Images per class.",
    )
    args = parser.parse_args(argv)

    if args.source is None:
        args.source = [
            Path("/tmp/afhq_pq/data/train-00000-of-00002.parquet"),
            Path("/tmp/afhq_pq/data/train-00001-of-00002.parquet"),
        ]

    sources = [s.resolve() for s in args.source]
    counts = extract(sources, args.output.resolve(), args.per_class)
    print("\n[extract] final counts:")
    for label, count in counts.items():
        print(f"  {label:<5} {count}")
    return 0 if all(c > 0 for c in counts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
