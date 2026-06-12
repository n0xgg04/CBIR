"""Filter an animal image dataset before CBIR ingestion.

The input tree must use the same layout as the seeding script:

    <source>/<animal_type>/<image>

The filter combines:

* Grounding DINO zero-shot detection for animal face/head localisation.
* CLIP positive-vs-negative prompt verification on the detected crop.
* Deterministic OpenCV quality gates for size, blur, exposure and clipping.

The script is non-destructive. It writes a CSV manifest and JSON summary; use
``--materialize`` to additionally create accepted/review/rejected trees using
hard links where possible and copies as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

SUPPORTED_EXTS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
STATUS = Literal["ACCEPT", "REVIEW", "REJECT"]


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str


@dataclass(frozen=True)
class Thresholds:
    detection_min: float = 0.20
    detection_accept: float = 0.32
    clip_reject_margin: float = -0.01
    clip_accept_margin: float = 0.02
    face_min_px: int = 64
    face_reject_ratio: float = 0.06
    face_accept_ratio: float = 0.15
    laplacian_reject: float = 35.0
    laplacian_accept: float = 80.0
    tenengrad_reject: float = 150.0
    tenengrad_accept: float = 400.0
    brightness_min: float = 30.0
    brightness_max: float = 225.0
    dark_ratio_max: float = 0.35
    bright_ratio_max: float = 0.25
    contrast_min: float = 20.0
    border_margin_ratio: float = 0.01
    multiple_face_area_fraction: float = 0.25


@dataclass
class FilterRecord:
    source_path: str
    animal_type: str
    sha256: str = ""
    width: int = 0
    height: int = 0
    status: STATUS = "REJECT"
    reasons: list[str] = field(default_factory=list)
    detection_count: int = 0
    detection_label: str = ""
    detection_confidence: float = 0.0
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    face_width: float = 0.0
    face_height: float = 0.0
    face_area_ratio: float = 0.0
    border_touch_count: int = 0
    clip_margin: float = 0.0
    laplacian_variance: float = 0.0
    tenengrad: float = 0.0
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    dark_ratio: float = 0.0
    bright_ratio: float = 0.0
    elapsed_ms: int = 0


class FaceDetector(Protocol):
    model_id: str

    def detect(self, image: Image.Image, animal_type: str) -> list[Detection]: ...


class CropVerifier(Protocol):
    model_id: str

    def verify(self, crop: Image.Image, animal_type: str) -> float: ...


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _move_inputs(inputs: Any, device: str) -> Any:
    return inputs.to(device)


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str,
        device: str,
        threshold: float,
        text_threshold: float,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "model dependencies missing; run `uv sync --extra filter`"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def detect(self, image: Image.Image, animal_type: str) -> list[Detection]:
        labels = [
            f"{animal_type} face",
            f"{animal_type} head",
            "animal face",
            "animal head",
        ]
        inputs = self.processor(images=image, text=[labels], return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=[labels],
        )[0]
        result_labels = (
            result["text_labels"] if "text_labels" in result else result["labels"]
        )
        detections = [
            Detection(
                box=tuple(float(value) for value in box.tolist()),
                score=float(score.item()),
                label=str(label),
            )
            for box, score, label in zip(
                result["boxes"], result["scores"], result_labels, strict=True
            )
        ]
        return non_max_suppression(detections)


class ClipCropVerifier:
    POSITIVE_TEMPLATES: Final[tuple[str, ...]] = (
        "a clear close-up photo of a {animal_type} face",
        "a clearly visible {animal_type} head",
        "an animal face with visible eyes and nose",
    )
    NEGATIVE_TEMPLATES: Final[tuple[str, ...]] = (
        "a full-body animal photographed from far away",
        "the back of an animal with no visible face",
        "a severely blurred or hidden animal face",
        "a landscape or background without an animal face",
    )

    def __init__(self, model_id: str, device: str) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "model dependencies missing; run `uv sync --extra filter`"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self._text_cache: dict[str, tuple[Any, int]] = {}

    def _text_features(self, animal_type: str) -> tuple[Any, int]:
        cached = self._text_cache.get(animal_type)
        if cached is not None:
            return cached
        positives = [
            template.format(animal_type=animal_type)
            for template in self.POSITIVE_TEMPLATES
        ]
        negatives = [
            template.format(animal_type=animal_type)
            for template in self.NEGATIVE_TEMPLATES
        ]
        prompts = positives + negatives
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = _move_inputs(inputs, self.device)
        with self._torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        cached = (features, len(positives))
        self._text_cache[animal_type] = cached
        return cached

    def verify(self, crop: Image.Image, animal_type: str) -> float:
        text_features, positive_count = self._text_features(animal_type)
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with self._torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)
        positive = similarities[:positive_count].max()
        negative = similarities[positive_count:].max()
        return float((positive - negative).item())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def box_overlap_over_smaller(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(box_area(left), box_area(right))
    return intersection / smaller if smaller > 0 else 0.0


def non_max_suppression(
    detections: Sequence[Detection],
    iou_threshold: float = 0.55,
    containment_threshold: float = 0.80,
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        duplicate = any(
            box_iou(detection.box, existing.box) >= iou_threshold
            or box_overlap_over_smaller(detection.box, existing.box)
            >= containment_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def clamp_box(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def expanded_crop(
    image: Image.Image, box: Sequence[float], expansion: float = 0.08
) -> Image.Image:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    expanded = clamp_box(
        (
            x1 - width * expansion,
            y1 - height * expansion,
            x2 + width * expansion,
            y2 + height * expansion,
        ),
        image.width,
        image.height,
    )
    return image.crop(tuple(int(round(value)) for value in expanded))


def quality_metrics(crop: Image.Image) -> dict[str, float]:
    rgb = np.asarray(crop.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    return {
        "laplacian_variance": laplacian,
        "tenengrad": tenengrad,
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "dark_ratio": float(np.mean(gray < 15)),
        "bright_ratio": float(np.mean(gray > 245)),
    }


def border_touch_count(
    box: Sequence[float], width: int, height: int, margin_ratio: float
) -> int:
    x1, y1, x2, y2 = box
    return sum(
        (
            x1 <= width * margin_ratio,
            y1 <= height * margin_ratio,
            x2 >= width * (1.0 - margin_ratio),
            y2 >= height * (1.0 - margin_ratio),
        )
    )


def _append_unique(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def classify_record(
    record: FilterRecord,
    detections: Sequence[Detection],
    thresholds: Thresholds,
) -> tuple[STATUS, list[str]]:
    reject: list[str] = []
    review: list[str] = []

    if record.face_width < thresholds.face_min_px or record.face_height < thresholds.face_min_px:
        reject.append("FACE_TOO_SMALL")
    if record.face_area_ratio < thresholds.face_reject_ratio:
        reject.append("FACE_AREA_TOO_SMALL")
    elif record.face_area_ratio < thresholds.face_accept_ratio:
        review.append("FACE_AREA_BORDERLINE")

    if record.detection_confidence < thresholds.detection_min:
        reject.append("LOW_DETECTION_CONFIDENCE")
    elif record.detection_confidence < thresholds.detection_accept:
        review.append("DETECTION_UNCERTAIN")

    if record.clip_margin < thresholds.clip_reject_margin:
        reject.append("CLIP_REJECTED_FACE")
    elif record.clip_margin < thresholds.clip_accept_margin:
        review.append("CLIP_UNCERTAIN")

    if record.laplacian_variance < thresholds.laplacian_reject:
        reject.append("TOO_BLURRY")
    elif (
        record.laplacian_variance < thresholds.laplacian_accept
        or record.tenengrad < thresholds.tenengrad_accept
    ):
        review.append("SHARPNESS_BORDERLINE")
    if record.tenengrad < thresholds.tenengrad_reject:
        _append_unique(reject, "TOO_BLURRY")

    if (
        record.brightness_mean < thresholds.brightness_min
        or record.dark_ratio > thresholds.dark_ratio_max
    ):
        reject.append("TOO_DARK")
    if (
        record.brightness_mean > thresholds.brightness_max
        or record.bright_ratio > thresholds.bright_ratio_max
    ):
        reject.append("OVEREXPOSED")
    if record.contrast_std < thresholds.contrast_min:
        reject.append("LOW_CONTRAST")

    # A face filling almost the entire frame is a normal close-up. Border
    # contact is only suspicious when substantial context remains in the crop.
    if record.face_area_ratio < 0.85:
        if record.border_touch_count >= 2:
            review.append("FACE_TOUCHES_MULTIPLE_BORDERS")
        elif record.border_touch_count == 1:
            review.append("FACE_TOUCHES_BORDER")

    if detections:
        primary_area = box_area(detections[0].box)
        significant = [
            detection
            for detection in detections[1:]
            if box_area(detection.box)
            >= primary_area * thresholds.multiple_face_area_fraction
        ]
        if significant:
            review.append("MULTIPLE_FACES")

    if reject:
        return "REJECT", reject + review
    if review:
        return "REVIEW", review
    return "ACCEPT", []


def iter_images(source: Path, max_per_class: int | None) -> Iterable[tuple[str, Path]]:
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        count = 0
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            yield class_dir.name, path
            count += 1
            if max_per_class is not None and count >= max_per_class:
                break


def inspect_image(
    path: Path,
    animal_type: str,
    detector: FaceDetector,
    verifier: CropVerifier,
    thresholds: Thresholds,
    known_hashes: set[str],
) -> FilterRecord:
    started = time.perf_counter()
    record = FilterRecord(source_path=str(path), animal_type=animal_type)
    try:
        record.sha256 = sha256_file(path)
        if record.sha256 in known_hashes:
            record.reasons = ["DUPLICATE"]
            return record
        known_hashes.add(record.sha256)

        with Image.open(path) as opened:
            image = opened.convert("RGB")
        record.width, record.height = image.size
    except (OSError, UnidentifiedImageError, ValueError):
        record.reasons = ["INVALID_IMAGE"]
        return record

    detections = detector.detect(image, animal_type)
    detections = [
        Detection(
            box=clamp_box(item.box, record.width, record.height),
            score=item.score,
            label=item.label,
        )
        for item in detections
        if item.score >= thresholds.detection_min
    ]
    detections = non_max_suppression(detections)
    record.detection_count = len(detections)
    if not detections:
        record.reasons = ["NO_ANIMAL_FACE"]
        return record

    primary = detections[0]
    record.detection_label = primary.label
    record.detection_confidence = primary.score
    (
        record.bbox_x1,
        record.bbox_y1,
        record.bbox_x2,
        record.bbox_y2,
    ) = primary.box
    record.face_width = record.bbox_x2 - record.bbox_x1
    record.face_height = record.bbox_y2 - record.bbox_y1
    record.face_area_ratio = box_area(primary.box) / (record.width * record.height)
    record.border_touch_count = border_touch_count(
        primary.box,
        record.width,
        record.height,
        thresholds.border_margin_ratio,
    )

    crop = expanded_crop(image, primary.box)
    record.clip_margin = verifier.verify(crop, animal_type)
    for name, value in quality_metrics(crop).items():
        setattr(record, name, value)
    record.status, record.reasons = classify_record(record, detections, thresholds)
    record.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return record


def _record_row(record: FilterRecord) -> dict[str, object]:
    row = asdict(record)
    row["reasons"] = "|".join(record.reasons)
    return row


def write_manifest(path: Path, records: Sequence[FilterRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_record_row(FilterRecord("", "")).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_record_row(record) for record in records)


def materialize_records(
    records: Sequence[FilterRecord], source: Path, output: Path
) -> None:
    status_dirs = {"ACCEPT": "accepted", "REVIEW": "review", "REJECT": "rejected"}
    for record in records:
        src = Path(record.source_path)
        relative = src.relative_to(source)
        destination = output / status_dirs[record.status] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, destination)
        except OSError:
            shutil.copy2(src, destination)


def write_previews(records: Sequence[FilterRecord], source: Path, output: Path) -> None:
    colors = {
        "ACCEPT": (22, 163, 74),
        "REVIEW": (217, 119, 6),
        "REJECT": (220, 38, 38),
    }
    for record in records:
        src = Path(record.source_path)
        relative = src.relative_to(source).with_suffix(".jpg")
        destination = output / "previews" / record.status.lower() / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as opened:
                image = opened.convert("RGB")
        except (OSError, UnidentifiedImageError):
            continue
        draw = ImageDraw.Draw(image)
        color = colors[record.status]
        if record.detection_count > 0:
            draw.rectangle(
                (
                    record.bbox_x1,
                    record.bbox_y1,
                    record.bbox_x2,
                    record.bbox_y2,
                ),
                outline=color,
                width=max(2, round(min(image.size) / 128)),
            )
        reason_text = ",".join(record.reasons) if record.reasons else "OK"
        label = f"{record.status} | {reason_text}"
        text_box = draw.textbbox((0, 0), label)
        text_height = text_box[3] - text_box[1]
        draw.rectangle((0, 0, image.width, text_height + 8), fill=(0, 0, 0))
        draw.text((4, 4), label, fill=color)
        image.save(destination, format="JPEG", quality=90)


def summary_payload(
    records: Sequence[FilterRecord],
    detector: FaceDetector,
    verifier: CropVerifier,
    device: str,
    thresholds: Thresholds,
) -> dict[str, object]:
    statuses = Counter(record.status for record in records)
    reasons = Counter(reason for record in records for reason in record.reasons)
    per_class: dict[str, dict[str, int]] = {}
    for record in records:
        counts = per_class.setdefault(
            record.animal_type, {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0}
        )
        counts[record.status] += 1
    return {
        "total": len(records),
        "statuses": dict(statuses),
        "reasons": dict(reasons.most_common()),
        "per_class": per_class,
        "detector_model": detector.model_id,
        "verifier_model": verifier.model_id,
        "device": device,
        "thresholds": asdict(thresholds),
        "mean_elapsed_ms": (
            float(np.mean([record.elapsed_ms for record in records])) if records else 0.0
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/face_filter"))
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--detector-model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument(
        "--verifier-model",
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--save-previews", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        print(f"source directory not found: {source}", file=sys.stderr)
        return 2
    if args.max_per_class is not None and args.max_per_class <= 0:
        print("--max-per-class must be positive", file=sys.stderr)
        return 2

    device = choose_device(args.device)
    thresholds = Thresholds(detection_min=args.detection_threshold)
    print(f"[filter] device={device}")
    print(f"[filter] loading detector={args.detector_model}")
    detector = GroundingDinoDetector(
        args.detector_model,
        device,
        args.detection_threshold,
        args.text_threshold,
    )
    print(f"[filter] loading verifier={args.verifier_model}")
    verifier = ClipCropVerifier(args.verifier_model, device)

    records: list[FilterRecord] = []
    known_hashes: set[str] = set()
    for index, (animal_type, path) in enumerate(
        iter_images(source, args.max_per_class), start=1
    ):
        try:
            record = inspect_image(
                path,
                animal_type,
                detector,
                verifier,
                thresholds,
                known_hashes,
            )
        except Exception as exc:
            record = FilterRecord(
                source_path=str(path),
                animal_type=animal_type,
                status="REVIEW",
                reasons=[f"MODEL_ERROR:{type(exc).__name__}"],
            )
            print(f"[filter] model error for {path}: {exc}", file=sys.stderr)
        records.append(record)
        print(
            f"[filter] {index:>4} {record.status:<6} "
            f"{animal_type:<10} {path.name} "
            f"{','.join(record.reasons) or 'OK'}"
        )

    write_manifest(output / "manifest.csv", records)
    summary = summary_payload(records, detector, verifier, device, thresholds)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.materialize:
        materialize_records(records, source, output)
    if args.save_previews:
        write_previews(records, source, output)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
