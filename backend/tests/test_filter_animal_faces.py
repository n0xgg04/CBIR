from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.filter_animal_faces import (
    Detection,
    FilterRecord,
    Thresholds,
    box_iou,
    box_overlap_over_smaller,
    classify_record,
    non_max_suppression,
    quality_metrics,
    write_previews,
)


def _good_record() -> FilterRecord:
    return FilterRecord(
        source_path="cat/example.jpg",
        animal_type="cat",
        width=512,
        height=512,
        detection_count=1,
        detection_confidence=0.8,
        face_width=256,
        face_height=256,
        face_area_ratio=0.25,
        clip_margin=0.1,
        laplacian_variance=150.0,
        tenengrad=800.0,
        brightness_mean=120.0,
        contrast_std=45.0,
        dark_ratio=0.01,
        bright_ratio=0.01,
    )


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_overlap_over_smaller_detects_nested_boxes() -> None:
    assert box_overlap_over_smaller((0, 0, 100, 100), (20, 20, 80, 80)) == 1.0


def test_nms_removes_duplicate_prompt_boxes() -> None:
    detections = [
        Detection((0, 0, 100, 100), 0.9, "cat face"),
        Detection((2, 2, 98, 98), 0.8, "animal face"),
        Detection((150, 150, 220, 220), 0.7, "animal face"),
    ]
    result = non_max_suppression(detections)
    assert len(result) == 2
    assert result[0].label == "cat face"


def test_nms_removes_nested_face_and_head_boxes() -> None:
    detections = [
        Detection((0, 0, 100, 100), 0.9, "animal head"),
        Detection((20, 20, 80, 80), 0.8, "animal face"),
    ]
    assert len(non_max_suppression(detections)) == 1


def test_good_record_is_accepted() -> None:
    record = _good_record()
    status, reasons = classify_record(
        record,
        [Detection((100, 100, 356, 356), 0.8, "cat face")],
        Thresholds(),
    )
    assert status == "ACCEPT"
    assert reasons == []


def test_blurry_record_is_rejected() -> None:
    record = _good_record()
    record.laplacian_variance = 10.0
    record.tenengrad = 20.0
    status, reasons = classify_record(
        record,
        [Detection((100, 100, 356, 356), 0.8, "cat face")],
        Thresholds(),
    )
    assert status == "REJECT"
    assert "TOO_BLURRY" in reasons


def test_uncertain_detection_is_reviewed() -> None:
    record = _good_record()
    record.detection_confidence = 0.25
    status, reasons = classify_record(
        record,
        [Detection((100, 100, 356, 356), 0.25, "cat face")],
        Thresholds(),
    )
    assert status == "REVIEW"
    assert "DETECTION_UNCERTAIN" in reasons


def test_full_frame_closeup_is_not_reviewed_only_for_border_contact() -> None:
    record = _good_record()
    record.face_area_ratio = 0.9
    record.border_touch_count = 4
    status, reasons = classify_record(
        record,
        [Detection((0, 0, 512, 512), 0.8, "cat face")],
        Thresholds(),
    )
    assert status == "ACCEPT"
    assert "FACE_TOUCHES_MULTIPLE_BORDERS" not in reasons


def test_quality_metrics_distinguish_flat_and_textured_images() -> None:
    flat = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
    yy, xx = np.indices((256, 256))
    checker = ((yy // 16) + (xx // 16)) % 2
    textured = Image.fromarray(
        np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
    )
    flat_metrics = quality_metrics(flat)
    textured_metrics = quality_metrics(textured)
    assert textured_metrics["laplacian_variance"] > flat_metrics["laplacian_variance"]
    assert textured_metrics["tenengrad"] > flat_metrics["tenengrad"]


def test_write_previews_draws_output(tmp_path) -> None:
    source = tmp_path / "source"
    image_path = source / "cat" / "cat.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8)).save(image_path)
    record = _good_record()
    record.source_path = str(image_path)
    record.width = 128
    record.height = 128
    record.status = "ACCEPT"
    record.detection_count = 1
    record.bbox_x1 = 16
    record.bbox_y1 = 16
    record.bbox_x2 = 112
    record.bbox_y2 = 112

    output = tmp_path / "output"
    write_previews([record], source, output)

    assert (output / "previews" / "accept" / "cat" / "cat.jpg").is_file()
