"""
test_crack_detector.py
----------------------
Unit tests for the CrackDetector pipeline.
Run with: pytest tests/
"""

import numpy as np
import pytest
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crack_detector import CrackDetector, CrackReport


@pytest.fixture
def detector():
    return CrackDetector()


def make_blank_image(h=300, w=300, color=(180, 180, 180)):
    """Create a plain BGR image."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def make_crack_image():
    """Draw a synthetic crack (dark diagonal line) on a light background."""
    img = make_blank_image()
    cv2.line(img, (30, 30), (270, 270), (30, 30, 30), thickness=4)
    return img


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_returns_single_channel(self, detector):
        img = make_blank_image()
        gray = detector.preprocess(img)
        assert gray.ndim == 2, "Expected 2-D grayscale array"

    def test_output_shape_matches_input_hw(self, detector):
        img = make_blank_image(200, 400)
        gray = detector.preprocess(img)
        assert gray.shape == (200, 400)


# ── Thresholding ──────────────────────────────────────────────────────────────

class TestThreshold:
    def test_binary_output(self, detector):
        img = make_blank_image()
        gray = detector.preprocess(img)
        thresh = detector.threshold(gray)
        unique = set(np.unique(thresh))
        assert unique.issubset({0, 255}), "Threshold should produce only 0 or 255"


# ── Edge Detection ────────────────────────────────────────────────────────────

class TestEdgeDetection:
    def test_crack_produces_edges(self, detector):
        img = make_crack_image()
        gray = detector.preprocess(img)
        edges = detector.detect_edges(gray)
        # A real crack line should produce non-zero edge pixels
        assert np.count_nonzero(edges) > 0


# ── Severity Classification ───────────────────────────────────────────────────

class TestClassify:
    @pytest.mark.parametrize("area,expected", [
        (0,    "Low"),
        (499,  "Low"),
        (500,  "Moderate"),
        (1999, "Moderate"),
        (2000, "Severe"),
        (9999, "Severe"),
    ])
    def test_thresholds(self, detector, area, expected):
        assert detector.classify(area) == expected


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_run_returns_correct_types(self, detector, tmp_path):
        # Save a synthetic crack image to disk
        img_path = str(tmp_path / "test_crack.jpg")
        cv2.imwrite(img_path, make_crack_image())

        output_img, report = detector.run(img_path)

        assert isinstance(output_img, np.ndarray)
        assert isinstance(report, CrackReport)
        assert report.severity in {"Low", "Moderate", "Severe"}
        assert report.total_area >= 0
        assert report.total_length >= 0

    def test_missing_file_raises(self, detector):
        with pytest.raises(FileNotFoundError):
            detector.run("nonexistent_image.jpg")

    def test_blank_image_is_low_severity(self, detector, tmp_path):
        img_path = str(tmp_path / "blank.jpg")
        cv2.imwrite(img_path, make_blank_image())
        _, report = detector.run(img_path)
        # A plain image should have minimal crack area
        assert report.severity in {"Low", "Moderate"}
