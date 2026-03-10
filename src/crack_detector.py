"""
crack_detector.py
-----------------
Smart Infrastructure – Crack Detection System
Uses OpenCV-based computer vision to detect, measure, and classify
structural cracks in aerial images of roads and bridges.

Authors:
    Aditya Francis Masih  (2361002)
    Amal Tom Ajith        (2361004)
    Anshika Bansal        (2361005)
    Aravindhan K          (2361006)

Project Batch: 32
"""

import cv2
import numpy as np
import argparse
import os
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Data class to hold detection results
# ─────────────────────────────────────────────

@dataclass
class CrackReport:
    total_area: float
    total_length: float
    contour_count: int
    severity: str
    severity_color: tuple  # BGR


# ─────────────────────────────────────────────
# Core detection pipeline
# ─────────────────────────────────────────────

class CrackDetector:
    """
    End-to-end crack detection pipeline.

    Steps:
        1. Grayscale conversion
        2. Gaussian blur (noise reduction)
        3. Adaptive thresholding
        4. Canny edge detection
        5. Morphological dilation
        6. Contour detection & measurement
        7. Severity classification
    """

    # Severity thresholds (area in pixels²)
    SEVERITY_THRESHOLDS = {
        "Low":      (0,    500),
        "Moderate": (500,  2000),
        "Severe":   (2000, float("inf")),
    }

    SEVERITY_COLORS = {
        "Low":      (0, 255, 0),    # Green
        "Moderate": (0, 165, 255),  # Orange
        "Severe":   (0, 0, 255),    # Red
    }

    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        dilate_iterations: int = 1,
        min_contour_area: float = 50.0,
    ):
        """
        Args:
            blur_kernel:        Gaussian blur kernel size (must be odd).
            canny_low:          Lower threshold for Canny edge detection.
            canny_high:         Upper threshold for Canny edge detection.
            dilate_iterations:  Number of dilation iterations for gap-closing.
            min_contour_area:   Minimum contour area to consider (filters noise).
        """
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.dilate_iterations = dilate_iterations
        self.min_contour_area = min_contour_area

    # ── Step 1 & 2: Grayscale + Blur ──────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian blur to reduce noise."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return blurred

    # ── Step 3: Adaptive Thresholding ─────────────────────────────────────────

    def threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Use adaptive (Gaussian) thresholding so the system handles
        uneven lighting across aerial images gracefully.
        """
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )

    # ── Step 4: Edge Detection ─────────────────────────────────────────────────

    def detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to highlight crack boundaries."""
        return cv2.Canny(gray, self.canny_low, self.canny_high)

    # ── Step 5: Morphological Dilation ────────────────────────────────────────

    def morphological_cleanup(self, edges: np.ndarray) -> np.ndarray:
        """
        Dilate edges to close small gaps in crack lines,
        making contours more continuous and measurable.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(edges, kernel, iterations=self.dilate_iterations)

    # ── Step 6: Contour Detection & Measurement ───────────────────────────────

    def find_cracks(self, processed: np.ndarray):
        """
        Find external contours and filter out noise by minimum area.

        Returns:
            List of contours that pass the area threshold.
        """
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

    def measure(self, contours: list) -> tuple[float, float]:
        """
        Calculate total crack area (px²) and perimeter length (px).

        Returns:
            (total_area, total_length)
        """
        total_area = sum(cv2.contourArea(c) for c in contours)
        total_length = sum(cv2.arcLength(c, closed=True) for c in contours)
        return total_area, total_length

    # ── Step 7: Severity Classification ──────────────────────────────────────

    def classify(self, total_area: float) -> str:
        """Classify crack severity based on total detected area."""
        for label, (low, high) in self.SEVERITY_THRESHOLDS.items():
            if low <= total_area < high:
                return label
        return "Severe"

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def run(self, image_path: str) -> tuple[np.ndarray, CrackReport]:
        """
        Execute the full detection pipeline on a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            (annotated_image, CrackReport)

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError:        If the image cannot be decoded by OpenCV.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")

        # Pipeline
        gray      = self.preprocess(image)
        thresh    = self.threshold(gray)
        edges     = self.detect_edges(gray)
        combined  = cv2.bitwise_or(thresh, edges)   # merge threshold + edges
        dilated   = self.morphological_cleanup(combined)
        contours  = self.find_cracks(dilated)

        total_area, total_length = self.measure(contours)
        severity = self.classify(total_area)

        # Annotate output image
        output = image.copy()
        color  = self.SEVERITY_COLORS[severity]
        cv2.drawContours(output, contours, -1, color, 2)

        label = (
            f"Severity: {severity}  |  "
            f"Area: {total_area:.0f}px²  |  "
            f"Length: {total_length:.0f}px  |  "
            f"Cracks: {len(contours)}"
        )
        cv2.putText(
            output, label, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
        )

        report = CrackReport(
            total_area=total_area,
            total_length=total_length,
            contour_count=len(contours),
            severity=severity,
            severity_color=color,
        )
        return output, report


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Infrastructure – Crack Detection System"
    )
    parser.add_argument("image", help="Path to input aerial image")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save annotated output image (optional)"
    )
    parser.add_argument("--show", action="store_true", help="Display result window")
    parser.add_argument("--canny-low",  type=int, default=50,  help="Canny lower threshold")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny upper threshold")
    parser.add_argument("--min-area",   type=float, default=50.0, help="Min contour area to keep")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = CrackDetector(
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        min_contour_area=args.min_area,
    )

    print(f"[INFO] Processing: {args.image}")
    output_img, report = detector.run(args.image)

    print("\n── Crack Detection Report ──────────────────")
    print(f"  Severity       : {report.severity}")
    print(f"  Total Area     : {report.total_area:.2f} px²")
    print(f"  Total Length   : {report.total_length:.2f} px")
    print(f"  Crack Regions  : {report.contour_count}")
    print("────────────────────────────────────────────\n")

    if args.output:
        cv2.imwrite(args.output, output_img)
        print(f"[INFO] Annotated image saved → {args.output}")

    if args.show:
        cv2.imshow("Crack Detection – Smart Infrastructure", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
