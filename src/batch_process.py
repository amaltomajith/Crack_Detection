"""
batch_process.py
----------------
Process an entire folder of aerial images through the crack detector
and export a CSV summary report.

Usage:
    python batch_process.py --input samples/ --output outputs/ --report report.csv
"""

import os
import csv
import argparse
from crack_detector import CrackDetector
import cv2


def run_batch(input_dir: str, output_dir: str, report_path: str):
    os.makedirs(output_dir, exist_ok=True)

    detector = CrackDetector()
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not image_files:
        print(f"[WARN] No supported images found in: {input_dir}")
        return

    rows = []
    for filename in sorted(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"annotated_{filename}")

        try:
            output_img, report = detector.run(input_path)
            cv2.imwrite(output_path, output_img)

            rows.append({
                "filename":      filename,
                "severity":      report.severity,
                "total_area_px": round(report.total_area, 2),
                "total_length_px": round(report.total_length, 2),
                "crack_regions": report.contour_count,
                "output_file":   output_path,
            })
            print(f"  ✓  {filename:40s}  →  {report.severity}")

        except Exception as e:
            print(f"  ✗  {filename}: {e}")
            rows.append({
                "filename": filename,
                "severity": "ERROR",
                "total_area_px": "",
                "total_length_px": "",
                "crack_regions": "",
                "output_file": str(e),
            })

    # Write CSV
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Processed {len(rows)} images.")
    print(f"[INFO] Report saved → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch crack detection processor")
    parser.add_argument("--input",  required=True, help="Input folder of images")
    parser.add_argument("--output", required=True, help="Output folder for annotated images")
    parser.add_argument("--report", default="report.csv", help="CSV report output path")
    args = parser.parse_args()

    run_batch(args.input, args.output, args.report)
