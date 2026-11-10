#!/usr/bin/env python3
"""
package_deliverables.py
Packages all deliverables into CLICK_APPEAL_COMPLETE_OUTPUT.zip.
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = BASE_DIR / "CLICK_APPEAL_COMPLETE_OUTPUT.zip"

INCLUDE_DIRS = [
    "data",
    "thumbnails_original",
    "annotated_thumbnails",
    "features",
    "scores",
    "analysis",
    "scripts",
    "documentation",
    "logs"
]

INCLUDE_FILES = [
    "requirements.txt",
    "README.md",
    "PROGRESS.md",
    "run_pipeline.py"
]


def create_zip():
    print(f"Creating ZIP archive at: {ZIP_PATH}")
    file_count = 0

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in INCLUDE_FILES:
            fp = BASE_DIR / f
            if fp.exists():
                zipf.write(fp, arcname=f"ClickAppeal_New/{f}")
                file_count += 1
                print(f"  + Added file: {f}")

        for d in INCLUDE_DIRS:
            dp = BASE_DIR / d
            if dp.exists():
                dir_files = 0
                for root, _, files in os.walk(dp):
                    for file in files:
                        full_p = Path(root) / file
                        rel_p = full_p.relative_to(BASE_DIR)
                        arcname = f"ClickAppeal_New/{str(rel_p).replace(chr(92), '/')}"
                        zipf.write(full_p, arcname=arcname)
                        file_count += 1
                        dir_files += 1
                print(f"  + Added directory: {d} ({dir_files} files)")

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print("\n================== PACKAGING COMPLETE ==================")
    print(f"  Output Archive : {ZIP_PATH.name}")
    print(f"  Total Files    : {file_count}")
    print(f"  Archive Size   : {size_mb:.2f} MB")
    print("========================================================\n")


if __name__ == "__main__":
    create_zip()
