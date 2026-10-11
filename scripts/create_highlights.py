#!/usr/bin/env python3
"""Guide-compatible wrapper for `soccer-vision highlights` (same flags). See `soccer-vision highlights --help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_vision.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["highlights", *sys.argv[1:]])
