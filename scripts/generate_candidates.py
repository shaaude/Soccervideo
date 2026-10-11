#!/usr/bin/env python3
"""Guide-compatible wrapper for `soccer-vision candidates` (same flags). See `soccer-vision candidates --help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_vision.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["candidates", *sys.argv[1:]])
