# -*- coding: utf-8 -*-
"""qa_page.py — لقطة سريعة لمنطقة فصل معين من dist/book_print.html عبر Edge (جزء #chN)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from svg2png import launch_screenshot

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "ch1"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 4200
    out = ROOT / "build" / f"qa_page_{label}.png"
    url = (DIST / "book_print.html").resolve().as_uri() + "#" + label
    launch_screenshot(url, out, 900, h, scale=1.0)
    print("saved", out)


if __name__ == "__main__":
    main()
