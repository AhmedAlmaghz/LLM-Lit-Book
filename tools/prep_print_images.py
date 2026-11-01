# -*- coding: utf-8 -*-
"""prep_print_images.py — نسخة خفيفة للمخططات (للطباعة/PDF/EPUB): أقصى عرض 1560px."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dist" / "images"
DST = ROOT / "dist" / "images_print"
MAXW = 1560


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for p in sorted(SRC.glob("*.png")):
        img = Image.open(p)
        if img.width > MAXW:
            nh = int(img.height * MAXW / img.width)
            img = img.resize((MAXW, nh), Image.LANCZOS)
        img.save(DST / p.name, "PNG", optimize=True)
        print(p.name, "->", (DST / p.name).stat().st_size)


if __name__ == "__main__":
    main()
