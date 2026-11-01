# -*- coding: utf-8 -*-
"""contact — تركيب لوحات تحقق (2×2) من PNGs لمراجعة بصرية سريعة."""
from pathlib import Path
from PIL import Image

def sheet(files, out, scale=0.62):
    imgs = [Image.open(p) for p in files]
    w = max(i.width for i in imgs) // 2
    h = max(i.height for i in imgs) // 2
    small = []
    for im in imgs:
        im2 = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        small.append(im2)
    pad = 12
    W = w * 2 + pad * 3
    H = h * 2 + pad * 3
    canvas = Image.new("RGB", (W, H), "#DDDDDD")
    for k, im in enumerate(small):
        x = pad + (k % 2) * (w + pad)
        y = pad + (k // 2) * (h + pad)
        canvas.paste(im, (x, y))
    canvas.save(out)
    print("sheet", out, "cells:", len(small))

if __name__ == "__main__":
    png = Path("figures/png")
    allp = sorted(png.glob("f*.png"))
    outdir = Path("build/qa")
    outdir.mkdir(parents=True, exist_ok=True)
    for g in range(0, len(allp), 4):
        sheet(allp[g:g + 4], outdir / f"sheet{g//4 + 1:02d}.png")
