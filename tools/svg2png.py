# -*- coding: utf-8 -*-
"""svg2png — تحويل مخططات SVG إلى PNG عالية الدقة عبر Edge بدون رأس.

الإصدار المركّب: صفحة HTML واحدة تضم دفعة من المخططات فوق بعضها، لقطة واحدة،
ثم تقطيع قطعًا دقيقًا بلقطات PIL حسب ارتفاع كل مخطط — يقلب 33 إطلاقًا إلى 5.
"""
import re, subprocess, tempfile, shutil
from pathlib import Path
from PIL import Image

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROFILE = r"C:\Users\AbuEmad\AppData\Local\Temp\opencode\edge_shared"


def vb_size(svg_text):
    m = re.search(r'viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)', svg_text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def launch_screenshot(html_path, png_path, w, h, scale=2.0, timeout=180, profile=None):
    ephemeral = None
    if profile is None:
        import tempfile
        ephemeral = tempfile.mkdtemp(prefix="edge_prof_")
        profile = ephemeral
    url = html_path.as_uri() if isinstance(html_path, Path) else str(html_path)
    args = [
        EDGE, "--headless", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
        f"--user-data-dir={profile}",
        f"--screenshot={png_path.resolve()}",
        f"--window-size={int(w)},{int(h)}",
        f"--force-device-scale-factor={scale}",
        "--default-background-color=FFFFFFFF",
        url,
    ]
    r = subprocess.run(args, capture_output=True, timeout=timeout)
    if not Path(png_path).exists():
        raise RuntimeError(r.stderr.decode("utf-8", "ignore")[-400:])
    if ephemeral:
        shutil.rmtree(ephemeral, ignore_errors=True)


def convert_all(src="figures", dst="figures/png", scale=2.0, batch=8, only=None):
    """يحول مجموعات SVG إلى PNG عبر لقطات مركّبة ثم تقطيع."""
    srcp, dstp = Path(src), Path(dst)
    dstp.mkdir(parents=True, exist_ok=True)
    figs = []
    for p in sorted(srcp.glob("*.svg")):
        if only and p.stem not in only:
            continue
        w, h = vb_size(p.read_text(encoding="utf-8"))
        figs.append((p, int(w), int(h)))

    gap = 24
    for k in range(0, len(figs), batch):
        chunk = figs[k:k + batch]
        total = sum(h for _, _, h in chunk) + gap * (len(chunk) - 1)
        parts = []
        y = 0
        rows = []
        for i, (p, w, h) in enumerate(chunk):
            parts.append(f'<img src="file:///{p.resolve().as_posix()}" style="position:absolute;left:0;top:{y}px" width="{w}" height="{h}">')
            rows.append((p, y, h))
            y += h + gap
        html = (f'<!doctype html><meta charset="utf-8">'
                f'<style>html,body{{margin:0;padding:0}}</style>'
                f'<div style="position:relative;width:1040px;height:{total}px">' + "".join(parts) + "</div>")
        tmp_html = dstp / f"_batch_{k//batch}.html"
        tmp_html.write_text(html, encoding="utf-8")
        big = dstp / f"_batch_{k//batch}.png"
        launch_screenshot(tmp_html, big, 1040, total, scale)
        img = Image.open(big)
        for pi, (p, yy, hh) in enumerate(rows):
            crop = img.crop((0, yy * scale, 1040 * scale, (yy + hh) * scale))
            out = dstp / f"{p.stem}.png"
            crop.save(out, "PNG")
            print("png", p.stem)
        tmp_html.unlink(missing_ok=True)
        big.unlink(missing_ok=True)


def convert_one(svg_path: Path, png_path: Path, scale=2.0, timeout=120, profile=PROFILE):
    """لقطة فردية (نادرًا ما تُستخدم — بطيئة إطلاقًا)."""
    w, h = vb_size(svg_path.read_text(encoding="utf-8"))
    html_path = png_path.with_suffix(".tmp.html")
    html_path.write_text(
        (f'<!doctype html><meta charset="utf-8">'
         f'<style>html,body{{margin:0;padding:0}}img{{display:block}}</style>'
         f'<img src="file:///{svg_path.resolve().as_posix()}" width="{w}" height="{h}">'),
        encoding="utf-8")
    args = [
        EDGE, "--headless", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
        f"--user-data-dir={profile}",
        f"--screenshot={png_path.resolve()}",
        f"--window-size={int(w)},{int(h)}",
        f"--force-device-scale-factor={scale}",
        "--default-background-color=FFFFFFFF",
        html_path.resolve().as_uri(),
    ]
    r = subprocess.run(args, capture_output=True, timeout=timeout)
    html_path.unlink(missing_ok=True)
    if not png_path.exists():
        raise RuntimeError(r.stderr.decode("utf-8", "ignore")[-400:])

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "figures"
    dst = sys.argv[2] if len(sys.argv) > 2 else "figures/png"
    convert_all(src, dst)
