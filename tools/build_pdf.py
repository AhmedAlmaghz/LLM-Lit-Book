# -*- coding: utf-8 -*-
"""
build_pdf.py — طباعة dist/book_print.html إلى dist/llm-ds.pdf عبر Edge بدون رأس.
الصور نسبية (images/*.png) بجانب الملف — لذا البيئة المحلية كافية.
"""
import subprocess, sys, time
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
SRC = DIST / "book_print.html"
OUT = DIST / "llm-ds.pdf"


def kill_headless():
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
             "Where-Object { $_.CommandLine -like '*--headless*' } | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
            capture_output=True, timeout=25)
    except Exception:
        pass


def main():
    if not SRC.exists():
        print("missing", SRC)
        sys.exit(1)
    kill_headless()
    args = [
        EDGE, "--headless", "--disable-gpu", "--no-first-run",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT.resolve()}",
        SRC.resolve().as_uri(),
    ]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, timeout=420)
    dt = time.time() - t0
    ok = OUT.exists() and OUT.stat().st_size > 100000
    size = OUT.stat().st_size if OUT.exists() else 0
    print(f"pdf ok={ok} bytes={size:,} t={dt:.1f}s rc={r.returncode}")
    if r.stderr:
        err = r.stderr.decode("utf-8", "ignore")[-500:]
        print("stderr tail:", err)
    if not ok:
        sys.exit(2)
    import fitz
    d = fitz.open(OUT)
    print("pages:", d.page_count)
    # قياس أول صفحة
    p0 = d[0]
    print("page size:", p0.rect)


if __name__ == "__main__":
    main()
