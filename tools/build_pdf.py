# -*- coding: utf-8 -*-
"""
build_pdf.py — طباعة dist/book_print.html إلى dist/llm-ds.pdf عبر Edge بدون رأس.
الصور نسبية (images/*.png) بجانب الملف — لذا البيئة المحلية كافية.

Edge (Chromium) يكتب ملف PDF ثم يظل معلّقًا بعد انتهاء العمل. لذلك لا ننتظر
خروج العملية: نراقب الملف حتى يستقر حجمه، ثم نقتل شجرة Edge ونكمل.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
SRC = DIST / "book_print.html"
OUT = DIST / "llm-ds.pdf"
LOG = ROOT / "build" / "_edge.log"

PRINT_TIMEOUT = 900      # سقف الطباعة (ثانية)
STABLE_TICKS = 4         # استقرار الحجم 4 ثوانٍ متتالية = اكتمل الملف
MIN_BYTES = 5_000_000    # أقل حجم معقول لكتاب بهذا العدد من الصفحات


def kill_headless():
    """يقتل كل عمليات Edge بوضع headless (بما فيها العمليات اليتيمة)."""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
             "Where-Object { $_.CommandLine -like '*--headless*' } | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
            capture_output=True, timeout=30)
    except Exception:
        pass


def wait_for_file(proc, deadline):
    """ينتظر ظهور OUT واستقرار حجمه. يعيد (ok, waited, reason)."""
    last = -1
    stable = 0
    while time.time() < deadline:
        if OUT.exists():
            size = OUT.stat().st_size
            if size >= MIN_BYTES and size == last:
                stable += 1
                if stable >= STABLE_TICKS:
                    return True, size, "stable"
            else:
                stable = 0
            last = size
        elif proc.poll() is not None:
            return False, 0, f"edge exited rc={proc.returncode} without output"
        time.sleep(1)
    # انتهى السقف الزمني: نقبل الملف إن كان موجودًا وكبيرًا
    if OUT.exists() and OUT.stat().st_size >= MIN_BYTES:
        return True, OUT.stat().st_size, "timeout-with-file"
    return False, 0, "timeout"


def main():
    if not SRC.exists():
        print("missing", SRC)
        sys.exit(1)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    kill_headless()
    OUT.unlink(missing_ok=True)

    args = [
        EDGE, "--headless=new", "--disable-gpu", "--no-first-run",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT.resolve()}",
        SRC.resolve().as_uri(),
    ]
    t0 = time.time()
    with LOG.open("wb") as logf:
        proc = subprocess.Popen(args, stdout=logf, stderr=logf)
        ok, size, reason = wait_for_file(proc, t0 + PRINT_TIMEOUT)

    # Edge يظل معلّقًا بعد الكتابة: ننهي العملية وشجرتها دائمًا
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
    kill_headless()

    dt = time.time() - t0
    print(f"pdf ok={ok} bytes={size:,} t={dt:.1f}s reason={reason}")
    if not ok:
        tail = ""
        try:
            data = LOG.read_bytes()[-800:]
            tail = data.decode("utf-8", "ignore").strip()
        except Exception:
            pass
        if tail:
            print("edge log tail:", tail)
        sys.exit(2)

    import pymupdf
    d = pymupdf.open(OUT)
    pages = d.page_count
    d.close()
    print("pages:", pages, "page size: A4")
    if pages < 100:
        sys.exit(3)

    # الرؤوس/التذييلات + أرقام الفهرس + Bookmarks
    if "--no-post" not in sys.argv:
        import postprocess_pdf
        postprocess_pdf.main()


if __name__ == "__main__":
    main()
