# -*- coding: utf-8 -*-
"""
postprocess_pdf.py — لمسات الطباعة الاحترافية بعد إخراج Edge:
- رأس كل صفحة: اسم الكتاب (يمين) + عنوان الفصل/القسم (يسار) مع خط فاصل
- تذييل: اسم المؤلف (يمين) + رقم الصفحة (يسار) مع خط علوي
- ملء أرقام صفحات الفهرس: نقرأ روابط الفهرس الداخلية (أهداف الفصول) ونرسم الأرقام
  مكان النقاط الموضعية في صفوف الفهرس
- فهرس PDF داخلي (Bookmarks) وبيانات وصفية
- يتخطى الغلاف، ولا يضع رأسًا على صفحات افتتاح الأجزاء/الفصول/الفهرس

ملاحظة تقنية مهمة: محتوى صفحات Edge يترك أحيانًا حالة رسومية غير متوازنة
(q بلا Q)، فيرث النص المضاف تحويل المقياس 0.24 وتظهر الرؤوس بحجم مجهري.
لذلك نستدعي wrap_contents() لكل صفحة قبل أي رسم.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
import pymupdf
import arabic_reshaper
from bidi.algorithm import get_display

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "dist" / "llm-ds.pdf"
TMP = ROOT / "dist" / "_llm-ds-final.pdf"

MUTED = (0.392, 0.455, 0.545)
BRAND = (0.263, 0.220, 0.792)
INDIGO = (0.216, 0.188, 0.639)
LINE = (0.886, 0.910, 0.941)

FONT_R = pymupdf.Font(fontfile=r"C:\Windows\Fonts\tahoma.ttf")
FONT_B = pymupdf.Font(fontfile=r"C:\Windows\Fonts\tahomabd.ttf")

COLORS = {"muted": MUTED, "indigo": INDIGO, "brand": BRAND}
FRONT_LABEL = {"F0": "الهداء", "F1": "المقدمة", "F2": "خطة الكتاب"}
LOG = ROOT / "build" / "_post.log"
_t0 = time.time()


def log(*a):
    line = f"[{time.time() - _t0:7.1f}s] " + " ".join(str(x) for x in a)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def shaped(s):
    return get_display(arabic_reshaper.reshape(s or ""))


def clip(t, n=58):
    t = (t or "").strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


class Text:
    """يجمع نصوص الصفحة حسب اللون ثم يكتبها في مجرى واحد لكل لون."""

    def __init__(self, page):
        self.page = page
        self.writers = {}
        self.used = set()

    def add(self, key, text, x, y, font, size, align="left", maxw=None):
        s = shaped(text)
        if not s:
            return
        if maxw:
            while size > 6.0 and font.text_length(s, fontsize=size) > maxw:
                size -= 0.2
        w = font.text_length(s, fontsize=size)
        if align == "right":
            x -= w
        elif align == "center":
            x -= w / 2
        tw = self.writers.get(key)
        if tw is None:
            tw = self.writers[key] = pymupdf.TextWriter(self.page.rect)
        tw.append((x, y), s, font=font, fontsize=size)
        self.used.add(key)

    def commit(self):
        for key in self.used:
            self.writers[key].write_text(self.page, color=COLORS[key])


def toc_tokens(book):
    tokens = ["F0", "F1", "F2"]
    for pt in book["part_titles"]:
        tokens.append(f"P{pt['num']}")
        for c in book["chapters"]:
            if c["part"] == pt["num"]:
                tokens.append(f"C{c['num']}")
    for i in range(len(book["appendices"])):
        tokens.append(f"X{i}")
    return tokens


def header_label(chapters, appx, token):
    if not token or token in ("I", "T", "Z", "X"):
        return None
    if token.startswith("P"):
        return None
    if token.startswith("F"):
        return FRONT_LABEL.get(token)
    if token.startswith("C"):
        n = int(token[1:])
        return f"الفصل {n} · {clip(chapters.get(n, ''), 52)}"
    if token.startswith("X"):
        i = int(token[1:])
        return f"ملحق · {clip(appx.get(i, ''), 50)}"
    return None


def main():
    if not PDF.exists():
        raise SystemExit("missing dist/llm-ds.pdf")
    TMP.unlink(missing_ok=True)
    LOG.unlink(missing_ok=True)
    book = B.load()
    meta = book["meta"]
    book_title = meta["title"]
    chapters = {c["num"]: c["title"] for c in book["chapters"]}
    appx = {i: a["title"] for i, a in enumerate(book["appendices"])}

    doc = pymupdf.open(PDF)
    total = doc.page_count
    pw = doc[0].rect.width
    log("start pages", total)

    # ---------- 0) موازنة الحالة الرسومية لكل صفحة قبل أي إضافة ----------
    for i in range(total):
        try:
            doc[i].wrap_contents()
        except Exception as e:
            log("wrap failed", i, repr(e))
    log("wrapped")

    # ---------- 1) صفحات الفهرس + أهداف الروابط (بترتيب القراءة) ----------
    toc_pages = []
    for i in range(min(14, total)):
        links = [l for l in doc[i].get_links()
                 if l.get("kind") == 4 and l.get("page") is not None]
        if len(links) >= 5:
            toc_pages.append(i)

    rows = []          # (page, col, y, target, rect)
    for i in toc_pages:
        for l in doc[i].get_links():
            if l.get("kind") != 4 or l.get("page") is None:
                continue
            r = pymupdf.Rect(l["from"])
            col = 0 if r.x0 > 300 else 1
            rows.append((i, col, r.y0, l["page"], r))
    rows.sort(key=lambda t: (t[0], t[1], t[2]))

    # ---------- 2) النقاط الموضعية في صفوف الفهرس ----------
    dots = []          # (page, col, y, origin, bbox)
    for i in toc_pages:
        data = doc[i].get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        for b in data.get("blocks", []):
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    if sp.get("text", "").strip() == "." and 7.4 <= sp.get("size", 0) <= 9.7:
                        r = pymupdf.Rect(sp["bbox"])
                        col = 0 if r.x0 > 250 else 1
                        dots.append((i, col, r.y0, tuple(sp.get("origin", (r.x0, r.y1))), r))
    dots.sort(key=lambda t: (t[0], t[1], t[2]))
    log("toc_pages", toc_pages, "rows", len(rows), "dots", len(dots))

    tokens = toc_tokens(book)
    starts = {}
    if len(rows) >= len(tokens):
        for tok, row in zip(tokens, rows):
            starts[tok] = row[3]
    else:
        raise SystemExit(f"TOC rows missing: rows={len(rows)} tokens={len(tokens)}")
    starts.setdefault("I", 1)
    starts["T"] = toc_pages[0] if toc_pages else 2
    if "X0" in starts:
        starts["X"] = max(0, starts["X0"] - 1)
    starts["Z"] = total - 1
    log("starts", len(starts))

    # ---------- 3) مطابقة كل صف بأقرب نقطة في العمود نفسه ورسم الأرقام ----------
    used = set()
    pairs = []
    for row in rows:
        best = None
        for k, dot in enumerate(dots):
            if k in used or dot[0] != row[0] or dot[1] != row[1]:
                continue
            d = abs(dot[2] - row[2])
            if best is None or d < best[0]:
                best = (d, k)
        if best is None:
            log("no dot for row", row[:3])
            continue
        used.add(best[1])
        pairs.append((row, dots[best[1]]))
    log("pairs", len(pairs), "of", len(rows))

    by_page = {}
    for row, dot in pairs:
        by_page.setdefault(dot[0], []).append((row, dot))
    for pi, items in by_page.items():
        page = doc[pi]
        shape = page.new_shape()
        for row, dot in items:
            r = pymupdf.Rect(dot[4])
            r.x0 -= 0.6
            r.y0 -= 0.6
            r.x1 += 0.6
            r.y1 += 0.6
            shape.draw_rect(r)
        shape.finish(color=None, fill=(1, 1, 1))
        shape.commit()
        txt = Text(page)
        for row, dot in items:
            x = max(42.5, dot[3][0] + 1.0)
            txt.add("brand", str(row[3]), x, dot[3][1], FONT_B, 8.6)
        txt.commit()
    log("toc numbers done")

    # ---------- 4) سياق كل صفحة (لأجل الرأس) ----------
    events = sorted((p, t) for t, p in starts.items())
    contexts = {}
    ei, current = 0, None
    for i in range(total):
        while ei < len(events) and events[ei][0] <= i:
            current = events[ei][1]
            ei += 1
        contexts[i] = current

    opener_pages = set(starts.values())
    no_header = opener_pages | set(toc_pages) | {0}

    # ---------- 5) الرؤوس والتذييلات ----------
    h = doc[0].rect.height
    x0, x1 = 42.5, pw - 42.5
    for i in range(total):
        if i % 100 == 0:
            log("headers", i)
        page = doc[i]
        label = header_label(chapters, appx, contexts.get(i))
        txt = Text(page)
        shape = page.new_shape()
        drew = False
        if i not in no_header and label:
            txt.add("muted", book_title, x1, 28, FONT_R, 7.6, "right", maxw=245)
            txt.add("indigo", label, x0, 28, FONT_B, 8.1, "left", maxw=245)
            shape.draw_line(pymupdf.Point(x0, 38.6), pymupdf.Point(x1, 38.6))
            shape.finish(color=LINE, width=0.55)
            shape.draw_line(pymupdf.Point(x1 - 34, 38.6), pymupdf.Point(x1, 38.6))
            shape.finish(color=BRAND, width=1.5)
            shape.draw_rect(pymupdf.Rect(x1 - 2.2, 35.6, x1, 37.8))
            shape.finish(color=None, fill=BRAND)
            drew = True
        if i > 0:
            txt.add("muted", "م. أحمد صالح المغز", x1, h - 24, FONT_R, 7.5, "right")
            txt.add("brand", str(i), x0, h - 24, FONT_B, 8.6)
            shape.draw_line(pymupdf.Point(x0, h - 39), pymupdf.Point(x1, h - 39))
            shape.finish(color=LINE, width=0.55)
            shape.draw_line(pymupdf.Point(x0, h - 39), pymupdf.Point(x0 + 26, h - 39))
            shape.finish(color=BRAND, width=1.3)
            drew = True
        if drew:
            shape.commit()
        txt.commit()
    log("h/f done")

    # ---------- 6) فهرس PDF داخلي ----------
    toc = [[1, "الغلاف", 1]]
    if "I" in starts:
        toc.append([1, "صفحة العنوان", starts["I"] + 1])
    if "T" in starts:
        toc.append([1, "فهرس الكتاب", starts["T"] + 1])
    for key in ("F0", "F1", "F2"):
        if key in starts:
            toc.append([1, FRONT_LABEL[key], starts[key] + 1])
    for pt in book["part_titles"]:
        pn = pt["num"]
        if f"P{pn}" in starts:
            toc.append([1, f"الجزء {pt['ordinal']}: {pt['title']}", starts[f"P{pn}"] + 1])
        for c in book["chapters"]:
            if c["part"] != pn:
                continue
            if f"C{c['num']}" in starts:
                toc.append([2, f"الفصل {c['num']}: {c['title']}", starts[f"C{c['num']}"] + 1])
    if "X" in starts:
        toc.append([1, "الملاحق التسعة", starts["X"] + 1])
    for i, a in enumerate(book["appendices"]):
        if f"X{i}" in starts:
            toc.append([2, a["title"], starts[f"X{i}"] + 1])
    if "Z" in starts:
        toc.append([1, "خاتمة الكتاب", starts["Z"] + 1])
    doc.set_toc(toc)
    log("bookmarks", len(toc))

    doc.set_metadata({
        "title": meta["title"],
        "author": meta.get("author", ""),
        "subject": meta["subtitle"],
        "keywords": "LLM, Transformers, MoE, Mamba, DPO, GRPO, RAG, Agents, Arabic",
        "creator": "BookAI publishing pipeline",
        "producer": "PyMuPDF + Microsoft Edge",
    })

    # ---------- 7) حفظ ----------
    log("saving")
    doc.save(TMP, garbage=1, deflate=True)
    doc.close()
    TMP.replace(PDF)
    log("saved", PDF.stat().st_size)
    print(f"pdf postprocess: pages={total} markers={len(starts)} slots={len(pairs)} "
          f"bookmarks={len(toc)} size={PDF.stat().st_size:,}")


if __name__ == "__main__":
    main()
