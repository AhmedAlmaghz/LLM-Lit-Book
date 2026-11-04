# -*- coding: utf-8 -*-
"""
build_md.py — يبني الإصدار الرئيسي للمخطوطة بصيغة Markdown:
  dist/llm-ds.md
المصدر: build/book.json (+ عناوين الأجزاء من build/manuscript.json)
يُنسخ كل المخططات إلى dist/images/
الهيكل: صفحة العنوان → فهرس الكتاب → فهرس الأشكال → الهداء → المقدمة → خطة الكتاب →
        10 أجزاء × 48 فصلًا (مع مخططات نهاية الفصل) → الملاحق → الخاتمة
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
from booklib import load, ensure_dist_images, quote_label, FIGS, DIST
import mathlite


def mth(s):
    """رياضيات MD → صيغة يونيكود مقروءة (لا تظهر رموز LaTeX الخام في أي قارئ)."""
    return mathlite.render(s)


def img_md(key, num_label):
    m = FIGS[key]
    return f"![{m['title']}](images/{key}.png)\n*{num_label}: {m['title']}*"


def block_md(b):
    """بلوك واحد → نص MD (سلاسل متعددة الأسطر داخل فقرة واحدة)."""
    t = b["t"]
    if t in ("h1", "h2", "h3"):
        return "#" * {"h1": 1, "h2": 2, "h3": 3}[t] + " " + mth(b["text"])
    if t == "p":
        return mth(b["text"])
    if t == "quote":
        return "\n".join("> " + seg.strip()
                         for seg in quote_label(mth(b["text"]), b.get("kind", "note")).split("\n"))
    if t == "code":
        return f"````{b.get('lang') or 'text'}\n{b['code'].rstrip()}\n````"
    if t == "ul":
        return "\n".join("- " + mth(it).replace("\n", " ") for it in b["items"])
    if t == "ol":
        return "\n".join(f"{k}. " + mth(it).replace("\n", " ") for k, it in enumerate(b["items"], 1))
    if t == "table":
        esc = lambda s: re.sub(r"\|", r"\\|", re.sub(r"\s*\n\s*", "<br>", mth(s)))
        head = b["header"]
        rows = b["rows"]
        out = ["| " + " | ".join(esc(h) for h in head) + " |",
               "|" + "|".join([" --- "] * len(head)) + "|"]
        out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
        return "\n".join(out)
    if t == "hr":
        return "---"
    raise ValueError(t)


def render_blocks(blocks):
    """قائمة فقرات، مع إسقاط --- المجاور للعناوين/الجداول/الكود."""
    out = []
    for i, b in enumerate(blocks):
        if b["t"] == "hr":
            prev = blocks[i - 1]["t"] if i else None
            nxt = blocks[i + 1]["t"] if i + 1 < len(blocks) else None
            if prev in (None, "h1", "h2", "h3", "hr", "code", "table", "quote") or \
               nxt in (None, "h1", "h2", "h3", "hr", "code", "table", "quote"):
                continue
        out.append(block_md(b))
    return out


def main():
    mathlite.last_failures.clear()
    book = load()
    ensure_dist_images()
    meta = book["meta"]
    parts = book["part_titles"]
    st = meta["stats"]

    chunks = []
    add = chunks.append

    # ---------- صفحة العنوان ----------
    add("\n\n".join([
        f"# {meta['title']}",
        f"#### {meta['subtitle']}",
        f"**{meta.get('author_display', meta.get('author', ''))}**",
        f"**{meta['edition']}** · {meta['publisher']} · ⏱ ~{st['minutes']} دقيقة قراءة",
    ]))
    add("| العنصر | القيمة |\n| --- | --- |\n"
        + "\n".join([
            f"| الأجزاء | 10 |", f"| الفصول | 48 |", f"| الملاحق | 9 |",
            f"| المخططات التوضيحية | 33 |", f"| الكلمات | ~{st['words']:,} |",
            f"| كتل الكود القابلة للتشغيل | {st['code_blocks']}+ |",
        ]))
    add("> **الحزمة المرفقة**: مجلد `code/` فيه نسخة `.py` قابلة للتشغيل لكل فصل (جاهزة لـ Colab/Kaggle)، "
        "وفي `code/notebooks/` دفتر Jupyter لكل جزء، و`images/` كل المخططات بدقة عالية.")

    # ---------- فهرس الكتاب ----------
    toc = ["## فهرس الكتاب", ""]
    for pt in parts:
        chs = [c for c in book["chapters"] if c["part"] == pt["num"]]
        toc.append(f"**الجزء {pt['ordinal']}: {mth(pt['title'])}** *(الفصول {chs[0]['num']}–{chs[-1]['num']})*")
        toc.extend(f"- الفصل {c['num']}: {mth(c['title'])}" for c in chs)
        toc.append("")
    toc.append("**الملاحق** *(ملحق أ — ط)*")
    toc.extend(f"- {mth(a['title'])}" for a in book["appendices"])
    add("\n".join(toc))

    # ---------- فهرس الأشكال ----------
    fi = ["## فهرس الأشكال", "كل مخطط مضمّن في نهاية فصله (أو شريحة افتتاح جزئه).", ""]
    for c in book["chapters"]:
        ks = B.figs_for_chapter(c["num"])
        for j, k in enumerate(ks, 1):
            fi.append(f"- **شكل {c['num']}.{j}** — {FIGS[k]['title']} *(الفصل {c['num']})*")
    for key, m in FIGS.items():
        if isinstance(m["chapter"], tuple):
            pn = m["chapter"][1]
            fi.append(f"- **شكل ج{pn}** — {m['title']} *(افتتاح الجزء {pn})*")
    add("\n".join(fi))

    # ---------- أعمال تأسيسية ----------
    fr = book["front"]
    add("## الهداء\n\n" + "\n\n".join(mth(b["text"]) for b in fr["dedication"][0:9] if b["t"] == "p"))
    add("## المقدمة")
    chunks.extend(render_blocks(fr["preface"]))
    add("## خطة الكتاب")
    chunks.extend(render_blocks(fr["summary"]))

    # ---------- الأجزاء والفصول ----------
    for pt in parts:
        pn = pt["num"]
        add(f"# الجزء {pt['ordinal']}: {mth(pt['title'])}")
        for key in B.figs_for_part(pn):
            add(img_md(key, f"شكل ج{pn}"))
        for c in [c for c in book["chapters"] if c["part"] == pn]:
            add(f"## الفصل {c['num']}: {mth(c['title'])}")
            chunks.extend(render_blocks(c["blocks"]))
            ks = B.figs_for_chapter(c["num"])
            if ks:
                head = "## المخططات التوضيحية" if len(ks) > 1 else "## المخطط التوضيحي"
                add(head)
                for j, key in enumerate(ks, 1):
                    add(img_md(key, f"شكل {c['num']}.{j}"))
                for key in ks:
                    m = FIGS[key]
                    if m["page_extra"]:
                        extra = "، ".join(str(x) for x in m["page_extra"])
                        add(f"> ⤵️ يتفرع هذا المخطط عمليًا في الفصل {extra} من هذا الكتاب.")

    # ---------- الملاحق ----------
    add("# الملاحق")
    for a in book["appendices"]:
        add(f"## {mth(a['title'])}")
        chunks.extend(render_blocks(a["blocks"]))

    # ---------- الخاتمة ----------
    add("---\n\n*انتهى كتاب «%s» — %s. أُنجز عبر خط أنابيب المنشور في `tools/`: "
        "~%s كلمة، %s كتلة كود، 33 مخططًا توضيحيًا، و5 صيغ إخراج (MD · HTML · PDF · DOCX · EPUB).*"
        % (meta['title'], meta['edition'], f"{st['words']:,}", st['code_blocks']))

    text = "\n\n".join(chunks) + "\n"
    text = re.sub(r"[ \t]+\n", "\n", text)

    out = DIST / "llm-ds.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("md bytes:", out.stat().st_size)


if __name__ == "__main__":
    main()
