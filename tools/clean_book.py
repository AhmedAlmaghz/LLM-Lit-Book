# -*- coding: utf-8 -*-
"""
تنظيف وتحرير المخطوطة إلى build/book.json:
- محلل بلوكات (عناوين، فقرات، اقتباسات تنبيهية، كود، جداول، قوائم)
- ماسح أسوار الكود بآلة حالات للسلاسل النصية (يحل أكواد f-string المدمجة)
- جدول إصلاحات موضعية (أخطاء كود مصدرية)
- إزالة الملاحق التحريرية (انتهى الفصل/دعوة للكتابة)
- إحصاءات لكل فصل (كلمات، دقائق قراءة، أكواد، تنبيهات)
"""
import re, json, io, sys, unicodedata
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
MS   = ROOT / "build" / "manuscript.json"
OUT  = ROOT / "build" / "book.json"

# ---------------------------------------------------------------- code fence
# ماسح أسوار متسامح: يتجاهل أي ``` يقع داخل سلسلة نصية ثلاثية في بايثون
def _python_string_mode(text):
    """يعيد True إذا كانت آخر حاجة للنص قربه واقعًا داخل سلسلة ثلاثية أو عادية مفتوحة."""
    i, n = 0, len(text)
    mode = None  # None | "'" | '"' | "'''T" | '"""T'
    while i < n:
        c = text[i]
        if mode is None:
            if text[i:i+3] in ('"""', "'''"):
                mode = "T" if text[i:i+3] == '"""' else "S3"
                i += 3; continue
            if c == '"': mode = 'D'; i += 1; continue
            if c == "'": mode = 'S'; i += 1; continue
            if c == '#':  # تعليق: تجاهل حتى نهاية السطر
                j = text.find("\n", i)
                i = n if j == -1 else j + 1
                continue
            if c == "\\" and i + 1 < n: i += 2; continue
            i += 1
        elif mode in ('D', 'S'):
            if c == "\\": i += 2; continue
            if (mode == 'D' and c == '"') or (mode == 'S' and c == "'"):
                mode = None
            if c == "\n":
                # سلسلة عادية لا تمتد أسطرًا عادةً؛ إن وصلنا نهاية السطر نعيد التهيئة
                mode = None
            i += 1
        elif mode == "T":
            if text[i:i+3] == '"""': mode = None; i += 3; continue
            if c == "\\": i += 2; continue
            i += 1
        elif mode == "S3":
            if text[i:i+3] == "'''": mode = None; i += 3; continue
            if c == "\\": i += 2; continue
            i += 1
    return mode

def _guess_lang(code, declared):
    if declared:
        return declared
    head = "\n".join(code.splitlines()[:6])
    # مخططات نصية/رموز عناصر رسومية
    if re.search(r"[\u2190-\u21FF\u2588\u2591\u2500-\u257F\u25A0\u25CF\u251C\u2514]", code) \
            and not re.search(r"\b(import |def |class |print\(|return\b)", code):
        return "text"
    # YAML محلّي
    if re.match(r"^# .*\.ya?ml", head) or re.search(r"^(version|name|author|created|prompt|model|data)\s*:\s", head, re.M):
        return "yaml"
    # قوالب أوامر (prompts)
    if re.search(r"^\s*(أنت |المهمة:|الملف:|إرشادات:|التنسيق:|السياق:)", head) and not re.search(r"\b(def |import )", code):
        return "prompt"
    if re.search(r"\b(import |from |\bdef |\bclass |print\()", code):
        return "python"
    # شبه-كود بخطوات مرقّمة أو مخطط مادي
    if re.search(r"^\s*\d+\.\s", code, re.M) or re.search(r"^\s*\$\w", code, re.M):
        return "text"
    if re.search(r"^\s*(npm |pip |python |cd |export |git |curl |wget |torchrun |docker |\w+\.\w+ )", code, re.M):
        return "bash"
    if re.search(r"^\s*[\u0621-\u064A\w]+=\w", code, re.M) and not re.search(r"^\s*\d+\.\s", code, re.M):
        return "python"
    return "txt"

def parse_blocks(text):
    """يحول نص Markdown إلى قائمة بلوكات منظمة."""
    lines = text.splitlines()
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        # --- أسوار الكود (بثلاثية أو أكثر) ---
        m = re.match(r"^(`{3,}|~{3,})([\w+-]*)\s*$", ln)
        if m:
            fence, declared = m.group(1)[0]*3, m.group(2)
            j, buf = i + 1, []
            while j < n:
                if re.match(r"^(`{3,}|~{3,})\s*$", lines[j]):
                    # إن كنا داخل بايثون وسلسلة نصية مفتوحة → هذا سطر نص داخل الكود
                    if _guess_lang("\n".join(buf), declared) == "python" and _python_string_mode("\n".join(buf)):
                        buf.append(lines[j]); j += 1; continue
                    break
                buf.append(lines[j]); j += 1
            code = "\n".join(buf)
            # تصحيح لغة: فنّان المصادر رشّ "python" على قالب YAML
            guessed = _guess_lang(code, declared)
            if guessed in ("python", "py") and declared in ("python", "py"):
                if re.match(r"^# .*\.ya?ml", code) or re.search(r"^version\s*:|\bcreated:\s*\d{4}", code):
                    guessed = "yaml"
            blocks.append({"t": "code", "lang": guessed, "code": code})
            i = j + 1
            continue
        # --- HR ---
        if re.match(r"^-{3,}$", s) or re.match(r"^_{3,}$", s):
            blocks.append({"t": "hr"}); i += 1; continue
        # --- Headings ---
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            blocks.append({"t": f"h{min(lvl,4)}", "text": txt}); i += 1; continue
        # --- Quote (callout) ---
        if s.startswith(">"):
            buf, j = [], i
            while j < n and lines[j].strip().startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[j])); j += 1
            qtxt = " ".join(x.strip() for x in buf)
            blocks.append({"t": "quote", "text": qtxt, "kind": callout_kind(qtxt)})
            i = j; continue
        # --- Table ---
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i+1] or ""):
            rows = []
            j = i
            while j < n and lines[j].strip().startswith("|"):
                r = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                rows.append(r); j += 1
            header = rows[0]
            body = rows[2:] if len(rows) > 2 else rows[1:]
            blocks.append({"t": "table", "header": header, "rows": body})
            i = j; continue
        # --- Lists ---
        if re.match(r"^[-*]\s+", s):
            buf, j = [], i
            while j < n and re.match(r"^\s*[-*]\s+", lines[j]):
                buf.append(re.sub(r"^\s*[-*]\s+", "", lines[j])); j += 1
            blocks.append({"t": "ul", "items": buf}); i = j; continue
        if re.match(r"^\d+[.)]\s+", s):
            buf, j = [], i
            while j < n and re.match(r"^\s*\d+[.)]\s+", lines[j]):
                buf.append(re.sub(r"^\s*\d+[.)]\s+", "", lines[j])); j += 1
            blocks.append({"t": "ol", "items": buf}); i = j; continue
        # --- Paragraph ---
        if s:
            buf, j = [ln], i + 1
            while j < n:
                nx = lines[j].strip()
                if (not nx or nx.startswith(("#", ">", "```", "~~~", "|", "-", "*", "—"))
                        or re.match(r"^\d+[.)]\s", nx) or re.match(r"^-{3,}$", nx)):
                    break
                buf.append(lines[j]); j += 1
            blocks.append({"t": "p", "text": " ".join(x.strip() for x in buf)})
            i = j; continue
        i += 1
    return blocks

# ------------------------------------------------------------- callout kinds
CALLOUTS = [
    ("نصيحة ذهبية", "gold"), ("أفضل ممارسة", "best"), ("أفضل الممارسات", "best"),
    ("خطأ شائع قاتل", "danger"), ("خطأ شائع جدًا", "danger"), ("خطأ شائع", "mistake"),
    ("تحذير قاتل", "danger"), ("تحذير", "warn"), ("تنبيه", "warn"),
    ("نصيحة عملية", "tip"), ("نصيحة", "tip"), ("تلميح", "tip"),
    ("سؤال يفتح الشهية", "hook"), ("سؤال يستحق التفكير", "think"), ("سؤال", "think"),
    ("مقارنة", "compare"), ("قاعدة عملية", "rule"), ("درس عملي", "lesson"),
    ("دراسة حالة", "case"), ("ملاحظة صادقة", "note"), ("ملاحظة", "note"), ("درس", "lesson"),
    ("تلميح للاختبار", "tip"), ("تلميح للمنافسة", "tip"), ("فرق مهم", "mistake"), ("احتفظ بهذه النقطة", "gold"),
]
def callout_kind(text):
    m = re.match(r"^\*{0,2}([\u0621-\u064A ]{3,20})\*{0,2}\s*:?", text.lstrip("*"))
    if m:
        head = m.group(1).strip()
        for k, v in CALLOUTS:
            if head.startswith(k):
                return v
    return "note"

# ---------------------------------------------------------------- fixes
FIXES = [
    # (رقم الفصل أو None، نص قديم، نص جديد) — إصلاحات موضعية تم التحقق منها يدويًا
    (1,
     "# التدرج التحليلي (الذي حسبناه في الكود السابق)\n_, a1, _, a2 = forward(X, W1, b1, W2, b2)\ndW1_analytic, _, _, _ = backward(X, y, _, a1, _, a2, W2)",
     "# التدرج التحليلي (الذي حسبناه في الكود السابق)\nz1_, a1_, z2_, a2_ = forward(X, W1, b1, W2, b2)\ndW1_analytic, _, _, _ = backward(X, y, z1_, a1_, z2_, a2_, W2)"),
    (1,
     "def loss_for_W1(W1_val):\n    W1_tmp = W1_val.copy()\n    _, _, _, a2 = forward(X, W1_tmp, b1, W2, b2)\n    return bce_loss(y, a2)",
     "def loss_for_W1(W1_val):\n    W1_tmp = W1_val.copy()\n    _, _, _, a2 = forward(X, W1_tmp, b1, W2, b2)\n    return bce_loss(y, a2)"),
    (2,
     "model.eval()    # قبل التقييم\nwith torch.no_grad():\n    # ... evaluation ...\n```",
     "model.eval()    # قبل التقييم\nwith torch.no_grad():\n    pass  # ... evaluation ...\n```"),
]

EDITS = [
    # إزالة الملاحق التحريرية الخفية في نهايات الفصول
    (re.compile(r"^\*\*انتهى .+?\.\*\*$", re.M), ""),
    (re.compile(r"^هل تريد أن أكتب .*", re.M), ""),
    (re.compile(r"^قُبل بالجزء .*", re.M), ""),
]

def apply_text_fixes(ch_num, text):
    for chap, old, new in FIXES:
        if chap in (None, ch_num):
            if old in text:
                text = text.replace(old, new)
    for rx, repl in EDITS:
        text = rx.sub(repl, text)
    # أسطر فارغة متعددة → واحدة
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------- title split
def split_title(ordinal, title):
    t = title.replace(f"الفصل {ordinal}", "", 1)
    t = t.lstrip(":—- ").strip()
    if " — " in t:
        a, b = t.split(" — ", 1)
        return a.strip(" —"), b.strip()
    if ": " in t and len(t.split(": ", 1)[0]) < 60:
        a, b = t.split(": ", 1)
        return a.strip(), b.strip()
    return t, ""

# ---------------------------------------------------------------- stats
def block_stats(blocks):
    words = 0; code = 0; quotes = Counter(); tables = 0
    for b in blocks:
        if b["t"] in ("p",):
            words += len(b["text"].split())
        elif b["t"].startswith("h"):
            words += len(b.get("text", "").split())
        elif b["t"] == "code":
            if b.get("lang") in ("python", "py"): code += 1
            words += len(b["code"].split()) // 3
        elif b["t"] == "quote":
            quotes[b.get("kind", "note")] += 1
            words += len(b.get("text", "").split())
        elif b["t"] == "table":
            tables += 1
            words += sum(len(" ".join(r).split()) for r in b.get("rows", []))
        elif b["t"] in ("ul", "ol"):
            words += sum(len(x.split()) for x in b.get("items", []))
    minutes = max(2, round(words / 180))
    return {"words": words, "minutes": minutes, "code": code,
            "quotes": dict(quotes), "tables": tables}

def sections_of(blocks):
    secs = []
    for b in blocks:
        if b["t"] == "h2":
            secs.append(b["text"])
    return secs

def augment_title(title):
    # فرصة لإزالة تكرار اسم الفصل في العنوان الفرعي
    return title

def clean_unit(raw_title, body, ch_num=None, letter=None):
    text = apply_text_fixes(ch_num, body)
    blocks = parse_blocks(text)
    # إسقاط hr الافتتاحي بعد العنوان
    while blocks and blocks[0]["t"] == "hr":
        blocks.pop(0)
    # إسقاط hr الختامي
    while blocks and blocks[-1]["t"] == "hr":
        blocks.pop()
    return blocks

def main():
    ms = json.loads(MS.read_text(encoding="utf-8"))
    out_ch = []
    for c in ms["chapters_raw"]:
        blocks = clean_unit(c["title"], c["body"], ch_num=c["num"])
        st = block_stats(blocks)
        out_ch.append({
            **{k: c[k] for k in ("num", "ordinal", "title", "part")},
            "blocks": blocks,
            "sections": sections_of(blocks),
            "stats": st,
        })
    out_app = []
    for a in ms["appendices_raw"]:
        blocks = clean_unit(a["title"], a["body"])
        out_app.append({"title": a["title"], "blocks": blocks, "stats": block_stats(blocks), "sections": sections_of(blocks)})
    front = {}
    for k in ("dedication", "preface", "summary"):
        front[k] = parse_blocks(ms["front"][k]) if ms["front"].get(k) else []

    data = {
        "meta": {"title": "المحوّلات والنماذج اللغوية الكبيرة",
                 "subtitle": "من النظرية إلى بناء أنظمة ذكية",
                 "edition": "الطبعة الأولى — 2026",
                 "publisher": "BookAI"},
        "front": front,
        "chapters": out_ch,
        "appendices": out_app,
    }
    total_words = sum(c["stats"]["words"] for c in out_ch)
    total_code = sum(c["stats"]["code"] for c in out_ch)
    total_minutes = sum(c["stats"]["minutes"] for c in out_ch)
    data["meta"]["stats"] = {"words": total_words, "code_blocks": total_code,
                             "minutes": total_minutes,
                             "pages_est": int(total_words / 320)}
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"chapters={len(out_ch)} appendix={len(out_app)} words≈{total_words:,} code={total_code} hours≈{total_minutes/60:.1f}")
    print("OK ->", OUT)

if __name__ == "__main__":
    main()
