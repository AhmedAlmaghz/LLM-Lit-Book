# -*- coding: utf-8 -*-
"""
استخراج وتنظيم مخطوطة LLM-ds.md إلى build/manuscript.json
- يقسّم الملف إلى وحدات: فصول (1..48)، ملاحق (أ..ط)، مقدمة، ملخص، إهداء
- يزيل أقسام التخطيط/العيّنات/التسويق من البداية والنهاية
- رمز الإخراج: JSON بخط UTF-8
"""
import re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "LLM-ds.md"
OUT  = ROOT / "build" / "manuscript.json"

SEP_RE   = re.compile(r"^[=]{8,}\s*$")
HEAD_RE  = re.compile(r"^# (الفصل |ملحق |مقدمة الكتاب|ملخص الكتاب|إهداء)")

PART_OF_CHAPTER = {
    1:1,2:1,3:1, 4:2,5:2,6:2, 7:3,8:3,9:3,10:3,11:3,
    12:4,13:4,14:4,15:4,16:4,17:4,18:4,19:4,
    20:5,21:5,22:5,23:5,24:5, 25:6,26:6,27:6,28:6,
    29:7,30:7,31:7,32:7,33:7,34:7,
    35:8,36:8,37:8,38:8,39:8,40:8, 41:9,42:9,43:9,44:9,45:9,
    46:10,47:10,48:10,
}
PART_TITLES = {
    1:"الأساسيات — من الشبكات العصبية إلى التعلّم العميق",
    2:"التضمينات والتمثيلات — كيف تتحول النصوص إلى أرقام",
    3:"المحوّلات — القلب النابض للنماذج اللغوية",
    4:"بناء LLM من الصفر — خط الأنابيب الكامل",
    5:"التقنيات المتقدمة — من هندسة الأوامر إلى الوكلاء",
    6:"النماذج متعددة الوسائط (Multimodal LLMs)",
    7:"البناء انطلاقًا من نماذج موجودة — بدلًا من الصفر",
    8:"التقييم والاختبار والنشر",
    9:"التطبيقات والمشاريع العملية",
    10:"المقابلات والمسار المهني",
}
AR_ORDINAL = ["الأول","الثاني","الثالث","الرابع","الخامس","السادس","السابع","الثامن","التاسع","العاشر"]

def arabic_num(n):
    m = {"0":"٠","1":"١","2":"٢","3":"٣","4":"٤","5":"٥","6":"٦","7":"٧","8":"٨","9":"٩"}
    return "".join(m[c] for c in str(n))

def parse():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    n = len(lines)
    starts = [i for i, ln in enumerate(lines) if HEAD_RE.match(ln)]
    chapters, appendices, front = [], [], {"dedication": None, "preface": None, "summary": None}
    for j, a in enumerate(starts):
        b = starts[j + 1] if j + 1 < len(starts) else n
        chunk = lines[a:b]
        while chunk and (chunk[-1].strip() == "" or SEP_RE.match(chunk[-1])):
            chunk.pop()
        title_line = chunk[0].strip()
        body = chunk[1:]
        if title_line.startswith("# ملحق"):
            appendices.append((title_line, body))
        elif title_line.startswith("# إهداء"):
            front["dedication"] = body
        elif title_line.startswith("# مقدمة الكتاب"):
            front["preface"] = body
        elif title_line.startswith("# ملخص الكتاب"):
            front["summary"] = body
        else:
            chapters.append((title_line, body))

    items = []
    chapter_out = []
    for idx, (title_line, body) in enumerate(chapters, start=1):
        t = title_line.replace("# الفصل ", "", 1)
        bits = t.split(":", 1)
        ordinal = bits[0].strip()
        short = bits[1].strip() if len(bits) > 1 else ordinal
        short = re.sub(r"\s*[—-]\s*$", "", short)
        sections = [re.sub(r"^## ", "", ln).strip() for ln in body if re.match(r"^## ", ln)]
        code_blocks = sum(1 for ln in body if ln.strip().startswith("```")) // 2
        items.append({
            "num": idx, "num_ar": arabic_num(idx), "ordinal": ordinal,
            "title": short, "sections": sections,
            "part": PART_OF_CHAPTER.get(idx, 0), "code_blocks": code_blocks,
        })
        chapter_out.append({"num": idx, "ordinal": ordinal, "title": short,
                            "part": PART_OF_CHAPTER.get(idx, 0), "body": "\n".join(body)})
    return items, appendices, front, chapter_out

def main():
    items, appendices, front, chapter_out = parse()
    data = {
        "parts": [{"num": k, "ordinal": AR_ORDINAL[k-1], "title": v} for k, v in PART_TITLES.items()],
        "chapters": items,
        "chapters_raw": chapter_out,
        "appendices": [{"title": t.replace("# ", "", 1), "lines": len(b)} for t, b in appendices],
        "appendices_raw": [{"title": t.replace("# ", "", 1), "body": "\n".join(b)} for t, b in appendices],
        "front": {k: ("\n".join(v) if v else None) for k, v in front.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for it in items:
        print(f"P{it['part']}|CH{it['num']:02d}|secs={len(it['sections']):02d}|code={it['code_blocks']}")
    print("appendices:", len(appendices), "| preface:", bool(front["preface"]), "| devotion:", bool(front["dedication"]), "| summary:", bool(front["summary"]))
    print("OK ->", OUT)

if __name__ == "__main__":
    main()
