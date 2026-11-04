# -*- coding: utf-8 -*-
"""
booklib.py — المكتبة المشتركة لكل أدوات البناء (MD · HTML · PDF · DOCX · EPUB)
مصدر الحقيقة: build/book.json + build/manuscript.json (عناوين الأجزاء) + ./figures/png
"""
import json, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
FIGS_PNG = ROOT / "figures" / "png"
DIST = ROOT / "dist"
IMG_ROOT = ROOT / "images"           # داخل مجلد dist

AUTHOR = "أحمد صالح المغز"
AUTHOR_DISPLAY = "م. أحمد صالح المغز"

KIND_LABEL = {
    "tip": "نصيحة", "warn": "تحذير", "note": "ملاحظة", "hook": "سؤال يفتح الشهية",
    "lesson": "درس عملي", "danger": "خطأ شائع قاتل", "rule": "قاعدة عملية",
    "gold": "نصيحة ذهبية", "think": "سؤال يستحق التفكير", "compare": "مقارنة سريعة",
    "mistake": "خطأ شائع", "best": "أفضل ممارسة",
}

# خريطة المخططات: مفتاح = اسم الملف داخل figures/png
# chapter: 1..48 (فصل أساسي) أو ("part", N) لشريحة افتتاح جزء
FIGS = {
    "f01_neuron":       {"chapter": 1,   "title": "الخلية العصبية: من المدخلات إلى التنشيط", "page_extra": []},
    "f02_backprop":     {"chapter": 1,   "title": "الرسم البياني الحسابي لقاعدة السلسلة", "page_extra": []},
    "f03_curves":       {"chapter": 2,   "title": "منحنيات التدريب: خمسة أنماط تُقرأ قبل الأرقام", "page_extra": []},
    "f04_regularizers": {"chapter": 2,   "title": "التنظيم: أدوات منع الحفظ الأعمى", "page_extra": []},
    "f05_amp":          {"chapter": 3,   "title": "بنية التدريب الحديثة: دقة مختلطة + تراكم + توزيع", "page_extra": []},
    "f06_bpe":          {"chapter": 4,   "title": "BPE: من بايتات حقيقية إلى وحدات ذات معنى", "page_extra": []},
    "f07_w2v":          {"chapter": 5,   "title": "فضاء التضمينات: الهندسة تحمل المعنى", "page_extra": []},
    "f08_lstm":         {"chapter": 6,   "title": "LSTM: بوابة إدارة الذاكرة", "page_extra": []},
    "f09_attention":    {"chapter": 7,   "title": "الانتباه الذاتي: Q·Kᵀ ÷ √dₖ ثم المزج مع V", "page_extra": []},
    "f10_transformer":  {"chapter": 8,   "title": "بنية المحوّل: من كتلة واحدة إلى معمارية كاملة", "page_extra": []},
    "f11_families":     {"chapter": 9,   "title": "ثلاث فلسفات في بناء العقل: BERT · GPT · T5", "page_extra": []},
    "f12_modern":       {"chapter": 10,  "title": "ما بعد المحوّل الكلاسيكي: FlashAttention · MLA · MoE · Mamba", "page_extra": []},
    "f13_quant":        {"chapter": 11,  "title": "سلّم التكميم: دقة مقابل ذاكرة", "page_extra": []},
    "f14_pipeline":     {"chapter": ("part", 4), "title": "خريطة البناء من الصفر: خط الأنابيب الكامل", "page_extra": []},
    "f15_mini_llm":     {"chapter": 14,  "title": "MiniLLM: بنية النموذج من الداخل", "page_extra": []},
    "f16_trainloop":    {"chapter": 15,  "title": "حلقة التدريب: تجميع تدرّجات + تقصير + استئناف", "page_extra": []},
    "f17_lora":         {"chapter": 16,  "title": "LoRA: زيادة صغيرة فوق أوزان جامدة", "page_extra": []},
    "f18_dpo":          {"chapter": 17,  "title": "من PPO إلى DPO: نفس الهدف، نصف العتاد", "page_extra": []},
    # f19 غير موجودة (الفصل 18: الاستدلال — بلا مخطط)
    "f20_distill":      {"chapter": 30,  "title": "التقطير: علّم الطالب أن يفكر كالمعلم", "page_extra": []},
    "f21_rag":          {"chapter": 23,  "title": "حلقة RAG: استرجاع ثم إجابة", "page_extra": [43]},
    "f22_agent":        {"chapter": 24,  "title": "حلقة الوكيل (ReAct): فكر، افعل، اطّلع، كرّر", "page_extra": []},
    "f23_vlm":          {"chapter": 25,  "title": "VLM: كيف يلتقي التدفق البصري مع اللغة", "page_extra": [26]},
    "f24_paths":        {"chapter": 29,  "title": "متى لا تبدأ من الصفر؟ محور التكلفة والتحكم", "page_extra": []},
    "f25_merge":        {"chapter": 31,  "title": "دمج النماذج: مزج الأقدار دون تدريب", "page_extra": []},
    "f26_prune":        {"chapter": 32,  "title": "التشذيب: احذف — ثم استعد المعنى", "page_extra": []},
    "f27_loraserve":    {"chapter": 33,  "title": "نموذج أساس واحد + مئة عميل: LoRA-as-a-Service", "page_extra": []},
    "f28_selfimp":      {"chapter": 34,  "title": "التحسين الذاتي: كل النموذج معلمًا لنفسه", "page_extra": []},
    "f29_metrics":      {"chapter": 35,  "title": "مصفوفة الالتباس: أبعاد الصواب وأفق المسامحة", "page_extra": []},
    "f30_judge":        {"chapter": 37,  "title": "LLM-as-a-Judge: حكم آلي مع حذر بشري", "page_extra": []},
    "f31_deploy":       {"chapter": 40,  "title": "من المختبر إلى الإنتاج: مخطط النشر", "page_extra": []},
    "f32_security":     {"chapter": 39,  "title": "هجوم الحقن: صوت خارج القرار", "page_extra": []},
    "f33_projects":     {"chapter": ("part", 9), "title": "خمسة مشاريع متكاملة — من وصفية إلى توظيف", "page_extra": []},
    "f34_patterns":     {"chapter": 47,  "title": "أنماط تصميم أنظمة LLM", "page_extra": []},
}

_sep_tail = re.compile(r"\s*[=]{5,}\s*$")
_pure_sep = re.compile(r"[=\-_\s\*]{4,}")


def load():
    """يعيد dict فيه كل ما يحتاجه أي باني."""
    book = json.loads((BUILD / "book.json").read_text(encoding="utf-8"))
    ms = json.loads((BUILD / "manuscript.json").read_text(encoding="utf-8"))
    parts = [{"num": p["num"], "ordinal": p["ordinal"], "title": p["title"]}
             for p in ms["parts"]]
    # الفصل 48 ينتهي بـ h1 "الملاحق" فهو فاصل ماسح — نحذفه هنا
    for c in book["chapters"]:
        if c["num"] == 48 and c["blocks"] and c["blocks"][-1].get("t") == "h1":
            c["blocks"] = c["blocks"][:-1]
    book["part_titles"] = parts
    book["meta"].setdefault("author", AUTHOR)
    book["meta"].setdefault("author_display", AUTHOR_DISPLAY)
    return book


def is_pure_sep(text):
    return bool(_pure_sep.fullmatch(text or ""))


def clean_text(t):
    """إزالة الفواصل المسحوبة التي التحمت بنهاية الفقرات."""
    t = _sep_tail.sub("", t)
    return t.strip()


def render_blocks(container):
    """يعيد قائمة البلوكات جاهزة للبناء بعد الإسقاطات المحلية.
    يقبل إما حاوية (dict فيه blocks) أو قائمة بلوكات مباشرة."""
    items = container["blocks"] if isinstance(container, dict) else container
    out = []
    for b in items:
        t = b["t"]
        if t == "h1" and b.get("text", "").strip() == "الملاحق":
            continue
        if t == "p" and is_pure_sep(b.get("text", "")):
            continue
        bb = dict(b)
        if t in ("p", "quote"):
            bb["text"] = clean_text(bb.get("text", ""))
        out.append(bb)
    return out


def quote_label(text, kind):
    """التسميات مدمجة في النص أصلاً؛ نضيفها فقط عند غيابها وعدم بدءه بعلامة اقتباس."""
    if text.startswith("**") or text.count("**", 0, 60) >= 2:
        return text
    head = text.lstrip()
    if head.startswith("“") or head.startswith('"'):
        return text
    lbl = KIND_LABEL.get(kind, "ملاحظة")
    return f"**{lbl}**: {text}"


# ---------------------------------------------------------------- الخريطة الفصلية
def figs_for_chapter(num):
    return [k for k, m in FIGS.items() if m["chapter"] == num]


def figs_for_part(n):
    return [k for k, m in FIGS.items() if isinstance(m["chapter"], tuple) and m["chapter"][1] == n]


def chapter_to_part(book, num):
    for c in book["chapters"]:
        if c["num"] == num:
            return c["part"]
    return None


def fig_path(key):
    return FIGS_PNG / f"{key}.png"


# ---------------------------------------------------------------- النسخ إلى dist
def ensure_dist_images():
    """dist/images/*.png : كل المخططات (لاحقًا + cover.png)"""
    dst = DIST / "images"
    dst.mkdir(parents=True, exist_ok=True)
    for key in FIGS:
        src = fig_path(key)
        if src.exists():
            shutil.copy2(src, dst / src.name)
    return dst
