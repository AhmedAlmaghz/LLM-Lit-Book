# -*- coding: utf-8 -*-
"""المخططات التوضيحية (2/2): من BN/GPT/T5 إلى النشر، مع شريحة مقدّمة الجزء الرابع"""
from figurekit import Fig, C
import math

def f11_families():
    F = Fig(1040, 560)
    F.arrows_bold("agrA")
    F.text(F.w/2, 34, "ثلاث فلسفات في بناء العقل اللغوي", size=21, weight=700)
    fams = [
        ("BERT — مُشفّر ثنائي الجانب", C["emerald"], "Bidirectional Masked LM",
         ["يقرأ من الجانبين (قبل/بعد)", "يخمّن المقنّع <MASK>", "استعمال: تصنيف، POS، فهم", "لا قيد على الرؤية"]),
        ("GPT — مُفكّك سببي", C["violet"], "Causal LM (next token)",
         ["يتنبأ بالتالي فقط (توليد)", "يعمل تدفّقيًا عبر KV cache", "استعمال: توليد، محادثة، كود", "قناع مثلثي (لا نظرة للأمام)"]),
        ("T5 — مُشفّر + مُفكّك", C["indigo"], "Text-to-Text Span Corruption",
         ["قارئ ثم كاتب (Enc→Dec)", "تدريب استعادة المقاطع (Span)", "استعمال: ترجمة، تلخيص، صياغة", "قناعان يجتمعان"]),
    ]
    cw = (F.w - 2*F.pad - 60)/3
    for k, (title, col, sub, lines) in enumerate(fams):
        x = F.pad + k*(cw+30)
        F.rrect(x, 84, cw, 400, "white", stroke=C["line"], r=16)
        F.parts.append(f'<path d="M{x+26},84 H{x+cw-26} a26,26 0 0 1 26,26 v44 h-{cw} v-44 a26,26 0 0 1 26,-26 z" fill="{col}"/>')
        F.text(x+cw/2, 112, title.split(" — ")[0], size=19, fill="#fff", weight=800)
        F.text(x+cw/2, 140, title.split(" — ")[1], size=13, fill="#fff", opacity=0.92)
        F.badge(x+cw/2-92, 186, sub, col, size=10.5, mono=False)
        yy = 240
        for j, ln in enumerate(lines):
            F.text(x+26, yy, ("• " + ln), size=13, fill=C["slate"], anchor="start")
            yy += 40
        # شريحة القناع
        ym = 380
        F.text(x+cw/2, ym-20, ("قناع ثنائي الجانب" if k == 0 else "قناع سببي مثلثي" if k == 1 else "قناعان معاً"), size=11.5, fill=C["mute"])
        cell = 15
        for i in range(5):
            for j in range(5):
                xx = x + cw/2 - (5*cell)/2 + j*cell
                yv = ym + i*cell
                on = True if k == 0 else (j <= i if k == 1 else (j <= i or True))
                if k == 2:
                    on = j <= i if j < 5 - i else True
                colr = col if on else C["panel2"]
                F.parts.append(f'<rect x="{xx}" y="{yv}" width="{cell-3}" height="{cell-3}" rx="3" fill="{colr}" opacity="{1.0 if on else 0.5}"/>')
        F.badge(x+cw/2-64, 448, ["فهم لا توليد", "توليد لا فهم", "فهم + توليد"][k], col, size=11)
    F.text(F.w/2, 530, "السؤال في المقابلات: متى تستخدم causal ومتى bidirectional؟ — الإجابة دائمًا: في نوع الهدف التدريبي!", size=13.3, fill=C["mute"])
    F.section_note("الفصل 9 — BERT · GPT · T5")
    return F

def f12_modern():
    F = Fig(1040, 560)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 34, "ما بعد المحوّل الكلاسيكي: معايير الإنتاج 2024–2026", size=21, weight=700)
    # FlashAttention band
    F.rrect(F.pad, 76, F.w-2*F.pad, 96, C["cyanL"], stroke="#0891B2", r=14)
    F.text(F.w/2, 100, "لا مصفوفة كاملة أبدًا — وفر ذاكرة HBM بالتقطيع والتسلسل الفوري", size=14.5, weight=700, fill=C["cyan"])
    F.text(F.w/2, 122, "FlashAttention 1/2/3 : Online Softmax + Tiling", size=12.5, mono=True, fill=C["cyan"])
    F.text(F.w/2 - 220, 148, "FP16 ~740 TFLOPS (FA-3 على H100)", size=12, mono=True, anchor="start", fill=C["cyan"])
    F.text(F.w/2 + 60, 148, "FP8 ~1.2 PFLOPS", size=12, mono=True, anchor="start", fill=C["cyan"])
    F.text(F.w/2 + 300, 148, "ذاكرة O(N) بدل O(N²)", size=12, mono=True, anchor="start", fill=C["cyan"])
    # ثلاث متّجهات
    archs = [
        ("Dense Transformer", C["indigo"], ["جميع الطبقات كثيفة", "MLA لتقليص KV cache", "أمثلة: LLaMA، Qwen"]),
        ("MoE (خبراء)", C["violet"], ["موجّه يفعّل k من N خبيرًا", "سعة أعلى بكلفة معاملات أقل", "DeepSeek-V3 · Mixtral"]),
        ("Mamba (SSM)", C["emerald"], ["تعاقد زمني مستمر (Recurrence)", "كفاءة O(N) بالتسلسل", "هجين مع الانتباه (Samba)"]),
    ]
    cw = (F.w - 2*F.pad - 60)/3
    for k, (t, col, lines) in enumerate(archs):
        x = F.pad + k*(cw+30)
        F.rrect(x, 208, cw, 210, "white", stroke=C["line"], r=14)
        F.parts.append(f'<rect x="{x}" y="208" width="6" height="210" rx="3" ry="3" fill="{col}"/>')
        F.text(x+cw/2+3, 234, t, size=16, weight=700)
        yy = 272
        for ln in lines:
            F.text(x+26, yy, "• " + ln, size=12.5, fill=C["slate"], anchor="start")
            yy += 36
        F.badge(x+26, 386, ["الأساس العالمي", "كفاءة على المعاملات", "سياقات هائلة"][k], col, size=10.5)
    # سطر تفصيل الكاش
    F.rrect(F.pad, 442, F.w-2*F.pad, 66, C["amberL"], stroke="#FCD34D", r=12)
    F.text(F.w/2, 464, "MHA vs GQA vs MLA — توفير الذاكرة في الاستدلال:", size=13.5, weight=700, fill=C["amber"])
    F.text(F.w/2 - 250, 490, "MHA: 1×   ", size=12, mono=True, anchor="start")
    F.text(F.w/2 - 100, 490, "GQA (8): 8× توفير", size=12, mono=True, anchor="start")
    F.text(F.w/2 + 90, 490, "MLA: كاش مضغوط مع جودة عالية", size=12, mono=True, anchor="start")
    F.text(F.w/2, 536, "الاختيار الهندسي: FlashAttention + GQA/MLA + تكميم KV هي ذاكرة تكفي CPU شخص واحد", size=12.8, fill=C["mute"])
    F.section_note("الفصل 10 — ما بعد المحوّل الكلاسيكي")
    return F

def f13_quant():
    F = Fig(1040, 520)
    F.text(F.w/2, 34, "سلّم التكميم: دقة مقابل ذاكرة (نموذج 8B تقريبًا)", size=21, weight=700)
    levels = [("FP32", 1.00, C["rose"], "مرجع الدقة الكاملة"),
              ("BF16/FP16", 0.50, C["amber"], "معيار التدريب"),
              ("INT8", 0.25, C["indigo"], "PTQ الشائع"),
              ("FP8 (E5M2)", 0.13, C["violet"], "على H100+"),
              ("INT4/FP4 · QLoRA", 0.08, C["emerald"], "تجارب — بحذر")]
    for k, (t, fr, col, note) in enumerate(levels):
        yy = 92 + k*56
        F.hbar(F.pad+30, yy, 340, fr, f"{t} — {note}", f"~{fr*64:.0f} GB", col, h=16, size=12.3)
    x0 = 520
    F.rrect(x0, 76, F.w - x0 - F.pad, 234, C["panel"], stroke=C["line"], r=14)
    F.text((x0+F.w-F.pad)/2, 104, "أحكام عملية قبل كل درجة", size=14.5, weight=700)
    rules = [
        ("PTQ (بعد التدريب): أسرع وأرخص — قِس perplexity بعد الانتقال", C["indigo"]),
        ("QAT (أثناء التدريب): أدق لكن مكلف — يُنظر فيه من 4 بت فأقل", C["violet"]),
        ("ابدأ دائمًا بـ INT8/FP8 ثم قِس", C["emerald"]),
        ("التكميم العدواني يُفقِد مهارات التفكير في النماذج الصغيرة", C["rose"]),
    ]
    yy = 140
    for t, col in rules:
        F.text(x0+26, yy, "• " + t, size=12.3, anchor="start", fill=C["slate"])
        yy += 30
    F.badge(x0+26, yy+4, "قِس: perplexity + جودة مهمة حقيقية", C["emerald"], size=10.5)
    # ذاكرة KV
    F.rrect(F.pad, 330, F.w-2*F.pad, 114, C["cyanL"], stroke="#0891B2", r=14)
    F.text(F.w/2, 356, "KV cache: رأس التكلفة في السياقات الطويلة", size=14.5, weight=700, fill=C["cyan"])
    F.text(F.w/2, 386, "سياق 128K على نموذج 70B مع GQA — الكاش قد يبلغ ~40GB؛ MLA + تكميم KV يجعلها خدمية على GPU واحدة", size=12.4, fill=C["slate"])
    F.badge(F.w/2-262, 410, "vLLM · PagedAttention", C["cyan"], size=11)
    F.badge(F.w/2-66, 410, "TGI", C["cyan"], size=11)
    F.badge(F.w/2+4, 410, "TensorRT-LLM", C["cyan"], size=11)
    F.badge(F.w/2+166, 410, "Tensor Parallelism", C["cyan"], size=11)
    F.text(F.w/2, 478, "ابدأ صغيرًا: perplexity + تحقق من مهمة حقيقية كافيان لتقرير كل درجة قبل النشر", size=12.8, fill=C["mute"])
    F.section_note("الفصل 11 — تسريع العمليات الحسابية")
    return F

def f14_pipeline():
    F = Fig(1040, 600, 30)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 40, "بناء LLM من الصفر: خط الأنابيب الكامل (الجزء الرابع)", size=21, weight=700)
    stages = [("12", "تخطيط المشروع", "البيانات والزحف", C["cyan"]),
              ("13", "Tokenizer", "BPE العربي 32K", C["indigo"]),
              ("14", "النموذج", "RoPE · SwiGLU · RMSNorm", C["violet"]),
              ("15", "Pre-training", "CPU/GPU · checkpoints", C["emerald"]),
              ("16", "SFT/LoRA", "تعليمات وتوحيد", C["amber"]),
              ("17", "DPO", "تفضيلات دقيقة", C["rose"]),
              ("18", "التوليد", "top-p · temperature", C["indigo"])]
    bw = (F.w - 2*F.pad)/7 - 12
    for k, (n, t, d, col) in enumerate(stages):
        x = F.pad + k*(bw+12)
        F.rrect(x, 92, bw, 110, "white", stroke=col, r=12, sw=1.5)
        F.parts.append(f'<path d="M{x+14},92 h{bw-28} a14,14 0 0 1 14,14 v22 h-{bw} v-22 a14,14 0 0 1 14,-14 z" fill="{col}"/>')
        F.text(x+bw/2, 121, f"CH{n}", size=13, fill="#fff", weight=800, mono=True)
        F.text(x+bw/2, 158, t, size=14, weight=700)
        F.text(x+bw/2, 184, d[:24], size=10.5, fill=C["mute"])
        if k < 6:
            F.line(x+bw, 147, x+bw+11, 147, stroke=C["line2"], sw=1.6, marker="agrA")
    # حلقة التقييم والنشر
    F.rrect(F.pad, 240, F.w-2*F.pad, 92, C["panel"], stroke=C["line"], r=14)
    F.text(F.w/2, 268, "حلقة التقييم (الفصول 19–40): قِس، أصلح، أعد — بلاها النموذج ملف أوزان فقط", size=14, weight=700)
    F.text(F.w/2, 296, "منحنيات التدريب ← حزمة تقييم ← A/B ← واقيات ← مراقبة؛ الحلقة لا تغلق بعد النشر", size=12.5, fill=C["mute"])
    # الشبكة الدنيا: ذنوب هندسيّة
    notes = [
        ("فحص العقل", "overfit دفعة واحدة قبل أي تدريب", C["rose"]),
        ("grad_norm", "تابعه: الانفجار يسبق NaN", C["amber"]),
        ("Checkpoints", "نموذج+مُحسّن+جدول+RNG معًا", C["indigo"]),
        ("عربية أولًا", "Tokenizer خاص = توفير 30–50%", C["violet"]),
    ]
    cw = (F.w-2*F.pad-60)/4
    for k, (t, d, col) in enumerate(notes):
        x = F.pad + k*(cw+20)
        F.rrect(x, 352, cw, 96, "white", stroke=col, r=12)
        F.text(x+cw/2, 380, t, size=14, weight=700, fill=col)
        F.text(x+cw/2, 410, d, size=11.4, fill=C["slate"])
    F.rrect(F.pad, 492, F.w-2*F.pad, 74, C["emeraldL"], stroke="#34D399", r=14)
    F.text(F.w/2, 520, "نموذج مصغّر (10–124M) على GPU واحدة = أسرع طريق لفهم الظواهر كلها قبل 70B", size=14.2, weight=700, fill=C["emerald"])
    F.text(F.w/2, 548, "لا تدرّب كبيرًا قبل أن تشعر — كل محوّل صغير يعلمك المفاهيم نفسها", size=12.2, fill=C["mute"])
    F.section_note("الجزء الرابع — خريطة البناء")
    return F

def f15_mini_llm():
    F = Fig(1040, 600, 40)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 36, "MiniLLaMA (نمط LLaMA مصغّر): لكل قطعة سبب وجودها", size=21, weight=700)
    # عمود المعمارية
    x0, w0 = 96, 420
    F.rrect(x0, 80, w0, 56, C["cyanL"], stroke="#0891B2", r=12)
    F.text(x0+w0/2, 104, "Token Embedding", size=15, weight=700, fill=C["cyan"])
    F.text(x0+w0/2, 126, "وحدات ← متجهات (بلا سياق بعد)", size=11.5, fill=C["slate"])
    F.rrect(x0, 152, w0, 252, "white", stroke=C["indigo"], r=14, sw=1.6)
    F.text(x0+w0/2, 178, "N × Transformer Block", size=15.5, weight=700, fill=C["indigo"])
    stack = [("RMSNorm", C["violet"]), ("Multi-Head Attention + RoPE", C["indigo"]), ("Residual +", C["emerald"]),
             ("RMSNorm", C["violet"]), ("FFN (SwiGLU · d_ff ≈ 2.7×d)", C["amber"]), ("Residual +", C["emerald"])]
    yy = 208
    for t, col in stack:
        F.rrect(x0+32, yy, w0-64, 28, C["panel"], stroke=col, r=7)
        F.text(x0+w0/2, yy+15, t, size=12.5, weight=600, fill=col)
        yy += 32
    F.rrect(x0, 416, w0, 56, C["violetL"], stroke="#A78BFA", r=12)
    F.text(x0+w0/2, 440, "Final RMSNorm + LM Head", size=15, weight=700, fill=C["violet"])
    F.text(x0+w0/2, 462, "وزن مشارَك مع تضمين الدخل (weight tying)", size=11.5, fill=C["slate"])
    # الجواب المتوطن يمين
    x1 = 570
    F.rrect(x1, 92, F.w - x1 - F.pad, 180, C["panel"], stroke=C["line"], r=14)
    F.text((x1+F.w-F.pad)/2, 118, "لماذا هذه القطع تحديدًا؟", size=15, weight=700)
    chips = [("RoPE", "موضع نسبي، سياق قابل للتوسع", C["indigo"]),
             ("RMSNorm", "أسرع من LayerNorm", C["violet"]),
             ("SwiGLU", "تنشيط سلس ببوابة", C["amber"]),
             ("Weight Tying", "أوفر في المعاملات", C["emerald"])]
    yy = 146
    for t, d, col in chips:
        F.badge(x1+26, yy, t, col, size=11.5)
        F.text(x1+150, yy+12, d, size=12, fill=C["slate"], anchor="start")
        yy += 30
    F.rrect(x1, 292, F.w-x1-F.pad, 150, "white", stroke=C["line"], r=14)
    F.text((x1+F.w-F.pad)/2, 318, "خط تكوين صغير (config)", size=14, weight=700)
    cfg = "MiniLLM(vocab=32k, d=512, layers=12, heads=8)"
    F.text(x1+30, 352, cfg[:46], size=14, mono=True, anchor="start", fill=C["slate"])
    F.rrect(x1+30, 372, F.w-x1-F.pad-60, 26, C["panel2"], r=7)
    F.text(x1+42, 386, "parameters: 16.4M (embed+head مربوطان) + 38.5M (12 طبقة)", size=11, mono=True, anchor="start", fill=C["mute"])
    F.badge(x1+30, 410, "≈ 55M معاملًا", C["indigo"], size=11)
    # صندوق الخطر
    F.rrect(F.pad, 492, F.w-2*F.pad, 86, C["roseL"], stroke="#FDA4AF", r=14)
    F.text(F.w/2, 518, "الأخطاء الثلاثة في الكود، لا في المفهوم", size=14.5, weight=700, fill=C["rose"])
    F.text(F.w/2, 546, "1. نسيان القناع السببي   2. خسارة على PAD (دون ignore_index=-100)   3. need_weights=True أثناء التدريب (مكلف بلا فائدة)", size=12.6, fill=C["slate"])
    F.text(F.w/2, 570, "تحقّق: شكل المخرجات أولًا، ثم forward كامل، ثم جرعة التدريب الفعلية", size=12.2, fill=C["mute"])
    F.section_note("الفصل 14 — بناء النموذج")
    return F

def f16_trainloop():
    F = Fig(1040, 560)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 36, "حلقة التدريب المسبق: سبع خطوات لا تغادر", size=21, weight=700)
    steps = [("batch", C["cyan"]), ("forward", C["indigo"]), ("CE loss (ignore PAD)", C["amber"]),
             ("backward", C["indigo"]), ("clip 1.0", C["rose"]), ("AdamW step", C["emerald"]), ("sched.step()", C["violet"])]
    bw = (F.w-2*F.pad)/7 - 12
    for k, (t, col) in enumerate(steps):
        x = F.pad + k*(bw+12)
        F.rrect(x, 100, bw, 56, "white", stroke=col, r=10, sw=1.5)
        F.parts.append(f'<rect x="{x}" y="100" width="7" height="56" rx="3.5" ry="3.5" fill="{col}"/>')
        F.text(x+bw/2+3, 134, t[:22], size=12.6, weight=600, mono=True if "loss" not in t else False)
        if k < 6:
            F.line(x+bw+2, 128, x+bw+10, 128, stroke=C["line2"], sw=1.5, marker="agrA")
    # حساب عود التدريب
    F.curve(f"M {F.w-F.pad-40} 128 C {F.w-F.pad-40} 210, {F.pad+40} 210, {F.pad+40} 130",
            stroke=C["indigo"], sw=1.7, dash="6 5", marker="agrA")
    F.text(F.w/2, 196, "كل خطوة تعود للبداية — الخسارة تنقص بالصبر لا بالمعجزات", size=12.5, fill=C["mute"])
    # جدول LR Scheduler
    x0, y0 = 90, 260
    F.rrect(x0-20, y0-20, 560, 200, C["white"], stroke=C["line"], r=14)
    F.text(x0+240, y0+6, "جدول معدل التعلم: Warmup ثم Cosine", size=14.5, weight=700)
    ox, oy, ow, oh = x0+40, y0+140, 460, 90
    F.line(ox, oy, ox+ow, oy, stroke=C["line2"], sw=1.2)
    F.line(ox, oy, ox, oy-oh-10, stroke=C["line2"], sw=1.2)
    pts = []
    max_steps, max_lr, warm = 1000, 1.0, 100
    for s in range(0, max_steps+1, 10):
        u = s/max_steps
        if s < warm:
            lr = max_lr*s/warm
        else:
            import math as mm
            lr = 0.02 + 0.98*0.5*(1+mm.cos(mm.pi*(s-warm)/(max_steps-warm)))
        pts.append((ox+ow*u, oy-min(1.0, lr)*oh))
    d = "M" + " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    F.curve(d, stroke=C["emerald"], sw=2.2)
    F.line(ox+ow*warm/max_steps, oy-4, ox+ow*warm/max_steps, oy-oh-8, stroke=C["line2"], sw=1, dash="4 3")
    F.text(ox+ow*warm/max_steps+8, oy-oh-6, "نهاية warmup", size=10.5, fill=C["mute"], anchor="start")
    F.text(ox, oy-oh-14, "LR", size=10.5, mono=True, fill=C["mute"], anchor="start")
    F.text(ox+ow, oy+16, "steps", size=10.5, mono=True, fill=C["mute"], anchor="start")
    # بطاقة الإحصاءات السفلية
    x1 = 660
    F.rrect(x1, y0-20, F.w - x1 - F.pad, 200, C["panel"], stroke=C["line"], r=14)
    F.text((x1+F.w-F.pad)/2, y0+6, "لوحة المراقبة المباشرة", size=14.5, weight=700)
    stats = [("loss", "0.89 → 0.31", C["indigo"]), ("grad_norm", "0.6 (قمة 2.1)", C["emerald"]),
             ("lr", " Scheduler 3e-4", C["violet"]), ("tokens/s", "41.2K", C["cyan"])]
    yy = y0 + 42
    for t, v, col in stats:
        F.text(x1+34, yy, t, size=12.5, mono=True, anchor="start", fill=col)
        F.text(F.w - F.pad - 34, yy, v, size=12.5, mono=True, anchor="end", fill=C["slate"])
        yy += 34
    F.badge(x1+30, y0+150, "قاعدة ذهبية: لا تثق بالرقم الأخير وحده", C["amber"], size=10.5)
    F.rrect(F.pad, 480, F.w-2*F.pad, 62, C["amberL"], stroke="#FCD34D", r=12)
    F.text(F.w/2, 504, "نصيحة تجربة: ارفع LR قصدًا لترى الانفجار مرة — بعدها تتعرف على grad_norm في السجل مباشرة", size=13, weight=600, fill=C["amber"])
    F.section_note("الفصل 15 — حلقة التدريب")
    return F

def f17_lora():
    F = Fig(1040, 540)
    F.arrows_bold("agrA")
    F.text(F.w/2, 34, "LoRA: زيادة صغيرة فوق أوزان جامدة", size=21, weight=700)
    # الشيفرة يمين: نموذج فريزين + LoRA
    x0, y0 = 110, 120
    F.rrect(x0, y0, 430, 84, C["panel"], stroke=C["indigo"], r=14, sw=1.5)
    F.text(x0+22, y0+34, "أوزان النموذج W (مجمّدة)", size=14.5, weight=700, fill=C["slate"], anchor="start")
    F.text(x0+22, y0+62, "W: d×k — مليارات؛ مثبّتة لا تدرّب", size=11.5, mono=True, fill=C["mute"], anchor="start")
    # مقاطع LoRA
    y0b = y0 + 104
    F.rrect(x0, y0b, 430, 104, C["emeraldL"], stroke="#34D399", r=14)
    F.text(x0+22, y0b+30, "تحديث LoRA: ΔW = B·A (رتبة r صغيرة)", size=14.5, weight=700, fill=C["emerald"], anchor="start")
    F.rrect(x0+40, y0b+52, 118, 40, "white", stroke=C["emerald"], r=8)
    F.text(x0+99, y0b+72, "A: r×k", size=13, mono=True, weight=700, fill=C["emerald"])
    F.line(x0+158, y0b+72, x0+196, y0b+72, stroke=C["emerald"], sw=1.6, marker="agrA")
    F.rrect(x0+200, y0b+52, 118, 40, "white", stroke=C["emerald"], r=8)
    F.text(x0+259, y0b+72, "B: d×r", size=13, mono=True, weight=700, fill=C["emerald"])
    F.badge(x0+330, y0b+58, "r = 8..64", C["emerald"], size=10.5)
    # الدمج
    y0c = y0b + 122
    F.rrect(x0, y0c, 430, 64, C["violetL"], stroke="#A78BFA", r=14)
    F.text(x0+22, y0c+24, "بعد التدريب: W ← W + ΔW (إذابة في القاعدة)", size=14, weight=700, fill=C["violet"], anchor="start")
    F.text(x0+22, y0c+47, "بلا زيادة في زمن الاستدلال عند الدمج", size=11.8, fill=C["slate"], anchor="start")
    # فروق الذاكرة
    x1 = 600
    F.rrect(x1, y0, F.w - x1 - F.pad, 288, C["white"], stroke=C["line"], r=14)
    F.text((x1+F.w-F.pad)/2, y0+30, "ماذا توفرو LoRA؟", size=15, weight=700)
    rows = [("أوزان قابلة للتدريب", "100%", "0.1–1%", C["rose"], C["emerald"]),
            ("ذاكرة التدريب (GPU)", "48–80GB", "12–24GB", C["rose"], C["emerald"]),
            ("زمن التدريب", "أيام/أسابيع", "ساعات", C["rose"], C["emerald"]),
            ("جودة المهمة", "مرجع", "قريبة جدًا", C["rose"], C["emerald"])]
    yy = y0 + 66
    for t, a, b, ca, cb in rows:
        F.text(x1+28, yy, t, size=13, weight=650, anchor="start")
        F.text(F.w - F.pad - 150, yy, a, size=12, mono=True, anchor="end", fill=ca)
        F.text(F.w - F.pad - 30, yy, b, size=12, mono=True, anchor="end", fill=cb)
        F.line(x1+24, yy+12, F.w-F.pad-24, yy+12, stroke=C["line"], sw=0.9, opacity=0.7)
        yy += 42
    F.text(x1+28, yy-12, "استخدام QLoRA؟ الأوزان الأساسية INT4", size=11.5, fill=C["mute"], anchor="start")
    # بطاقات الأدوات
    F.rrect(F.pad, 416, F.w-2*F.pad, 88, C["cyanL"], stroke="#0891B2", r=14)
    F.badge(F.pad+30, 440, "peft", C["cyan"], size=12)
    F.text(F.pad+110, 450, "PEFT API — سطران للتدريب:", size=12.5, mono=True, anchor="start")
    F.text(F.pad+110, 478, "get_peft_model(model, LoraConfig(r=16, target_modules=[\"q_proj\",\"v_proj\"]))", size=12.5, mono=True, anchor="start", fill=C["slate"])
    F.badge(F.w - F.pad - 232, 440, "QLoRA: 4-bit + LoRA", C["amber"], size=11)
    F.badge(F.w - F.pad - 84, 440, "merge", C["violet"], size=11)
    F.text(F.w/2, 522, "الخطأ الشائع: فتح كل الطبقات على بيانات صغيرة (overfit) — بقّص الهدف إلى 30% من الطبقات أولًا", size=12.6, fill=C["mute"])
    F.section_note("الفصل 16 — SFT وLoRA/QLoRA (التخصيص)")
    return F

def f18_dpo():
    F = Fig(1040, 520)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 34, "من PPO إلى DPO: نفس الهدف، نصف العتاد", size=21, weight=700)
    # PPO
    x0 = 60
    F.rrect(x0, 96, 420, 300, C["roseL"], stroke="#FDA4AF", r=16)
    F.text(x0+210, 126, "PPO (الكلاسيكي)", size=17, weight=800, fill=C["rose"])
    boxes = [("سياسة Policy", C["rose"]), ("مرجع Reference", C["amber"]), ("مكافأة Reward", C["indigo"]), ("قيمة Value", C["violet"])]
    for k, (t, col) in enumerate(boxes):
        xx = x0+40+(k%2)*200
        yy = 170+(k//2)*74
        F.rrect(xx, yy, 170, 54, "white", stroke=col, r=10, sw=1.5)
        F.text(xx+85, yy+28, t, size=13.5, weight=700)
    for k, _ in enumerate(boxes[:3]):
        F.line(x0+40+170, 197+(k//2)*74, x0+220, 197+(k//2)*74, stroke=C["line2"], sw=1.4, marker=None) if k in (0,1) else None
    F.text(x0+210, 370, "4 نماذج متزامنة — بنية تحتية معقدة وضبط حساس لـ KL", size=12.2, fill=C["rose"])
    # DPO
    x1 = 520
    F.rrect(x1, 96, 460, 300, C["emeraldL"], stroke="#34D399", r=16)
    F.text(x1+230, 126, "DPO (المباشر)", size=17, weight=800, fill=C["emerald"])
    for k, (t, col) in enumerate([("سياسة Policy", C["emerald"]), ("مرجع Reference", C["amber"])]):
        xx = x1+50+(k%2)*230
        yy = 170+(k//2)*84
        F.rrect(xx, yy, 200, 56, "white", stroke=col, r=10, sw=1.5)
        F.text(xx+100, yy+28, t, size=13.5, weight=700)
    F.rrect(x1+40, 260, 380, 96, "white", stroke=C["emerald"], r=12, dash="5 3", sw=1.3)
    F.text(x1+230, 286, "خسارة مغلقة الصيغة:", size=13, weight=700, fill=C["emerald"])
    F.text(x1+230, 316, "-log σ( β·[logπ/πref(w) − logπ/πref(l)] )", size=12.2, mono=True, fill=C["emerald"])
    F.text(x1+230, 342, "دون مكافأة، دون عينات أثناء التدريب", size=11.5, fill=C["slate"])
    # المقارنة السفلية
    F.rrect(F.pad, 424, F.w-2*F.pad, 76, C["panel"], stroke=C["line"], r=14)
    F.text(F.w/2, 450, "التوجه في بيئة الإنتاج (2025–2026): SFT قوي ثم DPO — PPO في حالات خاصة بمكافآت برمجية قابلة للتحقق (GRPO)", size=13, weight=600)
    F.text(F.w/2, 480, "ابدأ DPO بلا SFT مسبق؟ مخاطرة: قد يرتفع احتمال المرفوض — جهّز SFT أولًا، وتدرّج في β", size=12, fill=C["mute"])
    F.section_note("الفصل 17 — ضبط التفضيلات")
    return F
