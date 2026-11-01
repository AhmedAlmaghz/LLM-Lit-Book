# -*- coding: utf-8 -*-
"""المخططات التوضيحية (1/2): الأساسيات، التضمينات، المحوّلات"""
from figurekit import Fig, C
import re
import math


def f01_neuron():
    F = Fig(1040, 540, 30)
    F.arrows_bold("agrA", C["indigo"]); F.arrow_def("agrB", "#E11D48")
    F.text(F.w/2, 36, "الخلية العصبية: من الأرقام إلى القرار", size=21, weight=700)
    # مدخلات
    ins = ["سحب كثيفة", "رطوبة عالية", "هبوط ضغط", "رياح شرقية"]
    for k, t in enumerate(ins):
        x, y = 40, 90 + k * 74
        F.rrect(x, y, 176, 46, C["indigoL"], stroke="#6366F1", r=10)
        F.text(x + 88, y + 24, t, size=14, fill=C["indigo"], weight=600)
        F.text(x + 190, y + 24, "×w" + str(k+1), size=13, mono=True, fill=C["mute"], anchor="start")
        F.line(x + 184, y + 23, 414, 246, stroke="#94A3B8", sw=1.5, marker="agrA")
    # المعالجة
    x0 = 466
    F.grad("gSum", "#6366F1", "#4338CA")
    F.circle(x0, 246, 46, fill="url(#gSum)", stroke="#4F46E5", sw=2)
    F.text(x0, 238, "Σ", size=30, weight=700, fill="#fff")
    F.text(x0, 264, "z = Σw·x + b", size=10.5, mono=True, fill="#fff")
    F.line(x0 + 46, 246, x0 + 108, 246, stroke="#94A3B8", sw=1.8, marker="agrA")
    F.rrect(x0 + 112, 218, 128, 56, C["violetL"], stroke="#7C3AED", r=12)
    F.text(x0 + 176, 242, "تنشيط σ", size=15, weight=700, fill=C["violet"])
    F.text(x0 + 176, 262, "ReLU / GELU / SiLU", size=10.5, mono=True, fill=C["violet"])
    F.line(x0 + 240, 246, x0 + 298, 246, stroke="#94A3B8", sw=1.8, marker="agrA")
    F.grad("gOut", "#059669", "#047857")
    F.rrect(x0 + 302, 222, 128, 48, "url(#gOut)", stroke="none", r=12)
    F.text(x0 + 366, 240, "a", size=16, mono=True, fill="#fff", weight=700)
    F.text(x0 + 366, 258, "التنبأ", size=11.5, fill="#fff")
    F.line(x0 + 432, 246, x0 + 452, 246, stroke="#94A3B8", sw=1.8, marker="agrA")
    F.rrect(x0 + 456, 222, 112, 48, C["white"], stroke=C["line"], r=10)
    F.text(x0 + 512, 250, "«الليلة مطر»", size=14, weight=700, anchor="middle")
    # الرجوع العكسي
    F.curve("M 995 310 C 740 490, 320 480, 150 402", stroke=C["rose"], sw=2.2, dash="7 5", marker="agrB")
    F.rrect(300, 416, 392, 40, C["roseL"], stroke="#FB7185", r=20)
    F.text(496, 437, "الانتشار العكسي: خطأ التنبأ يعود لتصحيح الأوزان", size=13, weight=600, fill=C["rose"])
    F.text(F.w - 24, 470, "التعلّم = تقدير منصف للأوزان يختفض خطأ التنبأ", size=12.5, fill=C["mute"], anchor="end", opacity=0.9)
    F.badge(604, F.h - 40, "الانتشار الأمامي", C["indigo"])
    F.badge(768, F.h - 40, "الانتشار العكسي", C["rose"])
    F.section_note("الفصل 1 — الشبكات العصبية من الصفر")
    return F

def f02_backprop():
    F = Fig(1040, 520)
    F.arrows_bold("agrF", C["indigo"]); F.arrow_def("agrBk", "#E11D48")
    F.text(F.w/2, 34, "الرسم البياني الحسابي لقاعدة السلسلة", size=21, weight=700)
    nodes = [("x", C["cyan"]), ("z1 = W1·x + b1", C["slate"]), ("a1 = σ(z1)", C["slate"]),
             ("z2 = W2·a1 + b2", C["slate"]), ("a2 = σ(z2)", C["slate"]), ("L", C["rose"])]
    xk = F.pad + 20
    positions = []
    bw = [46, 176, 140, 190, 140, 56]
    gaps = [50, 60, 66, 66, 60]
    y0 = 150
    xx = xk
    for k, (t, col) in enumerate(nodes):
        w = bw[k]
        fill = C["roseL"] if col == C["rose"] else ("white")
        F.rrect(xx, y0 - 22, w, 44, C["roseL"] if t == "L" else "white", stroke=col, r=10, sw=1.7)
        F.text(xx + w/2, y0, t, size=14.5, mono=re.match(r"^[xzaAL]", t) is not None and len(t) < 12, weight=600)
        positions.append((xx, w))
        if k < len(gaps):
            xx += w + gaps[k]
    for k in range(5):
        sx = positions[k][0] + positions[k][1] + 4
        ex = positions[k+1][0] - 4
        F.line(sx, y0, ex, y0, stroke=C["indigo"], sw=1.7, marker="agrF")
    F.badge(660, y0 - 66, "الاتجاه الأمامي (تدفق الحساب)", C["indigo"], mono=False)
    # تدفق عكسي تحت
    y0b = 330
    d = ["∂L/∂a2", "∂L/∂z2 = ∂L/∂a2 · σ′(z2)", "∂L/∂W2", "∂L/∂a1", "∂L/∂z1", "∂L/∂W1", "∂L/∂b1"]
    xx = positions[5][0]
    seq = [(5, 0), (4, 1), (4, 2), (3, 3), (2, 4), (1, 5), (1, 6)]
    sy = y0b
    for (ni, li) in seq:
        F.text(positions[ni][0] + positions[ni][1]/2, sy, d[li], size=12.5, mono=True, fill=C["rose"])
    F.curve("M " + str(positions[5][0]+28) + f" {y0-22} C {positions[5][0]} {y0+90}, {positions[5][0]} {y0+220}, {positions[5][0]-4} {y0b-16}",
            stroke=C["rose"], sw=1.7, dash="5 4", marker="agrBk")
    for k in range(5, 0, -1):
        x1 = positions[k][0] - 6
        x2 = positions[k-1][0] + positions[k-1][1] + 2
        F.curve(f"M {x1} {y0b} C {(x1+x2)//2} {y0b+52}, {(x1+x2)//2} {y0b+52}, {x2} {y0b}",
                stroke=C["rose"], sw=1.5, marker="agrBk", opacity=0.85)
    F.badge(240, y0b + 76, "القاعدة: ∂L/∂w = (∂L/∂a)·(∂a/∂z)·(∂z/∂w) — طبقة تلو الأخرى نحو البداية", C["rose"], size=12)
    F.text(F.w/2, 454, "التدرج يسافر عكسيًا عبر نفس سلسلة العقد: كل عقدة تمرّر الفرق إلى من سبقها", size=13.5, fill=C["mute"])
    F.section_note("الفصل 1 — الانتشار العكسي")
    return F

def f03_curves():
    F = Fig(1040, 560)
    F.text(F.w/2, 34, "منحنيات التدريب: خمسة أنماط تُقرأ قبل الأرقام", size=21, weight=700)
    shapes = [
        ("توازُن مثالي", C["emerald"], "indigo", "emerald"),
        ("حفظ زائد (Overfit)", C["rose"], "indigo", "rose"),
        ("نقص تعلّم (Underfit)", C["amber"], "indigo", "amber"),
        ("معدل تعلّم كبير", C["rose"], "indigo", "rose"),
        ("معدل تعلّم صغير", C["cyan"], "indigo", "cyan"),
    ]
    # شبكة 3+2
    pw, ph, gap = 300, 190, 30
    for k, (title, col, tcol, kcol) in enumerate(shapes):
        xx = F.pad + (k % 3) * (pw + gap)
        yy = 90 + (k // 3) * (ph + 46)
        F.rrect(xx, yy, pw, ph, C["white"], stroke=C["line"], r=12)
        F.text(xx + pw/2, yy + 24, title, size=15, weight=700)
        ox, oy = xx + 46, yy + 52   # محاور
        oxw, oyh = pw - 70, ph - 92
        F.line(ox, oy, ox + oxw, oy, stroke=C["line2"], sw=1.2)
        F.line(ox, oy, ox, oy + oyh, stroke=C["line2"], sw=1.2)
        def path(fnf, color, delta=0):
            d = "M"
            for p in range(0, 41):
                u = p / 40
                v = max(0.0, min(1.0, fnf(u) / 3.0))
                x, yv = ox + oxw * u, oy + oyh * (1 - v)
                d += f" {x:.1f},{yv:.1f} " if p else f" {x:.1f},{yv:.1f} "
            F.curve(d, stroke=color, sw=2.1, opacity=0.95)
        if k == 0:
            path(lambda u: 2.4*math.exp(-2.4*u)+0.16, C["indigo"])
            path(lambda u: 2.4*math.exp(-2.4*u)+0.16 + 0.16 - 0.14*u, C["emerald"])
        elif k == 1:
            path(lambda u: 2.2*math.exp(-2.9*u)+0.10, C["indigo"])
            path(lambda u: max(0.05, 2.2*math.exp(-2.6*u)+0.32 - 0.0*(u<0.55) + (2.9*(u-0.55))**2*0.33*(u>0.55)), C["rose"])
        elif k == 2:
            path(lambda u: 0.68 - 0.12*u, C["indigo"])
            path(lambda u: 0.72 - 0.11*u, C["amber"])
        elif k == 3:
            pts = [0.3, 1.9, 0.6, 2.2, 0.9, 1.4, 1.1, 1.0, 1.6, 1.2]
            d = "M"
            for i, v in enumerate(pts):
                d += f" {ox + oxw*i/(len(pts)-1):.1f},{oy + oyh*max(0.03, 1.22 - v*0.5):.1f} "
            F.curve(d, stroke=C["rose"], sw=2.1)
        else:
            path(lambda u: 2.4*math.exp(-0.65*u)+0.16, C["indigo"])
            path(lambda u: 2.4*math.exp(-0.58*u)+0.3 - 0.16*u, C["cyan"])
        F.text(ox + oxw + 12, oy + oyh + 16, "Epochs", size=10.5, mono=True, fill=C["mute"])
        F.text(ox - 10, oy - 10, "Loss", size=10.5, mono=True, fill=C["mute"], anchor="start")
    # وسيلة إيضاح
    F.legend(760, 300, [("خسارة التدريب", C["indigo"]), ("خسارة التحقق", C["emerald"])])
    F.text(F.w/2, 532, "المنحنى يحكي القصة قبل أن يتكلم الرقم النهائي — الفجوة بين الكفّتين هي إشارتك", size=13.5, fill=C["mute"])
    F.section_note("الفصل 2 — قراءة منحنيات التدريب")
    return F

def f04_regularizers():
    F = Fig(1040, 480)
    F.text(F.w/2, 34, "التنظيم (Regularization): أدوات منع الحفظ الأعمى", size=21, weight=700)
    items = [
        ("Dropout", "تعطيل عشوائي 10–30% من الخلايا في كل خطوة تدريب", C["indigo"], "#6366F1", "شبكة مختلفة كل خطوة — قوة مع إسقاط"),
        ("Weight Decay", "عقوبة λΣw² تفضّل أوزانًا صغيرة وتُنعّم التعميم؛ عبر AdamW", C["emerald"], "#34D399", "ثبات السلوك ومقاومة الاهتزاز"),
        ("Early Stopping", "راقب خسارة التحقق وتوقّف عند patience خطوات مع حفظ أفضل نقطة", C["amber"], "#FBBF24", "أبسط تنظيم وأقوى أثر — بلا كلفة"),
        ("بيانات أكثر", "Augmentation أو بيانات حقيقية إضافية — أقوى علاج عملي", C["cyan"], "#22D3EE", "لا بديل عنها أبدًا"),
    ]
    for k, (h, d, col, col2, note) in enumerate(items):
        xx = F.pad + (k % 2) * (F.w - 2*F.pad - 20) / 1.0 if False else (F.pad + (k % 2) * 490)
        yy = 82 + (k // 2) * 172
        F.rrect(xx, yy, 470, 150, C["white"], stroke=C["line"], r=14)
        F.parts.append(f'<rect x="{xx}" y="{yy}" width="6" height="150" rx="3" ry="3" fill="{col}"/>')
        F.parts.append(f'<circle cx="{xx+54}" cy="{yy+52}" r="26" fill="{col}18" stroke="{col}" stroke-width="1.6"/>')
        # أيقونة مبسّطة
        if k == 0:
            for i, (a, b) in enumerate([(-11, -7), (2, -3), (12, 4)]):
                F.circle(xx + 54 + a, yy + 52 + b, 7, "white", stroke=col, sw=1.6)
            F.line(xx + 42, yy + 62, xx + 68, yy + 36, stroke=C["rose"], sw=1.8, marker=None)
        elif k == 1:
            F.parts.append(f'<path d="M{xx+42} {yy+52} l6 0 l4 -8 l6 16 l6 -16 l6 16 l6 -16 l4 8 l6 0" fill="none" stroke="{col}" stroke-width="1.8"/>')
        elif k == 2:
            d2 = "M"
            for i, u in enumerate(range(0, 30)):
                t = u/29
                d2 += f" {xx+40+t*30:.1f},{yy+60 - 22*math.exp(-2.2*t)+6*t:.1f} "
            F.curve(d2, stroke=col, sw=1.8)
        else:
            F.rrect(xx+40, yy+40, 30, 24, f"{col}20", stroke=col, r=4)
            F.rrect(xx+48, yy+34, 30, 24, "white", stroke=col, r=4)
        F.text(xx + 96, yy + 52, h, size=17.5, weight=700, fill=col, anchor="start")
        F.text(xx + 96, yy + 86, d, size=13.5, fill=C["slate"], anchor="start")
        F.text(xx + 96, yy + 118, note, size=12, fill=C["mute"], anchor="start")
    F.text(F.w/2, 450, "قاعدة عملية: اجمع تنظيمين قبل أن تفكر في نموذج أكبر", size=13.5, fill=C["mute"])
    F.section_note("الفصل 2 — التنظيم")
    return F

def f05_amp():
    F = Fig(1040, 560)
    F.arrows_bold("agrA")
    F.text(F.w/2, 34, "بنية التدريب الحديثة: دقة مختلطة + تراكم + توزيع", size=21, weight=700)
    # جانب الذاكرة: FP16/BF16
    stages = [
        ("Batch", "دفعات صغيرة", C["cyan"]),
        ("autocast(BF16)", "نسخ مؤقت للأوزان", C["violet"]),
        ("Forward", "حساب بدقة منخفضة", C["indigo"]),
        ("Backward", "FP16: scale× · BF16 مباشر", C["indigo"]),
        ("unscale + clip", "قص التدرجات قبل الخطوة", C["amber"]),
        ("Optimizer (FP32)", "تحديث FP32 عالي الدقة", C["emerald"]),
    ]
    bw = (F.w - 2*F.pad)/6 - 14
    for k, (t, d, col) in enumerate(stages):
        x = F.pad + k*(bw+14)
        F.rrect(x, 104, bw, 74, "white", stroke=col, r=10, sw=1.5)
        F.parts.append(f'<rect x="{x}" y="104" width="7" height="74" rx="3.5" ry="3.5" fill="{col}"/>')
        F.text(x+bw/2+3, 128, t, size=13, weight=700)
        F.text(x+bw/2+3, 152, d[:26], size=11, fill=C["mute"])
    # خط الفواصل (فجوات فقط — لا تعبر الصناديق)
    for k in range(5):
        x1 = F.pad + k*(bw+14) + bw + 2
        F.line(x1, 128, x1+10, 128, stroke=C["line2"], sw=1.6, marker="agrA")
    # شريط الذاكرة
    F.text(70, 234, "الذاكرة لكل GPU (نموذج 8B):", size=13, fill=C["slate"], anchor="start")
    F.hbar(70, 258, 300, 1.00, "FP32 كامل", "~64 GB", C["rose"])
    F.hbar(70, 300, 300, 0.5, "Mixed Precision (BF16/افتراضية)", "~32 GB", C["indigo"])
    F.hbar(70, 342, 300, 0.24, "Gradient Accumulation (نفس الدفعة الفعّالة)", "~16 GB", C["amber"])
    # منصّة التوزيع
    x0 = 430
    F.rrect(x0, 226, F.w - x0 - F.pad, 150, C["panel"], stroke=C["line"], r=14)
    F.text((x0 + F.w - F.pad)/2, 250, "التدريب الموزع", size=15.5, weight=700)
    for k in range(4):
        gxx = x0 + 44 + k*98
        F.rrect(gxx, 276, 76, 52, C["white"], stroke=C["indigo"], r=8, sw=1.4)
        F.text(gxx+38, 296, f"GPU {k}", size=12, weight=700, mono=True)
        F.text(gxx+38, 314, "8 GB" if k else "8 GB", size=10.5, mono=True, fill=C["mute"])
        if k < 3:
            F.line(gxx+76, 302, gxx+96, 302, stroke=C["line2"], sw=1.4, marker="agrA")
    F.badge(x0 + 10, 348, "DDP: نسخ متطابقة وall-reduce", C["indigo"], size=11)
    F.badge(x0 + 300, 348, "FSDP: تقسيم الأوزان عبر العتاد", C["violet"], size=11)
    # توصية سفلية
    F.rrect(F.pad, 392, F.w - 2*F.pad, 100, C["emeraldL"], stroke="#34D399", r=14)
    F.text(F.w/2, 418, "وصفة المحترف 2026", size=15, weight=700, fill=C["emerald"])
    F.text(F.pad + 30, 446, "BF16 افتراضيًا + clip_grad_norm=1.0 بين backward وstep", size=13, anchor="start", fill=C["slate"])
    F.text(F.pad + 30, 468, "تراكم التدرجات لتكبير الدفعة الفعّالة", size=13, anchor="start", fill=C["slate"])
    F.text(F.w - F.pad - 30, 446, "«grad_norm» مؤشر حيوي بعد الخسارة", size=13, anchor="end", fill=C["emerald"])
    F.text(F.w - F.pad - 30, 468, "انفجار مفاجئ = إنذار أحمر فوري", size=13, anchor="end", fill=C["emerald"])
    F.section_note("الفصل 3 — بنية التدريب الحديثة")
    return F

def f06_bpe():
    F = Fig(1040, 560)
    F.arrows_bold("agr")
    F.text(F.w/2, 34, "BPE: من بايتات حقيقة إلى وحدات ذات معنى", size=21, weight=700)
    # جهة اليسار: خط الدمج
    x0 = 70
    F.rrect(40, 84, 400, 380, C["panel"], stroke=C["line"], r=14)
    F.text(240, 112, "خط الدمج على «low, lower, lowest»", size=14.5, weight=700)
    rows = [("l o w │ l o w e r │ l o w e s t", "انطلاق: حروف"), ("l o w │ l o w e r │ l o w e s t", "زوج الأكثر تكرارًا: l o"),
            ("lo w │ lo w e r │ lo w e s t", "دمج l o → lo"), ("low │ low e r │ low e s t", "دمج lo w → low"),
            ("low │ low e r │ low e s t", "دمج e r → er"), ("low │ lower │ lowest", "النهاية حسب حجم المفردات")]
    for k, (t, note) in enumerate(rows):
        yy = 148 + k*52
        F.text(x0+8, yy, t, size=13.6, mono=True, fill=C["ink"], anchor="start")
        F.text(x0+8, yy+20, note, size=11.5, fill=C["mute"], anchor="start")
        if k < len(rows)-1:
            F.curve(f"M {x0+220} {yy+30} C {x0+230} {yy+42}, {x0+230} {yy+42}, {x0+220} {yy+50}", stroke=C["indigo"], sw=1.6, marker="agr")
    F.badge(x0+120, 440, "يعيد الدمج حتى استيفاء حجم المفردات", C["indigo"], size=11)
    # جهة اليمين: نتيجة على نص عربي
    xr = 480
    F.rrect(xr, 84, F.w - xr - F.pad, 380, C["indigoL"], stroke="#6366F1", r=14)
    F.text((xr + F.w - F.pad)/2, 112, "على نص عربي حقيقي (توضيحي)", size=14.5, weight=700, fill=C["indigo"])
    seqs = ["البرمجة", "فاعلموه", "متعدد الوسائط"]
    seqs_t = [["ال", "برم", "جة"], ["ف", "اعلمو", "ه"], ["متعدد", "ال", "وسائط"]]
    yy = 158
    for s, toks in zip(seqs, seqs_t):
        F.text(xr+24, yy, s, size=13, fill=C["ink"], anchor="start", weight=600)
        cx = xr + (F.w - xr - F.pad) - 28
        for tk in toks:
            wid = 12 + 11 * max(2, len(tk))
            cx -= wid
            F.rrect(cx, yy-16, wid, 28, "white", stroke=C["indigo"], r=7)
            F.text(cx+wid/2, yy-2, tk, size=12, fill=C["indigo"], weight=600)
        yy += 64
    F.text(xr + (F.w-xr-F.pad)/2, yy+10, "الشائع يبقى كما هو، والكلمات النادرة تُقصّ إلى مقاطع معلومة", size=12.5, fill=C["indigo"], opacity=0.8)
    F.badge(xr+26, 440, "شارات خاصة: مسافة سابقة قبل كلمة جديدة، وعلامة نهاية الوحدة", C["indigo"], size=10.5)
    F.text(F.w/2, 508, "BPE يلتقط اللواحق المتكررة في العربية (ال، ات، ها) من الإحصاء — دون قواعد صرف جاهزة", size=13.5, fill=C["mute"])
    F.section_note("الفصل 4 — الترميز إلى وحدات")
    return F

def f07_w2v():
    F = Fig(1040, 540)
    F.text(F.w/2, 34, "فضاء التضمينات: الهندسة تحمل المعنى", size=21, weight=700)
    F.arrow_def("agrE", C["emerald"])
    x0, y0, sx, sy = 430, 285, 95, 82
    pts = {
        "ملك": (-2.05, 1.0), "رجل": (-2.05, -1.0), "امرأة": (1.0, -1.0), "ملكة": (1.0, 1.0),
        "ولد": (-1.2, -2.1), "بنت": (1.05, -2.05), "أب": (-2.2, 2.1), "أم": (1.05, 2.1),
    }
    def P(p):
        return (x0 + p[0]*sx, y0 - p[1]*sy)
    # محاور خفيفة
    F.line(100, y0, 760, y0, stroke=C["line"], sw=1.2)
    F.line(x0, y0 - 195, x0, y0 + 195, stroke=C["line"], sw=1.2)
    F.text(x0 - 190, y0 - 200, "جهة المذكّر", size=11.5, fill=C["mute"])
    F.text(x0 + 170, y0 - 200, "جهة مؤنّثة", size=11.5, fill=C["mute"])
    arrows = [("رجل", "امرأة"), ("ملك", "ملكة"), ("أب", "أم"), ("ولد", "بنت")]
    for a, b in arrows:
        (x1, y1), (x2, y2) = P(pts[a]), P(pts[b])
        F.line(x1, y1, x2, y2, stroke=C["emerald"], sw=1.7, dash="6 4", marker="agrE")
    # صندوق الحساب: ملك − رجل + امرأة ≈؟
    (mx, my), (rx, ry), (ax, ay) = P(pts["ملك"]), P(pts["رجل"]), P(pts["امرأة"])
    (qx, qy) = P(pts["ملكة"])
    F.rrect(250, 92, 196, 58, C["roseL"], stroke=C["rose"], r=10, sw=1.4, dash="5 3")
    F.text(348, 114, "ملك − رجل + امرأة", size=11.4, mono=True, fill=C["rose"], weight=700)
    F.text(348, 136, "= ؟ ← الأقرب: ملكة", size=11.4, fill=C["rose"], weight=700)
    F.curve(f"M 452 112 C 500 112, {qx} 150, {qx} {qy - 24}", stroke=C["rose"], sw=1.5, dash="4 3", marker="agrE")
    # شرح يمين
    for i, ln in enumerate([
        "عملية القياس (analogy) تعني فحص",
        "الأقرب إلى نتيجة الفرق بين متجهات:",
        "نفس فكرة الجبر الخطي لكنه على المعاني!",
    ]):
        F.text(790, 150 + i * 24, ln, size=11.8, fill=C["slate"] if i < 2 else C["mute"], anchor="start")
    # العقد
    for t, p in pts.items():
        x, y = P(p)
        col = C["violet"] if t in ("ملك", "ملكة") else C["indigo"]
        F.circle(x, y, 21, fill=C["white"], stroke=col, sw=1.8)
        F.text(x, y + 1, t, size=12.8, weight=700, fill=col)
    F.text(F.w/2, 508, "حدود التضمين الثابت: «عين» تعني المريض عند الطبيب ولعبة عند الصائغ — الحل السياقي في الفصول 7–9", size=12.2, fill=C["mute"])
    F.section_note("الفصل 5 — تضمينات الكلمات (Word2Vec · GloVe)")
    return F

def f08_lstm():
    F = Fig(1040, 545)
    F.arrows_bold("agr")
    F.text(F.w/2, 34, "LSTM: بوابة إدارة الذاكرة", size=21, weight=700)
    cx, cy = 512, 232
    F.rrect(cx-210, cy-84, 420, 168, C["white"], stroke=C["indigo"], r=16, sw=1.6)
    F.badge(cx-190, cy-110, "خلية LSTM", C["indigo"], size=12)
    gates = [("i", "بوابة الدخل", "ما الذي يثر من جديد؟", C["cyan"]),
             ("f", "بوابة النسيان", "ماذا نُبقي من الماضي؟", C["violet"]),
             ("o", "بوابة الإخراج", "ماذا نُعلن الآن؟", C["emerald"])]
    for k, (tag, name, d, col) in enumerate(gates):
        gx = cx - 150 + k*150
        F.circle(gx, cy - 18, 26, C["panel2"], stroke=col, sw=1.8)
        F.text(gx, cy - 17, tag, size=16, mono=True, weight=700, fill=col)
        F.text(gx, cy + 26, name, size=12.5, weight=600)
        F.text(gx, cy + 44, d, size=11, fill=C["mute"])
    # شريط الحالة الامتداد
    y_st = cy + 76
    F.line(cx-210, y_st, cx+210, y_st, stroke=C["amber"], sw=3, marker=None)
    F.text(cx, y_st+22, "الخط العصبي C_t — شريط المعلومات الطويل (Conveyor Belt)", size=12, fill=C["amber"], weight=600)
    # تدفق غير ملفوف في الزمن
    yb = 428
    xs = []
    for k, t in enumerate(["t−2", "t−1", "t"]):
        x = F.pad + 70 + k*300
        xs.append(x)
        F.rrect(x, yb-42, 210, 84, C["panel"], stroke=C["line"], r=12)
        F.text(x+105, yb-16, t, size=14, weight=700, mono=True)
        F.text(x+105, yb+16, "h ← ذاكرة + معلومة جديدة", size=11.5, fill=C["mute"])
        if k < 2:
            F.line(x+210, yb-30, x+300, yb-30, stroke=C["amber"], sw=2.4, marker="agrG")
    F.arrow_def("agrG", C["amber"])
    F.text(F.w/2, 366, "الذاكرة تمتد عبر الزمن؛ والبوابات تنظّم ما يُنشر في كل خطوة", size=13.5, fill=C["mute"])
    F.rrect(F.pad+30, 504, F.w-60, 34, C["amberL"], stroke="#FCD34D", r=10)
    F.text(F.w/2, 521, "الحد: حساب الزمن تسلسلي بلا توازٍ؛ وهنا استحوذ المحوّل على الساحة (الفصل 8)", size=12.5, fill=C["amber"])
    F.section_note("الفصل 6 — تضمينات الجُمل (RNN · LSTM · GRU)")
    return F

def f09_attention():
    F = Fig(1040, 560)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 34, "الانتباه الذاتي: Q·Kᵀ ÷ √d_k ثم المزج مع V", size=21, weight=700)
    # السلسلة
    stages = [
        ("Q, K, V", "استعلام/مفتاح/قيمة", C["indigo"]),
        ("QKᵀ", "قرابة كل زوج مواضع", C["violet"]),
        ("÷ √d_k", "تثبيت الأعداد", C["cyan"]),
        ("+ Mask", "منع رؤية المستقبل", C["amber"]),
        ("softmax", "تحويل إلى أوزان", C["rose"]),
        ("· V", "مزج القيم المختارة", C["emerald"]),
    ]
    bw = (F.w - 2*F.pad)/6 - 16
    for k, (t, d, col) in enumerate(stages):
        x = F.pad + k*(bw+16)
        F.rrect(x, 118, bw, 84, "white", stroke=col, r=12, sw=1.6)
        F.parts.append(f'<rect x="{x}" y="118" width="7" height="84" rx="3.5" ry="3.5" fill="{col}"/>')
        F.text(x+bw/2+3, 148, t, size=15, mono=True, weight=700)
        F.text(x+bw/2+3, 174, d[:30], size=11.2, fill=C["mute"])
        if k < 5:
            F.line(x+bw+2, 160, x+bw+14, 160, stroke=C["line2"], sw=1.6, marker="agrA")
    # مثال الرسم الحسي (مصفوفة انتباه)
    x0, y0 = 120, 340
    cell = 30
    words = ["القطة", "جلست", "على", "السجادة", "لأنها", "متعبة"]
    F.text(x0 - 12, y0 - 40, "أوزان الانتباه (مثال حسي)", size=13, weight=700, anchor="start", fill=C["slate"])
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            v = 1 if i == j else (0.62 if (i, j) in ((4, 0), (0, 4)) else (0.44 if (i, j) in ((4, 2), (4, 1)) else 0.1))
            # مثلث بالقناع: أعلى فقط
            if j < i:
                yy = y0 + i*cell; xx = x0 + j*cell
                F.parts.append(f'<rect x="{xx}" y="{yy}" width="{cell-3}" height="{cell-3}" rx="4" fill="#4F46E5" opacity="{0.18 + v*0.78:.2f}"/>')
    for k, w1 in enumerate(words):
        F.text(x0 - 6, y0 + k*cell + 14, w1, size=10.5, anchor="end", fill=C["slate"])
        F.text(x0 + k*cell + 15, y0 - 8, w2 := words[k], size=10.5, anchor="start", fill=C["mute"], opacity=0.85 if k else 1)
    # شرح زاوية يمين
    xr = 620
    F.rrect(xr, 300, F.w - xr - F.pad - 4, 200, C["cyanL"], stroke="#0891B2", r=14)
    F.text((xr + F.w - F.pad)/2, 328, "القناع السببي (Causal Mask)", size=14.5, weight=700, fill=C["cyan"])
    F.text(xr+20, 356, "لأنها ← (القطة؟ السجادة؟) الآن يتضح:", size=12.5, fill=C["slate"], anchor="start")
    F.text(xr+20, 380, "«القطة» توزع وزن انتباه مرتفع — إحالة ضميرية صحيحة", size=12, fill=C["slate"], anchor="start")
    F.text(xr+20, 404, "المثلث البنفسجي: يمنع «رؤية» المستقبل أثناء التدريب", size=12, fill=C["slate"], anchor="start")
    F.badge(xr+22, 440, "نسيان القناع = أشهر أخطاء تدريب المولّدات", C["rose"], size=11)
    F.text(F.w/2, 530, "كل رأس يسأل سؤالًا مختلفًا: تتبع/تركيب/ضمائر — والرؤوس تتضاعف مع (Multi-Head)", size=13, fill=C["mute"])
    F.section_note("الفصل 7 — الانتباه الذاتي")
    return F

def f10_transformer():
    F = Fig(1040, 600, 40)
    F.arrows_bold("agrA", C["indigo"])
    F.text(F.w/2, 36, "كتلة المحوّل الحديثة (Pre-LN): بسيطة وقابلة للتوسع", size=21, weight=700)
    # هيكل المحوّل
    x0, y0 = 130, 108
    defs = [
        ("x", C["cyan"], 44),
        ("RMSNorm", C["violet"], 44),
        ("Multi-Head Attention", C["indigo"], 52),
        ("＋ Residual", C["emerald"], 40),
        ("RMSNorm", C["violet"], 44),
        ("Feed-Forward (SwiGLU)", C["amber"], 52),
        ("＋ Residual", C["emerald"], 40),
    ]
    yy = y0
    nodes = []
    for t, col, h in defs:
        nodes.append((yy, h))
        label = t if t != "x" else "المدخل — التضمين + الموضع (RoPE)"
        F.rrect(x0, yy, 300, h, "white", stroke=col, r=9, sw=1.5)
        F.parts.append(f'<rect x="{x0}" y="{yy}" width="7" height="{h}" rx="3.5" ry="3.5" fill="{col}"/>')
        F.text(x0+157, yy+h/2, label, size=13.8, weight=600 if t != "x" else 700)
        yy += h + 14
    F.line(x0+152, y0+44, x0+152, y0+48, stroke=C["line2"], sw=1.5, marker="agrA")
    for k in range(5):
        (ty, th) = nodes[k]
        (ny, _) = nodes[k+1]
        F.line(x0+152, ty+th, x0+152, ny-2, stroke=C["line2"], sw=1.5, marker="agrA")
        # خط بوثقة متبقية
        if defs[k+1][0] in ("＋ Residual",):
            mid = (ty+th + ny)/2
            F.curve(f"M {x0+302} {ty+th/2} C {x0+360} {ty+th/2}, {x0+360} {mid}, {x0+306} {mid}",
                    stroke=C["emerald"], sw=1.6, dash="5 4", marker="agrA")
    F.badge(x0-30, yy - 10, "× N طبقات (40–120 في النماذج الإنتاجية)", C["indigo"], size=11)
    # جانب: why Pre-LN
    x1 = 500
    F.rrect(x1, y0, F.w - x1 - F.pad, 300, C["panel"], stroke=C["line"], r=14)
    F.text((x1 + F.w - F.pad)/2, y0+28, "الديناميات الدقيقة", size=15, weight=700)
    F.text(x1 + 24, y0+60, "Pre-LN (قبل الطبقة الفرعية):", size=13, weight=700, anchor="start", fill=C["emerald"])
    F.text(x1 + 40, y0+84, "تدريب عميق مستقر؛ 95% من النماذج بعد 2020", size=12.2, anchor="start")
    F.text(x1 + 24, y0+116, "Post-LN (بعد الطبقة الفرعية):", size=13, weight=700, anchor="start", fill=C["rose"])
    F.text(x1 + 40, y0+140, "أفضل أداء؟ يحتاج warmup وصبر؛ هش في العمق", size=12.2, anchor="start")
    F.text(x1 + 24, y0+172, "RoPE:", size=13, weight=700, anchor="start", fill=C["indigo"])
    F.text(x1 + 40, y0+196, "الدوران الجيبي على الزوايا يجعل الموضع النسبي جاهزًا", size=12.2, anchor="start")
    F.badge(x1+26, y0+232, "RMSNorm مبسّط: بلا مجموع؛ أسرع 5–15%", C["emerald"], size=10.5)
    F.badge(x1+26, y0+266, "SwiGLU: بوابة GLU مزدوجة بلا انحياز", C["amber"], size=10.5)
    F.rrect(x1, y0+318, F.w - x1 - F.pad, 168, C["violetL"], stroke="#A78BFA", r=14)
    F.text((x1 + F.w - F.pad)/2, y0+344, "أين تذهب المعاملات؟ (7B تقريبًا)", size=14, weight=700, fill=C["violet"])
    F.hbar(x1+30, y0+388, F.w - x1 - F.pad - 90, 0.62, "طبقات FFN (SwiGLU)", "~62% (4.3B)", C["violet"], h=14)
    F.hbar(x1+30, y0+424, F.w - x1 - F.pad - 90, 0.23, "الانتباه (QKVO)", "~23% (1.6B)", C["indigo"], h=14)
    F.hbar(x1+30, y0+460, F.w - x1 - F.pad - 90, 0.15, "التضمين + رأس اللغة", "~15% (1.0B)", C["emerald"], h=14)
    F.section_note("الفصل 8 — بنية المحوّل")
    return F
