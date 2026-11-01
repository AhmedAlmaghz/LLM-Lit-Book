# -*- coding: utf-8 -*-
"""مسح صفحات PDF للعثور على علامات مرجعية (فصول/أشكال/جداول/كود)."""
import pymupdf, sys

d = pymupdf.open('dist/llm-ds.pdf')
targets = ['الفصل 14: بناء النموذج', 'شكل 14.1', 'المخطط التوضيحي', 'الفصل 1: الشبكات', 'الحلقة التدريبية',
           'import numpy', 'Transformer — الترميز', 'الجزء الرابع', 'كشف بعد تشغيل', 'خطة الكتاب', 'مصفوفة', 'بعد التشغيل']
hits = {}
for i in range(d.page_count):
    t = d[i].get_text()
    for tg in targets:
        if tg in t and tg not in hits:
            hits[tg] = i
print('total pages:', d.page_count)
for k in targets:
    print(k, '->', hits.get(k))
