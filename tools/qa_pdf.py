# -*- coding: utf-8 -*-
"""نقض الجودة للـ PDF: بقايا لاتكس خام، صفحات العينات للأشكال والكود."""
import pymupdf, re

d = pymupdf.open('dist/llm-ds.pdf')
leak_pages = []
for i in range(d.page_count):
    t = d[i].get_text()
    if re.search(r"\\frac\{|\bmathbf|\\sum_|\\begin\{|\[ \w = ", t):
        leak_pages.append(i)
print("pages:", d.page_count, "leaking latex:", len(leak_pages))
print("first leak pages:", leak_pages[:10])
# المخططات: عدد الصور في PDF
imgs = 0
for i in range(d.page_count):
    imgs += len(d[i].get_images(full=True))
print("embedded images:", imgs)
