# -*- coding: utf-8 -*-
"""فحص سريع لاستخدام التراجع الكودي والتاتكس داخل النص."""
import json, re
from collections import Counter

b = json.load(open('build/book.json', encoding='utf-8'))
bt = q = 0

def walk(blks):
    global bt, q
    for x in blks:
        t = x.get('t')
        if t in ('p', 'quote'):
            s = x.get('text', '')
            bt += s.count('`'); q += s.count(r'\text')
        elif t in ('ul', 'ol'):
            for it in x.get('items', []):
                bt += it.count('`'); q += it.count(r'\text')
        elif t == 'table':
            for r in x.get('rows', []):
                for c in r:
                    bt += c.count('`'); q += c.count(r'\text')

for grp in ('chapters', 'appendices'):
    for c in b[grp]:
        walk(c['blocks'])
walk(b['front']['preface']); walk(b['front']['summary'])
walk(b['front']['dedication'][:10])
print('backticks in prose:', bt)
print('text-command usages:', q)
