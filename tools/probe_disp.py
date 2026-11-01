# -*- coding: utf-8 -*-
import json, re
d = json.load(open('build/book.json', encoding='utf-8'))
conts = list(d['chapters']) + list(d['appendices'])
cnt_disp = 0
cnt_dollar = 0
samples = []
for c in conts:
    for b in c['blocks']:
        texts = [b.get('text', ''), b.get('code', '')]
        texts += b.get('items', []) if b.get('items') else []
        for t in texts:
            if '\\[' in t or '\\]' in t:
                cnt_disp += 1
                if len(samples) < 3: samples.append(t[:140])
            if '$$' in t:
                cnt_dollar += 1
print('display \\[ blocks:', cnt_disp, ' $$ blocks:', cnt_dollar)
for s in samples: print('SAMPLE:', s)
