# -*- coding: utf-8 -*-
import json, io

d = json.load(open('build/book.json', encoding='utf-8'))
out = io.StringIO()
p = lambda *a: print(*a, file=out)

ch = d['chapters']

# find h1 block
for c in ch:
    for b in c['blocks']:
        if b.get('t') == 'h1':
            p('H1 FOUND: ch', c['num'], json.dumps(b, ensure_ascii=False)[:300])
            break

# one sample per type, from a mid chapter with all types
seen = set()
c = None
for cc in ch:
    ts = {b['t'] for b in cc['blocks']}
    if {'quote','code','table','ul','ol','h3','hr','p'} <= ts:
        c = cc
        break
p('SAMPLE CHAPTER:', c['num'], c['title'][:60])
for b in c['blocks'][:220]:
    t = b['t']
    if t in seen:
        continue
    seen.add(t)
    p('--', t, '=>', json.dumps(b, ensure_ascii=False)[:400])

# quotes: collect kinds histogram
from collections import Counter
kinds = Counter()
for cc in ch + d['appendices']:
    for b in cc['blocks']:
        if b['t'] == 'quote':
            kinds[b.get('kind', '??')] += 1
p('QUOTE KINDS:', dict(kinds))

# code: keys and languages
langs = Counter()
codekeys = None
for cc in ch:
    for b in cc['blocks']:
        if b['t'] == 'code':
            codekeys = list(b.keys())
            langs[b.get('lang', '??')] += 1
p('CODE KEYS:', codekeys, 'LANGS:', dict(langs))
# sample code block (truncate code)
for cc in ch:
    for b in cc['blocks']:
        if b['t'] == 'code' and b.get('lang') == 'python':
            p('CODE SAMPLE keys:', list(b.keys()))
            p('code head:', json.dumps({k: (v[:160] if isinstance(v, str) and len(v) > 160 else v) for k, v in b.items()}, ensure_ascii=False))
            break
    else:
        continue
    break

# table sample
for cc in ch:
    for b in cc['blocks']:
        if b['t'] == 'table':
            p('TABLE KEYS:', list(b.keys()))
            head = b.get('head') or b.get('header')
            p('head:', json.dumps(head, ensure_ascii=False)[:250])
            rows = b.get('rows') or []
            p('rows n=', len(rows), 'row0:', json.dumps(rows[0], ensure_ascii=False)[:250])
            break
    else:
        continue
    break

# caption title without? check 'caption' presence across types
keys_all = Counter()
for cc in ch:
    for b in cc['blocks']:
        keys_all[b['t'] + ':' + ','.join(sorted(b.keys()))] += 1
for k, v in sorted(keys_all.items()):
    p('KEYSIG', k, v)

# front blocks types
for fk, fv in d['front'].items():
    p('FRONT', fk, 'types:', Counter(b['t'] for b in fv))
    qk = Counter(b.get('kind','') for b in fv if b['t'] == 'quote')
    p('  quote kinds:', dict(qk))

# dedication list length and any quote kinds inside dedication
dd = d['front']['dedication']
p('DEDICATION first 12 types:', [b['t'] for b in dd[:12]])

# statistics per part
from collections import defaultdict
byp = defaultdict(list)
for c in ch:
    byp[c['part']].append(c['num'])
for pt in sorted(byp):
    p('part', pt, '->', byp[pt])

open('build/probe_blocks.txt', 'w', encoding='utf-8').write(out.getvalue())
print('ok')
