# -*- coding: utf-8 -*-
import json, io, re
from collections import Counter

d = json.load(open('build/book.json', encoding='utf-8'))
out = io.StringIO()
p = lambda *a: print(*a, file=out)

# One sample quote per kind: does the text carry its own bold label?
kind_samples = {}
for c in d['chapters'] + d['appendices']:
    for b in c['blocks']:
        if b['t'] == 'quote' and b.get('kind') not in kind_samples:
            kind_samples[b['kind']] = b['text']
for k, v in kind_samples.items():
    p(f'--- {k}:', v[:110].replace('\n', ' / '))

# unlabeled quotes count: label pattern  = text starting with **...**:
lab = re.compile(r'^\*\*[^*\n]{1,24}\*\*:')
cnt = Counter()
for c in d['chapters'] + d['appendices']:
    for b in c['blocks']:
        if b['t'] == 'quote':
            capitalized = bool(lab.match(b['text']))
            cnt[(b.get('kind'), capitalized)] += 1
p('LABEL PRESENT PER KIND:', dict(cnt))

# ===== / ----- separator pollution inside p texts
sep_p = 0
tail_p = 0
samples = []
for c in d['chapters'] + d['appendices']:
    for b in c['blocks']:
        if b['t'] == 'p':
            t = b['text']
            if re.fullmatch(r'[={\-_\s*]+', t):
                sep_p += 1
                samples.append(t)
            else:
                stripped = re.sub(r'\s*[=\-]{5,}\s*$', '', t)
                if stripped != t:
                    tail_p += 1
p('pure-separator p blocks:', sep_p, ' p with separator tails:', tail_p)
p('h2 texts containing numbers sample:')
n = 0
for c in d['chapters']:
    for b in c['blocks']:
        if b['t'] == 'h2':
            if re.match(r'^\d+\.?\d*\s', b['text']):
                n += 1
                if n <= 5:
                    p('   ', c['num'], '|', b['text'])
p('h2 with numeric prefix total:', n)

# h2 duplicated titles?
tt = Counter()
for c in d['chapters']:
    for b in c['blocks']:
        if b['t'] == 'h2':
            tt[b['text']] += 1
dup = [k for k, v in tt.items() if v > 1]
p('duplicated h2 titles:', len(dup))
for k in dup[:15]:
    p('   DUP:', k, tt[k])

# section lists: check format (dicts?)
s0 = d['chapters'][0]['sections']
p('sections[0] sample:', json.dumps(s0[0], ensure_ascii=False)[:200])
p('sections[1] sample:', json.dumps(s0[1], ensure_ascii=False)[:200])

# front summary h3 items already list parts; appendix sections sample
ap = d['appendices'][0]
p('ap sections sample:', json.dumps(ap['sections'][0], ensure_ascii=False)[:200])

# quote inside preface/summary/dedication handled; check stats keys of appendix quotes
open('build/probe_quotes.txt', 'w', encoding='utf-8').write(out.getvalue())
print('ok')
