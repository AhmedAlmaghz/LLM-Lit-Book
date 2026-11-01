# -*- coding: utf-8 -*-
import json, io

d = json.load(open('build/book.json', encoding='utf-8'))
out = io.StringIO()
p = lambda *a: print(*a, file=out)

dd = d['front']['dedication']
p('== DEDICATION OUTLINE (%d blocks) ==' % len(dd))
for i, b in enumerate(dd):
    if b['t'] in ('h1', 'h2', 'h3'):
        p(f'[{i}] {b["t"]}: {b["text"][:90]}')
    elif b['t'] == 'quote':
        p(f'[{i}] quote({b.get("kind")}): {b["text"][:70]}')
    elif b['t'] == 'code':
        p(f'[{i}] code({b.get("lang")}): {b["code"][:60]!r}')
    elif b['t'] == 'p':
        p(f'[{i}] p: {b["text"][:70]}')
    else:
        p(f'[{i}] {b["t"]}')

for fk in ('preface', 'summary'):
    fv = d['front'][fk]
    p(f'== {fk.upper()} OUTLINE ({len(fv)} blocks) ==')
    for i, b in enumerate(fv):
        if b['t'] in ('h1', 'h2', 'h3'):
            p(f'[{i}] {b["t"]}: {b["text"][:90]}')

c48 = d['chapters'][47]
p('== CH48 blocks tail ==')
for i, b in enumerate(c48['blocks']):
    if b['t'] == 'h1':
        p('H1 index in ch48:', i, b['text'])
        p('preceding:', json.dumps(c48['blocks'][i-1], ensure_ascii=False)[:120])
        p('following:', json.dumps(c48['blocks'][i+1], ensure_ascii=False)[:120] if i+1 < len(c48['blocks']) else 'NONE (last)')

# part titles: any place in raw manuscript? check manuscript.json
m = json.load(open('build/manuscript.json', encoding='utf-8'))
p('== manuscript.json keys ==', list(m.keys()) if isinstance(m, dict) else type(m).__name__)
if isinstance(m, dict):
    for k, v in m.items():
        if isinstance(v, dict):
            p(' ', k, '-> dict keys', list(v.keys())[:10])
        elif isinstance(v, list):
            p(' ', k, '-> list', len(v))
        else:
            p(' ', k, '->', str(v)[:60])

open('build/probe_front.txt', 'w', encoding='utf-8').write(out.getvalue())
print('ok')
