# -*- coding: utf-8 -*-
import json, io, sys

d = json.load(open('build/book.json', encoding='utf-8'))
out = io.StringIO()

def p(*a):
    print(*a, file=out)

p('== meta ==')
p(json.dumps(d['meta'], ensure_ascii=False, indent=1))
p('== front keys ==', list(d['front'].keys()))
for k, v in d['front'].items():
    p(f'-- front.{k} type={type(v).__name__}')
    if isinstance(v, list):
        p('   len', len(v), 'first:', json.dumps(v[0], ensure_ascii=False)[:200] if v else '')
    elif isinstance(v, dict):
        p('   keys', list(v.keys()))
    else:
        p('   val:', json.dumps(v, ensure_ascii=False)[:200])

ch = d['chapters']
p('== chapters n ==', len(ch))
c1 = ch[0]
p('keys of ch[0]:', list(c1.keys()))
for k, v in c1.items():
    if isinstance(v, list):
        p(f'  {k}: list len={len(v)}')
        if v and isinstance(v[0], dict):
            p('    first item keys:', list(v[0].keys()))
            p('    first item:', json.dumps(v[0], ensure_ascii=False)[:300])
    elif isinstance(v, dict):
        p(f'  {k}: dict keys=', list(v.keys()))
    else:
        p(f'  {k}:', json.dumps(v, ensure_ascii=False)[:160])

p('== parts ==')
parts = sorted({c.get('part') for c in ch})
p(parts)
# chapter numbering styles
p('num field of first chapters:',
  [c.get('num', c.get('number')) for c in ch[:6]])
p('== appendices ==')
ap = d['appendices']
p('n=', len(ap))
if ap:
    a0 = ap[0]
    p('keys:', list(a0.keys()))
    p(json.dumps({k: (f"list[{len(v)}]" if isinstance(v, list) else str(v)[:120]) for k, v in a0.items()}, ensure_ascii=False)[:600])

# block types histogram
from collections import Counter
types = Counter()
for c in ch:
    for b in c.get('blocks', []):
        types[b.get('t') or b.get('type')] += 1
p('== block types across chapters ==', dict(types))

open('build/probe_book.txt', 'w', encoding='utf-8').write(out.getvalue())
print('wrote build/probe_book.txt')
