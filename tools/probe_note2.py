# -*- coding: utf-8 -*-
import json
d = json.load(open('build/book.json', encoding='utf-8'))
out = open('build/probe_note2.txt', 'w', encoding='utf-8')
n = 0
for c in d['chapters'] + d['appendices']:
    for b in c['blocks']:
        if b['t'] == 'quote' and b.get('kind') == 'note':
            if not b['text'].startswith('**'):
                n += 1
                if n <= 6:
                    print('CH', c.get('num', c.get('title', '')), '>>', b['text'][:100], file=out)
print('unlabeled total:', n, file=out)
print('front note samples:', file=out)
for b in d['front']['dedication']:
    if b['t'] == 'quote':
        print('  ', b['text'][:90], file=out)
out.close()
print('ok')
