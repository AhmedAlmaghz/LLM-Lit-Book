# -*- coding: utf-8 -*-
import sys, io
sys.path.insert(0, 'tools')
import mathlite as M
import re

out = io.StringIO()

def trace(s, label):
    t = M._convert(s)
    print(label, 'convert :', repr(t), file=out)
    return t

trace(r"\pi_\theta(y_w|x)", 'A')
trace(r"\pi_{\theta}(y_w|x)", 'B')
raw = r"\(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y|x)}\)"
r = M.render(raw)
print('render   :', repr(r), file=out)
print('renderS  :', r, file=out)
open('build/probe_dbg.txt', 'w', encoding='utf-8').write(out.getvalue())
print('ok')
