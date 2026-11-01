# -*- coding: utf-8 -*-
import io, sys
sys.path.insert(0, 'tools')
import mathlite as M

cases = [
    r"\(\sigma(z) = \frac{1}{1+e^{-z}}\)",
    r"Tanh: \(\frac{e^z - e^{-z}}{e^z + e^{-z}}\)",
    r"\(\mathrm{Focal} = -\alpha (1-\hat{y})^{\gamma} \log \hat{y}\)",
    r"\(P(\text{dropout})\) — ضَر капуста \(d_{model}\), \(X_{ij}\), \(10^{-5}\)",
    r"\(\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\)",
    r"\(W = W_0 + \frac{\alpha}{r}\,BA,\quad \Delta W = BA\)",
    r"\(\mathcal{L}_{OR} = -\log\sigma\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y|x)}\right)\)",
    r"\(d_{ff} = \frac{8}{3}d\),\(K \in \mathbb{R}^{n \times d_k}\), \(d_c \ll d_k, d_v\)",
    r"\(\sum_{i=1}^{d_k} q_i k_i\)", r"\(\mathrm{PPL} = e^{\mathrm{loss}} \approx 10^{-5}\)",
    r"\(\begin{pmatrix} a & b \\ c & d \end{pmatrix}\)",
    r"\(\nabla_{\theta} J(\theta)\), \(50\%\), \(\infty\), \(\Re\)",
]
out = io.StringIO()
for c in cases:
    print('IN :', c, file=out)
    print('OUT:', M.render(c), file=out)
    print(file=out)
(f := open('build/probe_math.txt', 'w', encoding='utf-8')).write(out.getvalue()); f.close()
print('fails:', M.last_failures)
