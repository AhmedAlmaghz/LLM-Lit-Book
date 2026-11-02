# -*- coding: utf-8 -*-
"""
الفصل 8: بنية المحوّل — من كتلة واحدة إلى معمارية كاملة

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثالث · الفصل 8 · عدد كتل بايثون في هذا الملف: 4

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch08_transformer-architecture.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch08_transformer-architecture.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/4] class RMSNorm(nn.Module): ════════


import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (..., dim)
        # نحسب متوسط المربعات على آخر بُعد
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


# ════════ [2/4] class SwiGLU(nn.Module): ════════


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, bias=False):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ════════ [3/4] class RoPE(nn.Module): ════════


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        # الترددات: من 1 إلى 1/base تقريبًا
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # حساب cos و sin مسبقًا لكل المواضع
        pos = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, inv_freq)  # (seq, dim/2)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k, seq_len):
        # q, k: (batch, heads, seq, dim)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim/2)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # تدوير الأزواج
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
        return q_rot, k_rot


# ════════ [4/4] class ModernTransformerBlock(nn.Module): ════════


class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # التيار المتبقي: نضيف، لا نستبدل
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
