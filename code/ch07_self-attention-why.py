# -*- coding: utf-8 -*-
"""
الفصل 7: الدافع وراء آلية الانتباه الذاتي — اللحظة التي توقفت فيها الشبكة عن النسيان

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثالث · الفصل 7 · عدد كتل بايثون في هذا الملف: 3

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch07_self-attention-why.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch07_self-attention-why.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/3] def scaled_dot_product_attention(Q, K, V, mask=None): ════════


import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n_heads, seq_len, d_k)
    K: (batch, n_heads, seq_len, d_k)
    V: (batch, n_heads, seq_len, d_v)
    mask: (batch, 1, seq_len, seq_len) أو (1, 1, seq_len, seq_len)
          True/1 للمواقع المسموح بها، False/0 للممنوعة
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq)
    output = torch.matmul(weights, V)    # (batch, heads, seq, d_v)
    return output, weights


# ════════ [2/3] class MultiHeadAttention(torch.nn.Module): ════════


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        # (B, L, d_model) -> (B, L, h, d_k) -> (B, h, L, d_k)
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ════════ [3/3] def causal_mask(seq_len, device="cpu"): ════════


def causal_mask(seq_len, device="cpu"):
    # True للمواقع المسموح بها (بما فيها القطر)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask  # (L, L)، سنضيف أبعاد batch/heads لاحقًا
