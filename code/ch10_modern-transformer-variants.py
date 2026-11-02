# -*- coding: utf-8 -*-
"""
الفصل 10: ما بعد المحوّل الكلاسيكي — FlashAttention وMLA وMoE وMamba

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثالث · الفصل 10 · عدد كتل بايثون في هذا الملف: 3

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch10_modern-transformer-variants.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch10_modern-transformer-variants.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/3] def flash_attention_forward(q, k, v, causal=True): ════════


import torch
import torch.nn.functional as F

def flash_attention_forward(q, k, v, causal=True):
    """
    q, k, v: (batch, heads, seq, dim)
    PyTorch سيختار تلقائيًا FlashAttention إن كان متاحًا.
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        is_causal=causal,
        # dropout_p=0.0 في التقييم
    )

# للتحقق من أن FlashAttention مفعّل:
print(torch.backends.cuda.flash_sdp_enabled())  # True


# ════════ [2/3] class MLA(nn.Module): ════════


class MLA(nn.Module):
    def __init__(self, d_model, d_kv, d_latent):
        super().__init__()
        # ضغط K و V إلى فضاء كامن
        self.compress_kv = nn.Linear(d_model, d_latent)
        # إعادة بناء K و V
        self.expand_k = nn.Linear(d_latent, d_kv)
        self.expand_v = nn.Linear(d_latent, d_kv)
        self.q_proj = nn.Linear(d_model, d_kv)

    def forward(self, x):
        # x: (batch, seq, d_model)
        q = self.q_proj(x)
        latent = self.compress_kv(x)       # (batch, seq, d_latent)
        k = self.expand_k(latent)           # (batch, seq, d_kv)
        v = self.expand_v(latent)
        # ... الانتباه المعتاد ...


# ════════ [3/3] def repeat_kv(x, n_rep): ════════


# في PyTorch، يمكنك محاكاة GQA عبر تكرار K/V
def repeat_kv(x, n_rep):
    """x: (batch, n_kv_heads, seq, dim) -> (batch, n_kv_heads*n_rep, seq, dim)"""
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
        x.size(0), -1, x.size(2), x.size(3)
    )
