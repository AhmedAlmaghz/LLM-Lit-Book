#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — الدافع وراء آلية الانتباه الذاتي — اللحظة التي توقفت فيها الشبكة عن النسيان
Transformers and Large Language Models — Self Attention Why

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-07-llm-ds (lesson 7)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part03_transformers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ خلية الإعداد — شغّلها أولًا (Colab / Kaggle / محليًا)
# تثبّت ما ينقص فقط، وتتجاهل ما لا يدعم جهازك.
import sys, subprocess, importlib

PKGS = [
    ("torch", "torch"), ("transformers", "transformers"),
    ("datasets", "datasets"), ("accelerate", "accelerate"),
    ("peft", "peft"), ("trl", "trl"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"), ("pandas", "pandas"),
    ("numpy", "numpy"), ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"), ("evaluate", "evaluate"),
]
todo = []
for mod, pipname in PKGS:
    try:
        importlib.import_module(mod)
    except Exception:
        todo.append(pipname)
if todo:
    print("تثبيت الحزم الناقصة:", todo)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *todo])
    except Exception as e:
        print("تعذّر تثبيت بعض الحزم (ثبّتها يدويًا إن احتجتها):", e)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print("device:", DEVICE)
print("جاهز ✅")

# ────────────────────────────────────────────────────────────────────────
# [002] code/notebooks/part03_transformers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part03_transformers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part03_transformers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def causal_mask(seq_len, device="cpu"):
    # True للمواقع المسموح بها (بما فيها القطر)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask  # (L, L)، سنضيف أبعاد batch/heads لاحقًا

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part03_transformers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part03_transformers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, bias=False):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part03_transformers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part03_transformers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part03_transformers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # نستخدم تمثيل [CLS] (أول رمز) كملخص للجملة
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

# مثال للاستخدام
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["I love this movie", "This film is terrible"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

model = BertClassifier()
logits = model(batch["input_ids"], batch["attention_mask"])
print(logits.softmax(dim=-1))  # احتمالات الفئتين

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part03_transformers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
class GPTBlock(nn.Module):
    """كتلة GPT: انتباه سببي + FFN، بدون انتباه متقاطع."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part03_transformers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_text = "translate English to French: The cat is sitting on the mat."
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# "Le chat est assis sur le tapis."

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part03_transformers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part03_transformers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part03_transformers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# في PyTorch، يمكنك محاكاة GQA عبر تكرار K/V
def repeat_kv(x, n_rep):
    """x: (batch, n_kv_heads, seq, dim) -> (batch, n_kv_heads*n_rep, seq, dim)"""
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
        x.size(0), -1, x.size(2), x.size(3)
    )

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part03_transformers.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# تكميم AWQ (مبسط)
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama-3-8b-awq",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# ────────────────────────────────────────────────────────────────────────
# [016] code/ch07_self-attention-why.py
# ────────────────────────────────────────────────────────────────────────
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
