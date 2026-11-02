# -*- coding: utf-8 -*-
"""
الفصل 14: بناء النموذج — من Embedding إلى GPT Block (كود كامل، مختبر، خطوة بخطوة)

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 14 · عدد كتل بايثون في هذا الملف: 13

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch14_build-model-gpt-block.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch14_build-model-gpt-block.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/13] class RMSNorm(nn.Module): ════════


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    من LLaMA، Qwen، DeepSeek.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # حساب في FP32 لتجنب عدم الاستقرار في BF16/FP16
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


# ════════ [2/13] norm = RMSNorm(64) ════════


norm = RMSNorm(64)
x = torch.randn(2, 10, 64)
out = norm(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
print(f"RMS of output: {out.float().pow(2).mean(-1).sqrt().mean():.4f}")
# يجب أن يكون قريبًا من 1.0 (بسبب weight=1 وعدم وجود انحياز)


# ════════ [3/13] class RoPE(nn.Module): ════════


class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    يُطبق على Q وK فقط، وليس على V.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len

        # الترددات: 1 / (base^(2i/dim)) لكل زوج
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # حساب cos و sin مسبقًا لكل المواضع
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len).float()
        freqs = torch.outer(pos, self.inv_freq)  # (seq, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        """
        q, k: (batch, n_heads, seq_len, head_dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim/2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # تدوير الأزواج: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
        return q_rot, k_rot


# ════════ [4/13] rope = RoPE(dim=64, max_seq_len=512) ════════


rope = RoPE(dim=64, max_seq_len=512)
q = torch.randn(2, 8, 128, 64)  # (B, H, L, D)
k = torch.randn(2, 8, 128, 64)
q_rot, k_rot = rope(q, k, seq_len=128)
print(f"q_rot shape: {q_rot.shape}")  # يجب أن يكون (2, 8, 128, 64)

# اختبار خاصية: الضرب النقطي يعتمد على الفرق في الموضع
q_test = torch.zeros(1, 1, 2, 64)
k_test = torch.zeros(1, 1, 2, 64)
q_test[0, 0, 0, :2] = torch.tensor([1.0, 0.0])   # موضع 0
k_test[0, 0, 1, :2] = torch.tensor([1.0, 0.0])   # موضع 1
q_rot, k_rot = rope(q_test, k_test, seq_len=2)
score = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
print(f"Score with relative position 1: {score:.4f}")
# يجب أن يعتمد على الزاوية، وليس على الموضع المطلق


# ════════ [5/13] class CausalSelfAttention(nn.Module): ════════


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        x: (batch, seq_len, d_model)
        attention_mask: (batch, seq_len) - 1 للكلمات الحقيقية، 0 للحشو
        """
        B, L, _ = x.shape

        # الإسقاطات
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # تطبيق RoPE على Q وK
        q, k = self.rope(q, k, seq_len=L)

        # استخدام F.scaled_dot_product_attention (FlashAttention تلقائيًا)
        # is_causal=True يطبق القناع السببي تلقائيًا
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,          # لا حاجة، is_causal كافٍ
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        # out: (B, H, L, head_dim)

        # دمج الرؤوس
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)


# ════════ [6/13] attn = CausalSelfAttention(d_model=256, n_heads=8, max_seq_len=512) ════════


attn = CausalSelfAttention(d_model=256, n_heads=8, max_seq_len=512)
x = torch.randn(2, 64, 256)
out = attn(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # (2, 64, 256)

# اختبار القناع السببي: تغيير رمز مستقبلي لا يجب أن يؤثر على المخرجات السابقة
x1 = x.clone()
x2 = x.clone()
x2[:, 10, :] = torch.randn(256)  # غيّر الرمز رقم 10
out1 = attn(x1)
out2 = attn(x2)
# يجب أن تكون المخرجات للمواضع 0..9 متطابقة
diff_before = (out1[:, :10] - out2[:, :10]).abs().max()
diff_after = (out1[:, 10:] - out2[:, 10:]).abs().max()
print(f"Diff before pos 10: {diff_before:.6f}")  # يجب أن يكون ~0
print(f"Diff after pos 10: {diff_after:.6f}")   # يجب أن يكون > 0


# ════════ [7/13] class SwiGLU(nn.Module): ════════


class SwiGLU(nn.Module):
    """
    SwiGLU FFN.
    d_ff الافتراضي = 8/3 * d_model (مع تقريب لمضاعف 256).
    """
    def __init__(self, d_model: int, d_ff: int = None, bias: bool = False):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # تقريب لأقرب مضاعف لـ 256 (لتحسين أداء العتاد)
            d_ff = ((d_ff + 255) // 256) * 256

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ════════ [8/13] ffn = SwiGLU(d_model=256) ════════


ffn = SwiGLU(d_model=256)
x = torch.randn(2, 64, 256)
out = ffn(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"d_ff: {ffn.gate_proj.out_features}")  # يجب أن يكون ~682 (8/3 * 256)


# ════════ [9/13] class TransformerBlock(nn.Module): ════════


class TransformerBlock(nn.Module):
    """
    كتلة محوّل حديثة (Pre-LN): RMSNorm + Attention + Residual + RMSNorm + FFN + Residual.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        d_ff: int = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # التيار المتبقي: نضيف، لا نستبدل
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ════════ [10/13] block = TransformerBlock(d_model=256, n_heads=8) ════════


block = TransformerBlock(d_model=256, n_heads=8)
x = torch.randn(2, 64, 256)
out = block(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # نفس الشكل


# ════════ [11/13] class MiniGPT(nn.Module): ════════


class MiniGPT(nn.Module):
    """
    نموذج لغوي بنمط LLaMA مصغّر.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # الكتل
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, max_seq_len, dropout, d_ff
            )
            for _ in range(n_layers)
        ])

        # التطبيع النهائي
        self.norm_f = RMSNorm(d_model)

        # رأس اللغة
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ربط الأوزان
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # التهيئة
        self.apply(self._init_weights)

        # تهيئة خاصة للـ o_proj و down_proj (لتقليل الانفجار)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        input_ids: (batch, seq_len)
        targets: (batch, seq_len) - اختياري
        """
        B, L = input_ids.shape

        # Embedding
        x = self.token_embedding(input_ids)  # (B, L, d_model)

        # المرور عبر الكتل
        for block in self.blocks:
            x = block(x, attention_mask)

        # التطبيع النهائي
        x = self.norm_f(x)

        # رأس اللغة
        logits = self.lm_head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_id: int = None,
    ):
        """توليد متسلسل مع KV cache ضمني (بسيط هنا، بدونه)."""
        self.eval()
        for _ in range(max_new_tokens):
            # اقتطاع إذا تجاوزنا max_seq_len
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                logits[mask] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


# ════════ [12/13] def test_model(): ════════


def test_model():
    """اختبار شامل للنموذج."""
    vocab_size = 1000
    batch_size = 2
    seq_len = 64

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=512,
    )

    # اختبار الأشكال
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)
    print(f"Logits shape: {logits.shape}")  # (2, 64, 1000)
    print(f"Loss: {loss.item():.4f}")
    print(f"Expected initial loss (log V): {math.log(vocab_size):.4f}")

    # اختبار التوليد
    generated = model.generate(
        input_ids[:1, :10],
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )
    print(f"Generated shape: {generated.shape}")  # (1, 30)

    # عدد المعاملات
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_emb = n_params - model.token_embedding.weight.numel()
    print(f"Total parameters: {n_params:,}")
    print(f"Parameters (no embedding, due to tying): {n_params_no_emb:,}")

    return model

model = test_model()


# ════════ [13/13] def count_parameters(model): ════════


def count_parameters(model):
    """حساب دقيق للمعاملات، مع تفصيل."""
    total = 0
    breakdown = {}

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        # تصنيف
        if "embedding" in name:
            key = "embedding"
        elif "attn" in name:
            key = "attention"
        elif "ffn" in name:
            key = "ffn"
        elif "norm" in name:
            key = "norm"
        elif "lm_head" in name:
            key = "lm_head"
        else:
            key = "other"

        breakdown[key] = breakdown.get(key, 0) + n

    print(f"Total parameters: {total:,}")
    print("Breakdown:")
    for key, n in breakdown.items():
        print(f"  {key}: {n:,} ({n/total*100:.1f}%)")

    return total, breakdown

# لـ 124M (مثل GPT-2 small)
model_124m = MiniGPT(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    max_seq_len=1024,
)
count_parameters(model_124m)
