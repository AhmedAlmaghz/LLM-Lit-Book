# -*- coding: utf-8 -*-
"""
الفصل 21: الضبط الدقيق الفعّال (PEFT) المتقدم — عندما يكون "الأقل" أكثر

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الخامس · الفصل 21 · عدد كتل بايثون في هذا الملف: 7

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch21_advanced-peft.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch21_advanced-peft.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/7] class PfeifferAdapter(nn.Module): ════════


import torch
import torch.nn as nn

class PfeifferAdapter(nn.Module):
    """
    Adapter بسيط: LayerNorm + Down + Up + Residual.
    """
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)

        # تهيئة: up بأصفار، حتى يكون التعديل صفرًا في البداية
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x: (batch, seq, d_model)
        residual = x
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.up(h)
        return residual + h


# ════════ [2/7] class TransformerBlockWithAdapter(nn.Module): ════════


class TransformerBlockWithAdapter(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, adapter_bottleneck=64):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.adapter1 = PfeifferAdapter(d_model, adapter_bottleneck)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.adapter2 = PfeifferAdapter(d_model, adapter_bottleneck)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.adapter1(x)
        x = x + self.ffn(self.norm2(x))
        x = self.adapter2(x)
        return x


# ════════ [3/7] class PrefixTuning(nn.Module): ════════


class PrefixTuning(nn.Module):
    """
    Prefix Tuning: تضيف بادئة قابلة للتعلم إلى K و V.
    """
    def __init__(self, n_layers: int, n_heads: int, head_dim: int, prefix_len: int = 10):
        super().__init__()
        self.prefix_len = prefix_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

        # بادئة K و V لكل طبقة
        # الشكل: (n_layers, 2, n_heads, prefix_len, head_dim)
        self.prefix = nn.Parameter(
            torch.randn(n_layers, 2, n_heads, prefix_len, head_dim) * 0.02
        )

    def get_prefix(self, layer_idx: int):
        # (2, n_heads, prefix_len, head_dim)
        return self.prefix[layer_idx]

class CausalSelfAttentionWithPrefix(nn.Module):
    def __init__(self, d_model, n_heads, layer_idx, prefix_tuning):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.layer_idx = layer_idx
        self.prefix_tuning = prefix_tuning

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # أضف البادئة
        prefix = self.prefix_tuning.get_prefix(self.layer_idx)  # (2, n_heads, prefix_len, head_dim)
        prefix_k = prefix[0].unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_heads, prefix_len, head_dim)
        prefix_v = prefix[1].unsqueeze(0).expand(B, -1, -1, -1)

        k = torch.cat([prefix_k, k], dim=2)  # (B, n_heads, prefix_len + L, head_dim)
        v = torch.cat([prefix_v, v], dim=2)

        # قناع سببي: يجب أن يسمح للبادئة دائمًا
        # نستخدم is_causal=False مع قناع صريح
        attn_mask = self._build_mask(L, self.prefix_len, x.device)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)

    def _build_mask(self, seq_len, prefix_len, device):
        # (1, 1, seq_len, prefix_len + seq_len)
        # البادئة مسموحة دائمًا
        mask = torch.ones(seq_len, prefix_len + seq_len, device=device, dtype=torch.bool)
        # المواقع المستقبلية في النص الأصلي ممنوعة
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask[:, prefix_len:] = causal
        return mask.unsqueeze(0).unsqueeze(0)


# ════════ [4/7] model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B" ════════


from peft import PrefixTuningConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,      # طول البادئة
    prefix_projection=True,      # استخدم MLP للإسقاط (أفضل)
    encoder_hidden_size=512,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~1M || all params: ~1.2B || trainable%: ~0.08%


# ════════ [5/7] class PromptTuning(nn.Module): ════════


class PromptTuning(nn.Module):
    """
    Prompt Tuning: يضيف متجهات قابلة للتعلم إلى المدخل.
    """
    def __init__(self, d_model: int, prompt_len: int = 20):
        super().__init__()
        self.prompt_len = prompt_len
        # متجهات البرومبت (قابلة للتعلم)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_len, d_model) * 0.02
        )

    def forward(self, input_embeds):
        """
        input_embeds: (batch, seq, d_model)
        """
        B = input_embeds.size(0)
        # كرر البرومبت لكل عينة في الدفعة
        prompt = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prompt, input_embeds], dim=1)  # (B, prompt_len + seq, d_model)

class MiniGPTWithPromptTuning(nn.Module):
    def __init__(self, base_model, prompt_len=20):
        super().__init__()
        self.base_model = base_model
        self.prompt_tuning = PromptTuning(base_model.d_model, prompt_len)

        # جمّد كل معاملات النموذج الأساسي
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, targets=None):
        # احصل على التضمينات
        x = self.base_model.token_embedding(input_ids)
        # أضف البرومبت
        x = self.prompt_tuning(x)
        # مرر عبر الكتل
        for block in self.base_model.blocks:
            x = block(x)
        x = self.base_model.norm_f(x)
        logits = self.base_model.lm_head(x)

        loss = None
        if targets is not None:
            # انتبه: البرومبت يزيح المواقع!
            # نحتاج إلى إضافة -100 للبرومبت في labels
            prompt_len = self.prompt_tuning.prompt_len
            extended_targets = torch.cat([
                torch.full((targets.size(0), prompt_len), -100, device=targets.device),
                targets,
            ], dim=1)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                extended_targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ════════ [6/7] class IA3Linear(nn.Module): ════════


class IA3Linear(nn.Module):
    """
    طبقة خطية مع IA3: تضرب المخرج في متجه قابل للتعلم.
    """
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base = base_linear
        # متجه قابل للتعلم، مهيأ بـ 1 (لا تغيير في البداية)
        self.ia3_vector = nn.Parameter(torch.ones(base_linear.out_features))

        # جمّد الأوزان الأساسية
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base(x) * self.ia3_vector

def apply_ia3_to_model(model):
    """
    يستبدل K وV وFFN down بـ IA3Linear.
    """
    for name, module in model.named_modules():
        # K و V في الانتباه
        if name.endswith("k_proj") or name.endswith("v_proj"):
            parent_name, _, attr_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, IA3Linear(module))
        # FFN down
        elif name.endswith("down_proj"):
            parent_name, _, attr_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, IA3Linear(module))
    return model


# ════════ [7/7] # في الممارسة، هذا معقد وقد لا يعمل بشكل جيد ════════


# في الممارسة، هذا معقد وقد لا يعمل بشكل جيد
# لكن الفكرة: أضف LoRA للأوزان، وأضف Prompt Tuning للمدخل
from peft import LoraConfig, get_peft_model, PromptTuningConfig

# لا يمكن دمجها بسهولة في peft مباشرة
# تحتاج إلى تطبيق يدوي، أو استخدام حلول مخصصة
