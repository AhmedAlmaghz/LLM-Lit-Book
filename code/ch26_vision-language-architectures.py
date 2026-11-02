# -*- coding: utf-8 -*-
"""
الفصل 26: معماريات Vision-Language المتقدمة — خمس فلسفات في رؤية واحدة

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السادس · الفصل 26 · عدد كتل بايثون في هذا الملف: 5

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch26_vision-language-architectures.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch26_vision-language-architectures.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/5] class LLaVAProjector(nn.Module): ════════


import torch
import torch.nn as nn

class LLaVAProjector(nn.Module):
    """
    MLP بسيط: Linear → GELU → Linear.
    هذا هو Projector المستخدم في LLaVA-1.5.
    """
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        """
        return self.fc2(self.act(self.fc1(vision_features)))


# ════════ [2/5] class QFormer(nn.Module): ════════


class QFormer(nn.Module):
    """
    Q-Former مبسط: 32 استعلامًا + Cross-Attention + Self-Attention.
    """
    def __init__(
        self,
        vision_dim: int = 1024,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries

        # الاستعلامات القابلة للتعلم
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # إسقاط رقع الصورة
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # طبقات Q-Former
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "norm1": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "norm2": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                "norm3": nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        يُرجع: (batch, num_queries, hidden_dim)
        """
        B = vision_features.size(0)
        x = self.vision_proj(vision_features)  # (B, P, H)
        q = self.queries.expand(B, -1, -1)      # (B, Q, H)

        for layer in self.layers:
            # Self-Attention على الاستعلامات
            h = layer["norm1"](q)
            h, _ = layer["self_attn"](h, h, h)
            q = q + h

            # Cross-Attention: استعلامات تنتبه إلى رقع الصورة
            h = layer["norm2"](q)
            h, _ = layer["cross_attn"](h, x, x)
            q = q + h

            # FFN
            q = q + layer["ffn"](layer["norm3"](q))

        return q


# ════════ [3/5] class FlamingoCrossAttention(nn.Module): ════════


class FlamingoCrossAttention(nn.Module):
    """
    Cross-Attention بين LLM والذاكرة البصرية.
    """
    def __init__(self, llm_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(llm_dim)
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # بوابة تان

    def forward(self, x, visual_memory, visual_mask=None):
        """
        x: (batch, seq, llm_dim) - تمثيل النص
        visual_memory: (batch, num_visual_tokens, llm_dim)
        """
        h = self.norm(x)
        attn_out, _ = self.cross_attn(
            query=h, key=visual_memory, value=visual_memory,
            key_padding_mask=visual_mask,
        )
        # بوابة تان: في البداية، 0، فالنموذج لا يتأثر
        return x + torch.tanh(self.gate) * attn_out


# ════════ [4/5] model_name = "Qwen/Qwen2.5-VL-7B-Instruct" ════════


from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

def ask_qwen(image: Image.Image, question: str) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# مثال: OCR عربي
image = Image.open("arabic_sign.jpg")
print(ask_qwen(image, "اقرأ كل النص العربي في هذه الصورة، وأعد النص كما هو."))


# ════════ [5/5] model_name = "microsoft/Florence-2-large" ════════


from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_name = "microsoft/Florence-2-large"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")

def florence_caption(image: Image.Image) -> str:
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    return processor.batch_decode(out, skip_special_tokens=True)[0]
