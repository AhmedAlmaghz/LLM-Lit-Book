# -*- coding: utf-8 -*-
"""
الفصل 25: مدخل إلى النماذج متعددة الوسائط (Multimodal) — عندما يفتح النموذج عينيه

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السادس · الفصل 25 · عدد كتل بايثون في هذا الملف: 8

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch25_multimodal-intro.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch25_multimodal-intro.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/8] def load_vision_encoder(model_name="google/siglip-so400m-patch14-384") ════════


import torch
from transformers import AutoModel, AutoImageProcessor

def load_vision_encoder(model_name="google/siglip-so400m-patch14-384"):
    """
    يحمل Vision Encoder من HuggingFace.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, processor

def encode_image(image, model, processor, device="cuda"):
    """
    image: PIL.Image
    يُرجع: (num_patches, vision_dim)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (batch, num_patches, vision_dim)
    return outputs.last_hidden_state


# ════════ [2/8] class MLPProjector(nn.Module): ════════


import torch.nn as nn

class MLPProjector(nn.Module):
    """
    Projector بسيط: طبقتان خطيتان مع GELU.
    """
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        """
        x = self.fc1(vision_features)
        x = self.act(x)
        x = self.fc2(x)
        return x  # (batch, num_patches, llm_dim)


# ════════ [3/8] class PerceiverResampler(nn.Module): ════════


class PerceiverResampler(nn.Module):
    """
    يضغط الرموز البصرية إلى عدد ثابت من الرموز.
    """
    def __init__(self, vision_dim: int, llm_dim: int, num_latents: int = 64, num_heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, llm_dim) * 0.02)
        self.proj_in = nn.Linear(vision_dim, llm_dim) if vision_dim != llm_dim else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)

        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim),
        )

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        يُرجع: (batch, num_latents, llm_dim)
        """
        x = self.proj_in(vision_features)  # (B, P, D)
        B = x.size(0)

        latents = self.latents.expand(B, -1, -1)  # (B, L, D)

        # Cross-attention: latents (query) تنتبه إلى vision_features (key/value)
        h = self.norm1(latents)
        h, _ = self.cross_attn(h, x, x)
        latents = latents + h

        # FFN
        latents = latents + self.ffn(self.norm2(latents))
        return latents


# ════════ [4/8] class SimpleVLM(nn.Module): ════════


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleVLM(nn.Module):
    """
    VLM بسيط: Vision Encoder + Projector + LLM.
    """
    def __init__(
        self,
        vision_encoder,
        llm,
        projector: nn.Module,
        image_token_id: int,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.projector = projector
        self.image_token_id = image_token_id

        # جمّد Vision Encoder و LLM
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        """
        input_ids: (batch, seq) — يحتوي على image_token_id في موضع الصورة
        pixel_values: (batch, 3, H, W)
        """
        # 1. ترميز الصورة
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        # vision_features: (batch, num_patches, vision_dim)

        # 2. إسقاط إلى فضاء LLM
        visual_tokens = self.projector(vision_features)
        # visual_tokens: (batch, num_patches, llm_dim)

        # 3. تضمين النص
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # text_embeds: (batch, seq, llm_dim)

        # 4. استبدال image_token_id بتضمينات الصورة
        # نفترض أن كل صورة لها موضع واحد في input_ids (image_token_id)
        batch_size = input_ids.size(0)
        new_embeds = []
        for i in range(batch_size):
            # ابحث عن موضع image_token
            img_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                new_embeds.append(text_embeds[i])
                continue

            pos = img_pos[0].item()
            # قبل الصورة، الصورة، بعد الصورة
            before = text_embeds[i, :pos]            # (pos, D)
            after = text_embeds[i, pos+1:]           # (seq-pos-1, D)
            combined = torch.cat([before, visual_tokens[i], after], dim=0)
            new_embeds.append(combined)

        # ملاحظة: الأطوال مختلفة الآن! نحتاج إلى حشو أو التعامل مع ذلك.
        # في التطبيق الحقيقي، نستخدم attention_mask ويتم التعامل مع الأطوال.
        # للتبسيط، نفترض أن كل العينات لها نفس العدد من image_tokens.
        inputs_embeds = torch.stack(new_embeds, dim=0)

        # 5. مرر عبر LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs


# ════════ [5/8] model_name = "Qwen/Qwen2-VL-2B-Instruct" ════════


from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests

# تحميل نموذج VLM جاهز
model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# تحميل صورة
image = Image.open("path/to/image.jpg")

# بناء المحادثة
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "صف هذه الصورة بالتفصيل."},
        ],
    }
]

# معالجة
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# توليد
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# فك الترميز
response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(response)


# ════════ [6/8] class ImageTextDataset(Dataset): ════════


from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

class ImageTextDataset(Dataset):
    """
    بيانات: (صورة، تعليق).
    """
    def __init__(self, pairs, processor, image_token="<image>"):
        self.pairs = pairs
        self.processor = processor
        self.image_token = image_token

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, caption = self.pairs[idx]
        # بناء النص: <image> + التعليق
        text = f"{self.image_token}\n{caption}"
        inputs = self.processor(
            text=text, images=image,
            return_tensors="pt", padding="max_length", truncation=True, max_length=512,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # labels = input_ids، مع -100 على رموز الصورة
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

def train_projector(vlm, dataset, epochs=1, lr=1e-3, batch_size=4):
    """
    يدرب Projector فقط (LLM و Vision Encoder مجمدان).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # فقط معاملات Projector
    optimizer = torch.optim.AdamW(vlm.projector.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)

    vlm.projector.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(vlm.llm.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = vlm(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return vlm


# ════════ [7/8] def evaluate_vqa(model, processor, dataset, max_samples=100): ════════


def evaluate_vqa(model, processor, dataset, max_samples=100):
    """
    تقييم بسيط على VQA.
    """
    correct = 0
    total = 0
    for i, (image, question, answer) in enumerate(dataset):
        if i >= max_samples:
            break
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        pred = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if pred.lower() == answer.lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


# ════════ [8/8] def test_arabic_ocr(model, processor, image): ════════


def test_arabic_ocr(model, processor, image):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "اقرأ كل النص العربي في هذه الصورة، وأعد النص كما هو."},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
