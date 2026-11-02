# -*- coding: utf-8 -*-
"""
الفصل 27: بناء VLM عمليًا من نموذج موجود — من الصفر إلى نموذج يرى بالعربية

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السادس · الفصل 27 · عدد كتل بايثون في هذا الملف: 13

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch27_build-vlm-arabic.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch27_build-vlm-arabic.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/13] AutoModelForCausalLM, AutoTokenizer, ════════


import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModel, AutoImageProcessor,
)
import torch.nn as nn

def load_components(device="cuda"):
    # 1. LLM
    llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.bfloat16, device_map=device
    )

    # 2. Vision Encoder
    vision_name = "google/siglip-so400m-patch14-384"
    vision_processor = AutoImageProcessor.from_pretrained(vision_name)
    vision_encoder = AutoModel.from_pretrained(
        vision_name, torch_dtype=torch.bfloat16
    ).to(device)

    return llm, tokenizer, vision_encoder, vision_processor

llm, tokenizer, vision_encoder, vision_processor = load_components()
print(f"LLM hidden dim: {llm.config.hidden_size}")          # 1536
print(f"Vision hidden dim: {vision_encoder.config.hidden_size}")  # 1152


# ════════ [2/13] class MLPProjector(nn.Module): ════════


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ════════ [3/13] class PooledProjector(nn.Module): ════════


class PooledProjector(nn.Module):
    """
    Projector مع AvgPool: يقلل الرموز من 729 إلى 144 (عبر pool_size=3).
    """
    def __init__(self, vision_dim, llm_dim, pool_size=3):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.pool = nn.AvgPool2d(pool_size)  # 27x27 -> 9x9

    def forward(self, x):
        # x: (B, 729, 1152) -> (B, 27, 27, 1152) -> (B, 1152, 27, 27)
        B, N, D = x.shape
        side = int(N ** 0.5)
        x = x.view(B, side, side, D).permute(0, 3, 1, 2)
        x = self.pool(x)  # (B, D, 9, 9)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, D)  # (B, 81, D)
        return self.fc2(self.act(self.fc1(x)))


# ════════ [4/13] class ImageCaptionDataset(Dataset): ════════


from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    """
    المرحلة 1: (صورة، تعليق).
    """
    def __init__(self, pairs, processor, tokenizer, image_token="<image>", max_length=512):
        self.pairs = pairs  # list of (image_path, caption)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length

        # أضف <image> كرمز خاص
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if self.image_token_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, caption = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")

        # النص: <image>\n{تعليق}
        text = f"{self.image_token}\n{caption}"

        # معالجة الصورة
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        # ترميز النص
        text_inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length,
            truncation=True, padding="max_length"
        )
        input_ids = text_inputs["input_ids"][0]
        attention_mask = text_inputs["attention_mask"][0]

        # labels: -100 على الحشو، و-100 على <image> (لا نتنبأ بالصورة)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


# ════════ [5/13] class ArabicVLM(nn.Module): ════════


class ArabicVLM(nn.Module):
    def __init__(self, vision_encoder, llm, projector):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.projector = projector

        # جمّد Vision Encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        # جمّد LLM (سنستخدم LoRA لاحقًا)
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        # 1. ترميز الصورة
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=pixel_values)
            vision_features = vision_out.last_hidden_state  # (B, 729, 1152)

        # 2. إسقاط
        visual_tokens = self.projector(vision_features)  # (B, 729, llm_dim)

        # 3. تضمين النص
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq, llm_dim)

        # 4. استبدال <image> برموز بصرية
        B, seq_len, D = text_embeds.shape
        num_visual = visual_tokens.size(1)

        # نفترض أن كل عينة لها <image> واحد
        new_embeds = []
        new_attention = []
        new_labels = []

        for i in range(B):
            # موضع <image>
            img_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_positions) == 0:
                new_embeds.append(text_embeds[i])
                new_attention.append(attention_mask[i] if attention_mask is not None else torch.ones(seq_len, device=input_ids.device))
                new_labels.append(labels[i] if labels is not None else torch.full((seq_len,), -100, device=input_ids.device))
                continue

            pos = img_positions[0].item()
            before = text_embeds[i, :pos]
            after = text_embeds[i, pos+1:]

            combined = torch.cat([before, visual_tokens[i], after], dim=0)
            new_embeds.append(combined)

            # attention_mask: 1 للصورة، ثم النص
            if attention_mask is not None:
                attn_before = attention_mask[i, :pos]
                attn_visual = torch.ones(num_visual, device=input_ids.device)
                attn_after = attention_mask[i, pos+1:]
                new_attention.append(torch.cat([attn_before, attn_visual, attn_after]))
            else:
                new_attention.append(torch.ones(combined.size(0), device=input_ids.device))

            # labels: -100 للصورة
            if labels is not None:
                lab_before = labels[i, :pos]
                lab_visual = torch.full((num_visual,), -100, device=input_ids.device)
                lab_after = labels[i, pos+1:]
                new_labels.append(torch.cat([lab_before, lab_visual, lab_after]))

        # حشو إلى أقصى طول (لأن الأطوال قد تختلف)
        max_len = max(e.size(0) for e in new_embeds)
        padded_embeds = torch.zeros(B, max_len, D, device=input_ids.device, dtype=text_embeds.dtype)
        padded_attn = torch.zeros(B, max_len, device=input_ids.device, dtype=torch.long)
        padded_labels = torch.full((B, max_len), -100, device=input_ids.device, dtype=torch.long)

        for i in range(B):
            L = new_embeds[i].size(0)
            padded_embeds[i, :L] = new_embeds[i]
            padded_attn[i, :L] = new_attention[i]
            if labels is not None:
                padded_labels[i, :L] = new_labels[i]

        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attn,
            labels=padded_labels if labels is not None else None,
        )
        return outputs


# ════════ [6/13] def train_alignment(vlm, dataset, epochs=1, batch_size=2, lr=1e-3, dev ════════


from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

def train_alignment(vlm, dataset, epochs=1, batch_size=2, lr=1e-3, device="cuda"):
    """
    المرحلة 1: تدريب Projector فقط.
    """
    vlm = vlm.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(vlm.projector.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)

    vlm.projector.train()
    vlm.llm.eval()
    vlm.vision_encoder.eval()

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = vlm(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    return vlm


# ════════ [7/13] def add_lora_to_llm(llm): ════════


from peft import LoraConfig, get_peft_model

def add_lora_to_llm(llm):
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(llm, config)

# استخدم LoRA
vlm.llm = add_lora_to_llm(vlm.llm)
vlm.llm.print_trainable_parameters()


# ════════ [8/13] class VQADataset(Dataset): ════════


class VQADataset(Dataset):
    """
    المرحلة 2: (صورة، سؤال، إجابة).
    """
    def __init__(self, samples, processor, tokenizer, image_token="<image>", max_length=1024):
        self.samples = samples
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, question, answer = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        # بناء النص: <image>\nسؤال: {question}\nجواب: {answer}
        prompt = f"{self.image_token}\nسؤال: {question}\nجواب: "
        full = prompt + answer

        # labels: -100 على prompt (بما فيها <image>)، ثم رموز الإجابة
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False, max_length=self.max_length, truncation=True)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # تأكد من الطول
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


# ════════ [9/13] def train_sft(vlm, dataset, epochs=3, batch_size=1, lr=2e-4, device="c ════════


def train_sft(vlm, dataset, epochs=3, batch_size=1, lr=2e-4, device="cuda", accum_steps=8):
    vlm = vlm.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # فقط معاملات Projector + LoRA
    trainable = list(vlm.projector.parameters()) + [p for p in vlm.llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs // accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, 20, total_steps)

    vlm.projector.train()
    vlm.llm.train()

    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = vlm(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item() * accum_steps:.4f}")

    return vlm


# ════════ [10/13] @torch.no_grad() ════════


@torch.no_grad()
def generate_answer(vlm, image: Image.Image, question: str, processor, tokenizer, max_new_tokens=100):
    vlm.eval()
    device = next(vlm.parameters()).device

    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt = f"<image>\nسؤال: {question}\nجواب: "
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # احصل على التضمينات
    vision_features = vlm.vision_encoder(pixel_values=pixel_values).last_hidden_state
    visual_tokens = vlm.projector(vision_features)

    text_embeds = vlm.llm.get_input_embeddings()(input_ids)

    # استبدال <image>
    img_pos = (input_ids[0] == vlm.image_token_id).nonzero(as_tuple=True)[0][0].item()
    combined = torch.cat([
        text_embeds[:, :img_pos],
        visual_tokens,
        text_embeds[:, img_pos+1:],
    ], dim=1)

    # توليد
    output_ids = vlm.llm.generate(
        inputs_embeds=combined,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ════════ [11/13] def evaluate_ocr(vlm, samples, processor, tokenizer, max_samples=20): ════════


def evaluate_ocr(vlm, samples, processor, tokenizer, max_samples=20):
    """
    samples: list of (image_path, expected_text)
    """
    correct = 0
    total = 0
    for image_path, expected in samples[:max_samples]:
        image = Image.open(image_path).convert("RGB")
        pred = generate_answer(vlm, image, "اقرأ كل النص في الصورة.", processor, tokenizer)
        # تنظيف بسيط
        pred = pred.strip()
        if expected.strip() in pred or pred in expected.strip():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


# ════════ [12/13] def merge_and_save(vlm, save_path="./arabic_vlm"): ════════


from peft import PeftModel

def merge_and_save(vlm, save_path="./arabic_vlm"):
    # دمج LoRA في LLM
    merged_llm = vlm.llm.merge_and_unload()

    # حفظ LLM
    merged_llm.save_pretrained(f"{save_path}/llm")
    vlm.tokenizer.save_pretrained(f"{save_path}/llm")

    # حفظ Projector
    torch.save(vlm.projector.state_dict(), f"{save_path}/projector.pt")

    # حفظ Vision Encoder (أو استخدم اسم النموذج الأصلي)
    vlm.vision_encoder.save_pretrained(f"{save_path}/vision_encoder")
    vlm.vision_processor.save_pretrained(f"{save_path}/vision_encoder")

    print(f"Saved to {save_path}")


# ════════ [13/13] bnb_config = BitsAndBytesConfig( ════════


from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# تحميل النموذج المدمج بـ 4-bit
llm_4bit = AutoModelForCausalLM.from_pretrained(
    f"{save_path}/llm",
    quantization_config=bnb_config,
    device_map="auto",
)
