# -*- coding: utf-8 -*-
"""
الفصل 17: ضبط التفضيلات (RLHF وDPO) — كيف تعلّم النموذج أن يختار "الأفضل" لا "الصحيح" فقط

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 17 · عدد كتل بايثون في هذا الملف: 8

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch17_rlhf-dpo.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch17_rlhf-dpo.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/8] class RewardModel(nn.Module): ════════


import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class RewardModel(nn.Module):
    """
    نموذج مكافأة: يأخذ (prompt + response)، ويعيد درجة عددية.
    """
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B"):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # (batch,)

def reward_model_loss(chosen_rewards, rejected_rewards):
    """
    خسارة Bradley-Terry: نريد chosen_rewards > rejected_rewards.
    """
    return -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()


# ════════ [2/8] def dpo_loss( ════════


import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (batch,)
    policy_rejected_logps: torch.Tensor,  # (batch,)
    ref_chosen_logps: torch.Tensor,       # (batch,)
    ref_rejected_logps: torch.Tensor,     # (batch,)
    beta: float = 0.1,
):
    """
    حساب خسارة DPO.
    logps: مجموع لوغاريتمات الاحتمالات لتسلسل الإجابة (بعد قناع -100 على التعليمات).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    # للتحليل: هامش التفضيل الضمني
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    margin = (chosen_rewards - rejected_rewards).mean()

    return loss, margin


# ════════ [3/8] def compute_logps(model, input_ids, labels, attention_mask): ════════


def compute_logps(model, input_ids, labels, attention_mask):
    """
    يحسب مجموع لوغاريتمات الاحتمالات للرموز التي ليست -100.
    """
    with torch.no_grad() if model.training is False else torch.enable_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # logits: (batch, seq, vocab)
    # نزيح: logits[:, :-1] تتنبأ بـ labels[:, 1:]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    # log_softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # نجمع فقط حيث labels != -100
    mask = (labels != -100).float()
    # gathered: (batch, seq)
    gathered = log_probs.gather(dim=-1, index=labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)  # (batch,)


# ════════ [4/8] model = AutoModelForCausalLM.from_pretrained( ════════


from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# بيانات تفضيل: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("json", data_files="preference_data.jsonl", split="train")

training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # الافتراضي الآمن
    learning_rate=5e-7,          # أقل بكثير من SFT!
    num_train_epochs=1,          # 1-2 حقب فقط
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    loss_type="sigmoid",         # DPO القياسي
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,              # TRL سينشئ مرجعًا تلقائيًا من النموذج الأساسي
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./dpo_final")


# ════════ [5/8] config = ORPOConfig( ════════


# في TRL: استخدم ORPOTrainer
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    output_dir="./orpo_output",
    learning_rate=8e-6,          # أعلى من DPO قليلًا
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
)
trainer = ORPOTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()


# ════════ [6/8] config = KTOConfig( ════════


from trl import KTOTrainer, KTOConfig

config = KTOConfig(
    output_dir="./kto_output",
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
)
trainer = KTOTrainer(model=model, args=config, train_dataset=kto_dataset, tokenizer=tokenizer)
trainer.train()


# ════════ [7/8] config = GRPOConfig( ════════


from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./grpo_output",
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,           # عدد الإجابات لكل تعليمات
    bf16=True,
)
trainer = GRPOTrainer(model=model, args=config, train_dataset=grpo_dataset, tokenizer=tokenizer)
trainer.train()


# ════════ [8/8] def format_preference_example(example, tokenizer): ════════


def format_preference_example(example, tokenizer):
    """
    example: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(example["chosen"], add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(example["rejected"], add_special_tokens=False)["input_ids"]

    # input_ids: prompt + chosen (أو rejected)
    # labels: -100 على prompt، ثم رموز الإجابة
    chosen_input = prompt_ids + chosen_ids
    chosen_labels = [-100] * len(prompt_ids) + chosen_ids

    rejected_input = prompt_ids + rejected_ids
    rejected_labels = [-100] * len(prompt_ids) + rejected_ids

    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": chosen_input,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input,
        "rejected_labels": rejected_labels,
    }
