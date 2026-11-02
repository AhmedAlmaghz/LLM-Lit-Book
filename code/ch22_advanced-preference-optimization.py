# -*- coding: utf-8 -*-
"""
الفصل 22: ضبط التفضيلات المتقدم — ORPO وKTO وSimPO وGRPO: عائلة DPO وأبناء العمومة

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الخامس · الفصل 22 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch22_advanced-preference-optimization.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch22_advanced-preference-optimization.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] def orpo_loss( ════════


import torch
import torch.nn.functional as F

def orpo_loss(
    policy_chosen_logps: torch.Tensor,    # (batch,) مجموع log P(y_w|x)
    policy_rejected_logps: torch.Tensor,  # (batch,) مجموع log P(y_l|x)
    chosen_lengths: torch.Tensor,         # (batch,) طول y_w
    rejected_lengths: torch.Tensor,       # (batch,) طول y_l
    lambda_or: float = 0.1,
):
    """
    خسارة ORPO: SFT + عقوبة نسبة الأرجحية.
    """
    # 1. SFT: متوسط log-prob لكل رمز (لتجنب تحيز الطول)
    chosen_logps_per_token = policy_chosen_logps / chosen_lengths
    sft_loss = -chosen_logps_per_token.mean()

    # 2. عقوبة التفضيل: log-odds ratio
    # log-odds = log(p) - log(1-p)
    # لكن p هنا هو احتمال التسلسل كاملًا، وهو صغير جدًا.
    # عمليًا، نستخدم الفرق في log-probs (تقريب).
    log_odds_chosen = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps).clamp(max=1 - 1e-6))
    log_odds_rejected = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps).clamp(max=1 - 1e-6))

    log_odds_ratio = log_odds_chosen - log_odds_rejected
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    return sft_loss + lambda_or * or_loss, sft_loss, or_loss


# ════════ [2/6] model = AutoModelForCausalLM.from_pretrained( ════════


from trl import ORPOTrainer, ORPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# بيانات: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("json", data_files="preference_data.jsonl", split="train")

config = ORPOConfig(
    output_dir="./orpo_output",
    learning_rate=8e-6,              # أعلى من DPO قليلًا
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    lambda_or=0.1,                   # معامل عقوبة التفضيل
    max_length=1024,
    max_prompt_length=512,
)

trainer = ORPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./orpo_final")


# ════════ [3/6] dataset = load_dataset("json", data_files="kto_data.jsonl", split="tra ════════


from trl import KTOTrainer, KTOConfig
from datasets import load_dataset

# بيانات: {"prompt": "...", "completion": "...", "label": true/false}
# label=true يعني "جيدة" (thumbs up)، false يعني "سيئة" (thumbs down)
dataset = load_dataset("json", data_files="kto_data.jsonl", split="train")

config = KTOConfig(
    output_dir="./kto_output",
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    beta=0.1,
    desirable_weight=1.0,            # وزن الإجابات الجيدة
    undesirable_weight=1.0,          # وزن الإجابات السيئة
    max_length=1024,
    max_prompt_length=512,
)

trainer = KTOTrainer(
    model=model,
    ref_model=None,                  # TRL سينشئ مرجعًا تلقائيًا
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./kto_final")


# ════════ [4/6] def simpo_loss( ════════


def simpo_loss(
    policy_chosen_logps: torch.Tensor,    # (batch,) مجموع log P(y_w|x)
    policy_rejected_logps: torch.Tensor,  # (batch,) مجموع log P(y_l|x)
    chosen_lengths: torch.Tensor,         # (batch,)
    rejected_lengths: torch.Tensor,       # (batch,)
    beta: float = 2.0,
    gamma: float = 0.5,
):
    """
    خسارة SimPO: بدون مرجع، مع تطبيع الطول وهامش.
    """
    # متوسط log-prob لكل رمز (بدلًا من المجموع)
    chosen_avg_logps = policy_chosen_logps / chosen_lengths
    rejected_avg_logps = policy_rejected_logps / rejected_lengths

    # الفرق، مطبقًا بـ beta، مع هامش gamma
    logits = beta * (chosen_avg_logps - rejected_avg_logps) - gamma
    loss = -F.logsigmoid(logits).mean()

    return loss


# ════════ [5/6] config = CPOConfig( ════════


from trl import CPOTrainer, CPOConfig   # SimPO متاح عبر CPOTrainer

config = CPOConfig(
    output_dir="./simpo_output",
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    loss_type="simpo",               # مفتاح: simpo بدلًا من sigmoid
    cpo_alpha=0.0,                   # لا تستخدم CPO (لأننا نريد SimPO)
    simpo_gamma=0.5,                 # الهامش
    beta=2.0,                        # أعلى من DPO
    max_length=1024,
    max_prompt_length=512,
)

trainer = CPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()


# ════════ [6/6] dataset = load_dataset("json", data_files="math_problems.jsonl", split ════════


from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# بيانات: {"prompt": "...", "answer": "..."} (الإجابة الصحيحة)
# أو: {"prompt": "..."} مع دالة مكافأة مخصصة
dataset = load_dataset("json", data_files="math_problems.jsonl", split="train")

# دالة مكافأة: تتحقق من صحة الإجابة
import re
def reward_function(completions, prompts, **kwargs):
    """
    completions: list[str] - الإجابات المولدة
    prompts: list[str] - الأسئلة
    **kwargs: يحتوي على "answer" (الإجابة الصحيحة) من البيانات
    """
    rewards = []
    for completion, gold in zip(completions, kwargs["answer"]):
        # استخرج الرقم النهائي من الإجابة
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if match:
            pred = match.group(1).strip()
            rewards.append(1.0 if pred == gold.strip() else 0.0)
        else:
            rewards.append(0.0)
    return rewards

config = GRPOConfig(
    output_dir="./grpo_output",
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,       # صغير، لأننا نولد G إجابات
    gradient_accumulation_steps=8,
    num_generations=8,                    # G: عدد الإجابات لكل تعليمات
    max_completion_length=512,
    max_prompt_length=512,
    bf16=True,
    beta=0.04,                            # معامل KL
    loss_type="grpo",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    reward_funcs=reward_function,
    tokenizer=tokenizer,
)
trainer.train()
