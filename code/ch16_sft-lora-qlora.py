# -*- coding: utf-8 -*-
"""
الفصل 16: الضبط الدقيق (SFT) والضبط الفعّال (LoRA/QLoRA) — كيف تعلّم نموذجًا أن يطيع دون أن تعيد بناءه

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 16 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch16_sft-lora-qlora.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch16_sft-lora-qlora.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] def format_sft_example(example, tokenizer): ════════


import torch
from transformers import AutoTokenizer

def format_sft_example(example, tokenizer):
    """
    example: {"instruction": "...", "response": "..."}
    يُرجع input_ids و labels مع -100 على رموز التعليمات.
    """
    # بناء النص الكامل باستخدام chat template
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # ترميز النص الكامل
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # لبناء labels، نكرر الترميز على التعليمات فقط لمعرفة طولها
    prompt_messages = [{"role": "user", "content": example["instruction"]}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ════════ [2/6] model = AutoModelForCausalLM.from_pretrained( ════════


from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,          # alpha = r
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",            # لا تدرب bias
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# مثال: trainable params: 2,162,688 || all params: 1,238,468,608 || trainable%: 0.17%


# ════════ [3/6] bnb_config = BitsAndBytesConfig( ════════


from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4
    bnb_4bit_use_double_quant=True,      # التكميم المزدوج
    bnb_4bit_compute_dtype=torch.bfloat16, # الحساب في BF16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,
    device_map="auto",
)

# يجب استدعاء هذا قبل get_peft_model!
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)


# ════════ [4/6] dataset = load_dataset("json", data_files="sft_data.jsonl", split="tra ════════


from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

training_args = SFTConfig(
    output_dir="./sft_lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    packing=False,                # ابدأ بدون packing
    dataset_text_field="text",    # أو استخدم formatting_func
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./sft_lora_final")


# ════════ [5/6] def train_lora(model, train_dataset, tokenizer, config): ════════


from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

def train_lora(model, train_dataset, tokenizer, config):
    device = "cuda"
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=0.01,
    )

    def collate_fn(batch):
        # batch: قائمة من dicts مع input_ids و labels
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            input_ids[i, :n] = x["input_ids"]
            labels[i, :n] = x["labels"]
            attention_mask[i, :n] = 1
        return input_ids, labels, attention_mask

    loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                        shuffle=True, collate_fn=collate_fn)

    total_steps = len(loader) * config["epochs"] // config["accum_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(config["epochs"]):
        for step, (input_ids, labels, attention_mask) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config["accum_steps"]

            loss.backward()

            if (step + 1) % config["accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item() * config['accum_steps']:.4f}")

    return model


# ════════ [6/6] base_model = AutoModelForCausalLM.from_pretrained( ════════


from peft import PeftModel

# تحميل النموذج الأساسي (بدون تكميم إذا أردت الدمج)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# تحميل المحولات
peft_model = PeftModel.from_pretrained(base_model, "./sft_lora_final")

# دمج
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
