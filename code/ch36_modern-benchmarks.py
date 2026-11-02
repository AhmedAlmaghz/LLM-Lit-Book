# -*- coding: utf-8 -*-
"""
الفصل 36: المعايير الحديثة — MMLU وBIG-Bench وHELM: كيف نقيس "الذكاء العام"؟

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثامن · الفصل 36 · عدد كتل بايثون في هذا الملف: 5

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch36_modern-benchmarks.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch36_modern-benchmarks.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/5] def load_mmlu(subject: str = "abstract_algebra", split: str = "test"): ════════


from datasets import load_dataset
import re

def load_mmlu(subject: str = "abstract_algebra", split: str = "test"):
    """يحمل MMLU لمادة محددة."""
    ds = load_dataset("cais/mmlu", subject, split=split)
    return ds

def format_mmlu_prompt(example):
    """يبني prompt لـ MMLU."""
    question = example["question"]
    choices = example["choices"]
    return f"""أجب على السؤال التالي باختيار الحرف الصحيح فقط (A, B, C, D).

السؤال: {question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

الإجابة:"""

def evaluate_mmlu(model, tokenizer, subject="abstract_algebra", num_samples=100):
    """تقييم بسيط على MMLU."""
    ds = load_mmlu(subject)
    correct = 0
    total = 0

    for example in ds.select(range(min(num_samples, len(ds)))):
        prompt = format_mmlu_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        response = tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()

        # استخرج الحرف
        match = re.search(r"[ABCD]", response)
        if match:
            predicted = "ABCD".index(match.group())
            if predicted == example["answer"]:
                correct += 1
        total += 1

    return correct / total if total > 0 else 0


# ════════ [2/5] def load_bbh(task: str = "logical_deduction_three_objects"): ════════


from datasets import load_dataset

def load_bbh(task: str = "logical_deduction_three_objects"):
    """يحمل مهمة من BIG-Bench Hard."""
    ds = load_dataset("lukaemon/bbh", task, split="test")
    return ds

def evaluate_bbh(model, tokenizer, task="logical_deduction_three_objects", num_samples=50):
    ds = load_bbh(task)
    correct = 0
    total = 0

    for example in ds.select(range(min(num_samples, len(ds)))):
        prompt = example["input"]
        expected = example["target"].strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()

        if response.startswith(expected) or expected in response:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


# ════════ [3/5] def evaluate_calibration(model, tokenizer, dataset, num_samples=100): ════════


# HELM يستخدم إطار عمل معقد. هنا تقييم مبسط لبعض المقاييس.
import torch
import numpy as np

def evaluate_calibration(model, tokenizer, dataset, num_samples=100):
    """
    يقيس Calibration: هل ثقة النموذج تتناسب مع دقته؟
    """
    confidences = []
    correctness = []

    for example in dataset.select(range(min(num_samples, len(dataset)))):
        prompt = example["question"]
        expected = example["answer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            predicted = probs.argmax().item()

        confidences.append(confidence)
        correctness.append(1 if predicted == expected else 0)

    # Expected Calibration Error (ECE)
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(len(bins) - 1):
        bin_mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
        if bin_mask.sum() > 0:
            bin_conf = np.mean(np.array(confidences)[bin_mask])
            bin_acc = np.mean(np.array(correctness)[bin_mask])
            ece += (bin_mask.sum() / len(confidences)) * abs(bin_conf - bin_acc)

    return {"ece": ece}


# ════════ [4/5] def evaluate_mt_bench(model, tokenizer, questions, judge_model="gpt-4o ════════


def evaluate_mt_bench(model, tokenizer, questions, judge_model="gpt-4o"):
    """
    تقييم مبسط: النموذج يجيب، وGPT-4 يقيّم.
    """
    from openai import OpenAI
    client = OpenAI()

    scores = []
    for question in questions:
        # 1. النموذج يجيب
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        answer = tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

        # 2. GPT-4 يقيّم
        judge_prompt = f"""قيّم الإجابة التالية على مقياس 1-10:

السؤال: {question}
الإجابة: {answer}

المعايير: الدقة، العمق، الوضوح، الفائدة.
أعد رقمًا واحدًا فقط (1-10)."""

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=10,
        )
        try:
            score = int(response.choices[0].message.content.strip())
            scores.append(score)
        except:
            scores.append(0)

    return {"mean_score": sum(scores) / len(scores) if scores else 0, "scores": scores}


# ════════ [5/5] @dataclass ════════


import json
from dataclasses import dataclass

@dataclass
class EvalSample:
    task: str
    input: str
    expected: str  # أو None إذا كانت الإجابة مفتوحة
    criteria: list[str]  # معايير التقييم

def evaluate_custom(model, tokenizer, samples: list[EvalSample], judge_model="gpt-4o"):
    """
    تقييم مخصص: النموذج يجيب، والحكم يقيّم.
    """
    from openai import OpenAI
    client = OpenAI()

    results = []
    for sample in samples:
        # 1. إجابة النموذج
        inputs = tokenizer(sample.input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        answer = tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

        # 2. تقييم
        criteria_text = "\n".join(f"- {c}" for c in sample.criteria)
        judge_prompt = f"""قيّم الإجابة التالية على مقياس 1-5 لكل معيار:

المدخل: {sample.input}
الإجابة: {answer}

المعايير:
{criteria_text}

أعد JSON: {{"criterion1": score, "criterion2": score, ...}}"""

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=100,
        )

        try:
            scores = json.loads(response.choices[0].message.content)
        except:
            scores = {}

        results.append({
            "task": sample.task,
            "input": sample.input,
            "answer": answer,
            "expected": sample.expected,
            "scores": scores,
        })

    return results

# مثال
samples = [
    EvalSample(
        task="customer_support",
        input="عميلي يشتكي من تأخر الشحنة. اكتب ردًا مهذبًا.",
        expected=None,
        criteria=["مهذب", "يتعاطف", "يقدم حلًا", "واضح"],
    ),
    EvalSample(
        task="summarization",
        input="لخص هذا التقرير المالي: ...",
        expected="ملخص قصير",
        criteria=["دقيق", "موجز", "يشمل الأرقام الأساسية"],
    ),
]
