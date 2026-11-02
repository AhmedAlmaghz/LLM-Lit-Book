# -*- coding: utf-8 -*-
"""
الفصل 34: Self-Improvement وSynthetic Data — كيف يتفوق النموذج على نفسه

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السابع · الفصل 34 · عدد كتل بايثون في هذا الملف: 5

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch34_self-improvement-synthetic-data.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch34_self-improvement-synthetic-data.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/5] client = OpenAI() ════════


from openai import OpenAI
import json
import random

client = OpenAI()

def generate_instructions(seed_instructions: list[str], num_new: int = 10) -> list[str]:
    """
    يولّد تعليمات جديدة بناءً على بذور.
    """
    prompt = f"""لديك هذه التعليمات كأمثلة:
{chr(10).join(f'- {s}' for s in seed_instructions)}

ولّد {num_new} تعليمات جديدة، متنوعة، بالعربية، تشبه الأمثلة لكنها مختلفة.
أعد قائمة JSON فقط، دون أي نص آخر: ["تعليمات 1", "تعليمات 2", ...]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return []

def generate_answer(instruction: str) -> str:
    """يولّد إجابة على تعليمات."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": instruction}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# مثال
seeds = [
    "اكتب فقرة عن أهمية القراءة.",
    "لخص هذا النص في جملتين.",
    "اشرح الفرق بين الذكاء الاصطناعي والتعلم الآلي.",
]

new_instructions = generate_instructions(seeds, num_new=5)
for inst in new_instructions:
    answer = generate_answer(inst)
    print(f"Q: {inst}\nA: {answer[:100]}...\n")


# ════════ [2/5] def self_data_distillation_step( ════════


def self_data_distillation_step(
    model, tokenizer, prompts, optimizer, temperature=2.0, max_new_tokens=256
):
    """
    خطوة Self-Data Distillation:
    1. النموذج يولد إجابات (rollouts).
    2. النموذج يتدرب على هذه الإجابات (تقليد الذات).
    """
    model.train()

    # 1. توليد
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    # 2. تدريب على الإجابات المولدة
    # labels = outputs (مع -100 على الـ prompt إذا أردت)
    labels = outputs.clone()
    # اختياري: تجاهل الـ prompt
    labels[:, :inputs["input_ids"].size(1)] = -100

    logits = model(outputs).logits
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ════════ [3/5] def soft_bootstrap(question: str, model, tokenizer, rounds: int = 3): ════════


def soft_bootstrap(question: str, model, tokenizer, rounds: int = 3):
    """
    يحسن الإجابة تدريجيًا عبر جولات.
    """
    answer = ""
    history = []

    for round in range(rounds):
        if round == 0:
            prompt = question
        else:
            prompt = f"""السؤال: {question}

إجابتك السابقة: {answer}

حاول تحسين إجابتك. كن أكثر دقة، وضوحًا، واكتمالًا.
الإجابة المحسنة:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        history.append(answer)

    # اختر الأفضل (هنا: الأخيرة، لكن يمكنك استخدام LLM-as-a-Judge)
    return history[-1], history

# مثال
best, all_answers = soft_bootstrap("اشرح مفهوم الانتباه في المحوّلات.", model, tokenizer)
print(f"Best answer: {best}")


# ════════ [4/5] def generate_math_problem() -> tuple[str, str]: ════════


import re
import sympy as sp

def generate_math_problem() -> tuple[str, str]:
    """يولّد مسألة رياضية وإجابتها (بسيطة)."""
    import random
    a, b = random.randint(1, 100), random.randint(1, 100)
    op = random.choice(["+", "-", "*"])
    question = f"احسب: {a} {op} {b}"
    answer = eval(f"{a} {op} {b}")
    return question, str(answer)

def verify_math_answer(question: str, answer: str) -> bool:
    """يتحقق من صحة الإجابة باستخدام sympy."""
    try:
        # استخرج التعبير من السؤال
        expr = re.search(r"احسب:\s*(.+)", question).group(1)
        expected = sp.sympify(expr).evalf()
        given = sp.sympify(answer).evalf()
        return abs(expected - given) < 1e-6
    except Exception:
        return False

def generate_verified_math_data(model, tokenizer, num_samples: int = 1000):
    """يولّد بيانات رياضية، ويفلترها بالتحقق."""
    verified = []
    for _ in range(num_samples):
        question, correct_answer = generate_math_problem()
        # النموذج يحاول الإجابة
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
        model_answer = tokenizer.decode(output[0], skip_special_tokens=True)

        # تحقق
        if verify_math_answer(question, model_answer):
            verified.append({"question": question, "answer": model_answer})

    print(f"Verified: {len(verified)}/{num_samples}")
    return verified


# ════════ [5/5] def distinct_n(texts: list[str], n: int = 2) -> float: ════════


from collections import Counter
import re

def distinct_n(texts: list[str], n: int = 2) -> float:
    """
    Distinct-n: نسبة n-grams الفريدة إلى إجمالي n-grams.
    كلما زادت، زاد التنوع.
    """
    ngrams = []
    for text in texts:
        tokens = re.findall(r"\w+", text.lower())
        ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

# مثال
texts = ["القطة تجلس على السجادة", "القطة تنام على السجادة", "القطة تأكل"]
print(f"Distinct-1: {distinct_n(texts, 1):.3f}")
print(f"Distinct-2: {distinct_n(texts, 2):.3f}")
