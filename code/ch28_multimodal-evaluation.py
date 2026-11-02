# -*- coding: utf-8 -*-
"""
الفصل 28: تقييم النماذج متعددة الوسائط — كيف تعرف أن النموذج “يرى” فعلًا؟

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السادس · الفصل 28 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch28_multimodal-evaluation.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch28_multimodal-evaluation.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] def llm_judge(image_path, question, answer, judge_model="gpt-4o-mini") ════════


def llm_judge(image_path, question, answer, judge_model="gpt-4o-mini"):
    """
    يستخدم GPT-4o لتقييم الإجابة.
    """
    from openai import OpenAI
    import base64

    client = OpenAI()
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""قيّم الإجابة التالية على مقياس 1-5:
- 5: دقيقة، كاملة، مستمدة من الصورة
- 3: صحيحة جزئيًا
- 1: خاطئة أو هلوسة

السؤال: {question}
الإجابة: {answer}

أعد رقمًا واحدًا فقط (1-5)."""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=10,
    )
    return int(response.choices[0].message.content.strip())


# ════════ [2/6] def evaluate_pope(vlm, samples, processor, tokenizer): ════════


def evaluate_pope(vlm, samples, processor, tokenizer):
    """
    samples: list of (image_path, object_name, exists: bool)
    """
    correct = 0
    total = 0
    false_positives = 0  # قال "نعم" لغير موجود

    for image_path, obj, exists in samples:
        image = Image.open(image_path).convert("RGB")
        question = f"هل يوجد {obj} في الصورة؟ أجب بـ 'نعم' أو 'لا' فقط."
        answer = generate_answer(vlm, image, question, processor, tokenizer).strip()

        pred_yes = "نعم" in answer
        if pred_yes == exists:
            correct += 1
        elif pred_yes and not exists:
            false_positives += 1
        total += 1

    accuracy = correct / total
    fp_rate = false_positives / total
    return {"accuracy": accuracy, "false_positive_rate": fp_rate}


# ════════ [3/6] def detect_hallucination_in_caption(image_path, caption, judge_model=" ════════


def detect_hallucination_in_caption(image_path, caption, judge_model="gpt-4o-mini"):
    """
    يستخدم GPT-4o للتحقق من كل جملة في الوصف.
    """
    from openai import OpenAI
    import base64

    client = OpenAI()
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""هل هذا الوصف دقيق بالكامل للصورة؟
إذا كان هناك أي شيء غير موجود في الصورة، اذكر ما هو.
إذا كان دقيقًا، أعد "دقيق".

الوصف: {caption}"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# ════════ [4/6] def character_error_rate(pred: str, gold: str) -> float: ════════


def character_error_rate(pred: str, gold: str) -> float:
    """
    CER: نسبة الأخطاء على مستوى الحرف.
    """
    import Levenshtein
    if not gold:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gold) / len(gold)

def evaluate_arabic_ocr(vlm, samples, processor, tokenizer):
    """
    samples: list of (image_path, expected_text)
    """
    cers = []
    for image_path, expected in samples:
        image = Image.open(image_path).convert("RGB")
        pred = generate_answer(vlm, image, "اقرأ كل النص العربي في الصورة.", processor, tokenizer)
        # تنظيف
        pred = pred.strip().replace("النص:", "").strip()
        cer = character_error_rate(pred, expected.strip())
        cers.append(cer)

    return {
        "mean_cer": sum(cers) / len(cers),
        "min_cer": min(cers),
        "max_cer": max(cers),
    }


# ════════ [5/6] def adversarial_test(vlm, image, processor, tokenizer): ════════


def adversarial_test(vlm, image, processor, tokenizer):
    """
    اختبارات عدائية على صورة واحدة.
    """
    tests = [
        # أسئلة سلبية
        ("هل يوجد فيل في هذه الصورة؟", "لا"),
        ("هل هذه صورة ليلية؟", None),  # لا نعرف
        # أسئلة عن تفاصيل
        ("ما النص المكتوب في الزاوية العلوية اليمنى؟", None),
        # سؤال محمل
        ("لماذا الشخص في الصورة حزين؟", "قد لا يكون حزينًا"),
    ]

    results = []
    for question, expected in tests:
        answer = generate_answer(vlm, image, question, processor, tokenizer)
        results.append({"question": question, "answer": answer, "expected": expected})

    return results


# ════════ [6/6] def human_eval_interface(samples): ════════


def human_eval_interface(samples):
    """
    واجهة بسيطة (CLI) لتقييم بشري.
    samples: list of (image_path, question, answer)
    """
    scores = []
    for image_path, question, answer in samples:
        print(f"\n{'='*60}")
        print(f"الصورة: {image_path}")
        print(f"السؤال: {question}")
        print(f"الإجابة: {answer}")
        print(f"{'='*60}")

        while True:
            score = input("قيّم (1-5): ").strip()
            if score in ["1", "2", "3", "4", "5"]:
                scores.append(int(score))
                break
            print("أدخل رقمًا من 1 إلى 5.")

    return {"mean_score": sum(scores) / len(scores), "scores": scores}
