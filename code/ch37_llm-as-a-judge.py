# -*- coding: utf-8 -*-
"""
الفصل 37: LLM-as-a-Judge — عندما يصبح النموذج قاضيًا على نفسه

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثامن · الفصل 37 · عدد كتل بايثون في هذا الملف: 10

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch37_llm-as-a-judge.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch37_llm-as-a-judge.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/10] client = OpenAI() ════════


from openai import OpenAI
import json

client = OpenAI()

def llm_judge_pointwise(
    question: str,
    answer: str,
    criteria: str = "الدقة، الفائدة، الوضوح",
    judge_model: str = "gpt-4o-mini",
) -> dict:
    """
    تقييم Pointwise: إجابة واحدة على مقياس 1-5.
    """
    prompt = f"""أنت مقيّم خبير. قيّم الإجابة التالية على مقياس 1-5.

السؤال: {question}

الإجابة: {answer}

المعايير: {criteria}

- 5: ممتازة (دقيقة، كاملة، واضحة، مفيدة)
- 4: جيدة (دقيقة، لكن قد تحتاج تحسينًا بسيطًا)
- 3: مقبولة (صحيحة جزئيًا، أو غير مكتملة)
- 2: ضعيفة (تحتوي أخطاء، أو غير ذات صلة)
- 1: سيئة (خاطئة، أو مضللة)

أعد JSON فقط:
{{"score": <1-5>, "reasoning": "<سبب مختصر>"}}"""

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)

def llm_judge_pairwise(
    question: str,
    answer_a: str,
    answer_b: str,
    judge_model: str = "gpt-4o-mini",
) -> dict:
    """
    تقييم Pairwise: أيهما أفضل، A أم B؟
    """
    prompt = f"""أنت مقيّم خبير. قارن الإجابتين التاليتين.

السؤال: {question}

الإجابة A: {answer_a}

الإجابة B: {answer_b}

أيهما أفضل؟ ضع في اعتبارك: الدقة، الاكتمال، الوضوح، الفائدة.

أعد JSON فقط:
{{"winner": "A" أو "B" أو "tie", "reasoning": "<سبب مختصر>"}}"""

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)


# ════════ [2/10] def detect_position_bias(judge_fn, samples): ════════


def detect_position_bias(judge_fn, samples):
    """
    يكشف Position Bias: هل الحكم يفضل A دائمًا؟
    """
    a_wins = 0
    b_wins = 0
    ties = 0

    for question, answer_a, answer_b in samples:
        # الحكم الأصلي (A أولًا، B ثانيًا)
        result1 = judge_fn(question, answer_a, answer_b)
        # الحكم المعكوس (B أولًا، A ثانيًا)
        result2 = judge_fn(question, answer_b, answer_a)

        # إذا قال "A" في الأول، و"A" في الثاني (الذي هو B الأصلي)، فهذا تفضيل للموضع
        if result1["winner"] == "A" and result2["winner"] == "A":
            a_wins += 1  # لم يتغير، قد يكون تفضيلًا حقيقيًا
        elif result1["winner"] == "A" and result2["winner"] == "B":
            a_wins += 1  # تفضيل حقيقي لـ A
        elif result1["winner"] == "B" and result2["winner"] == "A":
            b_wins += 1  # تفضيل حقيقي لـ B
        elif result1["winner"] == "B" and result2["winner"] == "B":
            b_wins += 1  # تفضيل حقيقي لـ B
        else:
            ties += 1

    # نسبة التفضيل للموضع الأول (A في الأول)
    # إذا كانت عالية، فهناك Position Bias
    position_a_pref = (a_wins + ties/2) / len(samples)

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "position_a_preference": position_a_pref,
        "has_position_bias": abs(position_a_pref - 0.5) > 0.1,  # أكثر من 10% انحراف
    }


# ════════ [3/10] def unbiased_pairwise_judge(judge_fn, question, answer_a, answer_b): ════════


def unbiased_pairwise_judge(judge_fn, question, answer_a, answer_b):
    """
    تقييم غير متحيز: قارن في الاتجاهين، وادمج النتائج.
    """
    result_ab = judge_fn(question, answer_a, answer_b)  # A أولًا
    result_ba = judge_fn(question, answer_b, answer_a)  # B أولًا

    # حول result_ba إلى نفس الإطار (A vs B)
    ba_winner = result_ba["winner"]
    if ba_winner == "A":
        ba_winner = "B"
    elif ba_winner == "B":
        ba_winner = "A"

    # إذا اتفق الاتجاهان، فالحكم ثابت
    if result_ab["winner"] == ba_winner:
        return {"winner": result_ab["winner"], "reasoning": result_ab["reasoning"]}
    else:
        # اختلاف: تعادل (أو إعادة تقييم)
        return {"winner": "tie", "reasoning": "اختلاف في الاتجاهين"}


# ════════ [4/10] def judge_with_cot(question, answer, judge_model="gpt-4o-mini"): ════════


def judge_with_cot(question, answer, judge_model="gpt-4o-mini"):
    prompt = f"""قيّم الإجابة التالية. اتبع هذه الخطوات:

1. حدد ما إذا كانت الإجابة تجيب على السؤال.
2. تحقق من صحة المعلومات.
3. قيّم الاكتمال (هل تغطي كل الجوانب؟).
4. قيّم الوضوح والأسلوب.
5. أعطِ درجة نهائية (1-5).

السؤال: {question}
الإجابة: {answer}

أعد JSON: {{"score": <1-5>, "reasoning": "<خطواتك>"}}"""
    # ...


# ════════ [5/10] criteria = """ ════════


criteria = """
- الدقة: هل المعلومات صحيحة؟ (وزن 40%)
- الاكتمال: هل تغطي كل الجوانب؟ (وزن 30%)
- الوضوح: هل سهلة الفهم؟ (وزن 20%)
- الأمان: هل خالية من محتوى ضار؟ (وزن 10%)
"""


# ════════ [6/10] examples = """ ════════


examples = """
مثال 1:
السؤال: ما عاصمة فرنسا؟
الإجابة: باريس.
الدرجة: 5 (دقيقة، كاملة، واضحة)

مثال 2:
السؤال: ما عاصمة فرنسا؟
الإجابة: باريس هي عاصمة فرنسا، وأكبر مدنها، ومركز الثقافة والفن. تقع على نهر السين...
الدرجة: 4 (دقيقة، لكن طويلة جدًا، فيها معلومات غير مطلوبة)

مثال 3:
السؤال: ما عاصمة فرنسا؟
الإجابة: لندن.
الدرجة: 1 (خاطئة تمامًا)
"""


# ════════ [7/10] def ensemble_judge(question, answer, judges=None): ════════


def ensemble_judge(question, answer, judges=None):
    if judges is None:
        judges = ["gpt-4o-mini", "claude-3-5-sonnet", "gemini-1.5-pro"]

    results = []
    for model in judges:
        result = llm_judge_pointwise(question, answer, judge_model=model)
        results.append(result)

    # متوسط الدرجات
    avg_score = sum(r["score"] for r in results) / len(results)
    return {"avg_score": avg_score, "individual": results}


# ════════ [8/10] def measure_agreement(llm_scores, human_scores): ════════


from sklearn.metrics import cohen_kappa_score

def measure_agreement(llm_scores, human_scores):
    """
    يقيس الاتفاق بين LLM والبشر.
    """
    # تحويل إلى فئات (منخفض، متوسط، مرتفع)
    def to_category(score):
        if score <= 2: return 0
        elif score == 3: return 1
        else: return 2

    llm_cat = [to_category(s) for s in llm_scores]
    human_cat = [to_category(s) for s in human_scores]

    kappa = cohen_kappa_score(llm_cat, human_cat)
    return {"kappa": kappa, "agreement": sum(a == b for a, b in zip(llm_cat, human_cat)) / len(llm_cat)}


# ════════ [9/10] def validate_judge(judge_fn, known_samples): ════════


def validate_judge(judge_fn, known_samples):
    """
    يختبر الحكم على عينات معروفة الإجابة (جيدة/سيئة).
    known_samples: list of (question, answer, expected_score)
    """
    correct = 0
    for question, answer, expected in known_samples:
        result = judge_fn(question, answer)
        # اعتبر صحيحًا إذا كان ضمن ±1
        if abs(result["score"] - expected) <= 1:
            correct += 1
    return correct / len(known_samples)


# ════════ [10/10] def evaluate_rag_triad(question, contexts, answer, judge_model="gpt-4o ════════


def evaluate_rag_triad(question, contexts, answer, judge_model="gpt-4o-mini"):
    """
    تقييم RAG: Context Relevance، Faithfulness، Answer Relevance.
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""قيّم الجوانب الثلاثة التالية لنظام RAG. أعد JSON فقط.

السؤال: {question}

السياق المسترجع:
{context_text}

الإجابة: {answer}

1. Context Relevance (1-5): هل السياق ذو صلة بالسؤال؟
2. Faithfulness (1-5): هل الإجابة مستمدة من السياق، دون هلوسة؟
3. Answer Relevance (1-5): هل الإجابة تجيب على السؤال؟

أعد JSON:
{{"context_relevance": <1-5>, "faithfulness": <1-5>, "answer_relevance": <1-5>, "reasoning": "<سبب مختصر>"}}"""

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
