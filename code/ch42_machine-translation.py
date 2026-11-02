# -*- coding: utf-8 -*-
"""
الفصل 42: الترجمة الآلية — عندما تخون الكلمة مترجمها

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء التاسع · الفصل 42 · عدد كتل بايثون في هذا الملف: 10

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch42_machine-translation.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch42_machine-translation.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/10] # مثال: الترجمة من الإنجليزية إلى العربية ════════


# مثال: الترجمة من الإنجليزية إلى العربية
# "The doctor said she was tired."
# النموذج قد يترجم: "قال الطبيب إنها متعبة." (الطبيب مذكر، لكن "she" مؤنث)
# الترجمة الصحيحة: "قالت الطبيبة إنها متعبة."


# ════════ [2/10] def load_nllb(model_name: str = "facebook/nllb-200-distilled-600M"): ════════


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def load_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    """
    يحمل NLLB للترجمة.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


# ════════ [3/10] def translate_nllb( ════════


def translate_nllb(
    text: str,
    model,
    tokenizer,
    src_lang: str = "arb_Arab",  # العربية
    tgt_lang: str = "eng_Latn",  # الإنجليزية
    max_length: int = 512,
) -> str:
    """
    يترجم نصًا باستخدام NLLB.
    """
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=max_length,
            num_beams=4,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# مثال
model, tokenizer = load_nllb()
print(translate_nllb("القطة تجلس على السجادة.", model, tokenizer))
# "The cat is sitting on the carpet."


# ════════ [4/10] client = OpenAI() ════════


from openai import OpenAI
client = OpenAI()

def translate_llm(text: str, src: str = "العربية", tgt: str = "الإنجليزية") -> str:
    """
    يترجم باستخدام LLM.
    """
    prompt = f"""ترجم النص التالي من {src} إلى {tgt}.
حافظ على المعنى، والأسلوب، والتعبيرات الثقافية.
لا تضف أي شرح أو تعليق. أعد الترجمة فقط.

النص: {text}

الترجمة:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ════════ [5/10] def translate_with_gender(text: str, gender: str = "auto") -> str: ════════


def translate_with_gender(text: str, gender: str = "auto") -> str:
    """
    يترجم مع التحكم في الجنس.
    gender: "male"، "female"، "auto"
    """
    gender_instruction = {
        "male": "استخدم الصيغة المذكرة عند الإشارة إلى الأشخاص.",
        "female": "استخدم الصيغة المؤنثة عند الإشارة إلى الأشخاص.",
        "auto": "حافظ على الجنس كما هو واضح من السياق.",
    }[gender]

    prompt = f"""ترجم النص التالي من الإنجليزية إلى العربية.
{gender_instruction}
لا تضف شرحًا.

النص: {text}

الترجمة:"""
    # ...


# ════════ [6/10] def translate_creative(text: str) -> str: ════════


def translate_creative(text: str) -> str:
    """ترجمة إبداعية: تحافظ على الأسلوب، وليس فقط المعنى."""
    prompt = f"""ترجم النص التالي من العربية إلى الإنجليزية.
هذه ترجمة إبداعية (أدبية). حافظ على:
- الأسلوب الأدبي
- الصور البلاغية
- الإيقاع
- التعبيرات الثقافية (ابحث عن مكافئ، وليس ترجمة حرفية)

النص: {text}

الترجمة:"""
    # ...


# ════════ [7/10] def compute_bleu(reference: str, candidate: str) -> float: ════════


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference: str, candidate: str) -> float:
    """BLEU: تشابه n-grams."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return sentence_bleu(
        [ref_tokens], cand_tokens,
        smoothing_function=SmoothingFunction().method1,
    )


# ════════ [8/10] def load_comet(model_name: str = "Unbabel/wmt22-comet-da"): ════════


from comet import download_model, load_from_checkpoint

def load_comet(model_name: str = "Unbabel/wmt22-comet-da"):
    """يحمل COMET."""
    model_path = download_model(model_name)
    return load_from_checkpoint(model_path)

def compute_comet(model, sources: list[str], hypotheses: list[str], references: list[str]):
    """COMET: تقييم دلالي."""
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=8, gpus=1)
    return output["system_score"]


# ════════ [9/10] def compute_bertscore(references: list[str], candidates: list[str], la ════════


from bert_score import score

def compute_bertscore(references: list[str], candidates: list[str], lang: str = "ar"):
    P, R, F1 = score(candidates, references, lang=lang)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}


# ════════ [10/10] def translate_image_text(vlm, image, processor, tokenizer, tgt_lang="E ════════


def translate_image_text(vlm, image, processor, tokenizer, tgt_lang="English"):
    """يقرأ النص في صورة ويترجمه."""
    prompt = f"اقرأ النص في هذه الصورة، ثم ترجمه إلى {tgt_lang}. أعد النص الأصلي والترجمة."
    # ...
