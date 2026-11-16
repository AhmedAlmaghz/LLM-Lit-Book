#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — تحليل المشاعر بالعربية — عندما تصبح "ما شاء الله" سخرية
Transformers and Large Language Models — Arabic Sentiment

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-41-llm-ds (lesson 41)
Code files: 51 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part09_practical-projects.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ خلية الإعداد — شغّلها أولًا (Colab / Kaggle / محليًا)
# تثبّت ما ينقص فقط، وتتجاهل ما لا يدعم جهازك.
import sys, subprocess, importlib

PKGS = [
    ("torch", "torch"), ("transformers", "transformers"),
    ("datasets", "datasets"), ("accelerate", "accelerate"),
    ("peft", "peft"), ("trl", "trl"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"), ("pandas", "pandas"),
    ("numpy", "numpy"), ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"), ("evaluate", "evaluate"),
]
todo = []
for mod, pipname in PKGS:
    try:
        importlib.import_module(mod)
    except Exception:
        todo.append(pipname)
if todo:
    print("تثبيت الحزم الناقصة:", todo)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *todo])
    except Exception as e:
        print("تعذّر تثبيت بعض الحزم (ثبّتها يدويًا إن احتجتها):", e)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print("device:", DEVICE)
print("جاهز ✅")

# ────────────────────────────────────────────────────────────────────────
# [002] code/notebooks/part09_practical-projects.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import re
import unicodedata

def normalize_arabic(text: str) -> str:
    """تطبيع عربي بسيط."""
    if not isinstance(text, str):
        return ""
    # إزالة التطويل
    text = re.sub(r"\u0640", "", text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # توحيد الهمزات
    text = re.sub(r"[إأآٱ]", "ا", text)
    # توحيد التاء المربوطة
    text = text.replace("ة", "ه")
    # توحيد الأرقام العربية
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    # إزالة الروابط
    text = re.sub(r"http\S+|www\S+", "", text)
    # إزالة الإشارات (@mentions)
    text = re.sub(r"@\w+", "", text)
    # توحيد المسافات
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset(path: str) -> pd.DataFrame:
    """يحمل بيانات ويطبع إحصاءات."""
    df = pd.read_csv(path)
    # افترض أن الأعمدة: text, label (positive/negative/neutral)
    df["text"] = df["text"].apply(normalize_arabic)
    df = df[df["text"].str.len() > 3]  # إزالة النصوص القصيرة جدًا
    df = df.drop_duplicates(subset=["text"])
    print(f"Loaded {len(df)} samples")
    print(df["label"].value_counts())
    return df

# مثال
# df = load_dataset("astd.csv")

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part09_practical-projects.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import emoji

def emoji_to_text(text: str) -> str:
    """يحول الإيموجي إلى وصف نصي."""
    return emoji.demojize(text, language="ar")

# مثال
print(emoji_to_text("الخدمة سيئة 😡"))  # "الخدمة سيئة :وجه_غاضب:"

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part09_practical-projects.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

def train_tfidf_baseline(df: pd.DataFrame):
    """يدرب خط أساس TF-IDF + LR."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            analyzer="word",
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    return pipeline, X_test, y_test

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part09_practical-projects.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch

def prepare_dataset(df: pd.DataFrame, tokenizer, label2id: dict, max_length: int = 128):
    """يحول DataFrame إلى Dataset."""
    ds = Dataset.from_pandas(df[["text", "label"]])

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.map(lambda x: {"labels": label2id[x["label"]]})
    ds = ds.remove_columns(["text", "label"])
    return ds

def train_small_model(
    df: pd.DataFrame,
    model_name: str = "aubmindlab/bert-base-arabertv02",
    output_dir: str = "./sentiment_model",
):
    """يدرب نموذج تصنيف مشاعر عربي."""
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # تقسيم
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_ds = prepare_dataset(train_df, tokenizer, label2id)
    val_ds = prepare_dataset(val_df, tokenizer, label2id)

    # إعدادات التدريب
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part09_practical-projects.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def prepare_chat_dataset(df: pd.DataFrame, tokenizer):
    """يحول البيانات إلى صيغة محادثة."""
    def format_example(row):
        return {
            "messages": [
                {"role": "system", "content": "أنت مصنف مشاعر. صنّف النص إلى: إيجابي، سلبي، محايد. أعد كلمة واحدة فقط."},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": row["label"]},
            ]
        }

    ds = Dataset.from_pandas(df)
    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds

def train_lora_sentiment(df: pd.DataFrame):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = prepare_chat_dataset(df, tokenizer)

    args = SFTConfig(
        output_dir="./sentiment_lora",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./sentiment_lora_final")
    return model, tokenizer

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part09_practical-projects.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
client = OpenAI()

def classify_with_gpt(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """أنت مصنف مشاعر عربي خبير. صنّف النص إلى: إيجابي، سلبي، محايد.
انتبه للتهكم: "ما شاء الله، وصل بعد شهر" هي سلبية.
انتبه للهجات: "مبسوط" = إيجابي، "مش بطال" = محايد/إيجابي.
أعد كلمة واحدة فقط."""},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part09_practical-projects.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def detect_sarcasm_heuristic(text: str) -> bool:
    """كشف تهكم بسيط ( heuristic)."""
    sarcasm_markers = [
        r"ما\s+شاء\s+الله.*(?:بعد|تأخر|انتظر)",
        r"الله\s+يعطيك\s+العافية.*(?:بعد|تأخر|انتظر)",
        r"(?:ممتاز|رائع|جميل).*!!!!",
        r"😍.*(?:سيء|سيئه|تأخر|فاشل)",
        r"(?:سيء|فاشل).*😍",
    ]
    for pattern in sarcasm_markers:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part09_practical-projects.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
dialect_mapping = {
    # خليجي
    "مبسوط": "سعيد",
    "زين": "جيد",
    "ما يخالف": "لا مشكلة",
    "الله يعافيك": "شكرًا",
    # مصري
    "مش بطال": "ليس سيئًا",
    "تحفة": "رائع",
    "زي الفل": "ممتاز",
    # شامي
    "منيح": "جيد",
    "كتير حلو": "جميل جدًا",
}

def normalize_dialect(text: str) -> str:
    """يحول بعض كلمات اللهجة إلى الفصحى."""
    for dialect, fusha in dialect_mapping.items():
        text = text.replace(dialect, fusha)
    return text

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part09_practical-projects.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# تحميل النموذج
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    confidence: float

@app.post("/classify", response_model=Response)
def classify(req: Request):
    text = normalize_arabic(req.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    return Response(
        label=model.config.id2label[pred],
        confidence=probs[pred].item(),
    )

@app.get("/health")
def health():
    return {"status": "ok"}

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part09_practical-projects.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# مثال: الترجمة من الإنجليزية إلى العربية
# "The doctor said she was tired."
# النموذج قد يترجم: "قال الطبيب إنها متعبة." (الطبيب مذكر، لكن "she" مؤنث)
# الترجمة الصحيحة: "قالت الطبيبة إنها متعبة."

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part09_practical-projects.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part09_practical-projects.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part09_practical-projects.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part09_practical-projects.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [016] code/notebooks/part09_practical-projects.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [017] code/notebooks/part09_practical-projects.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference: str, candidate: str) -> float:
    """BLEU: تشابه n-grams."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return sentence_bleu(
        [ref_tokens], cand_tokens,
        smoothing_function=SmoothingFunction().method1,
    )

# ────────────────────────────────────────────────────────────────────────
# [018] code/notebooks/part09_practical-projects.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [019] code/notebooks/part09_practical-projects.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
from bert_score import score

def compute_bertscore(references: list[str], candidates: list[str], lang: str = "ar"):
    P, R, F1 = score(candidates, references, lang=lang)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

# ────────────────────────────────────────────────────────────────────────
# [020] code/notebooks/part09_practical-projects.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
def translate_image_text(vlm, image, processor, tokenizer, tgt_lang="English"):
    """يقرأ النص في صورة ويترجمه."""
    prompt = f"اقرأ النص في هذه الصورة، ثم ترجمه إلى {tgt_lang}. أعد النص الأصلي والترجمة."
    # ...

# ────────────────────────────────────────────────────────────────────────
# [021] code/notebooks/part09_practical-projects.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import fitz  # PyMuPDF
import pdfplumber

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """
    يستخرج النص من PDF باستخدام PyMuPDF.
    يُرجع قائمة من الصفحات، كل صفحة نص.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages

def extract_text_pdfplumber(pdf_path: str) -> list[dict]:
    """
    يستخرج النص من PDF باستخدام pdfplumber.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": page_num + 1, "text": text})
    return pages

# ────────────────────────────────────────────────────────────────────────
# [022] code/notebooks/part09_practical-projects.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import re

def is_text_reversed(text: str) -> bool:
    """
    يكشف إذا كان النص العربي معكوسًا.
    """
    # إذا كان النص يبدأ بحرف عربي وينتهي بحرف عربي، لكن الكلمات معكوسة
    # طريقة بسيطة: ابحث عن كلمات شائعة معكوسة
    common_reversed = ["ةطقلا", "نبا", "نم", "ىلع"]  # القطة، ابن، من، على
    for word in common_reversed:
        if word in text:
            return True
    return False

def reverse_arabic_text(text: str) -> str:
    """يعكس النص العربي (إذا كان معكوسًا)."""
    return text[::-1]

# ────────────────────────────────────────────────────────────────────────
# [023] code/notebooks/part09_practical-projects.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
import pytesseract
from PIL import Image

def extract_text_ocr(image_path: str, lang: str = "ara") -> str:
    """
    يستخرج النص من صورة باستخدام Tesseract.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def extract_text_vlm(image_path: str, vlm, processor, tokenizer) -> str:
    """
    يستخرج النص باستخدام VLM (أدق للعربية).
    """
    image = Image.open(image_path).convert("RGB")
    prompt = "اقرأ كل النص العربي في هذه الصورة. أعد النص كما هو، دون شرح."
    return generate_answer(vlm, image, prompt, processor, tokenizer)

# ────────────────────────────────────────────────────────────────────────
# [024] code/notebooks/part09_practical-projects.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
import camelot

def extract_tables(pdf_path: str, pages: str = "all") -> list:
    """
    يستخرج الجداول من PDF باستخدام Camelot.
    """
    tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
    return [table.df.to_dict() for table in tables]

# ────────────────────────────────────────────────────────────────────────
# [025] code/notebooks/part09_practical-projects.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
import re

def split_arabic_sentences(text: str) -> list[str]:
    """
    يقسم النص إلى جمل، مع احترام علامات الترقيم العربية.
    """
    # علامات نهاية الجملة في العربية
    sentence_endings = r"[.!?؟۔؛]"
    # قسّم عند علامة نهاية، متبوعة بمسافة أو نهاية النص
    parts = re.split(f"(?<=[{sentence_endings}])\\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    يقسم النص إلى قطع، مع احترام حدود الجمل.
    """
    sentences = split_arabic_sentences(text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chunk_size:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            # ابدأ جديدة، مع تداخل (آخر overlap حرفًا من السابق)
            overlap_text = current[-overlap:] if len(current) > overlap else ""
            current = overlap_text + sent + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ────────────────────────────────────────────────────────────────────────
# [026] code/notebooks/part09_practical-projects.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
def hierarchical_chunk(text: str, max_chunk_size: int = 500) -> list[dict]:
    """
    تقسيم هرمي: أقسام -> فقرات -> جمل.
    """
    # افترض أن الأقسام مفصولة بعناوين (مثل "الفصل")
    sections = re.split(r"\n(?=الفصل|المادة|القسم)", text)

    chunks = []
    for section in sections:
        paragraphs = section.split("\n\n")
        for para in paragraphs:
            if len(para) <= max_chunk_size:
                chunks.append({"text": para.strip(), "type": "paragraph"})
            else:
                for sub in chunk_by_sentences(para, max_chunk_size):
                    chunks.append({"text": sub, "type": "sentence"})
    return chunks

# ────────────────────────────────────────────────────────────────────────
# [027] code/notebooks/part09_practical-projects.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

def embed_chunks(chunks: list[dict], model_name: str = "intfloat/multilingual-e5-large"):
    """
    يحول القطع إلى متجهات.
    """
    model = SentenceTransformer(model_name)
    texts = [f"passage: {c['text']}" for c in chunks]  # e5 يحتاج بادئة
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return embeddings, model

# ────────────────────────────────────────────────────────────────────────
# [028] code/notebooks/part09_practical-projects.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_retrieval(
    query: str,
    query_embedding: np.ndarray,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    k: int = 5,
):
    """
    بحث هجين: dense + BM25، مع RRF.
    """
    # 1. Dense: تشابه جيبي
    dense_scores = chunk_embeddings @ query_embedding
    dense_top = np.argsort(dense_scores)[::-1][:k*2]

    # 2. BM25
    tokenized = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())
    bm25_top = np.argsort(bm25_scores)[::-1][:k*2]

    # 3. RRF
    rrf = {}
    for rank, idx in enumerate(dense_top):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)

    top_indices = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [chunks[i] for i in top_indices]

# ────────────────────────────────────────────────────────────────────────
# [029] code/notebooks/part09_practical-projects.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

def rerank(query: str, chunks: list[dict], top_k: int = 3):
    """
    يعيد ترتيب القطع بـ Cross-Encoder.
    """
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [{"chunk": c, "score": s} for c, s in ranked[:top_k]]

# ────────────────────────────────────────────────────────────────────────
# [030] code/notebooks/part09_practical-projects.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    """
    يبني أمر RAG مع سياق ومصادر.
    """
    context_text = "\n\n".join([
        f"[مصدر {i+1}] (صفحة {c.get('page', '?')}):\n{c['text']}"
        for i, c in enumerate(contexts)
    ])

    return f"""أنت مساعد خبير يجيب على الأسئلة بناءً على المستندات المُقدمة فقط.

السياق:
{context_text}

السؤال: {query}

التعليمات:
- أجب بناءً على السياق فقط.
- إذا لم تجد الإجابة في السياق، قل: "لا أجد هذه المعلومة في المستندات المتاحة."
- اذكر رقم المصدر ورقم الصفحة (مثل: [مصدر 1، صفحة 5]).
- كن دقيقًا وموجزًا.
- إذا كان السؤال يحتاج إلى أرقام، اذكرها كما هي.

الإجابة:"""

# ────────────────────────────────────────────────────────────────────────
# [031] code/notebooks/part09_practical-projects.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def hierarchical_retrieval(query, sections_index, chunks_index, embeddings, k=5):
    """
    1. ابحث في ملخصات الأقسام.
    2. ثم ابحث في قطع القسم المختار.
    """
    # 1. بحث في الأقسام
    section_embeddings = sections_index["embeddings"]
    query_emb = embeddings.encode(query)
    section_scores = section_embeddings @ query_emb
    best_section = sections_index["names"][section_scores.argmax()]

    # 2. بحث في قطع القسم
    section_chunks = [c for c in chunks_index if c["section"] == best_section]
    # ... بحث dense + BM25 ...
    return section_chunks[:k]

# ────────────────────────────────────────────────────────────────────────
# [032] code/notebooks/part09_practical-projects.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def extractive_summarize(text: str, num_sentences: int = 5, embedder=None):
    """
    تلخيص استخراجي بـ TextRank (مبسط).
    """
    if embedder is None:
        embedder = SentenceTransformer("intfloat/multilingual-e5-small")

    sentences = split_arabic_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # تضمين الجمل
    embeddings = embedder.encode([f"passage: {s}" for s in sentences], normalize_embeddings=True)

    # مصفوفة التشابه
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    # PageRank (مبسط)
    scores = np.ones(len(sentences)) / len(sentences)
    damping = 0.85
    for _ in range(50):
        new_scores = (1 - damping) / len(sentences) + damping * sim_matrix.T @ scores
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    # اختر أعلى الجمل
    top_indices = np.argsort(scores)[::-1][:num_sentences]
    top_indices = sorted(top_indices)  # احتفظ بالترتيب الأصلي

    return " ".join([sentences[i] for i in top_indices])

# ────────────────────────────────────────────────────────────────────────
# [033] code/notebooks/part09_practical-projects.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
def mmr_selection(embeddings, scores, num_sentences: int, lambda_param: float = 0.5):
    """
    Maximal Marginal Relevance: اختر جملًا ذات صلة، لكن متنوعة.
    """
    selected = []
    candidates = list(range(len(scores)))

    for _ in range(num_sentences):
        if not candidates:
            break
        mmr_scores = []
        for i in candidates:
            relevance = scores[i]
            if selected:
                similarity = max(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] for j in selected)
            else:
                similarity = 0
            mmr = lambda_param * relevance - (1 - lambda_param) * similarity
            mmr_scores.append((i, mmr))
        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        candidates.remove(best)

    return selected

# ────────────────────────────────────────────────────────────────────────
# [034] code/notebooks/part09_practical-projects.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
def summarize_simple(text: str, llm, max_words: int = 200) -> str:
    """تلخيص بسيط بـ LLM."""
    prompt = f"""لخص النص التالي في {max_words} كلمة.
حافظ على الأفكار الرئيسية، والأرقام الأساسية.
لا تضف معلومات غير موجودة في النص.
اكتب بالعربية الفصحى.

النص:
{text}

الملخص:"""
    return llm(prompt)

# ────────────────────────────────────────────────────────────────────────
# [035] code/notebooks/part09_practical-projects.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
def map_reduce_summarize(
    text: str,
    llm,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    summary_words_per_chunk: int = 150,
    final_summary_words: int = 300,
) -> str:
    """
    Map-Reduce: قسّم، لخّص كل جزء، ثم لخّص الملخصات.
    """
    # 1. تقسيم
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])

    print(f"Split into {len(chunks)} chunks")

    # 2. Map: لخّص كل جزء
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""لخص الجزء التالي من مستند طويل في {summary_words_per_chunk} كلمة.
هذا الجزء {i+1} من {len(chunks)}. حافظ على المعلومات المهمة، والأرقام.
لا تضف معلومات غير موجودة.

الجزء:
{chunk}

الملخص:"""
        summary = llm(prompt)
        partial_summaries.append(summary)
        print(f"  Chunk {i+1}/{len(chunks)} summarized")

    # 3. Reduce: لخّص الملخصات
    combined = "\n\n".join([f"[جزء {i+1}]: {s}" for i, s in enumerate(partial_summaries)])

    final_prompt = f"""لديك ملخصات لأجزاء من مستند طويل. ادمجها في ملخص نهائي واحد ({(final_summary_words)} كلمة).
حافظ على التسلسل المنطقي. أزل التكرار. لا تفقد المعلومات الأساسية.
لا تضف معلومات غير موجودة.

الملخصات:
{combined}

الملخص النهائي:"""
    return llm(final_prompt)

# ────────────────────────────────────────────────────────────────────────
# [036] code/notebooks/part09_practical-projects.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
def refine_summarize(
    text: str,
    llm,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    target_words: int = 300,
) -> str:
    """
    Refine: لخّص تدريجيًا، مع تحسين الملخص في كل خطوة.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])

    # 1. الملخص الأولي
    current_summary = llm(f"""لخص النص التالي في {target_words} كلمة.
حافظ على المعلومات المهمة.

النص:
{chunks[0]}

الملخص:""")

    # 2. حسّن بالتدريج
    for i, chunk in enumerate(chunks[1:], start=1):
        prompt = f"""لديك ملخص حالي، وجزء جديد من نفس المستند.
حسّن الملخص بإضافة المعلومات المهمة من الجزء الجديد.
احتفظ بالطول التقريبي ({target_words} كلمة).
أزل التكرار. لا تفقد المعلومات السابقة.

الملخص الحالي:
{current_summary}

الجزء الجديد:
{chunk}

الملخص المحسن:"""
        current_summary = llm(prompt)
        print(f"  Refined with chunk {i+1}/{len(chunks)}")

    return current_summary

# ────────────────────────────────────────────────────────────────────────
# [037] code/notebooks/part09_practical-projects.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
def rag_summarize(query: str, chunks: list[dict], embedder, llm, top_k: int = 10):
    """
    تلخيص موجّه بسؤال: استرجع، ثم لخّص.
    """
    # 1. استرجاع
    query_emb = embedder.encode([f"query: {query}"], normalize_embeddings=True)
    chunk_embs = embedder.encode([f"passage: {c['text']}" for c in chunks], normalize_embeddings=True)
    scores = chunk_embs @ query_emb[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]

    # 2. تلخيص
    context = "\n\n".join([c["text"] for c in top_chunks])
    prompt = f"""لخص المعلومات التالية ذات الصلة بالسؤال: "{query}"
حافظ على الأرقام والتفاصيل. لا تضف معلومات غير موجودة.

المعلومات:
{context}

الملخص:"""
    return llm(prompt)

# ────────────────────────────────────────────────────────────────────────
# [038] code/notebooks/part09_practical-projects.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
def evaluate_summary(source: str, summary: str, llm_judge) -> dict:
    """
    تقييم شامل: أمانة، تغطية، سلاسة.
    """
    prompt = f"""قيّم الملخص التالي مقارنةً بالنص الأصلي. أعد JSON.

النص الأصلي (مقتطف):
{source[:3000]}

الملخص:
{summary}

المعايير (1-5):
1. الأمانة: هل كل ما في الملخص موجود في النص؟ (5 = لا هلوسة)
2. التغطية: هل يغطي الأفكار الرئيسية؟ (5 = كل الأفكار)
3. السلاسة: هل يقرأ كأن إنسانًا كتبه؟ (5 = سلس)
4. الإيجاز: هل هو موجز دون فقدان المعنى؟ (5 = مثالي)

أعد JSON:
{{"faithfulness": <1-5>, "coverage": <1-5>, "fluency": <1-5>, "conciseness": <1-5>, "overall": <1-5>, "missing_points": ["..."]}}"""

    import json
    response = llm_judge(prompt)
    return json.loads(response)

# ────────────────────────────────────────────────────────────────────────
# [039] code/notebooks/part09_practical-projects.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
def test_lost_in_middle(llm, needle: str, haystack_size: int = 50000):
    """
    يختبر إذا كان النموذج يجد معلومة في منتصف السياق.
    """
    filler = "هذا نص حشو. " * (haystack_size // 15)

    positions = {"beginning": 0, "middle": len(filler) // 2, "end": len(filler) - len(needle) - 100}

    results = {}
    for name, pos in positions.items():
        context = filler[:pos] + needle + filler[pos:]
        prompt = f"""النص التالي يحتوي على معلومة. ابحث عنها وأعدها.

{context}

المعلومة:"""
        response = llm(prompt)
        results[name] = needle.lower() in response.lower()

    return results

# ────────────────────────────────────────────────────────────────────────
# [040] code/notebooks/part09_practical-projects.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    user = db.execute(query).fetchone()
    return user is not None

# ────────────────────────────────────────────────────────────────────────
# [041] code/notebooks/part09_practical-projects.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
client = OpenAI()

def generate_code(description: str, language: str = "Python") -> str:
    """يولد كودًا من وصف."""
    prompt = f"""اكتب دالة {language} تفعل ما يلي:
{description}

المتطلبات:
- اكتب كودًا نظيفًا، مع تعليقات.
- استخدم أسماء متغيرات واضحة.
- تعامل مع الحالات الحدية (None، قائمة فارغة، إلخ).
- أعد الكود فقط، دون شرح."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [042] code/notebooks/part09_practical-projects.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
def generate_code_with_context(
    description: str,
    language: str,
    existing_code: str = "",
    libraries: list[str] = None,
    style: str = "PEP8",
) -> str:
    """يولد كودًا مع سياق."""
    libs = ", ".join(libraries) if libraries else "المكتبات القياسية فقط"

    prompt = f"""اكتب دالة {language} تفعل ما يلي:
{description}

السياق:
- المكتبات المتاحة: {libs}
- أسلوب الكود: {style}
- الكود الموجود (إن وجد):
```
{existing_code}
```

المتطلبات:
- اكتب كودًا نظيفًا، مع تعليقات.
- تعامل مع الحالات الحدية.
- لا تستخدم مكتبات غير مذكورة.
- أعد الكود فقط."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [043] code/notebooks/part09_practical-projects.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
def generate_code_with_tests(description: str, language: str = "Python") -> dict:
    """يولد كودًا واختبارات له."""
    prompt = f"""اكتب دالة {language} تفعل ما يلي:
{description}

ثم اكتب اختبارات (pytest) تغطي:
- الحالة العادية
- الحالات الحدية (None، فارغ، صفر)
- المدخلات غير الصالحة

أعد JSON:
{{"code": "الكود", "tests": "الاختبارات"}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    import json
    return json.loads(response.choices[0].message.content)

# مثال
result = generate_code_with_tests("دالة تحسب متوسط قائمة أرقام، وتعيد None إذا كانت فارغة")
print(result["code"])
print(result["tests"])

# ────────────────────────────────────────────────────────────────────────
# [044] code/notebooks/part09_practical-projects.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
def review_code(code: str, language: str = "Python") -> dict:
    """يراجع الكود ويكشف المشاكل."""
    prompt = f"""راجع الكود التالي ({language})، واكشف:

1. أخطاء منطقية (bugs)
2. ثغرات أمنية (SQL Injection، XSS، path traversal، eval، pickle)
3. مشاكل أداء (حلقات غير ضرورية، استعلامات N+1)
4. مشاكل أسلوب (أسماء غير واضحة، تعليقات ناقصة)
5. حالات حدية غير معالجة

الكود:
```
{code}
```

أعد JSON:
{{"issues": [{{"type": "...", "severity": "high/medium/low", "line": 0, "description": "...", "suggestion": "..."}}], "overall": "..."}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    import json
    return json.loads(response.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [045] code/notebooks/part09_practical-projects.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
import subprocess
import json

def run_bandit(code_path: str) -> dict:
    """يشغل Bandit لكشف الثغرات."""
    result = subprocess.run(
        ["bandit", "-r", code_path, "-f", "json"],
        capture_output=True, text=True,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Bandit failed", "raw": result.stdout}

# ────────────────────────────────────────────────────────────────────────
# [046] code/notebooks/part09_practical-projects.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
import ast

def detect_sql_injection(code: str) -> list[dict]:
    """
    يكشف SQL Injection في كود Python (تحليل AST).
    """
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues

    for node in ast.walk(tree):
        # نبحث عن f-strings أو .format أو % في استعلامات SQL
        if isinstance(node, ast.JoinedStr):  # f-string
            # تحقق إذا كان هناك متغير في f-string
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    issues.append({
                        "type": "SQL Injection (possible)",
                        "line": node.lineno,
                        "description": "f-string في استعلام SQL قد يكون خطرًا",
                        "suggestion": "استخدم parameterized queries: cursor.execute('... WHERE id=?', (id,))",
                    })
                    break
    return issues

# ────────────────────────────────────────────────────────────────────────
# [047] code/notebooks/part09_practical-projects.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
def fix_vulnerability(code: str, vulnerability: str) -> str:
    """يقترح إصلاحًا لثغرة."""
    prompt = f"""الكود التالي يحتوي على ثغرة: {vulnerability}

الكود:
```
{code}
```

أعد نسخة **آمنة** من الكود. أصلح الثغرة، مع الحفاظ على الوظيفة.
أعد الكود فقط."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [048] code/notebooks/part09_practical-projects.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
def generate_tests(code: str, language: str = "Python") -> str:
    """يولد اختبارات للكود."""
    prompt = f"""الكود التالي ({language}):
```
{code}
```

اكتب اختبارات (pytest) تغطي:
- الحالة العادية
- الحالات الحدية (None، فارغ، صفر، سالب)
- المدخلات غير الصالحة (نوع خاطئ، قيمة خارج النطاق)
- الحالات التي قد تسبب استثناءات

أعد الاختبارات فقط، دون شرح."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

# ────────────────────────────────────────────────────────────────────────
# [049] code/notebooks/part09_practical-projects.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
import subprocess
import tempfile
import os

def run_tests(code: str, tests: str) -> dict:
    """يشغل الاختبارات على الكود."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # اكتب الكود
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(code)

        # اكتب الاختبارات
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            f.write(f"from solution import *\n\n{tests}")

        # شغّل pytest
        result = subprocess.run(
            ["pytest", os.path.join(tmpdir, "test_solution.py"), "-v"],
            capture_output=True, text=True, timeout=30,
        )

        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

# ────────────────────────────────────────────────────────────────────────
# [050] code/notebooks/part09_practical-projects.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
def evaluate_pass_at_k(model, tokenizer, problems: list[dict], k: int = 1, num_samples: int = 1):
    """
    يقيس pass@k على HumanEval/MBPP.
    problems: list of {"prompt": ..., "tests": ...}
    """
    correct = 0
    for problem in problems:
        passed = False
        for _ in range(num_samples):
            # ولّد الكود
            inputs = tokenizer(problem["prompt"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2)
            code = tokenizer.decode(output[0], skip_special_tokens=True)

            # شغّل الاختبارات
            result = run_tests(code, problem["tests"])
            if result["passed"]:
                passed = True
                break
        if passed:
            correct += 1
    return correct / len(problems)

# ────────────────────────────────────────────────────────────────────────
# [051] code/ch41_arabic-sentiment.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
الفصل 41: تحليل المشاعر بالعربية — عندما تصبح "ما شاء الله" سخرية

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء التاسع · الفصل 41 · عدد كتل بايثون في هذا الملف: 9

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch41_arabic-sentiment.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch41_arabic-sentiment.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/9] def normalize_arabic(text: str) -> str: ════════


import pandas as pd
import re
import unicodedata

def normalize_arabic(text: str) -> str:
    """تطبيع عربي بسيط."""
    if not isinstance(text, str):
        return ""
    # إزالة التطويل
    text = re.sub(r"\u0640", "", text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # توحيد الهمزات
    text = re.sub(r"[إأآٱ]", "ا", text)
    # توحيد التاء المربوطة
    text = text.replace("ة", "ه")
    # توحيد الأرقام العربية
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    # إزالة الروابط
    text = re.sub(r"http\S+|www\S+", "", text)
    # إزالة الإشارات (@mentions)
    text = re.sub(r"@\w+", "", text)
    # توحيد المسافات
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset(path: str) -> pd.DataFrame:
    """يحمل بيانات ويطبع إحصاءات."""
    df = pd.read_csv(path)
    # افترض أن الأعمدة: text, label (positive/negative/neutral)
    df["text"] = df["text"].apply(normalize_arabic)
    df = df[df["text"].str.len() > 3]  # إزالة النصوص القصيرة جدًا
    df = df.drop_duplicates(subset=["text"])
    print(f"Loaded {len(df)} samples")
    print(df["label"].value_counts())
    return df

# مثال
# df = load_dataset("astd.csv")


# ════════ [2/9] def emoji_to_text(text: str) -> str: ════════


import emoji

def emoji_to_text(text: str) -> str:
    """يحول الإيموجي إلى وصف نصي."""
    return emoji.demojize(text, language="ar")

# مثال
print(emoji_to_text("الخدمة سيئة 😡"))  # "الخدمة سيئة :وجه_غاضب:"


# ════════ [3/9] def train_tfidf_baseline(df: pd.DataFrame): ════════


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

def train_tfidf_baseline(df: pd.DataFrame):
    """يدرب خط أساس TF-IDF + LR."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            analyzer="word",
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    return pipeline, X_test, y_test


# ════════ [4/9] AutoModelForSequenceClassification, ════════


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch

def prepare_dataset(df: pd.DataFrame, tokenizer, label2id: dict, max_length: int = 128):
    """يحول DataFrame إلى Dataset."""
    ds = Dataset.from_pandas(df[["text", "label"]])

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.map(lambda x: {"labels": label2id[x["label"]]})
    ds = ds.remove_columns(["text", "label"])
    return ds

def train_small_model(
    df: pd.DataFrame,
    model_name: str = "aubmindlab/bert-base-arabertv02",
    output_dir: str = "./sentiment_model",
):
    """يدرب نموذج تصنيف مشاعر عربي."""
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # تقسيم
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_ds = prepare_dataset(train_df, tokenizer, label2id)
    val_ds = prepare_dataset(val_df, tokenizer, label2id)

    # إعدادات التدريب
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


# ════════ [5/9] def prepare_chat_dataset(df: pd.DataFrame, tokenizer): ════════


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def prepare_chat_dataset(df: pd.DataFrame, tokenizer):
    """يحول البيانات إلى صيغة محادثة."""
    def format_example(row):
        return {
            "messages": [
                {"role": "system", "content": "أنت مصنف مشاعر. صنّف النص إلى: إيجابي، سلبي، محايد. أعد كلمة واحدة فقط."},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": row["label"]},
            ]
        }

    ds = Dataset.from_pandas(df)
    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds

def train_lora_sentiment(df: pd.DataFrame):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = prepare_chat_dataset(df, tokenizer)

    args = SFTConfig(
        output_dir="./sentiment_lora",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./sentiment_lora_final")
    return model, tokenizer


# ════════ [6/9] client = OpenAI() ════════


from openai import OpenAI
client = OpenAI()

def classify_with_gpt(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """أنت مصنف مشاعر عربي خبير. صنّف النص إلى: إيجابي، سلبي، محايد.
انتبه للتهكم: "ما شاء الله، وصل بعد شهر" هي سلبية.
انتبه للهجات: "مبسوط" = إيجابي، "مش بطال" = محايد/إيجابي.
أعد كلمة واحدة فقط."""},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


# ════════ [7/9] def detect_sarcasm_heuristic(text: str) -> bool: ════════


def detect_sarcasm_heuristic(text: str) -> bool:
    """كشف تهكم بسيط ( heuristic)."""
    sarcasm_markers = [
        r"ما\s+شاء\s+الله.*(?:بعد|تأخر|انتظر)",
        r"الله\s+يعطيك\s+العافية.*(?:بعد|تأخر|انتظر)",
        r"(?:ممتاز|رائع|جميل).*!!!!",
        r"😍.*(?:سيء|سيئه|تأخر|فاشل)",
        r"(?:سيء|فاشل).*😍",
    ]
    for pattern in sarcasm_markers:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ════════ [8/9] dialect_mapping = { ════════


dialect_mapping = {
    # خليجي
    "مبسوط": "سعيد",
    "زين": "جيد",
    "ما يخالف": "لا مشكلة",
    "الله يعافيك": "شكرًا",
    # مصري
    "مش بطال": "ليس سيئًا",
    "تحفة": "رائع",
    "زي الفل": "ممتاز",
    # شامي
    "منيح": "جيد",
    "كتير حلو": "جميل جدًا",
}

def normalize_dialect(text: str) -> str:
    """يحول بعض كلمات اللهجة إلى الفصحى."""
    for dialect, fusha in dialect_mapping.items():
        text = text.replace(dialect, fusha)
    return text


# ════════ [9/9] app = FastAPI() ════════


from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# تحميل النموذج
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    confidence: float

@app.post("/classify", response_model=Response)
def classify(req: Request):
    text = normalize_arabic(req.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    return Response(
        label=model.config.id2label[pred],
        confidence=probs[pred].item(),
    )

@app.get("/health")
def health():
    return {"status": "ok"}
