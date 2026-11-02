# -*- coding: utf-8 -*-
"""
الفصل 47: أنماط تصميم أنظمة LLM — عندما يصبح النموذج مجرد لبنة

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء العاشر · الفصل 47 · عدد كتل بايثون في هذا الملف: 9

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch47_llm-system-design-patterns.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch47_llm-system-design-patterns.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/9] class RAGSystem: ════════


from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np

class RAGSystem:
    def __init__(self, llm, embedder_name="intfloat/multilingual-e5-large",
                 reranker_name="BAAI/bge-reranker-v2-m3"):
        self.llm = llm
        self.embedder = SentenceTransformer(embedder_name)
        self.reranker = CrossEncoder(reranker_name)
        self.chunks = []
        self.embeddings = None

    def index(self, documents: list[str]):
        """فهرسة المستندات."""
        self.chunks = []
        for doc in documents:
            self.chunks.extend(self._chunk(doc))
        texts = [f"passage: {c}" for c in self.chunks]
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        print(f"Indexed {len(self.chunks)} chunks")

    def _chunk(self, text: str, size: int = 500) -> list[str]:
        """تقسيم بسيط (استخدم تقسيمًا أفضل في الإنتاج)."""
        sentences = text.split(". ")
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= size:
                current += sent + ". "
            else:
                chunks.append(current.strip())
                current = sent + ". "
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """استرجاع + إعادة ترتيب."""
        # 1. استرجاع كثيف
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb
        top_k = np.argsort(scores)[::-1][:k*2]
        candidates = [self.chunks[i] for i in top_k]

        # 2. إعادة ترتيب
        pairs = [(query, c) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:k]]

    def generate(self, query: str) -> str:
        """توليد الإجابة."""
        contexts = self.retrieve(query)
        context_text = "\n\n".join([f"[مصدر {i+1}]: {c}" for i, c in enumerate(contexts)])

        prompt = f"""أجب على السؤال بناءً على السياق فقط.
إذا لم تجد الإجابة، قل "لا أعرف".

السياق:
{context_text}

السؤال: {query}

الإجابة:"""
        return self.llm(prompt)


# ════════ [2/9] class ReActAgent: ════════


import json
import re

class ReActAgent:
    def __init__(self, llm, tools: dict, max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, query: str) -> str:
        history = []
        for step in range(self.max_steps):
            prompt = self._build_prompt(query, history)
            response = self.llm(prompt)

            # هل انتهى؟
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()

            # استخرج الأداة
            action, action_input = self._parse_action(response)
            if not action:
                history.append("Observation: لم أفهم. حاول مرة أخرى.")
                continue

            # نفذ
            if action not in self.tools:
                observation = f"الأداة {action} غير موجودة."
            else:
                try:
                    observation = str(self.tools[action](action_input))
                except Exception as e:
                    observation = f"خطأ: {e}"

            history.append(response.strip())
            history.append(f"Observation: {observation}")

        return "لم أتمكن من الإجابة."

    def _build_prompt(self, query, history):
        tools_desc = "\n".join([f"- {name}: {func.__doc__}" for name, func in self.tools.items()])
        history_text = "\n".join(history) if history else "(لا يوجد)"
        return f"""أنت وكيل. استخدم الأدوات للإجابة.

الأدوات:
{tools_desc}

التنسيق:
Thought: [تفكيرك]
Action: [اسم الأداة]
Action Input: [المدخل]

عندما تكون جاهزًا:
Final Answer: [الإجابة]

السؤال: {query}

التاريخ:
{history_text}
"""

    def _parse_action(self, response):
        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r"Action Input:\s*(.+)", response, re.DOTALL)
        if not action_match:
            return None, None
        return action_match.group(1), input_match.group(1).strip() if input_match else ""


# ════════ [3/9] class MultiAgentSystem: ════════


class MultiAgentSystem:
    def __init__(self, manager_llm, workers: dict):
        self.manager = manager_llm
        self.workers = workers

    def run(self, task: str) -> str:
        prompt = f"""أنت مدير. لديك الفريق:
{list(self.workers.keys())}

المهمة: {task}

اختر الوكيل المناسب، وأعد اسمه فقط."""
        worker_name = self.manager(prompt).strip()

        if worker_name not in self.workers:
            return f"لا يوجد وكيل باسم {worker_name}"

        return self.workers[worker_name].run(task)


# ════════ [4/9] def fine_tuning_pipeline( ════════


def fine_tuning_pipeline(
    data_path: str,
    base_model: str,
    output_dir: str,
    eval_fn,
    min_accuracy: float = 0.8,
):
    """
    خط أنابيب: تحميل، تدريب، تقييم، نشر.
    """
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # 1. تحميل
    df = pd.read_csv(data_path)

    # 2. تنسيق
    ds = Dataset.from_pandas(df)

    # 3. تدريب
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="bfloat16", device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=output_dir, num_train_epochs=2, bf16=True),
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 4. تقييم
    accuracy = eval_fn(model, tokenizer)
    if accuracy < min_accuracy:
        raise Exception(f"Accuracy {accuracy:.2f} < {min_accuracy}. Blocking deployment.")

    # 5. حفظ
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


# ════════ [5/9] class EvalPipeline: ════════


class EvalPipeline:
    def __init__(self, judge_llm, test_cases: list[dict]):
        self.judge = judge_llm
        self.test_cases = test_cases

    def run_offline(self, model, tokenizer) -> dict:
        """تقييم آلي على مجموعة اختبار."""
        scores = []
        for case in self.test_cases:
            # توليد
            answer = generate(model, tokenizer, case["input"])
            # تقييم
            score = self._judge(case["input"], answer, case.get("expected"))
            scores.append(score)
        return {
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        }

    def _judge(self, question, answer, expected):
        """LLM-as-a-Judge."""
        prompt = f"""قيّم الإجابة على مقياس 1-5.

السؤال: {question}
الإجابة: {answer}
الإجابة المتوقعة (إن وجدت): {expected}

أعد رقمًا فقط."""
        result = self.judge(prompt).strip()
        try:
            return int(result)
        except:
            return 0

    def run_online(self, user_feedback: list[dict]) -> dict:
        """تقييم من ملاحظات المستخدمين."""
        positive = sum(1 for f in user_feedback if f["rating"] >= 4)
        return {
            "satisfaction": positive / len(user_feedback) if user_feedback else 0,
            "total": len(user_feedback),
        }


# ════════ [6/9] app = FastAPI() ════════


from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://vllm:8000/v1", api_key="dummy")

class Request(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
def generate(req: Request):
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=req.max_tokens,
    )
    return {"text": response.choices[0].message.content}

@app.get("/health")
def health():
    return {"status": "ok"}


# ════════ [7/9] class Guardrails: ════════


import re

class Guardrails:
    def __init__(self):
        self.injection_patterns = [
            r"تجاهل\s+(كل\s+)?التعليمات",
            r"ignore\s+(all\s+)?instructions",
            r"system\s*:",
        ]
        self.pii_patterns = {
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            "phone": r"\b(?:\+966|0)?5\d{8}\b",
        }

    def check_input(self, text: str) -> tuple[bool, str]:
        """يفلتر المدخلات."""
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "injection_detected"
        return True, ""

    def check_output(self, text: str) -> tuple[bool, str]:
        """يفلتر المخرجات."""
        for name, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                text = re.sub(pattern, f"[{name.upper()}_REDACTED]", text)
        return True, text


# ════════ [8/9] class SemanticCache: ════════


import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, embedder, threshold=0.95):
        self.embedder = embedder
        self.threshold = threshold
        self.cache = []  # list of (embedding, response)

    def get(self, query: str):
        if not self.cache:
            return None
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)[0]
        for emb, response in self.cache:
            sim = np.dot(q_emb, emb)
            if sim >= self.threshold:
                return response
        return None

    def set(self, query: str, response: str):
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)[0]
        self.cache.append((q_emb, response))


# ════════ [9/9] def generate_with_fallback(prompt, primary, fallback, timeout=5): ════════


def generate_with_fallback(prompt, primary, fallback, timeout=5):
    try:
        return primary.generate(prompt, timeout=timeout)
    except Exception as e:
        print(f"Primary failed: {e}. Using fallback.")
        return fallback.generate(prompt)
