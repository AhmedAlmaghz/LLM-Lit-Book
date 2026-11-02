# -*- coding: utf-8 -*-
"""
الفصل 40: النشر (Deployment) — من المختبر إلى الإنتاج: كيف تبني نظامًا لا ينهار

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثامن · الفصل 40 · عدد كتل بايثون في هذا الملف: 10

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch40_deployment-production.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch40_deployment-production.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/10] def batch_inference(model_name: str, prompts: list[str], batch_size: i ════════


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def batch_inference(model_name: str, prompts: list[str], batch_size: int = 8, max_new_tokens: int = 256):
    """
    معالجة دفعات: أبسط وأرخص.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)

    return results


# ════════ [2/10] client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy") ════════


from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "مرحبًا، كيف حالك؟"}],
    max_tokens=100,
    temperature=0.7,
)
print(response.choices[0].message.content)


# ════════ [3/10] def benchmark_vllm(url: str, prompt: str, num_requests: int = 100, con ════════


import time
import requests
from concurrent.futures import ThreadPoolExecutor

def benchmark_vllm(url: str, prompt: str, num_requests: int = 100, concurrency: int = 10):
    """
    يقيس TTFT وITL وThroughput.
    """
    def send_request(_):
        start = time.time()
        response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
            },
        )
        return time.time() - start

    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        latencies = list(executor.map(send_request, range(num_requests)))
    total_time = time.time() - start

    return {
        "total_time": total_time,
        "throughput_rps": num_requests / total_time,
        "avg_latency": sum(latencies) / len(latencies),
        "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
    }


# ════════ [4/10] REQUEST_COUNT = Counter("llm_requests_total", "Total requests", ["mode ════════


from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# تعريف المقاييس
REQUEST_COUNT = Counter("llm_requests_total", "Total requests", ["model", "status"])
REQUEST_LATENCY = Histogram("llm_request_latency_seconds", "Request latency", ["model"])
TTFT = Histogram("llm_ttft_seconds", "Time to first token", ["model"])
ACTIVE_REQUESTS = Gauge("llm_active_requests", "Active requests")

def monitored_generate(model, tokenizer, prompt: str):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # قياس TTFT
        ttft_start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1)
        TTFT.labels(model=model.name_or_path).observe(time.time() - ttft_start)

        # التوليد الكامل
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)

        REQUEST_COUNT.labels(model=model.name_or_path, status="success").inc()
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        REQUEST_COUNT.labels(model=model.name_or_path, status="error").inc()
        raise e
    finally:
        REQUEST_LATENCY.labels(model=model.name_or_path).observe(time.time() - start)
        ACTIVE_REQUESTS.dec()

# تشغيل خادم المقاييس
start_http_server(9090)  # prometheus يقرأ من :9090/metrics


# ════════ [5/10] app = FastAPI() ════════


from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health")
def health():
    """فحص صحي بسيط."""
    try:
        # تحقق من النموذج
        result = model.generate("test", max_new_tokens=1)
        return {"status": "ok", "model": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/ready")
def ready():
    """هل النظام جاهز لاستقبال الطلبات؟"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


# ════════ [6/10] def ab_test_router(user_id: str, model_a, model_b, traffic_split: floa ════════


import random

def ab_test_router(user_id: str, model_a, model_b, traffic_split: float = 0.1):
    """
    يوجه 10% من المستخدمين إلى النموذج B، و90% إلى A.
    """
    # استخدم hash لتوجيه ثابت (نفس المستخدم يحصل على نفس النموذج)
    hash_val = hash(user_id) % 100
    if hash_val < traffic_split * 100:
        return model_b, "B"
    return model_a, "A"

def track_ab_result(user_id, model_version, success, latency):
    """سجّل النتائج للمقارنة."""
    import json, datetime
    with open("ab_results.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "model": model_version,
            "success": success,
            "latency": latency,
        }) + "\n")


# ════════ [7/10] def retry_with_backoff(max_retries=3, base_delay=1.0): ════════


import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise


# ════════ [8/10] def generate_with_fallback(prompt, primary_model, fallback_model): ════════


def generate_with_fallback(prompt, primary_model, fallback_model):
    try:
        return primary_model.generate(prompt, timeout=5)
    except Exception as e:
        print(f"Primary failed: {e}. Using fallback.")
        return fallback_model.generate(prompt)


# ════════ [9/10] def pre_deployment_check(model_path: str, test_cases: list[dict], min_ ════════


def pre_deployment_check(model_path: str, test_cases: list[dict], min_accuracy: float = 0.8):
    """
    يختبر النموذج قبل النشر.
    """
    model, tokenizer = load_model(model_path)
    correct = 0

    for case in test_cases:
        pred = generate_answer(model, tokenizer, case["input"])
        if case["expected"].lower() in pred.lower():
            correct += 1

    accuracy = correct / len(test_cases)
    if accuracy < min_accuracy:
        raise Exception(f"Accuracy {accuracy:.2f} < {min_accuracy}. Blocking deployment.")
    return {"accuracy": accuracy, "passed": True}


# ════════ [10/10] def canary_deploy(new_model, old_model, traffic_percent=5, duration_mi ════════


def canary_deploy(new_model, old_model, traffic_percent=5, duration_minutes=30):
    """
    ينشر النموذج الجديد لـ 5% من الطلبات، ويراقب.
    """
    import time
    start = time.time()
    metrics = {"new": [], "old": []}

    while time.time() - start < duration_minutes * 60:
        # توجيه 5% إلى الجديد
        model, version = ab_test_router("user", new_model, old_model, traffic_percent)
        # ... قس النتائج ...
        time.sleep(60)

    # قارن
    if metrics["new"]["error_rate"] > metrics["old"]["error_rate"] * 1.1:
        rollback()
    else:
        promote(new_model)
