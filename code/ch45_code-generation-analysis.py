# -*- coding: utf-8 -*-
"""
الفصل 45: توليد الكود وتحليله — عندما يكتب النموذج كودًا يعمل، لكنه يقتل

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء التاسع · الفصل 45 · عدد كتل بايثون في هذا الملف: 11

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch45_code-generation-analysis.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch45_code-generation-analysis.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/11] def login(username, password): ════════


def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    user = db.execute(query).fetchone()
    return user is not None


# ════════ [2/11] client = OpenAI() ════════


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


# ════════ [3/11] def generate_code_with_context( ════════


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


# ════════ [4/11] def generate_code_with_tests(description: str, language: str = "Python ════════


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


# ════════ [5/11] def review_code(code: str, language: str = "Python") -> dict: ════════


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


# ════════ [6/11] def run_bandit(code_path: str) -> dict: ════════


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


# ════════ [7/11] def detect_sql_injection(code: str) -> list[dict]: ════════


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


# ════════ [8/11] def fix_vulnerability(code: str, vulnerability: str) -> str: ════════


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


# ════════ [9/11] def generate_tests(code: str, language: str = "Python") -> str: ════════


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


# ════════ [10/11] def run_tests(code: str, tests: str) -> dict: ════════


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


# ════════ [11/11] def evaluate_pass_at_k(model, tokenizer, problems: list[dict], k: int  ════════


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
