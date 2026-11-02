# -*- coding: utf-8 -*-
"""
الفصل 24: الوكلاء (Agents) واستخدام الأدوات — عندما يتحول النموذج من مُجيب إلى فاعل

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الخامس · الفصل 24 · عدد كتل بايثون في هذا الملف: 5

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch24_agents-and-tools.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch24_agents-and-tools.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/5] class ReActAgent: ════════


import re
import json
from typing import Callable

class ReActAgent:
    def __init__(self, llm, tools: dict[str, Callable], max_steps: int = 10):
        """
        llm: دالة (prompt) -> response (نص)
        tools: dict من اسم الأداة إلى دالة
        """
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def build_prompt(self, question: str, history: list[str]) -> str:
        tools_desc = "\n".join([
            f"- {name}: {func.__doc__ or 'no description'}"
            for name, func in self.tools.items()
        ])

        history_text = "\n".join(history) if history else "(لا يوجد)"

        return f"""أنت وكيل ذكي. أجب على السؤال باستخدام الأدوات المتاحة.

الأدوات المتاحة:
{tools_desc}

التنسيق المطلوب:
Thought: [تفكيرك]
Action: [اسم الأداة]
Action Input: [المدخل، بصيغة JSON]

عندما تكون جاهزًا للإجابة النهائية:
Thought: لدي الإجابة.
Final Answer: [الإجابة]

السؤال: {question}

التاريخ:
{history_text}
"""

    def parse_action(self, response: str):
        """يستخرج Action وAction Input من استجابة النموذج."""
        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r"Action Input:\s*(.+)", response, re.DOTALL)

        if not action_match:
            return None, None

        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""

        return action, action_input

    def parse_final_answer(self, response: str):
        """يستخرج الإجابة النهائية."""
        match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        return match.group(1).strip() if match else None

    def run(self, question: str) -> str:
        history = []

        for step in range(self.max_steps):
            prompt = self.build_prompt(question, history)
            response = self.llm(prompt)

            # هل وصل إلى الإجابة النهائية؟
            final = self.parse_final_answer(response)
            if final:
                return final

            # استخرج Action
            action, action_input = self.parse_action(response)
            if not action:
                history.append(f"Observation: لم أفهم الاستجابة. حاول مرة أخرى.")
                continue

            # نفذ الأداة
            if action not in self.tools:
                observation = f"خطأ: الأداة '{action}' غير موجودة."
            else:
                try:
                    # حاول تحليل JSON
                    try:
                        args = json.loads(action_input)
                        if isinstance(args, dict):
                            observation = self.tools[action](**args)
                        else:
                            observation = self.tools[action](args)
                    except json.JSONDecodeError:
                        observation = self.tools[action](action_input)
                except Exception as e:
                    observation = f"خطأ في تنفيذ الأداة: {e}"

            history.append(response.strip())
            history.append(f"Observation: {observation}")

        return "لم أتمكن من الإجابة في الخطوات المتاحة."

# ===== مثال =====
def search_tool(query: str) -> str:
    """يبحث في الويب ويعيد النتائج."""
    # في الواقع، استخدم SerpAPI أو Tavily
    fake_db = {
        "عدد سكان فرنسا": "68,000,000",
        "عدد سكان مصر": "105,000,000",
    }
    return fake_db.get(query, "لا توجد نتائج.")

def calculator_tool(expression: str) -> str:
    """يحسب تعبيرًا رياضيًا. المدخل: {"expression": "2+2"}"""
    try:
        # تحذير: eval خطير في الإنتاج! استخدم مكتبة آمنة
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"خطأ: {e}"

# محاكاة LLM (في الواقع، استخدم OpenAI/Anthropic/Llama)
def fake_llm(prompt: str) -> str:
    if "عدد سكان فرنسا" in prompt and "search" in prompt and "68" not in prompt.split("Observation:")[-1]:
        return """Thought: أحتاج إلى معرفة عدد سكان فرنسا.
Action: search
Action Input: {"query": "عدد سكان فرنسا"}"""
    elif "68,000,000" in prompt and "calculator" in prompt and "136" not in prompt:
        return """Thought: الآن أضرب 68,000,000 في 2.
Action: calculator
Action Input: {"expression": "68000000 * 2"}"""
    elif "136000000" in prompt:
        return """Thought: لدي الإجابة.
Final Answer: 136,000,000"""
    return "Thought: لا أعرف.\nFinal Answer: لا أعرف."

agent = ReActAgent(fake_llm, {"search": search_tool, "calculator": calculator_tool})
print(agent.run("كم عدد سكان فرنسا مضروبًا في 2؟"))


# ════════ [2/5] client = OpenAI() ════════


from openai import OpenAI
import json

client = OpenAI()

# تعريف الأدوات (JSON Schema)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "احصل على حالة الطقس في مدينة.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "اسم المدينة"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "احصل على سعر سهم.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "رمز السهم"},
                },
                "required": ["symbol"],
            },
        },
    },
]

# دالة تنفيذ
def execute_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return json.dumps({"city": args["city"], "temp": 25, "unit": "celsius"})
    elif name == "get_stock_price":
        return json.dumps({"symbol": args["symbol"], "price": 150.5})
    return "unknown tool"

# حلقة الوكيل
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(10):  # max steps
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # إذا لم يكن هناك tool_call، فهذه إجابة نهائية
        if not msg.tool_calls:
            return msg.content

        # أضف رسالة المساعد (مع tool_calls) إلى التاريخ
        messages.append(msg)

        # نفذ كل أداة، وأضف النتائج
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "تجاوزت الحد الأقصى من الخطوات."

print(run_agent("ما هو طقس الرياض، وسعر سهم أبل؟"))


# ════════ [3/5] def plan_and_execute(goal: str, llm, tools: dict, max_replans: int = 3 ════════


def plan_and_execute(goal: str, llm, tools: dict, max_replans: int = 3):
    # 1. التخطيط
    plan_prompt = f"""ضع خطة من 3-7 خطوات لتحقيق الهدف التالي.
كل خطوة يجب أن تكون قابلة للتنفيذ باستخدام الأدوات المتاحة: {list(tools.keys())}.

الهدف: {goal}

أعد الخطة كقائمة JSON: [{{"step": 1, "action": "...", "tool": "..."}}, ...]"""

    plan_text = llm(plan_prompt)
    plan = json.loads(plan_text)

    results = []
    for step in plan:
        # نفذ الخطوة
        tool_name = step.get("tool")
        if tool_name and tool_name in tools:
            result = tools[tool_name](step.get("action", ""))
        else:
            # استخدم LLM للتفكير
            result = llm(f"الخطوة: {step}\nالسياق: {results}\nنفذ الخطوة.")

        results.append({"step": step, "result": result})

    # 2. إعادة التخطيط (اختياري)
    for _ in range(max_replans):
        review_prompt = f"""راجع النتائج التالية. هل الهدف تحقق؟
إذا لا، اقترح خطوات إضافية. إذا نعم، أعد "DONE".

الهدف: {goal}
النتائج: {results}"""

        review = llm(review_prompt)
        if "DONE" in review:
            break
        # ... نفذ الخطوات الإضافية ...

    return results


# ════════ [4/5] def safe_tool_call(tool, args, validators: list[Callable] = None): ════════


def safe_tool_call(tool, args, validators: list[Callable] = None):
    """
    ينفذ أداة مع تحقق مسبق وبعدي.
    """
    # 1. تحقق مسبق
    if validators:
        for v in validators:
            if not v(args):
                return f"مرفوض: فشل التحقق {v.__name__}"

    # 2. تنفيذ
    try:
        result = tool(**args)
    except Exception as e:
        return f"خطأ: {e}"

    # 3. تحقق بعدي (مثال: لا NaN)
    if isinstance(result, (int, float)) and (result != result):  # NaN check
        return "خطأ: النتيجة NaN"

    return result

# مثال: تحقق من مبلغ التحويل
def validate_amount(args):
    amount = args.get("amount", 0)
    return 0 < amount <= 10000  # لا تحويل أكثر من 10,000

def validate_recipient(args):
    iban = args.get("iban", "")
    return len(iban) == 24 and iban.startswith("SA")


# ════════ [5/5] class ManagerAgent: ════════


class ManagerAgent:
    def __init__(self, llm, workers: dict[str, "Agent"]):
        self.llm = llm
        self.workers = workers

    def delegate(self, task: str) -> str:
        prompt = f"""أنت مدير. لديك الفريق التالي:
{list(self.workers.keys())}

المهمة: {task}

اختر الوكيل المناسب، وأعد اسمه فقط."""
        worker_name = self.llm(prompt).strip()

        if worker_name not in self.workers:
            return f"لا يوجد وكيل باسم {worker_name}"

        return self.workers[worker_name].run(task)

# مثال
researcher = ReActAgent(fake_llm, {"search": search_tool})
writer = ReActAgent(fake_llm, {})  # وكيل كتابة بدون أدوات
manager = ManagerAgent(fake_llm, {"researcher": researcher, "writer": writer})
