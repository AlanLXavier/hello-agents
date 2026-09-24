"""
ReAct 智能体 —— 你的第一个"简历级"项目

它能让 DeepSeek 通过「思考 → 行动 → 观察」的循环，
自主调用工具来解决问题，而不是凭空瞎猜。

自带三个工具（都免费、无需额外 Key）：
  1. get_weather —— 查询城市实时天气（调用 wttr.in）
  2. calculator —— 计算数学表达式
  3. get_time —— 查询当前的北京时间

相比"自由文本 + 正则解析"，本版本使用「结构化输出」：
要求模型直接返回 JSON，用 json.loads 标准解析，更可靠。

运行方法：
  export DEEPSEEK_API_KEY="sk-你的key"   # 只需设置一次
  source .venv/bin/activate
  python react_agent.py
"""

import json
import os
import re
from openai import OpenAI

# ════════════════════════════════════════════════════════
# 第一部分：系统提示词（智能体的"说明书"）
# 作用：告诉 DeepSeek 它是什么角色、有哪些工具、必须按什么格式说话
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个智能助手，擅长通过调用工具来一步步解决问题。

# 你可以使用的工具：
1. get_weather —— 查询指定城市的实时天气，参数：city="城市名"
2. calculator —— 计算数学表达式，参数：expression="数学表达式"，例如 "123*456"
3. get_time —— 查询当前的北京时间，不需要参数

# 你的回复必须是一个严格的 JSON 对象，不要输出任何其他文字。格式如下：

调用工具时：
{"thought": "你的思考过程", "action": "get_weather", "args": {"city": "北京"}}

结束任务时：
{"thought": "你的思考过程", "action": "Finish", "args": {"answer": "最终答案"}}

# 重要规则：
- 只输出 JSON 对象本身，不要用 ```json 包裹，不要有任何多余文字
- 禁止重复调用同一个工具：如果之前已经调用过并拿到了结果，直接基于已有结果用 Finish 结束
- 当信息已经足够回答用户时，必须用 action="Finish" 结束
"""

# ════════════════════════════════════════════════════════
# 第二部分：工具函数（给智能体装的"手"）
# 每个工具就是一个普通的 Python 函数，接收参数，返回一段文字
# ════════════════════════════════════════════════════════

def get_weather(city: str) -> str:
    """查询城市实时天气（调用免费的 wttr.in 服务，无需 API Key）"""
    import requests
    url = f"https://wttr.in/{city}?format=j1"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        cur = data["current_condition"][0]
        desc = cur["weatherDesc"][0]["value"]
        temp = cur["temp_C"]
        return f"{city}当前天气：{desc}，气温{temp}℃"
    except Exception as e:
        return f"查询天气失败：{e}"


def calculator(expression: str) -> str:
    """安全地计算一个数学表达式"""
    # 只允许数字和运算符，防止模型让程序执行危险代码
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):
        return "表达式包含非法字符，只能包含数字和 + - * / ( ) %"
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


def get_time() -> str:
    """返回当前的北京时间（东八区）"""
    from datetime import datetime, timezone, timedelta
    beijing = timezone(timedelta(hours=8))
    now = datetime.now(beijing)
    return f"当前北京时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"


# 工具注册表：把工具名字和函数对应起来，主循环靠它找到并执行工具
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
    "get_time": get_time,
}

# ════════════════════════════════════════════════════════
# 第三部分：解析模型输出
# 模型输出的是 JSON，我们用标准解析器 json.loads 直接解析
# ════════════════════════════════════════════════════════

def parse_output(output: str):
    """把模型的 JSON 输出解析成 (thought, action, args)"""
    # 模型偶尔会输出 ```json ... ``` 或多余文字，先提取花括号部分
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return None, None, None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None, None
    return data.get("thought", ""), data.get("action", ""), data.get("args", {})


def execute_action(action: str, args: dict):
    """执行结构化的 Action，返回 (状态, 结果)"""
    # 情况1：结束任务
    if action == "Finish":
        return "FINISHED", args.get("answer", "")

    # 情况2：调用工具
    if action not in AVAILABLE_TOOLS:
        return "ERROR", f"未定义的工具 '{action}'"

    try:
        result = AVAILABLE_TOOLS[action](**args)
        return "OBSERVATION", result
    except Exception as e:
        return "ERROR", f"工具 {action} 执行失败：{e}"


# ════════════════════════════════════════════════════════
# 第四部分：主循环（智能体的"心跳"）
# 调模型 → 解析 JSON → 执行工具 → 把结果喂回去 → 循环，直到 Finish
# ════════════════════════════════════════════════════════

def run_agent(user_prompt: str, max_rounds: int = 6):
    """运行 ReAct 循环"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print('❌ 请先设置：export DEEPSEEK_API_KEY="sk-你的key"')
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 对话历史：system 说明书 + 用户的问题
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print(f"👤 用户：{user_prompt}\n" + "=" * 55)

    # 记录已经调用过的工具，用于检测"死循环"
    call_history = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n🔄 第 {round_num} 轮")

        # 1. 调用模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        output = response.choices[0].message.content
        print(f"\n🤔 模型输出：\n{output}\n")

        # 2. 把模型的回复记入历史
        messages.append({"role": "assistant", "content": output})

        # 3. 解析 JSON
        thought, action, args = parse_output(output)
        if action is None:
            print("⚠️  没解析到 JSON，模型可能没按要求输出，请重试。")
            break

        # 3.5 【防死循环】检测是否在重复调用同一个工具
        call_key = f"{action}:{sorted(args.items())}"
        if call_key not in call_history:
            call_history.append(call_key)
        else:
            print("🔁 检测到重复调用！给模型一个反思提示，而不是再次执行。")
            reflection = (
                f"⚠️ 你刚才已经调用过 {action} 工具了，结果在之前的 Observation 里。"
                "请不要重复调用，直接基于已有信息用 action=Finish 结束。"
            )
            messages.append({"role": "user", "content": f"Observation: {reflection}"})
            continue

        # 4. 执行 Action，得到结果
        status, result = execute_action(action, args)

        if status == "FINISHED":
            print("✅ 任务完成！最终答案：")
            print("-" * 55)
            print(result)
            print("-" * 55)
            break

        # 5. 把观察结果喂回给模型，进入下一轮
        observation = f"Observation: {result}"
        print(f"👀 观察结果：\n{observation}")
        messages.append({"role": "user", "content": observation})
    else:
        print(f"\n⚠️  达到最大轮数 {max_rounds} 仍未完成，模型可能陷入了循环。")


# ════════════════════════════════════════════════════════
# 程序入口：改这里的问题，就能让智能体做不同的事
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    question = "请帮我查一下北京的天气，顺便算一下 123 乘以 456 等于多少。"
    # question = "现在北京时间是几点？"

    run_agent(question)
