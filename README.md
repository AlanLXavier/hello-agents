# ReAct Agent —— 从零实现的智能体

> 一个不依赖任何框架、用约 150 行 Python 从零构建的 ReAct 智能体。
> 它通过「思考 → 行动 → 观察」的循环，自主调用工具解决真实问题。

## ✨ 项目亮点

- **零框架依赖**：只用 `openai` + `requests`，没有 LangChain 等重型框架，彻底暴露 Agent 的底层原理
- **兼容任意 LLM**：基于 OpenAI 接口规范，可无缝切换 DeepSeek、豆包、Qwen 等国产模型
- **真实工具调用**：天气查询（wttr.in 免费 API）、数学计算、北京时间查询
- **完整的开发闭环**：从"模型不会查时间"到"加上 get_time 工具"，演示了 Agent 迭代开发的真实过程

## 🧠 核心原理：ReAct 范式

ReAct = Reasoning（思考）+ Acting（行动）。大模型本身拿不到实时信息，ReAct 给它装上"手"——调用工具，让决策建立在真实数据之上。

```
        ┌──────────────────────────────────────┐
        │                                      │
        ▼                                      │
   ┌─────────┐    ┌─────────┐    ┌─────────┐   │
   │ Thought │ →  │ Action  │ →  │ Observe │───┘
   │  思考    │    │  行动    │    │  观察   │
   └─────────┘    └─────────┘    └─────────┘
```

每一轮模型只输出一对 `Thought + Action`，工具返回的结果（Observation）会喂回给模型，作为下一轮决策的依据，直到模型输出 `Finish[最终答案]`。

## 🛠️ 工具列表

| 工具 | 功能 | 依赖 |
|------|------|------|
| `get_weather(city)` | 查询城市实时天气 | 免费（wttr.in） |
| `calculator(expression)` | 计算数学表达式 | 无 |
| `get_time()` | 查询北京时间 | 无 |

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install openai requests

# 2. 设置 API Key
export DEEPSEEK_API_KEY="sk-你的key"

# 3. 运行
python react_agent.py
```

> 换其他模型只需改 `base_url` 和 `model` 两个参数（代码里已经集中写好了）。

## 📝 示例输出

```
👤 用户：请帮我查一下北京的天气，顺便算一下 123 乘以 456 等于多少。

🔄 第 1 轮
Thought: 我需要先查询北京天气。
Action: get_weather(city="北京")
Observation: 北京当前天气：晴，气温26℃

🔄 第 2 轮
Thought: 天气查到了，接下来计算 123*456。
Action: calculator(expression="123*456")
Observation: 计算结果：123*456 = 56088

🔄 第 3 轮
Thought: 信息齐全，可以回答。
Action: Finish[北京晴朗26℃；123*456=56088]
✅ 任务完成！
```

## 📂 代码结构

```
react_agent.py
├── 第一部分：系统提示词（告诉模型角色、工具、输出格式）
├── 第二部分：工具函数（get_weather / calculator / get_time）
├── 第三部分：解析器（把模型的文字翻译成函数调用）
└── 第四部分：主循环（调模型 → 解析 → 执行 → 喂回结果 → 循环）
```

## 🛣️ 技术栈

- Python 3.11
- DeepSeek API（`deepseek-chat`）
- OpenAI SDK
- requests
