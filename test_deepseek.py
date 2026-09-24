"""
第一个测试：验证你的 Mac 能正常调用 DeepSeek。

运行前，先在终端里设置你的 API Key（把 sk-xxx 换成你自己的 key）：
    export DEEPSEEK_API_KEY="sk-你的key"

然后运行：
    source .venv/bin/activate
    python test_deepseek.py

看到 DeepSeek 回复一句话，就说明一切正常，你的 Mac 完全够用！
"""

import os
from openai import OpenAI

# 1. 从环境变量读取 API Key（不要写死在代码里，安全！）
api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    print("❌ 没找到 API Key！")
    print("   请先运行： export DEEPSEEK_API_KEY=\"sk-你的key\"")
    exit(1)

# 2. 创建客户端，指向 DeepSeek 的服务器
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",  # 关键：把地址改成 DeepSeek
)

# 3. 发送一条消息，让 DeepSeek 回复
print("正在调用 DeepSeek...\n")
response = client.chat.completions.create(
    model="deepseek-chat",  # DeepSeek 的模型名
    messages=[
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": "用一句话证明你已经在线，并告诉我今天适合学什么。"},
    ],
)

# 4. 打印返回结果
answer = response.choices[0].message.content
print("DeepSeek 回复：")
print("-" * 40)
print(answer)
print("-" * 40)
print("\n✅ 成功！你的 Mac 已经能正常调用大模型了。")
