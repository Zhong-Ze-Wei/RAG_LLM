from llm_core.api_wrapper.deepseek_api import DeepSeekChat
import yaml
import os

# 从配置文件加载 API Key 和模型名
with open("config/model_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

api_key = os.getenv("DEEPSEEK_API_KEY")  # 从环境变量取值
# api_key = config['api']['api_key'] 如果写在本地中可以在这里读取
model_name = config['api']['model_name']

# 初始化模型对象
bot = DeepSeekChat(api_key=api_key, model_name=model_name)

# 用于存储完整对话历史
history = []


# 编码安全处理，避免乱码问题
def safe_str(s):
    if not isinstance(s, str):
        s = str(s)
    return s.encode('utf-8', 'ignore').decode('utf-8')


# 执行一次对话
def chat(query, temperature=0.5, top_p=0.95, stream=False):
    # 构造消息列表，包含系统角色设定和历史对话内容
    messages = [{"role": "system", "content": safe_str("你是一个有帮助的助手。")}]
    for turn in history:
        messages.append({"role": "user", "content": safe_str(turn['user'])})
        messages.append({"role": "assistant", "content": safe_str(turn['bot'])})
    messages.append({"role": "user", "content": safe_str(query)})

    # 调用底层 API 获取回复
    answer = bot.chat(messages, temperature=temperature, top_p=top_p, stream=stream)

    # 更新历史记录
    history.append({"user": query, "bot": answer})
    return answer


# 清空历史对话记录
def reset():
    history.clear()
