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

def load_system_prompt(role="default"):
    base_dir = os.path.join(os.path.dirname(__file__), "prompts")
    prompt_path = os.path.join(base_dir, f"{role}.txt")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[提示] 未找到 {role}.txt，尝试加载 default.txt")
        try:
            fallback_path = os.path.join(base_dir, "default.txt")
            with open(fallback_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[错误] 加载默认提示词失败，使用内置默认。错误: {e}")
            return "你是一个有帮助的助手。"

def chat(query, temperature=0.5, top_p=0.95, stream=False, role="default", history=None):
    if history is None:
        history = []
    # print(f"[DEBUG] history content before processing: {history}")

    system_prompt = load_system_prompt(role)
    messages = [{"role": "system", "content": safe_str(system_prompt)}]

    for turn in history:
        # print(f"[DEBUG] processing turn: {turn}")

        messages.append({"role": "user", "content": safe_str(turn['user'])})
        messages.append({"role": "assistant", "content": safe_str(turn['bot'])})

    messages.append({"role": "user", "content": safe_str(query)})

    answer = bot.chat(messages, temperature=temperature, top_p=top_p, stream=stream)

    # 如果是流式，answer可能不是字符串，不能直接append
    if stream:
        # 你可以选择不更新历史，或者用别的方法记录
        pass
    else:
        # 确保answer是字符串
        if not isinstance(answer, str):
            answer = str(answer)
        history.append({"user": query, "bot": answer})

    return answer


# 清空历史对话记录
def reset():
    history.clear()
