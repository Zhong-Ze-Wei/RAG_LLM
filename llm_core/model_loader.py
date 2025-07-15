from llm_core.api_wrapper.deepseek_api import DeepSeekChat
from chatbot import ChatBot

def load_model():
    config = load_config()
    mode = config.get("mode", "local")  # 支持 "local" 或 "api" 模式

    if mode == "local":
        model_path = config["local"]["model_path"]
        return ChatBot(model_path)  # 返回本地模型实例
    elif mode == "api":
        api_key = config["api"]["api_key"]
        model_name = config["api"]["model_name"]
        return DeepSeekChat(api_key=api_key, model_name=model_name)  # 返回 API 模型实例
    else:
        raise ValueError(f"Unsupported mode: {mode}")
