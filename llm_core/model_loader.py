import yaml
from llm_core.api_wrapper.deepseek_api import DeepSeekChat  # DeepSeek API 封装类
from chatbot import ChatBot  # 本地模型封装类（自定义）

# 加载 YAML 配置文件，默认路径为 config/model_config.yaml
def load_config(config_path="config/model_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 根据配置文件内容加载对应的模型实例
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
