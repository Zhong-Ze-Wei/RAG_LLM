from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage, SystemMessage

class DeepSeekChat:
    def __init__(self, api_key, model_name="deepseek-chat", **kwargs):
        self.llm = ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            **kwargs
        )

    def chat(self, messages, **kwargs):
        formatted_messages = [
            SystemMessage(content=msg['content']) if msg['role'] == 'system' else HumanMessage(content=msg['content'])
            for msg in messages
        ]
        response = self.llm.invoke(formatted_messages, **kwargs)
        return response.content.strip()
