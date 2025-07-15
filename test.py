# test.py or main.py
from llm_core.inference import chat

while True:
    q = input("你：")
    print("模型：", chat(q))
