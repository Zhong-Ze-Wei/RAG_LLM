from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llm_core.inference import chat, reset
import asyncio
import json
import inspect

app = FastAPI(title="LLM聊天接口")

class ChatRequest(BaseModel):
    query: str
    temperature: float = 0.5
    top_p: float = 0.95
    stream: bool = False
    role: str = "default"
    history: list[dict] = []

@app.post("/chat")
async def chat_api(req: ChatRequest):
    print(f"[chat_api] 接收到请求：query='{req.query}', role='{req.role}', temperature={req.temperature}, top_p={req.top_p}, stream={req.stream}")

    if req.stream:
        if inspect.isasyncgenfunction(chat):
            async def event_generator():
                async for chunk in chat(
                    req.query,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stream=True,        # 流式传输
                    role=req.role,
                    history=req.history
                ):
                    yield f"data: {json.dumps({'answer': chunk})}\n\n"
                    await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            answer = chat(
                req.query,
                temperature=req.temperature,
                top_p=req.top_p,
                stream=False,           #非流式传输
                role=req.role,
                history=req.history
            )
            return JSONResponse({"answer": answer})
    else:
        if inspect.iscoroutinefunction(chat):
            answer = await chat(
                req.query,
                temperature=req.temperature,
                top_p=req.top_p,
                stream=False,
                role=req.role,
                history=req.history
            )
        else:
            answer = chat(
                req.query,
                temperature=req.temperature,
                top_p=req.top_p,
                stream=False,
                role=req.role,
                history=req.history
            )
        return JSONResponse({"answer": answer})

@app.post("/reset")
async def reset_api():
    reset()
    return JSONResponse({"message": "历史对话已重置"})

app.mount("/", StaticFiles(directory="ui", html=True), name="static")
