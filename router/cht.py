# routers/cht.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# legacy OpenAI SDK (openai==0.28.x)
import openai
from typing import List, Dict

# ----------------------------
# CONFIG
# ----------------------------
WORKING_DIR = "./lightrag_index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

openai.api_key = OPENAI_API_KEY


# ----------------------------
# Legacy OpenAI wrapper functions
# ----------------------------
def openai_chat(messages: List[Dict[str, str]], model="gpt-4o-mini") -> str:
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"openai_chat error: {e}")


def openai_embedding(texts: List[str], model="text-embedding-3-large"):
    try:
        resp = openai.Embedding.create(
            model=model,
            input=texts
        )
        return [item["embedding"] for item in resp["data"]]
    except Exception as e:
        raise RuntimeError(f"openai_embedding error: {e}")


# ----------------------------
# LightRAG Setup
# ----------------------------
embedding_dim = 3072  # for text-embedding-3-large

embedding_func = EmbeddingFunc(
    embedding_dim=embedding_dim,
    max_token_size=8192,
    func=lambda texts: openai_embedding(texts, "text-embedding-3-large")
)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=lambda messages: openai_chat(messages, "gpt-4o-mini"),
    embedding_func=embedding_func,
)


# ----------------------------
# REQUEST MODEL
# ----------------------------
class ChatRequest(BaseModel):
    mode: str = "chat"
    message: str
    top_k: int = 10


router = APIRouter(prefix="/cht", tags=["chat"])


# ----------------------------
# ROUTE
# ----------------------------
@router.post("/chat")
def chat(req: ChatRequest):
    try:
        # normal chat
        if req.mode == "chat":
            response = openai_chat(
                [{"role": "user", "content": req.message}],
                model="gpt-4o-mini"
            )
            return {"mode": "chat", "response": response}

        # RAG mode
        elif req.mode == "rag":
            params = QueryParam(
                mode="hybrid",
                top_k=req.top_k
            )
            answer = rag.query(req.message, param=params)
            return {"mode": "rag", "response": answer}

        else:
            raise HTTPException(status_code=400, detail="Mode must be 'chat' or 'rag'")


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
