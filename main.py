# main.py
from fastapi import FastAPI
from router.cht import router as chat_router

app = FastAPI(title="LightRAG Chat API")

# include your router
app.include_router(chat_router)

# optional simple root
@app.get("/")
def root():
    return {"status": "ok", "routes": ["/cht/chat", "/health"]}

if __name__ == "__main__":
    import uvicorn
    # run with: python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


