"""FastAPI chat server.

Endpoints:
  GET  /                       — static chat UI (src/nova/app/ui/index.html)
  POST /api/chat               — JSON {reply} (non-stream)
  POST /api/chat/stream        — SSE stream of reply chunks
  POST /api/new                — reset the server-side conversation
  GET  /api/model              — {model info, capabilities, context}
  GET /api/health              — liveness

The server owns ONE conversation (single-user demo). Multi-user session
management is TODO (see ROADMAP).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nova import chat_template as ct
from nova.inference.generator import Generator
from nova.safety.boundaries import is_prohibited, REFUSAL


class ChatIn(BaseModel):
    """Chat request body (module level so Pydantic can resolve it)."""
    message: str
    temperature: float | None = None
    max_new_tokens: int | None = None

HERE = os.path.dirname(os.path.abspath(__file__))          # .../nova_assistant/src/nova/api
NOVA_PKG = os.path.dirname(HERE)                            # .../src/nova
SRC = os.path.dirname(NOVA_PKG)                             # .../src
PROJECT = os.path.dirname(SRC)                              # .../nova_assistant
UI = os.path.join(NOVA_PKG, "app", "ui")


def load_server_config(path: str) -> dict:
    if not os.path.exists(path):
        return {"checkpoint": None, "port": 8001, "temperature": 0.8,
                "max_new_tokens": 96, "top_k": 50}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_app(checkpoint: str = None, temperature: float = 0.8,
             max_new_tokens: int = 96, top_k: int = 50) -> FastAPI:
    gen = Generator.from_checkpoint(checkpoint)
    convo = ct.Conversation()
    settings = {"temperature": temperature, "max_new_tokens": max_new_tokens, "top_k": top_k}

    app = FastAPI(title="Nova Assistant")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/model")
    def model_info():
        return {
            "name": os.path.basename(os.path.dirname(checkpoint)),
            "checkpoint": checkpoint,
            "context_tokens": gen.model.cfg.max_seq_len,
            "vocab_size": gen.tok.vocab_size,
            "device": gen.device,
            "capabilities": "experimental character/subword baseline — NOT a production assistant",
            "settings": settings,
        }

    @app.post("/api/new")
    def new_chat():
        convo.reset()
        return {"ok": True, "turns": 0}

    @app.post("/api/chat")
    def chat(body: ChatIn):
        if is_prohibited(body.message):
            convo.add(body.message, REFUSAL)
            return {"reply": REFUSAL}
        convo.add(body.message)
        try:
            temp = body.temperature if body.temperature is not None else settings["temperature"]
            newtok = body.max_new_tokens or settings["max_new_tokens"]
            reply = gen.chat_reply(convo, max_new_tokens=newtok, temperature=temp,
                                   top_k=settings["top_k"], seed=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generation failed: {e}")
        convo.turns[-1]["assistant"] = reply
        return {"reply": reply}

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatIn):
        if is_prohibited(body.message):
            def deny():
                yield "data: " + json.dumps({"refusal": REFUSAL}) + "\n\n"
                yield "data: " + json.dumps({"done": True}) + "\n\n"
            return StreamingResponse(deny(), media_type="text/event-stream")
        convo.add(body.message)
        temp = body.temperature if body.temperature is not None else settings["temperature"]
        newtok = body.max_new_tokens or settings["max_new_tokens"]

        def sse():
            full = ""
            try:
                for piece in gen.stream(
                    prompt=_render_prompt(convo),
                    max_new_tokens=newtok, temperature=temp,
                    top_k=settings["top_k"],
                ):
                    full += piece
                    yield f"data: {json.dumps({'piece': piece})}\n\n"
                convo.turns[-1]["assistant"] = full
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                if not full:
                    raise
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    def _render_prompt(c):
        ids = ct.render(c, gen.tok, gen.model.cfg.max_seq_len - settings["max_new_tokens"])
        return gen.tok.decode(ids)

    if os.path.isdir(UI):
        app.mount("/", StaticFiles(directory=UI, html=True), name="ui")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT, "configs", "server.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = load_server_config(args.config)
    ckpt = args.checkpoint or cfg.get("checkpoint")
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint configured/available: {ckpt}")
    port = args.port or cfg.get("port", 8001)
    app = make_app(ckpt, cfg.get("temperature", 0.8), cfg.get("max_new_tokens", 96),
                   cfg.get("top_k", 50))
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
