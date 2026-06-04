import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from database import get_sessions, init_db, save_session

REQUESTY_API_KEY: str = os.getenv("REQUESTY_API_KEY", "")
REQUESTY_ROUTER: str = "https://router.requesty.ai/v1"
REQUESTY_MGMT: str = "https://api-v2.requesty.ai/v1"
CHAT_PASSWORD: str = os.getenv("CHAT_PASSWORD", "")

# Stable secret key: explicit env var wins; otherwise derive from CHAT_PASSWORD
# so the key survives server restarts without requiring a separate env var.
_raw_secret = os.getenv("SECRET_KEY") or CHAT_PASSWORD or secrets.token_hex(32)
SECRET_KEY: str = hashlib.sha256(_raw_secret.encode()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=False,   # Railway terminates TLS at the edge; Secure flag not needed
)
templates = Jinja2Templates(directory="templates")


def _auth(request: Request) -> bool:
    return True  # auth temporarily disabled


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/chat" if _auth(request) else "/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _auth(request):
        return RedirectResponse("/chat")
    return templates.TemplateResponse(request=request, name="login.html", context={"error": ""})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, password: str = Form(...)):
    if CHAT_PASSWORD and password == CHAT_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse("/chat", status_code=303)
    return templates.TemplateResponse(
        request=request, name="login.html", context={"error": "Incorrect password"}
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


# ── Page ───────────────────────────────────────────────────────────────────────

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    if not _auth(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse(request=request, name="chat.html", context={})


# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def api_models(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{REQUESTY_ROUTER}/models",
                headers={"Authorization": f"Bearer {REQUESTY_API_KEY}"},
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
            models = sorted([m["id"] for m in data.get("data", [])])
            return {"models": models}
        except Exception as exc:
            return {"models": [], "error": str(exc)}


@app.get("/api/balance")
async def api_balance(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{REQUESTY_MGMT}/manage/org",
                headers={"Authorization": f"Bearer {REQUESTY_API_KEY}"},
                timeout=10.0,
            )
            r.raise_for_status()
            return r.json()   # contains { name, balance }
        except Exception as exc:
            return {"error": str(exc)}


@app.post("/api/chat/stream")
async def api_chat_stream(request: Request):
    if not _auth(request):
        raise HTTPException(401)

    body = await request.json()
    model: str = body.get("model", "")
    messages: list = body.get("messages", [])

    async def event_stream():
        prompt_tokens = 0
        completion_tokens = 0
        total_cost: Optional[float] = None

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{REQUESTY_ROUTER}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {REQUESTY_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    timeout=120.0,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get(
                                "completion_tokens", completion_tokens
                            )
                            total_cost = (
                                usage.get("total_cost")
                                or usage.get("cost")
                                or total_cost
                            )

                        choices = chunk.get("choices") or []
                        if choices:
                            content = (choices[0].get("delta") or {}).get("content")
                            if content:
                                yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"

                first_user = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"),
                    "",
                )
                save_session(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_cost=total_cost,
                    first_message=str(first_user)[:120],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                done_payload = json.dumps({
                    "type": "done",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": total_cost,
                })
                yield f"data: {done_payload}\n\n"

            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/sessions")
async def api_sessions(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    return {"sessions": get_sessions(limit=50)}
