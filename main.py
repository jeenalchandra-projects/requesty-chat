import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from database import get_sessions, init_db, save_session

# ── Config ────────────────────────────────────────────────────────
REQUESTY_API_KEY: str = os.getenv("REQUESTY_API_KEY", "")
REQUESTY_ROUTER:  str = "https://router.requesty.ai/v1"
REQUESTY_MGMT:    str = "https://api-v2.requesty.ai/v1"
CHAT_PASSWORD:    str = os.getenv("CHAT_PASSWORD", "")
TAVILY_API_KEY:   str = os.getenv("TAVILY_API_KEY", "")

_raw_secret = os.getenv("SECRET_KEY") or CHAT_PASSWORD or secrets.token_hex(32)
SECRET_KEY: str = hashlib.sha256(_raw_secret.encode()).hexdigest()

# Memory file paths (on Railway volume at /data, or local fallback)
_data_dir    = os.getenv("DATA_DIR", "data")
MEMORY_PATH  = os.path.join(_data_dir, "memory.md")
SESSION_PATH = os.path.join(_data_dir, "session_context.md")


# ── Tool definitions ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for current information, recent news, prices, "
                "weather, facts, or anything that may have changed. Use this whenever "
                "the user asks about something that requires up-to-date knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": (
                "Save important information to permanent memory. "
                "Use ONLY when the user explicitly asks you to remember something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"}
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_memory",
            "description": (
                "Load everything stored in permanent memory. "
                "Use when the user asks you to recall, check, or refer to memory."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ── Memory helpers ────────────────────────────────────────────────
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def memory_load() -> str:
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content or "Memory is empty."
    except FileNotFoundError:
        return "Memory is empty."


def memory_append(content: str):
    _ensure_dir(MEMORY_PATH)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n---\n**{ts}**\n{content}\n")


def session_load() -> list:
    try:
        with open(SESSION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def session_save(messages: list):
    _ensure_dir(SESSION_PATH)
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f)


def session_clear():
    _ensure_dir(SESSION_PATH)
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)


# ── Tavily search ─────────────────────────────────────────────────
async def tavily_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return "Web search is not configured (TAVILY_API_KEY missing)."
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": True,
                },
                timeout=20.0,
            )
            data = r.json()
            parts = []
            if data.get("answer"):
                parts.append(f"**Summary:** {data['answer']}\n")
            for res in data.get("results", [])[:4]:
                parts.append(
                    f"• **{res.get('title', '')}**\n"
                    f"  {res.get('url', '')}\n"
                    f"  {(res.get('content') or '')[:350]}"
                )
            return "\n\n".join(parts) if parts else "No results found."
        except Exception as exc:
            return f"Search error: {exc}"


# ── Tool executor ─────────────────────────────────────────────────
async def execute_tool(name: str, args: dict) -> str:
    if name == "search_web":
        return await tavily_search(args.get("query", ""))
    if name == "save_to_memory":
        memory_append(args.get("content", ""))
        return "✓ Saved to memory."
    if name == "load_memory":
        return memory_load()
    return f"Unknown tool: {name}"


# ── App setup ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _ensure_dir(MEMORY_PATH)
    _ensure_dir(SESSION_PATH)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=False)
templates = Jinja2Templates(directory="templates")


def _auth(request: Request) -> bool:
    return True  # re-enable after testing: return bool(request.session.get("authenticated"))


# ── Auth ──────────────────────────────────────────────────────────
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
    session_clear()
    return RedirectResponse("/login")


# ── Chat page (GET + POST server-side fallback) ───────────────────
async def _fetch_models() -> list:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{REQUESTY_ROUTER}/models",
                headers={"Authorization": f"Bearer {REQUESTY_API_KEY}"},
                timeout=15.0,
            )
            return sorted([m["id"] for m in r.json().get("data", [])])
        except Exception:
            return []


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    if not _auth(request):
        return RedirectResponse("/login")
    models   = await _fetch_models()
    sessions = get_sessions(limit=30)
    local_spend = round(sum(s["total_cost"] or 0 for s in sessions), 6)
    # Restore session context
    saved_conversation = session_load()
    return templates.TemplateResponse(
        request=request, name="chat.html",
        context={
            "models": models, "sessions": sessions, "local_spend": local_spend,
            "conversation": saved_conversation, "selected_model": "",
            "error": "", "history_b64": "",
        },
    )


@app.post("/chat", response_class=HTMLResponse)
async def chat_submit(
    request: Request,
    model: str = Form(""),
    message: str = Form(""),
    history: str = Form("[]"),
):
    import base64
    if not _auth(request):
        return RedirectResponse("/login")
    try:
        conversation = json.loads(base64.b64decode(history).decode())
    except Exception:
        conversation = []

    error = ""
    if message.strip() and model:
        conversation.append({"role": "user", "content": message.strip()})
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(
                    f"{REQUESTY_ROUTER}/chat/completions",
                    headers={"Authorization": f"Bearer {REQUESTY_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model, "messages": conversation, "stream": False},
                    timeout=120.0,
                )
                r.raise_for_status()
                data   = r.json()
                reply  = data["choices"][0]["message"]["content"]
                usage  = data.get("usage", {})
                conversation.append({"role": "assistant", "content": reply})
                session_save(conversation)
                save_session(
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_cost=usage.get("total_cost") or usage.get("cost"),
                    first_message=message.strip()[:120],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as exc:
                error = str(exc)
                conversation.pop()

    models   = await _fetch_models()
    sessions = get_sessions(limit=30)
    local_spend = round(sum(s["total_cost"] or 0 for s in sessions), 6)
    history_b64 = __import__("base64").b64encode(json.dumps(conversation).encode()).decode()
    return templates.TemplateResponse(
        request=request, name="chat.html",
        context={
            "models": models, "sessions": sessions, "local_spend": local_spend,
            "conversation": conversation, "error": error,
            "selected_model": model, "history_b64": history_b64,
        },
    )


# ── API: models ───────────────────────────────────────────────────
@app.get("/api/models")
async def api_models(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    models = await _fetch_models()
    return {"models": models}


# ── API: chat (JS, with tool-use loop) ───────────────────────────
@app.post("/api/chat")
async def api_chat(request: Request):
    if not _auth(request):
        raise HTTPException(401)

    body     = await request.json()
    model:   str  = body.get("model", "")
    messages: list = body.get("messages", [])

    working = list(messages)
    reply   = ""

    async with httpx.AsyncClient() as client:
        try:
            for _round in range(6):  # max 5 tool-call rounds
                req_body: dict = {
                    "model":    model,
                    "messages": working,
                    "stream":   False,
                }
                if TAVILY_API_KEY:
                    req_body["tools"]       = TOOLS
                    req_body["tool_choice"] = "auto"

                r = await client.post(
                    f"{REQUESTY_ROUTER}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {REQUESTY_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=req_body,
                    timeout=120.0,
                )

                # Graceful fallback: model doesn't support tool use
                if r.status_code in (400, 422) and TAVILY_API_KEY:
                    req_body.pop("tools", None)
                    req_body.pop("tool_choice", None)
                    r = await client.post(
                        f"{REQUESTY_ROUTER}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {REQUESTY_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=req_body,
                        timeout=120.0,
                    )

                r.raise_for_status()
                data    = r.json()
                msg     = data["choices"][0]["message"]
                tc_list = msg.get("tool_calls") or []

                if not tc_list:
                    # Final answer
                    reply = msg.get("content") or ""
                    usage = data.get("usage", {})
                    first_user = next(
                        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
                    )
                    # Append assistant reply to working messages so session is complete
                    working.append({"role": "assistant", "content": reply})
                    session_save(working)
                    save_session(
                        model=model,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_cost=usage.get("total_cost") or usage.get("cost"),
                        first_message=str(first_user)[:120],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    break

                # Execute tool calls
                working.append(msg)
                for tc in tc_list:
                    fn_name  = tc["function"]["name"]
                    fn_args  = json.loads(tc["function"]["arguments"] or "{}")
                    result   = await execute_tool(fn_name, fn_args)
                    working.append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      result,
                    })

            return {
                "reply": reply,
                "prompt_tokens":     data["usage"].get("prompt_tokens", 0) if "usage" in data else 0,
                "completion_tokens": data["usage"].get("completion_tokens", 0) if "usage" in data else 0,
                "cost":              (data.get("usage") or {}).get("total_cost"),
            }

        except Exception as exc:
            return {"error": str(exc)}


# ── API: document upload ──────────────────────────────────────────
@app.post("/api/upload")
async def api_upload(request: Request, file: UploadFile = File(...)):
    if not _auth(request):
        raise HTTPException(401)

    raw      = await file.read()
    filename = file.filename or ""
    text     = ""

    try:
        if filename.lower().endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(BytesIO(raw))
            text   = "\n\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )
        elif filename.lower().endswith(".docx"):
            import docx as _docx
            doc  = _docx.Document(BytesIO(raw))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return {"error": f"Could not extract text: {exc}"}

    if len(text) > 60_000:
        text = text[:60_000] + "\n\n[Truncated at 60 000 characters]"

    return {"text": text, "filename": filename, "char_count": len(text)}


# ── API: balance ──────────────────────────────────────────────────
@app.get("/api/balance")
async def api_balance(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    sessions    = get_sessions(limit=10000)
    local_spend = round(sum(s["total_cost"] or 0 for s in sessions), 6)
    return {"local_spend": local_spend}


# ── API: sessions ─────────────────────────────────────────────────
@app.get("/api/sessions")
async def api_sessions(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    return {"sessions": get_sessions(limit=50)}


# ── API: memory ───────────────────────────────────────────────────
@app.get("/api/memory")
async def api_memory_get(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    return {"content": memory_load()}


@app.delete("/api/memory")
async def api_memory_delete(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)
    return {"ok": True}


# ── API: session context ──────────────────────────────────────────
@app.get("/api/session-context")
async def api_session_context(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    return {"messages": session_load()}


@app.delete("/api/session-context")
async def api_session_context_clear(request: Request):
    if not _auth(request):
        raise HTTPException(401)
    session_clear()
    return {"ok": True}
