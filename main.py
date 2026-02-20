import os
import time
import uuid
import logging
from typing import Optional, Dict

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from openai import AzureOpenAI

# -----------------------------
# Environment / dotenv (v0.3)
# -----------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
if ENVIRONMENT != "production":
    # In production (Azure), use App Service "Environment variables" (App settings)
    # In dev, load local .env
    load_dotenv()

app = FastAPI()

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# -----------------------------
# Configuration helpers (v0.3)
# -----------------------------
def get_provider() -> str:
    return os.getenv("AI_PROVIDER", "mock").strip().lower()

def get_system_prompt() -> str:
    return os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.").strip()

def get_max_output_tokens() -> int:
    raw = os.getenv("MAX_OUTPUT_TOKENS", "256").strip()
    try:
        val = int(raw)
    except ValueError:
        val = 256
    # Guardrails: keep it reasonable for safety/cost
    return max(1, min(val, 1024))

def get_max_prompt_chars() -> int:
    raw = os.getenv("MAX_PROMPT_CHARS", "4000").strip()
    try:
        val = int(raw)
    except ValueError:
        val = 4000
    # Guardrails: allow some flexibility but keep it sane
    return max(1, min(val, 8000))

def get_azure_config() -> Dict[str, Optional[str]]:
    return {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    }

def azure_config_ok(cfg: Dict[str, Optional[str]]) -> bool:
    return bool(cfg["endpoint"] and cfg["api_key"] and cfg["deployment"])

# Cache Azure client (avoid re-creating on every request)
_AZURE_CLIENT: Optional[AzureOpenAI] = None
_AZURE_CLIENT_KEY: Optional[str] = None

def get_azure_client(cfg: Dict[str, Optional[str]]) -> AzureOpenAI:
    global _AZURE_CLIENT, _AZURE_CLIENT_KEY

    key_fingerprint = f"{cfg['endpoint']}|{cfg['api_key']}|{cfg['api_version']}"
    if _AZURE_CLIENT is None or _AZURE_CLIENT_KEY != key_fingerprint:
        # NOTE: AzureOpenAI SDK supports different timeout options depending on version.
        # Keep this minimal for compatibility.
        _AZURE_CLIENT = AzureOpenAI(
            azure_endpoint=cfg["endpoint"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"] or "2024-02-01",
        )
        _AZURE_CLIENT_KEY = key_fingerprint
    return _AZURE_CLIENT


# ---- Middleware ----
class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.time()

        response = await call_next(request)

        duration_ms = int((time.time() - start) * 1000)
        response.headers["x-request-id"] = request_id

        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration_ms={duration_ms} "
            f"request_id={request_id}"
        )
        return response

app.add_middleware(RequestLogMiddleware)

# ---- Health / readiness ----
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    provider = get_provider()
    if provider == "mock":
        return {"ready": True, "provider": "mock"}

    if provider == "azure_openai":
        cfg = get_azure_config()
        ok = azure_config_ok(cfg)
        return {
            "ready": ok,
            "provider": "azure_openai",
            "deployment": cfg["deployment"],
            "endpoint_set": bool(cfg["endpoint"]),
            "api_version": cfg["api_version"],
            "reason": None if ok else "missing_azure_config",
        }

    return {"ready": False, "provider": provider, "reason": "unknown_provider"}

# ---- Safe debug config (no secrets) ----
@app.get("/config")
def config():
    cfg = get_azure_config()
    return {
        "environment": ENVIRONMENT,
        "provider": get_provider(),
        "max_output_tokens": get_max_output_tokens(),
        "max_prompt_chars": get_max_prompt_chars(),
        "system_prompt_set": bool(get_system_prompt()),
        "azure": {
            "deployment_set": bool(cfg["deployment"]),
            "endpoint_set": bool(cfg["endpoint"]),
            "api_version": cfg["api_version"],
        },
    }


# ---- Models ----
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)  # hard cap; further limited by env
    temperature: float = Field(0.2, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    provider: str
    output: str
    request_id: Optional[str] = None


# ---- Providers ----
def generate_mock(prompt: str) -> str:
    return f"MOCK_RESPONSE: {prompt.strip()}"

def generate_azure_openai(prompt: str, temperature: float, request_id: Optional[str]) -> str:
    cfg = get_azure_config()
    if not azure_config_ok(cfg):
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT.",
        )

    deployment = cfg["deployment"] or ""
    max_tokens = get_max_output_tokens()
    system_prompt = get_system_prompt()

    client = get_azure_client(cfg)

    # Measure only the AI call duration
    ai_start = time.time()
    try:
        resp = client.chat.completions.create(
            model=deployment,  # Azure uses deployment name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        ai_ms = int((time.time() - ai_start) * 1000)
        msg = str(e).lower()

        # Common Azure/OpenAI failure modes
        if "429" in msg or "rate limit" in msg or "quota" in msg:
            logger.warning(
                f"ai_generate provider=azure_openai ok=false "
                f"error=rate_limited ai_duration_ms={ai_ms} "
                f"deployment={deployment} request_id={request_id}"
            )
            raise HTTPException(status_code=429, detail="Azure OpenAI rate limited or quota exceeded.")

        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            logger.error(
                f"ai_generate provider=azure_openai ok=false "
                f"error=auth_failed ai_duration_ms={ai_ms} "
                f"deployment={deployment} request_id={request_id}"
            )
            raise HTTPException(status_code=502, detail="Azure OpenAI authentication/authorization failed.")

        if "timeout" in msg or "timed out" in msg:
            logger.warning(
                f"ai_generate provider=azure_openai ok=false "
                f"error=timeout ai_duration_ms={ai_ms} "
                f"deployment={deployment} request_id={request_id}"
            )
            raise HTTPException(status_code=504, detail="Azure OpenAI request timed out.")

        if "name or service not known" in msg or "dns" in msg:
            logger.error(
                f"ai_generate provider=azure_openai ok=false "
                f"error=dns ai_duration_ms={ai_ms} "
                f"deployment={deployment} request_id={request_id}"
            )
            raise HTTPException(status_code=502, detail="Azure OpenAI endpoint could not be reached (DNS/network).")

        # Wrong deployment/model not found (common)
        if ("not found" in msg or "resource not found" in msg) and ("deployment" in msg or "model" in msg):
            logger.error(
                f"ai_generate provider=azure_openai ok=false "
                f"error=deployment_not_found ai_duration_ms={ai_ms} "
                f"deployment={deployment} request_id={request_id}"
            )
            raise HTTPException(
                status_code=502,
                detail="Azure OpenAI deployment not found. Check AZURE_OPENAI_DEPLOYMENT matches your deployed model name.",
            )

        logger.error(
            f"ai_generate provider=azure_openai ok=false "
            f"error=unknown ai_duration_ms={ai_ms} "
            f"deployment={deployment} request_id={request_id}"
        )
        raise HTTPException(status_code=502, detail="Azure OpenAI request failed.")

    ai_ms = int((time.time() - ai_start) * 1000)
    content = (resp.choices[0].message.content or "").strip()

    logger.info(
        f"ai_generate provider=azure_openai ok=true "
        f"ai_duration_ms={ai_ms} deployment={deployment} "
        f"max_tokens={max_tokens} prompt_chars={len(prompt)} "
        f"request_id={request_id}"
    )
    return content


# ---- Routes ----
@app.post("/ai/generate", response_model=GenerateResponse)
def ai_generate(payload: GenerateRequest, request: Request):
    request_id = request.headers.get("x-request-id")
    provider = get_provider()

    # Enforce env-based prompt length cap (cost control)
    max_chars = get_max_prompt_chars()
    prompt = payload.prompt.strip()
    if len(prompt) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long. Max allowed is {max_chars} characters.",
        )

    if provider == "mock":
        out = generate_mock(prompt)
        return GenerateResponse(provider="mock", output=out, request_id=request_id)

    if provider == "azure_openai":
        out = generate_azure_openai(
            prompt=prompt,
            temperature=payload.temperature,
            request_id=request_id,
        )
        return GenerateResponse(provider="azure_openai", output=out, request_id=request_id)

    raise HTTPException(
        status_code=400,
        detail="Provider not configured. Use AI_PROVIDER=mock or AI_PROVIDER=azure_openai.",
    )
