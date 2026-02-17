import os
import time
import uuid
import logging
from typing import Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

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
    return {"ready": True}

# ---- AI (v0): provider switch ----
AI_PROVIDER = os.getenv("AI_PROVIDER", "mock").lower()
# Future env vars (for Azure OpenAI)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") 

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: float = Field(0.2, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    provider: str
    output: str
    request_id: Optional[str] = None

def generate_mock(prompt: str) -> str:
    # Simple deterministic behavior for now (safe + deployable)
    return f"MOCK_RESPONSE: {prompt.strip()}"

@app.post("/ai/generate", response_model=GenerateResponse)
def ai_generate(payload: GenerateRequest, request: Request):
    # Pull request id from middleware header if present
    request_id = request.headers.get("x-request-id")

    if AI_PROVIDER == "mock":
        out = generate_mock(payload.prompt)
        return GenerateResponse(provider="mock", output=out, request_id=request_id)

    # Placeholder for real provider(s) later
    return GenerateResponse(
        provider=AI_PROVIDER,
        output="Provider not configured yet. Set AI_PROVIDER=mock for now.",
        request_id=request_id
    )
