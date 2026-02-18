import os
import time
import uuid
import logging
from typing import Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
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

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: float = Field(0.2, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    provider: str
    output: str
    request_id: Optional[str] = None

def generate_mock(prompt: str) -> str:
    return f"MOCK_RESPONSE: {prompt.strip()}"

def generate_azure_openai(prompt: str, temperature: float = 0.2) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not endpoint or not api_key or not deployment:
        raise RuntimeError(
            "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT."
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )

    resp = client.chat.completions.create(
        model=deployment,  # Azure uses your *deployment name*
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return (resp.choices[0].message.content or "").strip()

@app.post("/ai/generate", response_model=GenerateResponse)
def ai_generate(payload: GenerateRequest, request: Request):
    request_id = request.headers.get("x-request-id")
    provider = os.getenv("AI_PROVIDER", "mock").lower()

    if provider == "mock":
        out = generate_mock(payload.prompt)
        return GenerateResponse(provider="mock", output=out, request_id=request_id)

    if provider == "azure_openai":
        out = generate_azure_openai(payload.prompt, temperature=payload.temperature)
        return GenerateResponse(provider="azure_openai", output=out, request_id=request_id)

    return GenerateResponse(
        provider=provider,
        output="Provider not configured. Use AI_PROVIDER=mock or AI_PROVIDER=azure_openai.",
        request_id=request_id,
    )
