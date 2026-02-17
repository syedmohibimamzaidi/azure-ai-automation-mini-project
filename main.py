import time
import uuid
import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()


logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

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

# Routes
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    return {"ready": True}