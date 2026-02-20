# Azure AI Automation Mini Project
A FastAPI-based AI backend deployed on Azure App Service, designed to explore cloud-hosted AI automation, Azure OpenAI integration, and production-ready API design patterns.

This project focuses on real-world deployment, configuration-driven architecture, and safe, observable AI inference rather than experimental or toy implementations.

Overview

This service exposes a REST API that routes AI generation requests to a configurable provider (currently Azure OpenAI) while enforcing input validation, token limits, and readiness checks.

The application is fully deployed to Azure App Service (Linux) using ZIP deployment, with environment-based configuration and health probes suitable for production environments.

Features (v0.3)

FastAPI backend with modular provider architecture

Azure OpenAI integration using deployment-based routing

Environment-variable–driven configuration (no hardcoded secrets)

Prompt length validation and output token guardrails

Cached Azure OpenAI client for efficiency

Structured logging with request IDs

Health (/health) and readiness (/ready) probes

Configuration inspection endpoint (/config)

Verified production deployment on Azure App Service

Tech Stack

Python

FastAPI

Azure App Service (Linux)

Azure OpenAI

Uvicorn

ZIP Deployment

python-dotenv (local development)

GitHub (CI/CD planned)

API Endpoints
Health & Readiness

GET /health
Basic service liveness check.

GET /ready
Validates Azure OpenAI configuration and deployment readiness.

GET /config
Returns a sanitized view of active runtime configuration (no secrets).

AI Generation

POST /ai/generate

Routes a prompt to the configured AI provider.

Request Body
{
  "prompt": "Say \"Version 0.3 running perfectly.\"",
  "temperature": 0.2
}
Response
{
  "provider": "azure_openai",
  "output": "Version 0.3 running perfectly.",
  "request_id": "optional-request-id"
}
Validation & Guardrails

Prompt length capped (default: 4000 characters)

Output token limit configurable via environment variable

Graceful handling of rate limits, authentication errors, and timeouts

Configuration

The application is fully configured using environment variables.

Core
AI_PROVIDER=azure_openai
PORT=8000
Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-10-21
Limits
MAX_OUTPUT_TOKENS=256

A .env.example file is included for reference.
Sensitive values are never committed to version control.

Deployment

Platform: Azure App Service (Linux)

Deployment method: ZIP Deployment

Server: Uvicorn

Startup: Custom startup.sh

The deployed service has been verified to behave identically to local execution.

Project Status

Version 0.3 – Stable

Azure OpenAI integration complete

Configuration and validation finalized

Health, readiness, and config endpoints implemented

Production deployment verified

Planned Improvements

API authentication (API keys or OAuth)

Rate limiting and usage quotas

GitHub Actions CI/CD pipeline

Support for additional AI providers

Usage metrics and cost visibility

Author

Mohib Zaidi
AI & Full-Stack Developer
University of Alberta
