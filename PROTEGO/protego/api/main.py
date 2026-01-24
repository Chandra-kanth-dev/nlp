"""
main.py
=======
FastAPI entry point for PROTEGO.

Responsibilities:
- App initialization
- Middleware configuration
- API routing
- Error handling
- Chatbot orchestration
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from protego.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResponseWithDebug,
)
from protego.api.chatbot_service import handle_message

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="PROTEGO API",
    description="AI-powered safety & support chatbot for crisis prevention",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# -----------------------------
# CORS configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    print("✅ PROTEGO API started successfully")

# -----------------------------
# Health check
# -----------------------------
@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "running",
        "service": "PROTEGO API",
        "message": "PROTEGO API is live 🛡️"
    }

# -----------------------------
# Chat endpoint (PRIMARY)
# -----------------------------
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat with PROTEGO"
)
async def chat_endpoint(request: ChatRequest):
    try:
        return handle_message(
            message=request.message,
            country=request.country,
            debug=False
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except RuntimeError as e:
        # 🔥 Show real internal error
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    except Exception as e:
        # 🔥 Catch-all with traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}"
        )

# -----------------------------
# Debug endpoint (CONTROLLED)
# -----------------------------
@app.post(
    "/chat/debug",
    response_model=ChatResponseWithDebug,
    tags=["Debug"],
    summary="Debug chatbot reasoning"
)
async def chat_debug_endpoint(request: ChatRequest):
    try:
        return handle_message(
            message=request.message,
            country=request.country,
            debug=True
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Debug endpoint failure",
                "details": str(e)
            }
        )
