"""SIGNBRIDGE streaming-recognition service contract.

This service intentionally does not generate predictions until a validated model
is installed. It must never substitute demo labels or fabricated translations.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI(title="SIGNBRIDGE ISL Streaming API", version="0.1.0")


def model_status() -> dict:
    # Replace only when a real, versioned and evaluated model is loaded.
    return {
        "service": "signbridge-isl-stream",
        "status": "model_not_configured",
        "model_loaded": False,
        "translation_enabled": False,
        "message": "No validated ISL recognition model is installed. No translation is available.",
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "signbridge-isl-stream"}


@app.get("/v1/recognition/status")
async def recognition_status():
    status = model_status()
    return JSONResponse(status_code=503, content=status)


@app.websocket("/v1/recognition/session")
async def recognition_session(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "recognition_unavailable",
        **model_status(),
        "safe_fallback": "Use text input or a qualified human interpreter.",
    })
    # Do not accept camera frames or emit guesses while no model is configured.
    await websocket.close(code=1013, reason="Validated recognition model not configured")
