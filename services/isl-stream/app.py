"""SIGNBRIDGE streaming-recognition service contract.

A checkpoint may be inspected and loaded, but live recognition remains disabled
until the camera preprocessing and evaluated streaming adapter are implemented.
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

try:
    from .model_runtime import ModelRuntime
except ImportError:  # Running uvicorn with this directory as its working directory.
    from model_runtime import ModelRuntime

app = FastAPI(title="SIGNBRIDGE Sign Streaming API", version="0.2.0")
runtime = ModelRuntime()


def model_status(*, attempt_load: bool = False) -> dict:
    return runtime.status(attempt_load=attempt_load)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "signbridge-sign-stream"}


@app.get("/v1/recognition/status")
async def recognition_status():
    status = model_status(attempt_load=True)
    # Even a successfully loaded research checkpoint is not a live translator.
    return JSONResponse(status_code=503, content=status)


@app.websocket("/v1/recognition/session")
async def recognition_session(websocket: WebSocket):
    await websocket.accept()
    status = model_status(attempt_load=True)
    await websocket.send_json({
        "type": "recognition_unavailable",
        **status,
        "safe_fallback": "Use text input or a qualified human interpreter.",
    })
    # Do not accept camera frames or emit guesses until the complete streaming
    # recognition and translation path has been independently evaluated.
    await websocket.close(code=1013, reason="Live recognition pipeline not integrated")
