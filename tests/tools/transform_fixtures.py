"""Deterministic transform implementation used by absorption authority tests."""

def transform(request: dict) -> dict:
    if request.get("mode") != "default":
        raise ValueError("unsupported mode")
    return {"status": "ok"}
