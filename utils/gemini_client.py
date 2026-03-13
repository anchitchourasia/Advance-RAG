import base64
import mimetypes
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from google import genai
from groq import Groq

from config import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    MODEL_NAME,
    GROQ_MODEL_NAME,
    GROQ_VISION_MODEL,
    APP_NAME,
    LANGSMITH_TRACING,
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
)


try:
    import langsmith as ls
    from langsmith import Client, traceable
    _LANGSMITH_AVAILABLE = True
except Exception:
    ls = None
    Client = None
    _LANGSMITH_AVAILABLE = False

    def traceable(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in environment variables.")


gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


langsmith_client = None
if _LANGSMITH_AVAILABLE and LANGSMITH_API_KEY:
    try:
        langsmith_client = Client(
            api_key=LANGSMITH_API_KEY,
            api_url=LANGSMITH_ENDPOINT,
        )
    except Exception:
        langsmith_client = None


_LAST_CALL_METRICS = {
    "text": {},
    "vision": {},
}


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    normalized = _normalize(text)
    if not normalized:
        return 0
    return max(1, int(len(normalized.split()) * 1.3))


def _record_metrics(
    call_type: str,
    provider: str,
    model: str,
    prompt: str,
    response: str,
    latency_ms: float,
    success: bool,
    fallback_used: bool = False,
):
    _LAST_CALL_METRICS[call_type] = {
        "provider": provider,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "fallback_used": fallback_used,
        "prompt_chars": len(prompt or ""),
        "response_chars": len(response or ""),
        "prompt_tokens_est": _estimate_tokens(prompt),
        "response_tokens_est": _estimate_tokens(response),
        "total_tokens_est": _estimate_tokens(prompt) + _estimate_tokens(response),
        "timestamp": time.time(),
    }


def get_last_call_metrics():
    return {
        "text": dict(_LAST_CALL_METRICS.get("text", {})),
        "vision": dict(_LAST_CALL_METRICS.get("vision", {})),
    }


def _langsmith_enabled() -> bool:
    return bool(_LANGSMITH_AVAILABLE and LANGSMITH_TRACING and langsmith_client is not None)


def _langsmith_extra():
    if langsmith_client is None:
        return None
    return {"client": langsmith_client}


def _trace_context(metadata=None, tags=None, enabled=None):
    if not _LANGSMITH_AVAILABLE or ls is None:
        return _NoOpContext()

    resolved_enabled = _langsmith_enabled() if enabled is None else bool(enabled and langsmith_client)

    if not resolved_enabled:
        return ls.tracing_context(enabled=False)

    return ls.tracing_context(
        enabled=True,
        project_name=LANGSMITH_PROJECT,
        metadata=metadata or {},
        tags=tags or [],
    )


@traceable(run_type="llm", name="gemini_text_call")
def _call_gemini_text(prompt: str, langsmith_extra=None) -> str:
    started = time.perf_counter()
    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    final_text = text.strip() if text and text.strip() else "I couldn’t generate a response right now."

    _record_metrics(
        call_type="text",
        provider="gemini",
        model=MODEL_NAME,
        prompt=prompt,
        response=final_text,
        latency_ms=(time.perf_counter() - started) * 1000,
        success=True,
        fallback_used=False,
    )
    return final_text


@traceable(run_type="llm", name="groq_text_call")
def _call_groq_text(prompt: str, langsmith_extra=None) -> str:
    if not groq_client:
        raise RuntimeError("Groq client is not configured.")

    started = time.perf_counter()
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are {APP_NAME}, an AI customer support assistant built for this application. "
                    "Keep this identity consistent. "
                    "Do not say you are Groq. "
                    "If asked what model powers you, say you are powered primarily by Google Gemini, "
                    "with Groq used as backup for text generation when needed."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_completion_tokens=800,
    )
    final_text = completion.choices[0].message.content.strip()

    _record_metrics(
        call_type="text",
        provider="groq",
        model=GROQ_MODEL_NAME,
        prompt=prompt,
        response=final_text,
        latency_ms=(time.perf_counter() - started) * 1000,
        success=True,
        fallback_used=True,
    )
    return final_text


@traceable(run_type="chain", name="ask_gemini_with_fallback")
def ask_gemini(prompt: str, langsmith_extra=None) -> str:
    trace_metadata = {
        "component": "text_generation",
        "primary_provider": "gemini",
        "backup_provider": "groq",
        "app_name": APP_NAME,
    }

    with _trace_context(metadata=trace_metadata, tags=["text", "fallback-router"]):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_gemini_text, prompt, _langsmith_extra())
                return future.result(timeout=20)

        except FutureTimeoutError:
            try:
                return _call_groq_text(prompt, _langsmith_extra())
            except Exception:
                fallback_text = (
                    "The primary language model timed out, and the backup text provider is also unavailable right now. "
                    "Please try again in a few moments."
                )
                _record_metrics(
                    call_type="text",
                    provider="router",
                    model="gemini->groq",
                    prompt=prompt,
                    response=fallback_text,
                    latency_ms=0.0,
                    success=False,
                    fallback_used=True,
                )
                return fallback_text

        except Exception as e:
            msg = str(e).lower()
            retryable = any(
                x in msg for x in [
                    "429", "500", "503", "unavailable",
                    "resource_exhausted", "deadline", "timeout"
                ]
            )

            if retryable:
                try:
                    return _call_groq_text(prompt, _langsmith_extra())
                except Exception:
                    fallback_text = (
                        "The primary language model is temporarily overloaded, and the backup text provider is also unavailable right now. "
                        "Please try again in a few moments."
                    )
                    _record_metrics(
                        call_type="text",
                        provider="router",
                        model="gemini->groq",
                        prompt=prompt,
                        response=fallback_text,
                        latency_ms=0.0,
                        success=False,
                        fallback_used=True,
                    )
                    return fallback_text

            error_text = f"I ran into a temporary issue while generating the response: {e}"
            _record_metrics(
                call_type="text",
                provider="gemini",
                model=MODEL_NAME,
                prompt=prompt,
                response=error_text,
                latency_ms=0.0,
                success=False,
                fallback_used=False,
            )
            return error_text


@traceable(run_type="llm", name="gemini_vision_call")
def _call_gemini_vision(image_path: str, user_query: str, langsmith_extra=None) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    started = time.perf_counter()
    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            user_query,
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_bytes,
                }
            },
        ],
    )
    text = getattr(response, "text", None)
    final_text = text.strip() if text and text.strip() else "I couldn’t analyze the image right now."

    _record_metrics(
        call_type="vision",
        provider="gemini",
        model=MODEL_NAME,
        prompt=user_query,
        response=final_text,
        latency_ms=(time.perf_counter() - started) * 1000,
        success=True,
        fallback_used=False,
    )
    return final_text


@traceable(run_type="llm", name="groq_vision_call")
def _call_groq_vision(image_path: str, user_query: str, langsmith_extra=None) -> str:
    if not groq_client:
        raise RuntimeError("Groq client is not configured.")

    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    started = time.perf_counter()
    completion = groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are {APP_NAME}, an AI customer support assistant built for this application. "
                    "Describe images clearly and concisely. "
                    "Do not say you are Groq."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded}"
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )
    final_text = completion.choices[0].message.content.strip()

    _record_metrics(
        call_type="vision",
        provider="groq",
        model=GROQ_VISION_MODEL,
        prompt=user_query,
        response=final_text,
        latency_ms=(time.perf_counter() - started) * 1000,
        success=True,
        fallback_used=True,
    )
    return final_text


@traceable(run_type="chain", name="analyze_image_with_fallback")
def analyze_image(image_path: str, user_query: str, langsmith_extra=None) -> str:
    last_error = None

    trace_metadata = {
        "component": "vision_generation",
        "primary_provider": "gemini",
        "backup_provider": "groq",
        "app_name": APP_NAME,
    }

    with _trace_context(metadata=trace_metadata, tags=["vision", "fallback-router"]):
        for attempt in range(3):
            try:
                return _call_gemini_vision(image_path, user_query, _langsmith_extra())
            except Exception as e:
                last_error = str(e).lower()
                retryable = any(
                    x in last_error for x in [
                        "429", "500", "503", "unavailable",
                        "resource_exhausted", "deadline", "timeout"
                    ]
                )

                if not retryable:
                    break

                wait = (2 ** attempt) + random.uniform(0.0, 0.8)
                time.sleep(wait)

        try:
            return _call_groq_vision(image_path, user_query, _langsmith_extra())
        except Exception:
            fallback_text = (
                "Image analysis is temporarily unavailable on both Gemini and Groq right now. "
                "Please try again in a moment."
            )
            _record_metrics(
                call_type="vision",
                provider="router",
                model="gemini->groq",
                prompt=user_query,
                response=fallback_text,
                latency_ms=0.0,
                success=False,
                fallback_used=True,
            )
            return fallback_text
