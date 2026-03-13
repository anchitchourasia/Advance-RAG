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
)

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in environment variables.")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _call_gemini_text(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()
    return "I couldn’t generate a response right now."


def _call_groq_text(prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq client is not configured.")

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
    return completion.choices[0].message.content.strip()


def ask_gemini(prompt: str) -> str:
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini_text, prompt)
            return future.result(timeout=20)
    except FutureTimeoutError:
        try:
            return _call_groq_text(prompt)
        except Exception:
            return (
                "The primary language model timed out, and the backup text provider is also unavailable right now. "
                "Please try again in a few moments."
            )
    except Exception as e:
        msg = str(e).lower()
        retryable = any(x in msg for x in [
            "429", "500", "503", "unavailable", "resource_exhausted", "deadline", "timeout"
        ])

        if retryable:
            try:
                return _call_groq_text(prompt)
            except Exception:
                return (
                    "The primary language model is temporarily overloaded, and the backup text provider is also unavailable right now. "
                    "Please try again in a few moments."
                )

        return f"I ran into a temporary issue while generating the response: {e}"


def _call_gemini_vision(image_path: str, user_query: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

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
    if text and text.strip():
        return text.strip()
    return "I couldn’t analyze the image right now."


def _call_groq_vision(image_path: str, user_query: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq client is not configured.")

    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

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
    return completion.choices[0].message.content.strip()


def analyze_image(image_path: str, user_query: str) -> str:
    last_error = None

    for attempt in range(3):
        try:
            return _call_gemini_vision(image_path, user_query)
        except Exception as e:
            last_error = str(e).lower()
            retryable = any(x in last_error for x in [
                "429", "500", "503", "unavailable", "resource_exhausted", "deadline", "timeout"
            ])

            if not retryable:
                break

            wait = (2 ** attempt) + random.uniform(0.0, 0.8) 
            time.sleep(wait)

    try:
        return _call_groq_vision(image_path, user_query)
    except Exception:
        return "Image analysis is temporarily unavailable on both Gemini and Groq right now. Please try again in a moment."
