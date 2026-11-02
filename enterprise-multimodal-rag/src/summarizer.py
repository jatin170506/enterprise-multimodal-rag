import os
import sys
import io
import time
import base64
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)

def compress_and_encode_image(image_path: str, max_dim: int = 800) -> str:
    """Downscales large screenshots/diagrams to max 800px and converts to JPEG."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_diagram_summary(image_path: str, max_retries: int = 4) -> str:
    """Sends compressed image to Gemini 2.5 Flash with exponential backoff for 503 errors."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file!")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    base64_image = compress_and_encode_image(image_path)

    prompt = (
        "Analyze this screenshot/diagram concisely. Identify:\n"
        "1. Context & Purpose (e.g., REPL session, system diagram)\n"
        "2. Key elements, inputs, outputs, or code methods visible\n"
        "Keep the summary structured, technical, and under 150 words."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }

    backoff = 2  # initial delay in seconds
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Sending image payload to Gemini (Attempt {attempt}/{max_retries})...")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            res_json = response.json()

            if response.status_code == 200:
                return res_json["candidates"][0]["content"]["parts"][0]["text"]

            if response.status_code == 503 and attempt < max_retries:
                print(f"[WARN] 503 High Demand detected. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
                continue

            error_msg = res_json.get("error", {}).get("message", str(res_json))
            raise RuntimeError(f"API Error ({response.status_code}): {error_msg}")

        except (requests.exceptions.RequestException, RuntimeError) as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2
            