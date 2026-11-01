import os
import sys
import io
import base64
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)

def compress_and_encode_image(image_path: str, max_dim: int = 800) -> str:
    """
    Downscales large screenshots/diagrams to max 800px and converts 
    to optimized JPEG to keep HTTP payload size under 100KB.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        # Resize maintaining aspect ratio
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_diagram_summary(image_path: str) -> str:
    """Sends compressed image to Gemini 2.5 Flash for rapid analysis."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file!")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    # Compress image down to ~80KB before sending
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

    print(f"[INFO] Sending compressed image payload to Gemini 2.5 Flash...")
    response = requests.post(url, headers=headers, json=payload, timeout=20)
    res_json = response.json()

    if response.status_code != 200:
        error_msg = res_json.get("error", {}).get("message", str(res_json))
        raise RuntimeError(f"API Error ({response.status_code}): {error_msg}")

    return res_json["candidates"][0]["content"]["parts"][0]["text"]
    