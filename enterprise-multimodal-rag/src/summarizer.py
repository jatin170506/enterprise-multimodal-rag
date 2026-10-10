import os
import sys
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv(override=True)

def generate_diagram_summary(image_path: str) -> str:
    """
    Sends an image artifact to Gemini Vision to generate a detailed, 
    search-optimized technical summary.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file!")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    # Initialize Gemini Client
    client = genai.Client(api_key=api_key)
    image = Image.open(image_path)

    prompt = (
        "You are an expert technical document analyst. Analyze this diagram/schematic in detail.\n"
        "Provide a comprehensive technical description containing:\n"
        "1. Diagram Type & Purpose (e.g., System Architecture, Flowchart, Circuit Schematic)\n"
        "2. Key Components & Labels mentioned\n"
        "3. Data Flow / Connections / Directional arrows\n"
        "4. Important technical values, protocols, or formulas visible."
    )

    print(f"[INFO] Processing diagram with Gemini 2.5 Flash: {image_path}")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt]
    )

    print("[SUCCESS] Diagram summary generated successfully.")
    return response.text


if __name__ == "__main__":
    print("=== Day 2: Vision Diagram Summarization Test (Gemini) ===")
    
    sample_img = "data/extracted_images/sample_diagram.png"
    
    if os.path.exists(sample_img):
        try:
            summary_output = generate_diagram_summary(sample_img)
            print("\n--- Generated Summary Output ---")
            print(summary_output)
        except Exception as e:
            print(f"[ERROR] Summarization failed: {e}", file=sys.stderr)
    else:
        print(f"\n[NOTICE] Place a sample diagram image at '{sample_img}' to test.")