import os
import base64
import sys
import traceback
from typing import Optional
from io import BytesIO
from PIL import Image

def preprocess_image_with_gemini(input_path: str, prompt: str, model: str, api_key: Optional[str] = None) -> Optional[str]:
    try:
        from google import genai
    except Exception as e:
        print(f"[NanoBanana] Failed to import google.genai: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[NanoBanana] GEMINI_API_KEY is missing. Skipping preprocessing.", file=sys.stderr)
        return None

    try:
        from google.genai import types
        client = genai.Client(api_key=api_key)
        img = Image.open(input_path)
        # Force image output to avoid text-only responses
        config = types.GenerateContentConfig(response_modalities=['Image'])
        resp = client.models.generate_content(model=model, contents=[prompt, img], config=config)
        # Find first image part
        out_img: Optional[Image.Image] = None
        if resp and getattr(resp, 'candidates', None):
            parts = resp.candidates[0].content.parts
            for p in parts:
                if getattr(p, 'inline_data', None) is not None:
                    mime = getattr(p.inline_data, 'mime_type', None)
                    data = p.inline_data.data
                    # Prefer raw bytes for inline_data; only base64-decode if it is a str
                    if isinstance(data, bytes):
                        raw_bytes = data
                    elif isinstance(data, str):
                        try:
                            raw_bytes = base64.b64decode(data)
                        except Exception as de:
                            print(f"[NanoBanana] Base64 decode failed: {de}", file=sys.stderr)
                            continue
                    else:
                        print("[NanoBanana] Inline data is neither bytes nor str; skipping.", file=sys.stderr)
                        continue
                    try:
                        out_img = Image.open(BytesIO(raw_bytes))
                        # Optional: verify image by loading
                        out_img.load()
                    except Exception as ie:
                        print(f"[NanoBanana] Inline data not an image (mime={mime}). Error: {ie}", file=sys.stderr)
                        out_img = None
                        continue
                    break
        else:
            print("[NanoBanana] No candidates returned by Gemini response.", file=sys.stderr)
        if out_img is None:
            try:
                # Log any text parts to help debugging prompt/model issues
                if resp and getattr(resp, 'candidates', None):
                    for p in resp.candidates[0].content.parts:
                        if getattr(p, 'text', None):
                            print(f"[NanoBanana] Text response: {p.text}", file=sys.stderr)
            except Exception:
                pass
            print("[NanoBanana] Gemini produced no image part. Skipping.", file=sys.stderr)
            return None
        base, _ = os.path.splitext(input_path)
        out_path = f"{base}_nb.png"
        out_img.save(out_path)
        return out_path
    except Exception as e:
        print(f"[NanoBanana] Exception during preprocessing: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


