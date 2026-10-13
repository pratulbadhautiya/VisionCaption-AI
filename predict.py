
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "Salesforce/blip-image-captioning-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_ai_model():
    """Load BLIP once and reuse it for future predictions."""
    print(f"Loading BLIP model on {DEVICE}...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    print("BLIP model loaded successfully.")
    return processor, model


def generate_caption(image_path):
    """Generate a natural English caption for an image."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Please provide a valid image file.") from exc

    processor, model = load_ai_model()

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            early_stopping=True
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    if not caption:
        return "Sorry, I could not generate a caption for this image."

    caption = caption[0].upper() + caption[1:]

    if caption[-1] not in ".!?":
        caption += "."

    return caption


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print('Usage: python predict.py "images/cat.jpg"')
        raise SystemExit(1)

    try:
        print("\nGenerated caption:")
        print(generate_caption(sys.argv[1]))
    except Exception as exc:
        print(f"Caption generation failed: {exc}")
        raise SystemExit(1)
