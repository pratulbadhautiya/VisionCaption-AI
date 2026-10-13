import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
CAPTION_FILE = BASE_DIR / "caption.txt"

def load_dataset():
    pairs = []
    missing_images = []
    invalid_lines = []

    if not CAPTION_FILE.exists():
        raise FileNotFoundError(
            f"Caption file not found: {CAPTION_FILE}"
        )

    with open(CAPTION_FILE, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",", 1)

            if len(parts) != 2:
                invalid_lines.append(line_number)
                continue

            image_name = parts[0].strip()
            caption = parts[1].strip()

            if not image_name or not caption:
                invalid_lines.append(line_number)
                continue

            image_path = IMAGE_DIR / image_name

            if not image_path.is_file():
                missing_images.append(image_name)
                continue

            pairs.append((image_name, caption))

    return pairs, missing_images, invalid_lines


if __name__ == "__main__":
    try:
        pairs, missing, invalid = load_dataset()

        print("\n===== DATASET REPORT =====")
        print("Valid image-caption pairs:", len(pairs))
        print("Missing images:", len(missing))
        print("Invalid lines:", len(invalid))

        if missing:
            print("\nFirst missing image filenames:")
            for name in missing[:20]:
                print("-", name)

        if invalid:
            print("\nInvalid line numbers:", invalid[:20])

        if not pairs:
            print("\nNo valid image-caption pairs found.")
            print("Check images/ and caption.txt.")

    except Exception as error:
        print("Dataset error:", error)