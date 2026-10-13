
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from predict import generate_caption


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ---------------- FILE VALIDATION ----------------

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_image(image_path):
    """
    Verify that the uploaded file contains a supported image.
    Returns the detected image format.
    """
    with Image.open(image_path) as image:
        image_format = image.format

        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format detected: {image_format}"
            )

        image.verify()

    return image_format


# ---------------- MAIN PAGE ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")

        # Check whether a file was selected.
        if uploaded is None or not uploaded.filename:
            error = "Please select an image before continuing."

        # Check the filename extension.
        elif not allowed_file(uploaded.filename):
            error = "Please upload a JPG, JPEG, PNG, or WEBP image."

        else:
            original_name = secure_filename(uploaded.filename)

            if not original_name or "." not in original_name:
                error = "Invalid filename. Please select another image."

            else:
                extension = original_name.rsplit(".", 1)[1].lower()
                saved_name = f"{uuid4().hex}.{extension}"
                image_path = UPLOAD_FOLDER / saved_name

                # Save the uploaded image.
                try:
                    uploaded.save(image_path)

                except OSError as exc:
                    print(f"FILE SAVE ERROR: {exc}")
                    error = "Unable to save the uploaded image."

                else:
                    # Validate the actual image contents.
                    try:
                        detected_format = validate_image(image_path)
                        print(
                            f"Image validation successful: "
                            f"{detected_format}"
                        )

                    except (
                        UnidentifiedImageError,
                        OSError,
                        ValueError,
                    ) as exc:
                        print(f"IMAGE VALIDATION ERROR: {exc}")
                        error = (
                            "The uploaded file could not be read as a "
                            "valid JPG, PNG, or WEBP image. "
                            "Please try another image."
                        )
                        image_path.unlink(missing_ok=True)

                    except Exception as exc:
                        print(f"UNEXPECTED IMAGE ERROR: {exc}")
                        error = "An unexpected error occurred while reading the image."
                        image_path.unlink(missing_ok=True)

                    else:
                        # Generate the caption only after validation.
                        try:
                            caption = generate_caption(str(image_path))

                            image_url = url_for(
                                "static",
                                filename=f"uploads/{saved_name}"
                            )

                            print("Caption generated successfully.")

                        except FileNotFoundError as exc:
                            print(f"MODEL FILE ERROR: {exc}")
                            error = (
                                "The trained AI model or tokenizer is "
                                "missing. Check the model files and run "
                                "python train.py after preparing your dataset."
                            )
                            image_path.unlink(missing_ok=True)

                        except Exception as exc:
                            print(f"CAPTION GENERATION ERROR: {exc}")
                            error = (
                                "The image was valid, but caption generation "
                                "failed. Check the terminal for details."
                            )
                            image_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        caption=caption,
        image_url=image_url,
        error=error
    )


# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        caption=None,
        image_url=None,
        error="Image is too large. Maximum allowed size is 10 MB."
    ), 413


@app.errorhandler(500)
def internal_server_error(_error):
    return render_template(
        "index.html",
        caption=None,
        image_url=None,
        error="An internal server error occurred. Check the terminal."
    ), 500


# ---------------- START APPLICATION ----------------


if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
