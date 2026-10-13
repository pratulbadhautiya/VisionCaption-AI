import pickle
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from prepare_dataset import load_dataset


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

MODEL_PATH = BASE_DIR / "image_caption_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"

MAX_LENGTH = 34
MAX_VOCAB_SIZE = 5000
EPOCHS = 20
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

pairs, missing_images, invalid_lines = load_dataset()

if not pairs:
    raise ValueError(
        "No valid image-caption pairs found. "
        "Check images/ and caption.txt."
    )

print(f"Valid image-caption pairs: {len(pairs)}")
print(f"Missing images reported by dataset loader: {len(missing_images)}")
print(f"Invalid caption lines: {len(invalid_lines)}")


# Normalize captions and group captions by image.
captions_by_image = {}

for image_name, caption in pairs:
    image_name = str(image_name).strip()
    caption = str(caption).lower().strip()

    if not image_name or not caption:
        continue

    # Avoid adding start/end tokens twice.
    caption = caption.replace("startseq", "").replace("endseq", "")
    caption = " ".join(caption.split())

    if not caption:
        continue

    caption = f"startseq {caption} endseq"

    captions_by_image.setdefault(image_name, []).append(caption)

if not captions_by_image:
    raise ValueError("No usable captions found in the dataset.")


# =========================================================
# TOKENIZER
# =========================================================

print("\nPreparing tokenizer...")

all_captions = [
    caption
    for image_captions in captions_by_image.values()
    for caption in image_captions
]

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<unk>",
    filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
)

tokenizer.fit_on_texts(all_captions)

vocab_size = min(
    len(tokenizer.word_index) + 1,
    MAX_VOCAB_SIZE
)

if vocab_size <= 2:
    raise ValueError(
        "Vocabulary is too small. Add more valid captions."
    )

if "startseq" not in tokenizer.word_index:
    raise ValueError("Tokenizer is missing the startseq token.")

if "endseq" not in tokenizer.word_index:
    raise ValueError("Tokenizer is missing the endseq token.")

print(f"Vocabulary size: {vocab_size}")
print(f"Maximum caption length: {MAX_LENGTH}")


# =========================================================
# RESNET50 FEATURE EXTRACTOR
# =========================================================

print("\nLoading ResNet50 image feature extractor...")

resnet = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

resnet.trainable = False


def extract_features(image_path):
    """Extract a 2048-dimensional feature vector from an image."""

    image = load_img(
        str(image_path),
        target_size=(224, 224),
        color_mode="rgb"
    )

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = resnet.predict(image, verbose=0)

    return features[0].astype(np.float32)


# =========================================================
# EXTRACT IMAGE FEATURES
# =========================================================

image_features = {}
valid_captions_by_image = {}

print("\nExtracting image features...")

total_images = len(captions_by_image)

for index, (image_name, image_captions) in enumerate(
    captions_by_image.items(),
    start=1
):
    image_path = IMAGE_DIR / image_name

    if not image_path.is_file():
        print(f"Skipping missing image: {image_path}")
        continue

    try:
        feature = extract_features(image_path)

        if feature.shape != (2048,):
            print(f"Skipping unexpected feature shape: {image_name}")
            continue

        image_features[image_name] = feature
        valid_captions_by_image[image_name] = image_captions

    except Exception as exc:
        print(f"Skipping unreadable image {image_name}: {exc}")

    if index % 10 == 0 or index == total_images:
        print(f"Processed {index}/{total_images} images")

if not image_features:
    raise ValueError(
        "No image features were extracted. "
        "Check image files and ResNet50 loading."
    )

print(f"Successfully processed images: {len(image_features)}")


# =========================================================
# BUILD TRAINING SEQUENCES
# =========================================================

print("\nPreparing training sequences...")

X_images = []
X_sequences = []
Y_words = []

for image_name, image_captions in valid_captions_by_image.items():
    feature = image_features[image_name]

    for caption in image_captions:
        sequence = tokenizer.texts_to_sequences([caption])[0]

        # Each prefix predicts the next word.
        for position in range(1, len(sequence)):
            input_sequence = sequence[:position]
            target_word = sequence[position]

            # Ignore targets outside the model vocabulary.
            if target_word <= 0 or target_word >= vocab_size:
                continue

            padded_sequence = pad_sequences(
                [input_sequence],
                maxlen=MAX_LENGTH,
                padding="post",
                truncating="pre"
            )[0]

            X_images.append(feature)
            X_sequences.append(padded_sequence)
            Y_words.append(target_word)

if len(X_images) == 0:
    raise ValueError(
        "No training samples were generated. "
        "Check caption.txt and image-caption pairs."
    )

X_images = np.asarray(X_images, dtype=np.float32)
X_sequences = np.asarray(X_sequences, dtype=np.int32)
Y_words = np.asarray(Y_words, dtype=np.int32)

print(f"Training samples: {len(X_images)}")
print(f"Image features shape: {X_images.shape}")
print(f"Caption sequences shape: {X_sequences.shape}")
print(f"Target words shape: {Y_words.shape}")


# =========================================================
# BUILD CAPTION GENERATION MODEL
# =========================================================

print("\nBuilding caption generation model...")

image_input = Input(
    shape=(2048,),
    name="image_features"
)

image_dense = Dense(
    256,
    activation="relu",
    name="image_projection"
)(image_input)

caption_input = Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="caption_input"
)

embedding = Embedding(
    input_dim=vocab_size,
    output_dim=256,
    mask_zero=True,
    name="word_embedding"
)(caption_input)

caption_lstm = LSTM(
    256,
    name="caption_lstm"
)(embedding)

merged = Add(name="image_caption_fusion")(
    [image_dense, caption_lstm]
)

dense = Dense(
    256,
    activation="relu",
    name="decoder_dense"
)(merged)

dropout = Dropout(
    0.4,
    name="decoder_dropout"
)(dense)

output = Dense(
    vocab_size,
    activation="softmax",
    name="next_word"
)(dropout)

model = Model(
    inputs=[image_input, caption_input],
    outputs=output,
    name="VisionCaption_Model"
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Avoid the terminal-width error from model.summary().
try:
    model.summary(line_length=100)
except (ValueError, TypeError) as exc:
    print(f"Model summary could not be displayed: {exc}")

print("Model architecture created successfully.")


# =========================================================
# TRAIN MODEL
# =========================================================

print("\nStarting training...")

# Reserve validation data only when the dataset is large enough.
use_validation = len(X_images) >= 20

callbacks = []

if use_validation:
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    )

    callbacks.append(
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    )
else:
    callbacks.append(
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="loss",
            save_best_only=True,
            verbose=1
        )
    )

history = model.fit(
    [X_images, X_sequences],
    Y_words,
    epochs=EPOCHS,
    batch_size=min(BATCH_SIZE, len(X_images)),
    validation_split=0.1 if use_validation else 0.0,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)


# =========================================================
# SAVE FINAL MODEL AND TOKENIZER
# =========================================================

print("\nSaving trained model...")

# Save the final model using the best weights available
# after EarlyStopping, when validation is enabled.
model.save(str(MODEL_PATH))

with open(TOKENIZER_PATH, "wb") as file:
    pickle.dump(tokenizer, file)

if not MODEL_PATH.is_file():
    raise RuntimeError("Model file was not created.")

if not TOKENIZER_PATH.is_file():
    raise RuntimeError("Tokenizer file was not created.")

print("\nTraining completed successfully!")
print(f"Model saved to: {MODEL_PATH}")
print(f"Tokenizer saved to: {TOKENIZER_PATH}")
print("Both required files are ready.")
