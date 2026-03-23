import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition


# ─────────────────────────────────────────────────────────────
# STEP 1: Load ArcFace model once (reuse across calls)
# ─────────────────────────────────────────────────────────────
_arcface_model = None

def get_arcface_model():
    global _arcface_model
    if _arcface_model is None:
        _arcface_model = DeepFace.build_model("ArcFace")
    return _arcface_model


# ─────────────────────────────────────────────────────────────
# STEP 2: Preprocess image the same way ArcFace expects
# ─────────────────────────────────────────────────────────────
def preprocess_for_arcface(pil_image: Image.Image) -> tf.Tensor:
    """Resize to 112x112, normalize to [-1, 1] as ArcFace expects."""
    img = pil_image.resize((112, 112))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - 127.5) / 128.0          # ArcFace normalization
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)   # shape: (1, 112, 112, 3)
    return img_tensor


# ─────────────────────────────────────────────────────────────
# STEP 3: Core FGSM using TensorFlow GradientTape
# ─────────────────────────────────────────────────────────────
def perform_fgsm(pil_image: Image.Image, epsilon: float = 0.05) -> Image.Image:
    """
    Apply FGSM to a PIL face image using ArcFace.

    x_adv = x + ε · sign(∇ₓ J(θ, x, y))

    Args:
        pil_image: original face as PIL Image
        epsilon:   perturbation strength (0.01 subtle → 0.1 aggressive)

    Returns:
        adversarial PIL Image (same size as input)
    """
    original_size = pil_image.size  # save original size to restore later
    model = get_arcface_model()

    # Preprocess
    img_tensor = preprocess_for_arcface(pil_image)
    img_var    = tf.Variable(img_tensor, trainable=True, dtype=tf.float32)

    # Compute gradient of embedding w.r.t. input image
    with tf.GradientTape() as tape:
        tape.watch(img_var)
        embedding = model.model(img_var, training=False)  # (1, 512)

        # Loss = maximize embedding magnitude → confuse the model
        # We use negative L2 norm as loss so gradient points away from identity
        loss = -tf.norm(embedding)

    gradient  = tape.gradient(loss, img_var)        # ∇ₓ J
    signed_grad = tf.sign(gradient)                 # sign(∇ₓ J)

    # Apply perturbation: x_adv = x + ε · sign(∇ₓ J)
    adv_tensor = img_var + epsilon * signed_grad

    # Clip back to valid ArcFace range [-1, 1]
    adv_tensor = tf.clip_by_value(adv_tensor, -1.0, 1.0)

    # Convert back to uint8 PIL image
    adv_array = adv_tensor.numpy()[0]               # (112, 112, 3)
    adv_array = (adv_array * 128.0 + 127.5)         # denormalize
    adv_array = np.clip(adv_array, 0, 255).astype(np.uint8)

    adv_image = Image.fromarray(adv_array)
    adv_image = adv_image.resize(original_size)     # restore original size

    return adv_image


# ─────────────────────────────────────────────────────────────
# STEP 4: Check if adversarial image still matches original
# ─────────────────────────────────────────────────────────────
def check_similarity(original_embedding: np.ndarray, adv_image: Image.Image) -> float:
    """
    Extract embedding from adversarial image and compare
    against the original stored embedding using cosine similarity.
    """
    adv_array = np.array(adv_image)

    try:
        result = DeepFace.represent(
            img_path=adv_array,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv"
        )
        adv_embedding = np.array(result[0]["embedding"]).astype(np.float32)
    except Exception:
        return 0.0

    similarity = float(np.dot(original_embedding, adv_embedding) / (
        np.linalg.norm(original_embedding) * np.linalg.norm(adv_embedding) + 1e-10
    ))
    return similarity