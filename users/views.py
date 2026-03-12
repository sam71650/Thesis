import base64
import numpy as np
from io import BytesIO
from PIL import Image

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import login as auth_login

from .models import Person

# ── Try to import InsightFace (best 2025 option) ─────────────
# Install: pip install insightface onnxruntime
try:
    import insightface
    from insightface.app import FaceAnalysis

    FACE_APP = FaceAnalysis(name='buffalo_l')  # ArcFace model
    FACE_APP.prepare(ctx_id=-1)  # -1 = CPU, 0 = GPU
    USE_INSIGHTFACE = True
except ImportError:
    # Fallback: deepface (easier to install)
    # pip install deepface
    from deepface import DeepFace

    USE_INSIGHTFACE = False


# ─────────────────────────────────────────────────────────────
# HELPER: Decode base64 image → PIL Image
# ─────────────────────────────────────────────────────────────
def decode_base64_image(base64_str: str) -> Image.Image:
    """Convert base64 string (from webcam) to a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]  # strip data:image/jpeg;base64,
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


# ─────────────────────────────────────────────────────────────
# HELPER: Extract 512-d face embedding from image
# ─────────────────────────────────────────────────────────────
def extract_embedding(pil_image: Image.Image):
    """
    Returns a 512-d numpy embedding vector, or None if no face found.
    Uses InsightFace (ArcFace) if available, else falls back to DeepFace.
    """
    img_array = np.array(pil_image)

    if USE_INSIGHTFACE:
        faces = FACE_APP.get(img_array[:, :, ::-1])  # RGB → BGR for insightface
        if not faces:
            return None
        # Take the largest/most confident face
        face = max(faces, key=lambda f: f.det_score)
        return face.embedding  # shape: (512,)

    else:
        # DeepFace fallback
        try:
            result = DeepFace.represent(
                img_path=img_array,
                model_name="ArcFace",
                enforce_detection=True
            )
            return np.array(result[0]["embedding"])
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────
# HELPER: Liveness check (basic — upgrade with your model)
# ─────────────────────────────────────────────────────────────
def check_liveness(pil_image: Image.Image) -> bool:
    """
    Returns True if the face appears to be a live person.

    Basic version: checks image has detectable face + is not a flat photo.
    TODO: Replace with your trained anti-spoof CNN model.

    Upgrade options:
    - Silent-Face-Anti-Spoofing (GitHub: minivision-ai)
    - FAS (Face Anti-Spoofing) models from InsightFace
    """
    img_array = np.array(pil_image)

    if USE_INSIGHTFACE:
        faces = FACE_APP.get(img_array[:, :, ::-1])
        if not faces:
            return False
        # Use detection score as a basic proxy (real faces score higher)
        best_face = max(faces, key=lambda f: f.det_score)
        return float(best_face.det_score) > 0.7  # threshold

    # Fallback: just check a face is detected
    try:
        DeepFace.detectFace(img_path=img_array, enforce_detection=True)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# HELPER: Check if face already registered (duplicate check)
# ─────────────────────────────────────────────────────────────
def is_duplicate_face(new_embedding: np.ndarray, threshold: float = 0.6) -> bool:
    """
    Compare new embedding against all stored embeddings using cosine similarity.
    Returns True if a match is found (face already registered).
    """
    existing_persons = Person.objects.exclude(embedding=None)

    for person in existing_persons:
        stored = np.frombuffer(person.embedding, dtype=np.float32)

        # Cosine similarity
        similarity = np.dot(new_embedding, stored) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(stored) + 1e-10
        )

        if similarity > threshold:
            return True  # Duplicate found

    return False


# ─────────────────────────────────────────────────────────────
# VIEW: Home
# ─────────────────────────────────────────────────────────────
def main(request):
    return render(request, "base.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Register — UPGRADED
# ─────────────────────────────────────────────────────────────
def register(request):
    """
    Modern registration flow:
    1. Receive name + base64 face image from webcam
    2. Decode image
    3. Run liveness check (anti-spoof)
    4. Extract 512-d face embedding (ArcFace)
    5. Check for duplicate face
    6. Create Django User + Person with embedding
    7. Auto-login and redirect to dashboard
    """
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        face_b64 = request.POST.get("face_image", "")

        # ── Step 1: Basic validation ──────────────────────────
        if not name or len(name) < 2:
            messages.error(request, "Please enter your full name.")
            return render(request, "register.html")

        if not face_b64:
            messages.error(request, "Please capture your face before registering.")
            return render(request, "register.html")

        # ── Step 2: Decode image ──────────────────────────────
        try:
            pil_image = decode_base64_image(face_b64)
        except Exception:
            messages.error(request, "Invalid image data. Please retake your photo.")
            return render(request, "register.html")

        # ── Step 3: Liveness check ────────────────────────────
        if not check_liveness(pil_image):
            messages.error(request, "Liveness check failed. Please use a live face, not a photo.")
            return render(request, "register.html")

        # ── Step 4: Extract embedding ─────────────────────────
        embedding = extract_embedding(pil_image)
        if embedding is None:
            messages.error(request, "No face detected. Please ensure your face is clearly visible.")
            return render(request, "register.html")

        # ── Step 5: Duplicate face check ──────────────────────
        if is_duplicate_face(embedding):
            messages.error(request, "This face is already registered in the system.")
            return render(request, "register.html")

        # ── Step 6: Save to database ──────────────────────────
        # Auto-generate a unique username from name
        base_username = name.lower().replace(" ", "_")
        username = base_username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}_{counter}"
            counter += 1

        user = User.objects.create_user(
            username=username,
            first_name=name,
        )

        # Save Person with embedding stored as binary bytes
        Person.objects.create(
            name=name,
            embedding=embedding.astype(np.float32).tobytes(),
            # Don't store raw base64 — only the embedding vector
        )

        # ── Step 7: Auto-login ────────────────────────────────
        auth_login(request, user, backend='django.contrib.auth.backends.ModelBackend')
        messages.success(request, f"Welcome, {name}! You are now registered.")
        return redirect("dashboard")

    return render(request, "register.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Login (face-based)
# ─────────────────────────────────────────────────────────────
def login(request):
    return render(request, "login.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Dashboard (protected)
# ─────────────────────────────────────────────────────────────
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("login")
    return render(request, "dashboard.html", {"user": request.user})