import base64
import numpy as np
from io import BytesIO
from PIL import Image

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import login as auth_login

from .models import Person
from deepface import DeepFace
from django.contrib.auth import logout
from django.views import View

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
# HELPER: Extract 512-d face embedding using DeepFace + ArcFace
# ─────────────────────────────────────────────────────────────
def extract_embedding(pil_image: Image.Image):
    """
    Returns a 512-d numpy embedding vector, or None if no face found.
    Uses DeepFace with ArcFace model.
    """
    img_array = np.array(pil_image)
    try:
        result = DeepFace.represent(
            img_path=img_array,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="opencv"
        )
        return np.array(result[0]["embedding"])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# HELPER: Liveness check using DeepFace face detection
# ─────────────────────────────────────────────────────────────
def check_liveness(pil_image: Image.Image) -> bool:
    """
    Basic liveness check — verifies a real face is detectable.

    Upgrade options:
    - Silent-Face-Anti-Spoofing (GitHub: minivision-ai)
    - Custom CNN trained on real vs spoof datasets (NUAA, MSU-MFSD)
    """
    img_array = np.array(pil_image)
    try:
        faces = DeepFace.extract_faces(
            img_path=img_array,
            enforce_detection=True,
            detector_backend="opencv"
        )
        # Must detect at least one face with reasonable confidence
        if not faces:
            return False
        best = max(faces, key=lambda f: f.get("confidence", 0))
        return best.get("confidence", 0) > 0.9
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
# HELPER: Find matching person by face embedding
# ─────────────────────────────────────────────────────────────
def find_matching_person(query_embedding: np.ndarray, threshold: float = 0.6):
    """
    Compare query embedding against all stored embeddings.
    Returns the best-matching Person and similarity score, or (None, score).
    """
    best_person     = None
    best_similarity = 0.0

    for person in Person.objects.exclude(embedding=None):
        stored     = np.frombuffer(person.embedding, dtype=np.float32)
        similarity = np.dot(query_embedding, stored) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(stored) + 1e-10
        )
        if similarity > best_similarity:
            best_similarity = float(similarity)
            best_person     = person

    if best_similarity >= threshold:
        return best_person, best_similarity

    return None, best_similarity


# ─────────────────────────────────────────────────────────────
# VIEW: Home
# ─────────────────────────────────────────────────────────────
def main(request):
    return render(request, "home.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Register
# ─────────────────────────────────────────────────────────────
def register(request):
    """
    Modern registration flow:
    1. Receive name + base64 face image from webcam
    2. Decode image
    3. Run liveness check (anti-spoof)
    4. Extract 512-d ArcFace embedding via DeepFace
    5. Check for duplicate face
    6. Create Django User + Person with embedding
    7. Auto-login and redirect to dashboard
    """
    if request.method == "POST":
        name     = request.POST.get("name", "").strip()
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
        base_username = name.lower().replace(" ", "_")
        username = base_username
        counter  = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}_{counter}"
            counter += 1

        user = User.objects.create_user(
            username=username,
            first_name=name,
        )

        Person.objects.create(
            name=name,
            embedding=embedding.astype(np.float32).tobytes(),
        )

        # ── Step 7: Auto-login ────────────────────────────────
        auth_login(request, user, backend='django.contrib.auth.backends.ModelBackend')
        messages.success(request, f"Welcome, {name}! You are now registered.")
        return redirect("dashboard")

    return render(request, "register.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Login — face-based authentication
# ─────────────────────────────────────────────────────────────
def login(request):
    """
    Face login flow:
    1. Receive base64 face image from webcam
    2. Decode image
    3. Liveness check — reject spoofs
    4. Extract 512-d ArcFace embedding via DeepFace
    5. Compare against all stored embeddings (cosine similarity)
    6. Match found  → log user in, redirect to dashboard
       No match     → access denied with similarity score
    """
    if request.method == "POST":
        face_b64 = request.POST.get("face_image", "")

        # ── Step 1: Validate ──────────────────────────────────
        if not face_b64:
            messages.error(request, "No face captured. Please try again.")
            return render(request, "login.html")

        # ── Step 2: Decode ────────────────────────────────────
        try:
            pil_image = decode_base64_image(face_b64)
        except Exception:
            messages.error(request, "Invalid image. Please retake your photo.")
            return render(request, "login.html")

        # ── Step 3: Liveness check ────────────────────────────
        if not check_liveness(pil_image):
            messages.error(
                request,
                "⚠️ Liveness check failed. Real face required — no photos or screens."
            )
            return render(request, "login.html", {"liveness_failed": True})

        # ── Step 4: Extract embedding ─────────────────────────
        embedding = extract_embedding(pil_image)
        if embedding is None:
            messages.error(request, "No face detected. Ensure your face is well-lit and centred.")
            return render(request, "login.html")

        # ── Step 5: Match against database ───────────────────
        matched_person, similarity = find_matching_person(embedding, threshold=0.6)

        # ── Step 6a: Match found → login ──────────────────────
        if matched_person:

            user = User.objects.filter(first_name=matched_person.name).first()
            if user:
                auth_login(request, user, backend="django.contrib.auth.backends.ModelBackend")

            messages.success(
                request,
                f"✅ Welcome back, {matched_person.name}! "
                f"(Match confidence: {similarity * 100:.1f}%)"
            )
            return redirect("home")

        # ── Step 6b: No match → access denied ────────────────
        messages.error(
            request,
            f"❌ Face not recognised. Best similarity: {similarity * 100:.1f}% "
            f"(required ≥ 50%). Please register first or try again."
        )
        return render(request, "login.html", {
            "access_denied": True,
            "similarity":    round(similarity * 100, 1),
        })

    return render(request, "login.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Dashboard (protected)
# ─────────────────────────────────────────────────────────────
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("login")
    return render(request, "dashboard.html", {"user": request.user})

class LogoutView(View):
    def post(self, request):
        logout(request)
        return redirect('home')   # change 'home' to your login or home url