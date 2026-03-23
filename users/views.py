import base64
import numpy as np
from io import BytesIO
from PIL import Image
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login as auth_login, logout
from django.views import View
from django.core.files.base import ContentFile
from .models import Person
from deepface import DeepFace
import os

# ─────────────────────────────────────────────────────────────
# HELPER: Decode base64 image → PIL Image
# ─────────────────────────────────────────────────────────────
def decode_base64_image(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


# ─────────────────────────────────────────────────────────────
# HELPER: Extract 512-d face embedding using DeepFace + ArcFace
# ─────────────────────────────────────────────────────────────
def extract_embedding(pil_image: Image.Image):
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
    img_array = np.array(pil_image)
    try:
        faces = DeepFace.extract_faces(
            img_path=img_array,
            enforce_detection=True,
            detector_backend="opencv"
        )
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
    for person in Person.objects.exclude(embedding=None):
        stored = np.frombuffer(person.embedding, dtype=np.float32)
        similarity = np.dot(new_embedding, stored) / (
            np.linalg.norm(new_embedding) * np.linalg.norm(stored) + 1e-10
        )
        if similarity > threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# HELPER: Find matching person by face embedding
# ─────────────────────────────────────────────────────────────
def find_matching_person(query_embedding: np.ndarray, threshold: float = 0.5):
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
    if request.method == "POST":
        name     = request.POST.get("name", "").strip()
        face_b64 = request.POST.get("face_image", "")

        # ── Step 1: Basic validation ──────────────────────────
        if not name or len(name) < 2:
            messages.error(request, "Please enter your full name.")
            return redirect("users:register")

        if not face_b64:
            messages.error(request, "Please capture your face before registering.")
            return redirect("users:register")

        # ── Step 2: Decode image ──────────────────────────────
        try:
            pil_image = decode_base64_image(face_b64)
        except Exception:
            messages.error(request, "Invalid image data. Please retake your photo.")
            return redirect("users:register")

        # ── Step 3: Liveness check ────────────────────────────
        if not check_liveness(pil_image):
            messages.error(request, "Liveness check failed. Please use a live face, not a photo.")
            return redirect("users:register")

        # ── Step 4: Extract embedding ─────────────────────────
        embedding = extract_embedding(pil_image)
        if embedding is None:
            messages.error(request, "No face detected. Please ensure your face is clearly visible.")
            return redirect("users:register")

        # ── Step 5: Duplicate face check ──────────────────────
        if is_duplicate_face(embedding):
            messages.error(request, "This face is already registered in the system.")
            return redirect("users:register")

        # ── Step 6: Build a unique username ───────────────────
        base_username = name.lower().replace(" ", "_")
        username      = base_username
        counter       = 1
        while Person.objects.filter(username=username).exists():
            username = f"{base_username}_{counter}"
            counter += 1

        # ── Step 7: Convert PIL image → Django file ───────────
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_file = ContentFile(buffer.getvalue(), name=f"{username}.jpg")

        # ── Step 8: Create ONE Person (which IS the user) ─────
        person = Person.objects.create_user(
            username=username,
            first_name=name,
            embedding=embedding.astype(np.float32).tobytes(),
            face_image=image_file,
        )

        # ── Step 9: Auto-login ────────────────────────────────
        auth_login(request, person, backend="django.contrib.auth.backends.ModelBackend")
        messages.success(request, f"Welcome, {name}! You are now registered.")
        return redirect("users:dashboard")

    return render(request, "register.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Login — face-based authentication
# ─────────────────────────────────────────────────────────────
def login(request):
    if request.method == "POST":
        face_b64 = request.POST.get("face_image", "")

        # ── Step 1: Validate ──────────────────────────────────
        if not face_b64:
            messages.error(request, "No face captured. Please try again.")
            return redirect("users:login")

        # ── Step 2: Decode ────────────────────────────────────
        try:
            pil_image = decode_base64_image(face_b64)
        except Exception:
            messages.error(request, "Invalid image. Please retake your photo.")
            return redirect("users:login")

        # ── Step 3: Liveness check ────────────────────────────
        if not check_liveness(pil_image):
            messages.error(
                request,
                "⚠️ Liveness check failed. Real face required — no photos or screens."
            )
            return redirect("users:login")

        # ── Step 4: Extract embedding ─────────────────────────
        embedding = extract_embedding(pil_image)
        if embedding is None:
            messages.error(request, "No face detected. Ensure your face is well-lit and centred.")
            return redirect("users:login")

        # ── Step 5: Match against database ───────────────────
        matched_person, similarity = find_matching_person(embedding, threshold=0.5)

        # ── Step 6a: Match found → login ──────────────────────
        if matched_person:
            auth_login(request, matched_person, backend="django.contrib.auth.backends.ModelBackend")
            messages.success(request, f"Welcome back, {matched_person.first_name}!")
            return redirect("users:dashboard")

        # ── Step 6b: No match → access denied ────────────────
        messages.error(
            request,
            f"❌ Face not recognised. Best similarity: {similarity * 100:.1f}% "
            f"(required ≥ 50%). Please register first or try again."
        )
        return redirect("users:login")

    return render(request, "login.html")


# ─────────────────────────────────────────────────────────────
# VIEW: Dashboard (protected)
# ─────────────────────────────────────────────────────────────
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("users:login")
    return render(request, "dashboard.html", {"user": request.user})


# ─────────────────────────────────────────────────────────────
# VIEW: Logout
# ─────────────────────────────────────────────────────────────
class LogoutView(View):
    def post(self, request):
        logout(request)
        return redirect("users:main")


# ─────────────────────────────────────────────────────────────
# VIEW: Delete Account
# ─────────────────────────────────────────────────────────────
@login_required
def delete_user(request):
    if request.method == "POST":
        user = request.user

        # ── Delete face image from disk ───────────────────────
        if user.face_image:
            if os.path.isfile(user.face_image.path):
                os.remove(user.face_image.path)

        # ── Delete adversarial image from disk ────────────────
        adv_path = os.path.join("media", "adversarial", f"adv_{user.username}.jpg")
        if os.path.isfile(adv_path):
            os.remove(adv_path)

        # ── Delete user from DB ───────────────────────────────
        logout(request)
        user.delete()

        messages.success(request, "Your account has been deleted.")
        return redirect("users:main")

    return render(request, "delete_account.html")