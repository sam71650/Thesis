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
import cv2
# import mediapipe as mp  # ← commented out due to version incompatibility
from skimage.feature import local_binary_pattern


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
# HELPER: Layer 1 — Basic face detection (original check)
# ─────────────────────────────────────────────────────────────
def check_face_detection(pil_image: Image.Image) -> bool:
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
# Next Phase: Implementing 3D liveness checks to prevent photo/screen spoofing
# HELPER: Layer 2 — 3D Facial Geometry via MediaPipe
# Checks Z-coordinate (depth) variation across face landmarks
# Real faces have high Z variance; flat photos have near zero
# NOTE: Disabled due to mediapipe version incompatibility
#       with Python 3.12. Re-enable after installing:
#       pip install mediapipe==0.10.11 -----> It does not work with mediapipe==0.10.11, which is required for Python 3.12.
#       This layer is currently disabled until a compatible version of mediapipe is available.
# ─────────────────────────────────────────────────────────────
# def check_3d_facial_geometry(pil_image: Image.Image) -> bool:
#     mp_face_mesh = mp.solutions.face_mesh
#     try:
#         with mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             refine_landmarks=True  # includes Z depth coordinates
#         ) as face_mesh:
#             results = face_mesh.process(np.array(pil_image))
#
#             if not results.multi_face_landmarks:
#                 return False
#
#             landmarks = results.multi_face_landmarks[0].landmark
#
#             # Extract Z coordinates (depth dimension)
#             z_coords = [lm.z for lm in landmarks]
#
#             # Real faces have significant 3D depth variation
#             # Flat printed photos have near-zero Z variance
#             z_variance = np.var(z_coords)
#             z_range    = max(z_coords) - min(z_coords)
#
#             return z_variance > 0.001 and z_range > 0.05
#     except Exception:
#         return False


# ─────────────────────────────────────────────────────────────
# HELPER: Layer 3 — Depth Variation via Laplacian
# Real faces have high frequency texture variation (3D surface)
# Flat printed photos have low Laplacian variance
# ─────────────────────────────────────────────────────────────
def check_depth_variation(pil_image: Image.Image) -> bool:
    try:
        gray      = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # High variance = 3D surface (real face)
        # Low variance  = flat surface (printed photo)
        depth_score = laplacian.var()

        return depth_score > 20.0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# HELPER: Layer 4 — Skin Texture + Reflectance Analysis
# Checks three 3D skin properties:
# (1) Micro-texture via LBP (skin pores)
# (2) Surface reflectance via FFT (light scatter)
# (3) Depth gradient (nose vs cheek region)
# ─────────────────────────────────────────────────────────────
def check_skin_texture_3d(pil_image: Image.Image) -> bool:
    try:
        img_array = np.array(pil_image)
        gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # ── (1) Micro-texture: LBP variance ──────────────────
        # Real skin has natural pore texture → high LBP variance
        # Printed paper/screen is smoother  → low LBP variance
        lbp          = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_variance = np.var(lbp)

        # ── (2) Surface reflectance: FFT high-frequency score ─
        # Real skin scatters light naturally → rich high-freq content
        # Flat surfaces reflect uniformly   → low high-freq content
        fft        = np.fft.fft2(gray)
        fft_shift  = np.fft.fftshift(fft)
        magnitude  = np.abs(fft_shift)
        high_freq  = magnitude[magnitude > np.percentile(magnitude, 90)]
        freq_score = np.mean(high_freq)

        # ── (3) Depth gradient: nose vs cheek brightness ──────
        # Real faces: nose protrudes → brightness difference
        # Flat photos: uniform depth → minimal brightness difference
        h, w         = gray.shape
        nose_region  = gray[h // 3:h // 2,  w // 3:2 * w // 3]
        cheek_region = gray[h // 3:h // 2,  :w // 4]
        depth_diff   = abs(np.mean(nose_region) - np.mean(cheek_region))

        texture_ok     = lbp_variance  > 5.0
        reflectance_ok = freq_score    > 200.0
        depth_ok       = depth_diff    > 3.0

        passed = sum([texture_ok, reflectance_ok, depth_ok])
        return passed >= 2
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# HELPER: Full 3D Liveness Check (4-layer pipeline)
#
# Layer 1: Face detection confidence     (does a face exist?)
# Layer 2: 3D geometry via MediaPipe     (is there depth?)     ← disabled
# Layer 3: Laplacian depth variation     (is face 3D not flat?)
# Layer 4: Skin texture + reflectance   (does skin look real?)
#
# All active layers must pass → LIVE
# Any layer fails             → SPOOF
# ─────────────────────────────────────────────────────────────
def check_liveness(pil_image: Image.Image) -> bool:

    # ── Layer 1: Basic face detection ────────────────────────
    if not check_face_detection(pil_image):
        return False

    # ── Layer 2: 3D facial geometry (MediaPipe Z-coords) ─────
    # Disabled due to mediapipe==0.10.11 required for Python 3.12
    # Uncomment after: pip install mediapipe==0.10.11
    # if not check_3d_facial_geometry(pil_image):
    #     return False

    # ── Layer 3: Depth variation (Laplacian variance) ─────────
    if not check_depth_variation(pil_image):
        return False

    # ── Layer 4: Skin texture + reflectance (LBP + FFT) ───────
    if not check_skin_texture_3d(pil_image):
        return False

    return True  # passed all active 3D liveness checks


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

        # ── Step 3: 3D Liveness check (4-layer) ──────────────
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

        # ── Step 3: 3D Liveness check (4-layer) ──────────────
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

        # ── Step 5: Match against database ────────────────────
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