import numpy as np
from PIL import Image
from django.shortcuts import render
from .fgsm import perform_fgsm, check_similarity
from django.core.files.base import ContentFile

def main(request):
    person          = request.user if request.user.is_authenticated else None
    adv_image_url   = None
    similarity      = None
    attack_result   = None

    if request.method == "POST" and person:

        # ── Load registered face from DB ──────────────────────
        pil_image          = Image.open(person.face_image.path).convert("RGB")
        original_embedding = np.frombuffer(person.embedding, dtype=np.float32)

        # ── Get epsilon from form (default 0.05) ──────────────
        epsilon = float(request.POST.get("epsilon", 0.05))

        # ── Apply FGSM ────────────────────────────────────────
        adv_image = perform_fgsm(pil_image, epsilon=epsilon)

        # ── Check similarity ──────────────────────────────────
        similarity = check_similarity(original_embedding, adv_image)

        # ── Decide result ─────────────────────────────────────
        if similarity >= 0.5:
            attack_result = "failed"    # model still recognises the face
        else:
            attack_result = "success"   # model fooled ✅

        # ── Save adversarial image temporarily to serve it ────
        import os
        from io import BytesIO


        buffer = BytesIO()
        adv_image.save(buffer, format="JPEG")
        buffer.seek(0)

        adv_filename = f"adv_{person.username}.jpg"
        adv_path     = os.path.join("media", "adversarial", adv_filename)
        os.makedirs(os.path.dirname(adv_path), exist_ok=True)

        with open(adv_path, "wb") as f:
            f.write(buffer.read())

        adv_image_url = f"/media/adversarial/{adv_filename}"

    return render(request, "research.html", {
        "person":        person,
        "adv_image_url": adv_image_url,
        "similarity":    round(similarity * 100, 1) if similarity is not None else None,
        "attack_result": attack_result,
    })