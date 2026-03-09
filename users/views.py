from django.shortcuts import render, redirect
from .models import Person
import numpy as np

def main(request):
    return render(request, "base.html")

def register(request):

    if request.method == "POST":

        name = request.POST.get("name")
        username = request.POST.get("username")
        face_image = request.POST.get("face_image")

        if name and username and face_image:

            Person.objects.create(
                name=name,
                username=username,
                face_image=face_image
            )

            return redirect("login")

    return render(request, "register.html")

def login(request):
    return render(request, "login.html")
